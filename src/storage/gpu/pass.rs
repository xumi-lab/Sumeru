//! 异构谓词掩码计算通道与多模式内核调度流水线 (GpuMaskedPass & GpuDispatchReceipt)
//!
//! 将 CPU 侧的 **倒排位图索引 (`ArchetypeBitSet`)**、**非迁移行级位平面 (`RowFlags`)**、
//! **关系二级位图索引 (`PairTargetIndex`)**、**64 行块增量纪元 (`ChangedSince`)** 与
//! GPU 侧的 **16KB UMA 零拷贝页 (`PageAlignedBuffer`)**、**硬件脏页折叠 (`PageDirtyBitmap`)**、
//! **跨行关系收集 (`PairTargetGpuLut`)** 融合成闭环的异构执行通道。

use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::core::archetype::{Archetype, RowBitMask};
use crate::storage::core::store::Store;
use crate::storage::core::column::PageAlignedBuffer;
use crate::storage::gpu::metal::MetalNoCopyBufferDescriptor;
use crate::storage::primitives::{DirtyPageRun, PageDirtyBitmap, HARDWARE_PAGE_SIZE};
use crate::storage::query::spec::ArchetypeHeader;
use serde::de::DeserializeOwned;
use serde::Serialize;

// ---------------------------------------------------------------------------
// 异构通道辅助原语：间接调度网格、谓词掩码缓冲区与关系目标查找表
// ---------------------------------------------------------------------------

/// GPU 间接调度参数（对齐 WebGPU `dispatchWorkgroupsIndirect` / Metal / Vulkan / CUDA 标准 3D 网格）
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectDispatchArgs {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

impl IndirectDispatchArgs {
    /// 按行数与线程组大小计算一维调度网格
    #[inline]
    pub const fn from_row_count(row_count: usize, workgroup_size: u32) -> Self {
        let wg = if workgroup_size == 0 { 64 } else { workgroup_size };
        let groups = (row_count as u32).div_ceil(wg);
        Self {
            workgroups_x: groups,
            workgroups_y: 1,
            workgroups_z: 1,
        }
    }

    /// 根据 `RowBitMask` 的有效跨度计算紧凑间接调度网格（全零掩码返回空网格）
    #[inline]
    pub fn from_active_mask(mask: &RowBitMask, workgroup_size: u32) -> Self {
        if mask.is_all_zero() {
            return Self {
                workgroups_x: 0,
                workgroups_y: 0,
                workgroups_z: 0,
            };
        }
        Self::from_row_count(mask.row_len(), workgroup_size)
    }
}

/// 将 `RowBitMask`（来自 `RowFlags`、`PairTargetIndex` 或 `ChangedSince`）封装为 16KB 页对齐的 GPU 谓词缓冲区
#[derive(Debug, Clone)]
pub struct GpuPredicationMaskBuffer {
    words_buf: PageAlignedBuffer<u64>,
    row_len: usize,
    active_rows: usize,
}

impl GpuPredicationMaskBuffer {
    /// 从 `RowBitMask` 构建 16KB 页对齐且尾页清零的 GPU 位平面掩码缓冲区
    pub fn from_row_bit_mask(mask: &RowBitMask) -> Self {
        let words_buf = PageAlignedBuffer::from_slice_zero_padded(mask.words());
        Self {
            words_buf,
            row_len: mask.row_len(),
            active_rows: mask.count_ones(),
        }
    }

    #[inline]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    #[inline]
    pub fn active_rows(&self) -> usize {
        self.active_rows
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        self.words_buf.as_slice()
    }

    #[inline]
    pub fn page_aligned_buffer(&self) -> &PageAlignedBuffer<u64> {
        &self.words_buf
    }

    /// 生成 Metal `newBufferWithBytesNoCopy` 零拷贝描述符（空掩码返回错误）
    pub fn metal_descriptor(&self) -> Result<MetalNoCopyBufferDescriptor, StateError> {
        let (ptr, allocated_bytes) = self.words_buf.as_metal_raw_parts();
        MetalNoCopyBufferDescriptor::validate_raw(
            ptr,
            allocated_bytes,
            self.words_buf.used_bytes(),
            self.words_buf.generation(),
        )
    }
}

/// 关系对 `Pair<R, Target>` 的 GPU 密集目标行索引查找表（LUT）
///
/// 将源原型表中每一行的 `Pair<R, EntityId>.target` 映射为目标原型表内的连续行号 `u32`。
/// 若目标实体不存在或不属于指定目标原型表，则对应槽位记为 `u32::MAX`（`INVALID_TARGET_ROW`）。
#[derive(Debug, Clone)]
pub struct PairTargetGpuLut {
    pub source_archetype_id: u32,
    pub target_archetype_id: u32,
    pub target_row_count: usize,
    pub valid_link_count: usize,
    pub target_rows: PageAlignedBuffer<u32>,
}

impl PairTargetGpuLut {
    /// 无效目标行哨兵值（在 WGSL / MSL 中通过 `target_row < target_row_count` 自动拦截）
    pub const INVALID_TARGET_ROW: u32 = u32::MAX;

    /// 从预解析的目标行切片构造 16KB 页对齐且尾页零填充的查找表
    pub fn from_target_rows(
        source_archetype_id: u32,
        target_archetype_id: u32,
        target_row_count: usize,
        rows: &[u32],
    ) -> Self {
        let mut buf = PageAlignedBuffer::with_capacity(rows.len());
        let mut valid_link_count = 0usize;
        for &r in rows {
            if (r as usize) < target_row_count {
                valid_link_count += 1;
            }
            buf.push(r);
        }
        buf.zero_pad_tail();
        Self {
            source_archetype_id,
            target_archetype_id,
            target_row_count,
            valid_link_count,
            target_rows: buf,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        self.target_rows.as_slice()
    }

    /// 生成 Metal `newBufferWithBytesNoCopy` 零拷贝描述符（空查找表返回错误）
    pub fn metal_descriptor(&self) -> Result<MetalNoCopyBufferDescriptor, StateError> {
        let (ptr, allocated_bytes) = self.target_rows.as_metal_raw_parts();
        MetalNoCopyBufferDescriptor::validate_raw(
            ptr,
            allocated_bytes,
            self.target_rows.used_bytes(),
            self.target_rows.generation(),
        )
    }
}

/// 表内行级位图动态过滤器闭包
type ArchetypeRowMaskFilter = Box<dyn Fn(&Archetype, &mut RowBitMask) -> bool + Send + Sync>;

/// 单张原型表在一次 `GpuMaskedPass` 中的调度遥测与脏页增量清单
#[derive(Debug, Clone)]
pub struct ArchetypeDispatchStats {
    pub archetype_id: u32,
    pub total_rows: usize,
    pub active_rows: usize,
    pub skipped_64row_chunks: usize,
    pub indirect_dispatch: IndirectDispatchArgs,
    pub dirty_page_runs: Vec<DirtyPageRun>,
    pub dirty_bytes: usize,
    pub total_column_page_bytes: usize,
}

/// `GpuMaskedPass` 执行完成后的异构调度收据
#[derive(Debug, Clone, Default)]
pub struct GpuDispatchReceipt {
    pub store_epoch: u64,
    pub matched_archetypes: usize,
    pub total_rows_scanned: usize,
    pub total_active_rows: usize,
    pub total_skipped_64row_chunks: usize,
    pub total_dirty_bytes: usize,
    pub total_column_page_bytes: usize,
    pub archetype_stats: Vec<ArchetypeDispatchStats>,
}

impl GpuDispatchReceipt {
    /// PCIe / NVLink 增量脏页传输节省率（`1.0 - dirty_bytes / total_column_page_bytes`）
    #[inline]
    pub fn dma_bandwidth_saved_ratio(&self) -> f64 {
        if self.total_column_page_bytes == 0 {
            0.0
        } else {
            let ratio = self.total_dirty_bytes as f64 / self.total_column_page_bytes as f64;
            (1.0 - ratio).clamp(0.0, 1.0)
        }
    }
}

/// 异构谓词掩码计算通道构建器与执行器 (`GpuMaskedPass`)
///
/// 支持叠加：
/// - `with_flag::<FLAG>()` / `without_flag::<FLAG>()`：行级非迁移位平面硬件掩码过滤；
/// - `with_pair::<R>(target)`：基于原型表内 `PairTargetIndex` 的零扫表目标位图过滤；
/// - `changed_since::<T>(since_epoch)`：基于 64 行物理块纪元的整块剪枝；
/// - 点对点行并行（Point-wise）、跨行关系目标收集（Relational Gather）、
///   跨行散射累加（Relational Scatter-Reduce）以及双精度 `f64` 归约（Threadgroup Reduction）。
pub struct GpuMaskedPass {
    workgroup_size: u32,
    page_size: usize,
    required_flags: Vec<u8>,
    excluded_flags: Vec<u8>,
    required_components: Vec<ComponentTypeId>,
    custom_mask_filters: Vec<ArchetypeRowMaskFilter>,
}

impl Default for GpuMaskedPass {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMaskedPass {
    /// 创建默认配置（`workgroup_size = 64`，硬件页粒度 `16KB`）的异构计算通道
    pub fn new() -> Self {
        Self {
            workgroup_size: 64,
            page_size: HARDWARE_PAGE_SIZE,
            required_flags: Vec::new(),
            excluded_flags: Vec::new(),
            required_components: Vec::new(),
            custom_mask_filters: Vec::new(),
        }
    }

    /// 配置线程组大小（如 Apple Silicon Simdgroup 32/64 或 CUDA Warp 32/128/256）
    pub fn with_workgroup_size(mut self, workgroup_size: u32) -> Self {
        self.workgroup_size = workgroup_size.max(1);
        self
    }

    /// 配置硬件物理页大小（如 4KB PCIe、16KB Apple UMA 或 64KB NVIDIA Grace Hopper NVLink-C2C）
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        assert!(
            page_size.is_power_of_two() && page_size >= 4096,
            "Hardware page size must be a power of two >= 4096"
        );
        self.page_size = page_size;
        self
    }

    /// 仅对行级位平面 `FLAG` 置 1 的活跃实体执行 GPU 计算
    pub fn with_flag<const FLAG: u8>(mut self) -> Self {
        self.required_flags.push(FLAG);
        self
    }

    /// 排除行级位平面 `FLAG` 置 1 的实体（如跳过 `Sleeping` / `Disabled` 粒子）
    pub fn without_flag<const FLAG: u8>(mut self) -> Self {
        self.excluded_flags.push(FLAG);
        self
    }

    /// 仅对包含关系对 `(R, target)` 的实体执行计算（自动利用原型内 `PairTargetIndex` 二级位图索引）
    pub fn with_pair<R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        mut self,
        target: EntityId,
    ) -> Self {
        let pair_cid = ComponentTypeId::of::<crate::al::Pair<R>>();
        self.required_components.push(pair_cid);
        self.custom_mask_filters.push(Box::new(move |arch, mask| {
            let Some(pair_mask) = arch.pair_target_row_mask::<R>(target) else {
                mask.words_mut().fill(0);
                return false;
            };
            mask.intersect_with(&pair_mask);
            !mask.is_all_zero()
        }));
        self
    }

    /// 仅对组件 `T` 所在 64 行物理块纪元大于 `since_epoch` 的增量脏块执行计算
    pub fn changed_since<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        mut self,
        since_epoch: u64,
    ) -> Self {
        let cid = ComponentTypeId::of::<T>();
        self.required_components.push(cid);
        self.custom_mask_filters.push(Box::new(move |arch, mask| {
            let (epochs, def_ep) = arch.effective_gpu_chunk_epochs(cid);
            let mut any_active = false;
            for (w_idx, word) in mask.words_mut().iter_mut().enumerate() {
                if *word == 0 {
                    continue;
                }
                let ep = epochs.get(w_idx).copied().unwrap_or(def_ep);
                if ep <= since_epoch {
                    *word = 0;
                } else {
                    any_active = true;
                }
            }
            any_active
        }));
        self
    }

    /// 计算单个原型表在所有位平面、关系索引与块纪元过滤后的联合 `RowBitMask`
    pub(crate) fn evaluate_archetype_mask(&self, arch: &Archetype) -> Option<RowBitMask> {
        let row_len = arch.len();
        if row_len == 0 {
            return None;
        }
        for &cid in &self.required_components {
            if !arch.contains_component(cid) {
                return None;
            }
        }
        let mut mask = RowBitMask::new_all_set(row_len);

        for &flag in &self.required_flags {
            let Some(words) = arch.flag_plane_words(flag) else {
                return None;
            };
            mask.intersect_words(words);
            if mask.is_all_zero() {
                return None;
            }
        }

        for &flag in &self.excluded_flags {
            if let Some(words) = arch.flag_plane_words(flag) {
                mask.difference_words(words);
                if mask.is_all_zero() {
                    return None;
                }
            }
        }

        for filter in &self.custom_mask_filters {
            if !filter(arch, &mut mask) {
                return None;
            }
        }

        if mask.is_all_zero() {
            None
        } else {
            Some(mask)
        }
    }

    /// 执行双组件原位并行核函数 `kernel(&mut Dst, &Src)`（如速度积分位置、场量阻尼更新）
    ///
    /// - 自动在 `RowBitMask` 上按 64 行机器字跳跃执行（`word == 0` 整块单周期跳过）；
    /// - 自动推进 `Store` 纪元，并精确更新被触碰的 64 行物理块纪元；
    /// - 自动计算 16KB/64KB 硬件脏页折叠 `DirtyPageRun`，生成 `GpuDispatchReceipt`。
    pub fn dispatch_binary_in_place<Dst, Src, K>(
        &self,
        store: &mut Store,
        mut kernel: K,
    ) -> Result<GpuDispatchReceipt, StateError>
    where
        Dst: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        Src: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        K: FnMut(usize, &mut Dst, &Src),
    {
        store.check_poisoned()?;
        let dst_cid = ComponentTypeId::of::<Dst>();
        let src_cid = ComponentTypeId::of::<Src>();
        if dst_cid == src_cid {
            return Err(StateError::BatchConflict(
                "Destination and source component types must be distinct in binary pass".into(),
            ));
        }

        let prev_epoch = store.epoch;
        let new_epoch = store
            .epoch
            .checked_add(1)
            .ok_or_else(|| StateError::CapacityExhausted("Store epoch exhausted".into()))?;
        store.epoch = new_epoch;

        let mut receipt = GpuDispatchReceipt {
            store_epoch: new_epoch,
            ..Default::default()
        };

        for arch in &mut store.archetypes {
            if !arch.contains_component(dst_cid) || !arch.contains_component(src_cid) {
                continue;
            }
            let Some(mask) = self.evaluate_archetype_mask(arch) else {
                continue;
            };

            let row_len = arch.len();
            let active_rows = mask.count_ones();
            let skipped_chunks = mask.words().iter().filter(|&&w| w == 0).count();

            // 构造 16KB 页对齐谓词缓冲区（验证 UMA 谓词绑定就绪）
            let _pred_buf = GpuPredicationMaskBuffer::from_row_bit_mask(&mask);
            let indirect_dispatch =
                IndirectDispatchArgs::from_active_mask(&mask, self.workgroup_size);

            // 直接通过零拷贝无借用冲突双列获取切片执行核函数
            if let Some((src_col, dst_col)) = arch.get_two_columns_ro_mut::<Src, Dst>() {
                let src_slice = src_col.as_slice();
                let dst_slice = dst_col.as_mut_slice();
                mask.for_each_set_row(|row| {
                    if let (Some(d), Some(s)) = (dst_slice.get_mut(row), src_slice.get(row)) {
                        kernel(row, d, s);
                    }
                });
            }

            // 仅推进掩码非零的 64 行物理块纪元
            arch.mark_column_masked_chunks_epoch(dst_cid, &mask, new_epoch);

            // 计算该列在本次 Pass 后折叠出的硬件脏页段（包含 swap_remove 物理搬迁页）
            let (epochs, def_ep) = arch.effective_gpu_chunk_epochs(dst_cid);
            let dirty_bitmap = PageDirtyBitmap::from_phys_chunk_epochs_with_page_size(
                &epochs,
                def_ep,
                prev_epoch,
                row_len,
                std::mem::size_of::<Dst>(),
                self.page_size,
            );

            let dirty_bytes = dirty_bitmap.total_dirty_bytes();
            let total_page_bytes = dirty_bitmap.total_pages() * self.page_size;

            receipt.matched_archetypes += 1;
            receipt.total_rows_scanned += row_len;
            receipt.total_active_rows += active_rows;
            receipt.total_skipped_64row_chunks += skipped_chunks;
            receipt.total_dirty_bytes += dirty_bytes;
            receipt.total_column_page_bytes += total_page_bytes;
            receipt.archetype_stats.push(ArchetypeDispatchStats {
                archetype_id: arch.id().0,
                total_rows: row_len,
                active_rows,
                skipped_64row_chunks: skipped_chunks,
                indirect_dispatch,
                dirty_page_runs: dirty_bitmap.dirty_runs().to_vec(),
                dirty_bytes,
                total_column_page_bytes: total_page_bytes,
            });
        }

        Ok(receipt)
    }

    /// 执行跨行关系对 `Pair<R, Target>` 目标收集核函数（Relational Cross-Row Gather）
    ///
    /// 打破传统 GPU ECS“单行隔离”限制：
    /// 源实体持有 `Pair<R, Target>` 与 `&mut Dst`，目标实体持有 `&TargetComp`（可位于同表或不同原型表）。
    /// 系统自动为每个源原型表生成 16KB 页对齐的 [`PairTargetGpuLut`]，将每个源行映射至目标实体所在行，
    /// 并在核函数中直接传入 `(source_row, &mut Dst, &TargetComp)`。
    pub fn dispatch_pair_gather<R, Dst, TargetComp, K>(
        &self,
        store: &mut Store,
        mut kernel: K,
    ) -> Result<GpuDispatchReceipt, StateError>
    where
        R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        Dst: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        TargetComp: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        K: FnMut(usize, &mut Dst, &TargetComp),
    {
        let pair_cid = ComponentTypeId::of::<crate::al::Pair<R>>();
        let dst_cid = ComponentTypeId::of::<Dst>();
        let target_cid = ComponentTypeId::of::<TargetComp>();

        let prev_epoch = store.epoch;
        let new_epoch = store
            .epoch
            .checked_add(1)
            .ok_or_else(|| StateError::CapacityExhausted("Store epoch exhausted".into()))?;
        store.epoch = new_epoch;

        // 1. 预先提取所有包含 TargetComp 的目标原型表只读快照（支持跨原型或同原型自引用 Gather）
        let mut target_tables: std::collections::BTreeMap<u32, Vec<TargetComp>> =
            std::collections::BTreeMap::new();
        for arch in &store.archetypes {
            if arch.contains_component(target_cid) {
                if let Some(col) = arch.get_column::<TargetComp>() {
                    target_tables.insert(arch.id().0, col.as_slice().to_vec());
                }
            }
        }

        // 2. 为匹配的源原型表构建 (target_arch_id, target_row) 映射并执行跨行 Gather
        let mut receipt = GpuDispatchReceipt {
            store_epoch: new_epoch,
            ..Default::default()
        };

        let allocator = &store.allocator;
        for arch in &mut store.archetypes {
            if !arch.contains_component(pair_cid) || !arch.contains_component(dst_cid) {
                continue;
            }
            let Some(mask) = self.evaluate_archetype_mask(arch) else {
                continue;
            };

            let row_len = arch.len();
            let active_rows = mask.count_ones();
            let skipped_chunks = mask.words().iter().filter(|&&w| w == 0).count();

            // 解析每一行的 Pair<R> 目标位置：(target_arch_id, target_row)
            let resolved_targets: Vec<Option<(u32, u32)>> = if let Some(pair_col) =
                arch.get_column::<crate::al::Pair<R>>()
            {
                pair_col
                    .as_slice()
                    .iter()
                    .map(|pair| {
                        allocator
                            .get_location(pair.target)
                            .map(|loc| (loc.archetype_id, loc.row_index))
                    })
                    .collect()
            } else {
                continue;
            };

            if let Some(dst_col) = arch.get_column_mut::<Dst>() {
                let dst_slice = dst_col.as_mut_slice();
                mask.for_each_set_row(|row| {
                    if let Some(Some((t_arch, t_row))) = resolved_targets.get(row) {
                        if let Some(t_slice) = target_tables.get(t_arch) {
                            if let (Some(d), Some(t_val)) =
                                (dst_slice.get_mut(row), t_slice.get(*t_row as usize))
                            {
                                kernel(row, d, t_val);
                            }
                        }
                    }
                });
            }

            arch.mark_column_masked_chunks_epoch(dst_cid, &mask, new_epoch);

            let (epochs, def_ep) = arch.effective_gpu_chunk_epochs(dst_cid);
            let dirty_bitmap = PageDirtyBitmap::from_phys_chunk_epochs_with_page_size(
                &epochs,
                def_ep,
                prev_epoch,
                row_len,
                std::mem::size_of::<Dst>(),
                self.page_size,
            );

            let dirty_bytes = dirty_bitmap.total_dirty_bytes();
            let total_page_bytes = dirty_bitmap.total_pages() * self.page_size;

            receipt.matched_archetypes += 1;
            receipt.total_rows_scanned += row_len;
            receipt.total_active_rows += active_rows;
            receipt.total_skipped_64row_chunks += skipped_chunks;
            receipt.total_dirty_bytes += dirty_bytes;
            receipt.total_column_page_bytes += total_page_bytes;
            receipt.archetype_stats.push(ArchetypeDispatchStats {
                archetype_id: arch.id().0,
                total_rows: row_len,
                active_rows,
                skipped_64row_chunks: skipped_chunks,
                indirect_dispatch: IndirectDispatchArgs::from_active_mask(
                    &mask,
                    self.workgroup_size,
                ),
                dirty_page_runs: dirty_bitmap.dirty_runs().to_vec(),
                dirty_bytes,
                total_column_page_bytes: total_page_bytes,
            });
        }

        Ok(receipt)
    }

    /// 执行跨行关系对 `Pair<R, Target>` 原子散射归约（Relational Cross-Row Scatter-Reduce）
    ///
    /// 源实体（如子粒子、弹簧边、网格单元）持有 `Pair<R, Target>` 与 `&SrcComp`，
    /// 将计算结果跨行累加/归约至目标实体（如父刚体、图中心节点）的 `&mut TargetComp` 上，
    /// 并精确标记目标实体所在 64 行物理块的脏纪元。
    pub fn dispatch_pair_scatter_reduce<R, SrcComp, TargetComp, K>(
        &self,
        store: &mut Store,
        mut accumulate: K,
    ) -> Result<GpuDispatchReceipt, StateError>
    where
        R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        SrcComp: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        TargetComp: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
        K: FnMut(&mut TargetComp, &SrcComp),
    {
        let pair_cid = ComponentTypeId::of::<crate::al::Pair<R>>();
        let src_cid = ComponentTypeId::of::<SrcComp>();
        let target_cid = ComponentTypeId::of::<TargetComp>();

        let prev_epoch = store.epoch;
        let new_epoch = store
            .epoch
            .checked_add(1)
            .ok_or_else(|| StateError::CapacityExhausted("Store epoch exhausted".into()))?;
        store.epoch = new_epoch;

        // 1. 收集所有活跃源行发出的 (target_arch_id, target_row, SrcComp) 散射项
        let mut scatter_ops: Vec<(u32, usize, SrcComp)> = Vec::new();
        let mut receipt = GpuDispatchReceipt {
            store_epoch: new_epoch,
            ..Default::default()
        };

        for arch in &store.archetypes {
            if !arch.contains_component(pair_cid) || !arch.contains_component(src_cid) {
                continue;
            }
            let Some(mask) = self.evaluate_archetype_mask(arch) else {
                continue;
            };
            let (Some(pair_col), Some(src_col)) = (
                arch.get_column::<crate::al::Pair<R>>(),
                arch.get_column::<SrcComp>(),
            ) else {
                continue;
            };
            let pairs = pair_col.as_slice();
            let srcs = src_col.as_slice();

            receipt.matched_archetypes += 1;
            receipt.total_rows_scanned += arch.len();
            receipt.total_active_rows += mask.count_ones();
            receipt.total_skipped_64row_chunks += mask.words().iter().filter(|&&w| w == 0).count();

            mask.for_each_set_row(|row| {
                if let (Some(p), Some(s)) = (pairs.get(row), srcs.get(row)) {
                    if let Some(loc) = store.allocator.get_location(p.target) {
                        scatter_ops.push((loc.archetype_id, loc.row_index as usize, s.clone()));
                    }
                }
            });
        }

        // 2. 按目标原型表应用归约并生成目标表的 64 行脏块掩码与硬件脏页段
        let mut touched_masks: std::collections::BTreeMap<u32, RowBitMask> =
            std::collections::BTreeMap::new();

        for (t_arch_id, t_row, src_val) in scatter_ops {
            if let Some(t_arch) = store.archetypes.get_mut(t_arch_id as usize) {
                if !t_arch.contains_component(target_cid) {
                    continue;
                }
                let t_len = t_arch.len();
                if let Some(t_col) = t_arch.get_column_mut::<TargetComp>() {
                    if let Some(dst_val) = t_col.as_mut_slice().get_mut(t_row) {
                        accumulate(dst_val, &src_val);
                        touched_masks
                            .entry(t_arch_id)
                            .or_insert_with(|| RowBitMask::new_empty(t_len))
                            .set(t_row);
                    }
                }
            }
        }

        for (t_arch_id, t_mask) in touched_masks {
            if let Some(t_arch) = store.archetypes.get_mut(t_arch_id as usize) {
                t_arch.mark_column_masked_chunks_epoch(target_cid, &t_mask, new_epoch);
                let (epochs, def_ep) = t_arch.effective_gpu_chunk_epochs(target_cid);
                let dirty_bitmap = PageDirtyBitmap::from_phys_chunk_epochs_with_page_size(
                    &epochs,
                    def_ep,
                    prev_epoch,
                    t_arch.len(),
                    std::mem::size_of::<TargetComp>(),
                    self.page_size,
                );
                let dirty_bytes = dirty_bitmap.total_dirty_bytes();
                let total_page_bytes = dirty_bitmap.total_pages() * self.page_size;
                receipt.total_dirty_bytes += dirty_bytes;
                receipt.total_column_page_bytes += total_page_bytes;
                receipt.archetype_stats.push(ArchetypeDispatchStats {
                    archetype_id: t_arch_id,
                    total_rows: t_arch.len(),
                    active_rows: t_mask.count_ones(),
                    skipped_64row_chunks: t_mask.words().iter().filter(|&&w| w == 0).count(),
                    indirect_dispatch: IndirectDispatchArgs::from_active_mask(
                        &t_mask,
                        self.workgroup_size,
                    ),
                    dirty_page_runs: dirty_bitmap.dirty_runs().to_vec(),
                    dirty_bytes,
                    total_column_page_bytes: total_page_bytes,
                });
            }
        }

        Ok(receipt)
    }

    /// 执行面向科学计算的双精度 `f64` 并行归约（如系统总动能、总质量或残差范数统计）
    pub fn dispatch_reduce_f64<T, M>(
        &self,
        store: &Store,
        map_row: M,
    ) -> (f64, GpuDispatchReceipt)
    where
        T: ModelComponent + Serialize + DeserializeOwned + 'static,
        M: Fn(&T) -> f64,
    {
        let cid = ComponentTypeId::of::<T>();
        let mut total_sum = 0.0f64;
        let mut receipt = GpuDispatchReceipt {
            store_epoch: store.epoch,
            ..Default::default()
        };

        for arch in &store.archetypes {
            if !arch.contains_component(cid) {
                continue;
            }
            let Some(mask) = self.evaluate_archetype_mask(arch) else {
                continue;
            };
            let Some(col) = arch.get_column::<T>() else {
                continue;
            };
            let slice = col.as_slice();
            let mut arch_sum = 0.0f64;
            mask.for_each_set_row(|row| {
                if let Some(val) = slice.get(row) {
                    arch_sum += map_row(val);
                }
            });
            total_sum += arch_sum;

            let row_len = arch.len();
            let active_rows = mask.count_ones();
            let skipped_chunks = mask.words().iter().filter(|&&w| w == 0).count();
            receipt.matched_archetypes += 1;
            receipt.total_rows_scanned += row_len;
            receipt.total_active_rows += active_rows;
            receipt.total_skipped_64row_chunks += skipped_chunks;
            receipt.archetype_stats.push(ArchetypeDispatchStats {
                archetype_id: arch.id().0,
                total_rows: row_len,
                active_rows,
                skipped_64row_chunks: skipped_chunks,
                indirect_dispatch: IndirectDispatchArgs::from_active_mask(
                    &mask,
                    self.workgroup_size,
                ),
                dirty_page_runs: Vec::new(),
                dirty_bytes: 0,
                total_column_page_bytes: 0,
            });
        }

        (total_sum, receipt)
    }
}

impl Store {
    /// 为指定源原型表与目标原型表构建 16KB 页对齐的 `Pair<R, Target>` GPU 目标行查找表 (`PairTargetGpuLut`)
    pub fn build_pair_target_gpu_lut<
        R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static,
    >(
        &self,
        source_archetype_id: u32,
        target_archetype_id: u32,
    ) -> Option<PairTargetGpuLut> {
        let src_arch = self.archetypes.get(source_archetype_id as usize)?;
        let dst_arch = self.archetypes.get(target_archetype_id as usize)?;
        let pair_col = src_arch.get_column::<crate::al::Pair<R>>()?;
        let dst_len = dst_arch.len();

        let rows: Vec<u32> = pair_col
            .as_slice()
            .iter()
            .map(|pair| {
                if let Some(loc) = self.allocator.get_location(pair.target) {
                    if loc.archetype_id == target_archetype_id && (loc.row_index as usize) < dst_len
                    {
                        return loc.row_index;
                    }
                }
                PairTargetGpuLut::INVALID_TARGET_ROW
            })
            .collect();

        Some(PairTargetGpuLut::from_target_rows(
            source_archetype_id,
            target_archetype_id,
            dst_len,
            &rows,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::query::QueryBuilder;
    use serde::Deserialize;

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct Pos3D {
        x: f64,
        y: f64,
        z: f64,
        w: f64, // 32 字节/行 -> 每 16KB 页 512 行（8 个 64 行物理块）
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct Vel3D {
        vx: f64,
        vy: f64,
        vz: f64,
        vw: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    struct ForceAccum {
        fx: f64,
        fy: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct AttachedTo;

    fn pos(x: f64, y: f64) -> Pos3D {
        Pos3D { x, y, z: 0.0, w: 1.0 }
    }

    #[test]
    fn test_64row_chunk_epochs_fold_to_16kb_dirty_page_runs() {
        // 2048 行 * 32B = 4 个 16KB 页；块 2、5 位于页 0，块 24 位于页 3
        let row_len = 2048;
        let elem_size = std::mem::size_of::<Pos3D>();
        assert_eq!(elem_size, 32);
        let mut chunk_epochs = vec![10u64; 32];
        chunk_epochs[2] = 15;
        chunk_epochs[5] = 15;
        chunk_epochs[24] = 16;

        let bitmap =
            PageDirtyBitmap::from_phys_chunk_epochs(&chunk_epochs, 10, 10, row_len, elem_size);
        assert_eq!(bitmap.total_pages(), 4);
        assert_eq!(bitmap.dirty_page_count(), 2);
        assert!(bitmap.is_page_dirty(0) && bitmap.is_page_dirty(3));
        assert!(!bitmap.is_page_dirty(1) && !bitmap.is_page_dirty(2));
        assert_eq!(
            bitmap.dirty_runs(),
            vec![
                DirtyPageRun {
                    first_page: 0,
                    page_count: 1,
                    byte_offset: 0,
                    byte_len: HARDWARE_PAGE_SIZE,
                },
                DirtyPageRun {
                    first_page: 3,
                    page_count: 1,
                    byte_offset: 3 * HARDWARE_PAGE_SIZE,
                    byte_len: HARDWARE_PAGE_SIZE,
                },
            ]
        );
    }

    #[test]
    fn test_gpu_compute_pass_with_row_flags_and_changed_since_sync() {
        let mut store = Store::new();
        for i in 0..600u128 {
            let e = EntityId::from_u128(5000 + i);
            store
                .insert((
                    e,
                    (
                        pos(i as f64, 0.0),
                        Vel3D {
                            vx: 10.0,
                            vy: 5.0,
                            vz: 0.0,
                            vw: 0.0,
                        },
                    ),
                ))
                .unwrap();
            if i == 10 || i == 20 {
                store.set_flag::<0>(e, true).unwrap();
            }
        }

        let epoch_before_gpu = store.epoch();
        let pass = GpuMaskedPass::new().with_flag::<0>();
        let receipt = pass
            .dispatch_binary_in_place::<Pos3D, Vel3D, _>(&mut store, |_row, pos, vel| {
                pos.x += vel.vx;
                pos.y += vel.vy;
            })
            .unwrap();

        // 600 行 = 10 个 64 行块，仅块 0 含活跃行 -> 跳过 9 块；600*32B 跨 2 页，仅页 0 脏
        assert_eq!(receipt.total_rows_scanned, 600);
        assert_eq!(receipt.total_active_rows, 2);
        assert_eq!(receipt.total_skipped_64row_chunks, 9);
        assert_eq!(receipt.total_dirty_bytes, HARDWARE_PAGE_SIZE);
        assert_eq!(receipt.total_column_page_bytes, 2 * HARDWARE_PAGE_SIZE);
        assert!((receipt.dma_bandwidth_saved_ratio() - 0.5).abs() < 1e-6);
        assert_eq!(
            store.get::<Pos3D>(EntityId::from_u128(5010)).unwrap().x,
            20.0
        );
        assert_eq!(store.get::<Pos3D>(EntityId::from_u128(5011)).unwrap().x, 11.0);

        // ChangedSince 仅命中被精确推进纪元的 64 行块
        let q_changed = QueryBuilder::<&Pos3D>::new()
            .changed_since::<Pos3D>(epoch_before_gpu)
            .build(&store)
            .unwrap();
        let mut changed_count = 0usize;
        q_changed
            .for_each(&store, |(_id, _pos)| changed_count += 1)
            .unwrap();
        assert_eq!(changed_count, 64);
    }

    #[test]
    fn test_cross_row_relational_pair_gather_and_scatter_reduce() {
        let mut store = Store::new();
        let anchor_a = EntityId::from_u128(1);
        let anchor_b = EntityId::from_u128(2);
        store
            .insert((anchor_a, (pos(100.0, 200.0), ForceAccum { fx: 0.0, fy: 0.0 })))
            .unwrap();
        store
            .insert((anchor_b, (pos(-50.0, -100.0), ForceAccum { fx: 0.0, fy: 0.0 })))
            .unwrap();
        for i in 0..20u128 {
            let target = if i % 2 == 0 { anchor_a } else { anchor_b };
            store
                .insert((
                    EntityId::from_u128(100 + i),
                    (
                        pos(1.0, 2.0),
                        ForceAccum { fx: 3.0, fy: 4.0 },
                        crate::al::Pair::new(AttachedTo, target),
                    ),
                ))
                .unwrap();
        }

        let pass = GpuMaskedPass::new();
        let gather_receipt = pass
            .dispatch_pair_gather::<AttachedTo, Pos3D, Pos3D, _>(
                &mut store,
                |_row, child_pos, anchor_pos| {
                    child_pos.x += anchor_pos.x;
                    child_pos.y += anchor_pos.y;
                },
            )
            .unwrap();
        assert_eq!(gather_receipt.total_active_rows, 20);
        assert_eq!(store.get::<Pos3D>(EntityId::from_u128(100)).unwrap().x, 101.0);
        assert_eq!(store.get::<Pos3D>(EntityId::from_u128(101)).unwrap().x, -49.0);

        let scatter_receipt = pass
            .dispatch_pair_scatter_reduce::<AttachedTo, ForceAccum, ForceAccum, _>(
                &mut store,
                |anchor_force, child_force| {
                    anchor_force.fx += child_force.fx;
                    anchor_force.fy += child_force.fy;
                },
            )
            .unwrap();
        assert_eq!(scatter_receipt.total_active_rows, 20);
        assert_eq!(
            store.get::<ForceAccum>(anchor_a).unwrap(),
            &ForceAccum { fx: 30.0, fy: 40.0 }
        );
        assert_eq!(
            store.get::<ForceAccum>(anchor_b).unwrap(),
            &ForceAccum { fx: 30.0, fy: 40.0 }
        );

        let (total_fx, _) = pass.dispatch_reduce_f64::<ForceAccum, _>(&store, |f| f.fx);
        assert!((total_fx - (30.0 + 30.0 + 20.0 * 3.0)).abs() < 1e-9);

        // 关系目标 LUT：子实体全部指向锚点所在原型表
        let src_arch = store.get_entity_location(EntityId::from_u128(100)).unwrap().archetype_id;
        let dst_arch = store.get_entity_location(anchor_a).unwrap().archetype_id;
        let lut = store
            .build_pair_target_gpu_lut::<AttachedTo>(src_arch, dst_arch)
            .unwrap();
        assert_eq!(lut.as_slice().len(), 20);
        assert_eq!(lut.valid_link_count, 20);
        assert!(lut.metal_descriptor().unwrap().allocated_bytes % HARDWARE_PAGE_SIZE == 0);
    }

    #[test]
    fn test_predication_mask_buffer_and_indirect_dispatch() {
        let mut mask = RowBitMask::new_empty(300);
        mask.set(3);
        mask.set(299);
        let buf = GpuPredicationMaskBuffer::from_row_bit_mask(&mask);
        assert_eq!(buf.row_len(), 300);
        assert_eq!(buf.active_rows(), 2);
        assert_eq!(buf.words().len(), 5);
        let desc = buf.metal_descriptor().unwrap();
        assert_eq!(desc.base_ptr as usize % HARDWARE_PAGE_SIZE, 0);

        let args = IndirectDispatchArgs::from_active_mask(&mask, 64);
        assert_eq!((args.workgroups_x, args.workgroups_y, args.workgroups_z), (5, 1, 1));
        let empty = IndirectDispatchArgs::from_active_mask(&RowBitMask::new_empty(300), 64);
        assert_eq!(empty.workgroups_x, 0);
    }
}
