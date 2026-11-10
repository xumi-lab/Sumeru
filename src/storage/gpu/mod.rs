//! 16KB 硬件页对齐 GPU 零拷贝视图、脏页位图与统一内存抽象 (Sumeru GPU Compute Layer)
//!
//! # 架构概览
//!
//! 1. **M0 — 16KB 硬件页对齐分配器 (`PageAlignedBuffer<T>`)**：
//!    物理列直接以 `16384` 字节对齐并在尾部补齐至 `16384` 字节整数倍，零填充尾页以防跨行越界读取。
//! 2. **M1 — 零拷贝 GPU 物理页视图与 16KB 脏页位图 (`ColumnGpuView` / `PageDirtyBitmap`)**：
//!    将 64 行物理修改纪元 `phys_chunk_epochs`（覆盖 `swap_remove` 尾行回填）折叠映射为 16KB 硬件页级脏位图与合并连续传输段 `DirtyPageRun`。
//! 3. **M2 — Apple Metal UMA 零拷贝与双后端管线 (`metal`)**：
//!    直接将 16KB 页对齐指针封装为 `MTLBuffer (StorageModeShared, NoCopy)`，按 `(ptr, bytes, generation)` 缓存句柄，并计算最优 Threadgroup 网格。
//! 4. **M3 — 多平台着色器内核与三层遍历体系 (`kernel`)**：
//!    通过 `#[gpu]` 统一宏同时生成 Metal Shading Language (MSL)、WebGPU (WGSL) 与 CPU 算子，支持：
//!    - `.for_each(...)`（CPU 单线程顺序执行）
//!    - `.par_for_each(...)` / `.par().for_each(...)`（CPU 多核 Rayon 并行执行）
//!    - `.gpu().for_each(...)`（GPU 硬件 16KB 页对齐零拷贝派发）

pub mod ir;
pub mod kernel;
pub mod metal;
pub mod pass;

#[cfg(test)]
mod tests;

pub use ir::*;
pub use kernel::*;
pub use metal::*;
pub use pass::{
    ArchetypeDispatchStats, GpuDispatchReceipt, GpuMaskedPass, GpuPredicationMaskBuffer,
    IndirectDispatchArgs, PairTargetGpuLut,
};

use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::{ArchetypeId, ComponentTypeId};
use crate::storage::column::page_aligned::HARDWARE_PAGE_SIZE;
use crate::storage::column::ColumnBlock;
use crate::storage::store::Store;
use bytemuck::{Pod, Zeroable};
use serde::de::DeserializeOwned;
use serde::Serialize;
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

/// 可直接映射至 GPU 缓冲区的 `#[repr(C)]` POD 组件契约。
///
/// # Safety
/// 实现类型必须为无填充未初始化字节的 `Pod + Zeroable` 平坦结构，且大小 > 0、对齐不超过 16KB。
pub unsafe trait GpuComponent:
    crate::storage::column::ArchiveComponent
    + ModelComponent
    + Copy
    + Serialize
    + DeserializeOwned
    + Pod
    + Zeroable
    + Send
    + Sync
    + 'static
{
    /// 编译期断言组件满足 GPU 页对齐要求。
    const IS_GPU_SAFE: bool = {
        assert!(
            std::mem::size_of::<Self>() > 0,
            "GpuComponent cannot be a zero-sized type (ZST)"
        );
        assert!(
            std::mem::align_of::<Self>() <= HARDWARE_PAGE_SIZE,
            "GpuComponent alignment exceeds 16KB hardware page size"
        );
        true
    };

    /// 触发编译期布局校验。
    #[inline(always)]
    fn assert_gpu_layout() {
        let _ = Self::IS_GPU_SAFE;
    }
}

/// 列的 16KB 硬件页几何信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageGeometry {
    /// 单个组件元素的字节大小 (`size_of::<T>()`)
    pub elem_size: usize,
    /// 当前活跃元素行数
    pub elem_count: usize,
    /// 活跃数据占用的精确字节数 (`elem_count * elem_size`)
    pub active_bytes: usize,
    /// 底层 16KB 页对齐分配的总字节数（16384 的整数倍）
    pub allocated_bytes: usize,
    /// 活跃数据覆盖的 16KB 硬件页数
    pub page_count: usize,
    /// 底层已分配的 16KB 硬件页总数
    pub allocated_page_count: usize,
}

impl PageGeometry {
    /// 计算给定元素大小、活跃行数与分配字节数的 16KB 页几何描述。
    #[inline]
    pub fn compute(elem_size: usize, elem_count: usize, allocated_bytes: usize) -> Self {
        let active_bytes = elem_count.saturating_mul(elem_size);
        let page_count = active_bytes.div_ceil(HARDWARE_PAGE_SIZE);
        let allocated_page_count = allocated_bytes / HARDWARE_PAGE_SIZE;
        Self {
            elem_size,
            elem_count,
            active_bytes,
            allocated_bytes,
            page_count,
            allocated_page_count,
        }
    }

    /// 单个 16KB 页可完整容纳的元素个数（若元素跨页则至少返回 1）。
    #[inline]
    pub fn elems_per_page(&self) -> usize {
        HARDWARE_PAGE_SIZE
            .checked_div(self.elem_size)
            .map(|v| v.max(1))
            .unwrap_or(0)
    }

    /// 给定行索引所在的首个 16KB 页索引。
    #[inline]
    pub fn page_for_row(&self, row: usize) -> usize {
        (row.saturating_mul(self.elem_size)) / HARDWARE_PAGE_SIZE
    }

    /// 给定 16KB 页索引覆盖的活跃行索引范围 `start..end`（按行起始字节归属，跨页不重复计数）。
    #[inline]
    pub fn row_range_for_page(&self, page_idx: usize) -> Range<usize> {
        if self.elem_size == 0 || self.elem_count == 0 || page_idx >= self.page_count {
            return 0..0;
        }
        let page_start_byte = page_idx * HARDWARE_PAGE_SIZE;
        let page_end_byte = ((page_idx + 1) * HARDWARE_PAGE_SIZE).min(self.active_bytes);
        let start_row = page_start_byte.div_ceil(self.elem_size).min(self.elem_count);
        let end_row = page_end_byte.div_ceil(self.elem_size).min(self.elem_count);
        start_row..end_row.max(start_row)
    }
}

/// 统一脏页类型（唯一定义点：[`crate::storage::primitives::dirty`]）。
///
/// `gpu` feature 仅在此追加基于列几何的扩展构造方法，不再重复定义类型，保证 feature 可加性。
pub use crate::storage::primitives::{DirtyPageRun, PageDirtyBitmap};

impl PageDirtyBitmap {
    /// 从 64 行粒度的物理修改纪元数组 (`phys_chunk_epochs`) 与 16KB 页对齐列几何构建脏页位图。
    ///
    /// 任何 `chunk_epoch > since_epoch` 的 64 行块，其字节跨度 `[chunk_start_byte, chunk_end_byte)`
    /// 触及的所有 16KB 硬件页均被标记为脏页；`chunk_epochs == None` 时按 `default_epoch` 整列判定。
    pub fn from_column_geometry(
        elem_size: usize,
        elem_count: usize,
        allocated_bytes: usize,
        chunk_epochs: Option<&[u64]>,
        default_epoch: u64,
        since_epoch: u64,
    ) -> Self {
        let geom = PageGeometry::compute(elem_size, elem_count, allocated_bytes);
        if geom.page_count == 0 {
            return Self::new_clean(0);
        }
        let Some(epochs) = chunk_epochs else {
            return if default_epoch > since_epoch {
                Self::new_all_dirty(geom.page_count)
            } else {
                Self::new_clean(geom.page_count)
            };
        };
        Self::from_phys_chunk_epochs(epochs, default_epoch, since_epoch, elem_count, elem_size)
    }
}

/// 利用 `Store::component_archetypes` 倒排位图索引对所需组件集合进行 64-bit word 求交剪枝。
#[inline]
fn candidate_archetypes_from_inverted_index(
    store: &Store,
    required_cids: &[ComponentTypeId],
) -> Option<crate::storage::primitives::ArchetypeBitSet> {
    let (&first_cid, rest) = required_cids.split_first()?;
    let comp_index = store.component_archetypes();
    let Some(first_bits) = comp_index.get(&first_cid) else {
        return Some(crate::storage::primitives::ArchetypeBitSet::new());
    };
    let mut candidate = first_bits.clone();
    for &cid in rest {
        let Some(bits) = comp_index.get(&cid) else {
            candidate.clear();
            return Some(candidate);
        };
        candidate.intersect_with(bits);
        if candidate.is_all_zero() {
            return Some(candidate);
        }
    }
    Some(candidate)
}

/// 单列的只读 16KB 页对齐 GPU 视图。
#[derive(Debug, Clone)]
pub struct ColumnGpuView<'a, T: GpuComponent> {
    base_ptr: NonNull<u8>,
    allocated_bytes: usize,
    generation: u64,
    len: usize,
    geometry: PageGeometry,
    dirty_bitmap: PageDirtyBitmap,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T: GpuComponent> ColumnGpuView<'a, T> {
    /// 构造列的只读 GPU 视图。
    #[inline]
    pub(crate) fn new(
        base_ptr: NonNull<u8>,
        allocated_bytes: usize,
        generation: u64,
        len: usize,
        dirty_bitmap: PageDirtyBitmap,
    ) -> Self {
        T::assert_gpu_layout();
        let geometry = PageGeometry::compute(std::mem::size_of::<T>(), len, allocated_bytes);
        Self {
            base_ptr,
            allocated_bytes,
            generation,
            len,
            geometry,
            dirty_bitmap,
            _marker: PhantomData,
        }
    }

    /// 物理基地址指针（16KB 对齐）。
    #[inline]
    pub fn base_ptr(&self) -> *const u8 {
        self.base_ptr.as_ptr()
    }

    /// 检查底层缓冲区是否严格满足 16KB (`16384` bytes) 首地址与容量对齐（可直接用于 Metal `newBufferWithBytesNoCopy`）。
    #[inline]
    pub fn is_zero_copy_aligned(&self) -> bool {
        (self.base_ptr.as_ptr() as usize).is_multiple_of(HARDWARE_PAGE_SIZE)
            && self.allocated_bytes.is_multiple_of(HARDWARE_PAGE_SIZE)
    }

    /// 底层 16KB 页对齐分配的总字节数。
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 缓冲区重分配代际号（用于 `MTLBuffer` 缓存命中校验）。
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 活跃元素行数。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 列是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 16KB 页几何信息。
    #[inline]
    pub fn geometry(&self) -> PageGeometry {
        self.geometry
    }

    /// 16KB 硬件页级脏位图。
    #[inline]
    pub fn dirty_bitmap(&self) -> &PageDirtyBitmap {
        &self.dirty_bitmap
    }

    /// 活跃元素的强类型切片。
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr.as_ptr() as *const T, self.len) }
    }

    /// 完整 16KB 页对齐分配区的字节切片（含已零初始化的尾页安全填充区）。
    #[inline]
    pub fn as_padded_page_bytes(&self) -> &'a [u8] {
        if self.allocated_bytes == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr.as_ptr(), self.allocated_bytes) }
    }

    /// 获取指定 16KB 页的 `16384` 字节切片。
    #[inline]
    pub fn page_bytes(&self, page_idx: usize) -> Option<&'a [u8]> {
        if page_idx >= self.geometry.allocated_page_count {
            return None;
        }
        let offset = page_idx * HARDWARE_PAGE_SIZE;
        Some(&self.as_padded_page_bytes()[offset..offset + HARDWARE_PAGE_SIZE])
    }

    /// 导出合并后的连续脏页段及其对应的对齐字节切片（专用于独显增量上传）。
    pub fn dirty_runs(&self) -> SmallVec<[(DirtyPageRun, &'a [u8]); 4]> {
        let padded = self.as_padded_page_bytes();
        self.dirty_bitmap
            .coalesced_runs()
            .into_iter()
            .filter_map(|run| {
                let end = run.byte_offset.checked_add(run.byte_len)?;
                if end <= padded.len() {
                    Some((run, &padded[run.byte_offset..end]))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// 单列的可变 16KB 页对齐 GPU 视图（用于 GPU Compute Shader 原位写回或 CPU/GPU 统一内存交互）。
#[derive(Debug)]
pub struct ColumnGpuMutView<'a, T: GpuComponent> {
    base_ptr: NonNull<u8>,
    allocated_bytes: usize,
    generation: u64,
    len: usize,
    geometry: PageGeometry,
    write_epoch: u64,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: GpuComponent> ColumnGpuMutView<'a, T> {
    #[inline]
    pub(crate) fn new(
        base_ptr: NonNull<u8>,
        allocated_bytes: usize,
        generation: u64,
        len: usize,
        write_epoch: u64,
    ) -> Self {
        T::assert_gpu_layout();
        let geometry = PageGeometry::compute(std::mem::size_of::<T>(), len, allocated_bytes);
        Self {
            base_ptr,
            allocated_bytes,
            generation,
            len,
            geometry,
            write_epoch,
            _marker: PhantomData,
        }
    }

    /// 物理基地址可变指针（16KB 对齐）。
    #[inline]
    pub fn base_mut_ptr(&mut self) -> *mut u8 {
        self.base_ptr.as_ptr()
    }

    /// 检查底层缓冲区是否严格满足 16KB (`16384` bytes) 首地址与容量对齐。
    #[inline]
    pub fn is_zero_copy_aligned(&self) -> bool {
        (self.base_ptr.as_ptr() as usize).is_multiple_of(HARDWARE_PAGE_SIZE)
            && self.allocated_bytes.is_multiple_of(HARDWARE_PAGE_SIZE)
    }

    /// 底层 16KB 页对齐分配的总字节数。
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 缓冲区重分配代际号。
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 活跃元素行数。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 列是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 16KB 页几何信息。
    #[inline]
    pub fn geometry(&self) -> PageGeometry {
        self.geometry
    }

    /// 本次可变视图绑定的 Store 写入纪元。
    #[inline]
    pub fn write_epoch(&self) -> u64 {
        self.write_epoch
    }

    /// 活跃元素的强类型只读切片。
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr.as_ptr() as *const T, self.len) }
    }

    /// 活跃元素的强类型可变切片。
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.base_ptr.as_ptr() as *mut T, self.len) }
    }

    /// 完整 16KB 页对齐分配区的可变字节切片。
    #[inline]
    pub fn as_padded_page_bytes_mut(&mut self) -> &mut [u8] {
        if self.allocated_bytes == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.base_ptr.as_ptr(), self.allocated_bytes) }
    }
}

/// 异步 GPU 派发期间固定底层 16KB 页对齐内存块生命周期的零拷贝句柄。
///
/// 通过持有 `Arc<ColumnBlock<T>>` 引用计数：
/// 1. 即使在 GPU 异步执行期间 CPU 端对 `Store` 发起写入，COW 机制也会为 CPU 克隆新块，
///    而 GPU 正在读取的物理内存页保持绝对不可变与有效，杜绝悬垂指针（Use-After-Free）与数据竞争。
#[derive(Debug, Clone)]
pub struct GpuColumnPin<T: GpuComponent> {
    archetype_id: ArchetypeId,
    block: Arc<ColumnBlock<T>>,
    base_ptr: *const u8,
    allocated_bytes: usize,
    generation: u64,
    len: usize,
}

// SAFETY: `T: GpuComponent` 约束了 `Send + Sync`，且 `Arc<ColumnBlock<T>>` 在ピン留期间只读共享。
unsafe impl<T: GpuComponent> Send for GpuColumnPin<T> {}
unsafe impl<T: GpuComponent> Sync for GpuColumnPin<T> {}

impl<T: GpuComponent> GpuColumnPin<T> {
    /// 所属 Archetype ID。
    #[inline]
    pub fn archetype_id(&self) -> ArchetypeId {
        self.archetype_id
    }

    /// 16KB 页对齐基地址指针。
    #[inline]
    pub fn base_ptr(&self) -> *const u8 {
        self.base_ptr
    }

    /// 16KB 页对齐分配字节数。
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 物理分配代际号。
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 活跃元素数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 强类型元素只读切片。
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let _ = &self.block;
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr as *const T, self.len) }
    }
}

/// 跨 Archetype 16KB 虚拟页表项（供 Compute Shader 在单次 Dispatch 中跨多张原型表寻址）。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuPageDescriptor {
    /// 所属 Archetype ID
    pub archetype_id: u32,
    /// 该页在对应 Archetype 列缓冲区内的页偏移（单位：16KB 页）
    pub column_page_index: u32,
    /// 该页起始行在全局逻辑线程空间中的起始索引
    pub global_row_start: u32,
    /// 该页内的有效活跃行数
    pub active_row_count: u32,
}

/// 跨 Archetype 汇总的 GPU 16KB 物理页表。
#[derive(Debug, Clone, Default)]
pub struct GpuPageTable {
    descriptors: Vec<GpuPageDescriptor>,
    total_active_rows: u32,
}

impl GpuPageTable {
    /// 页描述符切片（可直接作为 Uniform / Storage Buffer 传入 Compute Shader）。
    #[inline]
    pub fn descriptors(&self) -> &[GpuPageDescriptor] {
        &self.descriptors
    }

    /// 汇总的全局有效行总数。
    #[inline]
    pub fn total_active_rows(&self) -> u32 {
        self.total_active_rows
    }

    /// 汇总的 16KB 物理页总数。
    #[inline]
    pub fn total_pages(&self) -> usize {
        self.descriptors.len()
    }
}

/// 支持批量注册 GPU 查询中涉及的所有组件为 16KB 页对齐模式的辅助 Trait。
pub trait RegisterGpuQuery {
    /// 将查询涉及的所有组件类型注册到 `Store` 并提升为 16KB 硬件页对齐。
    fn register_in_store(store: &mut Store) -> Result<(), StateError>;
}

impl<A: GpuComponent> RegisterGpuQuery for &A {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        store.register_gpu_component::<A>()
    }
}

impl<A: GpuComponent> RegisterGpuQuery for &mut A {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        store.register_gpu_component::<A>()
    }
}

impl<A: RegisterGpuQuery> RegisterGpuQuery for (A,) {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)
    }
}

impl<A: RegisterGpuQuery, B: RegisterGpuQuery> RegisterGpuQuery for (A, B) {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)
    }
}

impl<A: RegisterGpuQuery, B: RegisterGpuQuery, C: RegisterGpuQuery> RegisterGpuQuery for (A, B, C) {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)
    }
}

impl<A: RegisterGpuQuery, B: RegisterGpuQuery, C: RegisterGpuQuery, D: RegisterGpuQuery>
    RegisterGpuQuery for (A, B, C, D)
{
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)?;
        D::register_in_store(store)
    }
}

impl<
    A: RegisterGpuQuery,
    B: RegisterGpuQuery,
    C: RegisterGpuQuery,
    D: RegisterGpuQuery,
    E: RegisterGpuQuery,
> RegisterGpuQuery for (A, B, C, D, E)
{
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)?;
        D::register_in_store(store)?;
        E::register_in_store(store)
    }
}

impl<
    A: RegisterGpuQuery,
    B: RegisterGpuQuery,
    C: RegisterGpuQuery,
    D: RegisterGpuQuery,
    E: RegisterGpuQuery,
    F: RegisterGpuQuery,
> RegisterGpuQuery for (A, B, C, D, E, F)
{
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)?;
        D::register_in_store(store)?;
        E::register_in_store(store)?;
        F::register_in_store(store)
    }
}

impl<
    A: RegisterGpuQuery,
    B: RegisterGpuQuery,
    C: RegisterGpuQuery,
    D: RegisterGpuQuery,
    E: RegisterGpuQuery,
    F: RegisterGpuQuery,
    G: RegisterGpuQuery,
> RegisterGpuQuery for (A, B, C, D, E, F, G)
{
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)?;
        D::register_in_store(store)?;
        E::register_in_store(store)?;
        F::register_in_store(store)?;
        G::register_in_store(store)
    }
}

impl<
    A: RegisterGpuQuery,
    B: RegisterGpuQuery,
    C: RegisterGpuQuery,
    D: RegisterGpuQuery,
    E: RegisterGpuQuery,
    F: RegisterGpuQuery,
    G: RegisterGpuQuery,
    H: RegisterGpuQuery,
> RegisterGpuQuery for (A, B, C, D, E, F, G, H)
{
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)?;
        D::register_in_store(store)?;
        E::register_in_store(store)?;
        F::register_in_store(store)?;
        G::register_in_store(store)?;
        H::register_in_store(store)
    }
}

impl Store {
    /// 注册 GPU 友好的 `#[repr(C)]` POD 组件（统一采用 `ColumnCodecId::Rkyv` 归档），并将其在所有原型中的物理列提升为 16KB (`16384` bytes) 页对齐缓冲区。
    pub fn register_gpu_component<T: GpuComponent>(&mut self) -> Result<(), StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let mut reg = crate::storage::column::ComponentRegistration::of::<T>();
        reg.column.promote_to_page_aligned();
        self.register_component_prototypes(vec![reg.column])
    }

    /// 一次性注册查询元组 `Q`（如 `(&mut Pos, &Vel)`）涉及的所有 GPU 组件并开启 16KB 页对齐。
    #[inline]
    pub fn register_gpu_query<Q: RegisterGpuQuery>(&mut self) -> Result<(), StateError> {
        Q::register_in_store(self)
    }

    /// 提取所有包含组件 `T` 的非空 Archetype 的只读 16KB 页对齐 GPU 视图与自 `since_epoch` 以来的 16KB 脏页位图。
    pub fn gpu_column_views<T: GpuComponent>(
        &self,
        since_epoch: u64,
    ) -> Result<Vec<(ArchetypeId, ColumnGpuView<'_, T>)>, StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let cid = ComponentTypeId::of::<T>();
        let elem_size = std::mem::size_of::<T>();
        let mut out = Vec::new();
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let Some(col) = arch.columns().get(&cid) else {
                return Err(StateError::ComponentNotFound(format!("{cid}")));
            };
            if col.len() != len {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} length {} does not match archetype {} row count {}",
                    std::any::type_name::<T>(),
                    col.len(),
                    arch.id().0,
                    len
                )));
            }
            let Some((ptr, allocated_bytes)) = col.metal_raw_parts() else {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} in archetype {} is not 16KB page-aligned; call store.register_gpu_component::<{}>() or use LayoutOptimization::PageAlignedGpu",
                    std::any::type_name::<T>(),
                    arch.id().0,
                    std::any::type_name::<T>(),
                )));
            };
            let active_bytes = len.checked_mul(elem_size).ok_or_else(|| {
                StateError::CapacityExhausted("GPU column active byte size overflow".into())
            })?;
            if active_bytes > allocated_bytes {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})",
                    std::any::type_name::<T>()
                )));
            }
            let generation = col.physical_version();
            let (merged_epochs, def_ep) = arch.effective_gpu_chunk_epochs(cid);
            let dirty = PageDirtyBitmap::from_column_geometry(
                elem_size,
                len,
                allocated_bytes,
                if merged_epochs.is_empty() {
                    None
                } else {
                    Some(merged_epochs.as_slice())
                },
                def_ep,
                since_epoch,
            );
            let base_ptr = NonNull::new(ptr).expect("PageAlignedBuffer pointer is non-null");
            out.push((
                arch.id(),
                ColumnGpuView::new(base_ptr, allocated_bytes, generation, len, dirty),
            ));
        }
        Ok(out)
    }

    /// 提取所有包含组件 `T` 的非空 Archetype 的可变 16KB 页对齐 GPU 视图，强制执行 COW 独占分离并自动推进 Store 纪元与块纪元。
    pub fn gpu_column_views_mut<T: GpuComponent>(
        &mut self,
    ) -> Result<Vec<(ArchetypeId, ColumnGpuMutView<'_, T>)>, StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let cid = ComponentTypeId::of::<T>();
        let elem_size = std::mem::size_of::<T>();

        // 阶段 1a：全表只读预校验（确保任意 Archetype 布局异常时绝不产生半边提升或 COW 剥离副作用）
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let arch_id = arch.id().0;
            let col = arch
                .columns()
                .get(&cid)
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            if col.len() != len {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} length {} does not match archetype {} row count {}",
                    std::any::type_name::<T>(),
                    col.len(),
                    arch_id,
                    len
                )));
            }
            let _ = len.checked_mul(elem_size).ok_or_else(|| {
                StateError::CapacityExhausted("GPU column active byte size overflow".into())
            })?;
        }

        // 阶段 1b：16KB 页对齐提升与 COW 独占分离（确保被 Pin / Clone 共享的块绝不被原位覆写）
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let arch_id = arch.id().0;
            let col = arch
                .get_erased_column_mut(cid)
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            if !col.is_page_aligned() {
                col.promote_to_page_aligned();
            }
            col.ensure_contiguous()?;
            let Some((_ptr, allocated_bytes)) = col.metal_raw_parts() else {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} in archetype {} failed to promote to 16KB page-aligned buffer",
                    std::any::type_name::<T>(),
                    arch_id
                )));
            };
            let active_bytes = len * elem_size;
            if active_bytes > allocated_bytes {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})",
                    std::any::type_name::<T>()
                )));
            }
            arch.rebuild_flat_columns();
        }

        // 阶段 2：原子推进纪元并导出独占可变视图
        self.epoch = next_epoch;
        let mut out = Vec::new();
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            arch.mark_column_row_range_epoch(cid, 0..len, next_epoch);
            let col = arch
                .columns()
                .get(&cid)
                .expect("checked component presence");
            let (ptr, allocated_bytes) = col
                .metal_raw_parts()
                .expect("column promoted and COW-unfrozen to 16KB page-aligned");
            let generation = col.physical_version();
            let base_ptr = NonNull::new(ptr).expect("PageAlignedBuffer pointer is non-null");
            out.push((
                arch.id(),
                ColumnGpuMutView::new(base_ptr, allocated_bytes, generation, len, next_epoch),
            ));
        }
        Ok(out)
    }

    /// 为所有包含组件 `T` 的非空 Archetype 创建 COW 安全的零拷贝 `GpuColumnPin<T>`，供异步 GPU 命令缓冲区安全读取。
    pub fn pin_gpu_columns<T: GpuComponent>(&mut self) -> Result<Vec<GpuColumnPin<T>>, StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let cid = ComponentTypeId::of::<T>();
        let elem_size = std::mem::size_of::<T>();

        // 阶段 1a：全表只读预校验（确保任意 Archetype 布局异常时绝不产生半边提升或冻结副作用）
        for arch in &self.archetypes {
            if arch.is_empty() || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let arch_id = arch.id();
            let arch_len = arch.len();
            let typed = arch
                .get_column::<T>()
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            if typed.len() != arch_len {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} length {} does not match archetype {} row count {}",
                    std::any::type_name::<T>(),
                    typed.len(),
                    arch_id.0,
                    arch_len
                )));
            }
            let _ = arch_len.checked_mul(elem_size).ok_or_else(|| {
                StateError::CapacityExhausted("GPU column active byte size overflow".into())
            })?;
        }

        // 阶段 1b：提升为 16KB 页对齐并导出 COW 零拷贝 Pin
        let mut pins = Vec::new();
        for arch in &mut self.archetypes {
            if arch.is_empty() || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let mut promoted = false;
            if let Some(col) = arch.get_erased_column_mut(cid)
                && !col.is_page_aligned()
            {
                col.promote_to_page_aligned();
                promoted = true;
            }
            if promoted {
                arch.rebuild_flat_columns();
            }
            let arch_id = arch.id();
            let arch_len = arch.len();
            let typed = arch
                .get_column_mut::<T>()
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            let (base_ptr, allocated_bytes) = typed
                .as_metal_raw_parts()
                .expect("column promoted to 16KB page-aligned");
            let active_bytes = arch_len * elem_size;
            if active_bytes > allocated_bytes {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})",
                    std::any::type_name::<T>()
                )));
            }
            let generation = typed.physical_version();
            let block = typed.share_root_block();
            pins.push(GpuColumnPin {
                archetype_id: arch_id,
                block,
                base_ptr,
                allocated_bytes,
                generation,
                len: arch_len,
            });
        }
        Ok(pins)
    }

    /// 构建组件 `T` 跨所有匹配 Archetype 的 16KB 硬件页表（供单次 GPU Dispatch 跨表寻址）。
    pub fn build_gpu_page_table<T: GpuComponent>(&self) -> Result<GpuPageTable, StateError> {
        let views = self.gpu_column_views::<T>(0)?;
        let mut descriptors = Vec::new();
        let mut global_row_start = 0u32;
        for (arch_id, view) in views {
            let geom = view.geometry();
            for page_idx in 0..geom.page_count {
                let range = geom.row_range_for_page(page_idx);
                let active_row_count = (range.end - range.start) as u32;
                descriptors.push(GpuPageDescriptor {
                    archetype_id: arch_id.0,
                    column_page_index: page_idx as u32,
                    global_row_start,
                    active_row_count,
                });
                global_row_start = global_row_start.saturating_add(active_row_count);
            }
        }
        Ok(GpuPageTable {
            descriptors,
            total_active_rows: global_row_start,
        })
    }

    /// 开启一个连续多 Kernel GPU 计算通道会话 (`GpuComputePass`)，消除逐次阻塞等待产生的 CPU-GPU 气泡。
    pub fn gpu_compute_pass<'a>(
        &'a mut self,
        engine: &'a mut MetalComputeEngine,
    ) -> Result<GpuComputePass<'a>, GpuExecutionError> {
        self.check_poisoned()?;
        let encoder = engine.begin_compute_pass()?;
        Ok(GpuComputePass::new(self, engine, encoder))
    }
}

/// 长期可复用查询句柄别名（与 `crate::storage::query::Query` 一致）
pub type QueryHandle<Q, F = crate::storage::query::All> = crate::storage::query::Query<Q, F>;

/// 多 Kernel 连续执行的 GPU 计算通道会话 (Multi-Kernel GPU Compute Pass)
///
/// 将多个连续 GPU Kernel 派发合并记录至单个 Metal 命令缓冲区中统一提交与同步，
/// 消除多 Query 连续执行时逐次阻塞等待产生的 CPU-GPU 气泡。
pub struct GpuComputePass<'a> {
    store: &'a mut Store,
    engine: &'a mut MetalComputeEngine,
    encoder: Box<dyn GpuPassEncoder + 'a>,
    pending_writes: Vec<(ArchetypeId, ComponentTypeId)>,
    reads_since_barrier: std::collections::HashSet<(ArchetypeId, ComponentTypeId)>,
    writes_since_barrier: std::collections::HashSet<(ArchetypeId, ComponentTypeId)>,
    pass_stats: GpuDispatchStats,
    dispatches: usize,
}

impl<'a> GpuComputePass<'a> {
    /// 构造新的多 Kernel 计算通道会话
    pub fn new(
        store: &'a mut Store,
        engine: &'a mut MetalComputeEngine,
        encoder: Box<dyn GpuPassEncoder + 'a>,
    ) -> Self {
        let execution_mode = engine.execution_mode();
        Self {
            store,
            engine,
            encoder,
            pending_writes: Vec::new(),
            reads_since_barrier: std::collections::HashSet::new(),
            writes_since_barrier: std::collections::HashSet::new(),
            pass_stats: GpuDispatchStats {
                execution_mode,
                ..Default::default()
            },
            dispatches: 0,
        }
    }

    fn check_and_record_hazards(
        &mut self,
        arch_ids: &[ArchetypeId],
        bindings: &[GpuKernelBindingSpec],
    ) -> Result<(), GpuExecutionError> {
        let mut needs_barrier = false;
        'outer: for &arch_id in arch_ids {
            for b in bindings {
                let key = (arch_id, b.type_id);
                if self.writes_since_barrier.contains(&key)
                    || (b.mutable && self.reads_since_barrier.contains(&key))
                {
                    needs_barrier = true;
                    break 'outer;
                }
            }
        }
        if needs_barrier {
            self.barrier()?;
        }
        for &arch_id in arch_ids {
            for b in bindings {
                let key = (arch_id, b.type_id);
                self.reads_since_barrier.insert(key);
                if b.mutable {
                    self.writes_since_barrier.insert(key);
                }
            }
        }
        Ok(())
    }

    /// 获取底层只读 Store 引用
    #[inline]
    pub fn store(&self) -> &Store {
        self.store
    }

    /// 获取底层计算引擎引用
    #[inline]
    pub fn engine(&self) -> &MetalComputeEngine {
        self.engine
    }

    /// 累计派发次数
    #[inline]
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }

    /// 获取当前计算通道累计统计数据
    #[inline]
    pub fn stats(&self) -> &GpuDispatchStats {
        &self.pass_stats
    }

    /// 在当前计算通道内追加派发一个强类型 GPU 内核计算任务（无 CPU 等待阻塞）
    ///
    /// 自动对所有匹配的原型表执行：
    /// 1. **阶段 0**：编译并缓存内核管线；
    /// 2. **阶段 1a**：校验列完整性与元素大小；
    /// 3. **阶段 1b**：按需执行 16KB 页对齐提升、可变列 COW 独占分离与零拷贝指针提取；
    /// 4. **编码阶段**：自动检测 RAW/WAW/WAR 依赖插入显存屏障，计算 SIMD-32 线程组网格并编码分派指令。
    pub fn dispatch<'k, K: GpuKernel<'k>>(
        &mut self,
        kernel: K,
    ) -> Result<&mut Self, GpuExecutionError> {
        self.store.check_poisoned()?;
        let desc = kernel.descriptor();
        let pipeline = self.engine.get_pipeline_handle(desc)?;
        let max_threads = pipeline.max_threads_per_threadgroup;
        let scalar_args = kernel.scalar_uniform_words();

        // 阶段 1a: 利用倒排位图索引匹配非空原型并预校验列完整性
        let binding_cids: SmallVec<[ComponentTypeId; 8]> =
            desc.bindings().iter().map(|b| b.type_id).collect();
        let mut candidate_indices = Vec::new();
        if let Some(bitset) = candidate_archetypes_from_inverted_index(self.store, &binding_cids) {
            bitset.for_each_set_bit(|id| candidate_indices.push(id as usize));
        } else {
            candidate_indices.extend(0..self.store.archetypes.len());
        }

        let mut matched_indices = Vec::new();
        let mut matched_arch_ids = Vec::new();
        for arch_idx in candidate_indices {
            let Some(archetype) = self.store.archetypes.get(arch_idx) else {
                continue;
            };
            let active_rows = archetype.len();
            if active_rows == 0 {
                continue;
            }
            let comp_slice = archetype.component_types_slice();
            if !desc
                .bindings()
                .iter()
                .all(|b| comp_slice.binary_search(&b.type_id).is_ok())
            {
                continue;
            }
            let arch_id = archetype.id();
            let cols = archetype.columns();
            for b in desc.bindings() {
                let col = cols.get(&b.type_id).ok_or_else(|| {
                    StateError::ComponentNotFound(format!("{}", b.type_id))
                })?;
                if col.len() != active_rows {
                    return Err(GpuExecutionError::State(StateError::AccessDenied(format!(
                        "Column {} length {} does not match archetype {} active_rows {}",
                        b.struct_name,
                        col.len(),
                        arch_id.0,
                        active_rows
                    ))));
                }
                let _active_bytes = active_rows.checked_mul(b.elem_size).ok_or_else(|| {
                    GpuExecutionError::State(StateError::CapacityExhausted(
                        "GPU column active byte size overflow".into(),
                    ))
                })?;
            }
            matched_indices.push(arch_idx);
            matched_arch_ids.push(arch_id);
        }

        self.check_and_record_hazards(&matched_arch_ids, desc.bindings())?;

        // 阶段 1b: 16KB 页对齐提升、COW 独占分离与命令编码
        for arch_idx in matched_indices {
            let archetype = &mut self.store.archetypes[arch_idx];
            let active_rows = archetype.len();
            let arch_id = archetype.id();
            let mut changed_any = false;
            for b in desc.bindings() {
                let needs_promote = archetype
                    .columns()
                    .get(&b.type_id)
                    .is_some_and(|col| !col.is_page_aligned());
                if needs_promote || b.mutable {
                    let col = archetype
                        .get_erased_column_mut(b.type_id)
                        .expect("verified in Phase 1a");
                    if needs_promote {
                        col.promote_to_page_aligned();
                    }
                    if b.mutable {
                        col.ensure_contiguous()?;
                    }
                    changed_any = true;
                }
            }
            if changed_any {
                archetype.rebuild_flat_columns();
            }

            let mut raw_buffers = Vec::with_capacity(desc.bindings().len());
            let mut arch_pages = 0usize;
            {
                let cols = archetype.columns();
                for b in desc.bindings() {
                    let col = cols.get(&b.type_id).expect("verified in Phase 1a");
                    let (ptr, allocated_bytes) = col.metal_raw_parts().ok_or_else(|| {
                        StateError::AccessDenied(format!(
                            "Column {} in archetype {} is not backed by a 16KB PageAlignedBuffer",
                            b.struct_name, arch_id.0
                        ))
                    })?;
                    let generation = col.physical_version();
                    if let Ok(buf_desc) = MetalNoCopyBufferDescriptor::validate_raw(
                        ptr,
                        allocated_bytes,
                        active_rows * b.elem_size,
                        generation,
                    ) {
                        let (_, hit, _) = self
                            .engine
                            .buffer_cache_mut()
                            .get_or_insert_with_evicted(arch_id, b.type_id, &buf_desc);
                        if hit {
                            self.pass_stats.buffer_cache_hits += 1;
                        } else {
                            self.pass_stats.buffer_cache_misses += 1;
                        }
                    }
                    let geom = PageGeometry::compute(b.elem_size, active_rows, allocated_bytes);
                    arch_pages += geom.page_count;
                    raw_buffers.push((ptr, allocated_bytes));
                    if b.mutable {
                        self.pending_writes.push((arch_id, b.type_id));
                    }
                }
            }

            let grid = MetalThreadgroupDispatch::compute_1d(active_rows, max_threads);
            let uniforms = GpuDispatchUniforms {
                active_rows: grid.active_rows,
                threads_per_group: grid.threads_per_threadgroup,
                archetype_id: arch_id.0,
                _pad: 0,
                scalar_args,
            };

            self.encoder
                .encode_dispatch(&pipeline, &grid, &raw_buffers, &uniforms)?;

            self.pass_stats.dispatched_archetypes += 1;
            self.pass_stats.total_rows += active_rows;
            self.pass_stats.total_pages_16k += arch_pages;
            self.dispatches += 1;
        }

        Ok(self)
    }

    /// `dispatch` 的算子直传别名
    #[inline]
    pub fn for_each<'k, K: GpuKernel<'k>>(
        &mut self,
        kernel: K,
    ) -> Result<&mut Self, GpuExecutionError> {
        self.dispatch(kernel)
    }

    /// 在当前计算通道内追加派发一个带有动态标量参数的 GPU 内核计算任务
    #[inline]
    pub fn dispatch_with_args<'k, K: GpuParameterizedKernel<'k>, A, const N: usize>(
        &mut self,
        kernel: K,
        args: A,
    ) -> Result<&mut Self, GpuExecutionError>
    where
        A: IntoGpuScalarUniforms<N>,
    {
        self.dispatch(GpuBoundKernel {
            kernel,
            scalar_words: args.into_scalar_words(),
        })
    }

    /// 基于已编译查询句柄及其过滤规则在当前通道内追加派发
    pub fn encode_query<Q, F, K>(
        &mut self,
        query: &crate::storage::query::Query<Q, F>,
        kernel: K,
    ) -> Result<&mut Self, GpuExecutionError>
    where
        Q: crate::storage::query::QueryMutShape + crate::storage::query::QueryMutData<'static>,
        F: crate::storage::query::QueryFilter + Clone,
        K: GpuKernel<'static, Query = Q>,
    {
        if F::IS_ROW_FILTERED {
            return Err(GpuExecutionError::State(StateError::AccessDenied(
                "GPU whole-column dispatch does not support row-level CPU predicate filters; use archetype filters (.with / .without)".into(),
            )));
        }
        self.store.check_poisoned()?;
        let matched = query.with_matched_slice(self.store, |_, _| Ok(query.matched_archetypes_arc()))?;
        let desc = kernel.descriptor();
        let pipeline = self.engine.get_pipeline_handle(desc)?;
        let max_threads = pipeline.max_threads_per_threadgroup;
        let scalar_args = kernel.scalar_uniform_words();

        let mut valid_indices = Vec::with_capacity(matched.len());
        let mut valid_arch_ids = Vec::with_capacity(matched.len());
        for &arch_id in matched.iter() {
            let arch_idx = arch_id.0 as usize;
            let Some(archetype) = self.store.archetypes.get(arch_idx) else {
                continue;
            };
            let active_rows = archetype.len();
            if active_rows == 0 || !query.filter.matches_archetype(archetype) {
                continue;
            }
            let comp_slice = archetype.component_types_slice();
            if !desc
                .bindings()
                .iter()
                .all(|b| comp_slice.binary_search(&b.type_id).is_ok())
            {
                continue;
            }
            let cols = archetype.columns();
            for b in desc.bindings() {
                let col = cols.get(&b.type_id).ok_or_else(|| {
                    StateError::ComponentNotFound(format!("{}", b.type_id))
                })?;
                if col.len() != active_rows {
                    return Err(GpuExecutionError::State(StateError::AccessDenied(format!(
                        "Column {} length {} does not match archetype {} active_rows {}",
                        b.struct_name,
                        col.len(),
                        arch_id.0,
                        active_rows
                    ))));
                }
            }
            valid_indices.push(arch_idx);
            valid_arch_ids.push(arch_id);
        }

        self.check_and_record_hazards(&valid_arch_ids, desc.bindings())?;

        for arch_idx in valid_indices {
            let archetype = &mut self.store.archetypes[arch_idx];
            let active_rows = archetype.len();
            let arch_id = archetype.id();
            let mut changed_any = false;
            for b in desc.bindings() {
                let needs_promote = archetype
                    .columns()
                    .get(&b.type_id)
                    .is_some_and(|col| !col.is_page_aligned());
                if needs_promote || b.mutable {
                    let col = archetype
                        .get_erased_column_mut(b.type_id)
                        .expect("verified in Phase 1a");
                    if needs_promote {
                        col.promote_to_page_aligned();
                    }
                    if b.mutable {
                        col.ensure_contiguous()?;
                    }
                    changed_any = true;
                }
            }
            if changed_any {
                archetype.rebuild_flat_columns();
            }

            let mut raw_buffers = Vec::with_capacity(desc.bindings().len());
            let mut arch_pages = 0usize;
            {
                let cols = archetype.columns();
                for b in desc.bindings() {
                    let col = cols.get(&b.type_id).expect("verified in Phase 1a");
                    let (ptr, allocated_bytes) = col.metal_raw_parts().ok_or_else(|| {
                        StateError::AccessDenied(format!(
                            "Column {} in archetype {} is not backed by a 16KB PageAlignedBuffer",
                            b.struct_name, arch_id.0
                        ))
                    })?;
                    let generation = col.physical_version();
                    if let Ok(buf_desc) = MetalNoCopyBufferDescriptor::validate_raw(
                        ptr,
                        allocated_bytes,
                        active_rows * b.elem_size,
                        generation,
                    ) {
                        let (_, hit, _) = self
                            .engine
                            .buffer_cache_mut()
                            .get_or_insert_with_evicted(arch_id, b.type_id, &buf_desc);
                        if hit {
                            self.pass_stats.buffer_cache_hits += 1;
                        } else {
                            self.pass_stats.buffer_cache_misses += 1;
                        }
                    }
                    let geom = PageGeometry::compute(b.elem_size, active_rows, allocated_bytes);
                    arch_pages += geom.page_count;
                    raw_buffers.push((ptr, allocated_bytes));
                    if b.mutable {
                        self.pending_writes.push((arch_id, b.type_id));
                    }
                }
            }

            let grid = MetalThreadgroupDispatch::compute_1d(active_rows, max_threads);
            let uniforms = GpuDispatchUniforms {
                active_rows: grid.active_rows,
                threads_per_group: grid.threads_per_threadgroup,
                archetype_id: arch_id.0,
                _pad: 0,
                scalar_args,
            };

            self.encoder
                .encode_dispatch(&pipeline, &grid, &raw_buffers, &uniforms)?;

            self.pass_stats.dispatched_archetypes += 1;
            self.pass_stats.total_rows += active_rows;
            self.pass_stats.total_pages_16k += arch_pages;
            self.dispatches += 1;
        }

        Ok(self)
    }

    /// 在当前计算通道内追加派发一个通过 `.bind(...)` 构造的 [`KernelLaunch`] 任务（自动推导读写冲突并插入屏障）
    pub fn encode_kernel_launch(
        &mut self,
        launch: &KernelLaunch<'_>,
    ) -> Result<&mut Self, GpuExecutionError> {
        self.store.check_poisoned()?;
        let valid_indices = launch.resolve_matched_archetype_indices(self.store)?;
        let desc = launch.descriptor;
        let pipeline = self.engine.get_pipeline_handle(desc)?;
        let max_threads = pipeline.max_threads_per_threadgroup;
        let scalar_args = launch.scalar_words;

        let valid_arch_ids: Vec<ArchetypeId> = valid_indices
            .iter()
            .map(|&idx| self.store.archetypes[idx].id())
            .collect();
        self.check_and_record_hazards(&valid_arch_ids, desc.bindings())?;

        for arch_idx in valid_indices {
            let archetype = &mut self.store.archetypes[arch_idx];
            let active_rows = archetype.len();
            let arch_id = archetype.id();
            let mut changed_any = false;
            for b in desc.bindings() {
                let needs_promote = archetype
                    .columns()
                    .get(&b.type_id)
                    .is_some_and(|col| !col.is_page_aligned());
                if needs_promote || b.mutable {
                    let col = archetype
                        .get_erased_column_mut(b.type_id)
                        .expect("verified in Phase 1a");
                    if needs_promote {
                        col.promote_to_page_aligned();
                    }
                    if b.mutable {
                        col.ensure_contiguous()?;
                    }
                    changed_any = true;
                }
            }
            if changed_any {
                archetype.rebuild_flat_columns();
            }

            let mut raw_buffers = Vec::with_capacity(desc.bindings().len());
            let mut arch_pages = 0usize;
            {
                let cols = archetype.columns();
                for b in desc.bindings() {
                    let col = cols.get(&b.type_id).expect("verified in Phase 1a");
                    let (ptr, allocated_bytes) = col.metal_raw_parts().ok_or_else(|| {
                        StateError::AccessDenied(format!(
                            "Column {} in archetype {} is not backed by a 16KB PageAlignedBuffer",
                            b.struct_name, arch_id.0
                        ))
                    })?;
                    let generation = col.physical_version();
                    if let Ok(buf_desc) = MetalNoCopyBufferDescriptor::validate_raw(
                        ptr,
                        allocated_bytes,
                        active_rows * b.elem_size,
                        generation,
                    ) {
                        let (_, hit, _) = self
                            .engine
                            .buffer_cache_mut()
                            .get_or_insert_with_evicted(arch_id, b.type_id, &buf_desc);
                        if hit {
                            self.pass_stats.buffer_cache_hits += 1;
                        } else {
                            self.pass_stats.buffer_cache_misses += 1;
                        }
                    }
                    let geom = PageGeometry::compute(b.elem_size, active_rows, allocated_bytes);
                    arch_pages += geom.page_count;
                    raw_buffers.push((ptr, allocated_bytes));
                    if b.mutable {
                        self.pending_writes.push((arch_id, b.type_id));
                    }
                }
            }

            let grid = MetalThreadgroupDispatch::compute_1d(active_rows, max_threads);
            let uniforms = GpuDispatchUniforms {
                active_rows: grid.active_rows,
                threads_per_group: grid.threads_per_threadgroup,
                archetype_id: arch_id.0,
                _pad: 0,
                scalar_args,
            };

            self.encoder
                .encode_dispatch(&pipeline, &grid, &raw_buffers, &uniforms)?;

            self.pass_stats.dispatched_archetypes += 1;
            self.pass_stats.total_rows += active_rows;
            self.pass_stats.total_pages_16k += arch_pages;
            self.dispatches += 1;
        }

        Ok(self)
    }

    /// 在连续 Kernel 之间插入 GPU 显存屏障（`memoryBarrierWithScope: MTLBarrierScopeBuffers`）
    #[inline]
    pub fn barrier(&mut self) -> Result<&mut Self, GpuExecutionError> {
        self.encoder.memory_barrier()?;
        self.pass_stats.barriers_inserted += 1;
        self.reads_since_barrier.clear();
        self.writes_since_barrier.clear();
        Ok(self)
    }

    /// 结束编码通道、提交硬件命令缓冲区并执行单次等待同步，批量推进 Store 与写入列的 64 行块纪元
    pub fn finish(mut self) -> Result<GpuDispatchStats, GpuExecutionError> {
        self.encoder.finish()?;
        if self.dispatches > 0 {
            let write_epoch = self.store.next_epoch()?;
            self.store.bump_epoch();
            for &(arch_id, cid) in &self.pending_writes {
                let arch_idx = arch_id.0 as usize;
                if let Some(arch) = self.store.archetypes.get_mut(arch_idx) {
                    arch.mark_column_all_chunks_epoch(cid, write_epoch);
                }
            }
            if self.engine.has_hardware_gpu() {
                self.pass_stats.command_buffer_submissions = 1;
            }
        }
        self.pass_stats.execution_mode = self.engine.execution_mode();
        Ok(self.pass_stats)
    }

    /// `finish` 的无返回值调用别名
    #[inline]
    pub fn execute(self) -> Result<(), GpuExecutionError> {
        self.finish()?;
        Ok(())
    }
}

// ============================================================================
// M2 & M3: 强类型列查询解构 (`QueryCol` / `queries_mut`)、`SharedColCache` 增量缓存、
//          Rayon `IntoParallelIterator` 与多算子元组管线 (`KernelLaunch` / `ComputeTask`)
// ============================================================================

/// 擦除具体过滤链类型的原型级匹配契约（供 [`KernelLaunch`] 与 [`SharedColCache`] 使用）
pub(crate) trait ErasedArchetypeFilter: Send + Sync {
    /// 是否包含行级 CPU 谓词过滤器
    fn is_row_filtered(&self) -> bool;
    /// 检查指定原型表是否满足过滤条件
    fn matches_archetype(&self, archetype: &crate::storage::archetype::Archetype) -> bool;
}

impl<F: crate::storage::query::QueryFilter> ErasedArchetypeFilter for F {
    #[inline]
    fn is_row_filtered(&self) -> bool {
        F::IS_ROW_FILTERED
    }

    #[inline]
    fn matches_archetype(&self, archetype: &crate::storage::archetype::Archetype) -> bool {
        crate::storage::query::QueryFilter::matches_archetype(self, archetype)
    }
}

/// 跨列共享的原型匹配增量缓存（对标 [`crate::storage::query::Query`] 内部的 `QueryCache`）
///
/// - 当未发生结构性变更（CR 增删组件/原型）时，后续算子派发与 Rayon 迭代 100% 命中缓存（0 原型重扫）；
/// - 当新增 Archetype 且未发生拓扑缩减时，按 $O(\Delta A)$ 仅增量扫描新增的原型表切片。
#[derive(Debug)]
pub struct SharedColCache {
    pub(crate) required_cids: SmallVec<[ComponentTypeId; 8]>,
    pub(crate) archetype_set_generation: u64,
    pub(crate) topology_epoch: u64,
    pub(crate) validated_archetype_len: usize,
    pub(crate) matched_archetypes: Arc<[ArchetypeId]>,
    pub(crate) cache_hits: std::sync::atomic::AtomicUsize,
    pub(crate) cache_misses: std::sync::atomic::AtomicUsize,
}

impl SharedColCache {
    /// 创建冷态共享列缓存
    #[inline]
    pub fn new(mut required_cids: SmallVec<[ComponentTypeId; 8]>) -> Self {
        required_cids.sort_unstable();
        required_cids.dedup();
        Self {
            required_cids,
            archetype_set_generation: u64::MAX,
            topology_epoch: u64::MAX,
            validated_archetype_len: 0,
            matched_archetypes: Arc::from([]),
            cache_hits: std::sync::atomic::AtomicUsize::new(0),
            cache_misses: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 当链式附加 `.with::<T>()` 或 `.without::<T>()` 时重置已缓存列表（惰性延迟至首次执行时构建）
    #[inline]
    pub fn invalidate_for_filter_change(&mut self) {
        self.archetype_set_generation = u64::MAX;
        self.topology_epoch = u64::MAX;
        self.validated_archetype_len = 0;
        self.matched_archetypes = Arc::from([]);
    }

    #[inline]
    fn is_up_to_date(&self, store: &Store) -> bool {
        self.archetype_set_generation == store.archetype_set_generation()
            && self.topology_epoch == store.archetype_topology_epoch()
            && self.validated_archetype_len == store.archetypes.len()
    }

    #[inline]
    fn archetype_has_required_components(
        &self,
        archetype: &crate::storage::archetype::Archetype,
    ) -> bool {
        let comp_slice = archetype.component_types_slice();
        self.required_cids
            .iter()
            .all(|cid| comp_slice.binary_search(cid).is_ok())
    }

    /// 解析当前匹配的原型表 ID 切片（命中时 $O(1)$ 零分配，新增原型时 $O(\Delta A)$ 增量刷新）
    pub(crate) fn resolve(
        cache_lock: &std::sync::RwLock<Self>,
        store: &Store,
        filter: &dyn ErasedArchetypeFilter,
    ) -> Arc<[ArchetypeId]> {
        {
            let guard = cache_lock.read().unwrap_or_else(|p| p.into_inner());
            if guard.is_up_to_date(store) {
                guard
                    .cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Arc::clone(&guard.matched_archetypes);
            }
        }

        let mut guard = cache_lock.write().unwrap_or_else(|p| p.into_inner());
        if guard.is_up_to_date(store) {
            guard
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Arc::clone(&guard.matched_archetypes);
        }

        guard
            .cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let current_gen = store.archetype_set_generation();
        let current_topo = store.archetype_topology_epoch();
        let current_len = store.archetypes.len();

        // O(\Delta A) 增量刷新路径：拓扑未重置且仅尾部追加了新 Archetype
        if guard.topology_epoch == current_topo && guard.validated_archetype_len <= current_len {
            if guard.validated_archetype_len < current_len {
                let mut delta = Vec::new();
                for archetype in &store.archetypes[guard.validated_archetype_len..current_len] {
                    if guard.archetype_has_required_components(archetype)
                        && filter.matches_archetype(archetype)
                    {
                        delta.push(archetype.id());
                    }
                }
                if !delta.is_empty() {
                    let mut all = Vec::with_capacity(guard.matched_archetypes.len() + delta.len());
                    all.extend_from_slice(&guard.matched_archetypes);
                    all.extend(delta);
                    guard.matched_archetypes = Arc::from(all.into_boxed_slice());
                }
            }
            guard.validated_archetype_len = current_len;
            guard.archetype_set_generation = current_gen;
            return Arc::clone(&guard.matched_archetypes);
        }

        // 全量构建路径（首次冷启动或拓扑重建后）：优先利用倒排 ArchetypeBitSet 64-bit word 求交剪枝
        let mut matched = Vec::new();
        if let Some(bitset) =
            candidate_archetypes_from_inverted_index(store, &guard.required_cids)
        {
            bitset.for_each_set_bit(|arch_raw| {
                if let Some(archetype) = store.archetypes.get(arch_raw as usize)
                    && guard.archetype_has_required_components(archetype)
                    && filter.matches_archetype(archetype)
                {
                    matched.push(archetype.id());
                }
            });
        } else {
            for archetype in &store.archetypes {
                if guard.archetype_has_required_components(archetype)
                    && filter.matches_archetype(archetype)
                {
                    matched.push(archetype.id());
                }
            }
        }
        guard.matched_archetypes = Arc::from(matched.into_boxed_slice());
        guard.archetype_set_generation = current_gen;
        guard.topology_epoch = current_topo;
        guard.validated_archetype_len = current_len;
        Arc::clone(&guard.matched_archetypes)
    }
}

/// 强类型单列查询视图（由 [`Store::queries_mut`] 或 [`Store::queries`] 解构生成）
///
/// - 共享 [`SharedColCache`] 原型匹配缓存（无 CR 变更时重复派发/迭代 0 重算，新增原型时 $O(\Delta A)$ 增量刷新）；
/// - 原生实现 [`IntoIterator`] 与 [`rayon::iter::IntoParallelIterator`]，支持 `(&mut pos, &vel).into_par_iter()`；
/// - 可直接传给 `#[kernel]` / `#[gpu]` 算子的 `.run(...)`、`.par(...)`、`.gpu(...)` 与 `.bind(...)`。
pub struct QueryCol<'a, Access, F = crate::storage::query::All> {
    pub(crate) store_ptr: NonNull<Store>,
    pub(crate) write_epoch: u64,
    pub(crate) filter: F,
    pub(crate) shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    pub(crate) _marker: PhantomData<(&'a mut Store, Access)>,
}

// SAFETY: `QueriesMutSplit::validate_no_alias` 保证同一元组解构出的所有可变 `QueryCol` 指向互不重叠的独立组件物理列，
// 且并行迭代在主线程预先提取互不重叠的连续切片 `&mut [T]` / `&[T]`。
unsafe impl<'a, Access: Send, F: Send + Sync> Send for QueryCol<'a, Access, F> {}
unsafe impl<'a, Access: Sync, F: Send + Sync> Sync for QueryCol<'a, Access, F> {}

impl<'a, Access, F: std::fmt::Debug> std::fmt::Debug for QueryCol<'a, Access, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCol")
            .field("type", &std::any::type_name::<Access>())
            .field("write_epoch", &self.write_epoch)
            .field("filter", &self.filter)
            .finish()
    }
}

impl<'a, Access, F: crate::storage::query::QueryFilter> QueryCol<'a, Access, F> {
    /// 附加原型级包含过滤规则 `.with::<T>()`
    #[inline]
    pub fn with<T: 'static>(
        self,
    ) -> QueryCol<'a, Access, crate::storage::query::With<T, F>> {
        if let Ok(mut guard) = self.shared_cache.write() {
            guard.invalidate_for_filter_change();
        }
        QueryCol {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            filter: crate::storage::query::With {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            shared_cache: self.shared_cache,
            _marker: PhantomData,
        }
    }

    /// 附加原型级排除过滤规则 `.without::<T>()`
    #[inline]
    pub fn without<T: 'static>(
        self,
    ) -> QueryCol<'a, Access, crate::storage::query::Without<T, F>> {
        if let Ok(mut guard) = self.shared_cache.write() {
            guard.invalidate_for_filter_change();
        }
        QueryCol {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            filter: crate::storage::query::Without {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            shared_cache: self.shared_cache,
            _marker: PhantomData,
        }
    }

    /// 获取底层 [`SharedColCache`] 的累计缓存命中次数
    #[inline]
    pub fn cache_hits(&self) -> usize {
        self.shared_cache
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .cache_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 获取底层 [`SharedColCache`] 的累计缓存未命中（构建/增量刷新）次数
    #[inline]
    pub fn cache_misses(&self) -> usize {
        self.shared_cache
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .cache_misses
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 显式触发或刷新底层 [`SharedColCache`]，返回当前匹配的 [`ArchetypeId`] 列表快照
    #[inline]
    pub fn matched_archetypes(&self) -> Arc<[ArchetypeId]> {
        let store = unsafe { self.store_ptr.as_ref() };
        SharedColCache::resolve(&self.shared_cache, store, &self.filter)
    }
}

impl<'a, T, F> QueryCol<'a, &'a T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned,
    F: crate::storage::query::QueryFilter,
{
    pub(crate) fn collect_read_slices<'b>(&'b self) -> (Vec<&'b [T]>, usize) {
        let store: &'b Store = unsafe { self.store_ptr.as_ref() };
        let matched = SharedColCache::resolve(&self.shared_cache, store, &self.filter);
        let mut slices = Vec::with_capacity(matched.len());
        let mut total_len = 0usize;
        for &arch_id in matched.iter() {
            let Some(arch) = store.archetypes.get(arch_id.0 as usize) else {
                continue;
            };
            let len = arch.len();
            if len == 0 || !self.filter.matches_archetype(arch) {
                continue;
            }
            if let Some(col) = arch.get_column::<T>() {
                let s = col.as_slice();
                if s.len() == len {
                    total_len += len;
                    slices.push(s);
                }
            }
        }
        (slices, total_len)
    }

    /// 顺序遍历列中的所有只读元素 `&T`
    #[inline]
    pub fn iter(&self) -> MultiSliceIter<'_, T> {
        let (slices, total_len) = self.collect_read_slices();
        MultiSliceIter::new(slices, total_len)
    }
}

impl<'a, T, F> QueryCol<'a, &'a mut T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned,
    F: crate::storage::query::QueryFilter,
{
    pub(crate) fn collect_read_slices_from_mut<'b>(&'b self) -> (Vec<&'b [T]>, usize) {
        let store: &'b Store = unsafe { self.store_ptr.as_ref() };
        let matched = SharedColCache::resolve(&self.shared_cache, store, &self.filter);
        let mut slices = Vec::with_capacity(matched.len());
        let mut total_len = 0usize;
        for &arch_id in matched.iter() {
            let Some(arch) = store.archetypes.get(arch_id.0 as usize) else {
                continue;
            };
            let len = arch.len();
            if len == 0 || !self.filter.matches_archetype(arch) {
                continue;
            }
            if let Some(col) = arch.get_column::<T>() {
                let s = col.as_slice();
                if s.len() == len {
                    total_len += len;
                    slices.push(s);
                }
            }
        }
        (slices, total_len)
    }

    pub(crate) fn collect_mut_slices<'b>(&'b mut self) -> (Vec<&'b mut [T]>, usize) {
        let store: &'b mut Store = unsafe { self.store_ptr.as_mut() };
        let matched = SharedColCache::resolve(&self.shared_cache, store, &self.filter);
        let cid = ComponentTypeId::of::<T>();
        let mut slices = Vec::with_capacity(matched.len());
        let mut total_len = 0usize;
        let arch_ptr = store.archetypes.as_mut_ptr();
        let arch_len = store.archetypes.len();
        for &arch_id in matched.iter() {
            let idx = arch_id.0 as usize;
            if idx >= arch_len {
                continue;
            }
            let arch = unsafe { &mut *arch_ptr.add(idx) };
            let len = arch.len();
            if len == 0 || !self.filter.matches_archetype(arch) {
                continue;
            }
            arch.mark_column_all_chunks_epoch(cid, self.write_epoch);
            arch.bump_columns_version_in_place();
            if let Some(col) = arch.get_column_mut::<T>() {
                let s = col.as_mut_slice();
                if s.len() == len {
                    total_len += len;
                    slices.push(s);
                }
            }
        }
        (slices, total_len)
    }

    /// 顺序遍历列中的所有只读元素 `&T`
    #[inline]
    pub fn iter(&self) -> MultiSliceIter<'_, T> {
        let (slices, total_len) = self.collect_read_slices_from_mut();
        MultiSliceIter::new(slices, total_len)
    }

    /// 顺序遍历列中的所有可变元素 `&mut T`
    #[inline]
    pub fn iter_mut(&mut self) -> MultiSliceIterMut<'_, T> {
        let (slices, total_len) = self.collect_mut_slices();
        MultiSliceIterMut::new(slices, total_len)
    }
}

/// 跨多张 Archetype 连续只读切片的精确长度双端迭代器
pub struct MultiSliceIter<'b, T> {
    inner: std::iter::Flatten<std::vec::IntoIter<&'b [T]>>,
    remaining: usize,
}

impl<'b, T> MultiSliceIter<'b, T> {
    #[inline]
    pub(crate) fn new(slices: Vec<&'b [T]>, remaining: usize) -> Self {
        Self {
            inner: slices.into_iter().flatten(),
            remaining,
        }
    }
}

impl<'b, T> Iterator for MultiSliceIter<'b, T> {
    type Item = &'b T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        self.remaining -= 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'b, T> DoubleEndedIterator for MultiSliceIter<'b, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let item = self.inner.next_back()?;
        self.remaining -= 1;
        Some(item)
    }
}

impl<'b, T> ExactSizeIterator for MultiSliceIter<'b, T> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

/// 跨多张 Archetype 连续可变切片的精确长度双端迭代器
pub struct MultiSliceIterMut<'b, T> {
    inner: std::iter::Flatten<std::vec::IntoIter<&'b mut [T]>>,
    remaining: usize,
}

impl<'b, T> MultiSliceIterMut<'b, T> {
    #[inline]
    pub(crate) fn new(slices: Vec<&'b mut [T]>, remaining: usize) -> Self {
        Self {
            inner: slices.into_iter().flatten(),
            remaining,
        }
    }
}

impl<'b, T> Iterator for MultiSliceIterMut<'b, T> {
    type Item = &'b mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        self.remaining -= 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'b, T> DoubleEndedIterator for MultiSliceIterMut<'b, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let item = self.inner.next_back()?;
        self.remaining -= 1;
        Some(item)
    }
}

impl<'b, T> ExactSizeIterator for MultiSliceIterMut<'b, T> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'b, 'a, T, F> IntoIterator for &'b QueryCol<'a, &'a T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b T;
    type IntoIter = MultiSliceIter<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'b, 'a, T, F> IntoIterator for &'b QueryCol<'a, &'a mut T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b T;
    type IntoIter = MultiSliceIter<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'b, 'a, T, F> IntoIterator for &'b mut QueryCol<'a, &'a mut T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b mut T;
    type IntoIter = MultiSliceIterMut<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// 跨多张 Archetype 连续只读切片的 Rayon 并行迭代器（支持 `MultiZip` 多列联合并行）
pub struct MultiSliceParIter<'b, T: Sync + 'b> {
    slices: Vec<&'b [T]>,
    total_len: usize,
}

struct MultiSliceProducer<'b, T: Sync + 'b> {
    slices: Vec<&'b [T]>,
    total_len: usize,
}

impl<'b, T: Sync + 'b> rayon::iter::plumbing::Producer for MultiSliceProducer<'b, T> {
    type Item = &'b T;
    type IntoIter = MultiSliceIter<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        MultiSliceIter::new(self.slices, self.total_len)
    }

    #[inline]
    fn min_len(&self) -> usize {
        crate::storage::primitives::SubchunkPlanner::default()
            .subchunk_rows(std::mem::size_of::<T>())
            .min(1024)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let right_len = self.total_len.saturating_sub(index);
        let mut left_slices = Vec::new();
        let mut right_slices = Vec::new();
        let mut rem = index;
        let mut switched_to_right = false;

        for s in self.slices {
            if !switched_to_right {
                if rem >= s.len() {
                    rem -= s.len();
                    left_slices.push(s);
                    if rem == 0 {
                        switched_to_right = true;
                    }
                } else if rem > 0 {
                    let (l, r) = s.split_at(rem);
                    left_slices.push(l);
                    right_slices.push(r);
                    rem = 0;
                    switched_to_right = true;
                } else {
                    switched_to_right = true;
                    right_slices.push(s);
                }
            } else {
                right_slices.push(s);
            }
        }

        (
            Self {
                slices: left_slices,
                total_len: index.min(self.total_len),
            },
            Self {
                slices: right_slices,
                total_len: right_len,
            },
        )
    }
}

impl<'b, T: Sync + 'b> rayon::iter::ParallelIterator for MultiSliceParIter<'b, T> {
    type Item = &'b T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.total_len)
    }
}

impl<'b, T: Sync + 'b> rayon::iter::IndexedParallelIterator for MultiSliceParIter<'b, T> {
    #[inline]
    fn len(&self) -> usize {
        self.total_len
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(MultiSliceProducer {
            slices: self.slices,
            total_len: self.total_len,
        })
    }
}

/// 跨多张 Archetype 连续可变切片的 Rayon 并行迭代器（支持 `MultiZip` 多列联合并行）
pub struct MultiSliceParIterMut<'b, T: Send + 'b> {
    slices: Vec<&'b mut [T]>,
    total_len: usize,
}

struct MultiSliceMutProducer<'b, T: Send + 'b> {
    slices: Vec<&'b mut [T]>,
    total_len: usize,
}

impl<'b, T: Send + 'b> rayon::iter::plumbing::Producer for MultiSliceMutProducer<'b, T> {
    type Item = &'b mut T;
    type IntoIter = MultiSliceIterMut<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        MultiSliceIterMut::new(self.slices, self.total_len)
    }

    #[inline]
    fn min_len(&self) -> usize {
        crate::storage::primitives::SubchunkPlanner::default()
            .subchunk_rows(std::mem::size_of::<T>())
            .min(1024)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let right_len = self.total_len.saturating_sub(index);
        let mut left_slices = Vec::new();
        let mut right_slices = Vec::new();
        let mut rem = index;
        let mut switched_to_right = false;

        for s in self.slices {
            if !switched_to_right {
                if rem >= s.len() {
                    rem -= s.len();
                    left_slices.push(s);
                    if rem == 0 {
                        switched_to_right = true;
                    }
                } else if rem > 0 {
                    let (l, r) = s.split_at_mut(rem);
                    left_slices.push(l);
                    right_slices.push(r);
                    rem = 0;
                    switched_to_right = true;
                } else {
                    switched_to_right = true;
                    right_slices.push(s);
                }
            } else {
                right_slices.push(s);
            }
        }

        (
            Self {
                slices: left_slices,
                total_len: index.min(self.total_len),
            },
            Self {
                slices: right_slices,
                total_len: right_len,
            },
        )
    }
}

impl<'b, T: Send + 'b> rayon::iter::ParallelIterator for MultiSliceParIterMut<'b, T> {
    type Item = &'b mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.total_len)
    }
}

impl<'b, T: Send + 'b> rayon::iter::IndexedParallelIterator for MultiSliceParIterMut<'b, T> {
    #[inline]
    fn len(&self) -> usize {
        self.total_len
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(MultiSliceMutProducer {
            slices: self.slices,
            total_len: self.total_len,
        })
    }
}

impl<'b, 'a, T, F> rayon::iter::IntoParallelIterator for &'b QueryCol<'a, &'a T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned + Sync,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b T;
    type Iter = MultiSliceParIter<'b, T>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        let (slices, total_len) = self.collect_read_slices();
        MultiSliceParIter { slices, total_len }
    }
}

impl<'b, 'a, T, F> rayon::iter::IntoParallelIterator for &'b QueryCol<'a, &'a mut T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned + Sync,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b T;
    type Iter = MultiSliceParIter<'b, T>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        let (slices, total_len) = self.collect_read_slices_from_mut();
        MultiSliceParIter { slices, total_len }
    }
}

impl<'b, 'a, T, F> rayon::iter::IntoParallelIterator for &'b mut QueryCol<'a, &'a mut T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned + Send + Sync,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b mut T;
    type Iter = MultiSliceParIterMut<'b, T>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        let (slices, total_len) = self.collect_mut_slices();
        MultiSliceParIterMut { slices, total_len }
    }
}

impl<'b, 'a, T, F> rayon::iter::IntoParallelIterator
    for &'b crate::storage::query::SelectBuilder<'a, &'a T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned + Sync,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b T;
    type Iter = MultiSliceParIter<'b, T>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        let mut slices = Vec::new();
        let mut total_len = 0usize;
        for arch in self.archetypes.iter() {
            let len = arch.len();
            if len == 0 || !self.filter.matches_archetype(arch) {
                continue;
            }
            if let Some(col) = arch.get_column::<T>() {
                let s = col.as_slice();
                if s.len() == len {
                    total_len += len;
                    slices.push(s);
                }
            }
        }
        MultiSliceParIter { slices, total_len }
    }
}

impl<'b, 'a, T, F> rayon::iter::IntoParallelIterator
    for &'b mut crate::storage::query::QueryMutBuilder<'a, &'a mut T, F>
where
    T: ModelComponent + Serialize + DeserializeOwned + Send + Sync,
    F: crate::storage::query::QueryFilter,
{
    type Item = &'b mut T;
    type Iter = MultiSliceParIterMut<'b, T>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        let cid = ComponentTypeId::of::<T>();
        let mut slices = Vec::new();
        let mut total_len = 0usize;
        for arch in self.archetypes.iter_mut() {
            let len = arch.len();
            if len == 0 || !self.filter.matches_archetype(arch) {
                continue;
            }
            arch.mark_column_all_chunks_epoch(cid, self.write_epoch);
            arch.bump_columns_version_in_place();
            if let Some(col) = arch.get_column_mut::<T>() {
                let s = col.as_mut_slice();
                if s.len() == len {
                    total_len += len;
                    slices.push(s);
                }
            }
        }
        MultiSliceParIterMut { slices, total_len }
    }
}

/// 单列参数提取契约（用于 [`Store::queries_mut`] 与 [`Store::queries`]）
pub trait SingleColParam<'a>: Sized {
    const IS_MUT: bool;
    fn component_id() -> ComponentTypeId;
    fn type_name() -> &'static str;
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> SingleColParam<'a> for &'a mut T {
    const IS_MUT: bool = true;
    #[inline]
    fn component_id() -> ComponentTypeId {
        ComponentTypeId::of::<T>()
    }
    #[inline]
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> SingleColParam<'a> for &'a T {
    const IS_MUT: bool = false;
    #[inline]
    fn component_id() -> ComponentTypeId {
        ComponentTypeId::of::<T>()
    }
    #[inline]
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }
}

/// 可变多列元组解构契约（支持 `store.queries_mut::<(&mut A, &mut B, ...)>()` 直接返回列视图元组）
pub trait QueriesMutSplit<'a>: Sized {
    type Output;
    fn validate_no_alias();
    fn required_cids() -> SmallVec<[ComponentTypeId; 8]>;
    unsafe fn split_from_store(
        store_ptr: NonNull<Store>,
        write_epoch: u64,
        shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    ) -> Self::Output;
}

/// 只读多列元组解构契约（支持 `store.queries::<(&A, &B, ...)>()` 直接返回只读列视图元组）
pub trait QueriesReadSplit<'a>: Sized {
    type Output;
    fn required_cids() -> SmallVec<[ComponentTypeId; 8]>;
    unsafe fn split_from_store(
        store_ptr: NonNull<Store>,
        shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    ) -> Self::Output;
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueriesMutSplit<'a> for &'a mut T {
    type Output = QueryCol<'a, &'a mut T, crate::storage::query::All>;

    #[inline]
    fn validate_no_alias() {}

    #[inline]
    fn required_cids() -> SmallVec<[ComponentTypeId; 8]> {
        smallvec::smallvec![ComponentTypeId::of::<T>()]
    }

    #[inline]
    unsafe fn split_from_store(
        store_ptr: NonNull<Store>,
        write_epoch: u64,
        shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    ) -> Self::Output {
        QueryCol {
            store_ptr,
            write_epoch,
            filter: crate::storage::query::All,
            shared_cache,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueriesMutSplit<'a> for &'a T {
    type Output = QueryCol<'a, &'a T, crate::storage::query::All>;

    #[inline]
    fn validate_no_alias() {}

    #[inline]
    fn required_cids() -> SmallVec<[ComponentTypeId; 8]> {
        smallvec::smallvec![ComponentTypeId::of::<T>()]
    }

    #[inline]
    unsafe fn split_from_store(
        store_ptr: NonNull<Store>,
        write_epoch: u64,
        shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    ) -> Self::Output {
        QueryCol {
            store_ptr,
            write_epoch,
            filter: crate::storage::query::All,
            shared_cache,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueriesReadSplit<'a> for &'a T {
    type Output = QueryCol<'a, &'a T, crate::storage::query::All>;

    #[inline]
    fn required_cids() -> SmallVec<[ComponentTypeId; 8]> {
        smallvec::smallvec![ComponentTypeId::of::<T>()]
    }

    #[inline]
    unsafe fn split_from_store(
        store_ptr: NonNull<Store>,
        shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    ) -> Self::Output {
        QueryCol {
            store_ptr,
            write_epoch: 0,
            filter: crate::storage::query::All,
            shared_cache,
            _marker: PhantomData,
        }
    }
}

/// 面向 [`QueryCol`] 及 `1..=8` 列视图元组的统一链式过滤扩展（支持元组直接 `.with::<T>()` / `.without::<T>()` 解构）
pub trait QueryTupleFilterExt: Sized {
    /// 附加 `.with::<T>()` 后的类型
    type With<T: 'static>;
    /// 附加 `.without::<T>()` 后的类型
    type Without<T: 'static>;

    /// 对元组内所有列查询同步附加 `.with::<T>()` 过滤条件
    fn with<T: 'static>(self) -> Self::With<T>;
    /// 对元组内所有列查询同步附加 `.without::<T>()` 过滤条件
    fn without<T: 'static>(self) -> Self::Without<T>;
    /// 保持向后兼容的显式列返回入口（恒等变换）
    #[inline]
    fn columns(self) -> Self {
        self
    }
}

macro_rules! impl_queries_tuple {
    ($( $P:ident : $idx:tt ),+) => {
        impl<'a, $( $P: SingleColParam<'a> ),+> QueriesMutSplit<'a> for ( $( $P, )+ ) {
            type Output = ( $( QueryCol<'a, $P, crate::storage::query::All>, )+ );

            fn validate_no_alias() {
                let ids = [ $( $P::component_id() ),+ ];
                let muts = [ $( $P::IS_MUT ),+ ];
                let names = [ $( $P::type_name() ),+ ];
                for i in 0..ids.len() {
                    for j in (i + 1)..ids.len() {
                        if ids[i] == ids[j] && (muts[i] || muts[j]) {
                            panic!(
                                "Aliasing violation: component '{}' requested multiple times with mutable access in queries_mut",
                                names[i]
                            );
                        }
                    }
                }
            }

            #[inline]
            fn required_cids() -> SmallVec<[ComponentTypeId; 8]> {
                smallvec::smallvec![ $( $P::component_id() ),+ ]
            }

            #[inline]
            unsafe fn split_from_store(
                store_ptr: NonNull<Store>,
                write_epoch: u64,
                shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
            ) -> Self::Output {
                (
                    $(
                        QueryCol::<$P, crate::storage::query::All> {
                            store_ptr,
                            write_epoch,
                            filter: crate::storage::query::All,
                            shared_cache: Arc::clone(&shared_cache),
                            _marker: PhantomData,
                        },
                    )+
                )
            }
        }

        impl<'a, $( $P: ModelComponent + Serialize + DeserializeOwned ),+> QueriesReadSplit<'a>
            for ( $( &'a $P, )+ )
        {
            type Output = ( $( QueryCol<'a, &'a $P, crate::storage::query::All>, )+ );

            #[inline]
            fn required_cids() -> SmallVec<[ComponentTypeId; 8]> {
                smallvec::smallvec![ $( ComponentTypeId::of::<$P>() ),+ ]
            }

            #[inline]
            unsafe fn split_from_store(
                store_ptr: NonNull<Store>,
                shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
            ) -> Self::Output {
                (
                    $(
                        QueryCol::<&'a $P, crate::storage::query::All> {
                            store_ptr,
                            write_epoch: 0,
                            filter: crate::storage::query::All,
                            shared_cache: Arc::clone(&shared_cache),
                            _marker: PhantomData,
                        },
                    )+
                )
            }
        }

        impl<'a, F, $( $P ),+> QueryTupleFilterExt
            for ( $( QueryCol<'a, $P, F>, )+ )
        where
            F: crate::storage::query::QueryFilter,
        {
            type With<T: 'static> = ( $( QueryCol<'a, $P, crate::storage::query::With<T, F>>, )+ );
            type Without<T: 'static> = ( $( QueryCol<'a, $P, crate::storage::query::Without<T, F>>, )+ );

            #[inline]
            fn with<T: 'static>(self) -> Self::With<T> {
                ( $( self.$idx.with::<T>(), )+ )
            }

            #[inline]
            fn without<T: 'static>(self) -> Self::Without<T> {
                ( $( self.$idx.without::<T>(), )+ )
            }
        }
    };
}

impl_queries_tuple!(P0:0);
impl_queries_tuple!(P0:0, P1:1);
impl_queries_tuple!(P0:0, P1:1, P2:2);
impl_queries_tuple!(P0:0, P1:1, P2:2, P3:3);
impl_queries_tuple!(P0:0, P1:1, P2:2, P3:3, P4:4);
impl_queries_tuple!(P0:0, P1:1, P2:2, P3:3, P4:4, P5:5);
impl_queries_tuple!(P0:0, P1:1, P2:2, P3:3, P4:4, P5:5, P6:6);
impl_queries_tuple!(P0:0, P1:1, P2:2, P3:3, P4:4, P5:5, P6:6, P7:7);

impl Store {
    /// 开启多列排他/混合查询并直接返回可解构的强类型列视图元组 `(QueryCol, ...)`
    ///
    /// - 支持链式 `.with::<T>()` 与 `.without::<T>()` 过滤并直接解构：`let (mut pos, mut vel) = store.queries_mut::<(&mut Pos, &mut Vel)>()?.with::<Active>();`
    /// - 内部共享 [`SharedColCache`]，在无 CR 结构修改时实现 100% 缓存命中与零原型重扫。
    #[inline]
    pub fn queries_mut<'a, Qs: QueriesMutSplit<'a>>(
        &'a mut self,
    ) -> Result<Qs::Output, StateError> {
        self.check_poisoned()?;
        Qs::validate_no_alias();
        let write_epoch = self.next_epoch()?;
        self.bump_epoch();
        let shared_cache = Arc::new(std::sync::RwLock::new(SharedColCache::new(
            Qs::required_cids(),
        )));
        let store_ptr = NonNull::from(self);
        Ok(unsafe { Qs::split_from_store(store_ptr, write_epoch, shared_cache) })
    }

    /// 开启多列只读查询并直接返回可解构的只读列视图元组 `(QueryCol, ...)`
    #[inline]
    pub fn queries<'a, Qs: QueriesReadSplit<'a>>(&'a self) -> Qs::Output {
        let shared_cache = Arc::new(std::sync::RwLock::new(SharedColCache::new(
            Qs::required_cids(),
        )));
        let store_ptr = NonNull::from(self);
        unsafe { Qs::split_from_store(store_ptr, shared_cache) }
    }
}

/// 绑定至 [`KernelLaunch`] 的单列运行时元数据
#[derive(Clone)]
pub struct KernelColBinding {
    pub(crate) store_ptr: NonNull<Store>,
    pub(crate) write_epoch: u64,
    pub(crate) shared_cache: Arc<std::sync::RwLock<SharedColCache>>,
    pub(crate) filter: Arc<dyn ErasedArchetypeFilter>,
    pub(crate) component_id: ComponentTypeId,
    pub(crate) mutable: bool,
}

// SAFETY: `KernelColBinding` 仅在受 `'store` 生命周期约束的 `KernelLaunch<'store>` 内部按顺序执行或编码。
unsafe impl Send for KernelColBinding {}
unsafe impl Sync for KernelColBinding {}

/// 算子整列参数绑定契约（支持将 `&mut QueryCol`、`&QueryCol` 或 `QueryCol` 传给算子的 `.run`/`.par`/`.gpu`/`.bind`）
///
/// 注意：借用局部变量的临时生命周期不会逃逸至返回的 [`KernelLaunch<'store>`]，
/// 仅保留底层 `'store` 存储借用生命周期，从而允许同一元组管线内多阶段重复借用同一 `QueryCol` 局部变量。
pub trait KernelColumnArg<'store, ExpectedParam> {
    /// 提取单列绑定描述符
    fn into_kernel_col_binding(self) -> KernelColBinding;
}

impl<'borrow, 'store, T, F> KernelColumnArg<'store, &'static mut T>
    for &'borrow mut QueryCol<'store, &'store mut T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: Arc::clone(&self.shared_cache),
            filter: Arc::new(self.filter.clone()),
            component_id: ComponentTypeId::of::<T>(),
            mutable: true,
        }
    }
}

impl<'store, T, F> KernelColumnArg<'store, &'static mut T>
    for QueryCol<'store, &'store mut T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: self.shared_cache,
            filter: Arc::new(self.filter),
            component_id: ComponentTypeId::of::<T>(),
            mutable: true,
        }
    }
}

impl<'borrow, 'store, T, F> KernelColumnArg<'store, &'static T>
    for &'borrow QueryCol<'store, &'store T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: Arc::clone(&self.shared_cache),
            filter: Arc::new(self.filter.clone()),
            component_id: ComponentTypeId::of::<T>(),
            mutable: false,
        }
    }
}

impl<'borrow, 'store, T, F> KernelColumnArg<'store, &'static T>
    for &'borrow QueryCol<'store, &'store mut T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: Arc::clone(&self.shared_cache),
            filter: Arc::new(self.filter.clone()),
            component_id: ComponentTypeId::of::<T>(),
            mutable: false,
        }
    }
}

impl<'borrow, 'store, T, F> KernelColumnArg<'store, &'static T>
    for &'borrow mut QueryCol<'store, &'store mut T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: Arc::clone(&self.shared_cache),
            filter: Arc::new(self.filter.clone()),
            component_id: ComponentTypeId::of::<T>(),
            mutable: false,
        }
    }
}

impl<'store, T, F> KernelColumnArg<'store, &'static T> for QueryCol<'store, &'store T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: self.shared_cache,
            filter: Arc::new(self.filter),
            component_id: ComponentTypeId::of::<T>(),
            mutable: false,
        }
    }
}

impl<'store, T, F> KernelColumnArg<'store, &'static T>
    for QueryCol<'store, &'store mut T, F>
where
    T: GpuComponent + GpuShaderStruct,
    F: crate::storage::query::QueryFilter + Clone + 'static,
{
    #[inline]
    fn into_kernel_col_binding(self) -> KernelColBinding {
        KernelColBinding {
            store_ptr: self.store_ptr,
            write_epoch: self.write_epoch,
            shared_cache: self.shared_cache,
            filter: Arc::new(self.filter),
            component_id: ComponentTypeId::of::<T>(),
            mutable: false,
        }
    }
}

/// 延迟绑定的算子整列派发描述符（由 `kernel.bind(...)` 创建，可单独执行或组成元组管线 `(k1.bind(...), k2.bind(...)).gpu()?`）
#[must_use = "a KernelLaunch does no work until .run(), .par(), or .gpu() is called"]
pub struct KernelLaunch<'store> {
    pub(crate) descriptor: &'static GpuKernelDescriptor,
    pub(crate) scalar_words: [u32; 4],
    pub(crate) col_bindings: SmallVec<[KernelColBinding; 8]>,
    pub(crate) _marker: PhantomData<&'store mut Store>,
}

impl<'store> std::fmt::Debug for KernelLaunch<'store> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelLaunch")
            .field("kernel", &self.descriptor.name())
            .field("num_bindings", &self.col_bindings.len())
            .finish()
    }
}

impl<'store> KernelLaunch<'store> {
    /// 从内核静态描述符、标量 Uniform 字与列绑定切片构造 [`KernelLaunch`]
    #[inline]
    pub fn from_bindings(
        descriptor: &'static GpuKernelDescriptor,
        scalar_words: [u32; 4],
        bindings: &[KernelColBinding],
    ) -> Self {
        Self {
            descriptor,
            scalar_words,
            col_bindings: bindings.iter().cloned().collect(),
            _marker: PhantomData,
        }
    }

    /// 使用线程本地默认 [`MetalComputeEngine`] 在 GPU 硬件计算通道上执行
    #[inline]
    pub fn gpu(self) -> Result<GpuDispatchStats, GpuExecutionError> {
        <Self as ComputeTask<'store>>::gpu(self)
    }

    /// 在指定的 [`MetalComputeEngine`] 上执行
    #[inline]
    pub fn gpu_on(
        self,
        engine: &mut MetalComputeEngine,
    ) -> Result<GpuDispatchStats, GpuExecutionError> {
        <Self as ComputeTask<'store>>::gpu_on(self, engine)
    }

    /// 在主机 CPU 上按 1024 行 Morsel 阈值自适应单核/多核并行执行
    #[inline]
    pub fn par(self) -> Result<GpuDispatchStats, GpuExecutionError> {
        self.execute_par_in_place()
    }

    /// 自动选择最优硬件后端执行（存在硬件 GPU 时走 `.gpu()`，否则走自适应多核 `.par()`）
    #[inline]
    pub fn run(self) -> Result<GpuDispatchStats, GpuExecutionError> {
        <Self as ComputeTask<'store>>::run(self)
    }

    pub(crate) fn validate_bindings(&self) -> Result<NonNull<Store>, GpuExecutionError> {
        let Some(first) = self.col_bindings.first() else {
            return Err(GpuExecutionError::State(StateError::AccessDenied(
                "KernelLaunch requires at least one column binding".into(),
            )));
        };
        for i in 0..self.col_bindings.len() {
            let bi = &self.col_bindings[i];
            if bi.store_ptr != first.store_ptr {
                return Err(GpuExecutionError::State(StateError::AccessDenied(
                    "All columns bound to a KernelLaunch must belong to the same Store".into(),
                )));
            }
            if bi.filter.is_row_filtered() {
                return Err(GpuExecutionError::State(StateError::AccessDenied(
                    "Whole-column kernel dispatch does not support row-level CPU predicate filters; use archetype filters (.with / .without)".into(),
                )));
            }
            for j in (i + 1)..self.col_bindings.len() {
                let bj = &self.col_bindings[j];
                if bi.component_id == bj.component_id && (bi.mutable || bj.mutable) {
                    return Err(GpuExecutionError::State(StateError::AccessDenied(format!(
                        "Aliasing violation in kernel `{}`: component {:?} bound multiple times with mutable access",
                        self.descriptor.name(),
                        bi.component_id
                    ))));
                }
            }
        }
        Ok(first.store_ptr)
    }

    pub(crate) fn resolve_matched_archetype_indices(
        &self,
        store: &Store,
    ) -> Result<Vec<usize>, GpuExecutionError> {
        self.validate_bindings()?;
        let first = &self.col_bindings[0];
        let matched = SharedColCache::resolve(&first.shared_cache, store, first.filter.as_ref());
        let desc = self.descriptor;
        let mut valid_indices = Vec::with_capacity(matched.len());

        for &arch_id in matched.iter() {
            let arch_idx = arch_id.0 as usize;
            let Some(archetype) = store.archetypes.get(arch_idx) else {
                continue;
            };
            let active_rows = archetype.len();
            if active_rows == 0 || !first.filter.matches_archetype(archetype) {
                continue;
            }
            if !self
                .col_bindings
                .iter()
                .skip(1)
                .all(|b| Arc::ptr_eq(&b.shared_cache, &first.shared_cache) || b.filter.matches_archetype(archetype))
            {
                continue;
            }
            let comp_slice = archetype.component_types_slice();
            if !desc
                .bindings()
                .iter()
                .all(|b| comp_slice.binary_search(&b.type_id).is_ok())
            {
                continue;
            }
            let cols = archetype.columns();
            for b in desc.bindings() {
                let col = cols.get(&b.type_id).ok_or_else(|| {
                    StateError::ComponentNotFound(format!("{}", b.type_id))
                })?;
                if col.len() != active_rows {
                    return Err(GpuExecutionError::State(StateError::AccessDenied(format!(
                        "Column {} length {} does not match archetype {} active_rows {}",
                        b.struct_name,
                        col.len(),
                        arch_id.0,
                        active_rows
                    ))));
                }
                let _active_bytes = active_rows.checked_mul(b.elem_size).ok_or_else(|| {
                    GpuExecutionError::State(StateError::CapacityExhausted(
                        "GPU column active byte size overflow".into(),
                    ))
                })?;
            }
            valid_indices.push(arch_idx);
        }
        Ok(valid_indices)
    }

    pub(crate) fn execute_par_in_place(&self) -> Result<GpuDispatchStats, GpuExecutionError> {
        let mut store_ptr = self.validate_bindings()?;
        let store = unsafe { store_ptr.as_mut() };
        store.check_poisoned()?;
        let valid_indices = self.resolve_matched_archetype_indices(store)?;
        let desc = self.descriptor;
        let write_epoch = self
            .col_bindings
            .iter()
            .map(|b| b.write_epoch)
            .max()
            .unwrap_or_else(|| store.epoch());

        struct ParArchSlice {
            arch_id: ArchetypeId,
            active_rows: usize,
            base_ptrs: SmallVec<[usize; 8]>,
            elem_sizes: SmallVec<[usize; 8]>,
        }

        let mut prepared = Vec::with_capacity(valid_indices.len());
        let mut total_rows = 0usize;
        let mut total_pages = 0usize;

        for arch_idx in valid_indices {
            let archetype = &mut store.archetypes[arch_idx];
            let active_rows = archetype.len();
            let arch_id = archetype.id();
            let mut changed_any = false;

            for b in desc.bindings() {
                if b.mutable {
                    let col = archetype
                        .get_erased_column_mut(b.type_id)
                        .expect("verified in resolve_matched_archetype_indices");
                    col.ensure_contiguous()?;
                    changed_any = true;
                }
            }
            if changed_any {
                archetype.rebuild_flat_columns();
            }
            for b in desc.bindings() {
                if b.mutable {
                    archetype.mark_column_all_chunks_epoch(b.type_id, write_epoch);
                }
            }
            if desc.bindings().iter().any(|b| b.mutable) {
                archetype.bump_columns_version_in_place();
            }

            let mut base_ptrs = SmallVec::<[usize; 8]>::with_capacity(desc.bindings().len());
            let mut elem_sizes = SmallVec::<[usize; 8]>::with_capacity(desc.bindings().len());
            let mut arch_pages = 0usize;
            {
                let cols = archetype.columns();
                for b in desc.bindings() {
                    let col = cols
                        .get(&b.type_id)
                        .expect("verified in resolve_matched_archetype_indices");
                    let col_desc = crate::storage::column::ColumnDescriptor::from_column_ref(&**col);
                    let (base, offset, end) = col_desc.locate(0);
                    if end < active_rows {
                        return Err(GpuExecutionError::State(StateError::AccessDenied(format!(
                            "Column {} in archetype {} is segmented across multiple blocks",
                            b.struct_name, arch_id.0
                        ))));
                    }
                    let ptr = unsafe { (base as *mut u8).add(offset * b.elem_size) };
                    let allocated_bytes = col
                        .metal_raw_parts()
                        .map(|(_, bytes)| bytes)
                        .unwrap_or(active_rows * b.elem_size);
                    let geom = PageGeometry::compute(b.elem_size, active_rows, allocated_bytes);
                    arch_pages += geom.page_count;
                    base_ptrs.push(ptr as usize);
                    elem_sizes.push(b.elem_size);
                }
            }

            total_rows += active_rows;
            total_pages += arch_pages;
            prepared.push(ParArchSlice {
                arch_id,
                active_rows,
                base_ptrs,
                elem_sizes,
            });
        }

        let runner = desc.cpu_runner();
        let scalar_args = self.scalar_words;
        let row_width: usize = desc.bindings().iter().map(|b| b.elem_size).sum();
        let subchunk_rows = crate::storage::primitives::SubchunkPlanner::default()
            .subchunk_rows(row_width);
        let par_threshold = subchunk_rows.min(1024);

        if total_rows < par_threshold {
            // 小于缓存子块并行阈值：直接在当前线程顺序执行，避免线程池唤醒开销
            for item in &prepared {
                let raw_ptrs: SmallVec<[*mut u8; 8]> =
                    item.base_ptrs.iter().map(|&p| p as *mut u8).collect();
                let uniforms = GpuDispatchUniforms {
                    active_rows: item.active_rows as u32,
                    threads_per_group: 256,
                    archetype_id: item.arch_id.0,
                    _pad: 0,
                    scalar_args,
                };
                unsafe {
                    runner(&raw_ptrs, item.active_rows, &uniforms);
                }
            }
        } else {
            // 达到并行阈值：按 SubchunkPlanner 规划的 L1/L2 缓存驻留子块在 Rayon 线程池上并行执行
            use rayon::prelude::*;
            let morsel_rows = subchunk_rows.min(1024);
            prepared.par_iter().for_each(|item| {
                let num_morsels = item.active_rows.div_ceil(morsel_rows);
                (0..num_morsels).into_par_iter().for_each(|morsel_idx| {
                    let start = morsel_idx * morsel_rows;
                    let end = (start + morsel_rows).min(item.active_rows);
                    let len = end - start;
                    if len == 0 {
                        return;
                    }
                    let raw_ptrs: SmallVec<[*mut u8; 8]> = item
                        .base_ptrs
                        .iter()
                        .zip(item.elem_sizes.iter())
                        .map(|(&base, &elem_sz)| (base + start * elem_sz) as *mut u8)
                        .collect();
                    let uniforms = GpuDispatchUniforms {
                        active_rows: len as u32,
                        threads_per_group: 256,
                        archetype_id: item.arch_id.0,
                        _pad: 0,
                        scalar_args,
                    };
                    unsafe {
                        runner(&raw_ptrs, len, &uniforms);
                    }
                });
            });
        }

        Ok(GpuDispatchStats {
            dispatched_archetypes: prepared.len(),
            total_rows,
            total_pages_16k: total_pages,
            buffer_cache_hits: 0,
            buffer_cache_misses: 0,
            execution_mode: GpuExecutionMode::CpuFallback,
            command_buffer_submissions: 0,
            barriers_inserted: 0,
        })
    }
}

/// 统一计算任务执行契约（同时支持单个 [`KernelLaunch`] 与 2..=8 个 [`KernelLaunch`] 组成的元组管线）
///
/// - `.gpu()` / `.gpu_on(&mut engine)`：在单个硬件 CommandBuffer 与 `GpuComputePass` 中编码全部阶段，
///   自动追踪列级 RAW/WAW/WAR 依赖并按需插入 `MTLBarrierScopeBuffers` 显存屏障，最后单次提交等待；
/// - `.par()`：按 1024 行 Morsel 阈值自适应在单核或 Rayon 多核线程池上执行；
/// - `.run()`：存在硬件 GPU 时自动使用 `.gpu()`，否则使用自适应 `.par()`。
pub trait ComputeTask<'store>: Sized {
    /// 使用线程本地默认 [`MetalComputeEngine`] 在 GPU 硬件计算通道上执行
    fn gpu(self) -> Result<GpuDispatchStats, GpuExecutionError> {
        thread_local! {
            static DEFAULT_TASK_ENGINE: std::cell::RefCell<MetalComputeEngine> =
                std::cell::RefCell::new(MetalComputeEngine::new());
        }
        DEFAULT_TASK_ENGINE.with(|cell| {
            let mut engine = cell.borrow_mut();
            self.gpu_on(&mut engine)
        })
    }

    /// 在指定的 [`MetalComputeEngine`] 上执行（多算子元组共享单次 CommandBuffer 提交与自动屏障管理）
    fn gpu_on(
        self,
        engine: &mut MetalComputeEngine,
    ) -> Result<GpuDispatchStats, GpuExecutionError>;

    /// 在主机 CPU 上按 1024 行 Morsel 阈值自适应单核/多核并行执行
    fn par(self) -> Result<GpuDispatchStats, GpuExecutionError>;

    /// 自动选择最优硬件后端执行（存在硬件 GPU 时走 `.gpu()`，否则走自适应多核 `.par()`）
    fn run(self) -> Result<GpuDispatchStats, GpuExecutionError> {
        thread_local! {
            static HAS_HW_GPU: bool = probe_system_gpu_driver().is_some();
        }
        let has_hw = HAS_HW_GPU.with(|&v| v);
        if has_hw {
            self.gpu()
        } else {
            self.par()
        }
    }
}

impl<'store> ComputeTask<'store> for KernelLaunch<'store> {
    fn gpu_on(
        self,
        engine: &mut MetalComputeEngine,
    ) -> Result<GpuDispatchStats, GpuExecutionError> {
        let mut store_ptr = self.validate_bindings()?;
        let store = unsafe { store_ptr.as_mut() };
        let mut pass = store.gpu_compute_pass(engine)?;
        pass.encode_kernel_launch(&self)?;
        pass.finish()
    }

    #[inline]
    fn par(self) -> Result<GpuDispatchStats, GpuExecutionError> {
        self.execute_par_in_place()
    }
}

macro_rules! impl_compute_task_tuple {
    ($( $K:ident : $idx:tt ),+) => {
        impl<'store, $( $K ),+> ComputeTask<'store> for ( $( $K, )+ )
        where
            $( $K: Into<KernelLaunch<'store>>, )+
        {
            fn gpu_on(
                self,
                engine: &mut MetalComputeEngine,
            ) -> Result<GpuDispatchStats, GpuExecutionError> {
                let launches: [KernelLaunch<'store>; [ $( $idx ),+ ].len()] = [
                    $( self.$idx.into(), )+
                ];
                let mut store_ptr = launches[0].validate_bindings()?;
                for launch in &launches[1..] {
                    let ptr = launch.validate_bindings()?;
                    if ptr != store_ptr {
                        return Err(GpuExecutionError::State(StateError::AccessDenied(
                            "All KernelLaunch stages in a ComputeTask tuple must target the same Store".into(),
                        )));
                    }
                }
                let store = unsafe { store_ptr.as_mut() };
                let mut pass = store.gpu_compute_pass(engine)?;
                for launch in &launches {
                    pass.encode_kernel_launch(launch)?;
                }
                pass.finish()
            }

            fn par(self) -> Result<GpuDispatchStats, GpuExecutionError> {
                let launches: [KernelLaunch<'store>; [ $( $idx ),+ ].len()] = [
                    $( self.$idx.into(), )+
                ];
                let mut agg = GpuDispatchStats {
                    execution_mode: GpuExecutionMode::CpuFallback,
                    ..Default::default()
                };
                for launch in &launches {
                    let s = launch.execute_par_in_place()?;
                    agg.dispatched_archetypes += s.dispatched_archetypes;
                    agg.total_rows += s.total_rows;
                    agg.total_pages_16k += s.total_pages_16k;
                }
                Ok(agg)
            }
        }
    };
}

impl_compute_task_tuple!(K0:0, K1:1);
impl_compute_task_tuple!(K0:0, K1:1, K2:2);
impl_compute_task_tuple!(K0:0, K1:1, K2:2, K3:3);
impl_compute_task_tuple!(K0:0, K1:1, K2:2, K3:3, K4:4);
impl_compute_task_tuple!(K0:0, K1:1, K2:2, K3:3, K4:4, K5:5);
impl_compute_task_tuple!(K0:0, K1:1, K2:2, K3:3, K4:4, K5:5, K6:6);
impl_compute_task_tuple!(K0:0, K1:1, K2:2, K3:3, K4:4, K5:5, K6:6, K7:7);

