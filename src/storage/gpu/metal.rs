//! Apple Metal 统一内存 (UMA) 零拷贝后端、Naga 着色器编译器校验与跨平台管线 (`gpu` / `gpu-metal`)
//!
//! 针对 Apple Silicon (M1 / M2 / M3 / M4) 统一内存架构 (Unified Memory Architecture) 与跨平台 WebGPU：
//! 1. **16KB 页对齐零拷贝 (`MTLStorageModeShared` + `newBufferWithBytesNoCopy`)**：
//!    当列存采用 [`PageAlignedBuffer`](crate::storage::column::PageAlignedBuffer) 时，首地址与容量均严格对齐至 `16384` 字节。
//!    Metal 驱动可直接将 CPU 物理页映射进 GPU 页表，实现 **0 字节拷贝、0 额外显存占用**。
//! 2. **真实的着色器编译与类型校验 (`naga`)**：
//!    每个 `#[gpu]` 内核在管线构建阶段均通过 `naga` 的 WGSL 前端解析器 (`naga::front::wgsl`)、
//!    IR 类型与边界验证器 (`naga::valid::Validator`) 以及 Apple MSL 后端代码生成器 (`naga::back::msl`) 的完整编译验证。
//! 3. **代际感知的 `MTLBuffer` 句柄缓存 (`MetalBufferCache`)**：
//!    以 `(ArchetypeId, ComponentTypeId)` 为键缓存零拷贝 Buffer 句柄，仅当 `(base_ptr, allocated_bytes, generation)` 发生改变时原地替换重建。
//! 4. **32 线程 SIMD-group 网格调度 (`MetalThreadgroupDispatch`)**：
//!    按 Apple GPU 的 32-wide threadExecutionWidth 自动计算最优 Threadgroup 大小与派发网格（带饱和防溢出保护）。
//! 5. **COW 安全生命周期票据 (`MetalAsyncCommandTicket`)**：
//!    通过持有底层列块的 `Arc` 引用，确保在外部引用存活期间 CPU 写操作自动触发写时复制 (COW) 隔离。

use super::{
    ColumnGpuMutView, ColumnGpuView, GpuColumnPin, GpuComponent, GpuCpuChunkRunner,
    GpuDispatchUniforms, GpuKernelDescriptor,
};
use crate::error::StateError;
use crate::id::{ArchetypeId, ComponentTypeId};
use crate::storage::column::page_aligned::HARDWARE_PAGE_SIZE;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Apple GPU SIMD-group (Wavefront) 硬件宽度（Apple Silicon 全系固定为 32 线程）。
pub const APPLE_SIMD_GROUP_WIDTH: u32 = 32;

/// Apple GPU 默认单 Threadgroup 最大线程数。
pub const DEFAULT_MAX_TOTAL_THREADS_PER_THREADGROUP: u32 = 256;

/// 默认 `MetalBufferCache` 最大缓存槽位数（超出时淘汰最旧条目，防止无限增长）。
pub const DEFAULT_MAX_BUFFER_CACHE_ENTRIES: usize = 4096;

/// Metal `MTLResourceOptions` 位标志定义（与 Apple `<Metal/MTLResource.h>` 二进制兼容）。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetalResourceOptions(pub u64);

impl MetalResourceOptions {
    /// `MTLResourceCPUCacheModeDefaultCache = 0`
    pub const CPU_CACHE_MODE_DEFAULT: Self = Self(0);
    /// `MTLResourceStorageModeShared = 0 << 4`（CPU 与 GPU 共享同一块物理内存页）
    pub const STORAGE_MODE_SHARED: Self = Self(0 << 4);
    /// `MTLResourceHazardTrackingModeDefault = 0 << 8`
    pub const HAZARD_TRACKING_MODE_DEFAULT: Self = Self(0 << 8);
    /// `MTLResourceHazardTrackingModeUntracked = 1 << 8`（由 Sumeru ECS 借用与 Epoch 显式同步）
    pub const HAZARD_TRACKING_MODE_UNTRACKED: Self = Self(1 << 8);

    /// Apple Silicon UMA 零拷贝标准配置：`StorageModeShared | CPUCacheModeDefault`。
    pub const UMA_SHARED_DEFAULT: Self = Self(0);
}

/// Metal `newBufferWithBytesNoCopy:length:options:deallocator:` 零拷贝内存描述符。
///
/// Apple Metal 硬件契约要求：
/// - `base_ptr` 必须按系统 VM 页大小（Apple Silicon 为 `16384` 字节）对齐；
/// - `allocated_bytes` 必须为 `16384` 字节的整数倍且大于 0；
/// - `active_bytes` 不得超过 `allocated_bytes`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalNoCopyBufferDescriptor {
    /// 16KB 页对齐的物理列首地址
    pub base_ptr: *mut u8,
    /// 16KB 整数倍的分配总字节数
    pub allocated_bytes: usize,
    /// 当前活跃元素的有效字节数 (`len * size_of::<T>()`)
    pub active_bytes: usize,
    /// 物理缓冲区重分配代际号
    pub generation: u64,
    /// Metal 资源选项 (`MTLStorageModeShared`)
    pub options: MetalResourceOptions,
}

// SAFETY: 描述符仅携带页对齐元信息，读写并发由上层 Store 借用与 GpuColumnPin 保证。
unsafe impl Send for MetalNoCopyBufferDescriptor {}
unsafe impl Sync for MetalNoCopyBufferDescriptor {}

impl MetalNoCopyBufferDescriptor {
    /// 从只读 GPU 列视图构造并校验 Metal `NoCopy` 描述符。
    pub fn from_view<T: GpuComponent>(view: &ColumnGpuView<'_, T>) -> Result<Self, StateError> {
        Self::validate_raw(
            view.base_ptr() as *mut u8,
            view.allocated_bytes(),
            view.geometry().active_bytes,
            view.generation(),
        )
    }

    /// 从可变 GPU 列视图构造并校验 Metal `NoCopy` 描述符。
    pub fn from_mut_view<T: GpuComponent>(
        view: &mut ColumnGpuMutView<'_, T>,
    ) -> Result<Self, StateError> {
        let active_bytes = view.geometry().active_bytes;
        let allocated_bytes = view.allocated_bytes();
        let generation = view.generation();
        let ptr = view.base_mut_ptr();
        Self::validate_raw(ptr, allocated_bytes, active_bytes, generation)
    }

    /// 校验原始指针与字节长度是否满足 Apple Metal `newBufferWithBytesNoCopy` 的 16KB 硬件页对齐与边界要求。
    pub fn validate_raw(
        base_ptr: *mut u8,
        allocated_bytes: usize,
        active_bytes: usize,
        generation: u64,
    ) -> Result<Self, StateError> {
        if base_ptr.is_null() {
            return Err(StateError::AccessDenied(
                "Metal NoCopy buffer pointer cannot be null".into(),
            ));
        }
        if !(base_ptr as usize).is_multiple_of(HARDWARE_PAGE_SIZE) {
            return Err(StateError::AccessDenied(format!(
                "Metal NoCopy buffer pointer {:p} is not 16KB ({HARDWARE_PAGE_SIZE}B) page-aligned",
                base_ptr
            )));
        }
        if allocated_bytes == 0 || !allocated_bytes.is_multiple_of(HARDWARE_PAGE_SIZE) {
            return Err(StateError::AccessDenied(format!(
                "Metal NoCopy buffer length {allocated_bytes} is not a positive multiple of 16KB ({HARDWARE_PAGE_SIZE}B)"
            )));
        }
        if active_bytes > allocated_bytes {
            return Err(StateError::AccessDenied(format!(
                "Metal NoCopy active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})"
            )));
        }
        Ok(Self {
            base_ptr,
            allocated_bytes,
            active_bytes,
            generation,
            options: MetalResourceOptions::UMA_SHARED_DEFAULT,
        })
    }
}

/// 缓存的 Metal Buffer 句柄条目。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedMetalBuffer {
    /// 后端分配的 `MTLBuffer` 句柄 ID
    pub handle_id: u64,
    /// 绑定时的物理列首地址
    pub base_ptr: usize,
    /// 绑定时的 16KB 页对齐分配字节数
    pub allocated_bytes: usize,
    /// 绑定时的 `PageAlignedBuffer` 代际号
    pub generation: u64,
}

/// 按 `(ArchetypeId, ComponentTypeId)` 组织的零拷贝 `MTLBuffer` 缓存表。
///
/// 避免每帧重复调用 `newBufferWithBytesNoCopy` 构造 Objective-C `MTLBuffer` 对象；
/// 当列发生扩容重分配或 COW 分离（`generation` 递增）时自动原地替换失效条目，并在超出容量上限时执行淘汰。
#[derive(Debug, Clone)]
pub struct MetalBufferCache {
    entries: HashMap<(ArchetypeId, ComponentTypeId), CachedMetalBuffer>,
    max_entries: usize,
    next_handle_id: u64,
    hits: u64,
    misses: u64,
}

impl Default for MetalBufferCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalBufferCache {
    /// 创建默认容量上限 (`4096`) 的空缓存。
    #[inline]
    pub fn new() -> Self {
        Self::with_max_entries(DEFAULT_MAX_BUFFER_CACHE_ENTRIES)
    }

    /// 创建指定容量上限的缓存。
    #[inline]
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: max_entries.max(1),
            next_handle_id: 1,
            hits: 0,
            misses: 0,
        }
    }

    /// 获取或注册零拷贝 `MTLBuffer` 句柄，返回 `(CachedMetalBuffer, is_cache_hit)`。
    pub fn get_or_insert(
        &mut self,
        archetype_id: ArchetypeId,
        component_id: ComponentTypeId,
        desc: &MetalNoCopyBufferDescriptor,
    ) -> (CachedMetalBuffer, bool) {
        let (entry, hit, _evicted) =
            self.get_or_insert_with_evicted(archetype_id, component_id, desc);
        (entry, hit)
    }

    /// 获取或注册零拷贝 `MTLBuffer` 句柄，同时返回被替换或淘汰的旧 `handle_id`（供底层驱动释放硬件句柄）。
    pub fn get_or_insert_with_evicted(
        &mut self,
        archetype_id: ArchetypeId,
        component_id: ComponentTypeId,
        desc: &MetalNoCopyBufferDescriptor,
    ) -> (CachedMetalBuffer, bool, Option<u64>) {
        let key = (archetype_id, component_id);
        let ptr_addr = desc.base_ptr as usize;
        if let Some(existing) = self.entries.get(&key)
            && existing.base_ptr == ptr_addr
            && existing.allocated_bytes == desc.allocated_bytes
            && existing.generation == desc.generation
        {
            self.hits += 1;
            return (existing.clone(), true, None);
        }
        let mut evicted_handle = None;
        if let Some(replaced) = self.entries.remove(&key) {
            evicted_handle = Some(replaced.handle_id);
        } else if self.entries.len() >= self.max_entries
            && let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, v)| v.handle_id)
                .map(|(k, _)| *k)
            && let Some(removed) = self.entries.remove(&oldest_key)
        {
            evicted_handle = Some(removed.handle_id);
        }
        self.misses += 1;
        let handle_id = self.next_handle_id;
        self.next_handle_id += 1;
        let entry = CachedMetalBuffer {
            handle_id,
            base_ptr: ptr_addr,
            allocated_bytes: desc.allocated_bytes,
            generation: desc.generation,
        };
        self.entries.insert(key, entry.clone());
        (entry, false, evicted_handle)
    }

    /// 移除指定 `(ArchetypeId, ComponentTypeId)` 的缓存条目（用于硬件缓冲区注册失败时的原子回滚）。
    #[inline]
    pub fn remove_entry(
        &mut self,
        archetype_id: ArchetypeId,
        component_id: ComponentTypeId,
    ) -> Option<CachedMetalBuffer> {
        self.entries.remove(&(archetype_id, component_id))
    }

    /// 淘汰指定 Archetype 的所有缓存句柄，并返回被移除的 `handle_id` 列表。
    pub fn evict_archetype(&mut self, archetype_id: ArchetypeId) -> Vec<u64> {
        let mut removed = Vec::new();
        self.entries.retain(|(aid, _), entry| {
            if *aid == archetype_id {
                removed.push(entry.handle_id);
                false
            } else {
                true
            }
        });
        removed
    }

    /// 缓存命中次数。
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// 缓存未命中（新建/扩容重建）次数。
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// 当前缓存条目数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 缓存是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 清空全部缓存句柄，并返回被移除的 `handle_id` 列表。
    #[inline]
    pub fn clear(&mut self) -> Vec<u64> {
        self.entries.drain().map(|(_, v)| v.handle_id).collect()
    }
}

/// Metal 1D Compute Shader 派发网格参数 (`dispatchThreadgroups:threadsPerThreadgroup:`)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalThreadgroupDispatch {
    /// 活跃实体总行数
    pub active_rows: u32,
    /// 每个 Threadgroup 的线程数（32 的整数倍，默认 256）
    pub threads_per_threadgroup: u32,
    /// 派发的 Threadgroup 数量
    pub threadgroup_count: u32,
    /// 总硬件网格线程数 (`threadgroup_count * threads_per_threadgroup`)
    pub total_grid_threads: u32,
}

impl MetalThreadgroupDispatch {
    /// 根据活跃实体行数与管线支持的最大线程数计算最优 32-wide SIMD 对齐派发配置（带饱和防溢出保护）。
    #[inline]
    pub fn compute_1d(active_rows: usize, max_threads_per_group: u32) -> Self {
        let rows = active_rows.min(u32::MAX as usize) as u32;
        if rows == 0 {
            return Self {
                active_rows: 0,
                threads_per_threadgroup: APPLE_SIMD_GROUP_WIDTH,
                threadgroup_count: 0,
                total_grid_threads: 0,
            };
        }
        let clamped_max = max_threads_per_group.clamp(APPLE_SIMD_GROUP_WIDTH, 1024);
        let aligned_group = (clamped_max / APPLE_SIMD_GROUP_WIDTH) * APPLE_SIMD_GROUP_WIDTH;
        let threads_per_threadgroup = if rows < aligned_group {
            (rows.saturating_add(APPLE_SIMD_GROUP_WIDTH - 1) / APPLE_SIMD_GROUP_WIDTH)
                * APPLE_SIMD_GROUP_WIDTH
        } else {
            aligned_group
        };
        let threadgroup_count = rows
            .saturating_add(threads_per_threadgroup - 1)
            / threads_per_threadgroup;
        let total_grid_threads = threadgroup_count.saturating_mul(threads_per_threadgroup);
        Self {
            active_rows: rows,
            threads_per_threadgroup,
            threadgroup_count,
            total_grid_threads,
        }
    }
}

/// 异步 Metal CommandBuffer 生命周期守卫票据。
///
/// 持有所有参与本次 Compute Dispatch 的 16KB 页对齐列块引用（`Arc<dyn Any + Send + Sync>`），
/// 在票据存活期间若 CPU 对同一列发起可变访问，将自动触发写时复制 (COW) 隔离。
#[derive(Default)]
pub struct MetalAsyncCommandTicket {
    command_buffer_id: u64,
    pinned_columns: Vec<Arc<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for MetalAsyncCommandTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalAsyncCommandTicket")
            .field("command_buffer_id", &self.command_buffer_id)
            .field("pinned_column_count", &self.pinned_columns.len())
            .finish()
    }
}

impl MetalAsyncCommandTicket {
    /// 创建指定命令缓冲区 ID 的票据。
    #[inline]
    pub fn new(command_buffer_id: u64) -> Self {
        Self {
            command_buffer_id,
            pinned_columns: Vec::new(),
        }
    }

    /// 将 `GpuColumnPin<T>` 绑定至当前生命周期票据中，延长其 16KB 物理内存页引用计数直至票据释放。
    #[inline]
    pub fn retain_pin<T: GpuComponent>(&mut self, pin: GpuColumnPin<T>) {
        self.pinned_columns.push(Arc::new(pin));
    }

    /// 命令缓冲区唯一序号。
    #[inline]
    pub fn command_buffer_id(&self) -> u64 {
        self.command_buffer_id
    }

    /// 当前锁定的列缓冲区数量。
    #[inline]
    pub fn pinned_column_count(&self) -> usize {
        self.pinned_columns.len()
    }
}

/// 已通过 `naga` 编译与类型验证的 Compute Pipeline 描述。
#[derive(Debug, Clone)]
pub struct MetalCompiledPipeline {
    /// 内核入口函数名
    pub entry_point: &'static str,
    /// 原始生成的 MSL 源码
    pub msl_source: String,
    /// 原始生成的 WGSL 源码
    pub wgsl_source: String,
    /// 经 `naga` WGSL 前端解析 + IR 验证器校验后由 `naga::back::msl` 生成的规范化 MSL 源码
    pub naga_translated_msl: String,
    /// 单 Threadgroup 最大线程数
    pub max_threads_per_threadgroup: u32,
    /// 是否已通过 `naga` 编译器前端与验证器完整校验
    pub shader_validated: bool,
}

/// 使用 `naga` 对内核描述符中的 WGSL 与 MSL 源码执行真实语法解析、IR 类型验证与 MSL 跨语言编译校验。
pub fn compile_and_validate_pipeline(
    descriptor: &GpuKernelDescriptor,
) -> Result<MetalCompiledPipeline, StateError> {
    let wgsl_src = descriptor.wgsl_source();
    let module = naga::front::wgsl::parse_str(wgsl_src).map_err(|err| {
        StateError::AccessDenied(format!(
            "WGSL shader parse error in kernel `{}`:\n{}\nSource:\n{}",
            descriptor.name(),
            err.emit_to_string(wgsl_src),
            wgsl_src
        ))
    })?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let module_info = validator.validate(&module).map_err(|err| {
        StateError::AccessDenied(format!(
            "WGSL shader IR validation error in kernel `{}`: {:?}\nSource:\n{}",
            descriptor.name(),
            err,
            wgsl_src
        ))
    })?;

    let mut msl_options = naga::back::msl::Options::default();
    let mut ep_resources = naga::back::msl::EntryPointResources::default();
    let total_bindings = descriptor.bindings().len() + 1;
    for idx in 0..total_bindings {
        let is_mut = if idx < descriptor.bindings().len() {
            let b = &descriptor.bindings()[idx];
            b.mutable && super::kernel::is_param_written_in_source(b.param_name, wgsl_src)
        } else {
            false
        };
        ep_resources.resources.insert(
            naga::ResourceBinding {
                group: 0,
                binding: idx as u32,
            },
            naga::back::msl::BindTarget {
                buffer: Some(idx as u8),
                mutable: is_mut,
                ..Default::default()
            },
        );
    }
    msl_options
        .per_entry_point_map
        .insert(descriptor.name().to_string(), ep_resources);

    let msl_pipeline_options = naga::back::msl::PipelineOptions::default();
    let (naga_translated_msl, _) =
        naga::back::msl::write_string(&module, &module_info, &msl_options, &msl_pipeline_options)
            .map_err(|err| {
                StateError::AccessDenied(format!(
                    "Naga MSL translation error in kernel `{}`: {:?}",
                    descriptor.name(),
                    err
                ))
            })?;

    Ok(MetalCompiledPipeline {
        entry_point: descriptor.name(),
        msl_source: descriptor.msl_source().to_string(),
        wgsl_source: wgsl_src.to_string(),
        naga_translated_msl,
        max_threads_per_threadgroup: DEFAULT_MAX_TOTAL_THREADS_PER_THREADGROUP,
        shader_validated: true,
    })
}

/// GPU 计算引擎运行时执行模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuExecutionMode {
    /// 当前机器已检测到受支持的 GPU 硬件驱动（如 Apple Silicon Metal UMA 设备），直接通过硬件 Compute Pipeline 派发。
    HardwareGpu,
    /// 当前机器未检测到可用 GPU 硬件（如无 GPU 的 Linux 容器/虚拟机），自动回退至 CPU 执行。
    #[default]
    CpuFallback,
}

/// 单张 Archetype 在硬件批量命令缓冲区中的编码项。
#[derive(Debug, Clone)]
pub struct GpuHardwareBatchItem {
    /// 该表的零拷贝 `MTLBuffer` 句柄列表
    pub buffer_handles: Vec<u64>,
    /// 该表的列原始指针列表（供 CPU fallback 或自定义驱动使用）
    pub raw_ptrs: Vec<*mut u8>,
    /// 活跃实体行数
    pub active_rows: usize,
    /// 1D 线程网格配置
    pub grid: MetalThreadgroupDispatch,
    /// 派发 Uniform 参数（含 `scalar_args`）
    pub uniforms: GpuDispatchUniforms,
}

/// 传入 `MetalComputeEngine::dispatch_archetypes_batch_raw` 的单张 Archetype 预检绑定项。
#[derive(Debug, Clone)]
pub struct GpuArchetypeBatchInput {
    /// 原型 ID
    pub archetype_id: ArchetypeId,
    /// 活跃实体行数
    pub active_rows: usize,
    /// 列绑定描述符列表
    pub bindings: Vec<(ComponentTypeId, MetalNoCopyBufferDescriptor)>,
}

/// GPU 计算与命令通道执行错误
#[derive(Debug, thiserror::Error)]
pub enum GpuExecutionError {
    #[error("Storage state error: {0}")]
    State(#[from] StateError),

    #[error("GPU hardware failure: {0}")]
    HardwareFailure(String),

    #[error("Pipeline error: {0}")]
    PipelineError(String),

    #[error("Pass encoding error: {0}")]
    EncodingError(String),
}

impl From<GpuExecutionError> for StateError {
    fn from(err: GpuExecutionError) -> Self {
        match err {
            GpuExecutionError::State(s) => s,
            other => StateError::AccessDenied(other.to_string()),
        }
    }
}

/// 编译后的计算管线句柄（承载硬件 PSO 指针与 CPU 降级执行器）
#[derive(Clone)]
pub struct GpuPipelineHandle {
    pub name: &'static str,
    pub raw_pso: *mut std::ffi::c_void,
    pub max_threads_per_threadgroup: u32,
    pub cpu_runner: Option<GpuCpuChunkRunner>,
}

unsafe impl Send for GpuPipelineHandle {}
unsafe impl Sync for GpuPipelineHandle {}

impl std::fmt::Debug for GpuPipelineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPipelineHandle")
            .field("name", &self.name)
            .field("raw_pso", &self.raw_pso)
            .field("max_threads_per_threadgroup", &self.max_threads_per_threadgroup)
            .finish()
    }
}

/// 多 Kernel 计算通道命令编码器契约
pub trait GpuPassEncoder: std::fmt::Debug + Send {
    /// 在当前计算通道内追加派发一个 Kernel 派发
    fn encode_dispatch(
        &mut self,
        pipeline: &GpuPipelineHandle,
        threadgroups: &MetalThreadgroupDispatch,
        buffers: &[(*mut u8, usize)],
        uniforms: &GpuDispatchUniforms,
    ) -> Result<(), GpuExecutionError>;

    /// 插入 GPU 显存屏障（`MTLBarrierScopeBuffers`）
    fn memory_barrier(&mut self) -> Result<(), GpuExecutionError>;

    /// 结束编码、提交命令缓冲区并单次等待完成
    fn finish(self: Box<Self>) -> Result<(), GpuExecutionError>;
}

/// 底层 GPU 硬件计算驱动契约（如 macOS 原生 Metal 硬件驱动或外部注入的 Vulkan / WebGPU 硬件驱动）。
pub trait GpuHardwareDriver: std::fmt::Debug + Send + Sync {
    /// 硬件设备名称（如 `"Apple M3 Max"`）。
    fn device_name(&self) -> &str;

    /// 硬件是否支持 CPU/GPU 统一内存架构 (UMA)。
    fn has_unified_memory(&self) -> bool;

    /// 获取指定名称的已编译原生 PSO 指针（若存在）
    fn get_pipeline_pso(&self, _name: &'static str) -> Option<*mut std::ffi::c_void> {
        None
    }

    /// 开启一个连续多 Kernel GPU 计算通道编码会话
    fn begin_compute_pass(&mut self) -> Result<Box<dyn GpuPassEncoder>, GpuExecutionError> {
        Ok(Box::new(CpuFallbackPassEncoder::default()))
    }

    /// 在硬件设备上编译指定内核的 Compute Pipeline State，返回该管线支持的最大 Threadgroup 宽度。
    fn compile_pipeline(
        &mut self,
        descriptor: &GpuKernelDescriptor,
        compiled: &MetalCompiledPipeline,
    ) -> Result<u32, StateError>;

    /// 在硬件设备上绑定 16KB 页对齐零拷贝缓冲区（如 `newBufferWithBytesNoCopy`）。
    ///
    /// # Safety
    /// 调用方必须保证 `desc.base_ptr` 在对应 `handle_id` 注销前始终有效且满足 16KB 页对齐。
    unsafe fn register_no_copy_buffer(
        &mut self,
        handle_id: u64,
        desc: &MetalNoCopyBufferDescriptor,
    ) -> Result<(), StateError>;

    /// 当 `MetalBufferCache` 淘汰或失效某个缓存条目时，通知硬件驱动释放对应的 `MTLBuffer` 句柄。
    fn unregister_buffer(&mut self, handle_id: u64);

    /// 在 GPU 硬件命令队列上编码并提交 1D Compute Shader 派发。
    ///
    /// # Safety
    /// 调用方必须保证 `buffer_handles` 与 `raw_ptrs` 与 `descriptor.bindings()` 一一对应且内存有效。
    unsafe fn dispatch_compute(
        &mut self,
        descriptor: &GpuKernelDescriptor,
        buffer_handles: &[u64],
        raw_ptrs: &[*mut u8],
        active_rows: usize,
        grid: &MetalThreadgroupDispatch,
        uniforms: &GpuDispatchUniforms,
    ) -> Result<(), StateError>;

    /// 在单个硬件 CommandBuffer 与 ComputeCommandEncoder 中批量编码并提交多张 Archetype 的 1D Compute 派发。
    ///
    /// 返回实际提交的硬件 CommandBuffer 数量（非空批次为 `1`，空批次为 `0`）。
    ///
    /// # Safety
    /// 调用方必须保证 `items` 中每个条目的 `buffer_handles` 与 `raw_ptrs` 均与 `descriptor.bindings()` 一一对应且内存有效。
    unsafe fn dispatch_compute_batch(
        &mut self,
        descriptor: &GpuKernelDescriptor,
        items: &[GpuHardwareBatchItem],
    ) -> Result<usize, StateError> {
        let mut submitted = 0usize;
        for item in items {
            if item.active_rows == 0 {
                continue;
            }
            unsafe {
                self.dispatch_compute(
                    descriptor,
                    &item.buffer_handles,
                    &item.raw_ptrs,
                    item.active_rows,
                    &item.grid,
                    &item.uniforms,
                )?;
            }
            submitted = 1;
        }
        Ok(submitted)
    }
}

#[cfg(target_os = "macos")]
mod macos_metal_driver {
    use super::*;
    use std::ffi::{CStr, CString, c_char, c_void};

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MTLSize {
        width: usize,
        height: usize,
        depth: usize,
    }

    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }

    #[link(name = "Foundation", kind = "framework")]
    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    struct AutoreleasePool(*mut c_void);

    impl AutoreleasePool {
        #[inline]
        unsafe fn new() -> Self {
            Self(unsafe { objc_autoreleasePoolPush() })
        }
    }

    impl Drop for AutoreleasePool {
        #[inline]
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { objc_autoreleasePoolPop(self.0) };
            }
        }
    }

    #[inline]
    unsafe fn sel(name: &[u8]) -> *mut c_void {
        unsafe { sel_registerName(name.as_ptr() as *const c_char) }
    }

    #[inline]
    unsafe fn objc_release(obj: *mut c_void) {
        if !obj.is_null() {
            let f: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            unsafe { f(obj, sel(b"release\0")) };
        }
    }

    unsafe fn nsstring_from_str(s: &str) -> Option<*mut c_void> {
        let cstr = CString::new(s).ok()?;
        let cls = unsafe { objc_getClass(c"NSString".as_ptr()) };
        if cls.is_null() {
            return None;
        }
        let alloc_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let raw = unsafe { alloc_fn(cls, sel(b"alloc\0")) };
        if raw.is_null() {
            return None;
        }
        let init_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char) -> *mut c_void =
            unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { init_fn(raw, sel(b"initWithUTF8String:\0"), cstr.as_ptr()) };
        Some(ns)
    }

    unsafe fn nserror_description(err: *mut c_void) -> String {
        if err.is_null() {
            return "unknown Metal error".to_string();
        }
        let desc_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns_desc = unsafe { desc_fn(err, sel(b"localizedDescription\0")) };
        if ns_desc.is_null() {
            return "unknown Metal error".to_string();
        }
        let utf8_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const c_char =
            unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let c_ptr = unsafe { utf8_fn(ns_desc, sel(b"UTF8String\0")) };
        if c_ptr.is_null() {
            "unknown Metal error".to_string()
        } else {
            unsafe { CStr::from_ptr(c_ptr).to_string_lossy().into_owned() }
        }
    }

    /// macOS 原生 `Metal.framework` 零拷贝硬件驱动（通过 Objective-C Runtime 直接调用 `MTLDevice` / `MTLCommandQueue`）。
    pub struct MacOsMetalDriver {
        device: *mut c_void,
        command_queue: *mut c_void,
        device_name: String,
        unified_memory: bool,
        pipelines: HashMap<&'static str, (*mut c_void, u32)>,
        buffers: HashMap<u64, *mut c_void>,
    }

    // SAFETY: `MTLDevice`、`MTLCommandQueue`、`MTLComputePipelineState` 与 `MTLBuffer` 在串行 `&mut self` 下安全使用。
    unsafe impl Send for MacOsMetalDriver {}
    unsafe impl Sync for MacOsMetalDriver {}

    impl std::fmt::Debug for MacOsMetalDriver {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MacOsMetalDriver")
                .field("device_name", &self.device_name)
                .field("unified_memory", &self.unified_memory)
                .field("compiled_pipelines", &self.pipelines.len())
                .field("registered_buffers", &self.buffers.len())
                .finish()
        }
    }

    impl MacOsMetalDriver {
        /// 动态探测当前 macOS 系统是否存在可用的 Metal 硬件 GPU。
        pub fn try_probe() -> Option<Box<dyn GpuHardwareDriver>> {
            unsafe {
                let _pool = AutoreleasePool::new();
                let device = MTLCreateSystemDefaultDevice();
                if device.is_null() {
                    return None;
                }
                let queue_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let command_queue = queue_fn(device, sel(b"newCommandQueue\0"));
                if command_queue.is_null() {
                    objc_release(device);
                    return None;
                }

                let name_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let ns_name = name_fn(device, sel(b"name\0"));
                let device_name = if !ns_name.is_null() {
                    let utf8_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const c_char =
                        std::mem::transmute(objc_msgSend as *const ());
                    let ptr = utf8_fn(ns_name, sel(b"UTF8String\0"));
                    if !ptr.is_null() {
                        CStr::from_ptr(ptr).to_string_lossy().into_owned()
                    } else {
                        "Apple Metal GPU".to_string()
                    }
                } else {
                    "Apple Metal GPU".to_string()
                };

                // Objective-C BOOL 在 x86_64 上为 signed char (i8/u8)，在 aarch64 上为 bool (u8)，统一按 u8 != 0 读取以保证跨架构 ABI 安全
                let uma_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u8 =
                    std::mem::transmute(objc_msgSend as *const ());
                let unified_memory = uma_fn(device, sel(b"hasUnifiedMemory\0")) != 0;

                Some(Box::new(Self {
                    device,
                    command_queue,
                    device_name,
                    unified_memory,
                    pipelines: HashMap::new(),
                    buffers: HashMap::new(),
                }))
            }
        }
    }

    impl GpuHardwareDriver for MacOsMetalDriver {
        fn device_name(&self) -> &str {
            &self.device_name
        }

        fn has_unified_memory(&self) -> bool {
            self.unified_memory
        }

        fn compile_pipeline(
            &mut self,
            descriptor: &GpuKernelDescriptor,
            compiled: &MetalCompiledPipeline,
        ) -> Result<u32, StateError> {
            if let Some(&(_, max_threads)) = self.pipelines.get(descriptor.name()) {
                return Ok(max_threads);
            }
            unsafe {
                let _pool = AutoreleasePool::new();
                let Some(ns_source) = nsstring_from_str(&compiled.msl_source) else {
                    return Err(StateError::AccessDenied("Failed to allocate NSString for MSL source".into()));
                };
                let mut err: *mut c_void = std::ptr::null_mut();
                let new_lib_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> *mut c_void = std::mem::transmute(objc_msgSend as *const ());
                let library = new_lib_fn(
                    self.device,
                    sel(b"newLibraryWithSource:options:error:\0"),
                    ns_source,
                    std::ptr::null_mut(),
                    &mut err,
                );
                objc_release(ns_source);
                if library.is_null() {
                    let msg = nserror_description(err);
                    return Err(StateError::AccessDenied(format!(
                        "Metal hardware shader compilation failed for `{}`: {msg}",
                        descriptor.name()
                    )));
                }

                let Some(ns_entry) = nsstring_from_str(descriptor.name()) else {
                    objc_release(library);
                    return Err(StateError::AccessDenied("Failed to allocate NSString for entry point".into()));
                };
                let new_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let function = new_fn(library, sel(b"newFunctionWithName:\0"), ns_entry);
                objc_release(ns_entry);
                objc_release(library);
                if function.is_null() {
                    return Err(StateError::AccessDenied(format!(
                        "Metal kernel function `{}` not found in compiled MTLLibrary",
                        descriptor.name()
                    )));
                }

                let mut pso_err: *mut c_void = std::ptr::null_mut();
                let new_pso_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> *mut c_void = std::mem::transmute(objc_msgSend as *const ());
                let pso = new_pso_fn(
                    self.device,
                    sel(b"newComputePipelineStateWithFunction:error:\0"),
                    function,
                    &mut pso_err,
                );
                objc_release(function);
                if pso.is_null() {
                    let msg = nserror_description(pso_err);
                    return Err(StateError::AccessDenied(format!(
                        "Metal ComputePipelineState creation failed for `{}`: {msg}",
                        descriptor.name()
                    )));
                }

                let max_threads_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                    std::mem::transmute(objc_msgSend as *const ());
                let max_threads = (max_threads_fn(pso, sel(b"maxTotalThreadsPerThreadgroup\0")) as u32)
                    .clamp(APPLE_SIMD_GROUP_WIDTH, DEFAULT_MAX_TOTAL_THREADS_PER_THREADGROUP);
                self.pipelines.insert(descriptor.name(), (pso, max_threads));
                Ok(max_threads)
            }
        }

        unsafe fn register_no_copy_buffer(
            &mut self,
            handle_id: u64,
            desc: &MetalNoCopyBufferDescriptor,
        ) -> Result<(), StateError> {
            self.unregister_buffer(handle_id);
            unsafe {
                let _pool = AutoreleasePool::new();
                let new_buf_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    usize,
                    u64,
                    *const c_void,
                ) -> *mut c_void = std::mem::transmute(objc_msgSend as *const ());
                let mtl_buf = new_buf_fn(
                    self.device,
                    sel(b"newBufferWithBytesNoCopy:length:options:deallocator:\0"),
                    desc.base_ptr as *mut c_void,
                    desc.allocated_bytes,
                    desc.options.0,
                    std::ptr::null(),
                );
                if mtl_buf.is_null() {
                    return Err(StateError::AccessDenied(format!(
                        "MTLDevice newBufferWithBytesNoCopy failed for ptr {:p} (len {})",
                        desc.base_ptr, desc.allocated_bytes
                    )));
                }
                self.buffers.insert(handle_id, mtl_buf);
                Ok(())
            }
        }

        fn unregister_buffer(&mut self, handle_id: u64) {
            if let Some(buf) = self.buffers.remove(&handle_id) {
                unsafe { objc_release(buf) };
            }
        }

        unsafe fn dispatch_compute(
            &mut self,
            descriptor: &GpuKernelDescriptor,
            buffer_handles: &[u64],
            _raw_ptrs: &[*mut u8],
            active_rows: usize,
            grid: &MetalThreadgroupDispatch,
            uniforms: &GpuDispatchUniforms,
        ) -> Result<(), StateError> {
            if active_rows == 0 {
                return Ok(());
            }
            let Some(&(pso, _)) = self.pipelines.get(descriptor.name()) else {
                return Err(StateError::AccessDenied(format!(
                    "Metal pipeline `{}` is not compiled",
                    descriptor.name()
                )));
            };
            unsafe {
                let _pool = AutoreleasePool::new();
                let cmd_buf_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let cmd_buf = cmd_buf_fn(self.command_queue, sel(b"commandBuffer\0"));
                if cmd_buf.is_null() {
                    return Err(StateError::AccessDenied("Failed to create MTLCommandBuffer".into()));
                }
                let enc_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let encoder = enc_fn(cmd_buf, sel(b"computeCommandEncoder\0"));
                if encoder.is_null() {
                    return Err(StateError::AccessDenied(
                        "Failed to create MTLComputeCommandEncoder".into(),
                    ));
                }

                let set_pso_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                set_pso_fn(encoder, sel(b"setComputePipelineState:\0"), pso);

                let set_buf_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    usize,
                    usize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                for (idx, handle_id) in buffer_handles.iter().enumerate() {
                    let Some(&mtl_buf) = self.buffers.get(handle_id) else {
                        let end_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                            std::mem::transmute(objc_msgSend as *const ());
                        end_fn(encoder, sel(b"endEncoding\0"));
                        return Err(StateError::AccessDenied(format!(
                            "Missing registered MTLBuffer for handle_id {handle_id}"
                        )));
                    };
                    set_buf_fn(encoder, sel(b"setBuffer:offset:atIndex:\0"), mtl_buf, 0, idx);
                }

                let set_bytes_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *const c_void,
                    usize,
                    usize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                set_bytes_fn(
                    encoder,
                    sel(b"setBytes:length:atIndex:\0"),
                    uniforms as *const GpuDispatchUniforms as *const c_void,
                    std::mem::size_of::<GpuDispatchUniforms>(),
                    buffer_handles.len(),
                );

                let dispatch_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    MTLSize,
                    MTLSize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                dispatch_fn(
                    encoder,
                    sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                    MTLSize {
                        width: grid.threadgroup_count as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: grid.threads_per_threadgroup as usize,
                        height: 1,
                        depth: 1,
                    },
                );

                let void_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                void_fn(encoder, sel(b"endEncoding\0"));
                void_fn(cmd_buf, sel(b"commit\0"));
                void_fn(cmd_buf, sel(b"waitUntilCompleted\0"));

                let status_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                    std::mem::transmute(objc_msgSend as *const ());
                let status = status_fn(cmd_buf, sel(b"status\0"));
                // MTLCommandBufferStatusCompleted == 4
                if status != 4 {
                    return Err(StateError::AccessDenied(format!(
                        "MTLCommandBuffer execution failed with status {status}"
                    )));
                }
                Ok(())
            }
        }

        unsafe fn dispatch_compute_batch(
            &mut self,
            descriptor: &GpuKernelDescriptor,
            items: &[GpuHardwareBatchItem],
        ) -> Result<usize, StateError> {
            if !items.iter().any(|item| item.active_rows > 0) {
                return Ok(0);
            }
            let Some(&(pso, _)) = self.pipelines.get(descriptor.name()) else {
                return Err(StateError::AccessDenied(format!(
                    "Metal pipeline `{}` is not compiled",
                    descriptor.name()
                )));
            };
            unsafe {
                let _pool = AutoreleasePool::new();
                let cmd_buf_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let cmd_buf = cmd_buf_fn(self.command_queue, sel(b"commandBuffer\0"));
                if cmd_buf.is_null() {
                    return Err(StateError::AccessDenied(
                        "Failed to create MTLCommandBuffer".into(),
                    ));
                }
                let enc_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let encoder = enc_fn(cmd_buf, sel(b"computeCommandEncoder\0"));
                if encoder.is_null() {
                    return Err(StateError::AccessDenied(
                        "Failed to create MTLComputeCommandEncoder".into(),
                    ));
                }

                let set_pso_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                set_pso_fn(encoder, sel(b"setComputePipelineState:\0"), pso);

                let set_buf_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    usize,
                    usize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                let set_bytes_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *const c_void,
                    usize,
                    usize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                let dispatch_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    MTLSize,
                    MTLSize,
                ) = std::mem::transmute(objc_msgSend as *const ());

                for item in items {
                    if item.active_rows == 0 {
                        continue;
                    }
                    for (idx, handle_id) in item.buffer_handles.iter().enumerate() {
                        let Some(&mtl_buf) = self.buffers.get(handle_id) else {
                            let end_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                                std::mem::transmute(objc_msgSend as *const ());
                            end_fn(encoder, sel(b"endEncoding\0"));
                            return Err(StateError::AccessDenied(format!(
                                "Missing registered MTLBuffer for handle_id {handle_id}"
                            )));
                        };
                        set_buf_fn(encoder, sel(b"setBuffer:offset:atIndex:\0"), mtl_buf, 0, idx);
                    }

                    set_bytes_fn(
                        encoder,
                        sel(b"setBytes:length:atIndex:\0"),
                        &item.uniforms as *const GpuDispatchUniforms as *const c_void,
                        std::mem::size_of::<GpuDispatchUniforms>(),
                        item.buffer_handles.len(),
                    );

                    dispatch_fn(
                        encoder,
                        sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                        MTLSize {
                            width: item.grid.threadgroup_count as usize,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: item.grid.threads_per_threadgroup as usize,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                let void_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                void_fn(encoder, sel(b"endEncoding\0"));
                void_fn(cmd_buf, sel(b"commit\0"));
                void_fn(cmd_buf, sel(b"waitUntilCompleted\0"));

                let status_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                    std::mem::transmute(objc_msgSend as *const ());
                let status = status_fn(cmd_buf, sel(b"status\0"));
                // MTLCommandBufferStatusCompleted == 4
                if status != 4 {
                    return Err(StateError::AccessDenied(format!(
                        "MTLCommandBuffer execution failed with status {status}"
                    )));
                }
                Ok(1)
            }
        }

        fn get_pipeline_pso(&self, name: &'static str) -> Option<*mut c_void> {
            self.pipelines.get(name).map(|&(pso, _)| pso)
        }

        fn begin_compute_pass(
            &mut self,
        ) -> Result<Box<dyn GpuPassEncoder>, GpuExecutionError> {
            unsafe {
                let pool = AutoreleasePool::new();
                let cmd_buf_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let cmd_buf = cmd_buf_fn(self.command_queue, sel(b"commandBuffer\0"));
                if cmd_buf.is_null() {
                    return Err(GpuExecutionError::HardwareFailure(
                        "Failed to create MTLCommandBuffer for compute pass".into(),
                    ));
                }
                let enc_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                    std::mem::transmute(objc_msgSend as *const ());
                let encoder = enc_fn(cmd_buf, sel(b"computeCommandEncoder\0"));
                if encoder.is_null() {
                    return Err(GpuExecutionError::HardwareFailure(
                        "Failed to create MTLComputeCommandEncoder for compute pass".into(),
                    ));
                }

                Ok(Box::new(MacOsMetalPassEncoder {
                    device: self.device,
                    cmd_buf,
                    encoder,
                    buffer_cache: HashMap::new(),
                    created_buffers: Vec::new(),
                    finished: false,
                    _pool: pool,
                }))
            }
        }
    }

    /// macOS 原生 Metal 多 Kernel 计算通道命令编码器
    pub struct MacOsMetalPassEncoder {
        device: *mut c_void,
        cmd_buf: *mut c_void,
        encoder: *mut c_void,
        buffer_cache: HashMap<usize, *mut c_void>,
        created_buffers: Vec<*mut c_void>,
        finished: bool,
        _pool: AutoreleasePool,
    }

    impl std::fmt::Debug for MacOsMetalPassEncoder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MacOsMetalPassEncoder")
                .field("cmd_buf", &self.cmd_buf)
                .field("encoder", &self.encoder)
                .field("created_buffers", &self.created_buffers.len())
                .field("finished", &self.finished)
                .finish()
        }
    }

    unsafe impl Send for MacOsMetalPassEncoder {}

    impl MacOsMetalPassEncoder {
        unsafe fn get_or_create_buffer(
            &mut self,
            ptr: *mut u8,
            len: usize,
        ) -> Result<*mut c_void, GpuExecutionError> {
            let key = ptr as usize;
            if let Some(&buf) = self.buffer_cache.get(&key) {
                return Ok(buf);
            }
            unsafe {
                let new_buf_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    usize,
                    u64,
                    *mut c_void,
                ) -> *mut c_void = std::mem::transmute(objc_msgSend as *const ());
                let mtl_buf = new_buf_fn(
                    self.device,
                    sel(b"newBufferWithBytesNoCopy:length:options:deallocator:\0"),
                    ptr as *mut c_void,
                    len,
                    0, // MTLResourceStorageModeShared = 0
                    std::ptr::null_mut(),
                );
                if mtl_buf.is_null() {
                    return Err(GpuExecutionError::HardwareFailure(format!(
                        "Failed to create zero-copy MTLBuffer for pointer {:p} with length {len}",
                        ptr
                    )));
                }
                self.created_buffers.push(mtl_buf);
                self.buffer_cache.insert(key, mtl_buf);
                Ok(mtl_buf)
            }
        }
    }

    impl GpuPassEncoder for MacOsMetalPassEncoder {
        fn encode_dispatch(
            &mut self,
            pipeline: &GpuPipelineHandle,
            threadgroups: &MetalThreadgroupDispatch,
            buffers: &[(*mut u8, usize)],
            uniforms: &GpuDispatchUniforms,
        ) -> Result<(), GpuExecutionError> {
            if threadgroups.active_rows == 0 || threadgroups.threadgroup_count == 0 {
                return Ok(());
            }
            if pipeline.raw_pso.is_null() {
                return Err(GpuExecutionError::PipelineError(format!(
                    "Metal pipeline `{}` has null PSO",
                    pipeline.name
                )));
            }
            unsafe {
                let set_pso_fn: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                set_pso_fn(
                    self.encoder,
                    sel(b"setComputePipelineState:\0"),
                    pipeline.raw_pso,
                );

                let set_buf_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    usize,
                    usize,
                ) = std::mem::transmute(objc_msgSend as *const ());

                for (idx, &(ptr, len)) in buffers.iter().enumerate() {
                    let mtl_buf = self.get_or_create_buffer(ptr, len)?;
                    set_buf_fn(self.encoder, sel(b"setBuffer:offset:atIndex:\0"), mtl_buf, 0, idx);
                }

                let set_bytes_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *const c_void,
                    usize,
                    usize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                set_bytes_fn(
                    self.encoder,
                    sel(b"setBytes:length:atIndex:\0"),
                    uniforms as *const GpuDispatchUniforms as *const c_void,
                    std::mem::size_of::<GpuDispatchUniforms>(),
                    buffers.len(),
                );

                let dispatch_fn: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    MTLSize,
                    MTLSize,
                ) = std::mem::transmute(objc_msgSend as *const ());
                dispatch_fn(
                    self.encoder,
                    sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                    MTLSize {
                        width: threadgroups.threadgroup_count as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threadgroups.threads_per_threadgroup as usize,
                        height: 1,
                        depth: 1,
                    },
                );

                Ok(())
            }
        }

        fn memory_barrier(&mut self) -> Result<(), GpuExecutionError> {
            unsafe {
                let void_fn_usize: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) =
                    std::mem::transmute(objc_msgSend as *const ());
                // MTLBarrierScopeBuffers = 1 << 0 = 1
                void_fn_usize(self.encoder, sel(b"memoryBarrierWithScope:\0"), 1);
                Ok(())
            }
        }

        fn finish(mut self: Box<Self>) -> Result<(), GpuExecutionError> {
            unsafe {
                let void_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                void_fn(self.encoder, sel(b"endEncoding\0"));
                void_fn(self.cmd_buf, sel(b"commit\0"));
                void_fn(self.cmd_buf, sel(b"waitUntilCompleted\0"));
                self.finished = true;

                let status_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                    std::mem::transmute(objc_msgSend as *const ());
                let status = status_fn(self.cmd_buf, sel(b"status\0"));
                // MTLCommandBufferStatusCompleted == 4
                if status != 4 {
                    return Err(GpuExecutionError::HardwareFailure(format!(
                        "MTLCommandBuffer execution failed with status {status}"
                    )));
                }

                for buf in self.created_buffers.drain(..) {
                    objc_release(buf);
                }
                Ok(())
            }
        }
    }

    impl Drop for MacOsMetalPassEncoder {
        fn drop(&mut self) {
            unsafe {
                if !self.finished && !self.encoder.is_null() {
                    let void_fn: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                        std::mem::transmute(objc_msgSend as *const ());
                    void_fn(self.encoder, sel(b"endEncoding\0"));
                }
                for buf in self.created_buffers.drain(..) {
                    objc_release(buf);
                }
            }
        }
    }

    impl Drop for MacOsMetalDriver {
        fn drop(&mut self) {
            unsafe {
                for (_, buf) in self.buffers.drain() {
                    objc_release(buf);
                }
                for (_, (pso, _)) in self.pipelines.drain() {
                    objc_release(pso);
                }
                objc_release(self.command_queue);
                objc_release(self.device);
            }
        }
    }
}

/// 无硬件 GPU 时的 CPU 回退驱动
#[derive(Debug, Default)]
pub struct CpuFallbackDriver;

impl GpuHardwareDriver for CpuFallbackDriver {
    fn device_name(&self) -> &str {
        "Sumeru CPU Fallback Driver"
    }

    fn has_unified_memory(&self) -> bool {
        true
    }

    fn compile_pipeline(
        &mut self,
        _descriptor: &GpuKernelDescriptor,
        _compiled: &MetalCompiledPipeline,
    ) -> Result<u32, StateError> {
        Ok(256)
    }

    unsafe fn register_no_copy_buffer(
        &mut self,
        _handle_id: u64,
        _desc: &MetalNoCopyBufferDescriptor,
    ) -> Result<(), StateError> {
        Ok(())
    }

    fn unregister_buffer(&mut self, _handle_id: u64) {}

    unsafe fn dispatch_compute(
        &mut self,
        descriptor: &GpuKernelDescriptor,
        _buffer_handles: &[u64],
        raw_ptrs: &[*mut u8],
        active_rows: usize,
        _grid: &MetalThreadgroupDispatch,
        uniforms: &GpuDispatchUniforms,
    ) -> Result<(), StateError> {
        if active_rows > 0 {
            unsafe {
                (descriptor.cpu_runner())(raw_ptrs, active_rows, uniforms);
            }
        }
        Ok(())
    }

    fn begin_compute_pass(&mut self) -> Result<Box<dyn GpuPassEncoder>, GpuExecutionError> {
        Ok(Box::new(CpuFallbackPassEncoder::default()))
    }
}

/// CPU 回退计算通道编码器
#[derive(Debug, Default)]
pub struct CpuFallbackPassEncoder {
    pub dispatches: usize,
}

impl GpuPassEncoder for CpuFallbackPassEncoder {
    fn encode_dispatch(
        &mut self,
        pipeline: &GpuPipelineHandle,
        _threadgroups: &MetalThreadgroupDispatch,
        buffers: &[(*mut u8, usize)],
        uniforms: &GpuDispatchUniforms,
    ) -> Result<(), GpuExecutionError> {
        let active_rows = uniforms.active_rows as usize;
        if active_rows == 0 {
            return Ok(());
        }
        let runner = pipeline.cpu_runner.as_ref().ok_or_else(|| {
            GpuExecutionError::PipelineError(format!(
                "No CPU fallback runner available for pipeline `{}`",
                pipeline.name
            ))
        })?;
        let raw_ptrs: Vec<*mut u8> = buffers.iter().map(|&(p, _)| p).collect();
        unsafe {
            runner(&raw_ptrs, active_rows, uniforms);
        }
        self.dispatches += 1;
        Ok(())
    }

    fn memory_barrier(&mut self) -> Result<(), GpuExecutionError> {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), GpuExecutionError> {
        Ok(())
    }
}

/// 动态探测当前运行机器是否存在受支持的硬件 GPU 驱动后端。
///
/// - 在 macOS (`aarch64` / `x86_64`) 上通过 `MTLCreateSystemDefaultDevice()` 探测真实 Metal GPU 硬件；
/// - 若当前机器无可用 GPU 硬件（如无 GPU 的容器/VM），返回 `None`，由 [`MetalComputeEngine`] 自动切换至 [`GpuExecutionMode::CpuFallback`]。
pub fn probe_system_gpu_driver() -> Option<Box<dyn GpuHardwareDriver>> {
    #[cfg(target_os = "macos")]
    {
        macos_metal_driver::MacOsMetalDriver::try_probe()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// 单次 Archetype 列派发的统计回执。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuDispatchStats {
    /// 派发的非空 Archetype 数量
    pub dispatched_archetypes: usize,
    /// 处理的活跃实体总行数
    pub total_rows: usize,
    /// 涉及的 16KB 硬件页总数
    pub total_pages_16k: usize,
    /// `MTLBuffer` 缓存命中次数
    pub buffer_cache_hits: usize,
    /// `MTLBuffer` 缓存新建次数
    pub buffer_cache_misses: usize,
    /// 本次派发实际采用的执行模式（`HardwareGpu` 真实硬件执行或 `CpuFallback` 无硬件回退）
    pub execution_mode: GpuExecutionMode,
    /// 实际向 GPU 硬件提交的 CommandBuffer 次数（多表合并派发在单次查询中固定为 1，空查询或 CPU Fallback 为 0）
    pub command_buffer_submissions: usize,
    /// 多算子计算通道中根据读写冲突自动插入或手动插入的显存屏障 (`MTLBarrierScopeBuffers`) 数量
    pub barriers_inserted: usize,
}

/// Metal / 跨平台 16KB 页对齐计算引擎（管理硬件 GPU 驱动探测、Naga Pipeline 编译验证缓存、`MTLBuffer` 零拷贝句柄缓存与计算派发）。
#[derive(Debug)]
pub struct MetalComputeEngine {
    device_name: String,
    unified_memory: bool,
    execution_mode: GpuExecutionMode,
    driver: Option<Box<dyn GpuHardwareDriver>>,
    buffer_cache: MetalBufferCache,
    pipelines: HashMap<&'static str, MetalCompiledPipeline>,
    next_command_buffer_id: u64,
    total_dispatches: u64,
}

impl Default for MetalComputeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalComputeEngine {
    /// 创建计算引擎实例：自动探测当前机器是否支持硬件 GPU；若支持则启用硬件 GPU 派发，若不支持则平滑回退至 CPU 执行。
    pub fn new() -> Self {
        match probe_system_gpu_driver() {
            Some(driver) => Self::with_driver(driver),
            None => Self::new_cpu_fallback(),
        }
    }

    /// 使用指定的硬件 GPU 驱动构造计算引擎（启用 [`GpuExecutionMode::HardwareGpu`]）。
    pub fn with_driver(driver: Box<dyn GpuHardwareDriver>) -> Self {
        let device_name = driver.device_name().to_string();
        let unified_memory = driver.has_unified_memory();
        Self {
            device_name,
            unified_memory,
            execution_mode: GpuExecutionMode::HardwareGpu,
            driver: Some(driver),
            buffer_cache: MetalBufferCache::new(),
            pipelines: HashMap::new(),
            next_command_buffer_id: 1,
            total_dispatches: 0,
        }
    }

    /// 显式构造无硬件 GPU 时的 CPU 回退计算引擎（启用 [`GpuExecutionMode::CpuFallback`]）。
    pub fn new_cpu_fallback() -> Self {
        Self {
            device_name: "Sumeru 16KB Page-Aligned CPU Fallback Engine (Naga WGSL/MSL Validated)"
                .to_string(),
            unified_memory: true,
            execution_mode: GpuExecutionMode::CpuFallback,
            driver: None,
            buffer_cache: MetalBufferCache::new(),
            pipelines: HashMap::new(),
            next_command_buffer_id: 1,
            total_dispatches: 0,
        }
    }

    /// 后端设备名称。
    #[inline]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// 当前引擎是否绑定了可用的真实 GPU 硬件驱动。
    #[inline]
    pub fn has_hardware_gpu(&self) -> bool {
        self.driver.is_some()
    }

    /// 当前引擎的运行时执行模式（[`GpuExecutionMode::HardwareGpu`] 或 [`GpuExecutionMode::CpuFallback`]）。
    #[inline]
    pub fn execution_mode(&self) -> GpuExecutionMode {
        self.execution_mode
    }

    /// 是否支持 CPU/GPU 统一内存 (`StorageModeShared` + `NoCopy`)。
    #[inline]
    pub fn has_unified_memory(&self) -> bool {
        self.unified_memory
    }

    /// 获取内部零拷贝 `MTLBuffer` 缓存的只读引用。
    #[inline]
    pub fn buffer_cache(&self) -> &MetalBufferCache {
        &self.buffer_cache
    }

    /// 获取内部零拷贝 `MTLBuffer` 缓存的可变引用（支持手动清理或按 Archetype 淘汰）。
    #[inline]
    pub fn buffer_cache_mut(&mut self) -> &mut MetalBufferCache {
        &mut self.buffer_cache
    }

    /// 淘汰指定 Archetype 的零拷贝缓存句柄，并同步释放底层 GPU 驱动中的硬件缓冲区句柄。
    pub fn evict_archetype_buffers(&mut self, archetype_id: ArchetypeId) {
        let removed = self.buffer_cache.evict_archetype(archetype_id);
        if let Some(driver) = &mut self.driver {
            for handle_id in removed {
                driver.unregister_buffer(handle_id);
            }
        }
    }

    /// 清空全部零拷贝缓存句柄，并同步释放底层 GPU 驱动中的硬件缓冲区句柄。
    pub fn clear_buffer_cache(&mut self) {
        let removed = self.buffer_cache.clear();
        if let Some(driver) = &mut self.driver {
            for handle_id in removed {
                driver.unregister_buffer(handle_id);
            }
        }
    }

    /// 累计派发次数。
    #[inline]
    pub fn total_dispatches(&self) -> u64 {
        self.total_dispatches
    }

    /// 确保指定内核已通过 `naga` WGSL 解析、IR 类型校验与 MSL 编译转换（若存在硬件 GPU 驱动则同步编译硬件 PSO），并缓存管线。
    pub fn ensure_pipeline_compiled(
        &mut self,
        descriptor: &GpuKernelDescriptor,
    ) -> Result<&MetalCompiledPipeline, StateError> {
        let name = descriptor.name();
        if !self.pipelines.contains_key(name) {
            let mut compiled = compile_and_validate_pipeline(descriptor)?;
            if let Some(driver) = &mut self.driver {
                let hw_max_threads = driver.compile_pipeline(descriptor, &compiled)?;
                compiled.max_threads_per_threadgroup = hw_max_threads;
            }
            self.pipelines.insert(name, compiled);
        }
        Ok(self.pipelines.get(name).unwrap())
    }

    /// 获取或即时编译指定内核的 `MetalCompiledPipeline`（若着色器语法/类型非法将 panic）。
    pub fn get_or_compile_pipeline(
        &mut self,
        descriptor: &GpuKernelDescriptor,
    ) -> &MetalCompiledPipeline {
        self.ensure_pipeline_compiled(descriptor)
            .expect("GPU kernel shader failed Naga WGSL/MSL compilation and validation")
    }

    /// 获取指定内核的编译管线句柄（包含硬件 PSO 与 CPU 降级执行器）。
    pub fn get_pipeline_handle(
        &mut self,
        descriptor: &GpuKernelDescriptor,
    ) -> Result<GpuPipelineHandle, GpuExecutionError> {
        let compiled = self.ensure_pipeline_compiled(descriptor)?;
        let max_threads = compiled.max_threads_per_threadgroup;
        let raw_pso = self
            .driver
            .as_ref()
            .and_then(|d| d.get_pipeline_pso(descriptor.name()))
            .unwrap_or(std::ptr::null_mut());

        Ok(GpuPipelineHandle {
            name: descriptor.name(),
            raw_pso,
            max_threads_per_threadgroup: max_threads,
            cpu_runner: Some(descriptor.cpu_runner()),
        })
    }

    /// 开启一个连续多 Kernel GPU 计算通道编码会话 (`GpuPassEncoder`)。
    pub fn begin_compute_pass(
        &mut self,
    ) -> Result<Box<dyn GpuPassEncoder>, GpuExecutionError> {
        match &mut self.driver {
            Some(driver) => driver.begin_compute_pass(),
            None => Ok(Box::new(CpuFallbackPassEncoder::default())),
        }
    }

    /// 分配新的生命周期守卫命令票据。
    #[inline]
    pub fn next_command_ticket(&mut self) -> MetalAsyncCommandTicket {
        let id = self.next_command_buffer_id;
        self.next_command_buffer_id += 1;
        MetalAsyncCommandTicket::new(id)
    }

    /// 对单个 Archetype 的 16KB 页对齐列缓冲区集合执行零拷贝校验、句柄缓存绑定与 Compute 内核派发。
    ///
    /// - 当检测到可用硬件 GPU 驱动（`self.driver.is_some()`）时，通过硬件驱动绑定零拷贝 `MTLBuffer` 并提交 GPU Compute 派发；
    /// - 当机器不支持或未检测到硬件 GPU（`self.driver.is_none()`）时，自动回退至 CPU 执行器 `descriptor.cpu_runner()`。
    ///
    /// # Safety
    /// 调用方必须保证 `bindings` 与 `descriptor.bindings()` 一一对应，且可变绑定具有独占访问权。
    pub unsafe fn dispatch_archetype_raw(
        &mut self,
        archetype_id: ArchetypeId,
        active_rows: usize,
        descriptor: &GpuKernelDescriptor,
        bindings: &[(ComponentTypeId, MetalNoCopyBufferDescriptor)],
    ) -> Result<(MetalThreadgroupDispatch, usize, usize), StateError> {
        let max_threads = self
            .ensure_pipeline_compiled(descriptor)?
            .max_threads_per_threadgroup;
        let dispatch = MetalThreadgroupDispatch::compute_1d(active_rows, max_threads);

        let mut hits = 0usize;
        let mut misses = 0usize;
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(bindings.len());
        let mut buffer_handles: Vec<u64> = Vec::with_capacity(bindings.len());

        for (cid, buf_desc) in bindings {
            let (entry, hit, evicted) =
                self.buffer_cache
                    .get_or_insert_with_evicted(archetype_id, *cid, buf_desc);
            if let Some(evicted_id) = evicted
                && let Some(driver) = &mut self.driver
            {
                driver.unregister_buffer(evicted_id);
            }
            if hit {
                hits += 1;
            } else {
                misses += 1;
                if let Some(driver) = &mut self.driver
                    && let Err(err) =
                        unsafe { driver.register_no_copy_buffer(entry.handle_id, buf_desc) }
                {
                    self.buffer_cache.remove_entry(archetype_id, *cid);
                    return Err(err);
                }
            }
            buffer_handles.push(entry.handle_id);
            ptrs.push(buf_desc.base_ptr);
        }

        let uniforms = GpuDispatchUniforms {
            active_rows: dispatch.active_rows,
            threads_per_group: dispatch.threads_per_threadgroup,
            archetype_id: archetype_id.0,
            _pad: 0,
            scalar_args: [0; 4],
        };

        if active_rows > 0 {
            match &mut self.driver {
                Some(driver) => unsafe {
                    driver.dispatch_compute(
                        descriptor,
                        &buffer_handles,
                        &ptrs,
                        active_rows,
                        &dispatch,
                        &uniforms,
                    )?;
                },
                None => unsafe {
                    (descriptor.cpu_runner())(&ptrs, active_rows, &uniforms);
                },
            }
        }
        self.total_dispatches += 1;
        Ok((dispatch, hits, misses))
    }

    /// 在单个硬件 CommandBuffer 与 ComputeCommandEncoder 中对多张 Archetype 的 16KB 页对齐列缓冲区集合执行批量零拷贝绑定与 Compute 派发。
    ///
    /// 返回 `Ok((total_hits, total_misses, command_buffer_submissions))`。
    ///
    /// # Safety
    /// 调用方必须保证 `inputs` 中每张表的 `bindings` 与 `descriptor.bindings()` 一一对应，且可变绑定具有独占访问权。
    pub unsafe fn dispatch_archetypes_batch_raw(
        &mut self,
        descriptor: &GpuKernelDescriptor,
        inputs: &[GpuArchetypeBatchInput],
        scalar_args: [u32; 4],
    ) -> Result<(usize, usize, usize), StateError> {
        let max_threads = self
            .ensure_pipeline_compiled(descriptor)?
            .max_threads_per_threadgroup;
        if inputs.is_empty() {
            return Ok((0, 0, 0));
        }

        let mut total_hits = 0usize;
        let mut total_misses = 0usize;
        let mut batch_items = Vec::with_capacity(inputs.len());

        for input in inputs {
            let dispatch = MetalThreadgroupDispatch::compute_1d(input.active_rows, max_threads);
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(input.bindings.len());
            let mut buffer_handles: Vec<u64> = Vec::with_capacity(input.bindings.len());

            for (cid, buf_desc) in &input.bindings {
                let (entry, hit, evicted) = self.buffer_cache.get_or_insert_with_evicted(
                    input.archetype_id,
                    *cid,
                    buf_desc,
                );
                if let Some(evicted_id) = evicted
                    && let Some(driver) = &mut self.driver
                {
                    driver.unregister_buffer(evicted_id);
                }
                if hit {
                    total_hits += 1;
                } else {
                    total_misses += 1;
                    if let Some(driver) = &mut self.driver
                        && let Err(err) =
                            unsafe { driver.register_no_copy_buffer(entry.handle_id, buf_desc) }
                    {
                        self.buffer_cache.remove_entry(input.archetype_id, *cid);
                        return Err(err);
                    }
                }
                buffer_handles.push(entry.handle_id);
                ptrs.push(buf_desc.base_ptr);
            }

            let uniforms = GpuDispatchUniforms {
                active_rows: dispatch.active_rows,
                threads_per_group: dispatch.threads_per_threadgroup,
                archetype_id: input.archetype_id.0,
                _pad: 0,
                scalar_args,
            };

            batch_items.push(GpuHardwareBatchItem {
                buffer_handles,
                raw_ptrs: ptrs,
                active_rows: input.active_rows,
                grid: dispatch,
                uniforms,
            });
        }

        let submissions = match &mut self.driver {
            Some(driver) => unsafe { driver.dispatch_compute_batch(descriptor, &batch_items)? },
            None => {
                for item in &batch_items {
                    if item.active_rows > 0 {
                        unsafe {
                            (descriptor.cpu_runner())(
                                &item.raw_ptrs,
                                item.active_rows,
                                &item.uniforms,
                            );
                        }
                    }
                }
                0
            }
        };

        self.total_dispatches += batch_items.len() as u64;
        Ok((total_hits, total_misses, submissions))
    }
}

