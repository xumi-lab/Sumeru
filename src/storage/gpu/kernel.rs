//! 多平台 GPU Compute 内核抽象、着色器生成 (MSL / WGSL) 与三层并行算子体系 (`#[gpu]`)
//!
//! 支持在同一份 Rust 代码中定义组件与内核函数：
//! ```rust,ignore
//! #[gpu]
//! pub struct SimPosition { pub x: f32, pub y: f32 }
//!
//! #[gpu]
//! pub struct SimVelocity { pub vx: f32, pub vy: f32 }
//!
//! #[gpu]
//! pub fn integrate_sim_motion(pos: &mut SimPosition, vel: &SimVelocity) {
//!     pos.x += vel.vx;
//!     pos.y += vel.vy;
//! }
//! ```
//! 并支持三层显式遍历方式：
//! 1. **CPU 单线程**：闭包 `store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.for_each(|(pos, vel)| { pos.x += vel.vx; })`
//!    或算子直传 `store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.run(integrate_sim_motion)`
//! 2. **CPU 多核并行 (Rayon)**：闭包 `store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.par_for_each(|(pos, vel)| ...)`
//!    或算子直传 `.par_run(integrate_sim_motion)` / `.par().run(integrate_sim_motion)`
//! 3. **GPU 硬件并行 (16KB 页对齐零拷贝)**：`store.query_mut::<(&mut SimPosition, &SimVelocity)>()?.gpu().for_each(integrate_sim_motion)?`
//!    或 `.gpu().run(integrate_sim_motion)?`

use super::ir::GpuScalarArg;
use super::GpuComponent;
use crate::id::ComponentTypeId;
use crate::storage::query::QueryMutData;
pub use crate::storage::query::{
    ClosureOpMarker, KernelOpMarker, QueryForEachOp, QueryParForEachOp, QueryRowItem,
};
use bytemuck::{Pod, Zeroable};

/// 传递给 GPU Compute Shader 常量缓冲区的派发元信息 (`[[buffer(N)]]` / `@group(0) @binding(N)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDispatchUniforms {
    /// 当前 Archetype 的有效元素行数（线程越界保护上界 `if (gid >= uniforms.active_rows) return;`）
    pub active_rows: u32,
    /// 每个 Threadgroup / Workgroup 的线程数
    pub threads_per_group: u32,
    /// 当前派发的 Archetype ID
    pub archetype_id: u32,
    /// 16 字节对齐保留位
    pub _pad: u32,
    /// 运行时动态标量参数槽位（按 32-bit 位模式承载最多 4 个 `f32` / `i32` / `u32` / `bool` 参数）
    pub scalar_args: [u32; 4],
}

/// 将 1~4 个运行时动态标量参数打包为 `GpuDispatchUniforms::scalar_args` 的转换契约。
pub trait IntoGpuScalarUniforms<const N: usize>: Copy + Send + Sync + 'static {
    /// 打包为 4 个 32-bit Uniform 字。
    fn into_scalar_words(self) -> [u32; 4];
}

impl<T0: GpuScalarArg> IntoGpuScalarUniforms<1> for T0 {
    #[inline(always)]
    fn into_scalar_words(self) -> [u32; 4] {
        [self.to_uniform_word(), 0, 0, 0]
    }
}

impl<T0: GpuScalarArg> IntoGpuScalarUniforms<1> for (T0,) {
    #[inline(always)]
    fn into_scalar_words(self) -> [u32; 4] {
        [self.0.to_uniform_word(), 0, 0, 0]
    }
}

impl<T0: GpuScalarArg, T1: GpuScalarArg> IntoGpuScalarUniforms<2> for (T0, T1) {
    #[inline(always)]
    fn into_scalar_words(self) -> [u32; 4] {
        [self.0.to_uniform_word(), self.1.to_uniform_word(), 0, 0]
    }
}

impl<T0: GpuScalarArg, T1: GpuScalarArg, T2: GpuScalarArg> IntoGpuScalarUniforms<3>
    for (T0, T1, T2)
{
    #[inline(always)]
    fn into_scalar_words(self) -> [u32; 4] {
        [
            self.0.to_uniform_word(),
            self.1.to_uniform_word(),
            self.2.to_uniform_word(),
            0,
        ]
    }
}

impl<T0: GpuScalarArg, T1: GpuScalarArg, T2: GpuScalarArg, T3: GpuScalarArg>
    IntoGpuScalarUniforms<4> for (T0, T1, T2, T3)
{
    #[inline(always)]
    fn into_scalar_words(self) -> [u32; 4] {
        [
            self.0.to_uniform_word(),
            self.1.to_uniform_word(),
            self.2.to_uniform_word(),
            self.3.to_uniform_word(),
        ]
    }
}

/// 绑定了运行时动态标量参数的 GPU/CPU 三层算子包装器（由 `kernel.with_args(...)` 构造）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBoundKernel<K> {
    /// 算子零大小标识类型
    pub kernel: K,
    /// 编码后的 4 个 32-bit 动态标量位字
    pub scalar_words: [u32; 4],
}

/// 跨平台着色器字段 32-bit 标量/紧凑数组类型枚举（保证 Rust `#[repr(C)]`、Apple MSL 与 WebGPU WGSL 内存布局 100% 一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFieldType {
    F32,
    I32,
    U32,
    Vec2F32,
    Vec3F32,
    Vec4F32,
    Vec2U32,
    Vec3U32,
    Vec4U32,
    Vec2I32,
    Vec3I32,
    Vec4I32,
}

impl GpuFieldType {
    /// 映射为 Apple Metal Shading Language (MSL) 类型名（4 字节对齐）。
    #[inline]
    pub const fn msl_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::Vec2F32 => "packed_float2",
            Self::Vec3F32 => "packed_float3",
            Self::Vec4F32 => "packed_float4",
            Self::Vec2U32 => "packed_uint2",
            Self::Vec3U32 => "packed_uint3",
            Self::Vec4U32 => "packed_uint4",
            Self::Vec2I32 => "packed_int2",
            Self::Vec3I32 => "packed_int3",
            Self::Vec4I32 => "packed_int4",
        }
    }

    /// 映射为 WebGPU Shading Language (WGSL) 类型名（4 字节对齐，与 Rust 数组及 MSL `packed_*` 布局完全一致）。
    #[inline]
    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Vec2F32 => "array<f32, 2>",
            Self::Vec3F32 => "array<f32, 3>",
            Self::Vec4F32 => "array<f32, 4>",
            Self::Vec2U32 => "array<u32, 2>",
            Self::Vec3U32 => "array<u32, 3>",
            Self::Vec4U32 => "array<u32, 4>",
            Self::Vec2I32 => "array<i32, 2>",
            Self::Vec3I32 => "array<i32, 3>",
            Self::Vec4I32 => "array<i32, 4>",
        }
    }

    /// 字段字节大小。
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::Vec2F32 | Self::Vec2U32 | Self::Vec2I32 => 8,
            Self::Vec3F32 | Self::Vec3U32 | Self::Vec3I32 => 12,
            Self::Vec4F32 | Self::Vec4U32 | Self::Vec4I32 => 16,
        }
    }

    /// 字段字节对齐要求（所有跨平台着色器标量/紧凑数组统一为 4 字节对齐）。
    #[inline]
    pub const fn byte_align(self) -> usize {
        4
    }
}

/// Rust 基础 32-bit 数值/数组类型到 `GpuFieldType` 的编译期映射 Trait。
pub trait GpuScalarType: Copy + Send + Sync + 'static {
    const FIELD_TYPE: GpuFieldType;
}

impl GpuScalarType for f32 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::F32;
}
impl GpuScalarType for i32 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::I32;
}
impl GpuScalarType for u32 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::U32;
}
impl GpuScalarType for [f32; 2] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2F32;
}
impl GpuScalarType for [f32; 3] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3F32;
}
impl GpuScalarType for [f32; 4] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4F32;
}
impl GpuScalarType for [u32; 2] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2U32;
}
impl GpuScalarType for [u32; 3] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3U32;
}
impl GpuScalarType for [u32; 4] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4U32;
}
impl GpuScalarType for [i32; 2] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2I32;
}
impl GpuScalarType for [i32; 3] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3I32;
}
impl GpuScalarType for [i32; 4] {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4I32;
}
impl GpuScalarType for super::ir::Vec2 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec2F32;
}
impl GpuScalarType for super::ir::Vec3 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec3F32;
}
impl GpuScalarType for super::ir::Vec4 {
    const FIELD_TYPE: GpuFieldType = GpuFieldType::Vec4F32;
}

/// GPU 组件结构体字段元数据。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuStructField {
    pub name: &'static str,
    pub field_type: GpuFieldType,
}

/// 具备自动生成 MSL / WGSL 结构体定义能力的 GPU 组件契约。
pub trait GpuShaderStruct: GpuComponent {
    /// 着色器内的结构体名称。
    const STRUCT_NAME: &'static str;
    /// 字段列表。
    const FIELDS: &'static [GpuStructField];

    /// 校验 Rust `#[repr(C)]` 结构体布局与着色器字段无隐式填充字节差异。
    #[inline]
    fn assert_shader_layout() {
        <Self as GpuComponent>::assert_gpu_layout();
        let expected_size: usize = Self::FIELDS.iter().map(|f| f.field_type.byte_size()).sum();
        assert_eq!(
            std::mem::size_of::<Self>(),
            expected_size,
            "GpuShaderStruct `{}` has unrepresented padding bytes (Rust size {} != shader field sum {})",
            Self::STRUCT_NAME,
            std::mem::size_of::<Self>(),
            expected_size
        );
        assert_eq!(
            std::mem::align_of::<Self>(),
            4,
            "GpuShaderStruct `{}` must have 4-byte alignment",
            Self::STRUCT_NAME
        );
    }

    /// 生成 Apple Metal Shading Language (MSL) `struct` 声明。
    fn msl_struct_definition() -> String {
        let mut s = format!("struct {} {{\n", Self::STRUCT_NAME);
        for f in Self::FIELDS {
            s.push_str(&format!("    {} {};\n", f.field_type.msl_type(), f.name));
        }
        s.push_str("};\n");
        s
    }

    /// 生成 WebGPU Shading Language (WGSL) `struct` 声明。
    fn wgsl_struct_definition() -> String {
        let mut s = format!("struct {} {{\n", Self::STRUCT_NAME);
        for f in Self::FIELDS {
            s.push_str(&format!("    {}: {},\n", f.name, f.field_type.wgsl_type()));
        }
        s.push_str("};\n");
        s
    }
}

/// 内核单个列参数的绑定规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuKernelBindingSpec {
    /// 内核形参名（如 `"pos"`、`"vel"`）
    pub param_name: &'static str,
    /// 组件结构体名（如 `"SimPosition"`）
    pub struct_name: &'static str,
    /// 组件 `ComponentTypeId`
    pub type_id: ComponentTypeId,
    /// 单个元素字节大小
    pub elem_size: usize,
    /// 是否为可变写绑定 (`&mut T`)
    pub mutable: bool,
}

/// CPU 镜像/回退整列块执行函数签名。
pub type GpuCpuChunkRunner = unsafe fn(
    raw_ptrs: &[*mut u8],
    active_rows: usize,
    uniforms: &GpuDispatchUniforms,
);

/// 完整的跨平台 GPU Compute 内核描述符（包含 MSL 源码、WGSL 源码、列绑定规格与 CPU 镜像执行器）。
///
/// 字段对外只读，仅能通过 `unsafe fn new`（或 `#[gpu]` 宏）构造，防止外部伪造错配的 `bindings` 与 `cpu_runner`。
#[derive(Debug, Clone)]
pub struct GpuKernelDescriptor {
    /// 内核函数名称
    name: &'static str,
    /// 列参数绑定表（顺序与 `Query` 元组严格一致）
    bindings: Vec<GpuKernelBindingSpec>,
    /// 自动生成的 Apple Metal Shading Language (MSL) 完整内核源码
    msl_source: String,
    /// 自动生成的 WebGPU Shading Language (WGSL) 完整内核源码
    wgsl_source: String,
    /// CPU 回退 / UMA 镜像执行函数
    cpu_runner: GpuCpuChunkRunner,
}

impl GpuKernelDescriptor {
    /// 构造内核描述符。
    ///
    /// # Safety
    /// 调用方必须保证 `bindings` 与 `cpu_runner` 解引用的列指针数量、类型大小、对齐及读写权限严格一致。
    #[inline]
    pub unsafe fn new(
        name: &'static str,
        bindings: Vec<GpuKernelBindingSpec>,
        msl_source: String,
        wgsl_source: String,
        cpu_runner: GpuCpuChunkRunner,
    ) -> Self {
        Self {
            name,
            bindings,
            msl_source,
            wgsl_source,
            cpu_runner,
        }
    }

    /// 内核函数名称。
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 列参数绑定规格列表。
    #[inline]
    pub fn bindings(&self) -> &[GpuKernelBindingSpec] {
        &self.bindings
    }

    /// Apple Metal Shading Language (MSL) 完整源码。
    #[inline]
    pub fn msl_source(&self) -> &str {
        &self.msl_source
    }

    /// WebGPU Shading Language (WGSL) 完整源码。
    #[inline]
    pub fn wgsl_source(&self) -> &str {
        &self.wgsl_source
    }

    /// CPU 整列块回退执行器。
    #[inline]
    pub fn cpu_runner(&self) -> GpuCpuChunkRunner {
        self.cpu_runner
    }
}

/// 强类型 GPU 计算内核契约：将内核结构体静态绑定至对应的 ECS 可变查询类型 `Self::Query`。
///
/// # Safety
/// 实现者必须保证 `descriptor()` 返回的 `bindings` 与 `Self::Query` 的列顺序、类型、`elem_size` 及可变性完全一致，
/// 且 `cpu_runner` 仅按 `bindings` 声明的权限与长度访问列切片。
pub unsafe trait GpuKernel<'a>: Copy + Send + Sync + 'static {
    /// 匹配的 `Store::query_mut::<Q>()` 查询类型参数（如 `(&'a mut SimPosition, &'a SimVelocity)`）。
    type Query: QueryMutData<'a>;

    /// 获取静态缓存的多平台编译描述符（MSL + WGSL + CPU Runner）。
    fn descriptor(&self) -> &'static GpuKernelDescriptor;

    /// 获取当前派发绑定的运行时动态标量 Uniform 位字（默认全 0；由 `.with_args(...)` 注入实际值）。
    #[inline]
    fn scalar_uniform_words(&self) -> [u32; 4] {
        [0; 4]
    }
}

/// 带有运行时动态标量 Uniform 参数（通过 `.with_args(...)` 绑定）的 GPU 内核契约。
///
/// # Safety
/// 实现者必须保证 `descriptor()` 返回的 `bindings` 与 `Self::Query` 的列顺序、类型、`elem_size` 及可变性完全一致。
pub unsafe trait GpuParameterizedKernel<'a>: Copy + Send + Sync + 'static {
    /// 匹配的 `Store::query_mut::<Q>()` 查询类型参数。
    type Query: QueryMutData<'a>;

    /// 获取静态缓存的多平台编译描述符（MSL + WGSL + CPU Runner）。
    fn descriptor(&self) -> &'static GpuKernelDescriptor;

    /// 在 CPU 单行上附带已打包的 4 个 32-bit Uniform 标量槽位执行算子。
    fn call_row_with_uniforms(
        &self,
        item: <Self::Query as QueryMutData<'a>>::Item,
        scalar_words: [u32; 4],
    );
}

unsafe impl<'a, K: GpuParameterizedKernel<'a>> GpuKernel<'a> for GpuBoundKernel<K> {
    type Query = K::Query;

    #[inline(always)]
    fn descriptor(&self) -> &'static GpuKernelDescriptor {
        self.kernel.descriptor()
    }

    #[inline(always)]
    fn scalar_uniform_words(&self) -> [u32; 4] {
        self.scalar_words
    }
}

impl<'a, K: GpuParameterizedKernel<'a>>
    crate::QueryForEachOp<<K::Query as QueryMutData<'a>>::Item, KernelOpMarker>
    for GpuBoundKernel<K>
{
    #[inline(always)]
    fn call_row(&mut self, item: <K::Query as QueryMutData<'a>>::Item) {
        self.kernel.call_row_with_uniforms(item, self.scalar_words);
    }
}

impl<'a, K: GpuParameterizedKernel<'a>>
    crate::QueryParForEachOp<<K::Query as QueryMutData<'a>>::Item, KernelOpMarker>
    for GpuBoundKernel<K>
where
    <K::Query as QueryMutData<'a>>::Item: Send,
{
    #[inline(always)]
    fn call_row_par(&self, item: <K::Query as QueryMutData<'a>>::Item) {
        self.kernel.call_row_with_uniforms(item, self.scalar_words);
    }
}

/// 检测形参 `param_name` 是否在着色器函数体 `body` 中被修改（赋值或传引用）。
pub fn is_param_written_in_source(param_name: &str, body: &str) -> bool {
    let bytes = body.as_bytes();
    let p_bytes = param_name.as_bytes();
    let p_len = p_bytes.len();
    let len = bytes.len();
    let mut i = 0;
    while i + p_len <= len {
        // Skip line comments
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            i += 2;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Skip block comments
        if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(len);
            continue;
        }
        // Match identifier
        if &bytes[i..i + p_len] == p_bytes {
            let prev_ok = i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_');
            let next_idx = i + p_len;
            let next_ok = next_idx == len
                || (!bytes[next_idx].is_ascii_alphanumeric() && bytes[next_idx] != b'_');
            if prev_ok && next_ok {
                let mut cursor = next_idx;
                while cursor < len && bytes[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
                while cursor < len {
                    if bytes[cursor] == b'.' {
                        cursor += 1;
                        while cursor < len
                            && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
                        {
                            cursor += 1;
                        }
                        while cursor < len && bytes[cursor].is_ascii_whitespace() {
                            cursor += 1;
                        }
                    } else if bytes[cursor] == b'[' {
                        let mut depth = 1;
                        cursor += 1;
                        while cursor < len && depth > 0 {
                            if bytes[cursor] == b'[' {
                                depth += 1;
                            } else if bytes[cursor] == b']' {
                                depth -= 1;
                            }
                            cursor += 1;
                        }
                        while cursor < len && bytes[cursor].is_ascii_whitespace() {
                            cursor += 1;
                        }
                    } else {
                        break;
                    }
                }
                if cursor < len {
                    let rest = &body[cursor..];
                    if (rest.starts_with('=') && !rest.starts_with("=="))
                        || rest.starts_with("+=")
                        || rest.starts_with("-=")
                        || rest.starts_with("*=")
                        || rest.starts_with("/=")
                        || rest.starts_with("%=")
                        || rest.starts_with("&=")
                        || rest.starts_with("|=")
                        || rest.starts_with("^=")
                        || rest.starts_with("<<=")
                        || rest.starts_with(">>=")
                    {
                        return true;
                    }
                }
                if i > 0 {
                    let mut pre = i - 1;
                    while pre > 0 && bytes[pre].is_ascii_whitespace() {
                        pre -= 1;
                    }
                    if bytes[pre] == b'&' {
                        return true;
                    }
                    // Check if param_name is passed as an argument in a function call: fn(..., param_name, ...)
                    let mut post = next_idx;
                    while post < len && bytes[post].is_ascii_whitespace() {
                        post += 1;
                    }
                    let is_post_arg = post < len && (bytes[post] == b',' || bytes[post] == b')');
                    if is_post_arg {
                        if bytes[pre] == b',' {
                            return true;
                        } else if bytes[pre] == b'(' && pre > 0 {
                            let mut fn_end = pre - 1;
                            while fn_end > 0 && bytes[fn_end].is_ascii_whitespace() {
                                fn_end -= 1;
                            }
                            let mut fn_start = fn_end;
                            while fn_start > 0
                                && (bytes[fn_start - 1].is_ascii_alphanumeric()
                                    || bytes[fn_start - 1] == b'_')
                            {
                                fn_start -= 1;
                            }
                            let fn_ident = &body[fn_start..=fn_end];
                            if !fn_ident.is_empty()
                                && fn_ident != "if"
                                && fn_ident != "while"
                                && fn_ident != "for"
                                && fn_ident != "switch"
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        i += 1;
    }
    let ptr_alias = format!("sumeru_ptr_{param_name}");
    if body.contains(&ptr_alias) {
        return true;
    }
    false
}

/// 根据结构体定义、绑定表与已翻译的 MSL 函数体生成完整的 Apple Metal Shading Language (MSL) 源码。
pub fn build_msl_kernel_source(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    msl_body: &str,
) -> String {
    build_msl_kernel_source_with_helpers(kernel_name, struct_defs, bindings, "", msl_body)
}

/// 根据结构体定义、绑定表、前置 `#[gpu]` 辅助函数与已翻译的 MSL 函数体生成完整的 MSL 源码。
pub fn build_msl_kernel_source_with_helpers(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    msl_helpers: &str,
    msl_body: &str,
) -> String {
    let mut src = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         struct GpuDispatchUniforms {\n\
         \x20   uint active_rows;\n\
         \x20   uint threads_per_group;\n\
         \x20   uint archetype_id;\n\
         \x20   uint _pad;\n\
         \x20   uint scalar_arg_0;\n\
         \x20   uint scalar_arg_1;\n\
         \x20   uint scalar_arg_2;\n\
         \x20   uint scalar_arg_3;\n\
         };\n\n",
    );
    let mut seen_structs: Vec<(&str, ComponentTypeId)> = Vec::new();
    for (idx, b) in bindings.iter().enumerate() {
        if let Some((_, existing_tid)) = seen_structs.iter().find(|(name, _)| *name == b.struct_name)
        {
            assert_eq!(
                *existing_tid, b.type_id,
                "GPU kernel `{kernel_name}` binds two distinct component types with the same shader struct name `{}`",
                b.struct_name
            );
            continue;
        }
        seen_structs.push((b.struct_name, b.type_id));
        if let Some(def) = struct_defs.get(idx) {
            src.push_str(def);
            src.push('\n');
        }
    }
    if !msl_helpers.trim().is_empty() {
        src.push_str(msl_helpers.trim());
        src.push_str("\n\n");
    }

    src.push_str(&format!("kernel void {kernel_name}(\n"));
    for (idx, b) in bindings.iter().enumerate() {
        let is_written = b.mutable && is_param_written_in_source(b.param_name, msl_body);
        let qualifier = if is_written { "device" } else { "const device" };
        src.push_str(&format!(
            "    {qualifier} {}* sumeru_buf_{} [[buffer({idx})]],\n",
            b.struct_name, b.param_name
        ));
    }
    let uniform_idx = bindings.len();
    src.push_str(&format!(
        "    constant GpuDispatchUniforms& uniforms [[buffer({uniform_idx})]],\n\
         \x20   uint gid [[thread_position_in_grid]]\n\
         ) {{\n\
         \x20   if (gid >= uniforms.active_rows) {{ return; }}\n"
    ));
    for b in bindings {
        let is_written = b.mutable && is_param_written_in_source(b.param_name, msl_body);
        if is_written {
            src.push_str(&format!(
                "    {} {} = sumeru_buf_{}[gid];\n",
                b.struct_name, b.param_name, b.param_name
            ));
        } else {
            src.push_str(&format!(
                "    const {} {} = sumeru_buf_{}[gid];\n",
                b.struct_name, b.param_name, b.param_name
            ));
        }
    }
    src.push_str("    ");
    src.push_str(msl_body.trim());
    src.push('\n');
    for b in bindings {
        let is_written = b.mutable && is_param_written_in_source(b.param_name, msl_body);
        if is_written {
            src.push_str(&format!(
                "    sumeru_buf_{}[gid] = {};\n",
                b.param_name, b.param_name
            ));
        }
    }
    src.push_str("}\n");
    src
}

/// 根据结构体定义、绑定表与已翻译的 WGSL 函数体生成完整的 WebGPU Shading Language (WGSL) 源码。
pub fn build_wgsl_kernel_source(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    wgsl_body: &str,
) -> String {
    build_wgsl_kernel_source_with_helpers(kernel_name, struct_defs, bindings, "", wgsl_body)
}

/// 根据结构体定义、绑定表、前置 `#[gpu]` 辅助函数与已翻译的 WGSL 函数体生成完整的 WGSL 源码。
pub fn build_wgsl_kernel_source_with_helpers(
    kernel_name: &str,
    struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    wgsl_helpers: &str,
    wgsl_body: &str,
) -> String {
    let mut src = String::from(
        "struct GpuDispatchUniforms {\n\
         \x20   active_rows: u32,\n\
         \x20   threads_per_group: u32,\n\
         \x20   archetype_id: u32,\n\
         \x20   _pad: u32,\n\
         \x20   scalar_arg_0: u32,\n\
         \x20   scalar_arg_1: u32,\n\
         \x20   scalar_arg_2: u32,\n\
         \x20   scalar_arg_3: u32,\n\
         };\n\n",
    );
    let mut seen_structs: Vec<(&str, ComponentTypeId)> = Vec::new();
    for (idx, b) in bindings.iter().enumerate() {
        if let Some((_, existing_tid)) = seen_structs.iter().find(|(name, _)| *name == b.struct_name)
        {
            assert_eq!(
                *existing_tid, b.type_id,
                "GPU kernel `{kernel_name}` binds two distinct component types with the same shader struct name `{}`",
                b.struct_name
            );
            continue;
        }
        seen_structs.push((b.struct_name, b.type_id));
        if let Some(def) = struct_defs.get(idx) {
            src.push_str(def);
            src.push('\n');
        }
    }
    if !wgsl_helpers.trim().is_empty() {
        src.push_str(wgsl_helpers.trim());
        src.push_str("\n\n");
    }
    for (idx, b) in bindings.iter().enumerate() {
        let is_written = b.mutable && is_param_written_in_source(b.param_name, wgsl_body);
        let access = if is_written { "read_write" } else { "read" };
        src.push_str(&format!(
            "@group(0) @binding({idx}) var<storage, {access}> sumeru_buf_{}: array<{}>;\n",
            b.param_name, b.struct_name
        ));
    }
    let uniform_idx = bindings.len();
    src.push_str(&format!(
        "@group(0) @binding({uniform_idx}) var<uniform> uniforms: GpuDispatchUniforms;\n\n\
         @compute @workgroup_size(256)\n\
         fn {kernel_name}(@builtin(global_invocation_id) global_id: vec3<u32>) {{\n\
         \x20   let gid = global_id.x;\n\
         \x20   if (gid >= uniforms.active_rows) {{ return; }}\n"
    ));
    for b in bindings {
        let is_written = b.mutable && is_param_written_in_source(b.param_name, wgsl_body);
        let kw = if is_written { "var" } else { "let" };
        src.push_str(&format!(
            "    {kw} {} = sumeru_buf_{}[gid];\n",
            b.param_name, b.param_name
        ));
    }
    src.push_str("    ");
    src.push_str(wgsl_body.trim());
    src.push('\n');
    for b in bindings {
        let is_written = b.mutable && is_param_written_in_source(b.param_name, wgsl_body);
        if is_written {
            src.push_str(&format!(
                "    sumeru_buf_{}[gid] = {};\n",
                b.param_name, b.param_name
            ));
        }
    }
    src.push_str("}\n");
    src
}

/// 声明式 GPU 组件定义宏（由 `#[gpu]` 与 `#[gpu_component]` 展开调用，也可直接使用）。
#[macro_export]
macro_rules! sumeru_gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $Name:ident {
            $(
                $(#[$fmeta:meta])*
                $fvis:vis $fname:ident : $fty:ty
            ),* $(,)?
        }
    ) => {
        #[repr(C)]
        $(#[$meta])*
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            PartialEq,
            $crate::__private::bytemuck::Pod,
            $crate::__private::bytemuck::Zeroable,
            $crate::__private::serde::Serialize,
            $crate::__private::serde::Deserialize,
            $crate::__private::rkyv::Archive,
            $crate::__private::rkyv::Serialize,
            $crate::__private::rkyv::Deserialize,
        )]
        $vis struct $Name {
            $(
                $(#[$fmeta])*
                $fvis $fname : $fty,
            )*
        }

        // SAFETY: #[repr(C)] + bytemuck::Pod + Zeroable guaranteed above.
        unsafe impl $crate::GpuComponent for $Name {}

        impl $crate::GpuShaderStruct for $Name {
            const STRUCT_NAME: &'static str = stringify!($Name);
            const FIELDS: &'static [$crate::GpuStructField] = &[
                $(
                    $crate::GpuStructField {
                        name: stringify!($fname),
                        field_type: <$fty as $crate::GpuScalarType>::FIELD_TYPE,
                    },
                )*
            ];
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_param_type {
    ($lt:lifetime, [mut] $T:ty) => {
        &$lt mut $T
    };
    ($lt:lifetime, [ref] $T:ty) => {
        &$lt $T
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_elided_param_type {
    ([mut] $T:ty) => {
        &mut $T
    };
    ([ref] $T:ty) => {
        &$T
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_is_mut {
    ([mut]) => {
        true
    };
    ([ref]) => {
        false
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_slice_from_raw {
    ([mut] $T:ty, $ptr:expr, $len:expr) => {
        ::std::slice::from_raw_parts_mut($ptr as *mut $T, $len)
    };
    ([ref] $T:ty, $ptr:expr, $len:expr) => {
        ::std::slice::from_raw_parts($ptr as *const $T, $len)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_elem_borrow {
    ([mut] $slice:ident, $idx:ident) => {
        &mut $slice[$idx]
    };
    ([ref] $slice:ident, $idx:ident) => {
        &$slice[$idx]
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_query_type {
    ($lt:lifetime, [$m0:ident] $T0:ty) => {
        $crate::__sumeru_gpu_param_type!($lt, [$m0] $T0)
    };
    ($lt:lifetime, [$m0:ident] $T0:ty, $( [$m:ident] $T:ty ),+) => {
        (
            $crate::__sumeru_gpu_param_type!($lt, [$m0] $T0),
            $( $crate::__sumeru_gpu_param_type!($lt, [$m] $T) ),+
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_row_pat {
    ($p0:ident) => {
        $p0
    };
    ($p0:ident, $( $p:ident ),+) => {
        ( $p0, $( $p ),+ )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __sumeru_gpu_maybe_mut_ident {
    ([mut] $id:ident) => {
        mut $id
    };
    ([ref] $id:ident) => {
        $id
    };
}

/// 统一 `#[gpu]` 函数展开宏：生成同名可调用算子结构体与静态描述符，支持 1~8 个组件参数及 0~4 个动态标量参数。
#[macro_export]
macro_rules! sumeru_gpu_fn {
    // Typed Shader IR arm with 1..=8 component params and 0 dynamic scalar uniforms
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $( $p:ident : [$m:ident] $T:ty ),+ $(,)?
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            #[allow(unused_mut, clippy::too_many_arguments, clippy::no_effect)]
            pub fn call( $( $p: $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+ ) {
                $($body)*
            }

            #[allow(non_snake_case)]
            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let mut __ptr_idx = 0usize;
                $(
                    let $p = unsafe {
                        let ptr = raw_ptrs[__ptr_idx];
                        __ptr_idx += 1;
                        $crate::__sumeru_gpu_slice_from_raw!([$m] $T, ptr, active_rows)
                    };
                )+
                let _ = __ptr_idx;
                for __row_idx in 0..active_rows {
                    Self::call(
                        $( $crate::__sumeru_gpu_elem_borrow!([$m] $p, __row_idx) ),+
                    );
                }
            }

            #[inline]
            #[allow(dead_code)]
            pub fn gpu_helper_descriptor() -> &'static $crate::GpuHelperFnDescriptor {
                static HELPER_DESC: ::std::sync::OnceLock<$crate::GpuHelperFnDescriptor> = ::std::sync::OnceLock::new();
                HELPER_DESC.get_or_init(|| {
                    $crate::GpuHelperFnDescriptor {
                        name: stringify!($fn_name),
                        params: ::std::vec![
                            $(
                                $crate::GpuHelperParamSpec {
                                    name: stringify!($p),
                                    is_mut: $crate::__sumeru_gpu_is_mut!([$m]),
                                    by_mut_ref: $crate::__sumeru_gpu_is_mut!([$m]),
                                    kind: $crate::GpuHelperParamKind::Struct {
                                        struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                        type_id: $crate::ComponentTypeId::of::<$T>(),
                                        fields: <$T as $crate::GpuShaderStruct>::FIELDS,
                                        msl_def: <$T as $crate::GpuShaderStruct>::msl_struct_definition,
                                        wgsl_def: <$T as $crate::GpuShaderStruct>::wgsl_struct_definition,
                                    },
                                }
                            ),+
                        ],
                        return_type: ::std::option::Option::None,
                        body: $ir_block,
                    }
                })
            }

            /// 将整列查询视图绑定为可组合的延迟算子派发描述符 [`KernelLaunch`]
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn bind<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+
            ) -> $crate::KernelLaunch<'__store> {
                $crate::KernelLaunch::from_bindings(
                    <Self as $crate::GpuKernel<'static>>::descriptor(self),
                    [0u32; 4],
                    &[ $( $p.into_kernel_col_binding() ),+ ],
                )
            }

            /// 自动最优整列派发（存在硬件 GPU 时自动使用 16KB 页对齐零拷贝 GPU 加速，无硬件 GPU 时自适应切换至多核并行 `.par`）
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn run<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+).run()
            }

            /// 自适应 CPU 并行整列派发（< 1024 行在当前线程顺序执行，>= 1024 行按 1024 行 Morsel 在 Rayon 线程池并行执行）
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn par<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+).par()
            }

            /// 显式 GPU 硬件 16KB 页对齐零拷贝整列派发（使用线程本地默认计算引擎）
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn gpu<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+).gpu()
            }

            /// 在指定的 [`MetalComputeEngine`] 上执行 GPU 硬件 16KB 页对齐零拷贝整列派发
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn gpu_on<'__store>(
                &self,
                engine: &mut $crate::MetalComputeEngine,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+).gpu_on(engine)
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = fn( $( $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+ );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: fn( $( $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+ ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            #[allow(non_snake_case)]
            fn call_row(
                &mut self,
                (_entity, $crate::__sumeru_gpu_row_pat!( $( $p ),+ )): (
                    $crate::EntityId,
                    $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
                ),
            ) {
                Self::call( $( $p ),+ );
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            #[allow(non_snake_case)]
            fn call_row_par(
                &self,
                (_entity, $crate::__sumeru_gpu_row_pat!( $( $p ),+ )): (
                    $crate::EntityId,
                    $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
                ),
            ) {
                Self::call( $( $p ),+ );
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+);

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    $( <$T as $crate::GpuShaderStruct>::assert_shader_layout(); )+
                    let bindings = ::std::vec![
                        $(
                            $crate::GpuKernelBindingSpec {
                                param_name: stringify!($p),
                                struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                type_id: $crate::ComponentTypeId::of::<$T>(),
                                elem_size: ::std::mem::size_of::<$T>(),
                                mutable: $crate::__sumeru_gpu_is_mut!([$m]),
                            }
                        ),+
                    ];
                    let components = [
                        $(
                            $crate::GpuComponentSchema {
                                param_name: stringify!($p),
                                struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                fields: <$T as $crate::GpuShaderStruct>::FIELDS,
                                mutable: $crate::__sumeru_gpu_is_mut!([$m]),
                            }
                        ),+
                    ];
                    let msl_defs = ::std::vec![
                        $( <$T as $crate::GpuShaderStruct>::msl_struct_definition() ),+
                    ];
                    let wgsl_defs = ::std::vec![
                        $( <$T as $crate::GpuShaderStruct>::wgsl_struct_definition() ),+
                    ];
                    let ir: $crate::GpuBlock = $ir_block;
                    let (msl_source, wgsl_source) = $crate::compile_kernel_ir_sources(
                        stringify!($fn_name),
                        &components,
                        &msl_defs,
                        &wgsl_defs,
                        &bindings,
                        &ir,
                    );
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // Typed Shader IR arm with 1..=8 component params AND 1..=4 dynamic scalar uniforms
    (
        @ir_with_uniforms ($num_uniforms:literal)
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            components = [ $( $p:ident : [$m:ident] $T:ty ),+ ],
            uniforms = [ $( ($u_slot:literal) $u_ident:ident : [$u_mut:ident] $u_ty:ty => $u_kind:expr ),+ ]
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            #[allow(unused_mut, clippy::too_many_arguments, clippy::no_effect)]
            pub fn call(
                $( $p: $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+,
                $( $crate::__sumeru_gpu_maybe_mut_ident!([$u_mut] $u_ident) : $u_ty ),+
            ) {
                $($body)*
            }

            /// 绑定运行时动态标量参数（单参数可直接传 `kernel.with_args(dt)`，多参数传元组 `kernel.with_args((dt, limit))`）。
            #[inline(always)]
            #[allow(dead_code)]
            pub fn with_args<__Args: $crate::IntoGpuScalarUniforms<$num_uniforms>>(
                self,
                args: __Args,
            ) -> $crate::GpuBoundKernel<Self> {
                $crate::GpuBoundKernel {
                    kernel: self,
                    scalar_words: args.into_scalar_words(),
                }
            }

            #[allow(non_snake_case)]
            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let mut __ptr_idx = 0usize;
                $(
                    let $p = unsafe {
                        let ptr = raw_ptrs[__ptr_idx];
                        __ptr_idx += 1;
                        $crate::__sumeru_gpu_slice_from_raw!([$m] $T, ptr, active_rows)
                    };
                )+
                let _ = __ptr_idx;
                $(
                    let $u_ident: $u_ty = <$u_ty as $crate::GpuScalarArg>::from_uniform_word(
                        uniforms.scalar_args[$u_slot]
                    );
                )+
                for __row_idx in 0..active_rows {
                    Self::call(
                        $( $crate::__sumeru_gpu_elem_borrow!([$m] $p, __row_idx) ),+,
                        $( $u_ident ),+
                    );
                }
            }

            /// 获取底层静态编译描述符。
            #[inline]
            pub fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    $( <$T as $crate::GpuShaderStruct>::assert_shader_layout(); )+
                    let bindings = ::std::vec![
                        $(
                            $crate::GpuKernelBindingSpec {
                                param_name: stringify!($p),
                                struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                type_id: $crate::ComponentTypeId::of::<$T>(),
                                elem_size: ::std::mem::size_of::<$T>(),
                                mutable: $crate::__sumeru_gpu_is_mut!([$m]),
                            }
                        ),+
                    ];
                    let components = [
                        $(
                            $crate::GpuComponentSchema {
                                param_name: stringify!($p),
                                struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                fields: <$T as $crate::GpuShaderStruct>::FIELDS,
                                mutable: $crate::__sumeru_gpu_is_mut!([$m]),
                            }
                        ),+
                    ];
                    let scalar_uniforms = [
                        $(
                            $crate::GpuScalarUniformSpec {
                                param_name: stringify!($u_ident),
                                mutable: $crate::__sumeru_gpu_is_mut!([$u_mut]),
                                kind: $u_kind,
                                slot: $u_slot,
                            }
                        ),+
                    ];
                    let msl_defs = ::std::vec![
                        $( <$T as $crate::GpuShaderStruct>::msl_struct_definition() ),+
                    ];
                    let wgsl_defs = ::std::vec![
                        $( <$T as $crate::GpuShaderStruct>::wgsl_struct_definition() ),+
                    ];
                    let ir: $crate::GpuBlock = $ir_block;
                    let (msl_source, wgsl_source) = $crate::compile_kernel_ir_sources_with_uniforms(
                        stringify!($fn_name),
                        &components,
                        &scalar_uniforms,
                        &msl_defs,
                        &wgsl_defs,
                        &bindings,
                        &ir,
                    );
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }

            #[inline]
            #[allow(dead_code)]
            pub fn gpu_helper_descriptor() -> &'static $crate::GpuHelperFnDescriptor {
                static HELPER_DESC: ::std::sync::OnceLock<$crate::GpuHelperFnDescriptor> = ::std::sync::OnceLock::new();
                HELPER_DESC.get_or_init(|| {
                    $crate::GpuHelperFnDescriptor {
                        name: stringify!($fn_name),
                        params: ::std::vec![
                            $(
                                $crate::GpuHelperParamSpec {
                                    name: stringify!($p),
                                    is_mut: $crate::__sumeru_gpu_is_mut!([$m]),
                                    by_mut_ref: $crate::__sumeru_gpu_is_mut!([$m]),
                                    kind: $crate::GpuHelperParamKind::Struct {
                                        struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                        type_id: $crate::ComponentTypeId::of::<$T>(),
                                        fields: <$T as $crate::GpuShaderStruct>::FIELDS,
                                        msl_def: <$T as $crate::GpuShaderStruct>::msl_struct_definition,
                                        wgsl_def: <$T as $crate::GpuShaderStruct>::wgsl_struct_definition,
                                    },
                                }
                            ),+,
                            $(
                                $crate::GpuHelperParamSpec {
                                    name: stringify!($u_ident),
                                    is_mut: $crate::__sumeru_gpu_is_mut!([$u_mut]),
                                    by_mut_ref: false,
                                    kind: $crate::GpuHelperParamKind::Scalar($u_kind),
                                }
                            ),+
                        ],
                        return_type: ::std::option::Option::None,
                        body: $ir_block,
                    }
                })
            }

            /// 将整列查询视图与动态标量参数绑定为可组合的延迟算子派发描述符 [`KernelLaunch`]
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn bind<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+,
                $( $u_ident: $u_ty ),+
            ) -> $crate::KernelLaunch<'__store> {
                let mut scalar_words = [0u32; 4];
                $(
                    scalar_words[$u_slot] = <$u_ty as $crate::GpuScalarArg>::to_uniform_word($u_ident);
                )+
                $crate::KernelLaunch::from_bindings(
                    Self::descriptor(self),
                    scalar_words,
                    &[ $( $p.into_kernel_col_binding() ),+ ],
                )
            }

            /// 自动最优整列派发（存在硬件 GPU 时自动使用 16KB 页对齐零拷贝 GPU 加速，无硬件 GPU 时自适应切换至多核并行 `.par`）
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn run<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+,
                $( $u_ident: $u_ty ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+, $( $u_ident ),+).run()
            }

            /// 自适应 CPU 并行整列派发（< 1024 行在当前线程顺序执行，>= 1024 行按 1024 行 Morsel 在 Rayon 线程池并行执行）
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn par<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+,
                $( $u_ident: $u_ty ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+, $( $u_ident ),+).par()
            }

            /// 显式 GPU 硬件 16KB 页对齐零拷贝整列派发（使用线程本地默认计算引擎）
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn gpu<'__store>(
                &self,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+,
                $( $u_ident: $u_ty ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+, $( $u_ident ),+).gpu()
            }

            /// 在指定的 [`MetalComputeEngine`] 上执行 GPU 硬件 16KB 页对齐零拷贝整列派发
            #[inline]
            #[allow(dead_code, clippy::too_many_arguments)]
            pub fn gpu_on<'__store>(
                &self,
                engine: &mut $crate::MetalComputeEngine,
                $( $p: impl $crate::KernelColumnArg<'__store, $crate::__sumeru_gpu_param_type!('static, [$m] $T)> ),+,
                $( $u_ident: $u_ty ),+
            ) -> ::std::result::Result<$crate::GpuDispatchStats, $crate::GpuExecutionError> {
                self.bind($( $p ),+, $( $u_ident ),+).gpu_on(engine)
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = fn(
                $( $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+,
                $( $u_ty ),+
            );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: fn(
                    $( $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+,
                    $( $u_ty ),+
                ) = $fn_name::call;
                &FN_PTR
            }
        }

        unsafe impl<'a> $crate::GpuParameterizedKernel<'a> for $fn_name {
            type Query = $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+);

            #[inline(always)]
            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                Self::descriptor(self)
            }

            #[inline(always)]
            #[allow(non_snake_case)]
            fn call_row_with_uniforms(
                &self,
                (_entity, $crate::__sumeru_gpu_row_pat!( $( $p ),+ )): (
                    $crate::EntityId,
                    $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
                ),
                scalar_words: [u32; 4],
            ) {
                $(
                    let $u_ident: $u_ty = <$u_ty as $crate::GpuScalarArg>::from_uniform_word(
                        scalar_words[$u_slot]
                    );
                )+
                Self::call( $( $p ),+, $( $u_ident ),+ );
            }
        }
    };

    // Legacy @translated arm for direct macro_rules! callers (1..=8 params)
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $( $p:ident : [$m:ident] $T:ty ),+ $(,)?
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            #[allow(unused_mut, clippy::too_many_arguments, clippy::no_effect)]
            pub fn call( $( $p: $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+ ) {
                $($body)*
            }

            #[allow(non_snake_case)]
            unsafe fn __cpu_chunk_runner(
                raw_ptrs: &[*mut u8],
                active_rows: usize,
                _uniforms: &$crate::GpuDispatchUniforms,
            ) {
                let mut __ptr_idx = 0usize;
                $(
                    let $p = unsafe {
                        let ptr = raw_ptrs[__ptr_idx];
                        __ptr_idx += 1;
                        $crate::__sumeru_gpu_slice_from_raw!([$m] $T, ptr, active_rows)
                    };
                )+
                let _ = __ptr_idx;
                for __row_idx in 0..active_rows {
                    Self::call(
                        $( $crate::__sumeru_gpu_elem_borrow!([$m] $p, __row_idx) ),+
                    );
                }
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = fn( $( $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+ );

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: fn( $( $crate::__sumeru_gpu_elided_param_type!([$m] $T) ),+ ) = $fn_name::call;
                &FN_PTR
            }
        }

        impl<'a> $crate::QueryForEachOp<(
            $crate::EntityId,
            $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            #[allow(non_snake_case)]
            fn call_row(
                &mut self,
                (_entity, $crate::__sumeru_gpu_row_pat!( $( $p ),+ )): (
                    $crate::EntityId,
                    $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
                ),
            ) {
                Self::call( $( $p ),+ );
            }
        }

        impl<'a> $crate::QueryParForEachOp<(
            $crate::EntityId,
            $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
        ), $crate::KernelOpMarker> for $fn_name {
            #[inline(always)]
            #[allow(non_snake_case)]
            fn call_row_par(
                &self,
                (_entity, $crate::__sumeru_gpu_row_pat!( $( $p ),+ )): (
                    $crate::EntityId,
                    $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+)
                ),
            ) {
                Self::call( $( $p ),+ );
            }
        }

        unsafe impl<'a> $crate::GpuKernel<'a> for $fn_name {
            type Query = $crate::__sumeru_gpu_query_type!('a, $( [$m] $T ),+);

            fn descriptor(&self) -> &'static $crate::GpuKernelDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuKernelDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    $( <$T as $crate::GpuShaderStruct>::assert_shader_layout(); )+
                    let bindings = ::std::vec![
                        $(
                            $crate::GpuKernelBindingSpec {
                                param_name: stringify!($p),
                                struct_name: <$T as $crate::GpuShaderStruct>::STRUCT_NAME,
                                type_id: $crate::ComponentTypeId::of::<$T>(),
                                elem_size: ::std::mem::size_of::<$T>(),
                                mutable: $crate::__sumeru_gpu_is_mut!([$m]),
                            }
                        ),+
                    ];
                    let msl_defs = ::std::vec![
                        $( <$T as $crate::GpuShaderStruct>::msl_struct_definition() ),+
                    ];
                    let wgsl_defs = ::std::vec![
                        $( <$T as $crate::GpuShaderStruct>::wgsl_struct_definition() ),+
                    ];
                    let msl_source = $crate::build_msl_kernel_source(stringify!($fn_name), &msl_defs, &bindings, $msl_body);
                    let wgsl_source = $crate::build_wgsl_kernel_source(stringify!($fn_name), &wgsl_defs, &bindings, $wgsl_body);
                    unsafe {
                        $crate::GpuKernelDescriptor::new(
                            stringify!($fn_name),
                            bindings,
                            msl_source,
                            wgsl_source,
                            Self::__cpu_chunk_runner,
                        )
                    }
                })
            }
        }
    };

    // Fallback @normalized arm
    (
        @normalized
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident ( $($params:tt)* ) { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @translated
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            msl = stringify!($($body)*),
            wgsl = stringify!($($body)*),
            body = { $($body)* }
        }
    };
}

/// 显式内核类型名展开宏（供 `#[gpu_kernel(CustomKernelName)]` 使用）。
#[macro_export]
macro_rules! sumeru_gpu_kernel {
    (
        @ir
        $(#[$meta:meta])*
        $vis:vis kernel $KernelType:ident fn $fn_name:ident (
            $($params:tt)*
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @ir
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            ir = $ir_block,
            body = { $($body)* }
        }
        #[allow(dead_code)]
        $vis type $KernelType = $fn_name;
    };
    (
        @ir_with_uniforms ($num_uniforms:literal)
        $(#[$meta:meta])*
        $vis:vis kernel $KernelType:ident fn $fn_name:ident (
            $($params:tt)*
        )
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @ir_with_uniforms ($num_uniforms)
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            ir = $ir_block,
            body = { $($body)* }
        }
        #[allow(dead_code)]
        $vis type $KernelType = $fn_name;
    };
    (
        @translated
        $(#[$meta:meta])*
        $vis:vis kernel $KernelType:ident fn $fn_name:ident (
            $($params:tt)*
        )
        msl = $msl_body:expr,
        wgsl = $wgsl_body:expr,
        body = { $($body:tt)* }
    ) => {
        $crate::sumeru_gpu_fn! {
            @translated
            $(#[$meta])*
            $vis fn $fn_name ( $($params)* )
            msl = $msl_body,
            wgsl = $wgsl_body,
            body = { $($body)* }
        }
        #[allow(dead_code)]
        $vis type $KernelType = $fn_name;
    };
}

/// 可组合 `#[gpu]` 辅助函数（Device Helper Function，支持标量、数组、`&Comp` / `&mut Comp` 参数及可选返回值）展开宏。
#[macro_export]
macro_rules! sumeru_gpu_helper_fn {
    (
        $(#[$meta:meta])*
        $vis:vis fn $fn_name:ident (
            $( $p_ident:ident : [$p_mut_binding:ident] $p_ty:ty => $p_spec:expr ),* $(,)?
        ) -> $ret_ty:ty => $ret_kind_opt:expr,
        ir = $ir_block:expr,
        body = { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Default)]
        $vis struct $fn_name;

        impl $fn_name {
            #[inline(always)]
            #[allow(unused_mut, clippy::too_many_arguments, clippy::no_effect)]
            pub fn call(
                $( $crate::__sumeru_gpu_maybe_mut_ident!([$p_mut_binding] $p_ident) : $p_ty ),*
            ) -> $ret_ty {
                $($body)*
            }

            #[inline]
            #[allow(dead_code)]
            pub fn gpu_helper_descriptor() -> &'static $crate::GpuHelperFnDescriptor {
                static DESC: ::std::sync::OnceLock<$crate::GpuHelperFnDescriptor> = ::std::sync::OnceLock::new();
                DESC.get_or_init(|| {
                    $crate::GpuHelperFnDescriptor {
                        name: stringify!($fn_name),
                        params: ::std::vec![
                            $( $p_spec ),*
                        ],
                        return_type: $ret_kind_opt,
                        body: $ir_block,
                    }
                })
            }
        }

        impl ::std::ops::Deref for $fn_name {
            type Target = fn( $( $p_ty ),* ) -> $ret_ty;

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                const FN_PTR: fn( $( $p_ty ),* ) -> $ret_ty = $fn_name::call;
                &FN_PTR
            }
        }
    };
}
