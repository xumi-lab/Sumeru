//! 强类型着色器中间表示 (Typed Shader IR)、双向符号表类型推导与多后端代码生成器 (`GpuIrCompiler`)
//!
//! 本模块提供目标无关的着色器 AST/IR 树（[`GpuBlock`]、[`GpuStmt`]、[`GpuExpr`]）：
//! 1. **零字符串模板宏展开**：`crates/sumeru-macros` 仅负责将 Rust `syn` AST 展开为本模块的强类型 IR 构造调用；
//! 2. **借力 `rustc` 编译期常量求值 ([`GpuConstValue`])**：核函数内引用的外部 `const` 常量（如 `DT`、`std::f32::consts::PI`）
//!    直接由 Rust 编译器求值并内联为强类型字面量 [`GpuLit`]；
//! 3. **可组合 `#[gpu]` 辅助函数 ([`GpuHelperFnDescriptor`])**：支持标量、定长向量数组 (`[T; 2..=4]`) 与组件结构体引用 (`&Comp` / `&mut Comp`) 参数，
//!    支持标量返回值或无返回值 (`()`) 子算子调用，自动按调用依赖拓扑序链接至目标 WGSL 与 MSL 模块；
//! 4. **全表达式双向符号表类型推导 ([`ShaderTypeEnv`])**：结合 [`GpuShaderStruct::FIELDS`] 元数据自动推导组件字段、
//!    数组下标、局部变量、左值赋值、二元算术/比较/位运算、函数实参与 `for i in 0..cfg.steps` 循环边界的有符号/无符号类型；
//! 5. **运行时动态标量 Uniform 参数 ([`GpuScalarUniformSpec`])**：支持算子尾部直接声明 `dt: f32` 等按值标量参数，通过 `GpuDispatchUniforms::scalar_args` 零分配传入。

use super::kernel::{GpuFieldType, GpuKernelBindingSpec, GpuStructField};
use crate::id::ComponentTypeId;
use std::collections::HashMap;

/// 着色器标量基础类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuScalarKind {
    F32,
    I32,
    U32,
    Bool,
}

impl GpuScalarKind {
    /// 映射为 Apple Metal Shading Language (MSL) 标量类型名。
    #[inline]
    pub const fn msl_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::Bool => "bool",
        }
    }

    /// 映射为 WebGPU Shading Language (WGSL) 标量类型名。
    #[inline]
    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Bool => "bool",
        }
    }
}

/// 可作为 `#[gpu]` 算子运行时动态标量参数（Uniform Scalar Argument）的类型契约。
pub trait GpuScalarArg: Copy + Send + Sync + 'static {
    /// 对应的着色器标量类型。
    const SCALAR_KIND: GpuScalarKind;

    /// 将当前标量值编码为 `GpuDispatchUniforms::scalar_args` 的 32-bit 位字。
    fn to_uniform_word(self) -> u32;

    /// 从 `GpuDispatchUniforms::scalar_args` 的 32-bit 位字还原标量值。
    fn from_uniform_word(word: u32) -> Self;
}

impl GpuScalarArg for f32 {
    const SCALAR_KIND: GpuScalarKind = GpuScalarKind::F32;
    #[inline(always)]
    fn to_uniform_word(self) -> u32 {
        self.to_bits()
    }
    #[inline(always)]
    fn from_uniform_word(word: u32) -> Self {
        f32::from_bits(word)
    }
}

impl GpuScalarArg for i32 {
    const SCALAR_KIND: GpuScalarKind = GpuScalarKind::I32;
    #[inline(always)]
    fn to_uniform_word(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn from_uniform_word(word: u32) -> Self {
        word as i32
    }
}

impl GpuScalarArg for u32 {
    const SCALAR_KIND: GpuScalarKind = GpuScalarKind::U32;
    #[inline(always)]
    fn to_uniform_word(self) -> u32 {
        self
    }
    #[inline(always)]
    fn from_uniform_word(word: u32) -> Self {
        word
    }
}

impl GpuScalarArg for usize {
    const SCALAR_KIND: GpuScalarKind = GpuScalarKind::U32;
    #[inline(always)]
    fn to_uniform_word(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn from_uniform_word(word: u32) -> Self {
        word as usize
    }
}

impl GpuScalarArg for bool {
    const SCALAR_KIND: GpuScalarKind = GpuScalarKind::Bool;
    #[inline(always)]
    fn to_uniform_word(self) -> u32 {
        u32::from(self)
    }
    #[inline(always)]
    fn from_uniform_word(word: u32) -> Self {
        word != 0
    }
}

/// 内核尾部按值传入的运行时动态标量 Uniform 参数规格（映射至 `uniforms.scalar_arg_{slot}`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuScalarUniformSpec {
    pub param_name: &'static str,
    pub mutable: bool,
    pub kind: GpuScalarKind,
    pub slot: usize,
}

/// 符号表推导出的表达式值类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuValueType {
    Scalar(GpuScalarKind),
    Vector(GpuScalarKind, usize),
    Array(GpuScalarKind, usize),
    Struct(&'static str),
    Unknown,
}

impl GpuValueType {
    /// 从组件结构体字段类型 [`GpuFieldType`] 转换为 [`GpuValueType`]。
    #[inline]
    pub const fn from_field_type(ft: GpuFieldType) -> Self {
        match ft {
            GpuFieldType::F32 => Self::Scalar(GpuScalarKind::F32),
            GpuFieldType::I32 => Self::Scalar(GpuScalarKind::I32),
            GpuFieldType::U32 => Self::Scalar(GpuScalarKind::U32),
            GpuFieldType::Vec2F32 => Self::Array(GpuScalarKind::F32, 2),
            GpuFieldType::Vec3F32 => Self::Array(GpuScalarKind::F32, 3),
            GpuFieldType::Vec4F32 => Self::Array(GpuScalarKind::F32, 4),
            GpuFieldType::Vec2U32 => Self::Array(GpuScalarKind::U32, 2),
            GpuFieldType::Vec3U32 => Self::Array(GpuScalarKind::U32, 3),
            GpuFieldType::Vec4U32 => Self::Array(GpuScalarKind::U32, 4),
            GpuFieldType::Vec2I32 => Self::Array(GpuScalarKind::I32, 2),
            GpuFieldType::Vec3I32 => Self::Array(GpuScalarKind::I32, 3),
            GpuFieldType::Vec4I32 => Self::Array(GpuScalarKind::I32, 4),
        }
    }

    /// 映射为 Apple Metal Shading Language (MSL) 类型名。
    pub fn msl_type_name(self) -> Option<&'static str> {
        match self {
            Self::Scalar(k) => Some(k.msl_type()),
            Self::Vector(GpuScalarKind::F32, 2) => Some("float2"),
            Self::Vector(GpuScalarKind::F32, 3) => Some("float3"),
            Self::Vector(GpuScalarKind::F32, 4) => Some("float4"),
            Self::Vector(GpuScalarKind::I32, 2) => Some("int2"),
            Self::Vector(GpuScalarKind::I32, 3) => Some("int3"),
            Self::Vector(GpuScalarKind::I32, 4) => Some("int4"),
            Self::Vector(GpuScalarKind::U32, 2) => Some("uint2"),
            Self::Vector(GpuScalarKind::U32, 3) => Some("uint3"),
            Self::Vector(GpuScalarKind::U32, 4) => Some("uint4"),
            Self::Vector(GpuScalarKind::Bool, 2) => Some("bool2"),
            Self::Vector(GpuScalarKind::Bool, 3) => Some("bool3"),
            Self::Vector(GpuScalarKind::Bool, 4) => Some("bool4"),
            Self::Array(GpuScalarKind::F32, 2) => Some("packed_float2"),
            Self::Array(GpuScalarKind::F32, 3) => Some("packed_float3"),
            Self::Array(GpuScalarKind::F32, 4) => Some("packed_float4"),
            Self::Array(GpuScalarKind::I32, 2) => Some("packed_int2"),
            Self::Array(GpuScalarKind::I32, 3) => Some("packed_int3"),
            Self::Array(GpuScalarKind::I32, 4) => Some("packed_int4"),
            Self::Array(GpuScalarKind::U32, 2) => Some("packed_uint2"),
            Self::Array(GpuScalarKind::U32, 3) => Some("packed_uint3"),
            Self::Array(GpuScalarKind::U32, 4) => Some("packed_uint4"),
            Self::Struct(name) => Some(name),
            _ => None,
        }
    }

    /// 映射为 WebGPU Shading Language (WGSL) 类型名。
    pub fn wgsl_type_name(self) -> Option<&'static str> {
        match self {
            Self::Scalar(k) => Some(k.wgsl_type()),
            Self::Vector(GpuScalarKind::F32, 2) => Some("vec2<f32>"),
            Self::Vector(GpuScalarKind::F32, 3) => Some("vec3<f32>"),
            Self::Vector(GpuScalarKind::F32, 4) => Some("vec4<f32>"),
            Self::Vector(GpuScalarKind::I32, 2) => Some("vec2<i32>"),
            Self::Vector(GpuScalarKind::I32, 3) => Some("vec3<i32>"),
            Self::Vector(GpuScalarKind::I32, 4) => Some("vec4<i32>"),
            Self::Vector(GpuScalarKind::U32, 2) => Some("vec2<u32>"),
            Self::Vector(GpuScalarKind::U32, 3) => Some("vec3<u32>"),
            Self::Vector(GpuScalarKind::U32, 4) => Some("vec4<u32>"),
            Self::Vector(GpuScalarKind::Bool, 2) => Some("vec2<bool>"),
            Self::Vector(GpuScalarKind::Bool, 3) => Some("vec3<bool>"),
            Self::Vector(GpuScalarKind::Bool, 4) => Some("vec4<bool>"),
            Self::Array(GpuScalarKind::F32, 2) => Some("array<f32, 2>"),
            Self::Array(GpuScalarKind::F32, 3) => Some("array<f32, 3>"),
            Self::Array(GpuScalarKind::F32, 4) => Some("array<f32, 4>"),
            Self::Array(GpuScalarKind::I32, 2) => Some("array<i32, 2>"),
            Self::Array(GpuScalarKind::I32, 3) => Some("array<i32, 3>"),
            Self::Array(GpuScalarKind::I32, 4) => Some("array<i32, 4>"),
            Self::Array(GpuScalarKind::U32, 2) => Some("array<u32, 2>"),
            Self::Array(GpuScalarKind::U32, 3) => Some("array<u32, 3>"),
            Self::Array(GpuScalarKind::U32, 4) => Some("array<u32, 4>"),
            Self::Struct(name) => Some(name),
            _ => None,
        }
    }

    #[inline]
    const fn as_vector_or_array(self) -> Option<(GpuScalarKind, usize)> {
        match self {
            Self::Vector(k, len) | Self::Array(k, len) => Some((k, len)),
            _ => None,
        }
    }
}

#[inline]
fn swizzle_component_index(field: &str, len: usize) -> Option<usize> {
    match field {
        "x" | "r" if len >= 1 => Some(0),
        "y" | "g" if len >= 2 => Some(1),
        "z" | "b" if len >= 3 => Some(2),
        "w" | "a" if len >= 4 => Some(3),
        _ => None,
    }
}

/// 强类型着色器字面量节点。
#[derive(Debug, Clone, PartialEq)]
pub enum GpuLit {
    /// 源码内联浮点字面量（保留原始十进制文本，如 `"100.0"`、`"1e-5"`）
    FloatRaw(&'static str),
    /// 由外部 Rust `const` 常量求值得到的 `f32` 字面量
    F32(f32),
    /// 源码内未带类型后缀的整数字面量（在 `U32` 上下文中可自动提升为 `u` 后缀）
    UntypedInt(i64),
    /// 显式有符号 32 位整数字面量 (`i32`)
    I32(i32),
    /// 显式无符号 32 位整数字面量 (`u32` / `usize`)
    U32(u64),
    /// 布尔字面量 (`bool`)
    Bool(bool),
}

impl GpuLit {
    /// 格式化为着色器源码字面量文本。
    pub fn format_shader(&self, coerce_unsigned: bool) -> String {
        match self {
            Self::FloatRaw(s) => (*s).to_string(),
            Self::F32(v) => {
                assert!(
                    v.is_finite(),
                    "Non-finite f32 constant `{v}` cannot be embedded into a GPU shader"
                );
                let s = format!("{v:?}");
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    s
                } else {
                    format!("{s}.0")
                }
            }
            Self::UntypedInt(n) => {
                if coerce_unsigned && *n >= 0 {
                    format!("{n}u")
                } else {
                    format!("{n}")
                }
            }
            Self::I32(n) => format!("{n}"),
            Self::U32(n) => format!("{n}u"),
            Self::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        }
    }
}

/// 将外部 Rust `const` 常量桥接为着色器 IR 表达式 [`GpuExpr`] 的编译期转换契约。
pub trait GpuConstValue {
    /// 将当前 Rust 常量值转换为强类型着色器字面量表达式。
    fn to_gpu_expr(&self) -> GpuExpr;
}

impl GpuConstValue for f32 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::F32(*self))
    }
}

impl GpuConstValue for i32 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::I32(*self))
    }
}

impl GpuConstValue for u32 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::U32(*self as u64))
    }
}

impl GpuConstValue for usize {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::U32(*self as u64))
    }
}

impl GpuConstValue for bool {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::Lit(GpuLit::Bool(*self))
    }
}

/// 着色器二元运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
}

impl GpuBinaryOp {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Rem => "%",
            Self::And => "&&",
            Self::Or => "||",
            Self::BitXor => "^",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Eq => "==",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Ne => "!=",
            Self::Ge => ">=",
            Self::Gt => ">",
        }
    }

    #[inline]
    pub const fn is_comparison_or_logical(self) -> bool {
        matches!(
            self,
            Self::And
                | Self::Or
                | Self::Eq
                | Self::Lt
                | Self::Le
                | Self::Ne
                | Self::Ge
                | Self::Gt
        )
    }
}

/// 着色器赋值与复合赋值运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitXorAssign,
    BitAndAssign,
    BitOrAssign,
    ShlAssign,
    ShrAssign,
}

impl GpuAssignOp {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assign => "=",
            Self::AddAssign => "+=",
            Self::SubAssign => "-=",
            Self::MulAssign => "*=",
            Self::DivAssign => "/=",
            Self::RemAssign => "%=",
            Self::BitXorAssign => "^=",
            Self::BitAndAssign => "&=",
            Self::BitOrAssign => "|=",
            Self::ShlAssign => "<<=",
            Self::ShrAssign => ">>=",
        }
    }
}

/// 着色器一元运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuUnaryOp {
    Neg,
    Not,
}

/// 跨平台着色器内建数学函数枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMathIntrinsic {
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Round,
    Fract,
    Exp,
    Ln,
    Min,
    Max,
    Atan2,
    Powf,
    Clamp,
    Fma,
    Dot,
    Length,
    Normalize,
    Distance,
}

/// 左值表达式节点（支持标识符、结构体字段访问与数组下标索引）。
#[derive(Debug, Clone)]
pub enum GpuLValue {
    Var(&'static str),
    Field {
        base: Box<GpuLValue>,
        field: &'static str,
    },
    Index {
        base: Box<GpuLValue>,
        index: Box<GpuExpr>,
    },
}

/// `#[gpu]` 辅助函数形参类型分类（支持标量、定长向量数组与组件结构体）。
#[derive(Debug, Clone, Copy)]
pub enum GpuHelperParamKind {
    /// 基础标量 (`f32`, `i32`, `u32`, `bool`)
    Scalar(GpuScalarKind),
    /// 原生 SIMD 向量 (`vec2`, `vec3`, `vec4`)
    Vector {
        elem: GpuScalarKind,
        len: usize,
    },
    /// 定长数组 (`[f32; 2..=4]`, `[i32; 2..=4]`, `[u32; 2..=4]`)
    Array {
        elem: GpuScalarKind,
        len: usize,
    },
    /// 组件结构体 (`&Comp` 或 `&mut Comp`)
    Struct {
        struct_name: &'static str,
        type_id: ComponentTypeId,
        fields: &'static [GpuStructField],
        msl_def: fn() -> String,
        wgsl_def: fn() -> String,
    },
}

impl GpuHelperParamKind {
    /// 转换为符号表 [`GpuValueType`]。
    #[inline]
    pub const fn to_value_type(self) -> GpuValueType {
        match self {
            Self::Scalar(k) => GpuValueType::Scalar(k),
            Self::Vector { elem, len } => GpuValueType::Vector(elem, len),
            Self::Array { elem, len } => GpuValueType::Array(elem, len),
            Self::Struct { struct_name, .. } => GpuValueType::Struct(struct_name),
        }
    }
}

/// `#[gpu]` 辅助函数单个形参规格。
#[derive(Debug, Clone, Copy)]
pub struct GpuHelperParamSpec {
    /// 形参名
    pub name: &'static str,
    /// 函数体内是否允许修改该绑定
    pub is_mut: bool,
    /// 是否按可变引用传递 (`&mut T`)，修改将写回调用方实参
    pub by_mut_ref: bool,
    /// 形参类型
    pub kind: GpuHelperParamKind,
}

/// 可组合 `#[gpu]` 辅助函数（Device Function）静态描述符。
#[derive(Debug, Clone)]
pub struct GpuHelperFnDescriptor {
    /// 辅助函数名称
    pub name: &'static str,
    /// 形参列表
    pub params: Vec<GpuHelperParamSpec>,
    /// 返回值标量类型（`None` 表示无返回值 `void` 子算子，`Some(kind)` 表示返回标量）
    pub return_type: Option<GpuScalarKind>,
    /// 函数体强类型 IR 块
    pub body: GpuBlock,
}

/// 强类型着色器右值表达式树节点。
#[derive(Debug, Clone)]
pub enum GpuExpr {
    /// 局部/形参变量引用
    Var(&'static str),
    /// 强类型字面量（含编译期求值的外部 `const` 常量）
    Lit(GpuLit),
    /// 定长数组初始化表达式 (`[a, b, c]`)
    ArrayInit(Vec<GpuExpr>),
    /// 定长数组重复初始化表达式 (`[val; N]`)
    ArrayRepeat {
        elem: Box<GpuExpr>,
        len: usize,
    },
    /// 原生 SIMD 向量构造表达式 (`vec2`, `vec3`, `vec4` 或从数组加载转换)
    VecConstruct {
        elem_ty: GpuScalarKind,
        elements: Vec<GpuExpr>,
    },
    /// 将原生 SIMD 向量转换为紧凑定长数组 (`array<T, N>` / `packed_T{N}`)
    VecToArray(Box<GpuExpr>),
    /// 结构体字段读取 (`base.field`)
    Field {
        base: Box<GpuExpr>,
        field: &'static str,
    },
    /// 数组元素下标读取 (`base[index]`)
    Index {
        base: Box<GpuExpr>,
        index: Box<GpuExpr>,
    },
    /// 二元运算表达式 (`lhs op rhs`)
    Binary {
        op: GpuBinaryOp,
        lhs: Box<GpuExpr>,
        rhs: Box<GpuExpr>,
    },
    /// 一元运算表达式 (`-expr` / `!expr`)
    Unary {
        op: GpuUnaryOp,
        expr: Box<GpuExpr>,
    },
    /// 括号分组表达式 (`(expr)`)
    Paren(Box<GpuExpr>),
    /// 标量类型强转 (`expr as ty`)
    Cast {
        expr: Box<GpuExpr>,
        ty: GpuScalarKind,
    },
    /// 跨平台数学内建函数调用
    MathCall {
        receiver: Box<GpuExpr>,
        intrinsic: GpuMathIntrinsic,
        args: Vec<GpuExpr>,
    },
    /// 调用另一个 `#[gpu]` 辅助函数或子算子
    HelperCall {
        helper: &'static GpuHelperFnDescriptor,
        args: Vec<GpuExpr>,
    },
}

impl GpuExpr {
    /// 构造 2D 浮点向量 (`vec2<f32>` / `float2`)
    pub fn vec2(x: GpuExpr, y: GpuExpr) -> Self {
        Self::VecConstruct {
            elem_ty: GpuScalarKind::F32,
            elements: vec![x, y],
        }
    }

    /// 构造 3D 浮点向量 (`vec3<f32>` / `float3`)
    pub fn vec3(x: GpuExpr, y: GpuExpr, z: GpuExpr) -> Self {
        Self::VecConstruct {
            elem_ty: GpuScalarKind::F32,
            elements: vec![x, y, z],
        }
    }

    /// 构造 4D 浮点向量 (`vec4<f32>` / `float4`)
    pub fn vec4(x: GpuExpr, y: GpuExpr, z: GpuExpr, w: GpuExpr) -> Self {
        Self::VecConstruct {
            elem_ty: GpuScalarKind::F32,
            elements: vec![x, y, z, w],
        }
    }

    /// 从定长紧凑数组加载为原生 SIMD 向量
    pub fn vec_from_array(elem_ty: GpuScalarKind, arr: GpuExpr) -> Self {
        Self::VecConstruct {
            elem_ty,
            elements: vec![arr],
        }
    }
}

/// 强类型着色器语句节点。
#[derive(Debug, Clone)]
pub enum GpuStmt {
    /// 局部变量声明 (`let [mut] name [: ty] = init;`)
    Let {
        name: &'static str,
        mutable: bool,
        explicit_ty: Option<GpuValueType>,
        init: GpuExpr,
    },
    /// 赋值或复合赋值语句 (`target op= value;`)
    Assign {
        op: GpuAssignOp,
        target: GpuLValue,
        value: GpuExpr,
    },
    /// 独立表达式语句（如调用无返回值 `#[gpu]` 辅助函数 `apply_drag(vel, 0.98);`）
    Expr(GpuExpr),
    /// 条件分支语句 (`if (cond) { then } else { ... }`)
    If {
        cond: GpuExpr,
        then_block: GpuBlock,
        else_branch: Option<GpuElseBranch>,
    },
    /// `while` 条件循环语句
    While {
        cond: GpuExpr,
        body: GpuBlock,
    },
    /// `loop { ... }` 无限循环语句（需配合 `break` 退出）
    Loop(GpuBlock),
    /// `for var in start..end` 范围循环语句
    ForRange {
        var: &'static str,
        start: GpuExpr,
        end: GpuExpr,
        inclusive: bool,
        body: GpuBlock,
    },
    /// 嵌套代码块
    Block(GpuBlock),
    /// 函数返回语句（核函数/无返回辅助函数为 `None`，有返回辅助函数为 `Some(expr)`）
    Return(Option<GpuExpr>),
    /// 循环跳出 (`break;`)
    Break,
    /// 循环继续 (`continue;`)
    Continue,
}

/// `if` 语句的 `else` 分支形态。
#[derive(Debug, Clone)]
pub enum GpuElseBranch {
    ElseIf(Box<GpuStmt>),
    ElseBlock(GpuBlock),
}

/// 着色器语句块。
#[derive(Debug, Clone, Default)]
pub struct GpuBlock {
    pub stmts: Vec<GpuStmt>,
}

impl GpuBlock {
    #[inline]
    pub fn new(stmts: Vec<GpuStmt>) -> Self {
        Self { stmts }
    }
}

/// 组件结构体 Schema 符号项（供 [`ShaderTypeEnv`] 查询字段类型）。
#[derive(Debug, Clone, Copy)]
pub struct GpuComponentSchema {
    pub param_name: &'static str,
    pub struct_name: &'static str,
    pub fields: &'static [GpuStructField],
    pub mutable: bool,
}

/// 词法作用域与组件字段双向类型推导环境。
#[derive(Clone)]
struct ShaderTypeEnv<'a> {
    schemas: HashMap<&'static str, &'static [GpuStructField]>,
    scopes: Vec<HashMap<&'static str, GpuValueType>>,
    mut_params: &'a [&'static str],
    helper_mut_ptr_params: Vec<&'static str>,
    return_type: Option<GpuScalarKind>,
}

impl<'a> ShaderTypeEnv<'a> {
    fn for_kernel(
        components: &[GpuComponentSchema],
        scalar_uniforms: &[GpuScalarUniformSpec],
        mut_params: &'a [&'static str],
    ) -> Self {
        let mut schemas = HashMap::with_capacity(components.len());
        let mut root_scope = HashMap::with_capacity(components.len() + scalar_uniforms.len());
        for c in components {
            schemas.insert(c.struct_name, c.fields);
            root_scope.insert(c.param_name, GpuValueType::Struct(c.struct_name));
        }
        for u in scalar_uniforms {
            root_scope.insert(u.param_name, GpuValueType::Scalar(u.kind));
        }
        Self {
            schemas,
            scopes: vec![root_scope],
            mut_params,
            helper_mut_ptr_params: Vec::new(),
            return_type: None,
        }
    }

    fn for_helper(params: &[GpuHelperParamSpec], return_type: Option<GpuScalarKind>) -> Self {
        let mut schemas = HashMap::new();
        let mut root_scope = HashMap::with_capacity(params.len());
        let mut helper_mut_ptr_params = Vec::new();
        for p in params {
            if let GpuHelperParamKind::Struct {
                struct_name,
                fields,
                ..
            } = p.kind
            {
                schemas.insert(struct_name, fields);
            }
            if p.by_mut_ref {
                helper_mut_ptr_params.push(p.name);
            }
            root_scope.insert(p.name, p.kind.to_value_type());
        }
        Self {
            schemas,
            scopes: vec![root_scope],
            mut_params: &[],
            helper_mut_ptr_params,
            return_type,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &'static str, ty: GpuValueType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_var(&self, name: &str) -> GpuValueType {
        for scope in self.scopes.iter().rev() {
            if let Some(&ty) = scope.get(name) {
                return ty;
            }
        }
        GpuValueType::Unknown
    }

    fn lookup_field(&self, struct_name: &str, field_name: &str) -> GpuValueType {
        if let Some(&fields) = self.schemas.get(struct_name)
            && let Some(f) = fields.iter().find(|f| f.name == field_name)
        {
            return GpuValueType::from_field_type(f.field_type);
        }
        GpuValueType::Unknown
    }

    fn infer_lvalue_type(&self, lv: &GpuLValue) -> GpuValueType {
        match lv {
            GpuLValue::Var(name) => self.lookup_var(name),
            GpuLValue::Field { base, field } => {
                let base_ty = self.infer_lvalue_type(base);
                match base_ty {
                    GpuValueType::Struct(sname) => self.lookup_field(sname, field),
                    GpuValueType::Vector(k, len) | GpuValueType::Array(k, len)
                        if swizzle_component_index(field, len).is_some() =>
                    {
                        GpuValueType::Scalar(k)
                    }
                    _ => GpuValueType::Unknown,
                }
            }
            GpuLValue::Index { base, .. } => {
                let base_ty = self.infer_lvalue_type(base);
                match base_ty {
                    GpuValueType::Array(elem_kind, _) | GpuValueType::Vector(elem_kind, _) => {
                        GpuValueType::Scalar(elem_kind)
                    }
                    _ => GpuValueType::Unknown,
                }
            }
        }
    }

    fn infer_expr_type(&self, expr: &GpuExpr) -> GpuValueType {
        match expr {
            GpuExpr::Var(name) => self.lookup_var(name),
            GpuExpr::Lit(lit) => match lit {
                GpuLit::FloatRaw(_) | GpuLit::F32(_) => GpuValueType::Scalar(GpuScalarKind::F32),
                GpuLit::UntypedInt(_) | GpuLit::I32(_) => GpuValueType::Scalar(GpuScalarKind::I32),
                GpuLit::U32(_) => GpuValueType::Scalar(GpuScalarKind::U32),
                GpuLit::Bool(_) => GpuValueType::Scalar(GpuScalarKind::Bool),
            },
            GpuExpr::ArrayInit(elems) => {
                let len = elems.len();
                let elem_kind = elems
                    .first()
                    .and_then(|e| match self.infer_expr_type(e) {
                        GpuValueType::Scalar(k) => Some(k),
                        _ => None,
                    })
                    .unwrap_or(GpuScalarKind::F32);
                GpuValueType::Array(elem_kind, len)
            }
            GpuExpr::ArrayRepeat { elem, len } => {
                let elem_kind = match self.infer_expr_type(elem) {
                    GpuValueType::Scalar(k) => k,
                    _ => GpuScalarKind::F32,
                };
                GpuValueType::Array(elem_kind, *len)
            }
            GpuExpr::VecConstruct { elem_ty, elements } => {
                if elements.len() == 1 {
                    match self.infer_expr_type(&elements[0]) {
                        GpuValueType::Array(_, len) | GpuValueType::Vector(_, len) => {
                            GpuValueType::Vector(*elem_ty, len)
                        }
                        _ => GpuValueType::Vector(*elem_ty, 1),
                    }
                } else {
                    GpuValueType::Vector(*elem_ty, elements.len())
                }
            }
            GpuExpr::VecToArray(inner) => match self.infer_expr_type(inner) {
                GpuValueType::Vector(k, len) | GpuValueType::Array(k, len) => {
                    GpuValueType::Array(k, len)
                }
                _ => GpuValueType::Unknown,
            },
            GpuExpr::Field { base, field } => {
                match self.infer_expr_type(base) {
                    GpuValueType::Struct(sname) => self.lookup_field(sname, field),
                    GpuValueType::Vector(k, len) | GpuValueType::Array(k, len)
                        if swizzle_component_index(field, len).is_some() =>
                    {
                        GpuValueType::Scalar(k)
                    }
                    _ => GpuValueType::Unknown,
                }
            }
            GpuExpr::Index { base, .. } => match self.infer_expr_type(base) {
                GpuValueType::Array(elem_kind, _) | GpuValueType::Vector(elem_kind, _) => {
                    GpuValueType::Scalar(elem_kind)
                }
                _ => GpuValueType::Unknown,
            },
            GpuExpr::Binary { op, lhs, rhs } => {
                if op.is_comparison_or_logical() {
                    return GpuValueType::Scalar(GpuScalarKind::Bool);
                }
                if matches!(op, GpuBinaryOp::Shl | GpuBinaryOp::Shr) {
                    return self.infer_expr_type(lhs);
                }
                let lt = self.infer_expr_type(lhs);
                let rt = self.infer_expr_type(rhs);
                // Vector/Array - Vector/Array
                if let (Some((lk, llen)), Some((rk, rlen))) =
                    (lt.as_vector_or_array(), rt.as_vector_or_array())
                {
                    let k = if lk == GpuScalarKind::U32 || rk == GpuScalarKind::U32 {
                        GpuScalarKind::U32
                    } else {
                        lk
                    };
                    return GpuValueType::Vector(k, llen.max(rlen));
                }
                // Vector/Array - Scalar
                if let Some((lk, len)) = lt.as_vector_or_array() {
                    return GpuValueType::Vector(lk, len);
                }
                // Scalar - Vector/Array
                if let Some((rk, len)) = rt.as_vector_or_array() {
                    return GpuValueType::Vector(rk, len);
                }
                if matches!(lt, GpuValueType::Scalar(GpuScalarKind::U32))
                    || matches!(rt, GpuValueType::Scalar(GpuScalarKind::U32))
                {
                    return GpuValueType::Scalar(GpuScalarKind::U32);
                }
                if !matches!(lt, GpuValueType::Unknown) {
                    lt
                } else {
                    rt
                }
            }
            GpuExpr::Unary { op, expr } => match op {
                GpuUnaryOp::Not => GpuValueType::Scalar(GpuScalarKind::Bool),
                GpuUnaryOp::Neg => match self.infer_expr_type(expr) {
                    GpuValueType::Array(k, len) => GpuValueType::Vector(k, len),
                    other => other,
                },
            },
            GpuExpr::Paren(inner) => self.infer_expr_type(inner),
            GpuExpr::Cast { ty, .. } => GpuValueType::Scalar(*ty),
            GpuExpr::MathCall {
                receiver,
                intrinsic,
                ..
            } => match intrinsic {
                GpuMathIntrinsic::Abs
                | GpuMathIntrinsic::Min
                | GpuMathIntrinsic::Max
                | GpuMathIntrinsic::Clamp => match self.infer_expr_type(receiver) {
                    GpuValueType::Array(k, len) => GpuValueType::Vector(k, len),
                    other => other,
                },
                GpuMathIntrinsic::Dot | GpuMathIntrinsic::Length | GpuMathIntrinsic::Distance => {
                    match self.infer_expr_type(receiver) {
                        GpuValueType::Vector(k, _) | GpuValueType::Array(k, _) => {
                            GpuValueType::Scalar(k)
                        }
                        _ => GpuValueType::Scalar(GpuScalarKind::F32),
                    }
                }
                GpuMathIntrinsic::Normalize => match self.infer_expr_type(receiver) {
                    GpuValueType::Array(k, len) => GpuValueType::Vector(k, len),
                    other => other,
                },
                _ => GpuValueType::Scalar(GpuScalarKind::F32),
            },
            GpuExpr::HelperCall { helper, .. } => match helper.return_type {
                Some(k) => GpuValueType::Scalar(k),
                None => GpuValueType::Unknown,
            },
        }
    }

    fn is_strictly_unsigned_expr(&self, expr: &GpuExpr) -> bool {
        match expr {
            GpuExpr::Lit(GpuLit::UntypedInt(_)) => false,
            GpuExpr::VecConstruct { elem_ty, .. } => *elem_ty == GpuScalarKind::U32,
            GpuExpr::Paren(inner) => self.is_strictly_unsigned_expr(inner),
            GpuExpr::Binary { op, lhs, rhs } => {
                if op.is_comparison_or_logical() {
                    false
                } else if matches!(op, GpuBinaryOp::Shl | GpuBinaryOp::Shr) {
                    self.is_strictly_unsigned_expr(lhs)
                } else {
                    self.is_strictly_unsigned_expr(lhs) || self.is_strictly_unsigned_expr(rhs)
                }
            }
            _ => matches!(
                self.infer_expr_type(expr),
                GpuValueType::Scalar(GpuScalarKind::U32)
                    | GpuValueType::Vector(GpuScalarKind::U32, _)
            ),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IrBackendTarget {
    Msl,
    Wgsl,
}

struct IrCodeEmitter {
    target: IrBackendTarget,
}

impl IrCodeEmitter {
    fn new(target: IrBackendTarget) -> Self {
        Self { target }
    }

    fn emit_block(&self, block: &GpuBlock, env: &mut ShaderTypeEnv<'_>) -> String {
        env.push_scope();
        let mut out = String::new();
        for (idx, stmt) in block.stmts.iter().enumerate() {
            if idx > 0 {
                out.push(' ');
            }
            out.push_str(&self.emit_stmt(stmt, env));
        }
        env.pop_scope();
        out
    }

    fn emit_stmt(&self, stmt: &GpuStmt, env: &mut ShaderTypeEnv<'_>) -> String {
        match stmt {
            GpuStmt::Let {
                name,
                mutable,
                explicit_ty,
                init,
            } => {
                let raw_init_ty = env.infer_expr_type(init);
                let inferred = if let Some(vt) = explicit_ty {
                    *vt
                } else if env.is_strictly_unsigned_expr(init) {
                    GpuValueType::Scalar(GpuScalarKind::U32)
                } else {
                    raw_init_ty
                };
                let coerce_u32 = matches!(
                    inferred,
                    GpuValueType::Scalar(GpuScalarKind::U32)
                        | GpuValueType::Array(GpuScalarKind::U32, _)
                        | GpuValueType::Vector(GpuScalarKind::U32, _)
                );
                let rhs = match (inferred, raw_init_ty) {
                    (GpuValueType::Vector(_, vlen), GpuValueType::Array(_, alen))
                        if vlen == alen =>
                    {
                        self.emit_expr_as_vec(init, env, coerce_u32)
                    }
                    (GpuValueType::Array(ak, alen), GpuValueType::Vector(_, vlen))
                        if alen == vlen && self.target == IrBackendTarget::Wgsl =>
                    {
                        let inner = self.emit_expr(init, env, coerce_u32);
                        let k_str = ak.wgsl_type();
                        let comps: Vec<String> =
                            (0..alen).map(|i| format!("({inner})[{i}]")).collect();
                        format!("array<{k_str}, {alen}>({})", comps.join(", "))
                    }
                    _ => self.emit_expr(init, env, coerce_u32),
                };
                env.define_var(name, inferred);
                match (self.target, *mutable, explicit_ty) {
                    (IrBackendTarget::Msl, true, Some(ty)) => {
                        let tname = ty.msl_type_name().unwrap_or("auto");
                        format!("{tname} {name} = {rhs};")
                    }
                    (IrBackendTarget::Msl, false, Some(ty)) => {
                        let tname = ty.msl_type_name().unwrap_or("auto");
                        format!("const {tname} {name} = {rhs};")
                    }
                    (IrBackendTarget::Msl, true, None) => format!("auto {name} = {rhs};"),
                    (IrBackendTarget::Msl, false, None) => format!("const auto {name} = {rhs};"),
                    (IrBackendTarget::Wgsl, true, Some(ty)) => {
                        if let Some(tname) = ty.wgsl_type_name() {
                            format!("var {name}: {tname} = {rhs};")
                        } else {
                            format!("var {name} = {rhs};")
                        }
                    }
                    (IrBackendTarget::Wgsl, false, Some(ty)) => {
                        if let Some(tname) = ty.wgsl_type_name() {
                            format!("let {name}: {tname} = {rhs};")
                        } else {
                            format!("let {name} = {rhs};")
                        }
                    }
                    (IrBackendTarget::Wgsl, true, None) => format!("var {name} = {rhs};"),
                    (IrBackendTarget::Wgsl, false, None) => format!("let {name} = {rhs};"),
                }
            }
            GpuStmt::Assign { op, target, value } => {
                let target_ty = env.infer_lvalue_type(target);
                let value_ty = env.infer_expr_type(value);
                let is_shift = matches!(op, GpuAssignOp::ShlAssign | GpuAssignOp::ShrAssign);
                let coerce_u32 = is_shift
                    || matches!(
                        target_ty,
                        GpuValueType::Scalar(GpuScalarKind::U32)
                            | GpuValueType::Array(GpuScalarKind::U32, _)
                            | GpuValueType::Vector(GpuScalarKind::U32, _)
                    )
                    || env.is_strictly_unsigned_expr(value);
                let lhs = self.emit_lvalue(target, env);
                if let (GpuValueType::Vector(_, vlen), GpuValueType::Array(_, alen)) =
                    (target_ty, value_ty)
                    && vlen == alen
                {
                    let rhs = self.emit_expr_as_vec(value, env, coerce_u32);
                    return format!("{lhs} {} {rhs};", op.as_str());
                }
                if self.target == IrBackendTarget::Wgsl {
                    if let (GpuValueType::Array(ak, alen), GpuValueType::Vector(_, vlen)) =
                        (target_ty, value_ty)
                    {
                        if alen == vlen {
                            let rhs = self.emit_expr(value, env, coerce_u32);
                            let k_str = ak.wgsl_type();
                            let bin_op = match op {
                                GpuAssignOp::Assign => None,
                                GpuAssignOp::AddAssign => Some("+"),
                                GpuAssignOp::SubAssign => Some("-"),
                                GpuAssignOp::MulAssign => Some("*"),
                                GpuAssignOp::DivAssign => Some("/"),
                                _ => None,
                            };
                            if let Some(bop) = bin_op {
                                let comps: Vec<String> = (0..alen)
                                    .map(|i| format!("({lhs})[{i}] {bop} ({rhs})[{i}]"))
                                    .collect();
                                return format!("{lhs} = array<{k_str}, {alen}>({});", comps.join(", "));
                            } else if *op == GpuAssignOp::Assign {
                                let comps: Vec<String> =
                                    (0..alen).map(|i| format!("({rhs})[{i}]")).collect();
                                return format!("{lhs} = array<{k_str}, {alen}>({});", comps.join(", "));
                            }
                        }
                    } else if let (GpuValueType::Array(ak, alen), GpuValueType::Array(_, rlen)) =
                        (target_ty, value_ty)
                        && alen == rlen
                    {
                        let bin_op = match op {
                            GpuAssignOp::AddAssign => Some("+"),
                            GpuAssignOp::SubAssign => Some("-"),
                            GpuAssignOp::MulAssign => Some("*"),
                            GpuAssignOp::DivAssign => Some("/"),
                            _ => None,
                        };
                        if let Some(bop) = bin_op {
                            let rhs = self.emit_expr(value, env, coerce_u32);
                            let k_str = ak.wgsl_type();
                            let comps: Vec<String> = (0..alen)
                                .map(|i| format!("({lhs})[{i}] {bop} ({rhs})[{i}]"))
                                .collect();
                            return format!("{lhs} = array<{k_str}, {alen}>({});", comps.join(", "));
                        }
                    } else if let (GpuValueType::Array(ak, alen), GpuValueType::Scalar(_)) =
                        (target_ty, value_ty)
                    {
                        let bin_op = match op {
                            GpuAssignOp::AddAssign => Some("+"),
                            GpuAssignOp::SubAssign => Some("-"),
                            GpuAssignOp::MulAssign => Some("*"),
                            GpuAssignOp::DivAssign => Some("/"),
                            _ => None,
                        };
                        if let Some(bop) = bin_op {
                            let rhs = self.emit_expr(value, env, coerce_u32);
                            let k_str = ak.wgsl_type();
                            let comps: Vec<String> = (0..alen)
                                .map(|i| format!("({lhs})[{i}] {bop} ({rhs})"))
                                .collect();
                            return format!("{lhs} = array<{k_str}, {alen}>({});", comps.join(", "));
                        }
                    }
                }
                let rhs = self.emit_expr(value, env, coerce_u32);
                format!("{lhs} {} {rhs};", op.as_str())
            }
            GpuStmt::Expr(expr) => {
                let s = self.emit_expr(expr, env, false);
                format!("{s};")
            }
            GpuStmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                let cond_str = self.emit_expr(cond, env, false);
                let then_str = self.emit_block(then_block, env);
                let mut s = format!("if ({cond_str}) {{ {then_str} }}");
                if let Some(eb) = else_branch {
                    match eb {
                        GpuElseBranch::ElseIf(elif_stmt) => {
                            let elif_str = self.emit_stmt(elif_stmt, env);
                            s.push_str(&format!(" else {elif_str}"));
                        }
                        GpuElseBranch::ElseBlock(eblock) => {
                            let else_str = self.emit_block(eblock, env);
                            s.push_str(&format!(" else {{ {else_str} }}"));
                        }
                    }
                }
                s
            }
            GpuStmt::While { cond, body } => {
                let cond_str = self.emit_expr(cond, env, false);
                let body_str = self.emit_block(body, env);
                format!("while ({cond_str}) {{ {body_str} }}")
            }
            GpuStmt::Loop(body) => {
                let body_str = self.emit_block(body, env);
                format!("while (true) {{ {body_str} }}")
            }
            GpuStmt::ForRange {
                var,
                start,
                end,
                inclusive,
                body,
            } => {
                let is_unsigned =
                    env.is_strictly_unsigned_expr(start) || env.is_strictly_unsigned_expr(end);
                let start_str = self.emit_expr(start, env, is_unsigned);
                let end_str = self.emit_expr(end, env, is_unsigned);
                let step_str = if is_unsigned { "1u" } else { "1" };
                let cmp_str = if *inclusive { "<=" } else { "<" };

                env.push_scope();
                env.define_var(
                    var,
                    GpuValueType::Scalar(if is_unsigned {
                        GpuScalarKind::U32
                    } else {
                        GpuScalarKind::I32
                    }),
                );
                let body_str = self.emit_block(body, env);
                env.pop_scope();

                match self.target {
                    IrBackendTarget::Msl => format!(
                        "for (auto {var} = {start_str}; {var} {cmp_str} {end_str}; {var} += {step_str}) {{ {body_str} }}"
                    ),
                    IrBackendTarget::Wgsl => format!(
                        "for (var {var} = {start_str}; {var} {cmp_str} {end_str}; {var} += {step_str}) {{ {body_str} }}"
                    ),
                }
            }
            GpuStmt::Block(inner) => {
                let inner_str = self.emit_block(inner, env);
                format!("{{ {inner_str} }}")
            }
            GpuStmt::Return(opt_val) => {
                if let Some(val) = opt_val {
                    let coerce_u32 = env.return_type == Some(GpuScalarKind::U32)
                        || env.is_strictly_unsigned_expr(val);
                    let val_str = self.emit_expr(val, env, coerce_u32);
                    if self.target == IrBackendTarget::Wgsl && !env.helper_mut_ptr_params.is_empty()
                    {
                        let mut wb = format!("let sumeru_ret = {val_str}; ");
                        for p in &env.helper_mut_ptr_params {
                            wb.push_str(&format!("*sumeru_ptr_{p} = {p}; "));
                        }
                        wb.push_str("return sumeru_ret;");
                        wb
                    } else {
                        format!("return {val_str};")
                    }
                } else {
                    match self.target {
                        IrBackendTarget::Msl => {
                            let mut wb = String::new();
                            for p in env.mut_params {
                                wb.push_str(&format!("sumeru_buf_{p}[gid] = {p}; "));
                            }
                            wb.push_str("return;");
                            wb
                        }
                        IrBackendTarget::Wgsl => {
                            let mut wb = String::new();
                            for p in env.mut_params {
                                wb.push_str(&format!("sumeru_buf_{p}[gid] = {p}; "));
                            }
                            for p in &env.helper_mut_ptr_params {
                                wb.push_str(&format!("*sumeru_ptr_{p} = {p}; "));
                            }
                            wb.push_str("return;");
                            wb
                        }
                    }
                }
            }
            GpuStmt::Break => "break;".to_string(),
            GpuStmt::Continue => "continue;".to_string(),
        }
    }

    fn emit_lvalue(&self, lv: &GpuLValue, env: &mut ShaderTypeEnv<'_>) -> String {
        match lv {
            GpuLValue::Var(name) => (*name).to_string(),
            GpuLValue::Field { base, field } => {
                let base_ty = env.infer_lvalue_type(base);
                let b = self.emit_lvalue(base, env);
                if self.target == IrBackendTarget::Wgsl
                    && let GpuValueType::Array(_, len) = base_ty
                    && let Some(idx) = swizzle_component_index(field, len)
                {
                    return format!("{b}[{idx}]");
                }
                format!("{b}.{field}")
            }
            GpuLValue::Index { base, index } => {
                let b = self.emit_lvalue(base, env);
                let idx = self.emit_expr(index, env, false);
                format!("{b}[{idx}]")
            }
        }
    }

    fn emit_expr_as_vec(
        &self,
        expr: &GpuExpr,
        env: &mut ShaderTypeEnv<'_>,
        coerce_unsigned: bool,
    ) -> String {
        let ty = env.infer_expr_type(expr);
        let raw = self.emit_expr(expr, env, coerce_unsigned);
        if let GpuValueType::Array(elem_ty, len) = ty {
            match self.target {
                IrBackendTarget::Msl => {
                    let tname = GpuValueType::Vector(elem_ty, len)
                        .msl_type_name()
                        .unwrap_or("float3");
                    format!("{tname}({raw})")
                }
                IrBackendTarget::Wgsl => {
                    let k_str = elem_ty.wgsl_type();
                    let comps: Vec<String> = (0..len).map(|i| format!("({raw})[{i}]")).collect();
                    format!("vec{len}<{k_str}>({})", comps.join(", "))
                }
            }
        } else {
            raw
        }
    }

    fn emit_expr(
        &self,
        expr: &GpuExpr,
        env: &mut ShaderTypeEnv<'_>,
        coerce_unsigned: bool,
    ) -> String {
        match expr {
            GpuExpr::Var(name) => (*name).to_string(),
            GpuExpr::Lit(lit) => lit.format_shader(coerce_unsigned),
            GpuExpr::ArrayInit(elems) => {
                let elem_u32 = coerce_unsigned
                    || elems.iter().any(|e| env.is_strictly_unsigned_expr(e));
                let arr_ty = match env.infer_expr_type(expr) {
                    GpuValueType::Array(_, len) if elem_u32 => {
                        GpuValueType::Array(GpuScalarKind::U32, len)
                    }
                    other => other,
                };
                let tname = match self.target {
                    IrBackendTarget::Msl => arr_ty.msl_type_name().unwrap_or("packed_float3"),
                    IrBackendTarget::Wgsl => arr_ty.wgsl_type_name().unwrap_or("array<f32, 3>"),
                };
                let parts: Vec<String> = elems
                    .iter()
                    .map(|e| self.emit_expr(e, env, elem_u32))
                    .collect();
                format!("{tname}({})", parts.join(", "))
            }
            GpuExpr::ArrayRepeat { elem, len } => {
                let elem_u32 = coerce_unsigned || env.is_strictly_unsigned_expr(elem);
                let arr_ty = match env.infer_expr_type(expr) {
                    GpuValueType::Array(_, n) if elem_u32 => {
                        GpuValueType::Array(GpuScalarKind::U32, n)
                    }
                    other => other,
                };
                let tname = match self.target {
                    IrBackendTarget::Msl => arr_ty.msl_type_name().unwrap_or("packed_float3"),
                    IrBackendTarget::Wgsl => arr_ty.wgsl_type_name().unwrap_or("array<f32, 3>"),
                };
                let val_str = self.emit_expr(elem, env, elem_u32);
                let parts = vec![val_str; *len];
                format!("{tname}({})", parts.join(", "))
            }
            GpuExpr::VecConstruct { elem_ty, elements } => {
                if elements.len() == 1 {
                    let inner_ty = env.infer_expr_type(&elements[0]);
                    let len = match inner_ty {
                        GpuValueType::Array(_, l) | GpuValueType::Vector(_, l) => l,
                        _ => 3,
                    };
                    let inner = self.emit_expr(&elements[0], env, false);
                    match self.target {
                        IrBackendTarget::Msl => {
                            let tname = match (*elem_ty, len) {
                                (GpuScalarKind::F32, 2) => "float2",
                                (GpuScalarKind::F32, 3) => "float3",
                                (GpuScalarKind::F32, 4) => "float4",
                                (GpuScalarKind::I32, 2) => "int2",
                                (GpuScalarKind::I32, 3) => "int3",
                                (GpuScalarKind::I32, 4) => "int4",
                                (GpuScalarKind::U32, 2) => "uint2",
                                (GpuScalarKind::U32, 3) => "uint3",
                                (GpuScalarKind::U32, 4) => "uint4",
                                (GpuScalarKind::Bool, 2) => "bool2",
                                (GpuScalarKind::Bool, 3) => "bool3",
                                (GpuScalarKind::Bool, 4) => "bool4",
                                _ => "float3",
                            };
                            format!("{tname}({inner})")
                        }
                        IrBackendTarget::Wgsl => {
                            let k_str = elem_ty.wgsl_type();
                            let comps: Vec<String> =
                                (0..len).map(|i| format!("({inner})[{i}]")).collect();
                            format!("vec{len}<{k_str}>({})", comps.join(", "))
                        }
                    }
                } else {
                    let len = elements.len();
                    let elem_u32 = *elem_ty == GpuScalarKind::U32;
                    let parts: Vec<String> = elements
                        .iter()
                        .map(|e| self.emit_expr(e, env, elem_u32))
                        .collect();
                    match self.target {
                        IrBackendTarget::Msl => {
                            let tname = match (*elem_ty, len) {
                                (GpuScalarKind::F32, 2) => "float2",
                                (GpuScalarKind::F32, 3) => "float3",
                                (GpuScalarKind::F32, 4) => "float4",
                                (GpuScalarKind::I32, 2) => "int2",
                                (GpuScalarKind::I32, 3) => "int3",
                                (GpuScalarKind::I32, 4) => "int4",
                                (GpuScalarKind::U32, 2) => "uint2",
                                (GpuScalarKind::U32, 3) => "uint3",
                                (GpuScalarKind::U32, 4) => "uint4",
                                (GpuScalarKind::Bool, 2) => "bool2",
                                (GpuScalarKind::Bool, 3) => "bool3",
                                (GpuScalarKind::Bool, 4) => "bool4",
                                _ => "float3",
                            };
                            format!("{tname}({})", parts.join(", "))
                        }
                        IrBackendTarget::Wgsl => {
                            let k_str = elem_ty.wgsl_type();
                            format!("vec{len}<{k_str}>({})", parts.join(", "))
                        }
                    }
                }
            }
            GpuExpr::VecToArray(inner) => {
                let inner_ty = env.infer_expr_type(inner);
                let (elem_ty, len) = match inner_ty {
                    GpuValueType::Vector(k, l) | GpuValueType::Array(k, l) => (k, l),
                    _ => (GpuScalarKind::F32, 3),
                };
                let inner_str = self.emit_expr(inner, env, false);
                match self.target {
                    IrBackendTarget::Msl => {
                        let tname = match (elem_ty, len) {
                            (GpuScalarKind::F32, 2) => "packed_float2",
                            (GpuScalarKind::F32, 3) => "packed_float3",
                            (GpuScalarKind::F32, 4) => "packed_float4",
                            (GpuScalarKind::I32, 2) => "packed_int2",
                            (GpuScalarKind::I32, 3) => "packed_int3",
                            (GpuScalarKind::I32, 4) => "packed_int4",
                            (GpuScalarKind::U32, 2) => "packed_uint2",
                            (GpuScalarKind::U32, 3) => "packed_uint3",
                            (GpuScalarKind::U32, 4) => "packed_uint4",
                            _ => "packed_float3",
                        };
                        format!("{tname}({inner_str})")
                    }
                    IrBackendTarget::Wgsl => {
                        let k_str = elem_ty.wgsl_type();
                        let comps: Vec<String> =
                            (0..len).map(|i| format!("({inner_str})[{i}]")).collect();
                        format!("array<{k_str}, {len}>({})", comps.join(", "))
                    }
                }
            }
            GpuExpr::Field { base, field } => {
                let base_ty = env.infer_expr_type(base);
                let b = self.emit_expr(base, env, false);
                if self.target == IrBackendTarget::Wgsl
                    && let GpuValueType::Array(_, len) = base_ty
                    && let Some(idx) = swizzle_component_index(field, len)
                {
                    return format!("{b}[{idx}]");
                }
                format!("{b}.{field}")
            }
            GpuExpr::Index { base, index } => {
                let b = self.emit_expr(base, env, false);
                let idx = self.emit_expr(index, env, false);
                format!("{b}[{idx}]")
            }
            GpuExpr::Binary { op, lhs, rhs } => {
                if matches!(op, GpuBinaryOp::Shl | GpuBinaryOp::Shr) {
                    let lhs_u32 = coerce_unsigned || env.is_strictly_unsigned_expr(lhs);
                    let l = self.emit_expr_as_vec(lhs, env, lhs_u32);
                    // WGSL requires shift count (RHS) to always be u32
                    let r = self.emit_expr_as_vec(rhs, env, true);
                    return format!("{l} {} {r}", op.as_str());
                }
                let child_u32 = (!op.is_comparison_or_logical() && coerce_unsigned)
                    || env.is_strictly_unsigned_expr(lhs)
                    || env.is_strictly_unsigned_expr(rhs);
                let l = self.emit_expr_as_vec(lhs, env, child_u32);
                let r = self.emit_expr_as_vec(rhs, env, child_u32);
                format!("{l} {} {r}", op.as_str())
            }
            GpuExpr::Unary { op, expr } => {
                let inner = match op {
                    GpuUnaryOp::Neg => self.emit_expr_as_vec(expr, env, false),
                    GpuUnaryOp::Not => self.emit_expr(expr, env, false),
                };
                match op {
                    GpuUnaryOp::Neg => format!("-({inner})"),
                    GpuUnaryOp::Not => format!("!({inner})"),
                }
            }
            GpuExpr::Paren(inner) => {
                let s = self.emit_expr(inner, env, coerce_unsigned);
                format!("({s})")
            }
            GpuExpr::Cast { expr, ty } => {
                let inner = self.emit_expr(expr, env, false);
                match self.target {
                    IrBackendTarget::Msl => format!("{}({inner})", ty.msl_type()),
                    IrBackendTarget::Wgsl => format!("{}({inner})", ty.wgsl_type()),
                }
            }
            GpuExpr::MathCall {
                receiver,
                intrinsic,
                args,
            } => {
                let math_u32 = matches!(
                    intrinsic,
                    GpuMathIntrinsic::Min | GpuMathIntrinsic::Max | GpuMathIntrinsic::Clamp
                ) && (coerce_unsigned
                    || env.is_strictly_unsigned_expr(receiver)
                    || args.iter().any(|a| env.is_strictly_unsigned_expr(a)));
                let recv = self.emit_expr_as_vec(receiver, env, math_u32);
                let emitted_args: Vec<String> = args
                    .iter()
                    .map(|a| self.emit_expr_as_vec(a, env, math_u32))
                    .collect();
                match intrinsic {
                    GpuMathIntrinsic::Abs => format!("abs({recv})"),
                    GpuMathIntrinsic::Sqrt => format!("sqrt({recv})"),
                    GpuMathIntrinsic::Sin => format!("sin({recv})"),
                    GpuMathIntrinsic::Cos => format!("cos({recv})"),
                    GpuMathIntrinsic::Tan => format!("tan({recv})"),
                    GpuMathIntrinsic::Floor => format!("floor({recv})"),
                    GpuMathIntrinsic::Ceil => format!("ceil({recv})"),
                    GpuMathIntrinsic::Round => format!("round({recv})"),
                    GpuMathIntrinsic::Fract => format!("fract({recv})"),
                    GpuMathIntrinsic::Exp => format!("exp({recv})"),
                    GpuMathIntrinsic::Ln => format!("log({recv})"),
                    GpuMathIntrinsic::Min => format!("min({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Max => format!("max({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Atan2 => format!("atan2({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Powf => format!("pow({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Clamp => {
                        format!("clamp({recv}, {}, {})", emitted_args[0], emitted_args[1])
                    }
                    GpuMathIntrinsic::Fma => {
                        format!("fma({recv}, {}, {})", emitted_args[0], emitted_args[1])
                    }
                    GpuMathIntrinsic::Dot => format!("dot({recv}, {})", emitted_args[0]),
                    GpuMathIntrinsic::Length => format!("length({recv})"),
                    GpuMathIntrinsic::Normalize => format!("normalize({recv})"),
                    GpuMathIntrinsic::Distance => {
                        format!("distance({recv}, {})", emitted_args[0])
                    }
                }
            }
            GpuExpr::HelperCall { helper, args } => {
                let emitted_args: Vec<String> = args
                    .iter()
                    .enumerate()
                    .map(|(idx, a)| {
                        let p = helper.params.get(idx);
                        let arg_u32 = matches!(
                            p.map(|spec| spec.kind),
                            Some(GpuHelperParamKind::Scalar(GpuScalarKind::U32))
                        );
                        let s = self.emit_expr(a, env, arg_u32);
                        if self.target == IrBackendTarget::Wgsl
                            && p.is_some_and(|spec| spec.by_mut_ref)
                        {
                            format!("&{s}")
                        } else {
                            s
                        }
                    })
                    .collect();
                format!("{}({})", helper.name, emitted_args.join(", "))
            }
        }
    }

    fn emit_helper_fn(&self, helper: &GpuHelperFnDescriptor) -> String {
        let mut env = ShaderTypeEnv::for_helper(&helper.params, helper.return_type);
        match self.target {
            IrBackendTarget::Msl => {
                let mut params_str = Vec::with_capacity(helper.params.len());
                for p in &helper.params {
                    let ty_str = p.kind.to_value_type().msl_type_name().unwrap_or("float");
                    if p.by_mut_ref {
                        params_str.push(format!("thread {ty_str}& {}", p.name));
                    } else {
                        params_str.push(format!("{ty_str} {}", p.name));
                    }
                }
                let ret_str = helper.return_type.map_or("void", |k| k.msl_type());
                let body_str = self.emit_block(&helper.body, &mut env);
                format!(
                    "inline {ret_str} {}({}) {{\n    {}\n}}\n",
                    helper.name,
                    params_str.join(", "),
                    body_str
                )
            }
            IrBackendTarget::Wgsl => {
                let mut prelude = String::new();
                let mut epilogue = String::new();
                let mut params_str = Vec::with_capacity(helper.params.len());
                for p in &helper.params {
                    let ty_str = p.kind.to_value_type().wgsl_type_name().unwrap_or("f32");
                    if p.by_mut_ref {
                        prelude.push_str(&format!(
                            "var {}: {ty_str} = *sumeru_ptr_{}; ",
                            p.name, p.name
                        ));
                        if helper.return_type.is_none() {
                            epilogue.push_str(&format!(" *sumeru_ptr_{} = {};", p.name, p.name));
                        }
                        params_str.push(format!("sumeru_ptr_{}: ptr<function, {ty_str}>", p.name));
                    } else if p.is_mut {
                        prelude.push_str(&format!(
                            "var {}: {ty_str} = sumeru_arg_{}; ",
                            p.name, p.name
                        ));
                        params_str.push(format!("sumeru_arg_{}: {ty_str}", p.name));
                    } else {
                        params_str.push(format!("{}: {ty_str}", p.name));
                    }
                }
                let body_str = self.emit_block(&helper.body, &mut env);
                if let Some(ret_kind) = helper.return_type {
                    format!(
                        "fn {}({}) -> {} {{\n    {}{}\n}}\n",
                        helper.name,
                        params_str.join(", "),
                        ret_kind.wgsl_type(),
                        prelude,
                        body_str
                    )
                } else {
                    format!(
                        "fn {}({}) {{\n    {}{}{}\n}}\n",
                        helper.name,
                        params_str.join(", "),
                        prelude,
                        body_str,
                        epilogue
                    )
                }
            }
        }
    }
}

/// 递归按后序（被调用者在前，调用者在后）收集 `GpuBlock` 中引用的全部 `#[gpu]` 辅助函数。
fn collect_helpers_topological(
    block: &GpuBlock,
    out: &mut Vec<&'static GpuHelperFnDescriptor>,
) {
    for stmt in &block.stmts {
        collect_helpers_in_stmt(stmt, out);
    }
}

fn collect_helpers_in_stmt(stmt: &GpuStmt, out: &mut Vec<&'static GpuHelperFnDescriptor>) {
    match stmt {
        GpuStmt::Let { init, .. } | GpuStmt::Expr(init) => collect_helpers_in_expr(init, out),
        GpuStmt::Assign { target, value, .. } => {
            collect_helpers_in_lvalue(target, out);
            collect_helpers_in_expr(value, out);
        }
        GpuStmt::If {
            cond,
            then_block,
            else_branch,
        } => {
            collect_helpers_in_expr(cond, out);
            collect_helpers_topological(then_block, out);
            if let Some(eb) = else_branch {
                match eb {
                    GpuElseBranch::ElseIf(s) => collect_helpers_in_stmt(s, out),
                    GpuElseBranch::ElseBlock(b) => collect_helpers_topological(b, out),
                }
            }
        }
        GpuStmt::While { cond, body } => {
            collect_helpers_in_expr(cond, out);
            collect_helpers_topological(body, out);
        }
        GpuStmt::Loop(body) | GpuStmt::Block(body) => collect_helpers_topological(body, out),
        GpuStmt::ForRange {
            start, end, body, ..
        } => {
            collect_helpers_in_expr(start, out);
            collect_helpers_in_expr(end, out);
            collect_helpers_topological(body, out);
        }
        GpuStmt::Return(Some(e)) => collect_helpers_in_expr(e, out),
        GpuStmt::Return(None) | GpuStmt::Break | GpuStmt::Continue => {}
    }
}

fn collect_helpers_in_lvalue(lv: &GpuLValue, out: &mut Vec<&'static GpuHelperFnDescriptor>) {
    match lv {
        GpuLValue::Var(_) => {}
        GpuLValue::Field { base, .. } => collect_helpers_in_lvalue(base, out),
        GpuLValue::Index { base, index } => {
            collect_helpers_in_lvalue(base, out);
            collect_helpers_in_expr(index, out);
        }
    }
}

fn collect_helpers_in_expr(expr: &GpuExpr, out: &mut Vec<&'static GpuHelperFnDescriptor>) {
    match expr {
        GpuExpr::Var(_) | GpuExpr::Lit(_) => {}
        GpuExpr::ArrayInit(elems) | GpuExpr::VecConstruct { elements: elems, .. } => {
            for e in elems {
                collect_helpers_in_expr(e, out);
            }
        }
        GpuExpr::ArrayRepeat { elem, .. }
        | GpuExpr::VecToArray(elem)
        | GpuExpr::Field { base: elem, .. }
        | GpuExpr::Unary { expr: elem, .. }
        | GpuExpr::Paren(elem)
        | GpuExpr::Cast { expr: elem, .. } => collect_helpers_in_expr(elem, out),
        GpuExpr::Index { base, index } => {
            collect_helpers_in_expr(base, out);
            collect_helpers_in_expr(index, out);
        }
        GpuExpr::Binary { lhs, rhs, .. } => {
            collect_helpers_in_expr(lhs, out);
            collect_helpers_in_expr(rhs, out);
        }
        GpuExpr::MathCall { receiver, args, .. } => {
            collect_helpers_in_expr(receiver, out);
            for a in args {
                collect_helpers_in_expr(a, out);
            }
        }
        GpuExpr::HelperCall { helper, args } => {
            for a in args {
                collect_helpers_in_expr(a, out);
            }
            if !out.iter().any(|h| std::ptr::eq(*h, *helper)) {
                collect_helpers_topological(&helper.body, out);
                if !out.iter().any(|h| h.name == helper.name) {
                    out.push(helper);
                }
            }
        }
    }
}

/// 收集 `GpuBlock` 中被修改（作为赋值目标或以 `by_mut_ref` 传参）的根变量名集合。
pub fn collect_written_variables(block: &GpuBlock) -> std::collections::HashSet<&'static str> {
    let mut written = std::collections::HashSet::new();
    for stmt in &block.stmts {
        collect_written_in_stmt(stmt, &mut written);
    }
    written
}

fn root_var_of_lvalue(lv: &GpuLValue) -> Option<&'static str> {
    match lv {
        GpuLValue::Var(name) => Some(*name),
        GpuLValue::Field { base, .. } => root_var_of_lvalue(base),
        GpuLValue::Index { base, .. } => root_var_of_lvalue(base),
    }
}

fn root_var_of_expr(expr: &GpuExpr) -> Option<&'static str> {
    match expr {
        GpuExpr::Var(name) => Some(*name),
        GpuExpr::Field { base, .. } => root_var_of_expr(base),
        GpuExpr::Index { base, .. } => root_var_of_expr(base),
        _ => None,
    }
}

fn collect_written_in_stmt(stmt: &GpuStmt, out: &mut std::collections::HashSet<&'static str>) {
    match stmt {
        GpuStmt::Assign { target, value, .. } => {
            if let Some(name) = root_var_of_lvalue(target) {
                out.insert(name);
            }
            collect_written_in_expr(value, out);
        }
        GpuStmt::Let { init, .. } | GpuStmt::Expr(init) => {
            collect_written_in_expr(init, out);
        }
        GpuStmt::If {
            cond,
            then_block,
            else_branch,
        } => {
            collect_written_in_expr(cond, out);
            for s in &then_block.stmts {
                collect_written_in_stmt(s, out);
            }
            if let Some(eb) = else_branch {
                match eb {
                    GpuElseBranch::ElseIf(s) => collect_written_in_stmt(s, out),
                    GpuElseBranch::ElseBlock(b) => {
                        for s in &b.stmts {
                            collect_written_in_stmt(s, out);
                        }
                    }
                }
            }
        }
        GpuStmt::While { cond, body } => {
            collect_written_in_expr(cond, out);
            for s in &body.stmts {
                collect_written_in_stmt(s, out);
            }
        }
        GpuStmt::Loop(body) | GpuStmt::Block(body) => {
            for s in &body.stmts {
                collect_written_in_stmt(s, out);
            }
        }
        GpuStmt::ForRange {
            start, end, body, ..
        } => {
            collect_written_in_expr(start, out);
            collect_written_in_expr(end, out);
            for s in &body.stmts {
                collect_written_in_stmt(s, out);
            }
        }
        GpuStmt::Return(Some(e)) => collect_written_in_expr(e, out),
        GpuStmt::Return(None) | GpuStmt::Break | GpuStmt::Continue => {}
    }
}

fn collect_written_in_expr(expr: &GpuExpr, out: &mut std::collections::HashSet<&'static str>) {
    match expr {
        GpuExpr::Var(_) | GpuExpr::Lit(_) => {}
        GpuExpr::ArrayInit(elems) | GpuExpr::VecConstruct { elements: elems, .. } => {
            for e in elems {
                collect_written_in_expr(e, out);
            }
        }
        GpuExpr::ArrayRepeat { elem, .. }
        | GpuExpr::VecToArray(elem)
        | GpuExpr::Field { base: elem, .. }
        | GpuExpr::Unary { expr: elem, .. }
        | GpuExpr::Paren(elem)
        | GpuExpr::Cast { expr: elem, .. } => collect_written_in_expr(elem, out),
        GpuExpr::Index { base, index } => {
            collect_written_in_expr(base, out);
            collect_written_in_expr(index, out);
        }
        GpuExpr::Binary { lhs, rhs, .. } => {
            collect_written_in_expr(lhs, out);
            collect_written_in_expr(rhs, out);
        }
        GpuExpr::MathCall { receiver, args, .. } => {
            collect_written_in_expr(receiver, out);
            for a in args {
                collect_written_in_expr(a, out);
            }
        }
        GpuExpr::HelperCall { helper, args } => {
            for (idx, p) in helper.params.iter().enumerate() {
                if idx < args.len() {
                    if p.by_mut_ref
                        && let Some(name) = root_var_of_expr(&args[idx])
                    {
                        out.insert(name);
                    }
                    collect_written_in_expr(&args[idx], out);
                }
            }
        }
    }
}

/// 从强类型 [`GpuBlock`] IR 与组件 Schema 元数据一次性编译生成完整的 MSL 2.0 与 WGSL 内核源码。
pub fn compile_kernel_ir_sources(
    kernel_name: &str,
    components: &[GpuComponentSchema],
    msl_struct_defs: &[String],
    wgsl_struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    ir_block: &GpuBlock,
) -> (String, String) {
    compile_kernel_ir_sources_with_uniforms(
        kernel_name,
        components,
        &[],
        msl_struct_defs,
        wgsl_struct_defs,
        bindings,
        ir_block,
    )
}

/// 从强类型 [`GpuBlock`] IR、组件 Schema 元数据与动态标量 Uniform 参数规格编译生成完整的 MSL 2.0 与 WGSL 内核源码。
pub fn compile_kernel_ir_sources_with_uniforms(
    kernel_name: &str,
    components: &[GpuComponentSchema],
    scalar_uniforms: &[GpuScalarUniformSpec],
    msl_struct_defs: &[String],
    wgsl_struct_defs: &[String],
    bindings: &[GpuKernelBindingSpec],
    ir_block: &GpuBlock,
) -> (String, String) {
    let mut helpers = Vec::new();
    collect_helpers_topological(ir_block, &mut helpers);

    let written_vars = collect_written_variables(ir_block);

    let mut_params: Vec<&'static str> = components
        .iter()
        .filter(|c| c.mutable && written_vars.contains(c.param_name))
        .map(|c| c.param_name)
        .collect();

    let msl_emitter = IrCodeEmitter::new(IrBackendTarget::Msl);
    let wgsl_emitter = IrCodeEmitter::new(IrBackendTarget::Wgsl);

    // Collect any extra struct definitions referenced only by helper functions
    let mut extra_msl_structs = String::new();
    let mut extra_wgsl_structs = String::new();
    let mut seen_extra_structs: Vec<&'static str> =
        bindings.iter().map(|b| b.struct_name).collect();
    for h in &helpers {
        for p in &h.params {
            if let GpuHelperParamKind::Struct {
                struct_name,
                msl_def,
                wgsl_def,
                ..
            } = p.kind
                && !seen_extra_structs.contains(&struct_name)
            {
                seen_extra_structs.push(struct_name);
                extra_msl_structs.push_str(&msl_def());
                extra_msl_structs.push('\n');
                extra_wgsl_structs.push_str(&wgsl_def());
                extra_wgsl_structs.push('\n');
            }
        }
    }

    let mut msl_helpers = extra_msl_structs;
    let mut wgsl_helpers = extra_wgsl_structs;
    for h in &helpers {
        msl_helpers.push_str(&msl_emitter.emit_helper_fn(h));
        msl_helpers.push('\n');
        wgsl_helpers.push_str(&wgsl_emitter.emit_helper_fn(h));
        wgsl_helpers.push('\n');
    }

    let mut msl_env = ShaderTypeEnv::for_kernel(components, scalar_uniforms, &mut_params);
    let mut wgsl_env = ShaderTypeEnv::for_kernel(components, scalar_uniforms, &mut_params);

    let mut msl_prelude = String::new();
    let mut wgsl_prelude = String::new();
    for u in scalar_uniforms {
        let slot = u.slot;
        let pname = u.param_name;
        let msl_kw = if u.mutable { "" } else { "const " };
        let wgsl_kw = if u.mutable { "var" } else { "let" };
        match u.kind {
            GpuScalarKind::F32 => {
                msl_prelude.push_str(&format!(
                    "{msl_kw}float {pname} = as_type<float>(uniforms.scalar_arg_{slot}); "
                ));
                wgsl_prelude.push_str(&format!(
                    "{wgsl_kw} {pname}: f32 = bitcast<f32>(uniforms.scalar_arg_{slot}); "
                ));
            }
            GpuScalarKind::I32 => {
                msl_prelude.push_str(&format!(
                    "{msl_kw}int {pname} = as_type<int>(uniforms.scalar_arg_{slot}); "
                ));
                wgsl_prelude.push_str(&format!(
                    "{wgsl_kw} {pname}: i32 = bitcast<i32>(uniforms.scalar_arg_{slot}); "
                ));
            }
            GpuScalarKind::U32 => {
                msl_prelude.push_str(&format!(
                    "{msl_kw}uint {pname} = uniforms.scalar_arg_{slot}; "
                ));
                wgsl_prelude.push_str(&format!(
                    "{wgsl_kw} {pname}: u32 = uniforms.scalar_arg_{slot}; "
                ));
            }
            GpuScalarKind::Bool => {
                msl_prelude.push_str(&format!(
                    "{msl_kw}bool {pname} = (uniforms.scalar_arg_{slot} != 0u); "
                ));
                wgsl_prelude.push_str(&format!(
                    "{wgsl_kw} {pname}: bool = (uniforms.scalar_arg_{slot} != 0u); "
                ));
            }
        }
    }

    let msl_body = format!(
        "{msl_prelude}{}",
        msl_emitter.emit_block(ir_block, &mut msl_env)
    );
    let wgsl_body = format!(
        "{wgsl_prelude}{}",
        wgsl_emitter.emit_block(ir_block, &mut wgsl_env)
    );

    let mut effective_bindings: Vec<GpuKernelBindingSpec> = bindings.to_vec();
    for b in &mut effective_bindings {
        if b.mutable && !written_vars.contains(b.param_name) {
            b.mutable = false;
        }
    }

    let msl_source = super::kernel::build_msl_kernel_source_with_helpers(
        kernel_name,
        msl_struct_defs,
        &effective_bindings,
        &msl_helpers,
        &msl_body,
    );
    let wgsl_source = super::kernel::build_wgsl_kernel_source_with_helpers(
        kernel_name,
        wgsl_struct_defs,
        &effective_bindings,
        &wgsl_helpers,
        &wgsl_body,
    );
    (msl_source, wgsl_source)
}

// -----------------------------------------------------------------------------
// CPU 端三层遍历融合：原生 SIMD 向量函数与扩展 Trait
// -----------------------------------------------------------------------------

/// 构造 2D 浮点向量 (`[f32; 2]`)
#[inline]
pub fn vec2(x: f32, y: f32) -> [f32; 2] {
    [x, y]
}

/// 构造 3D 浮点向量 (`[f32; 3]`)
#[inline]
pub fn vec3(x: f32, y: f32, z: f32) -> [f32; 3] {
    [x, y, z]
}

/// 构造 4D 浮点向量 (`[f32; 4]`)
#[inline]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> [f32; 4] {
    [x, y, z, w]
}

/// 计算两个向量的点积 (Dot Product)
#[inline]
pub fn dot<const N: usize>(a: [f32; N], b: [f32; N]) -> f32 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += a[i] * b[i];
    }
    sum
}

/// 计算向量的欧几里得长度 (Length / Norm)
#[inline]
pub fn length<const N: usize>(a: [f32; N]) -> f32 {
    dot(a, a).sqrt()
}

/// 计算两个向量之间的欧氏距离 (Distance)
#[inline]
pub fn distance<const N: usize>(a: [f32; N], b: [f32; N]) -> f32 {
    let mut diff = [0.0; N];
    for i in 0..N {
        diff[i] = a[i] - b[i];
    }
    length(diff)
}

/// 归一化向量为单位向量 (Normalize)
#[inline]
pub fn normalize<const N: usize>(a: [f32; N]) -> [f32; N] {
    let len = length(a);
    if len > 0.0 {
        let mut res = [0.0; N];
        for i in 0..N {
            res[i] = a[i] / len;
        }
        res
    } else {
        a
    }
}

/// 为定长数组扩展的向量几何与数学方法 Trait，对齐 CPU 单线程与多核 Rayon 遍历。
pub trait GpuVectorExt<const N: usize>: Sized + Copy {
    fn dot(self, other: Self) -> f32;
    fn length(self) -> f32;
    fn normalize(self) -> Self;
    fn distance(self, other: Self) -> f32;
}

impl<const N: usize> GpuVectorExt<N> for [f32; N] {
    #[inline]
    fn dot(self, other: Self) -> f32 {
        dot(self, other)
    }
    #[inline]
    fn length(self) -> f32 {
        length(self)
    }
    #[inline]
    fn normalize(self) -> Self {
        normalize(self)
    }
    #[inline]
    fn distance(self, other: Self) -> f32 {
        distance(self, other)
    }
}

// -----------------------------------------------------------------------------
// 一等公民 `Vec2` / `Vec3` / `Vec4` POD 向量类型与算术运算符重载
// -----------------------------------------------------------------------------

/// 2D 连续内存浮点向量（与 `[f32; 2]`、Apple MSL `packed_float2` 及 WebGPU WGSL `array<f32, 2>` 100% 二进制布局兼容）。
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    #[inline(always)]
    pub const fn from_array(arr: [f32; 2]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
        }
    }

    #[inline(always)]
    pub const fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 { self / len } else { self }
    }

    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }
}

impl std::ops::Deref for Vec2 {
    type Target = [f32; 2];
    #[inline(always)]
    fn deref(&self) -> &[f32; 2] {
        bytemuck::cast_ref(self)
    }
}

impl std::ops::DerefMut for Vec2 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [f32; 2] {
        bytemuck::cast_mut(self)
    }
}

impl From<[f32; 2]> for Vec2 {
    #[inline(always)]
    fn from(arr: [f32; 2]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec2> for [f32; 2] {
    #[inline(always)]
    fn from(v: Vec2) -> Self {
        v.to_array()
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Add<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.x + rhs, self.y + rhs)
    }
}

impl std::ops::Add<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self + rhs.x, self + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Sub<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.x - rhs, self.y - rhs)
    }
}

impl std::ops::Sub<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self - rhs.x, self - rhs.y)
    }
}

impl std::ops::Mul for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self * rhs.x, self * rhs.y)
    }
}

impl std::ops::Div for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl std::ops::Div<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn div(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self / rhs.x, self / rhs.y)
    }
}

impl std::ops::AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::AddAssign<f32> for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
    }
}

impl std::ops::SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl std::ops::SubAssign<f32> for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
    }
}

impl std::ops::MulAssign for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl std::ops::MulAssign<f32> for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl std::ops::DivAssign for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl std::ops::DivAssign<f32> for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

/// 3D 连续内存浮点向量（与 `[f32; 3]`、Apple MSL `packed_float3` 及 WebGPU WGSL `array<f32, 3>` 100% 二进制布局兼容）。
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline(always)]
    pub const fn from_array(arr: [f32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }

    #[inline(always)]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 { self / len } else { self }
    }

    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }
}

impl std::ops::Deref for Vec3 {
    type Target = [f32; 3];
    #[inline(always)]
    fn deref(&self) -> &[f32; 3] {
        bytemuck::cast_ref(self)
    }
}

impl std::ops::DerefMut for Vec3 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [f32; 3] {
        bytemuck::cast_mut(self)
    }
}

impl From<[f32; 3]> for Vec3 {
    #[inline(always)]
    fn from(arr: [f32; 3]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec3> for [f32; 3] {
    #[inline(always)]
    fn from(v: Vec3) -> Self {
        v.to_array()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Add<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl std::ops::Add<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self + rhs.x, self + rhs.y, self + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Sub<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl std::ops::Sub<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self - rhs.x, self - rhs.y, self - rhs.z)
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl std::ops::Div for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl std::ops::Div<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn div(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::AddAssign<f32> for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl std::ops::SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl std::ops::SubAssign<f32> for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

impl std::ops::MulAssign for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl std::ops::MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl std::ops::DivAssign for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// 4D 连续内存浮点向量（与 `[f32; 4]`、Apple MSL `packed_float4` 及 WebGPU WGSL `array<f32, 4>` 100% 二进制布局兼容）。
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }

    #[inline(always)]
    pub const fn from_array(arr: [f32; 4]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }

    #[inline(always)]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 { self / len } else { self }
    }

    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }
}

impl std::ops::Deref for Vec4 {
    type Target = [f32; 4];
    #[inline(always)]
    fn deref(&self) -> &[f32; 4] {
        bytemuck::cast_ref(self)
    }
}

impl std::ops::DerefMut for Vec4 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [f32; 4] {
        bytemuck::cast_mut(self)
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline(always)]
    fn from(arr: [f32; 4]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec4> for [f32; 4] {
    #[inline(always)]
    fn from(v: Vec4) -> Self {
        v.to_array()
    }
}

impl std::ops::Add for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl std::ops::Add<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs, self.w + rhs)
    }
}

impl std::ops::Add<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn add(self, rhs: Vec4) -> Vec4 {
        Vec4::new(self + rhs.x, self + rhs.y, self + rhs.z, self + rhs.w)
    }
}

impl std::ops::Sub for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl std::ops::Sub<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs, self.w - rhs)
    }
}

impl std::ops::Sub<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn sub(self, rhs: Vec4) -> Vec4 {
        Vec4::new(self - rhs.x, self - rhs.y, self - rhs.z, self - rhs.w)
    }
}

impl std::ops::Mul for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
            self.w * rhs.w,
        )
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl std::ops::Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4::new(self * rhs.x, self * rhs.y, self * rhs.z, self * rhs.w)
    }
}

impl std::ops::Div for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
            self.w / rhs.w,
        )
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl std::ops::Div<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn div(self, rhs: Vec4) -> Vec4 {
        Vec4::new(self / rhs.x, self / rhs.y, self / rhs.z, self / rhs.w)
    }
}

impl std::ops::AddAssign for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl std::ops::AddAssign<f32> for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
        self.w += rhs;
    }
}

impl std::ops::SubAssign for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl std::ops::SubAssign<f32> for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
        self.w -= rhs;
    }
}

impl std::ops::MulAssign for Vec4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
        self.w *= rhs.w;
    }
}

impl std::ops::MulAssign<f32> for Vec4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl std::ops::DivAssign for Vec4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        self.w /= rhs.w;
    }
}

impl std::ops::DivAssign<f32> for Vec4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

impl std::ops::Neg for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl GpuConstValue for Vec2 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::vec2(
            GpuExpr::Lit(GpuLit::F32(self.x)),
            GpuExpr::Lit(GpuLit::F32(self.y)),
        )
    }
}

impl GpuConstValue for Vec3 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::vec3(
            GpuExpr::Lit(GpuLit::F32(self.x)),
            GpuExpr::Lit(GpuLit::F32(self.y)),
            GpuExpr::Lit(GpuLit::F32(self.z)),
        )
    }
}

impl GpuConstValue for Vec4 {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::vec4(
            GpuExpr::Lit(GpuLit::F32(self.x)),
            GpuExpr::Lit(GpuLit::F32(self.y)),
            GpuExpr::Lit(GpuLit::F32(self.z)),
            GpuExpr::Lit(GpuLit::F32(self.w)),
        )
    }
}

impl GpuConstValue for [f32; 2] {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::vec2(
            GpuExpr::Lit(GpuLit::F32(self[0])),
            GpuExpr::Lit(GpuLit::F32(self[1])),
        )
    }
}

impl GpuConstValue for [f32; 3] {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::vec3(
            GpuExpr::Lit(GpuLit::F32(self[0])),
            GpuExpr::Lit(GpuLit::F32(self[1])),
            GpuExpr::Lit(GpuLit::F32(self[2])),
        )
    }
}

impl GpuConstValue for [f32; 4] {
    #[inline]
    fn to_gpu_expr(&self) -> GpuExpr {
        GpuExpr::vec4(
            GpuExpr::Lit(GpuLit::F32(self[0])),
            GpuExpr::Lit(GpuLit::F32(self[1])),
            GpuExpr::Lit(GpuLit::F32(self[2])),
            GpuExpr::Lit(GpuLit::F32(self[3])),
        )
    }
}


