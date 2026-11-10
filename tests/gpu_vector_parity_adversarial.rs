#![cfg(feature = "gpu")]

use sumeru::storage::gpu::ir::{
    distance, dot, length, normalize, vec3, GpuBlock, GpuComponentSchema, GpuExpr, GpuLit,
    GpuMathIntrinsic, GpuScalarKind, GpuStmt, GpuValueType,
};
use sumeru::storage::gpu::kernel::{GpuKernelBindingSpec, GpuKernelDescriptor};
use sumeru::storage::gpu::metal::compile_and_validate_pipeline;
use sumeru::{
    gpu, EntityId, GpuShaderStruct, LayoutOptimization, MetalComputeEngine, Store,
};

#[gpu]
pub struct VectorState {
    pub v: [f32; 3],
    pub out_norm: [f32; 3],
    pub out_dot: f32,
    pub out_len: f32,
    pub out_dist: f32,
    pub out_add: [f32; 3],
    pub out_sub: [f32; 3],
    pub out_scaled: [f32; 3],
}

#[gpu]
pub struct VectorParam {
    pub dest_v: [f32; 3],
    pub scale: f32,
}

#[gpu]
pub fn kernel_vector_all_ops(state: &mut VectorState, param: &VectorParam) {
    let v = vec3(state.v[0], state.v[1], state.v[2]);
    let dest_v = vec3(param.dest_v[0], param.dest_v[1], param.dest_v[2]);

    // Intrinsics
    state.out_len = length(v);
    let norm = normalize(v);
    state.out_norm[0] = norm[0];
    state.out_norm[1] = norm[1];
    state.out_norm[2] = norm[2];

    state.out_dot = dot(v, dest_v);
    state.out_dist = distance(v, dest_v);

    // Arithmetic
    state.out_add[0] = v[0] + dest_v[0];
    state.out_add[1] = v[1] + dest_v[1];
    state.out_add[2] = v[2] + dest_v[2];

    state.out_sub[0] = v[0] - dest_v[0];
    state.out_sub[1] = v[1] - dest_v[1];
    state.out_sub[2] = v[2] - dest_v[2];

    state.out_scaled[0] = v[0] * param.scale;
    state.out_scaled[1] = v[1] * param.scale;
    state.out_scaled[2] = v[2] * param.scale;
}

/// 1. Naga 静态验证全覆盖测试：涵盖 2D/3D/4D 向量运算、算术二元算子、几何内建函数
#[test]
fn test_naga_exhaustive_vector_shader_validation() {
    let block = GpuBlock {
        stmts: vec![
            // 2D 向量构建与运算
            GpuStmt::Let {
                name: "v2a",
                mutable: false,
                explicit_ty: Some(GpuValueType::Vector(GpuScalarKind::F32, 2)),
                init: GpuExpr::vec2(GpuExpr::Lit(GpuLit::F32(3.0)), GpuExpr::Lit(GpuLit::F32(4.0))),
            },
            GpuStmt::Let {
                name: "v2b",
                mutable: false,
                explicit_ty: Some(GpuValueType::Vector(GpuScalarKind::F32, 2)),
                init: GpuExpr::vec2(GpuExpr::Lit(GpuLit::F32(1.0)), GpuExpr::Lit(GpuLit::F32(2.0))),
            },
            GpuStmt::Let {
                name: "v2_add",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: sumeru::storage::gpu::ir::GpuBinaryOp::Add,
                    lhs: Box::new(GpuExpr::Var("v2a")),
                    rhs: Box::new(GpuExpr::Var("v2b")),
                },
            },
            GpuStmt::Let {
                name: "v2_sub",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: sumeru::storage::gpu::ir::GpuBinaryOp::Sub,
                    lhs: Box::new(GpuExpr::Var("v2a")),
                    rhs: Box::new(GpuExpr::Var("v2b")),
                },
            },
            GpuStmt::Let {
                name: "v2_mul_scalar",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: sumeru::storage::gpu::ir::GpuBinaryOp::Mul,
                    lhs: Box::new(GpuExpr::Var("v2_add")),
                    rhs: Box::new(GpuExpr::Lit(GpuLit::F32(2.5))),
                },
            },
            GpuStmt::Let {
                name: "v2_vec_mul",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: sumeru::storage::gpu::ir::GpuBinaryOp::Mul,
                    lhs: Box::new(GpuExpr::Var("v2a")),
                    rhs: Box::new(GpuExpr::Var("v2b")),
                },
            },
            GpuStmt::Let {
                name: "v2_dot",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v2a")),
                    intrinsic: GpuMathIntrinsic::Dot,
                    args: vec![GpuExpr::Var("v2b")],
                },
            },
            GpuStmt::Let {
                name: "v2_len",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v2_vec_mul")),
                    intrinsic: GpuMathIntrinsic::Length,
                    args: vec![],
                },
            },
            GpuStmt::Let {
                name: "v2_dist",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v2a")),
                    intrinsic: GpuMathIntrinsic::Distance,
                    args: vec![GpuExpr::Var("v2b")],
                },
            },
            // 3D 向量构建与定长数组转换
            GpuStmt::Let {
                name: "v3a",
                mutable: false,
                explicit_ty: Some(GpuValueType::Vector(GpuScalarKind::F32, 3)),
                init: GpuExpr::vec3(
                    GpuExpr::Lit(GpuLit::F32(-1.0)),
                    GpuExpr::Lit(GpuLit::F32(2.0)),
                    GpuExpr::Lit(GpuLit::F32(-3.0)),
                ),
            },
            GpuStmt::Let {
                name: "v3_norm",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v3a")),
                    intrinsic: GpuMathIntrinsic::Normalize,
                    args: vec![],
                },
            },
            GpuStmt::Let {
                name: "v3_arr",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::VecToArray(Box::new(GpuExpr::Var("v3_norm"))),
            },
            // 4D 向量构建与运算
            GpuStmt::Let {
                name: "v4a",
                mutable: false,
                explicit_ty: Some(GpuValueType::Vector(GpuScalarKind::F32, 4)),
                init: GpuExpr::vec4(
                    GpuExpr::Lit(GpuLit::F32(1.0)),
                    GpuExpr::Lit(GpuLit::F32(2.0)),
                    GpuExpr::Lit(GpuLit::F32(3.0)),
                    GpuExpr::Lit(GpuLit::F32(4.0)),
                ),
            },
            GpuStmt::Let {
                name: "v4_norm",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v4a")),
                    intrinsic: GpuMathIntrinsic::Normalize,
                    args: vec![],
                },
            },
            GpuStmt::Assign {
                op: sumeru::storage::gpu::ir::GpuAssignOp::Assign,
                target: sumeru::storage::gpu::ir::GpuLValue::Field {
                    base: Box::new(sumeru::storage::gpu::ir::GpuLValue::Var("state")),
                    field: "out_len",
                },
                value: GpuExpr::Var("v2_len"),
            },
        ],
    };

    let comp = GpuComponentSchema {
        param_name: "state",
        struct_name: <VectorState as GpuShaderStruct>::STRUCT_NAME,
        fields: <VectorState as GpuShaderStruct>::FIELDS,
        mutable: true,
    };
    let binding = GpuKernelBindingSpec {
        param_name: "state",
        struct_name: <VectorState as GpuShaderStruct>::STRUCT_NAME,
        type_id: sumeru::ComponentTypeId::of::<VectorState>(),
        elem_size: std::mem::size_of::<VectorState>(),
        mutable: true,
    };

    let (msl, wgsl) = sumeru::storage::gpu::ir::compile_kernel_ir_sources(
        "test_naga_vector_all_exhaustive",
        &[comp],
        &[<VectorState as GpuShaderStruct>::msl_struct_definition()],
        &[<VectorState as GpuShaderStruct>::wgsl_struct_definition()],
        &[binding],
        &block,
    );

    unsafe fn cpu_dummy(_: &[*mut u8], _: usize, _: &sumeru::GpuDispatchUniforms) {}

    let desc = unsafe {
        GpuKernelDescriptor::new(
            "test_naga_vector_all_exhaustive",
            vec![binding],
            msl,
            wgsl,
            cpu_dummy,
        )
    };

    let compiled = compile_and_validate_pipeline(&desc)
        .expect("Naga validation MUST strictly pass for all 2D/3D/4D vector arithmetic and geometric intrinsics");
    assert!(compiled.shader_validated);
}

/// 2. 三层遍历契约与非对齐边界行数遍历测试
#[test]
fn test_three_tier_vector_math_parity_and_edge_cases() {
    let mut engine = MetalComputeEngine::new();
    let has_gpu = engine.has_hardware_gpu();
    println!("Hardware GPU available: {has_gpu}");

    let test_cases = [
        ([0.0f32, 0.0, 0.0], [1.0f32, 2.0, 3.0], 2.0f32),
        ([-3.0, -4.0, 0.0], [3.0, 4.0, 0.0], -1.5),
        ([1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], 0.25),
        ([1e7, 2e7, 3e7], [1e7, 0.0, -1e7], 0.5),
        ([1e-5, 2e-5, -3e-5], [4e-5, -5e-5, 6e-5], 3.0),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([100.0, 200.0, 300.0], [-10.0, -20.0, -30.0], 10.0),
        ([0.0, 5.0, 0.0], [0.0, -5.0, 0.0], -1.0),
    ];

    let row_counts = [
        0, 1, 16, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 1365, 1366, 2048,
        2049, 4095, 4096, 4097, 8192, 16384,
    ];

    for &num_rows in &row_counts {
        println!("Testing row count: {num_rows}...");

        let mut store_t1 = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
        for i in 0..num_rows {
            let e = EntityId::from_u128(i as u128 + 1);
            let case = &test_cases[i % test_cases.len()];
            store_t1
                .insert((
                    e,
                    (
                        VectorState {
                            v: case.0,
                            out_norm: [0.0; 3],
                            out_dot: 0.0,
                            out_len: 0.0,
                            out_dist: 0.0,
                            out_add: [0.0; 3],
                            out_sub: [0.0; 3],
                            out_scaled: [0.0; 3],
                        },
                        VectorParam {
                            dest_v: case.1,
                            scale: case.2,
                        },
                    ),
                ))
                .unwrap();
        }

        let mut store_t2 = store_t1.clone();
        let mut store_t3 = store_t1.clone();
        let mut store_t3_pass = store_t1.clone();

        // Tier 1: CPU Single-threaded
        store_t1
            .query_mut::<(&mut VectorState, &VectorParam)>()
            .unwrap()
            .run(kernel_vector_all_ops);

        // Tier 2: CPU Rayon Morsel Parallel
        store_t2
            .query_mut::<(&mut VectorState, &VectorParam)>()
            .unwrap()
            .par_run(kernel_vector_all_ops);

        // Tier 3: GPU Single Query Execution
        store_t3
            .query_mut::<(&mut VectorState, &VectorParam)>()
            .unwrap()
            .gpu()
            .for_each_on(&mut engine, kernel_vector_all_ops)
            .unwrap();

        // Tier 3+: Multi-Kernel Pass
        let mut pass = store_t3_pass.gpu_compute_pass(&mut engine).unwrap();
        pass.dispatch(kernel_vector_all_ops).unwrap();
        pass.finish().unwrap();

        // Verification across all rows
        for i in 0..num_rows {
            let e = EntityId::from_u128(i as u128 + 1);
            let s1 = *store_t1.get::<VectorState>(e).unwrap();
            let s2 = *store_t2.get::<VectorState>(e).unwrap();
            let s3 = *store_t3.get::<VectorState>(e).unwrap();
            let s3p = *store_t3_pass.get::<VectorState>(e).unwrap();

            let is_zero_vec = s1.v == [0.0, 0.0, 0.0];

            // 1. Tier 1 vs Tier 2 (CPU Single vs Rayon Parallel) - 必须严格 100% 位一致
            assert_eq!(s1.out_len, s2.out_len, "T1 vs T2 out_len mismatch at row {i}");
            assert_eq!(s1.out_dot, s2.out_dot, "T1 vs T2 out_dot mismatch at row {i}");
            assert_eq!(s1.out_dist, s2.out_dist, "T1 vs T2 out_dist mismatch at row {i}");
            assert_eq!(s1.out_add, s2.out_add, "T1 vs T2 out_add mismatch at row {i}");
            assert_eq!(s1.out_sub, s2.out_sub, "T1 vs T2 out_sub mismatch at row {i}");
            assert_eq!(s1.out_scaled, s2.out_scaled, "T1 vs T2 out_scaled mismatch at row {i}");
            assert_eq!(s1.out_norm, s2.out_norm, "T1 vs T2 out_norm mismatch at row {i}");

            // 2. Tier 3 vs Tier 3+ (GPU Single vs GpuComputePass) - 必须严格 100% 一致
            assert_eq!(s3.out_len, s3p.out_len, "T3 vs T3+ out_len mismatch at row {i}");
            assert_eq!(s3.out_dot, s3p.out_dot, "T3 vs T3+ out_dot mismatch at row {i}");
            assert_eq!(s3.out_dist, s3p.out_dist, "T3 vs T3+ out_dist mismatch at row {i}");
            assert_eq!(s3.out_add, s3p.out_add, "T3 vs T3+ out_add mismatch at row {i}");
            assert_eq!(s3.out_sub, s3p.out_sub, "T3 vs T3+ out_sub mismatch at row {i}");
            assert_eq!(s3.out_scaled, s3p.out_scaled, "T3 vs T3+ out_scaled mismatch at row {i}");

            if is_zero_vec {
                assert!(s3.out_norm[0].is_nan() && s3p.out_norm[0].is_nan());
                assert!(s3.out_norm[1].is_nan() && s3p.out_norm[1].is_nan());
                assert!(s3.out_norm[2].is_nan() && s3p.out_norm[2].is_nan());
            } else {
                assert_eq!(s3.out_norm, s3p.out_norm, "T3 vs T3+ out_norm mismatch at row {i}");
            }

            // 3. CPU (T1) vs GPU (T3) 数值精度对比
            for axis in 0..3 {
                let diff_add = (s1.out_add[axis] - s3.out_add[axis]).abs();
                let rel_add = diff_add / (s1.out_add[axis].abs().max(1.0));
                assert!(rel_add < 1e-4, "CPU vs GPU out_add[{axis}] mismatch at row {i}");

                let diff_sub = (s1.out_sub[axis] - s3.out_sub[axis]).abs();
                let rel_sub = diff_sub / (s1.out_sub[axis].abs().max(1.0));
                assert!(rel_sub < 1e-4, "CPU vs GPU out_sub[{axis}] mismatch at row {i}");

                let diff_scaled = (s1.out_scaled[axis] - s3.out_scaled[axis]).abs();
                let rel_scaled = diff_scaled / (s1.out_scaled[axis].abs().max(1.0));
                assert!(rel_scaled < 1e-4, "CPU vs GPU out_scaled[{axis}] mismatch at row {i}");
            }

            let diff_dot = (s1.out_dot - s3.out_dot).abs();
            let rel_dot = diff_dot / (s1.out_dot.abs().max(1.0));
            assert!(rel_dot < 1e-4, "CPU vs GPU out_dot mismatch at row {i}");

            let diff_len = (s1.out_len - s3.out_len).abs();
            let rel_len = diff_len / (s1.out_len.abs().max(1.0));
            assert!(rel_len < 1e-4, "CPU vs GPU out_len mismatch at row {i}");

            let diff_dist = (s1.out_dist - s3.out_dist).abs();
            let rel_dist = diff_dist / (s1.out_dist.abs().max(1.0));
            assert!(rel_dist < 1e-4, "CPU vs GPU out_dist mismatch at row {i}");

            if !is_zero_vec {
                for axis in 0..3 {
                    let diff_norm = (s1.out_norm[axis] - s3.out_norm[axis]).abs();
                    assert!(
                        diff_norm < 1e-4,
                        "CPU vs GPU out_norm[{axis}] mismatch at row {i}"
                    );
                }
            }
        }
    }
}

/// 3. 极端边缘用例压力测试：零向量归一化差异与数值溢出行为实证
#[test]
fn test_extreme_edge_cases_and_divergence_behavior() {
    let mut engine = MetalComputeEngine::new();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e_zero = EntityId::from_u128(1);
    let e_overflow = EntityId::from_u128(2);

    store
        .insert((
            e_zero,
            (
                VectorState {
                    v: [0.0, 0.0, 0.0],
                    out_norm: [0.0; 3],
                    out_dot: 0.0,
                    out_len: 0.0,
                    out_dist: 0.0,
                    out_add: [0.0; 3],
                    out_sub: [0.0; 3],
                    out_scaled: [0.0; 3],
                },
                VectorParam {
                    dest_v: [0.0, 0.0, 0.0],
                    scale: 1.0,
                },
            ),
        ))
        .unwrap();

    store
        .insert((
            e_overflow,
            (
                VectorState {
                    v: [1e20, 1e20, 1e20],
                    out_norm: [0.0; 3],
                    out_dot: 0.0,
                    out_len: 0.0,
                    out_dist: 0.0,
                    out_add: [0.0; 3],
                    out_sub: [0.0; 3],
                    out_scaled: [0.0; 3],
                },
                VectorParam {
                    dest_v: [1e20, 1e20, 1e20],
                    scale: 2.0,
                },
            ),
        ))
        .unwrap();

    let mut store_cpu = store.clone();
    let mut store_gpu = store.clone();

    // CPU execution
    store_cpu
        .query_mut::<(&mut VectorState, &VectorParam)>()
        .unwrap()
        .run(kernel_vector_all_ops);

    // GPU execution
    store_gpu
        .query_mut::<(&mut VectorState, &VectorParam)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, kernel_vector_all_ops)
        .unwrap();

    let cpu_zero = *store_cpu.get::<VectorState>(e_zero).unwrap();
    let gpu_zero = *store_gpu.get::<VectorState>(e_zero).unwrap();

    let cpu_overflow = *store_cpu.get::<VectorState>(e_overflow).unwrap();
    let gpu_overflow = *store_gpu.get::<VectorState>(e_overflow).unwrap();

    // 零向量归一化实证事实：
    // CPU 端有 `if len > 0.0 { a / len } else { a }` 防御保护，返回 [0.0, 0.0, 0.0]
    assert_eq!(cpu_zero.out_norm, [0.0, 0.0, 0.0]);
    // GPU 端因硬件 ALU 规范 `v * rsqrt(dot(v, v))` 导致 0 * Inf = NaN
    assert!(gpu_zero.out_norm[0].is_nan());
    assert!(gpu_zero.out_norm[1].is_nan());
    assert!(gpu_zero.out_norm[2].is_nan());

    // 溢出行为实证事实：1e20 点积超过 f32::MAX (3.4e38)，产生 +Inf
    assert!(cpu_overflow.out_dot.is_infinite());
    assert!(gpu_overflow.out_dot.is_infinite());
    assert!(cpu_overflow.out_len.is_infinite());
    assert!(gpu_overflow.out_len.is_infinite());
}
