//! 13. Typed Shader IR: External `const` Evaluation, `#[gpu]` Helper Composition, Dynamic Uniforms & Batch Dispatch
//!
//! Demonstrates Sumeru's Type-Driven Shader IR (`GpuBlock` / `GpuStmt` / `GpuExpr`):
//! 1. **External Rust `const` Constants (`GpuConstValue`)**: Directly reference module constants
//!    (`SPEED_LIMIT`) and standard library constants (`std::f32::consts::FRAC_PI_4`)
//!    inside `#[gpu]` functions — evaluated at compile time by `rustc`;
//! 2. **Composable `#[gpu]` Helpers (Scalar & `&mut` Component Refs)**: Define modular device functions
//!    (`soft_clamp_axis` -> `damped_substep_displacement` -> `step_position_in_place(pos: &mut SimPosition, ...)`),
//!    automatically linked in topological dependency order into both MSL and WGSL;
//! 3. **Full-Expression Bidirectional `u32` Inference (`ShaderTypeEnv`)**: Write `cfg.substeps += 1`,
//!    `if cfg.substeps > 0`, and `for _step in 0..cfg.substeps` with unsuffixed integer literals —
//!    `ShaderTypeEnv` inspects `SimStepConfig::FIELDS`, infers `cfg.substeps: u32`, and automatically
//!    promotes literals to `u32` (`0u`, `1u`);
//! 4. **Runtime Dynamic Scalar Uniforms (`.with_args(dt)`) & Single-`MTLCommandBuffer` Batch Dispatch**:
//!    Pass runtime scalar parameters (`dt: f32`) without recompiling shaders, and dispatch across
//!    multiple matched Archetypes in a single hardware `MTLCommandBuffer`.

use crate::common::*;

const SPEED_LIMIT: f32 = 8.0;

#[gpu]
pub struct SimStepConfig {
    pub damping: f32,
    pub substeps: u32,
}

#[gpu]
pub struct SimBatchTag {
    pub group: u32,
}

#[gpu]
pub fn soft_clamp_axis(val: f32, limit: f32) -> f32 {
    if val > limit {
        limit
    } else if val < -limit {
        -limit
    } else {
        val
    }
}

#[gpu]
pub fn damped_substep_displacement(vel_axis: f32, damping: f32, dt: f32) -> f32 {
    let clamped = soft_clamp_axis(vel_axis, SPEED_LIMIT);
    clamped * dt * damping
}

#[gpu]
pub fn step_position_in_place(
    pos: &mut SimPosition,
    vel: &SimVelocity,
    damping: f32,
    dt: f32,
    angle_bias: f32,
) {
    pos.x += damped_substep_displacement(vel.vx, damping, dt) + angle_bias;
    pos.y += damped_substep_displacement(vel.vy, damping, dt);
}

#[gpu]
pub fn integrate_typed_ir_substeps(
    pos: &mut SimPosition,
    vel: &SimVelocity,
    cfg: &mut SimStepConfig,
    dt: f32,
) {
    const BIAS_SCALE: f32 = 0.0;
    let angle_bias = std::f32::consts::FRAC_PI_4 * BIAS_SCALE;
    // Bidirectional u32 literal coercion: `1`, `0` automatically promoted to `1u`, `0u`
    cfg.substeps += 1;
    if cfg.substeps > 0 {
        for _step in 0..cfg.substeps {
            step_position_in_place(pos, vel, cfg.damping, dt, angle_bias);
        }
    }
}

pub fn main() {
    clear_log();

    let bound_kernel = integrate_typed_ir_substeps.with_args(0.25f32);
    let desc = bound_kernel.descriptor();
    let compiled = compile_and_validate_pipeline(desc).unwrap();
    let msl = desc.msl_source();
    let wgsl = desc.wgsl_source();

    // Verify topological order of linked helper functions (callee before caller)
    let wgsl_callee_idx = wgsl.find("fn soft_clamp_axis").unwrap();
    let wgsl_mid_idx = wgsl.find("fn damped_substep_displacement").unwrap();
    let wgsl_caller_idx = wgsl.find("fn step_position_in_place").unwrap();
    let topo_ordered = wgsl_callee_idx < wgsl_mid_idx
        && wgsl_mid_idx < wgsl_caller_idx
        && msl.find("inline float soft_clamp_axis").unwrap()
            < msl.find("inline float damped_substep_displacement").unwrap();
    let inferred_u32_loop = wgsl.contains("cfg.substeps += 1u;")
        && wgsl.contains("for (var _step = 0u; _step < cfg.substeps; _step += 1u)");

    let mut store_gpu =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e1 = EntityId::from_u128(1);
    let e2 = EntityId::from_u128(2);
    // Archetype 1: (SimPosition, SimVelocity, SimStepConfig)
    store_gpu
        .insert((
            e1,
            (
                SimPosition { x: 2.0, y: 10.0 },
                SimVelocity { vx: 20.0, vy: -4.0 }, // vx=20 clamped to 8.0; vy=-4 stays -4.0
                SimStepConfig {
                    damping: 0.5,
                    substeps: 3, // incremented to 4 in shader
                },
            ),
        ))
        .unwrap();
    // Archetype 2: (SimPosition, SimVelocity, SimStepConfig, SimBatchTag)
    store_gpu
        .insert((
            e2,
            (
                SimPosition { x: 2.0, y: 10.0 },
                SimVelocity { vx: 20.0, vy: -4.0 },
                SimStepConfig {
                    damping: 0.5,
                    substeps: 3,
                },
                SimBatchTag { group: 1 },
            ),
        ))
        .unwrap();
    let mut store_cpu = store_gpu.clone();

    let stats = store_gpu
        .query_mut::<(&mut SimPosition, &SimVelocity, &mut SimStepConfig)>()
        .unwrap()
        .gpu()
        .for_each(integrate_typed_ir_substeps.with_args(0.25f32))
        .unwrap();

    store_cpu
        .query_mut::<(&mut SimPosition, &SimVelocity, &mut SimStepConfig)>()
        .unwrap()
        .run(integrate_typed_ir_substeps.with_args(0.25f32));

    let p1_gpu = *store_gpu.get::<SimPosition>(e1).unwrap();
    let p1_cpu = *store_cpu.get::<SimPosition>(e1).unwrap();
    let p2_gpu = *store_gpu.get::<SimPosition>(e2).unwrap();
    assert_eq!(p1_gpu, p1_cpu);
    assert_eq!(p1_gpu, p2_gpu);
    assert_eq!(stats.dispatched_archetypes, 2);

    elog!(
        "typed_ir: validated={}, helpers_topo={}, u32_loop_inferred={}, archetypes={}",
        compiled.shader_validated,
        topo_ordered,
        inferred_u32_loop,
        stats.dispatched_archetypes
    );
    elog!("typed_ir_pos=({:.1}, {:.1})", p1_gpu.x, p1_gpu.y);
}

#[test]
fn test_gpu_typed_ir_consts_and_helpers() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "typed_ir: validated=true, helpers_topo=true, u32_loop_inferred=true, archetypes=2",
            "typed_ir_pos=(6.0, 8.0)",
        ]
    );
}
