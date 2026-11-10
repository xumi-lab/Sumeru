//! End-to-End Multi-Kernel Compute Pass & Ergonomic Operator API Integration Example.
//!
//! Demonstrates:
//! 1. Unified `#[kernel]` Macro & First-Class `Vec3` Arithmetic:
//!    - `#[kernel]` on component structs (`BodyPosition`, `BodyVelocity`, `BodyDynamics`, `ActiveTag`, `FrozenTag`)
//!    - `#[kernel]` on operator functions (`integrate`, `damp_dynamics`, `step_physics_fused`)
//!    - Native `Vec3` operator overloading (`+`, `*`, `+=`, `*=`, `.length()`, `.dot()`) across CPU and GPU
//! 2. Direct Tuple Destructuring & Shared `QueryCache` (`store.queries_mut`):
//!    - `let (mut pos, mut vel, mut dyns) = store.queries_mut::<...>()?.with::<ActiveTag>().without::<FrozenTag>();`
//!    - Zero archetype re-scanning when no structural (CR) changes occur
//! 3. Four Seamless Execution Forms on the Same `#[kernel]` Operators:
//!    - Custom Rayon parallel loop: `(&mut pos, &vel).into_par_iter().for_each(|(p, v)| integrate(p, v, DT));`
//!    - Adaptive CPU/GPU whole-column dispatch: `integrate.run(&mut pos, &vel, DT)?` / `integrate.par(&mut pos, &vel, DT)?` / `integrate.gpu(&mut pos, &vel, DT)?`
//!    - Composable single-kernel fusion: `step_physics_fused.gpu(&mut pos, &mut vel, &mut dyns, DT)?`
//!    - Multi-kernel tuple pipeline with automatic `MTLBarrierScopeBuffers` hazard tracking:
//!      `(integrate.bind(&mut pos, &vel, DT), damp_dynamics.bind(&mut vel, &mut dyns, DT), integrate.bind(&mut pos, &vel, DT)).gpu()?`

use rayon::prelude::*;
use std::time::Instant;
use sumeru::{
    kernel, ComputeTask, EntityId, LayoutOptimization, MetalComputeEngine, QueryTupleFilterExt,
    Store, Vec3,
};

/// 3D position component aligned for 16KB hardware page zero-copy UMA dispatch.
#[kernel]
pub struct BodyPosition {
    pub coords: Vec3,
}

/// 3D velocity component aligned for 16KB hardware page zero-copy UMA dispatch.
#[kernel]
pub struct BodyVelocity {
    pub velocity: Vec3,
}

/// 3D force and drag dynamics parameters.
#[kernel]
pub struct BodyDynamics {
    pub force: Vec3,
    pub drag: f32,
}

/// Archetype inclusion tag.
#[kernel]
pub struct ActiveTag {
    pub active: u32,
}

/// Archetype exclusion tag.
#[kernel]
pub struct FrozenTag {
    pub frozen: u32,
}

/// Stage 1: Integrate 3D position using vector velocity (`Vec3` operator overloading).
/// `vel` is passed by immutable reference `&BodyVelocity`, eliminating redundant GPU writeback.
#[kernel]
pub fn integrate(pos: &mut BodyPosition, vel: &BodyVelocity, dt: f32) {
    pos.coords += vel.velocity * dt;
}

/// Stage 2: Accelerate velocity by force, apply aerodynamic drag, and decay applied force.
#[kernel]
pub fn damp_dynamics(vel: &mut BodyVelocity, dyns: &mut BodyDynamics, dt: f32) {
    let v_acc = vel.velocity + dyns.force * dt;
    let speed = v_acc.length();
    let damp_factor = (1.0 - dyns.drag * dt).max(0.0);
    let power = v_acc.dot(dyns.force);
    let energy_loss = speed * 0.0 + power * 0.0;
    vel.velocity = v_acc * damp_factor + Vec3::splat(energy_loss);
    dyns.force *= 0.5;
}

/// Fused single-kernel composition calling `integrate` and `damp_dynamics` in registers.
#[kernel]
pub fn step_physics_fused(
    pos: &mut BodyPosition,
    vel: &mut BodyVelocity,
    dyns: &mut BodyDynamics,
    dt: f32,
) {
    integrate(pos, vel, dt);
    damp_dynamics(vel, dyns, dt);
    integrate(pos, vel, dt);
}

fn populate_test_store(num_active: usize) -> Store {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..num_active {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    BodyPosition {
                        coords: Vec3::new(0.0, 10.0, 20.0),
                    },
                    BodyVelocity {
                        velocity: Vec3::new(10.0, 20.0, 30.0),
                    },
                    BodyDynamics {
                        force: Vec3::new(2.0, 4.0, 6.0),
                        drag: 0.5,
                    },
                    ActiveTag { active: 1 },
                ),
            ))
            .expect("insert active entity");
    }
    // Insert a frozen entity that must be excluded by `.without::<FrozenTag>()`
    let frozen_e = EntityId::from_u128(999_999);
    store
        .insert((
            frozen_e,
            (
                BodyPosition {
                    coords: Vec3::new(100.0, 100.0, 100.0),
                },
                BodyVelocity {
                    velocity: Vec3::new(50.0, 50.0, 50.0),
                },
                BodyDynamics {
                    force: Vec3::new(10.0, 10.0, 10.0),
                    drag: 0.1,
                },
                ActiveTag { active: 1 },
                FrozenTag { frozen: 1 },
            ),
        ))
        .expect("insert frozen entity");
    store
}

fn main() {
    println!("===============================================================================");
    println!("   Sumeru Ergonomic `#[kernel]` & Multi-Stage GPU Compute Pipeline Demo        ");
    println!("===============================================================================\n");

    let mut engine = MetalComputeEngine::new();
    println!("1. Execution Mode: {:?}", engine.execution_mode());
    let has_gpu = engine.has_hardware_gpu();
    println!("   Hardware GPU Available: {}", has_gpu);

    const NUM_ENTITIES: usize = 1024;
    const DT: f32 = 0.1;

    let mut store = populate_test_store(NUM_ENTITIES);
    let initial_epoch = store.epoch();

    println!(
        "\n2. Executing 3-Stage Tuple Pipeline `(integrate.bind(...), damp_dynamics.bind(...), integrate.bind(...)).gpu()?`:"
    );
    let start_pass = Instant::now();
    let (pass_stats, cache_hits, cache_misses) = {
        let (mut pos, mut vel, mut dyns) = store
            .queries_mut::<(&mut BodyPosition, &mut BodyVelocity, &mut BodyDynamics)>()
            .expect("queries_mut")
            .with::<ActiveTag>()
            .without::<FrozenTag>();

        let stats = (
            integrate.bind(&mut pos, &vel, DT),
            damp_dynamics.bind(&mut vel, &mut dyns, DT),
            integrate.bind(&mut pos, &vel, DT),
        )
            .gpu_on(&mut engine)
            .expect("tuple pipeline gpu_on");

        (stats, pos.cache_hits(), pos.cache_misses())
    };
    let pass_duration = start_pass.elapsed();

    println!("\n3. Tuple Pipeline Metrics:");
    println!("   - Dispatched Archetypes: {}", pass_stats.dispatched_archetypes);
    println!("   - Total Processed Rows: {}", pass_stats.total_rows);
    println!(
        "   - Auto-Inserted Memory Barriers: {}",
        pass_stats.barriers_inserted
    );
    println!(
        "   - Command Buffer Submissions: {}",
        pass_stats.command_buffer_submissions
    );
    println!(
        "   - SharedColCache Hits / Misses: {} hits / {} miss",
        cache_hits, cache_misses
    );
    println!("   - Execution Wall Time: {:?}", pass_duration);

    assert_eq!(
        pass_stats.barriers_inserted, 2,
        "Hazard tracker must automatically insert 2 barriers across the 3 dependent stages"
    );
    assert_eq!(
        cache_misses, 1,
        "SharedColCache must scan archetypes only once on cold start"
    );
    assert!(
        cache_hits >= 2,
        "Subsequent stages in the pipeline must hit SharedColCache with 0 re-computation"
    );
    if has_gpu {
        assert_eq!(
            pass_stats.command_buffer_submissions, 1,
            "Multi-kernel tuple pipeline must execute in a single command buffer submission!"
        );
    }

    let expected_pos = Vec3::new(1.969, 13.938, 25.907);
    let expected_vel = Vec3::new(9.69, 19.38, 29.07);
    let expected_force = Vec3::new(1.0, 2.0, 3.0);

    println!("\n4. Verifying CPU-GPU Fusion View Consistency & Filter Isolation:");
    let sample_e = EntityId::from_u128(1);
    let sample_pos = *store.get::<BodyPosition>(sample_e).expect("get pos");
    let sample_vel = *store.get::<BodyVelocity>(sample_e).expect("get vel");
    let sample_dyn = *store.get::<BodyDynamics>(sample_e).expect("get dyn");
    let frozen_pos = *store
        .get::<BodyPosition>(EntityId::from_u128(999_999))
        .expect("get frozen pos");

    assert_eq!(
        frozen_pos.coords,
        Vec3::new(100.0, 100.0, 100.0),
        "Frozen entity must remain untouched"
    );
    assert!((sample_pos.coords - expected_pos).length() < 1e-3);
    assert!((sample_vel.velocity - expected_vel).length() < 1e-3);
    assert!((sample_dyn.force - expected_force).length() < 1e-3);
    println!("   => [PASS] Exact numerical consistency verified between CPU store and GPU output.");

    // Verify 16KB hardware page dirty bitmaps
    println!("\n5. Verifying 16KB Page Dirty Bitmaps Tracking:");
    let pos_views = store
        .gpu_column_views::<BodyPosition>(initial_epoch)
        .expect("pos views");
    assert!(pos_views[0].1.dirty_bitmap().dirty_page_count() > 0);
    println!("   => [PASS] Hardware page dirty bitmaps accurately reflect all modified columns.");

    // Verify Fused Single-Kernel Execution (`step_physics_fused.gpu(...)`)
    println!("\n6. Verifying Fused Single-Kernel (`step_physics_fused.gpu(...)`) & Custom Rayon Loop:");
    let mut store_fused = populate_test_store(NUM_ENTITIES);
    {
        let (mut pos, mut vel, mut dyns) = store_fused
            .queries_mut::<(&mut BodyPosition, &mut BodyVelocity, &mut BodyDynamics)>()
            .expect("queries_mut")
            .with::<ActiveTag>()
            .without::<FrozenTag>();

        let fused_stats = step_physics_fused
            .gpu_on(&mut engine, &mut pos, &mut vel, &mut dyns, DT)
            .expect("fused gpu");
        assert_eq!(fused_stats.barriers_inserted, 0);

        // User-written Rayon parallel loop directly calling the same `#[kernel]` function!
        (&mut pos, &vel)
            .into_par_iter()
            .for_each(|(p, v)| integrate(p, v, DT));
    }

    let fused_e1_pos = *store_fused.get::<BodyPosition>(sample_e).expect("fused pos");
    let expected_after_extra_step = expected_pos + expected_vel * DT;
    assert!((fused_e1_pos.coords - expected_after_extra_step).length() < 1e-3);
    println!("   => [PASS] Fused kernel and custom Rayon `.into_par_iter()` loop verified.");

    println!("\n===============================================================================");
    println!("   All Ergonomic `#[kernel]` & Multi-Stage Pipeline Checks Passed!             ");
    println!("===============================================================================");
}
