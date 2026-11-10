#![cfg(feature = "gpu")]

use sumeru::storage::gpu::ir::{dot, length, vec3};
use sumeru::{
    gpu, EntityId, LayoutOptimization, MetalComputeEngine, Store,
};

// -----------------------------------------------------------------------------
// Component Definitions (POD, 16-byte / 32-byte aligned structures)
// -----------------------------------------------------------------------------

#[gpu]
pub struct AdvCoord {
    pub pos: [f32; 3],
    pub seed: f32,
}

#[gpu]
pub struct AdvMotion {
    pub vel: [f32; 3],
    pub speed_limit: f32,
}

#[gpu]
pub struct AdvField {
    pub gravity: [f32; 3],
    pub damping: f32,
}

#[gpu]
pub struct AdvEnergy {
    pub kinetic: f32,
    pub potential: f32,
    pub work_done: f32,
    pub generation: u32,
}

// -----------------------------------------------------------------------------
// GPU Kernels: Multi-Stage Dependency Chain (RAW, WAR, WAW Hazards)
// -----------------------------------------------------------------------------

/// Stage 1 (Kinematics Integration):
/// Reads `motion.vel`, writes `coord.pos` (RAW source for Stage 2 & 4).
#[gpu]
pub fn adv_stage1_integrate_pos(coord: &mut AdvCoord, motion: &AdvMotion, dt: f32) {
    let v = vec3(motion.vel[0], motion.vel[1], motion.vel[2]);
    coord.pos[0] += v[0] * dt;
    coord.pos[1] += v[1] * dt;
    coord.pos[2] += v[2] * dt;
    coord.seed += 1.0;
}

/// Stage 2 (Dynamics Update & Acceleration):
/// Reads `coord.pos` (RAW from Stage 1), reads `field`, writes `motion.vel` (RAW for Stage 3 & 4).
#[gpu]
pub fn adv_stage2_update_dynamics(motion: &mut AdvMotion, coord: &AdvCoord, field: &AdvField, dt: f32) {
    let p = vec3(coord.pos[0], coord.pos[1], coord.pos[2]);
    let dist = length(p).max(0.001);
    let g = vec3(field.gravity[0], field.gravity[1], field.gravity[2]);

    let acc_x = g[0] - (p[0] / dist) * 0.1;
    let acc_y = g[1] - (p[1] / dist) * 0.1;
    let acc_z = g[2] - (p[2] / dist) * 0.1;

    let v_raw_x = motion.vel[0] + acc_x * dt;
    let v_raw_y = motion.vel[1] + acc_y * dt;
    let v_raw_z = motion.vel[2] + acc_z * dt;

    let damp = (1.0 - field.damping * dt).max(0.0);
    let mut vx = v_raw_x * damp;
    let mut vy = v_raw_y * damp;
    let mut vz = v_raw_z * damp;

    let spd = length(vec3(vx, vy, vz));
    if spd > motion.speed_limit && spd > 0.0001 {
        let scale = motion.speed_limit / spd;
        vx *= scale;
        vy *= scale;
        vz *= scale;
    }

    motion.vel[0] = vx;
    motion.vel[1] = vy;
    motion.vel[2] = vz;
}

/// Stage 3 (Energy Calculation):
/// Reads `motion.vel` (RAW from Stage 2) and `coord.pos`, writes `energy` (RAW/WAW).
#[gpu]
pub fn adv_stage3_compute_energy(
    energy: &mut AdvEnergy,
    motion: &AdvMotion,
    coord: &AdvCoord,
    field: &AdvField,
) {
    let v = vec3(motion.vel[0], motion.vel[1], motion.vel[2]);
    let p = vec3(coord.pos[0], coord.pos[1], coord.pos[2]);
    let spd_sq = dot(v, v);
    let m = 1.0f32;

    energy.kinetic = 0.5 * m * spd_sq;
    let height = p[1].max(0.0);
    energy.potential = m * 9.81 * height;
    energy.work_done += (energy.kinetic + energy.potential) * field.damping * 0.01;
    energy.generation += 1;
}

/// Stage 4 (Secondary Integration & WAW / WAR Verification):
/// Reads `motion.vel` (updated in Stage 2), writes `coord.pos` (WAW with Stage 1),
/// and updates `energy.generation` (WAW with Stage 3).
#[gpu]
pub fn adv_stage4_refine_pos_and_energy(
    coord: &mut AdvCoord,
    energy: &mut AdvEnergy,
    motion: &AdvMotion,
    dt: f32,
) {
    let v = vec3(motion.vel[0], motion.vel[1], motion.vel[2]);
    coord.pos[0] += v[0] * (dt * 0.5);
    coord.pos[1] += v[1] * (dt * 0.5);
    coord.pos[2] += v[2] * (dt * 0.5);
    energy.generation += 10;
}

// -----------------------------------------------------------------------------
// Test 1: Data Hazards & Barrier Correctness (RAW, WAR, WAW Chains)
// -----------------------------------------------------------------------------

#[test]
fn test_adversarial_data_hazards_and_single_submission() {
    let mut engine = MetalComputeEngine::new();
    let has_gpu = engine.has_hardware_gpu();

    const COUNT: usize = 2048;
    const DT: f32 = 0.05;

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let initial_epoch = store.epoch();

    for i in 0..COUNT {
        let e = EntityId::from_u128(i as u128 + 100);
        store
            .insert((
                e,
                (
                    AdvCoord {
                        pos: [1.0, 10.0, 2.0],
                        seed: i as f32,
                    },
                    AdvMotion {
                        vel: [5.0, -2.0, 3.0],
                        speed_limit: 100.0,
                    },
                    AdvField {
                        gravity: [0.0, -9.81, 0.0],
                        damping: 0.1,
                    },
                    AdvEnergy {
                        kinetic: 0.0,
                        potential: 0.0,
                        work_done: 0.0,
                        generation: 0,
                    },
                ),
            ))
            .expect("insert entity");
    }

    // Step A: Multi-Kernel Pass with 4 dependent kernels and barriers
    let mut pass = store.gpu_compute_pass(&mut engine).expect("begin pass");
    assert_eq!(pass.dispatches(), 0);

    pass.dispatch_with_args(adv_stage1_integrate_pos, DT)
        .expect("stage 1");
    pass.barrier().expect("barrier 1");

    pass.dispatch_with_args(adv_stage2_update_dynamics, DT)
        .expect("stage 2");
    pass.barrier().expect("barrier 2");

    pass.dispatch(adv_stage3_compute_energy).expect("stage 3");
    pass.barrier().expect("barrier 3");

    pass.dispatch_with_args(adv_stage4_refine_pos_and_energy, DT)
        .expect("stage 4");

    let stats = pass.finish().expect("finish pass");

    if has_gpu {
        // Crucial empirical check: exactly 1 command buffer submission
        assert_eq!(stats.command_buffer_submissions, 1);
    }
    assert_eq!(stats.dispatched_archetypes, 4);
    assert_eq!(stats.total_rows, COUNT * 4);
    assert!(store.epoch() > initial_epoch);

    // Step B: Sequential Single-Query Execution Ground Truth Store
    let mut ground_truth = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..COUNT {
        let e = EntityId::from_u128(i as u128 + 100);
        ground_truth
            .insert((
                e,
                (
                    AdvCoord {
                        pos: [1.0, 10.0, 2.0],
                        seed: i as f32,
                    },
                    AdvMotion {
                        vel: [5.0, -2.0, 3.0],
                        speed_limit: 100.0,
                    },
                    AdvField {
                        gravity: [0.0, -9.81, 0.0],
                        damping: 0.1,
                    },
                    AdvEnergy {
                        kinetic: 0.0,
                        potential: 0.0,
                        work_done: 0.0,
                        generation: 0,
                    },
                ),
            ))
            .expect("insert entity to ground truth");
    }

    let q1 = ground_truth
        .query_mut::<(&mut AdvCoord, &AdvMotion)>()
        .expect("q1");
    q1.gpu()
        .for_each_on(&mut engine, adv_stage1_integrate_pos.with_args(DT))
        .expect("single q1");

    let q2 = ground_truth
        .query_mut::<(&mut AdvMotion, &AdvCoord, &AdvField)>()
        .expect("q2");
    q2.gpu()
        .for_each_on(&mut engine, adv_stage2_update_dynamics.with_args(DT))
        .expect("single q2");

    let q3 = ground_truth
        .query_mut::<(&mut AdvEnergy, &AdvMotion, &AdvCoord, &AdvField)>()
        .expect("q3");
    q3.gpu()
        .for_each_on(&mut engine, adv_stage3_compute_energy)
        .expect("single q3");

    let q4 = ground_truth
        .query_mut::<(&mut AdvCoord, &mut AdvEnergy, &AdvMotion)>()
        .expect("q4");
    q4.gpu()
        .for_each_on(&mut engine, adv_stage4_refine_pos_and_energy.with_args(DT))
        .expect("single q4");

    // Step C: Verify 100% Bitwise Parity across all entities between Multi-Kernel Pass and Sequential Dispatches
    for i in 0..COUNT {
        let e = EntityId::from_u128(i as u128 + 100);

        let c_pass = *store.get::<AdvCoord>(e).expect("get c_pass");
        let m_pass = *store.get::<AdvMotion>(e).expect("get m_pass");
        let en_pass = *store.get::<AdvEnergy>(e).expect("get en_pass");

        let c_single = *ground_truth.get::<AdvCoord>(e).expect("get c_single");
        let m_single = *ground_truth.get::<AdvMotion>(e).expect("get m_single");
        let en_single = *ground_truth.get::<AdvEnergy>(e).expect("get en_single");

        for axis in 0..3 {
            assert!(
                (c_pass.pos[axis] - c_single.pos[axis]).abs() < 1e-4,
                "Coord pos[{axis}] mismatch at entity {i}: pass={}, single={}",
                c_pass.pos[axis],
                c_single.pos[axis]
            );
            assert!(
                (m_pass.vel[axis] - m_single.vel[axis]).abs() < 1e-4,
                "Motion vel[{axis}] mismatch at entity {i}: pass={}, single={}",
                m_pass.vel[axis],
                m_single.vel[axis]
            );
        }

        assert_eq!(
            c_pass.seed, c_single.seed,
            "Seed mismatch at entity {i}: pass={}, single={}",
            c_pass.seed, c_single.seed
        );
        assert!(
            (en_pass.kinetic - en_single.kinetic).abs() < 1e-4,
            "Kinetic energy mismatch at {i}: pass={}, single={}",
            en_pass.kinetic,
            en_single.kinetic
        );
        assert!(
            (en_pass.potential - en_single.potential).abs() < 1e-4,
            "Potential energy mismatch at {i}: pass={}, single={}",
            en_pass.potential,
            en_single.potential
        );
        assert_eq!(
            en_pass.generation, en_single.generation,
            "Generation mismatch at {i}: pass={}, single={}",
            en_pass.generation, en_single.generation
        );
        assert_eq!(
            en_pass.generation, 11,
            "Expected 1 + 10 = 11 generations at {i}"
        );
    }
}

// -----------------------------------------------------------------------------
// Test 2: Scaling Across Entity Counts (64, 1024, 4096, 16384)
// -----------------------------------------------------------------------------

#[test]
fn test_adversarial_entity_count_scaling_parity() {
    let mut engine = MetalComputeEngine::new();
    let has_gpu = engine.has_hardware_gpu();

    let entity_counts = [64, 1024, 4096, 16384];

    for &count in &entity_counts {
        let mut store_pass = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
        let mut store_seq = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

        for i in 0..count {
            let e = EntityId::from_u128(i as u128 + 1);
            let init_c = AdvCoord {
                pos: [i as f32 * 0.01, 100.0, 50.0],
                seed: (i % 37) as f32,
            };
            let init_m = AdvMotion {
                vel: [1.0, 2.0, -1.0],
                speed_limit: 50.0,
            };
            let init_f = AdvField {
                gravity: [0.0, -9.81, 0.0],
                damping: 0.05,
            };
            let init_en = AdvEnergy {
                kinetic: 0.0,
                potential: 0.0,
                work_done: 0.0,
                generation: 0,
            };

            store_pass.insert((e, (init_c, init_m, init_f, init_en))).unwrap();
            store_seq.insert((e, (init_c, init_m, init_f, init_en))).unwrap();
        }

        // Pass execution
        let mut pass = store_pass.gpu_compute_pass(&mut engine).unwrap();
        pass.dispatch_with_args(adv_stage1_integrate_pos, 0.1).unwrap();
        pass.barrier().unwrap();
        pass.dispatch_with_args(adv_stage2_update_dynamics, 0.1).unwrap();
        pass.barrier().unwrap();
        pass.dispatch(adv_stage3_compute_energy).unwrap();
        let stats = pass.finish().unwrap();

        if has_gpu {
            assert_eq!(stats.command_buffer_submissions, 1);
        }
        assert_eq!(stats.total_rows, count * 3);

        // Sequential execution
        let q1 = store_seq.query_mut::<(&mut AdvCoord, &AdvMotion)>().unwrap();
        q1.gpu().for_each_on(&mut engine, adv_stage1_integrate_pos.with_args(0.1)).unwrap();

        let q2 = store_seq.query_mut::<(&mut AdvMotion, &AdvCoord, &AdvField)>().unwrap();
        q2.gpu().for_each_on(&mut engine, adv_stage2_update_dynamics.with_args(0.1)).unwrap();

        let q3 = store_seq.query_mut::<(&mut AdvEnergy, &AdvMotion, &AdvCoord, &AdvField)>().unwrap();
        q3.gpu().for_each_on(&mut engine, adv_stage3_compute_energy).unwrap();

        // Compare first 10, middle 10, and last 10 entities
        let sample_indices: Vec<usize> = if count <= 100 {
            (0..count).collect()
        } else {
            let mut v = Vec::new();
            v.extend(0..10);
            let mid = count / 2;
            v.extend((mid - 5)..(mid + 5));
            v.extend((count - 10)..count);
            v
        };

        for &i in &sample_indices {
            let e = EntityId::from_u128(i as u128 + 1);
            let cp = store_pass.get::<AdvCoord>(e).unwrap();
            let cs = store_seq.get::<AdvCoord>(e).unwrap();
            let mp = store_pass.get::<AdvMotion>(e).unwrap();
            let ms = store_seq.get::<AdvMotion>(e).unwrap();
            let ep = store_pass.get::<AdvEnergy>(e).unwrap();
            let es = store_seq.get::<AdvEnergy>(e).unwrap();

            for ax in 0..3 {
                assert!((cp.pos[ax] - cs.pos[ax]).abs() < 1e-4, "scale {count} coord mismatch");
                assert!((mp.vel[ax] - ms.vel[ax]).abs() < 1e-4, "scale {count} vel mismatch");
            }
            assert!((ep.kinetic - es.kinetic).abs() < 1e-4, "scale {count} kinetic mismatch");
            assert!((ep.potential - es.potential).abs() < 1e-4, "scale {count} potential mismatch");
            assert_eq!(ep.generation, es.generation, "scale {count} generation mismatch");
        }
    }
}

// -----------------------------------------------------------------------------
// Test 3: Multiple Archetypes Heterogeneous Dispatch & Isolation
// -----------------------------------------------------------------------------

#[test]
fn test_adversarial_multi_archetype_heterogeneous_dispatch() {
    let mut engine = MetalComputeEngine::new();
    let has_gpu = engine.has_hardware_gpu();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    // Archetype 1: (Coord, Motion, Field, Energy) — 500 entities
    for i in 0..500 {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    AdvCoord {
                        pos: [1.0, 2.0, 3.0],
                        seed: 1.0,
                    },
                    AdvMotion {
                        vel: [10.0, 20.0, 30.0],
                        speed_limit: 100.0,
                    },
                    AdvField {
                        gravity: [0.0, -9.81, 0.0],
                        damping: 0.1,
                    },
                    AdvEnergy {
                        kinetic: 0.0,
                        potential: 0.0,
                        work_done: 0.0,
                        generation: 0,
                    },
                ),
            ))
            .unwrap();
    }

    // Archetype 2: (Coord, Motion, Field) without Energy — 500 entities
    for i in 500..1000 {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    AdvCoord {
                        pos: [10.0, 20.0, 30.0],
                        seed: 2.0,
                    },
                    AdvMotion {
                        vel: [1.0, 2.0, 3.0],
                        speed_limit: 50.0,
                    },
                    AdvField {
                        gravity: [0.0, -9.81, 0.0],
                        damping: 0.2,
                    },
                ),
            ))
            .unwrap();
    }

    // Archetype 3: (Coord, Motion) without Field or Energy — 500 entities
    for i in 1000..1500 {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    AdvCoord {
                        pos: [100.0, 200.0, 300.0],
                        seed: 3.0,
                    },
                    AdvMotion {
                        vel: [5.0, 5.0, 5.0],
                        speed_limit: 80.0,
                    },
                ),
            ))
            .unwrap();
    }

    // Archetype 4: (Motion, Field) without Coord — 500 entities (should be completely untouched by stage 1)
    for i in 1500..2000 {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    AdvMotion {
                        vel: [-1.0, -2.0, -3.0],
                        speed_limit: 20.0,
                    },
                    AdvField {
                        gravity: [0.0, 0.0, 0.0],
                        damping: 0.0,
                    },
                ),
            ))
            .unwrap();
    }

    // Execute multi-kernel compute pass
    let mut pass = store.gpu_compute_pass(&mut engine).unwrap();

    // Kernel 1: stage 1 matches Arch 1, Arch 2, Arch 3 (3 archetypes = 1500 rows)
    pass.dispatch_with_args(adv_stage1_integrate_pos, 0.1).unwrap();
    pass.barrier().unwrap();

    // Kernel 2: stage 2 matches Arch 1, Arch 2 (2 archetypes = 1000 rows)
    pass.dispatch_with_args(adv_stage2_update_dynamics, 0.1).unwrap();
    pass.barrier().unwrap();

    // Kernel 3: stage 3 matches only Arch 1 (1 archetype = 500 rows)
    pass.dispatch(adv_stage3_compute_energy).unwrap();

    let stats = pass.finish().unwrap();

    if has_gpu {
        assert_eq!(stats.command_buffer_submissions, 1);
    }
    // Dispatched archetypes: 3 (stage 1) + 2 (stage 2) + 1 (stage 3) = 6
    assert_eq!(stats.dispatched_archetypes, 6);
    assert_eq!(stats.total_rows, 1500 + 1000 + 500);

    // Verify Archetype 4 was NOT modified by stage 1 or stage 3
    for i in 1500..2000 {
        let e = EntityId::from_u128(i as u128 + 1);
        let m = store.get::<AdvMotion>(e).unwrap();
        // Since Arch 4 has (Motion, Field) and no Coord, Stage 1 (needs Coord, Motion) skipped it.
        // Stage 2 (needs Motion, Coord, Field) also skipped it!
        // Stage 3 (needs Energy, Motion, Coord, Field) skipped it!
        // Vel must be EXACTLY initial value: [-1.0, -2.0, -3.0]
        assert_eq!(m.vel, [-1.0, -2.0, -3.0]);
    }

    // Verify Archetype 3 had Stage 1 applied, but Stage 2 skipped (no Field)
    for i in 1000..1500 {
        let e = EntityId::from_u128(i as u128 + 1);
        let c = store.get::<AdvCoord>(e).unwrap();
        let m = store.get::<AdvMotion>(e).unwrap();
        // Pos = [100, 200, 300] + [5, 5, 5] * 0.1 = [100.5, 200.5, 300.5]
        assert!((c.pos[0] - 100.5).abs() < 1e-4);
        assert!((c.pos[1] - 200.5).abs() < 1e-4);
        assert!((c.pos[2] - 300.5).abs() < 1e-4);
        // Vel remains [5, 5, 5]
        assert_eq!(m.vel, [5.0, 5.0, 5.0]);
    }
}

// -----------------------------------------------------------------------------
// Test 4: Dirty Page Bitmaps & Modified vs Unmodified Column Tracking
// -----------------------------------------------------------------------------

#[test]
fn test_adversarial_dirty_page_bitmap_exactness() {
    let mut engine = MetalComputeEngine::new();
    const ROWS: usize = 4096; // 4096 * 16B = 64KB = 4 hardware 16KB pages

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    for i in 0..ROWS {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    AdvCoord {
                        pos: [0.0, 0.0, 0.0],
                        seed: 0.0,
                    },
                    AdvMotion {
                        vel: [1.0, 1.0, 1.0],
                        speed_limit: 10.0,
                    },
                    AdvField {
                        gravity: [0.0, -9.81, 0.0],
                        damping: 0.1,
                    },
                ),
            ))
            .unwrap();
    }

    let pre_pass_epoch = store.epoch();

    // Pass: ONLY modifies AdvCoord. AdvMotion and AdvField are read-only!
    let mut pass = store.gpu_compute_pass(&mut engine).unwrap();
    pass.dispatch_with_args(adv_stage1_integrate_pos, 0.1).unwrap();
    pass.finish().unwrap();

    let post_pass_epoch = store.epoch();
    assert!(post_pass_epoch > pre_pass_epoch);

    // 1. AdvCoord was modified: its 16KB dirty page bitmap MUST report dirty pages > 0 (all pages dirty)
    let coord_views = store.gpu_column_views::<AdvCoord>(pre_pass_epoch).unwrap();
    assert_eq!(coord_views.len(), 1);
    let coord_dirty = coord_views[0].1.dirty_bitmap();
    assert!(
        coord_dirty.dirty_page_count() > 0,
        "AdvCoord dirty page count should be > 0"
    );
    assert_eq!(
        coord_dirty.dirty_page_count(),
        coord_views[0].1.geometry().page_count,
        "All active pages of AdvCoord should be marked dirty"
    );

    // 2. AdvMotion was read-only: its dirty page bitmap since pre_pass_epoch MUST be 0!
    let motion_views = store.gpu_column_views::<AdvMotion>(pre_pass_epoch).unwrap();
    assert_eq!(motion_views.len(), 1);
    let motion_dirty = motion_views[0].1.dirty_bitmap();
    assert_eq!(
        motion_dirty.dirty_page_count(),
        0,
        "AdvMotion was only read, dirty page count must be 0!"
    );

    // 3. AdvField was untouched: its dirty page bitmap must also be 0!
    let field_views = store.gpu_column_views::<AdvField>(pre_pass_epoch).unwrap();
    assert_eq!(field_views.len(), 1);
    let field_dirty = field_views[0].1.dirty_bitmap();
    assert_eq!(
        field_dirty.dirty_page_count(),
        0,
        "AdvField was not accessed, dirty page count must be 0!"
    );

    // 4. Querying since post_pass_epoch: AdvCoord dirty page count must now be 0!
    let coord_views_post = store.gpu_column_views::<AdvCoord>(post_pass_epoch).unwrap();
    assert_eq!(
        coord_views_post[0].1.dirty_bitmap().dirty_page_count(),
        0,
        "Querying since post_pass_epoch must show 0 dirty pages"
    );
}

// -----------------------------------------------------------------------------
// Test 5: Edge Cases: Empty Pass, Zero-Row Archetypes, Consecutive Barriers
// -----------------------------------------------------------------------------

#[test]
fn test_adversarial_edge_cases_empty_pass_and_barriers() {
    let mut engine = MetalComputeEngine::new();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let epoch_before = store.epoch();

    // Edge Case 1: Empty pass (0 dispatches, only finish)
    {
        let pass = store.gpu_compute_pass(&mut engine).expect("begin empty pass");
        assert_eq!(pass.dispatches(), 0);
        let stats = pass.finish().expect("finish empty pass");
        assert_eq!(stats.dispatched_archetypes, 0);
        assert_eq!(stats.total_rows, 0);
        // Since 0 dispatches occurred, epoch should NOT have bumped
        assert_eq!(store.epoch(), epoch_before);
    }

    // Edge Case 2: Consecutive memory barriers without dispatches in between
    {
        let mut pass = store.gpu_compute_pass(&mut engine).expect("begin barrier pass");
        pass.barrier().expect("barrier 1");
        pass.barrier().expect("barrier 2");
        pass.barrier().expect("barrier 3");
        let stats = pass.finish().expect("finish barrier pass");
        assert_eq!(stats.dispatched_archetypes, 0);
    }

    // Edge Case 3: Empty archetype in Store (0 rows)
    {
        // Register query to create archetype with 0 rows
        store.register_gpu_query::<(&mut AdvCoord, &AdvMotion)>().unwrap();

        let mut pass = store.gpu_compute_pass(&mut engine).unwrap();
        // Dispatching on store with 0 rows should succeed gracefully with 0 dispatches recorded
        pass.dispatch_with_args(adv_stage1_integrate_pos, 0.1).unwrap();
        let stats = pass.finish().unwrap();
        assert_eq!(stats.dispatched_archetypes, 0);
        assert_eq!(stats.total_rows, 0);
    }
}
