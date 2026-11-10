#![cfg(feature = "gpu")]

use sumeru::storage::gpu::ir::{dot, length, vec3};
use sumeru::{
    gpu, EntityId, LayoutOptimization, MetalComputeEngine, Store,
};

#[gpu]
pub struct E2eBodyCoord {
    pub pos: [f32; 3],
}

#[gpu]
pub struct E2eBodyMotion {
    pub vel: [f32; 3],
}

#[gpu]
pub struct E2eBodyAero {
    pub force: [f32; 3],
    pub drag: f32,
}

#[gpu]
pub fn e2e_stage1_integrate(coord: &mut E2eBodyCoord, motion: &E2eBodyMotion, dt: f32) {
    let v = vec3(motion.vel[0], motion.vel[1], motion.vel[2]);
    coord.pos[0] += v[0] * dt;
    coord.pos[1] += v[1] * dt;
    coord.pos[2] += v[2] * dt;
}

#[gpu]
pub fn e2e_stage2_damp_decay(motion: &mut E2eBodyMotion, aero: &mut E2eBodyAero, dt: f32) {
    let v_init = vec3(motion.vel[0], motion.vel[1], motion.vel[2]);
    let f = vec3(aero.force[0], aero.force[1], aero.force[2]);

    let v_acc = vec3(
        v_init[0] + f[0] * dt,
        v_init[1] + f[1] * dt,
        v_init[2] + f[2] * dt,
    );

    let speed = length(v_acc);
    let damp_factor = (1.0 - aero.drag * dt).max(0.0);
    let power = dot(v_acc, f);
    let energy_loss = speed * 0.0 + power * 0.0;

    motion.vel[0] = v_acc[0] * damp_factor + energy_loss;
    motion.vel[1] = v_acc[1] * damp_factor + energy_loss;
    motion.vel[2] = v_acc[2] * damp_factor + energy_loss;

    aero.force[0] *= 0.5;
    aero.force[1] *= 0.5;
    aero.force[2] *= 0.5;
}

#[test]
fn test_gpu_multi_kernel_compute_pass_e2e_integration() {
    let mut engine = MetalComputeEngine::new();
    let has_gpu = engine.has_hardware_gpu();

    const NUM_ENTITIES: usize = 512;
    const DT: f32 = 0.1;

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut E2eBodyCoord, &E2eBodyMotion)>()
        .expect("register query 1");
    store
        .register_gpu_query::<(&mut E2eBodyMotion, &mut E2eBodyAero)>()
        .expect("register query 2");

    let initial_epoch = store.epoch();

    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        store
            .insert((
                e,
                (
                    E2eBodyCoord {
                        pos: [0.0, 10.0, 20.0],
                    },
                    E2eBodyMotion {
                        vel: [10.0, 20.0, 30.0],
                    },
                    E2eBodyAero {
                        force: [2.0, 4.0, 6.0],
                        drag: 0.5,
                    },
                ),
            ))
            .expect("insert entity");
    }

    // Execute multi-kernel compute pass
    let mut pass = store.gpu_compute_pass(&mut engine).expect("begin pass");
    assert_eq!(pass.dispatches(), 0);

    pass.dispatch_with_args(e2e_stage1_integrate, DT).expect("stage 1");
    assert_eq!(pass.dispatches(), 1);

    pass.barrier().expect("barrier 1");

    pass.dispatch_with_args(e2e_stage2_damp_decay, DT).expect("stage 2");
    assert_eq!(pass.dispatches(), 2);

    pass.barrier().expect("barrier 2");

    pass.dispatch_with_args(e2e_stage1_integrate, DT).expect("stage 3");
    assert_eq!(pass.dispatches(), 3);

    let stats = pass.finish().expect("finish pass");

    if has_gpu {
        assert_eq!(stats.command_buffer_submissions, 1);
    }
    assert_eq!(stats.dispatched_archetypes, 3);
    assert_eq!(stats.total_rows, NUM_ENTITIES * 3);

    // Verify Store global epoch bumped
    assert!(store.epoch() > initial_epoch);

    // Verify CPU read consistency
    let expected_pos = [1.969, 13.938, 25.907];
    let expected_vel = [9.69, 19.38, 29.07];
    let expected_force = [1.0, 2.0, 3.0];

    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        let pos = *store.get::<E2eBodyCoord>(e).expect("get pos");
        let vel = *store.get::<E2eBodyMotion>(e).expect("get vel");
        let aero = *store.get::<E2eBodyAero>(e).expect("get aero");

        for axis in 0..3 {
            assert!((pos.pos[axis] - expected_pos[axis]).abs() < 1e-3);
            assert!((vel.vel[axis] - expected_vel[axis]).abs() < 1e-3);
            assert!((aero.force[axis] - expected_force[axis]).abs() < 1e-3);
        }
    }

    // Verify dirty bitmaps
    let pos_views = store.gpu_column_views::<E2eBodyCoord>(initial_epoch).expect("pos views");
    let vel_views = store.gpu_column_views::<E2eBodyMotion>(initial_epoch).expect("vel views");
    let aero_views = store.gpu_column_views::<E2eBodyAero>(initial_epoch).expect("aero views");

    assert!(!pos_views.is_empty());
    assert!(!vel_views.is_empty());
    assert!(!aero_views.is_empty());

    assert!(pos_views[0].1.dirty_bitmap().dirty_page_count() > 0);
    assert!(vel_views[0].1.dirty_bitmap().dirty_page_count() > 0);
    assert!(aero_views[0].1.dirty_bitmap().dirty_page_count() > 0);

    // Comparison with single query .gpu().for_each(...) execution
    let mut store_single = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        store_single
            .insert((
                e,
                (
                    E2eBodyCoord {
                        pos: [0.0, 10.0, 20.0],
                    },
                    E2eBodyMotion {
                        vel: [10.0, 20.0, 30.0],
                    },
                    E2eBodyAero {
                        force: [2.0, 4.0, 6.0],
                        drag: 0.5,
                    },
                ),
            ))
            .expect("insert entity");
    }

    let q1 = store_single.query_mut::<(&mut E2eBodyCoord, &E2eBodyMotion)>().expect("q1");
    q1.gpu().for_each_on(&mut engine, e2e_stage1_integrate.with_args(DT)).expect("single q1");

    let q2 = store_single.query_mut::<(&mut E2eBodyMotion, &mut E2eBodyAero)>().expect("q2");
    q2.gpu().for_each_on(&mut engine, e2e_stage2_damp_decay.with_args(DT)).expect("single q2");

    let q3 = store_single.query_mut::<(&mut E2eBodyCoord, &E2eBodyMotion)>().expect("q3");
    q3.gpu().for_each_on(&mut engine, e2e_stage1_integrate.with_args(DT)).expect("single q3");

    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        let pass_pos = *store.get::<E2eBodyCoord>(e).unwrap();
        let pass_vel = *store.get::<E2eBodyMotion>(e).unwrap();
        let pass_aero = *store.get::<E2eBodyAero>(e).unwrap();

        let single_pos = *store_single.get::<E2eBodyCoord>(e).unwrap();
        let single_vel = *store_single.get::<E2eBodyMotion>(e).unwrap();
        let single_aero = *store_single.get::<E2eBodyAero>(e).unwrap();

        for axis in 0..3 {
            assert!((pass_pos.pos[axis] - single_pos.pos[axis]).abs() < 1e-4);
            assert!((pass_vel.vel[axis] - single_vel.vel[axis]).abs() < 1e-4);
            assert!((pass_aero.force[axis] - single_aero.force[axis]).abs() < 1e-4);
        }
    }
}
