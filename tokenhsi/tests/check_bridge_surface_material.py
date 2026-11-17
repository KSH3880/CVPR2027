import argparse
import json
from pathlib import Path
import sys

from isaacgym import gymapi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.tasks.multi_agent.bridge_surface import load_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output_etc/check/bridge_surface_20261112/material_cpu')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if Path('output_etc/check').resolve() not in output.parents:
        raise ValueError('Material checks must use output_etc/check/')
    output.mkdir(parents=True, exist_ok=True)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt, params.substeps = 1/30, 2
    params.up_axis, params.gravity = gymapi.UP_AXIS_Z, gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = params.use_gpu_pipeline = False
    params.physx.num_threads, params.physx.solver_type = 2, 1
    params.physx.num_position_iterations, params.physx.num_velocity_iterations = 4, 0
    params.physx.contact_offset, params.physx.rest_offset = .02, 0.
    sim = gym.create_sim(-1, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('CPU PhysX simulator creation failed')
    plane = gymapi.PlaneParams()
    plane.normal, plane.static_friction, plane.dynamic_friction = gymapi.Vec3(0, 0, 1), 1., 1.
    gym.add_ground(sim, plane)
    size = [5.1, 2.2, 1.5]
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link, fixed.density = True, 3.
    single = gym.create_box(sim, *size, fixed)
    split = load_asset(gym, sim, output/'assets', size, fixed, .005)
    dynamic = gymapi.AssetOptions()
    dynamic.density = 1000.
    shoe_asset = gym.create_box(sim, .2, .1, .08, dynamic)
    cases, mass_checks = [], []
    for label, asset, friction in [('single04', single, .4), ('split04', split, .4), ('split10', split, 1.)]:
        env = gym.create_env(sim, gymapi.Vec3(-4, -4, 0), gymapi.Vec3(4, 4, 4), 3)
        pose = gymapi.Transform()
        pose.p.z = .75
        actor = gym.create_actor(env, asset, pose, 'bridge', len(cases), 0)
        props = gym.get_actor_rigid_shape_properties(env, actor)
        for prop in props:
            prop.friction = .4
            prop.restitution = 0.
        props[-1].friction = friction
        gym.set_actor_rigid_shape_properties(env, actor, props)
        actual = gym.get_actor_rigid_shape_properties(env, actor)
        pose.p.z = 1.7
        shoe = gym.create_actor(env, shoe_asset, pose, 'shoe', len(cases), 0)
        props = gym.get_actor_rigid_shape_properties(env, shoe)
        for prop in props:
            prop.friction = 1.
        gym.set_actor_rigid_shape_properties(env, shoe, props)
        cases.append(dict(label=label, env=env, actor=shoe, friction=[p.friction for p in actual], trace=[]))
    for controlled, density in [(False, 3.), (True, 500.)]:
        for span in (1, 2, 3):
            dims = [1.7*span, 2.2, 1.5]
            options = gymapi.AssetOptions()
            options.density, options.disable_gravity = density, controlled
            if controlled:
                options.max_angular_velocity = 0.
            baseline = gym.create_box(sim, *dims, options)
            capped = load_asset(gym, sim, output/'assets', dims, options, .005, controlled)
            env = gym.create_env(sim, gymapi.Vec3(-4, -4, 0), gymapi.Vec3(4, 4, 4), 3)
            bodies = []
            for index, asset in enumerate((baseline, capped)):
                pose = gymapi.Transform()
                pose.p.x, pose.p.z = index*10., .75
                handle = gym.create_actor(env, asset, pose, 'mass_check', 10+len(mass_checks), 0)
                prop = gym.get_actor_rigid_body_properties(env, handle)[0]
                bodies.append(dict(mass=prop.mass, com=[prop.com.x, prop.com.y, prop.com.z],
                    inertia=[prop.inertia.x.x, prop.inertia.y.y, prop.inertia.z.z], flags=prop.flags))
            for key in ('mass', 'com', 'inertia'):
                np.testing.assert_allclose(bodies[0][key], bodies[1][key], rtol=2e-5, atol=1e-5)
            assert bodies[0]['flags'] == bodies[1]['flags']
            mass_checks.append(dict(controlled=controlled, span=span, baseline=bodies[0], capped=bodies[1]))
    gym.prepare_sim(sim)
    for _ in range(30):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    for case in cases:
        state = gym.get_actor_rigid_body_states(case['env'], case['actor'], gymapi.STATE_ALL)
        state['vel']['linear']['x'] = 1.
        gym.set_actor_rigid_body_states(case['env'], case['actor'], state, gymapi.STATE_ALL)
    for _ in range(60):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        for case in cases:
            state = gym.get_actor_rigid_body_states(case['env'], case['actor'], gymapi.STATE_ALL)[0]
            case['trace'].append([float(state['pose']['p'][axis]) for axis in ('x', 'y', 'z')])
    results = [dict(label=c['label'], friction=c['friction'], slide_m=c['trace'][-1][0],
        settled_z_std_m=float(np.std(np.array(c['trace'])[30:, 2]))) for c in cases]
    np.testing.assert_allclose(results[0]['slide_m'], results[1]['slide_m'], rtol=.01, atol=1e-3)
    assert results[2]['slide_m'] < results[1]['slide_m']*.9
    assert all(r['settled_z_std_m'] < .001 for r in results)
    gym.destroy_sim(sim)
    report = dict(cpu_physics=True, not_policy_evaluation=True, material=results, mass_checks=mass_checks)
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
