"""Compare passive release of the real door asset; no humans or training."""
from isaacgym import gymapi
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from utils.door_hold import closing_torque


def main():
    root = Path(__file__).resolve().parents[3]
    experiment = os.environ.get('DOOR_COMPARE_EXPERIMENT', 'open_toy_strong_close')
    configs = [yaml.safe_load((root / f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())['env']
               for name in ('open_toy', experiment)]
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = False
    params.physx.num_threads = 1
    params.physx.num_position_iterations = 4
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    asset = gym.load_asset(sim, str(root / 'tokenhsi/assets/door'), 'door_hold.urdf', options)
    cases = []
    for config in configs:
        for angle, velocity, tap in ((math.pi/2, 0., 0.), (math.pi/2, 3., 0.), (0., 0., 2.), (0., 0., 20.)):
            env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 3), 2)
            actor = gym.create_actor(env, asset, gymapi.Transform(), 'door', len(cases), 0)
            props = gym.get_actor_dof_properties(env, actor)
            props['driveMode'] = gymapi.DOF_MODE_EFFORT
            props['stiffness'] = 0
            props['damping'] = config['door']['hinge']['damping']
            props['friction'] = config['door']['hinge']['friction']
            gym.set_actor_dof_properties(env, actor, props)
            state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            state['pos'] = angle
            state['vel'] = velocity
            gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
            cases.append(dict(env=env, actor=actor, spring=config['door']['autoClose'],
                              initial_angle=angle, initial_velocity=velocity, tap=tap, angles=[]))
    gym.prepare_sim(sim)
    try:
        for tick in range(600):
            for case in cases:
                state = gym.get_actor_dof_states(case['env'], case['actor'], gymapi.STATE_ALL)
                torque = closing_torque(torch.tensor(state['pos'].copy()),
                                        torch.tensor(state['vel'].copy()), case['spring'])
                if tick < 9:
                    torque += case['tap']  # 150ms opening impulse, then release.
                gym.apply_actor_dof_efforts(case['env'], case['actor'], torque.numpy().astype(np.float32))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            for case in cases:
                angle = float(gym.get_actor_dof_states(case['env'], case['actor'], gymapi.STATE_POS)['pos'][0])
                assert math.isfinite(angle)
                case['angles'].append(angle)
        report = []
        for case in cases:
            angles = case['angles']
            closed = next((i / 60 for i, a in enumerate(angles, 1) if a < math.radians(10)), None)
            report.append(dict(spring=case['spring'], initial_angle_deg=math.degrees(case['initial_angle']),
                               initial_velocity_rad_s=case['initial_velocity'],
                               tap_torque_Nm=case['tap'], tap_duration_s=.15,
                               max_angle_deg=math.degrees(max(angles)),
                               first_below_10_deg_seconds=closed,
                               angle_at_1s_deg=math.degrees(angles[59]),
                               final_angle_deg=math.degrees(angles[-1])))
        for old, new in zip(report[:4], report[4:]):
            if old['initial_angle_deg'] > 0 and experiment == 'open_toy_strong_close':
                assert new['first_below_10_deg_seconds'] < old['first_below_10_deg_seconds']
            elif old['initial_angle_deg'] == 0 and experiment == 'open_toy_strong_close':
                assert new['max_angle_deg'] < old['max_angle_deg']
                if old['tap_torque_Nm'] == 2:
                    assert new['max_angle_deg'] < .5
                else:
                    assert new['max_angle_deg'] > 1, 'Door must remain openable by sufficient force'
            assert abs(new['final_angle_deg']) < 1
        out = root / os.environ.get('OUTPUT_PATH', f'output/{experiment}_physics_check')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'audit.json').write_text(json.dumps(dict(status='passed', training=False, cases=report), indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
