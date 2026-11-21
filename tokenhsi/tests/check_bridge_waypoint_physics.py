import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym.torch_utils import quat_rotate
import torch

from qualitative.bridge_waypoint_viewer import environment
from qualitative.bridge_waypoint_physics import MODES, physics_info
from qualitative.bridge_waypoint_policy import WaypointPolicy
from qualitative.bridge_waypoint import bridge_goal


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bridge-physics', choices=MODES, required=True)
    parser.add_argument('--bridge-start', choices=('present', 'near', 'insert'), default='present')
    parser.add_argument('--cap-lift', type=float, default=0.)
    parser.add_argument('--out')
    opts = parser.parse_args()
    opts.envs, opts.seed, opts.headless = 4, 42, True
    opts.out = opts.out or 'output_etc/check/bridge_contact_20261117/reset_'+opts.bridge_physics
    torch.set_num_threads(4)
    task, env, output = environment(opts)
    policy = WaypointPolicy(task, torch.load(opts.checkpoint, map_location='cpu', weights_only=False))
    ids = torch.arange(task.num_envs, device=task.device)
    starts, records = [], []
    fixed = opts.bridge_physics in ('fixed', 'fixed_carry', 'platform')
    for episode in range(3):
        obs = env.reset()
        assert (task._bridge_start_mode == ('present', 'near', 'insert').index(opts.bridge_start)).all()
        starts.append(dict(root=task._root_states.cpu().clone(), dof=task._dof_state.cpu().clone(),
            goals=task._tar_pos.cpu().clone(), slots=task._bridge_slot.cpu().clone()))
        initial = task._box_states[ids, task._bridge_slot].clone()
        if opts.bridge_physics == 'follow_cap':
            cap = task._root_states[task._cap_actor_ids[ids, task._bridge_slot-task._bridge_bank]]
            offset = torch.zeros_like(initial[:, :3])
            offset[:, 2] = (task._box_size[ids, task._bridge_slot, 2]-.005)/2+opts.cap_lift
            expected = initial[:, :3]+quat_rotate(initial[:, 3:7], offset)
            torch.testing.assert_close(cap[:, :3], expected, rtol=0, atol=2e-5)
        props = physics_info(task)
        for row in props:
            if opts.bridge_physics in ('carry', 'fixed_carry', 'carry_nocap'):
                assert abs(row['density']-row['carry_density']) < 1e-3
                assert all(abs(p['friction']-row['carry_friction']) < 1e-6 for p in row['shapes'])
            if opts.bridge_physics == 'platform':
                assert len(row['shapes']) == 1
                terrain = task._compact_actor_ids[row['env'], task._compact_active[row['env']]][0]
                handle = int(terrain) - row['env']*task.gym.get_actor_count(task.envs[row['env']])
                platform = task.gym.get_actor_rigid_shape_properties(task.envs[row['env']], handle)[0]
                for key in ('friction', 'restitution', 'contact_offset', 'rest_offset'):
                    assert row['shapes'][0][key] == getattr(platform, key)
            elif opts.bridge_physics == 'follow_cap':
                assert len(row['shapes']) == 1
                assert abs(row['density']-3.) < 1e-3
                assert row['cap']['fixed_base'] and row['cap']['friction'] == 1.
            elif opts.bridge_physics.endswith('_nocap'):
                assert len(row['shapes']) == 1
                if opts.bridge_physics == 'baseline_nocap':
                    assert abs(row['density']-3.) < 1e-3
                    assert abs(row['shapes'][0]['friction']-.4) < 1e-6
            else:
                assert len(row['shapes']) == 2
        for step in range(20):
            leader = task._roles[:, 1]
            payload = task._agent_box_assignment[ids, leader]
            goal = bridge_goal(task._box_states[ids, task._bridge_slot],
                task._box_size[ids, task._bridge_slot], task._box_size[ids, payload])
            actions = policy.actions(obs, leader, goal-task._env_origins, torch.ones_like(ids, dtype=torch.bool))
            obs, _, _, _ = env.step(actions)
            assert torch.isfinite(obs).all()
            if fixed:
                torch.testing.assert_close(task._box_states[ids, task._bridge_slot], initial, rtol=0, atol=1e-6)
        roots = task._humanoid_root_states
        for person in (task._roles[:, 1], task._roles[:, 3]):
            assert (roots[ids, person, 2] > task._surface_height[:, 1]+.5).all()
        records.append(dict(episode=episode, physics=props, fixed_pose_preserved=fixed, platform_support=True))
    torch.save(starts, output/'initial_states.pt')
    (output/'checks.json').write_text(json.dumps(records, indent=2)+'\n')
    print('PASS: 4 environments x 3 resets; material, support and fixed poses checked', flush=True)
    task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
