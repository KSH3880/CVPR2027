import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_euler_xyz
import torch

from qualitative.bridge_waypoint_cap import CAP_FILTER
from qualitative.bridge_waypoint_viewer import environment


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cap-lift', type=float, default=0.)
    parser.add_argument('--out', default='output_etc/check/bridge_contact_20261117/cap_motion')
    opts = parser.parse_args()
    opts.envs, opts.seed, opts.headless, opts.bridge_physics = 4, 42, True, 'follow_cap'
    torch.set_num_threads(4)
    task, env, output = environment(opts)
    env.reset()
    ids = torch.arange(task.num_envs, device=task.device)
    slots = task._bridge_slot
    actors = task._cap_actor_ids[ids, slots-task._bridge_bank]
    cap_bodies = []
    for i in range(task.num_envs):
        cap = task._cap_handles[i*3+int(slots[i])-task._bridge_bank]
        cap_bodies.append(task.gym.get_actor_rigid_body_index(task.envs[i], cap, 0, gymapi.DOMAIN_SIM))
        for actor in range(task.gym.get_actor_count(task.envs[i])):
            props = task.gym.get_actor_rigid_shape_properties(task.envs[i], actor)
            paired = actor in task._cap_handles[i*3:i*3+3] or actor in task._box_handles[i*task.num_physical_objects+task._bridge_bank:i*task.num_physical_objects+task._bridge_bank+3]
            assert all(bool(prop.filter & CAP_FILTER) == paired for prop in props)
    cap_bodies = torch.tensor(cap_bodies, device=task.device)
    original = task._box_states[ids, slots].clone()
    errors, falls = [], []
    for step in range(20):
        body = original.clone()
        body[:, 0] += .05*step
        body[:, 1] += 20.
        body[:, 2] += 5.
        angle = torch.full_like(ids, step*.025, dtype=torch.float)
        body[:, 3:7] = quat_from_euler_xyz(angle*.25, -angle*.2, angle)
        body[:, 7:13] = 0.
        task._box_states[ids, slots] = body
        task._sync_follow_cap(reset=True)
        expected = task._root_states[actors, :7].clone()
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()
        actual = task._rigid_body_state[cap_bodies, :7]
        errors.append(float((actual-expected).abs().max()))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
        assert (task._root_states[actors, 7:13].abs() < 1e-6).all()
        velocity = task._box_states[ids, slots, 9]
        assert (velocity < -.01).all()
        falls.append(velocity.cpu().tolist())
    result = dict(environments=4, moving_poses=20, position_rotation_max_error=max(errors),
        cap_velocity_zero=True, body_remains_dynamic=True, only_bridge_cap_filter_shared=True,
        body_vertical_velocities=falls)
    (output/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
