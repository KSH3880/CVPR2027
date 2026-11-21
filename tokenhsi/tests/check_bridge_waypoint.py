import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch

from qualitative.bridge_waypoint import bridge_goal
from qualitative.bridge_waypoint_policy import WaypointPolicy
from qualitative.bridge_waypoint_viewer import environment
from env.tasks.multi_agent import square_sensor_scene as scene


@torch.no_grad()
def main():
    opts = SimpleNamespace(checkpoint=sys.argv[1], out='output_etc/check/bridge_waypoint/geometry',
                           seed=42, envs=8, headless=True)
    torch.set_num_threads(4)
    task, env, output = environment(opts)
    source = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
    policy = WaypointPolicy(task, source)
    obs = env.reset()
    ids = torch.arange(task.num_envs, device=task.device)
    leader = task._roles[:, 1]
    payload = task._agent_box_assignment[ids, leader]
    bridge = task._box_states[ids, task._bridge_slot]
    yaw = task._structure_yaw[:, 1]
    positions = torch.stack((task._humanoid_root_states[ids, leader, :2],
        task._box_states[ids, payload, :2], task._tar_pos[ids, leader, :2]), 1)
    local = scene.rotate_xy(positions-bridge[:, None, :2], -yaw[:, None])
    lateral_error = local[..., 1].abs().max()
    assert lateral_error < 1e-4
    assert (local[:, 0, 0]*local[:, 1, 0] > 0).all()
    assert (local[:, 1, 0]*local[:, 2, 0] < 0).all()
    assert (local[:, 0, 0].abs() > local[:, 1, 0].abs()).all()
    assert not task._before_same_platform.any()
    assert task._bridge_start_mode.eq(0).all()
    assert task._bridge_pushers.sum(-1).eq(2).all()
    torch.testing.assert_close(bridge[:, :3], task._bridge_landing_world, rtol=0, atol=1e-4)
    goal = bridge_goal(bridge, task._box_size[ids, task._bridge_slot], task._box_size[ids, payload])
    original = task._tar_pos.clone()
    rng = torch.cuda.get_rng_state()
    reference = policy.net.eval_actor(policy.rms(obs[:, :policy.width]))[0].clamp(-1, 1)
    torch.cuda.set_rng_state(rng)
    guided = policy.actions(obs, leader, goal-task._env_origins, torch.ones_like(ids, dtype=torch.bool))
    others = torch.ones(task.num_envs, 4, device=task.device, dtype=torch.bool)
    others[ids, leader] = False
    torch.testing.assert_close(guided.view(task.num_envs, 4, -1)[others],
                              reference.view(task.num_envs, 4, -1)[others], rtol=0, atol=0)
    torch.cuda.set_rng_state(rng)
    restored = policy.actions(obs, leader, goal-task._env_origins, torch.zeros_like(ids, dtype=torch.bool))
    torch.testing.assert_close(restored, reference, rtol=0, atol=0)
    torch.testing.assert_close(task._tar_pos, original, rtol=0, atol=0)
    changed = (guided.view(task.num_envs, 4, -1)[ids, leader]-reference.view(task.num_envs, 4, -1)[ids, leader]).abs()
    assert (changed.amax(-1) > 1e-5).all()
    result = dict(envs=task.num_envs, centerline_error=float(lateral_error),
        original_model_tensors=len(source['model']), nonguided_actions_exact=True,
        switched_actions_exact=True, actual_goals_exact=True,
        leader_action_change_max=float(changed.max()), initial_goal=goal.cpu().tolist(),
        initial_positions=positions.cpu().tolist())
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
