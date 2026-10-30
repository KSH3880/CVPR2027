"""CPU inference check of fixed targets, reset isolation, AT geometry and markers."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from torch import nn
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.coupled_push_spec import region_states
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.stage2_push_network import Stage2PushBuilder, transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def inspect(task):
    whole = task._assigned_box_values(task._box_states)
    cells = task._logical_box_values(task._box_states)[:, :4]
    goals = task._task_goal_positions()
    delta = task._tar_pos - whole[..., :3]
    torch.testing.assert_close(goals - cells[..., :3], delta, atol=5e-5, rtol=1e-5)
    push = task._templates_by_owner == 5
    distance = delta[..., :2].norm(dim=-1)
    assert ((distance[push] >= 2-5e-5) & (distance[push] <= 10+5e-5)).all()
    shared = task._members > 1
    if shared.any():
        assert (task._region_goal_offsets[shared].abs().sum(-1) > 0).any()
        assert not task._region_goal_offsets[~shared].any()
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(-1, 4, 5)
    assert (packet[..., 4][push] == torch.arange(12, 16, device=task.device)[None].expand(task.num_envs, -1)[push]).all()
    obs = task._compute_clean_scene_obs()
    offset = 4*223 + 8*30 + 4
    poses = obs[:, offset:offset+16*7].reshape(-1, 16, 7)
    torch.testing.assert_close(poses[:, -4:, :3], goals-task._env_origins[:, None], atol=5e-5, rtol=1e-5)
    assert torch.equal(task._evaluation_box_values(task._box_states), task._logical_box_values(task._box_states))
    return dict(shared_people=int(shared.sum()), min_push_distance=float(distance[push].min()) if push.any() else None,
                max_push_distance=float(distance[push].max()) if push.any() else None)


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test = args.eval = True
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    assert task._region_goals
    builder = Stage2PushBuilder()
    builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=4, num_objects=8,
        observation_mode='clean_scene', scene_entity_sizes=task.get_scene_entity_sizes(),
        scene_kinematic_size=7, scene_arena_scale=task.get_scene_arena_scale(),
        relation_reward_mode=task._relation_cfg['mode'], relation_graph_spec=task._relation_graph_spec,
        device=task.device).to(task.device)
    model = nn.Module()
    model.a2c_network = net
    checkpoint = torch.load(args.checkpoint, map_location=task.device, weights_only=False)
    report = transfer_stage1(model, checkpoint)
    assert report['source_epoch'] == 21000 and len(report['copied']) == 165
    rms = SceneRunningMeanStd([223, 30, 1], [4, 8, 4], normalized_sizes=[223, 30, 0],
        kinematic_size=7, extra_passthrough_size=24).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    rms.eval()
    model.eval()
    saved_weights = {k: v.clone() for k, v in model.state_dict().items()}
    cases = []
    with torch.no_grad():
        for preset in ('all_coupled', 'random_scenario', 'push', 'holding_at'):
            task._task_graph_preset = preset
            obs = env.reset()
            case = inspect(task)
            goals = task._task_goal_positions().clone()
            for _ in range(8):
                mu, _ = net.eval_actor(rms(obs))
                value = net.eval_critic(rms(obs))
                obs, reward, done, _ = env.step(mu)
                assert all(torch.isfinite(x).all() for x in (obs, mu, value, reward))
                if not done.any():
                    torch.testing.assert_close(task._task_goal_positions(), goals, atol=0, rtol=0)
                goals = task._task_goal_positions().clone()
            untouched = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
            old_goals = task._task_goal_positions()[1:].clone()
            old_offsets = task._region_goal_offsets[1:].clone()
            task._reset_envs(torch.tensor([0], device=task.device))
            assert torch.equal(untouched, task._root_states.view(task.num_envs, -1, 13)[1:])
            assert torch.equal(old_goals, task._task_goal_positions()[1:])
            assert torch.equal(old_offsets, task._region_goal_offsets[1:])
            cases.append(dict(preset=preset, finite=True, fixed_goals=True, partial_reset=True, **case))
        # Verify native AT reward on physical simulator states at the intended pose.
        task._task_graph_preset = 'all_coupled'
        env.reset()
        assigned = task._agent_box_assignment
        saved = task._box_states.clone()
        for owner in range(4):
            rows = torch.arange(task.num_envs, device=task.device)
            task._box_states[rows, assigned[:, owner], :3] = task._tar_pos[:, owner]
        phi, diag = task._evaluate_ontop_context()
        at = task.relation_runtime.graph.edge_relation == 7
        torch.testing.assert_close(phi[at], torch.ones_like(phi[at]), atol=1e-5, rtol=0)
        assert task.relation_runtime.gate.all()
        # Turn at the target center: goals stay fixed and outer cells lose success.
        goals = task._task_goal_positions().clone()
        turn = .5
        q = task._box_states[..., 3:7].clone()
        task._box_states[..., 5] = q[..., 2]*torch.cos(torch.tensor(turn/2))+q[..., 3]*torch.sin(torch.tensor(turn/2))
        task._box_states[..., 6] = q[..., 3]*torch.cos(torch.tensor(turn/2))-q[..., 2]*torch.sin(torch.tensor(turn/2))
        phi, diag = task._evaluate_ontop_context()
        assert (diag['distance'][at] > .2).any()
        assert torch.equal(goals, task._task_goal_positions())
        task._box_states.copy_(saved)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved_weights.items())
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(source_epoch=21000, optimizer_steps=0, checkpoint_created=False,
        weights_unchanged=True, region_at_reward=True, rotation_changes_error=True,
        cases=cases)
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('REGION_GOALS_CHECK', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
