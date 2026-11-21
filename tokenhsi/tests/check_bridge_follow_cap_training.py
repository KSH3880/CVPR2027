import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate
import torch

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.episode_layout_cache import buckets


@torch.no_grad()
def main():
    name = 'stage2_sensor_900_bridge_follow_cap_waypoint_progress'
    output = Path('output_etc/check/bridge_follow_cap_training_20261118/runtime')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = next(output.parent.glob('progress/*/nn/*.pth'))
    sys.argv = [sys.argv[0], '--task', 'HumanoidMACoupledPush', '--test', '--eval', '--headless',
        '--checkpoint', str(checkpoint),
        '--cfg_train', f'tokenhsi/data/cfg/train/rlg/{name}.yaml',
        '--cfg_env', f'tokenhsi/data/cfg/multi_agent/{name}.yaml',
        '--num_envs', '32', '--num_agents', '4', '--num_objects', '8', '--seed', '42',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--no_video', '--output_path', str(output)]
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    set_seed(42)
    torch.set_num_threads(4)
    task, env = parse_task(args, cfg, train, parse_sim_params(args, cfg, train))
    env.reset()
    ids = torch.arange(task.num_envs, device=task.device)
    task._prepare_layout_cache()
    # Evaluation has no training RSI counters; exercise only cache state restoration here.
    task._record_cached_sampling = lambda rows: None
    mode = ids % 3
    family = torch.tensor([1, 2, 4], device=task.device)[ids % 3]
    same = ids % 5 == 0
    alpha = torch.ones_like(ids, dtype=torch.float)
    kinds = buckets(mode, family, alpha, same, True)
    def restore(rows):
        task._layout_cache.restore(kinds[rows], rows, torch.zeros_like(rows), task._layout_tensors())
        task._restore_layouts(rows)
    restore(ids)
    assert torch.equal(task._before_same_platform, same)
    assert torch.equal(task._bridge_start_mode, mode)
    assert (task._bridge_pushers.sum(-1) == 2).all()
    original_goals = task._tar_pos.clone()
    root_before = task._humanoid_root_states.clone()
    body_before = task._box_states.clone()
    edges = task._leader_at_edges(ids)
    leader = task._roles[:, 1]
    payload = task._agent_box_assignment[ids, leader]
    graph = task.relation_runtime.graph
    assert (graph.edge_owner[ids, edges] == leader).all()
    assert (graph.edge_relation[ids, edges] == 7).all()
    goal, _, _, radius = task._waypoint_geometry(ids)
    task._waypoint_switched.zero_()
    totals = []
    for offset in (.0001, -.0001, 2.):
        task._box_states[ids, payload, :3] = goal
        task._box_states[ids, payload, 0] += radius+offset
        task._compute_reward(torch.zeros(task.num_envs*4, task.num_dof, device=task.device))
        result = task.relation_runtime.last_result
        runtime = task.relation_runtime
        phi, diag = task._evaluate_relations()
        baseline = runtime.components(runtime._gated(phi, graph, runtime.gate),
            runtime._gated(diag['progress'], graph, runtime.gate), diag['z_error'],
            diag['feet_height_error'], diag['region_error'], graph)
        unchanged = torch.ones_like(graph.edge_valid)
        unchanged[ids[~same], edges[~same]] = False
        torch.testing.assert_close(result['progress_component'][unchanged], baseline['progress_component'][unchanged])
        for key in ('state_component', 'success_component', 'own_success'):
            torch.testing.assert_close(result[key], baseline[key])
        total = result['progress_component'][ids, edges]+task._compute_bridge_center_progress()[ids, leader]
        totals.append(total.clone())
        task._compute_observations()
        assert not task.obs_buf[same, -4].any()
        assert not task._waypoint_switched[same].any()
        assert torch.equal(task._tar_pos, original_goals)
        if offset < 0.:
            assert task._waypoint_switched[~same].all()
            torch.testing.assert_close(total[~same], torch.full_like(total[~same], .2), atol=1e-5, rtol=0)
    assert (totals[1][~same] >= totals[0][~same]-1e-5).all()
    assert (totals[2][~same] >= .2).all()
    untouched = ids[1::2]
    latch = task._waypoint_switched[untouched].clone()
    reference = task._waypoint_reference[untouched].clone()
    restore(ids[::2])
    assert torch.equal(task._waypoint_switched[untouched], latch)
    assert torch.equal(task._waypoint_reference[untouched], reference)
    torch.testing.assert_close(task._humanoid_root_states[ids[::2]], root_before[ids[::2]], rtol=0, atol=0)
    torch.testing.assert_close(task._box_states[ids[::2]], body_before[ids[::2]], rtol=0, atol=0)
    restore(ids)
    _, initial_switch, _, _ = task._waypoint_geometry(ids)
    assert torch.equal(initial_switch, task._waypoint_switched)
    actors = task._cap_actor_ids[ids, task._bridge_slot-task._bridge_bank]
    cap_bodies = []
    for i in range(task.num_envs):
        handle = task._cap_handles[i*3+int(task._bridge_slot[i])-task._bridge_bank]
        cap_bodies.append(task.gym.get_actor_rigid_body_index(task.envs[i], handle, 0, gymapi.DOMAIN_SIM))
    cap_bodies = torch.tensor(cap_bodies, device=task.device)
    errors = []
    for step in range(4):
        task._box_states[ids, task._bridge_slot, 0] += .15
        task._box_states[ids, task._bridge_slot, 2] += .1
        task._sync_follow_cap(reset=True)
        expected = task._root_states[actors, :7].clone()
        bridge = task._box_states[ids, task._bridge_slot]
        height = task._box_size[ids, task._bridge_slot, 2]
        offset = torch.zeros_like(bridge[:, :3])
        offset[:, 2] = (height-.005)/2+.05
        torch.testing.assert_close(expected[:, :3], bridge[:, :3]+quat_rotate(bridge[:, 3:7], offset))
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()
        errors.append(float((task._rigid_body_state[cap_bodies, :7]-expected).abs().max()))
        torch.testing.assert_close(task._rigid_body_state[cap_bodies, :7], expected, atol=1e-4, rtol=0)
    report = dict(environments=32, cache_layouts_per_env=36, same_platform=int(same.sum()),
        cap_lift=.05, cap_pose_error_max=max(errors), staged_reward_continuous=True,
        unchanged_other_reward_components=True, same_platform_native_reward=True,
        final_goals_unchanged=True, partial_reset_isolated=True, cached_roots_exact=True,
        observation_width=task.obs_buf.shape[1])
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
