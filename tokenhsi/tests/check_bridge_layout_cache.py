import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.episode_layout_cache import buckets
from env.tasks.multi_agent import square_sensor_scene as scene


def check_before_mix(task):
    ids = torch.arange(task.num_envs, device=task.device)
    leader, follower = task._roles[:, 1], task._roles[:, 3]
    if task._state_init == task.StateInit.Default:
        torch.testing.assert_close(task._tar_pos[ids, follower],
            task._humanoid_root_states[ids, follower, :3], rtol=0, atol=1e-3)
        nodes = sum(size*count for size, count in zip(task.get_scene_entity_sizes(),
                    (task.num_agents, task.num_objects, task.num_agents)))
        poses = task.obs_buf[:, nodes:nodes+(2*task.num_agents+task.num_objects)*7].view(task.num_envs, -1, 7)
        torch.testing.assert_close(poses[ids, task.num_agents+task.num_objects+follower, :3],
                                  poses[ids, follower, :3], rtol=0, atol=1e-3)
    same = task._before_same_platform
    source = task._start_surface[ids, leader]
    target = torch.where(same, source, 3-source)
    assert torch.equal(task._surface[ids, leader], target)
    assert torch.equal(task._start_surface[ids, follower], target)
    assert task._bridge_pushers.sum(-1).eq(2).all()
    assert task._templates[task._bridge_pushers].eq(5).all()
    torch.testing.assert_close(task._center_tar_pos[task._bridge_pushers],
        task._bridge_landing_world[:, None].expand(-1, 4, -1)[task._bridge_pushers])
    payload = task._agent_box_assignment[ids, leader]
    all_envs = torch.ones_like(same)
    positions = [(task._kinematic_humanoid_rigid_body_states[ids, leader, 0, :2], source, all_envs),
                 (task._box_states[ids, payload, :2], source, all_envs),
                 (task._kinematic_humanoid_rigid_body_states[ids, follower, 0, :2], target, all_envs),
                 (task._tar_pos[ids, leader, :2], target, all_envs)]
    own = task._agent_box_assignment[ids, follower]
    ontop = task._templates[ids, follower] == 4
    for xy, surface, selected in positions+[(task._box_states[ids, own, :2], target, ontop)]:
        local = scene.rotate_xy(xy-task._env_origins[:, :2]-task._structure_shift[:, 1],
                                -task._structure_yaw[:, 1])
        center = task._surface_center[ids, 1, surface-1]
        half = task._surface_half[ids, 1, surface-1]
        assert (((local-center).abs() <= half+1e-4).all(-1) | ~selected).all()
    reward = task._compute_bridge_center_progress()
    assert reward[same].eq(0).all()
    others = torch.ones_like(reward, dtype=torch.bool)
    others[ids[~same], leader[~same]] = False
    assert reward[others].eq(0).all()
    assert (reward[ids[~same], leader[~same]] > 0).all()
    return int(same.sum())


@torch.no_grad()
def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    set_seed(args.seed)
    task, env = parse_task(args, cfg, train['params'], parse_sim_params(args, cfg, train))
    env.reset()
    stored = task._layout_cache.data['tensors']
    roots = stored['actors'][..., :task.num_agents, :3]
    followers = stored['_before_predecessor'] >= 0
    target_error = (stored['_tar_pos']-roots).norm(dim=-1)[followers]
    if task._state_init == task.StateInit.Default and target_error.max() > 1e-3:
        raise RuntimeError('Invalid inactive Target coordinates in prepared cache')
    ids = torch.arange(0, task.num_envs, 2, device=task.device)
    other = torch.arange(1, task.num_envs, 2, device=task.device)
    untouched = {name: value[other].clone() for name, value in task._layout_tensors().items()}
    modes = torch.ones_like(ids)
    families = torch.full_like(ids, 4)
    alpha = torch.ones(len(ids), device=task.device)
    same = torch.zeros_like(ids, dtype=torch.bool)
    task._generate_layouts(ids, torch.ones_like(ids), torch.full_like(ids, 4),
                           alpha)
    expected = {name: value[ids].clone() for name, value in task._layout_tensors().items()}
    expected_obs = task.obs_buf[ids].clone()
    expected_runtime = {name: value[ids].clone() for name, value in vars(task.relation_runtime).items()
                        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == task.num_envs}
    bucket = buckets(modes, families, alpha, same, task._layout_final_only)
    slots = task._layout_cache.data['next'][ids, bucket].clone()
    task._layout_cache.store(bucket, ids, alpha, task._layout_tensors())
    task._generate_layouts(ids, torch.full_like(ids, 2), torch.ones_like(ids),
                           torch.ones(len(ids), device=task.device))
    task._layout_cache.restore(bucket, ids, slots, task._layout_tensors())
    task._restore_layouts(ids)
    for name, value in task._layout_tensors().items():
        torch.testing.assert_close(value[ids], expected[name], rtol=0, atol=1e-5, msg=name)
        torch.testing.assert_close(value[other], untouched[name], rtol=0, atol=1e-5, msg='partial/'+name)
    torch.testing.assert_close(task.obs_buf[ids], expected_obs, rtol=0, atol=1e-5)
    for name, value in expected_runtime.items():
        torch.testing.assert_close(getattr(task.relation_runtime, name)[ids], value, rtol=0, atol=1e-5, msg='runtime/'+name)
    starts = torch.zeros(3, dtype=torch.long, device=task.device)
    followers = torch.zeros(5, dtype=torch.long, device=task.device)
    task._layout_spec['refresh_probability'] = 0.
    same_count = 0
    generate = task._generate_layouts
    if task._layout_final_only:
        if not task._layout_cache.data['alpha'].eq(1).all():
            raise RuntimeError('Final-distance cache was not fully prepared')
        def unexpected_generation(*args, **kwargs):
            raise RuntimeError('Cached reset unexpectedly generated a new layout')
        task._generate_layouts = unexpected_generation
    for _ in range(10):
        env.reset()
        same_count += check_before_mix(task)
        starts += torch.bincount(task._bridge_start_mode, minlength=3)
        family = task._templates[torch.arange(task.num_envs, device=task.device), task._roles[:, 3]]
        followers += torch.bincount(family, minlength=5)
        if not torch.isfinite(task.obs_buf).all() or task._place_failed.any():
            raise RuntimeError('Invalid cached reset')
    for update in (0, 501, 1400, 2500):
        task._bridge_curriculum_update = update
        env.reset()
        alpha = task._compact_distance.clone()
        expected_alpha = (0., min(1., max(0., (update-500)/2000)), 1.)
        valid = alpha.eq(1) if task._layout_final_only else torch.stack([alpha.eq(a) for a in expected_alpha]).any(0)
        if not valid.all():
            raise RuntimeError('Reset distance differs from the configured distribution')
    task._generate_layouts = generate
    obs, _, _, _ = env.step(torch.zeros(task.num_envs*task.num_agents, task.num_dof, device=task.device))
    if not torch.isfinite(obs).all():
        raise RuntimeError('Nonfinite simulation after cache restore')
    result = dict(environments=task.num_envs, exact_restore_fields=len(expected),
                  exact_runtime_fields=len(expected_runtime), partial_reset=True, amp_history_exact=True,
                  starts=starts.tolist(), followers=followers[[1, 2, 4]].tolist(),
                  final_distance_only=task._layout_final_only,
                  cache_buckets=task._layout_cache.bucket_count,
                  cache_neutral_target_max_error=float(target_error.max()),
                  reset_generation_disabled=task._layout_final_only, step_finite=True)
    result['same_platform_episodes'] = same_count
    Path(args.output_path, 'probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
