"""1024-env geometry, state/cache preservation and reset latency checks."""
import json
from pathlib import Path
import statistics
import sys
import time
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import isaacgym
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.scene_validation import scene_infeasible
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from env.tasks.multi_agent.warehouse_before_cache import STATE_FIELDS, CONTEXT_FIELDS
from utils.sensor_stage2_spec import WAREHOUSE
from utils.warehouse_waypoint import ROLE_OBS_WIDTH, role_eligibility


def inspect(task, obs):
    ids = torch.arange(task.num_envs, device=task.device)
    pairs = task._warehouse_before_pairs
    metadata = obs[:, -task._waypoint_obs_width:]
    assert (metadata[pairs, :2] == -1).all()
    report = {'before_pairs': int(pairs.sum())}
    if task._waypoint_obs_width == ROLE_OBS_WIDTH:
        people, eligible = role_eligibility(task._scene_ids, pairs, task._before_predecessor, task._templates)
        torch.testing.assert_close(metadata[:, [0, 5]].long(), people.masked_fill(~eligible, -1))
        assert (metadata[:, [1, 6]][~eligible] == -1).all()
        assert (metadata[:, 2:5][pairs] == 0).all()
        report['ontop_followers'] = int(eligible[:, 1].sum())
    ids = ids[~pairs]
    pre = task._before_predecessor[ids]
    follower = (pre >= 0).long().argmax(-1)
    leader = pre.gather(1, follower[:, None]).squeeze(1)
    roots = task._humanoid_root_states[ids, :, :2]
    origin = task._env_origins[ids, :2]+task._structure_shift[ids, WAREHOUSE]
    yaw = task._structure_yaw[ids, WAREHOUSE]
    local = scene.rotate_xy(roots-origin[:, None], -yaw[:, None])
    local[..., 0] += task._surface_half[ids, WAREHOUSE, 0, 0, None]
    rows = torch.arange(len(ids), device=task.device)
    p = local[rows, leader]
    assert ((p[:, 0] >= -3.501) & (p[:, 0] <= -2.499)).all()
    assert ((p[:, 1].abs() >= 2.399) & (p[:, 1].abs() <= 3.201)).all()
    slot = task._agent_box_assignment[ids, leader]
    box = scene.rotate_xy(task._box_states[ids, slot, :2]-origin, -yaw)
    half = task._surface_half[ids, WAREHOUSE, 0]
    box[:, 0] += half[:, 0]
    size = task._box_size[ids, slot, :2]/2
    gap = torch.stack((box[:, 0]-size[:, 0], 2*half[:, 0]-box[:, 0]-size[:, 0],
                       half[:, 1]-box[:, 1].abs()-size[:, 1]), -1)
    assert (gap >= 1.199).all()
    assert not scene_infeasible(task, torch.arange(task.num_envs, device=task.device)).any()
    assert not invalid_before_layout(task, torch.arange(task.num_envs, device=task.device)).any()
    return dict(report, min_wall_clearance=float(gap.min()))


def main():
    args = get_args(); torch.set_num_threads(4)
    cfg, train, _ = load_cfg(args); cfg['env']['motion_file'] = args.motion_file
    set_seed(args.seed)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    try:
        obs = env.reset(); report = inspect(task, obs)
        cache = task._warehouse_before_cache
        cache.refresh_probability = 0.
        ids = torch.arange(task.num_envs, device=task.device)
        # Freeze already sampled conditions to measure restoration itself.
        sample = task._sample_episode_graph
        task._sample_episode_graph = lambda selected: None
        # Seed any missing keys and synchronize the restored state with Isaac Gym.
        task._reset_ontop_context_envs(ids)
        names = (*STATE_FIELDS, *CONTEXT_FIELDS, '_hist_amp_obs_buf')
        for size in (16, 64, 256, 1024):
            selected = ids[:size]
            kept = {name: getattr(task, name)[size:].clone() for name in names}
            task._waypoint_phase[:] = 3
            task._waypoint_phase[selected] = 0
            task._reset_ontop_context_envs(selected)
            assert len(cache.hits) == size
            for name, value in kept.items():
                restored = getattr(task, name)
                torch.testing.assert_close(restored[cache.hits], cache.data[name][cache.hit_indices],
                                           atol=.001, rtol=1e-6,
                                           msg=lambda message: f'{name} after cache restore: {message}')
                torch.testing.assert_close(restored[size:], value, atol=.001, rtol=1e-6,
                                           msg=lambda message: f'{name} outside reset: {message}')
            assert (task._waypoint_phase[size:] == 3).all()
        timings = {}
        for size in (16, 64, 256, 1024):
            for mode, refresh in (('fresh', 1.), ('cached', 0.)):
                cache.refresh_probability = refresh
                seconds = []
                for _ in range(5):
                    torch.cuda.synchronize(); started = time.perf_counter()
                    task._reset_ontop_context_envs(ids[:size])
                    torch.cuda.synchronize(); seconds.append(time.perf_counter()-started)
                timings[f'{size}/{mode}'] = dict(median=statistics.median(seconds), p95=max(seconds), samples=seconds)
        task._sample_episode_graph = sample
        cache.refresh_probability = .25
        reset_reports = []
        for _ in range(4):
            obs = env.reset(ids*4); reset_reports.append(inspect(task, obs))
            assert torch.isfinite(obs).all()
        action = torch.zeros(task.num_envs*4, task.get_action_size(), device=task.device)
        for _ in range(3):
            obs, reward, done, _ = env.step(action)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        report.update(resets=reset_reports, reset_timings=timings, states_and_amp_restored=True,
                      non_reset_preserved=True, finite_physics=True, cache_warmup_seconds=cache.warmup_seconds)
        path = Path(args.output_path); path.mkdir(parents=True, exist_ok=True)
        (path/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        task.gym.destroy_sim(task.sim)

if __name__ == '__main__': main()
