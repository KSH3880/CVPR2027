import json
from pathlib import Path
import statistics
import sys
import time

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import isaacgym
import torch

from env.tasks.multi_agent.scene_validation import scene_infeasible
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from env.tasks.multi_agent.warehouse_before_cache import STATE_FIELDS, CONTEXT_FIELDS
from check_warehouse_beforepairs import inspect
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task


def main():
    args = get_args()
    torch.set_num_threads(4)
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    env.reset()
    stats = dict(envs=0, flagged=0, followers=dict(SIT=0, CLIMB=0, ON_TOP=0), root_height_above_apron_min=1e9)
    inspect(task, stats)
    cache = task._warehouse_before_cache
    ids = task._warehouse_before_pairs.nonzero().flatten()
    untouched = (~task._warehouse_before_pairs).nonzero().flatten()
    names = (*STATE_FIELDS, *CONTEXT_FIELDS, '_hist_amp_obs_buf')
    expected = {name: getattr(task, name)[ids].clone() for name in names}
    other = {name: getattr(task, name)[untouched].clone() for name in names}
    sample, place = task._sample_episode_graph, task._place_scene
    task._sample_episode_graph = lambda selected: None
    cache.refresh_probability = 0.

    def unexpected_place(*args, **kwargs):
        raise RuntimeError('A cache hit entered the layout sampler')

    task._place_scene = unexpected_place
    for name in (*STATE_FIELDS, '_hist_amp_obs_buf'):
        getattr(task, name)[ids] = -999.
    task._reset_ontop_context_envs(ids)
    for name in names:
        torch.testing.assert_close(getattr(task, name)[ids], expected[name])
        torch.testing.assert_close(getattr(task, name)[untouched], other[name])
    assert not scene_infeasible(task, ids).any()
    assert not invalid_before_layout(task, ids).any()
    task._place_scene = place
    timings = {}
    for label, probability in (('fresh', 1.), ('cached', 0.)):
        cache.refresh_probability = probability
        seconds = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            task._reset_ontop_context_envs(ids)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter()-start)
            assert not scene_infeasible(task, ids).any()
            assert not invalid_before_layout(task, ids).any()
        timings[label] = statistics.mean(seconds)
    inspect(task, stats)
    task._sample_episode_graph = sample
    cache.refresh_probability = .25
    action = torch.zeros(task.num_envs*task.num_agents, task.get_action_size(), device=task.device)
    for _ in range(30):
        obs, reward, _, _ = env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    stats.update(apron_envs=len(ids), same_condition_reset_seconds=timings,
                 all_hit_restored_state_and_amp=True, non_reset_state_unchanged=True,
                 cache_entries=sum(len(pool) for entries in cache.entries for pool in entries.values()))
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    (output/'cache_check.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
