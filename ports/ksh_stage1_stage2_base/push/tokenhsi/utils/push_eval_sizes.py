"""Physical size overrides for loco-only five-task PUSH evaluation."""
import json
import math
import os
from pathlib import Path

import torch


def parse_range(value):
    values = [float(x) for x in value.split(',')]
    if len(values) == 1:
        values *= 2
    if len(values) != 2 or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('PUSH_EVAL_SIZE_X/Y/Z must be positive min,max or a fixed size in metres')
    lo, hi = values
    if hi < lo or abs((hi-lo)/.05-round((hi-lo)/.05)) > 1e-5:
        raise ValueError('Size ranges must increase on a 0.05m grid')
    return values


def override_sizes(task, candidates):
    requested = {axis: os.environ.get('PUSH_EVAL_SIZE_'+axis, '') for axis in 'XYZ'}
    if not any(requested.values()):
        return candidates
    args = task.cfg['args']
    if (not args.eval or task.__class__.__name__ != 'HumanoidMAFiveTask'
            or task._task_graph_preset != 'push_at'):
        raise ValueError('Size override requires five-task PUSH --test --eval with TASK_GRAPH=push_at')
    evaluation = task.cfg['env']['eval']
    if any(skill != 'loco' and prob > 0 for skill, prob in
           zip(evaluation['skill'], evaluation['skillInitProb'])):
        raise ValueError('Unscreened size zero-shot requires EVAL_SKILLS=loco (no PUSH RSI)')
    if os.environ.get('PUSH_GOAL_SWITCH_ANGLES') or os.environ.get('PUSH_BOX_SWITCH_MODES'):
        raise ValueError('Size zero-shot uses fixed goals/boxes; unset the relocation options')
    ranges = {axis: parse_range(value) for axis, value in requested.items() if value}
    sizes = candidates.clone()
    for index, axis in enumerate('XYZ'):
        if axis in ranges:
            lo, hi = ranges[axis]
            levels = round((hi-lo)/.05)+1
            sizes[:, :task.num_agents, index] = lo+.05*torch.randint(
                levels, sizes[:, :task.num_agents, index].shape, device=sizes.device)
    task._push_eval_size_report = dict(
        ranges_m=ranges, interval_m=.05,
        sampled_sizes_m=sizes[:, :task.num_agents].cpu().tolist(),
        training_push_box=task.cfg['env']['pushBox'],
        mass_kg=task.cfg['env']['pushBox']['mass'],
        friction=task.cfg['env']['pushBox']['friction'],
        size_changes_during_episode=False, rsi=False)
    return sizes


def save_report(task):
    if not hasattr(task, '_push_eval_size_report'):
        return
    path = Path(task.cfg['args'].output_path)/'size_zero_shot.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task._push_eval_size_report, indent=2)+'\n')
    print('[PUSH size zero-shot]', task._push_eval_size_report['ranges_m'],
          'actual actor sizes:', task._push_eval_size_report['sampled_sizes_m'], flush=True)
