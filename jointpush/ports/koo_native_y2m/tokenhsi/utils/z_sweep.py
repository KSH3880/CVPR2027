"""Evaluation-only controlled box geometry and trial summaries (metres)."""
import math
from collections import defaultdict

TASKS = ('holding', 'at', 'ontop', 'sit', 'climb')
TEMPLATES = dict(holding='HOLDING', at='HOLDING_AT', ontop='HOLDING_ON_TOP',
                 sit='SIT', climb='CLIMB')


def make_conditions(x=.4, y=.4, base_z=.4, sizes=None, offsets=None):
    sizes = sizes if sizes is not None else [round(.2 + .05*i, 2) for i in range(10)]
    offsets = offsets if offsets is not None else [round(.1*i, 2) for i in range(6)]
    if not all(math.isfinite(v) and v > 0 for v in [x, y, base_z] + list(sizes)):
        raise ValueError('Dimensions must be positive and finite')
    if not offsets or not sizes or not all(math.isfinite(v) and v >= 0 for v in offsets):
        raise ValueError('Offsets must be nonnegative and finite; grids must be nonempty')
    conditions = []
    for task in TASKS:
        axes = [('source_size_z', sizes), ('source_bottom_z', offsets)]
        if task == 'ontop':
            axes += [('support_size_z', sizes), ('support_bottom_z', offsets)]
        if task == 'at':
            axes += [('goal_bottom_z', offsets)]
        for axis, grid in axes:
            for value in grid:
                source_z = value if axis == 'source_size_z' else base_z
                support_z = value if axis == 'support_size_z' else base_z
                conditions.append(dict(task=task, axis=axis, value=value, x=x, y=y,
                    source_size_z=source_z, support_size_z=support_z,
                    source_bottom_z=value if axis == 'source_bottom_z' else 0.,
                    support_bottom_z=value if axis == 'support_bottom_z' else 0.,
                    goal_bottom_z=value if axis == 'goal_bottom_z' else 0.))
    return conditions


def box_sizes(conditions, scenes_per_condition):
    if scenes_per_condition < 1:
        raise ValueError('scenes_per_condition must be positive')
    return [[[c['x'], c['y'], c['source_size_z'] if o < 2 else c['support_size_z']]
             for o in range(4)] for c in conditions for _ in range(scenes_per_condition)]


def graph_rows(conditions, scenes_per_condition):
    templates, bindings = [], []
    for c in conditions:
        template = TEMPLATES[c['task']]
        binding = ((0, 0), (1, 1)) if c['task'] == 'at' else (
            ((0, 2), (1, 3)) if c['task'] == 'ontop' else ((0,), (1,)))
        for _ in range(scenes_per_condition):
            templates.append((template, template)); bindings.append(binding)
    return templates, bindings


def wilson(successes, total):
    if not total:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0., center-radius), min(1., center+radius)


def summarize(rows, conditions):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['condition']].append(row)
    result = []
    for i, c in enumerate(conditions):
        trials = grouped[i]
        record = dict(condition=i, **c, trials=len(trials))
        for field in ('ever_success', 'final_success', 'initial_success', 'terminated'):
            count = sum(int(r[field]) for r in trials)
            record[field + '_count'] = count
            record[field + '_rate'] = count / len(trials) if trials else None
        # Two agents share a scene. Confidence intervals apply to the binary
        # scene-all outcome, not to correlated individual agent trials.
        scenes = defaultdict(list)
        for row in trials:
            scenes[(row['repeat'], row['scene'])].append(row)
        record['scene_trials'] = len(scenes)
        for field in ('ever_success', 'final_success'):
            count = sum(all(int(r[field]) for r in rs) for rs in scenes.values())
            record[field + '_scene_all_rate'] = count/len(scenes) if scenes else None
            lo, hi = wilson(count, len(scenes))
            record[field + '_scene_wilson_low'] = lo
            record[field + '_scene_wilson_high'] = hi
        result.append(record)
    return result
