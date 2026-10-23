"""Opt-in fifth task; original task IDs, sizes and reward contracts are preserved."""
from copy import deepcopy
import math
import torch

from utils.task_role_spec import TASKS, TASK_PROBS
from utils.size_rsi import task_probabilities


def push_assets(sizes, env):
    spec = env['pushBox']
    return (sizes[..., 2] - spec['size_z_m']).abs() < 1e-5


def legacy_size_probabilities(sizes, env, push_mask=None):
    mask = push_assets(sizes, env) if push_mask is None else push_mask
    safe = sizes.clone()
    safe[mask] = safe.new_tensor([.4, .4, .4])
    result = task_probabilities(safe, env['box']['build']['taskSizeRanges'])
    return result.masked_fill(mask[..., None], 0)


def size_probabilities(sizes, env, push_mask=None):
    from utils.task_role_spec import task_size_probabilities
    mask = push_assets(sizes, env) if push_mask is None else push_mask
    safe = sizes.clone()
    safe[mask] = safe.new_tensor([.4, .4, .4])
    original = task_size_probabilities(safe, env['box']['build']['taskSizeRanges'])
    return torch.cat((original.masked_fill(mask[..., None], 0), mask[..., None].float()), -1)


def validate_env(env):
    from utils.unified_training import SKILLS, validate_unified_env
    from utils.edge_scenario_spec import validate_sampler
    if env.get('multiTaskPush') is not True or env['relationReward'].get('task_types') != list(TASKS):
        raise ValueError('Five-task training requires unchanged IDs 0..3 and PUSH_AT=4')
    graph = deepcopy(env['relationGraph'])
    if env.get('sharedSupport'):
        graph['sampler'] = 'two_agent_four_object_independent_unified'
        graph['edge_capacity'] = 4
    validate_sampler(graph, env['numAgents'], env['numObjects'])
    p = env['relationGraph']['template_probabilities']['PUSH_AT']
    if not 0 < p < 1 or not math.isfinite(p):
        raise ValueError('PUSH_AT and original tasks must both have positive probability')
    expected = [v * (1-p) for v in TASK_PROBS] + [p]
    actual = list(env['relationGraph']['template_probabilities'].values())
    if any(abs(a-b) > 1e-8 for a, b in zip(actual, expected)):
        raise ValueError('Original task proportions must be preserved within the non-PUSH tasks')
    if tuple(env['skill']) != SKILLS + ('push',):
        raise ValueError('Append PUSH motion without changing existing skill indices')
    spec = env['pushTask']
    if 'loco_spawn_mode' in spec:
        from utils.push_random_spawn import MODE
        bounds = spec.get('loco_surface_clearance_range_m', [])
        if (spec['loco_spawn_mode'] != MODE or len(bounds) != 2
                or any(not math.isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= bounds[1]
                or not math.isfinite(spec.get('loco_other_box_clearance_m', -1))
                or spec.get('loco_other_box_clearance_m', -1) < .02
                or 'loco_distance_range_m' in spec):
            raise ValueError('Random PUSH starts require an unambiguous positive collider-surface gap')
    if (spec['rsi_retarget'] != 'none' or spec['rsi_sampling'] != 'native_size_cache'
            or spec.get('rsi_max_hand_midpoint_height_m') != 1.
            or spec.get('contact_objective') != 'common_face_xy_v1'):
        raise ValueError('Five-task PUSH requires native screened RSI and common-face CONTACT')
    if env['numAgents'] not in (2, 4) or (env.get('sharedSupport') is None and
            (env['numAgents'], env['numObjects']) != (2, 4)):
        raise ValueError('Five-task Stage1 uses 2 humans/4 objects; Stage2 uses 4 humans/12 objects')
    shared = env.get('sharedSupport')
    push_rsi = [1.] + [0.]*8 if shared else [.5] + [0.]*7 + [.5]
    if env['templateRsi']['PUSH_AT'] != push_rsi:
        raise ValueError('PUSH_AT RSI differs from the Stage1/Stage2 contract')
    if shared:
        if (shared.get('coupled_task') != 'PUSH_AT' or shared.get('shared_task_probabilities') != [0.,0.,0.,1.]
                or not 0 <= shared['coupled_probability'] <= 1
                or env['numObjects'] != 3*env['numAgents']
                or any(v != [1.]+[0.]*8 for v in env['templateRsi'].values())):
            raise ValueError('Stage2 uses loco-only resets and PUSH_AT-only coupling')
    base = deepcopy(env)
    for key in ('multiTaskPush', 'pushTask', 'pushBox', 'pushPointReward', 'pushGoalRandomization'):
        base.pop(key)
    base['relationReward'].pop('task_types')
    base['relationGraph']['template_probabilities'].pop('PUSH_AT')
    base['relationGraph']['template_probabilities'] = dict(zip(base['relationGraph']['template_probabilities'], TASK_PROBS))
    base['skill'] = list(SKILLS)
    for key in ('skillInitProb', 'skillDiscProb'):
        base[key] = base[key][:-1]
        total = sum(base[key])
        base[key] = [v / total for v in base[key]]
    base['templateRsi'].pop('PUSH_AT')
    base['templateRsi'] = {k: v[:-1] for k, v in base['templateRsi'].items()}
    if shared:
        base.pop('sharedSupport')
        base['relationGraph']['sampler'] = 'two_agent_four_object_independent_unified'
        base['relationGraph']['edge_capacity'] = 4
        for name in ('HOLDING_AT', 'HOLDING_ON_TOP'):
            base['templateRsi'][name] = [.5,0.,0.,0.,0.,.1,.3,.1]
    validate_unified_env(base)


def demo_distribution(env):
    """Four real AMP experts; PUSH never shares the carry discriminator label."""
    from utils.unified_training import EXPERT_PROBS
    p = env['relationGraph']['template_probabilities']['PUSH_AT']
    families = [.65*(1-p), .1*(1-p), .25*(1-p), p]
    experts = [list(row) + [0.] for row in EXPERT_PROBS]
    experts.append([1/3] + [0.]*7 + [2/3])
    return families, experts
