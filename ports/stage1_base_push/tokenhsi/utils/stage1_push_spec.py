"""Add PUSH_AT without renumbering any original Stage1 task or template."""
from copy import deepcopy
from dataclasses import replace
import math
import torch
from utils.edge_scenario_spec import (CLIMB_TEMPLATES, compose_canonical_graph,
    validate_sampler as validate_original_sampler)
from utils.edge_ontop_spec import select_graph

SAMPLER = 'two_agent_four_object_independent_unified_push'
PUSH = 12
PUSH_TEMPLATE = 5
PUSH_TASK = 4
TEMPLATES = CLIMB_TEMPLATES + ('PUSH_AT',)


def stable_push_profiles():
    """XY 0.1m grid, Z 0.05m grid; integer comparison avoids boundary drift."""
    return [[x/20., y/20., z/20.] for x in range(32,45,2)
            for y in range(32,45,2) for z in range(29,35)
            if 10*min(x,y) >= 11*z]


def original_spec(spec):
    original = deepcopy(spec)
    original['sampler'] = 'two_agent_four_object_independent_unified'
    original['template_probabilities'] = dict(zip(CLIMB_TEMPLATES, (0., .1, .25, .325, .325)))
    return original


def validate_sampler(spec, m=2, o=4):
    validate_original_sampler(original_spec(spec), m, o)
    probabilities = spec['template_probabilities']
    if (spec['sampler'] != SAMPLER or tuple(probabilities) != TEMPLATES
            or probabilities['HOLDING'] != 0 or (m, o) != (2, 4)
            or any(isinstance(p, bool) or not isinstance(p, (int, float))
                   or not math.isfinite(p) or p < 0 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.)):
        raise ValueError('Stage1 PUSH requires two people, four objects and six template probabilities')


def compose_graph(templates, shuffle=False):
    push = templates == PUSH_TEMPLATE
    graph = compose_canonical_graph(torch.where(push, 3, templates), shuffle=shuffle, num_objects=4)
    owner = graph.edge_owner
    per_edge_push = push.gather(1, owner)
    return replace(graph, edge_relation=torch.where(
        per_edge_push & (graph.edge_relation == 6), PUSH, graph.edge_relation))


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    return select_graph(compose_graph(torch.full((1, m), PUSH_TEMPLATE, device=device, dtype=torch.long)), 0)


def validate_env(env):
    from utils.edge_stage1_spec import validate_stage1_context_config
    from utils.unified_training import SKILLS
    validate_stage1_context_config(env['relationReward'])
    validate_sampler(env['relationGraph'], env['numAgents'], env['numObjects'])
    if tuple(env['skill']) != SKILLS + ('push',) or env.get('sharedSupport') or env.get('stage2'):
        raise ValueError('Keep original eight skills plus PUSH; Stage2 is excluded')
    if env.get('ampTaskConditioning') is not True or env.get('goalRotation') != 'identity':
        raise ValueError('Preserve Stage1 policy observation and conditioned AMP')
    if tuple(env['templateRsi']) != TEMPLATES:
        raise ValueError('Keep all original template RSI rows plus PUSH_AT')
    extension = env['pushExtension']
    finetune = extension.get('sampling_profile') == 'stage1_square_cone_v1'
    if extension['revision'] != 1 or extension['contact_weights'] != ([.2, .2, .2] if finetune else [.4, .2, .2]):
        raise ValueError('Use current CONTACT state/progress/success weights')
    from utils.unified_training import EXPERT_PROBS
    mapping=env['ampExpertMapping']
    expected=dict(zip(('carry','sit','climb'),[list(row)+[0.] for row in EXPERT_PROBS]))
    expected['push']=[.5]+[0.]*7+[.5]
    if mapping!=expected:
        raise ValueError('Keep original three expert mixtures and current loco/PUSH50/50')
    probability=env['openPushRSI']['interaction_probability']
    if env['templateRsi']['PUSH_AT']!=[1-probability]+[0.]*7+[probability]:
        raise ValueError('PUSH_AT RSI row and native interaction probability must agree')
    if env['openPushRSI'].get('uniform_goal_distance', False):
        if not finetune or env['openPushRSI'].get('goal_distance_range') != [2., 10.]:
            raise ValueError('This uniform RSI variant requires fine-tune and 2..10m bounds')
    mode = extension.get('distractor_mode')
    if mode not in (None, 'stage1_support_v1'):
        raise ValueError('Unknown PUSH distractor layout')
    if mode == 'stage1_support_v1' and not finetune:
        raise ValueError('Stage1 support distractors require the fine-tune sampler')
    box = env['pushBox']
    expected = [[round(1. + i / 10, 1), 2., 1.3] for i in range(6)]
    friction = .6
    if finetune:
        expected = [[round(1.+i/10,1),2.,round(1.3+j/10,1)] for i in range(6) for j in range(3)]
        friction = .5
        if extension.get('geometry_profile') == 'stable_footprint_v1':
            expected, friction = stable_push_profiles(), .4
            if box['size'] != [2.2,2.2,1.7] or env['pushScene']['path_separation'] != 3.5:
                raise ValueError('Stable PUSH requires conservative 2.2m footprint / 3.5m paths')
        if env['pushScene']['random_at_goal'] != dict(sampling='stage1_square_cone',reach=4.5,
                min_distance=1.,agent_clearance=.5,half_angle_degrees=45.):
            raise ValueError('Fine-tune requires the Stage1 square with a forward +/-45 degree cone')
    if box['size_profiles'] != expected or box['mass'] != 20. or box['friction'] != friction:
        raise ValueError('PUSH size profiles, mass or friction differ from this experiment')


def sample_asset_sizes(n, env, device):
    """Keep legacy task sizes and sample the configured PUSH profile family."""
    from utils.size_rsi import task_probabilities, STAGE1_V2_RANGES
    probs = torch.tensor(list(env['relationGraph']['template_probabilities'].values()), device=device)
    seeds = torch.multinomial(probs, n * 2, replacement=True).reshape(n, 2)
    sizes = torch.empty(n, 4, 3, device=device)
    profiles = torch.tensor(env['pushBox']['size_profiles'], device=device)
    native_support = env['pushExtension'].get('distractor_mode') == 'stage1_support_v1'
    for owner in range(2):
        for template, name in enumerate(TEMPLATES):
            rows = (seeds[:, owner] == template).nonzero().flatten()
            if not len(rows):
                continue
            if template == PUSH_TEMPLATE:
                sizes[rows, owner] = profiles[torch.randint(len(profiles), (len(rows),), device=device)]
                if not native_support:
                    sizes[rows, owner + 2] = profiles[torch.randint(len(profiles), (len(rows),), device=device)]
            else:
                for axis, (lo, hi) in enumerate(STAGE1_V2_RANGES[name]):
                    sizes[rows, owner, axis] = torch.randint(round(lo*20), round(hi*20)+1, (len(rows),), device=device)/20
                for axis, (lo, hi) in enumerate(STAGE1_V2_RANGES['SUPPORT']):
                    sizes[rows, owner+2, axis] = torch.randint(round(lo*20), round(hi*20)+1, (len(rows),), device=device)/20
    if native_support:
        for axis, (lo, hi) in enumerate(STAGE1_V2_RANGES['SUPPORT']):
            sizes[:, 2:, axis] = torch.randint(round(lo*20), round(hi*20)+1, (n, 2), device=device)/20
    posterior = torch.zeros(n, 2, 6, device=device)
    push = seeds == PUSH_TEMPLATE
    posterior[..., PUSH_TEMPLATE] = push
    if (~push).any():
        old = task_probabilities(sizes[:, :2][~push], STAGE1_V2_RANGES)
        old = old * probs[:5] / old.new_tensor((.05, .1, .25, .3, .3))
        old[:, 0] = 0
        old /= old.sum(-1, keepdim=True)
        posterior[..., :5][~push] = old
    return sizes, posterior


def checkpoint_contract(env):
    import hashlib
    from pathlib import Path
    return dict(revision=3, upstream='28f02b3278f0a3e703cfa22a1e8cc376bb8ec512',
        tasks=['sit','climb','carry_at','carry_ontop','push_at'],
        amp_families=['carry','sit','climb','push'], amp_input=1330, observation=634,
        amp_experts=env['ampExpertMapping'],
        graph=env['relationGraph'], extension=env['pushExtension'],
        push_box=env['pushBox'], push_scene=env['pushScene'], rsi=env['openPushRSI'],
        bank_sha256=hashlib.sha256(Path(env['openPushRSI']['bank_file']).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(env['motion_file']).read_bytes()).hexdigest())


def check_checkpoint(weights, env, for_training=False):
    if for_training and weights.get('validation_only'):
        raise ValueError('Untrained validation fixtures cannot resume training')
    if weights.get('stage1_push_contract') != checkpoint_contract(env):
        raise ValueError('Checkpoint is not this Stage1+PUSH five-task/four-box contract; '
                         'the old 570D/6000-iteration PUSH model cannot resume or evaluate here')
