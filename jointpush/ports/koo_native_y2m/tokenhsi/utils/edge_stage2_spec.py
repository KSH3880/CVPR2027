import math
import re

import torch

from utils.edge_context_spec import EdgeContextGraph, HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, expand_graph, permute_graph, select_graph
from utils.edge_interaction_spec import SIT, CLIMB


STAGE2_CONTEXT_MODE = 'state_relation_edge_stage2_v1'
STAGE2_SAMPLER = 'two_agent_stage2_cooperative'
STAGE2_PRESETS = ('random_scenario', 'place_climb', 'place_sit', 'place_stack', 'independent')
RELATIONS = {'HOLDING': HOLDING, 'AT': AT, 'ON_TOP': ON_TOP, 'SIT': SIT, 'CLIMB': CLIMB}


def validate_stage2_config(config):
    from utils.edge_stage1_spec import STAGE1_CONTEXT_MODE, validate_stage1_context_config
    if config.get('mode') != STAGE2_CONTEXT_MODE or config.get('schema_version') != 10:
        raise ValueError('Stage 2 requires its own mode and schema 10')
    variant = config.get('stage1_variant')
    if variant not in (
            'scenario_independent_stage1_plane', 'scenario_stage2_sit_plane_self_sum'):
        raise ValueError('Unsupported Stage-2 plane success geometry')
    expected_training = {
        'near_start': {'probability_start': .8, 'probability_end': .3,
                       'anneal_steps': 300000, 'at_xy_range': [.75, 1.5],
                       'ontop_xy_range': [1.1, 1.8]},
        'climb_rsi': {'late_fraction': .7, 'phase_range': [.65, .98]},
    }
    if variant == 'scenario_stage2_sit_plane_self_sum' and \
            config.get('independent_training') != expected_training:
        raise ValueError('Stage-2 independent training must match Stage-1 experiment 34')
    if variant == 'scenario_independent_stage1_plane' and 'independent_training' in config:
        raise ValueError('Baseline Stage-2 has no independent curriculum')
    legacy = dict(config, mode=STAGE1_CONTEXT_MODE, schema_version=9)
    legacy.pop('independent_training', None)
    validate_stage1_context_config(legacy)


def validate_sampler(spec, m=2, o=3):
    expected = {'mode', 'sampler', 'semantic_only', 'edge_capacity',
                'max_edges_per_agent', 'cooperative_probability',
                'family_probabilities', 'shuffle_edge_order'}
    if set(spec) not in (expected, expected | {'independent_template_probabilities'}) or \
            spec['mode'] != 'edge_composition' or \
            spec['sampler'] != STAGE2_SAMPLER or spec['semantic_only'] is not True or \
            spec['edge_capacity'] != 4 or spec['max_edges_per_agent'] != 2 or \
            (m, o) != (2, 3) or type(spec['shuffle_edge_order']) is not bool:
        raise ValueError('Stage-2 training requires two humans, three objects and four edge slots')
    p = spec['cooperative_probability']
    families = spec['family_probabilities']
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1 or \
            set(families) != {'place_climb', 'place_sit', 'place_stack'} or \
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v < 0 for v in families.values()) or \
            not math.isclose(sum(families.values()), 1., abs_tol=1e-8):
        raise ValueError('Invalid Stage-2 scenario probabilities')
    if 'independent_template_probabilities' in spec:
        from utils.edge_scenario_spec import CLIMB_TEMPLATES
        probabilities = spec['independent_template_probabilities']
        if set(probabilities) != set(CLIMB_TEMPLATES) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in probabilities.values()) or \
                not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-8):
            raise ValueError('Invalid Stage-2 independent template probabilities')


def _empty_graph(n, m, o, capacity, device):
    longs = [torch.zeros(n, capacity, dtype=torch.long, device=device) for _ in range(4)]
    valid = torch.zeros(n, capacity, dtype=torch.bool, device=device)
    required = valid.clone()
    pre = torch.zeros(n, capacity, capacity, dtype=torch.bool, device=device)
    term = torch.full((n, capacity), -1, dtype=torch.long, device=device)
    return EdgeContextGraph(tuple('slot{}'.format(i) for i in range(capacity)), m, o,
                            *longs, valid, required, pre, term)


def _put(graph, batch, index, owner, source, target, relation, required):
    graph.edge_src[batch, index] = source
    graph.edge_dst[batch, index] = target
    graph.edge_relation[batch, index] = relation
    graph.edge_owner[batch, index] = owner
    graph.edge_valid[batch, index] = True
    graph.required_goal[batch, index] = required


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None):
    validate_sampler(spec)
    if preset not in STAGE2_PRESETS:
        raise ValueError('Unknown Stage-2 graph preset: ' + preset)
    graph = _empty_graph(n, 2, 3, 4, device)
    family_names = tuple(spec['family_probabilities'])
    family_weights = torch.tensor([spec['family_probabilities'][name] for name in family_names],
                                  device=device)
    independent_spec = dict(mode='edge_composition',
        sampler='two_agent_three_object_independent_with_climb', semantic_only=True,
        edge_capacity=4, max_edges_per_agent=2,
        template_probabilities=spec.get('independent_template_probabilities',
            {'HOLDING': .2, 'SIT': .2, 'CLIMB': .2,
             'HOLDING_AT': .2, 'HOLDING_ON_TOP': .2}),
        random_binding={'objects': 'disjoint', 'goals': 'disjoint'},
        shuffle_edge_order=spec['shuffle_edge_order'])
    from utils.edge_scenario_spec import sample_graph as sample_independent
    for batch in range(n):
        independent = preset == 'independent' or (preset == 'random_scenario' and
            bool(torch.rand((), device=device, generator=generator) >= spec['cooperative_probability']))
        if independent:
            old = sample_independent(1, independent_spec, device, generator=generator)
            for key in ('edge_src', 'edge_dst', 'edge_relation', 'edge_owner',
                        'edge_valid', 'required_goal'):
                getattr(graph, key)[batch] = getattr(old, key)[0]
            continue
        family = (family_names[int(torch.multinomial(family_weights, 1,
            generator=generator))] if preset == 'random_scenario' else preset)
        carrier = (int(torch.randint(2, (), device=device, generator=generator))
                   if preset == 'random_scenario' else int(role_swap))
        other = 1 - carrier
        objects = torch.randperm(3, device=device, generator=generator)
        goals = torch.randperm(2, device=device, generator=generator)
        base, child = 2 + int(objects[0]), 2 + int(objects[1])
        _put(graph, batch, 0, carrier, carrier, base, HOLDING, False)
        _put(graph, batch, 1, carrier, base, 5 + int(goals[0]), AT, True)
        if family == 'place_climb':
            _put(graph, batch, 2, other, other, base, CLIMB, True)
        elif family == 'place_sit':
            _put(graph, batch, 2, other, other, base, SIT, True)
        else:
            _put(graph, batch, 2, other, other, child, HOLDING, False)
            _put(graph, batch, 3, other, child, base, ON_TOP, True)
    if spec['shuffle_edge_order'] and preset == 'random_scenario':
        graph = permute_graph(graph, torch.rand(n, 4, device=device,
                                               generator=generator).argsort(-1))
    validate_graph(graph)
    return graph


def compile_explicit_graph(spec, m, o, device=None):
    if spec.get('mode') != 'stage2_explicit' or not isinstance(spec.get('edges'), list):
        raise ValueError('Expected Stage-2 explicit edge list')
    if spec.get('semantic_only') is not True or m < 2 or o < m:
        raise ValueError('Explicit Stage-2 graph needs semantic packet and O>=M>=2')
    capacity = spec.get('edge_capacity', len(spec['edges']))
    if type(capacity) is not int or capacity < len(spec['edges']) or capacity < 1:
        raise ValueError('Invalid explicit edge capacity')
    graph = _empty_graph(1, m, o, capacity, device or 'cpu')
    def node(value):
        match = re.fullmatch(r'([HOG])_(\d+)', str(value))
        if match is None:
            raise ValueError('Invalid Stage-2 graph node: ' + str(value))
        kind, index = match[1], int(match[2])
        if index >= (o if kind == 'O' else m):
            raise ValueError('Stage-2 graph node out of range: ' + value)
        return index + {'H': 0, 'O': m, 'G': m + o}[kind]
    for index, edge in enumerate(spec['edges']):
        if set(edge) != {'owner', 'source', 'target', 'relation', 'required_goal'}:
            raise ValueError('Explicit Stage-2 edge fields do not match schema')
        owner, relation = edge['owner'], edge['relation']
        if type(owner) is not int or not 0 <= owner < m or relation not in RELATIONS or \
                type(edge['required_goal']) is not bool:
            raise ValueError('Invalid Stage-2 edge owner, relation or required_goal')
        _put(graph, 0, index, owner, node(edge['source']), node(edge['target']),
             RELATIONS[relation], edge['required_goal'])
    result = select_graph(graph, 0)
    validate_graph(result)
    return result


def compile_graph(spec, m, o, device=None):
    if spec.get('mode') == 'stage2_explicit':
        return compile_explicit_graph(spec, m, o, device)
    validate_sampler(spec, m, o)
    return select_graph(sample_graph(1, spec, device or 'cpu', 'place_climb'), 0)


def classify_family(graph):
    relation, owner, source, target, valid = (graph.edge_relation, graph.edge_owner,
        graph.edge_src, graph.edge_dst, graph.edge_valid)
    at = valid & (relation == AT)
    shared = at[:, :, None] & valid[:, None, :] & \
        (source[:, :, None] == target[:, None, :]) & \
        (owner[:, :, None] != owner[:, None, :])
    downstream = relation[:, None, :]
    climb = (shared & (downstream == CLIMB)).any((1, 2))
    sit = (shared & (downstream == SIT)).any((1, 2))
    stack = (shared & (downstream == ON_TOP)).any((1, 2))
    return torch.where(climb, 1, torch.where(sit, 2,
                       torch.where(stack, 3, 0)))


def expand_explicit(spec, n, m, o, device=None):
    return expand_graph(compile_explicit_graph(spec, m, o, device), n)


def validate_graph(graph):
    if graph.edge_src.ndim != 1:
        for row in range(graph.edge_src.shape[0]):
            validate_graph(select_graph(graph, row))
        return
    m, o = graph.num_agents, graph.num_objects
    edges = graph.edge_valid.nonzero().flatten().tolist()
    if not edges or graph.prereq_mask.any() or (graph.term_index >= 0).any():
        raise ValueError('Stage-2 phase needs current edges without temporal DAG')
    held = {}
    supports = {}
    goals = set()
    for index in edges:
        owner = int(graph.edge_owner[index])
        src, dst, relation = (int(graph.edge_src[index]), int(graph.edge_dst[index]),
                              int(graph.edge_relation[index]))
        if not 0 <= owner < m or relation not in RELATIONS.values():
            raise ValueError('Invalid Stage-2 edge owner/relation')
        source_type = 'H' if src < m else 'O' if src < m + o else 'G'
        target_type = 'H' if dst < m else 'O' if dst < m + o else 'G'
        expected = {HOLDING: ('H', 'O'), AT: ('O', 'G'), ON_TOP: ('O', 'O'),
                    SIT: ('H', 'O'), CLIMB: ('H', 'O')}[relation]
        if (source_type, target_type) != expected or (source_type == 'H' and src != owner):
            raise ValueError('Invalid Stage-2 edge endpoint types or owner')
        if relation == HOLDING:
            if dst in held:
                raise ValueError('Joint HOLDING of one object is not supported')
            held[dst] = owner
        if relation == AT:
            if dst in goals:
                raise ValueError('Two placements cannot target the same goal')
            goals.add(dst)
        if relation == ON_TOP:
            if src == dst or dst in supports.values():
                raise ValueError('Invalid ON_TOP support')
            supports[src] = dst
    for source in supports:
        seen = set()
        current = source
        while current in supports:
            if current in seen:
                raise ValueError('Cyclic ON_TOP supports')
            seen.add(current)
            current = supports[current]
    for owner in range(m):
        owned = [index for index in edges if int(graph.edge_owner[index]) == owner]
        if len(owned) not in (1, 2):
            raise ValueError('Each Stage-2 agent needs one bundle')
        relations = [int(graph.edge_relation[index]) for index in owned]
        if len(owned) == 1:
            if relations[0] not in (HOLDING, SIT, CLIMB) or not bool(graph.required_goal[owned[0]]):
                raise ValueError('Invalid standalone Stage-2 bundle')
            continue
        if sorted(relations) not in ([HOLDING, AT], [HOLDING, ON_TOP]):
            raise ValueError('Stage-2 placement needs one HOLDING edge')
        hold = owned[relations.index(HOLDING)]
        place = owned[1 - relations.index(HOLDING)]
        if graph.edge_dst[hold] != graph.edge_src[place] or \
                bool(graph.required_goal[hold]) or not bool(graph.required_goal[place]):
            raise ValueError('Stage-2 HOLDING/placement binding mismatch')
