"""SIT/CLIMB extension of the replayable sampled edge graph (no simulator)."""
from copy import deepcopy
import math

import torch

from utils.edge_context_spec import EdgeContextGraph, HOLDING, AT, compile_edge_context_graph
from utils.edge_ontop_spec import (ON_TOP, PACKET_FIELDS,
    permute_graph, select_graph, validate_graph as validate_ontop_graph)


INTERACTION_CONTEXT_MODE = 'state_relation_edge_interaction_v1'
SIT, CLIMB = 9, 10
PATTERNS = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP',
            'HOLDING_CLIMB', 'HOLDING_SIT')
PRESETS = ('random', 'sit_only', 'climb_only', 'hold_sit', 'hold_climb',
           'at_then_sit', 'at_then_climb', 'ontop_then_climb')
DEFAULT_GRAPH = dict(
    mode='edge_composition', sampler='two_agent_three_object_interaction',
    edge_capacity=4, max_edges_per_agent=2,
    pattern_probabilities=dict(zip(PATTERNS, (.10, .10, .10, .25, .20, .15, .10))),
    target_sampling={'placed_object_probability': .5,
                     'reject_shared_support': True},
    shuffle_edge_order=True)


def validate_interaction_context_config(c):
    from utils.edge_ontop_spec import validate_ontop_context_config
    base = deepcopy(c)
    sit = base.pop('sit', None)
    climb = base.pop('climb', None)
    if base.get('mode') != INTERACTION_CONTEXT_MODE or base.get('schema_version') != 4:
        raise ValueError('Expected interaction scalar-context schema 4')
    if sit != dict(state_definition='tokenhsi_tar_sit_pos', near_distance_scale=10.,
                   target_local_offset=[0., 0., 0.1381430834425038]):
        raise ValueError('Unsupported SIT target geometry')
    expected_climb = dict(state_definition='root_target', near_distance_scale=10.,
                          target_height='rotated_bbox_top_plus_char_h')
    if (not isinstance(climb, dict) or set(climb) != set(expected_climb) | {'feet_height_tolerance'}
            or any(climb.get(k) != v for k, v in expected_climb.items())):
        raise ValueError('Unsupported CLIMB target geometry')
    tolerance = climb['feet_height_tolerance']
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (float, int))
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('CLIMB feet_height_tolerance must be finite and positive')
    base.update(mode='state_relation_edge_ontop_v1', schema_version=3)
    validate_ontop_context_config(base)


def validate_sampler(spec, m=2, o=3):
    if (m, o) != (2, 3):
        raise ValueError('Interaction sampler/presets require M=2, O=3; use explicit edges for generic inference')
    if set(spec) != set(DEFAULT_GRAPH):
        raise ValueError('Unsupported interaction edge composition fields')
    for key in ('mode', 'sampler', 'edge_capacity', 'max_edges_per_agent', 'target_sampling'):
        if spec[key] != DEFAULT_GRAPH[key]:
            raise ValueError('Unsupported interaction edge composition ' + key)
    p = spec['pattern_probabilities']
    if (tuple(p) != PATTERNS or any(isinstance(v, bool) or not isinstance(v, (float, int))
            or not math.isfinite(v) or v < 0 for v in p.values())
            or not math.isclose(sum(p.values()), 1., abs_tol=1e-8)):
        raise ValueError('Pattern probabilities must follow the seven-pattern contract and sum to one')
    if sum(p[k] for k in ('SIT', 'CLIMB', 'HOLDING_CLIMB', 'HOLDING_SIT')) > .5 + 1e-8:
        raise ValueError('SIT/CLIMB pattern mass must be <=0.5 for conflict-free exact marginals')
    if type(spec['shuffle_edge_order']) is not bool:
        raise ValueError('shuffle_edge_order must be boolean')


def _draw_feasible_patterns(n, probabilities, device, generator):
    """Preserve each agent's marginal probabilities while excluding two human supports."""
    p = torch.tensor([probabilities[k] for k in PATTERNS], device=device)
    interaction_ids = torch.tensor([1, 2, 5, 6], device=device)
    other_ids = torch.tensor([0, 3, 4], device=device)
    q = p[interaction_ids].sum()
    if q > .5 + 1e-8:
        raise ValueError('SIT/CLIMB pattern mass must be <=0.5 for conflict-free exact marginals')
    pattern = other_ids[torch.multinomial(p[other_ids], n * 2, replacement=True,
                                          generator=generator)].reshape(n, 2)
    joint = torch.rand(n, device=device, generator=generator)
    only0 = joint < q
    only1 = (joint >= q) & (joint < 2 * q)
    for agent, mask in ((0, only0), (1, only1)):
        count = int(mask.sum())
        if count:
            choice = torch.multinomial(p[interaction_ids], count, replacement=True,
                                       generator=generator)
            pattern[mask, agent] = interaction_ids[choice]
    return pattern


def _terminal_relation(pattern):
    table = torch.tensor([0, 0, 0, AT, ON_TOP, CLIMB, SIT], device=pattern.device)
    return table[pattern]


def compose_graph(pattern, target, shuffle=False, generator=None):
    """Compile local patterns and logical object targets into a replayable graph."""
    n = pattern.shape[0]
    if pattern.shape != (n, 2) or target.shape != (n, 2):
        raise ValueError('Expected pattern/target [N,2]')
    if ((pattern < 0) | (pattern >= len(PATTERNS))).any() or ((target < 0) | (target >= 3)).any():
        raise ValueError('Pattern or logical support index out of range')
    device = pattern.device
    a = torch.arange(2, device=device)[None].expand(n, -1)
    composite = pattern >= 3
    standalone_interaction = (pattern == 1) | (pattern == 2)
    interaction = standalone_interaction | (pattern == 5) | (pattern == 6)
    support = interaction | (pattern == 4)

    rel0 = torch.where(pattern == 1, SIT, torch.where(pattern == 2, CLIMB, HOLDING))
    rel1 = _terminal_relation(pattern)
    valid = torch.stack([torch.ones_like(pattern, dtype=torch.bool), composite], -1).flatten(1)
    relation = torch.stack([rel0, rel1], -1).flatten(1)
    owner = a.repeat_interleave(2, -1)
    src0 = a
    dst0 = torch.where(standalone_interaction, 2 + target, 2 + a)
    src1 = torch.where((rel1 == SIT) | (rel1 == CLIMB), a, 2 + a)
    dst1 = torch.where(rel1 == AT, 5 + a, 2 + target)
    src = torch.stack([src0, src1], -1).flatten(1)
    dst = torch.stack([dst0, dst1], -1).flatten(1)

    required0 = (pattern <= 2) | (pattern >= 5)
    required1 = (pattern >= 3) & (pattern <= 6)
    required = torch.stack([required0, required1], -1).flatten(1) & valid
    term0 = torch.where((pattern == 3) | (pattern == 4), 2 * a + 1, -1)
    term = torch.stack([term0, torch.full_like(a, -1)], -1).flatten(1)
    pre = torch.zeros(n, 4, 4, dtype=torch.bool, device=device)
    pre[:, 1, 0] = composite[:, 0]
    pre[:, 3, 2] = composite[:, 1]

    # A support object may be another agent's object only when its AT/ON_TOP edge
    # prepares it. That placement edge becomes context; reward remains ungated.
    for agent in (0, 1):
        other = 1 - agent
        edge = 2 * agent + torch.where(composite[:, agent], 1, 0)
        uses_other = support[:, agent] & (target[:, agent] == other)
        other_places = (pattern[:, other] == 3) | (pattern[:, other] == 4)
        rows = (uses_other & other_places).nonzero(as_tuple=False).flatten()
        if rows.numel():
            pre[rows, edge[rows], 2 * other + 1] = True

    src = src.masked_fill(~valid, 0)
    dst = dst.masked_fill(~valid, 0)
    relation = relation.masked_fill(~valid, 0)
    owner = owner.masked_fill(~valid, 0)
    graph = EdgeContextGraph(('slot0', 'slot1', 'slot2', 'slot3'), 2, 3,
        src, dst, relation, owner, valid, required, pre, term)
    validate_graph(graph)
    if shuffle:
        graph = permute_graph(graph, torch.rand(n, 4, device=device, generator=generator).argsort(-1))
    return graph


def _random_targets(pattern, generator=None):
    n = pattern.shape[0]
    device = pattern.device
    target = torch.full_like(pattern, 2)
    terminal = _terminal_relation(pattern)
    support = (pattern == 1) | (pattern == 2) | (pattern == 4) | (pattern == 5) | (pattern == 6)
    interaction = (pattern == 1) | (pattern == 2) | (pattern == 5) | (pattern == 6)
    standalone = (pattern == 1) | (pattern == 2)

    both = support.all(-1)
    both_top = both & (terminal == ON_TOP).all(-1)
    rows = both_top.nonzero(as_tuple=False).flatten()
    if rows.numel():
        lower = (torch.rand(len(rows), device=device, generator=generator) >= .5).long()
        upper = 1 - lower
        target[rows, lower] = 2
        target[rows, upper] = lower

    mixed = both & (interaction.sum(-1) == 1)
    rows = mixed.nonzero(as_tuple=False).flatten()
    if rows.numel():
        inter_agent = interaction[rows].long().argmax(-1)
        top_agent = 1 - inter_agent
        target[rows, top_agent] = 2
        target[rows, inter_agent] = top_agent

    single = support & ~both[:, None]
    for agent in (0, 1):
        rows = single[:, agent].nonzero(as_tuple=False).flatten()
        if not rows.numel():
            continue
        other = 1 - agent
        placed_other = ((pattern[rows, other] == 3) | (pattern[rows, other] == 4))
        own_allowed = standalone[rows, agent]
        # Uniform over the valid set {O_i when not holding, prepared O_j, O_X}.
        weights = torch.stack([own_allowed.float(), placed_other.float(), torch.ones_like(own_allowed, dtype=torch.float)], -1)
        choice = torch.multinomial(weights, 1, generator=generator).flatten()
        candidates = torch.stack([torch.full_like(choice, agent), torch.full_like(choice, other), torch.full_like(choice, 2)], -1)
        target[rows, agent] = candidates.gather(1, choice[:, None]).flatten()
    return target


def sample_graph(n, spec, device='cpu', preset='random', role_swap=False, generator=None):
    validate_sampler(spec)
    if preset not in PRESETS:
        raise ValueError('Unknown TASK_GRAPH preset: ' + preset)
    if preset == 'random':
        # Anti-correlate the two interaction indicators: each agent retains the
        # configured marginal distribution, but the O=3 toy never assigns two humans
        # to the same sole stable support in one episode.
        pattern = _draw_feasible_patterns(n, spec['pattern_probabilities'], device, generator)
        target = _random_targets(pattern, generator)
    else:
        table = {
            'sit_only': ((1, 0), (2, 2)),
            'climb_only': ((2, 0), (2, 2)),
            'hold_sit': ((6, 0), (2, 2)),
            'hold_climb': ((5, 0), (2, 2)),
            'at_then_sit': ((3, 1), (2, 0)),
            'at_then_climb': ((3, 2), (2, 0)),
            'ontop_then_climb': ((4, 5), (2, 0)),
        }
        patterns, targets = table[preset]
        pattern = torch.tensor(patterns, device=device).expand(n, -1).clone()
        target = torch.tensor(targets, device=device).expand(n, -1).clone()
        if role_swap:
            pattern = pattern.flip(-1)
            target = target.flip(-1)
            target = torch.where(target < 2, 1 - target, target)
    return compose_graph(pattern, target,
        shuffle=spec['shuffle_edge_order'] if preset == 'random' else False,
        generator=generator)


def compile_interaction_graph(spec, m, o, device=None):
    if spec.get('mode') == 'edge_composition':
        validate_sampler(spec, m, o)
        return select_graph(sample_graph(1, spec, device=device or 'cpu', preset='hold_sit'), 0)
    if set(spec) != {'edges'}:
        raise ValueError('Expected edge_composition or explicit edges')
    adapted = deepcopy(spec)
    restored = []
    for i, edge in enumerate(adapted['edges']):
        relation = edge.get('relation')
        if relation == 'ON_TOP':
            dst = edge.get('dst', '')
            if not dst.startswith('O_'):
                raise ValueError('ON_TOP target must be an Object')
            restored.append((i, ON_TOP, int(dst[2:])))
            edge.update(relation='AT', dst='G_0')
        elif relation in ('SIT', 'CLIMB'):
            restored.append((i, SIT if relation == 'SIT' else CLIMB, None))
            edge['relation'] = 'HOLDING'
    graph = compile_edge_context_graph(adapted, m, o, device)
    for i, relation, target in restored:
        graph.edge_relation[i] = relation
        if relation == ON_TOP:
            if target >= o:
                raise ValueError('ON_TOP target must be an existing Object')
            graph.edge_dst[i] = m + target
    validate_graph(graph)
    return graph


def validate_graph(graph):
    validate_ontop_graph(graph)
    if graph.edge_src.ndim != 1:
        for i in range(graph.edge_src.shape[0]):
            validate_graph(select_graph(graph, i))
        return
    active = graph.edge_valid.nonzero(as_tuple=False).flatten().tolist()
    human_supports = set()
    placements = {int(graph.edge_src[i]): i for i in active
                  if int(graph.edge_relation[i]) in (AT, ON_TOP)}
    held = {int(graph.edge_dst[i]) for i in active if int(graph.edge_relation[i]) == HOLDING}
    for i in active:
        relation = int(graph.edge_relation[i])
        src, dst = int(graph.edge_src[i]), int(graph.edge_dst[i])
        if relation in (SIT, CLIMB):
            if not 0 <= src < graph.num_agents or not graph.num_agents <= dst < graph.num_agents + graph.num_objects:
                raise ValueError('SIT/CLIMB must bind Human -> Object')
            if dst in human_supports:
                raise ValueError('Multiple humans cannot consume one SIT/CLIMB support')
            if dst in held and dst not in placements:
                raise ValueError('Moving HOLDING-only object is not a stable support')
            human_supports.add(dst)
