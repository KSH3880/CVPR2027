"""Context-free joint scenarios with random object and goal bindings."""
import itertools
import math

import torch

from utils.edge_context_spec import EdgeContextGraph, HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, permute_graph, select_graph, validate_graph as validate_base_graph
from utils.edge_interaction_spec import SIT, CLIMB


SCENARIO_SAMPLER = 'two_agent_three_object_scenario_no_climb'
SCENARIO_CLIMB_SAMPLER = 'two_agent_three_object_scenario_with_climb'
SCENARIO_INDEPENDENT_SAMPLER = 'two_agent_three_object_independent_climb_placement'
TEMPLATES = ('HOLDING', 'SIT', 'HOLDING_AT', 'HOLDING_ON_TOP')
CLIMB_TEMPLATES = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')
PRESETS = ('random_scenario', 'holding', 'sit', 'climb', 'holding_at', 'holding_ontop')
INDEPENDENT_PRESETS = ('random_scenario', 'climb', 'holding_at', 'holding_ontop',
                       'climb_ontop', 'at_ontop')


def scenario_templates(spec):
    return (CLIMB_TEMPLATES if spec.get('sampler') in
            (SCENARIO_CLIMB_SAMPLER, SCENARIO_INDEPENDENT_SAMPLER) else TEMPLATES)


def _options(template):
    if template in ('HOLDING', 'SIT', 'CLIMB'):
        return [(o,) for o in range(3)]
    if template == 'HOLDING_AT':
        return list(itertools.product(range(3), range(2)))
    if template == 'HOLDING_ON_TOP':
        return [(source, support) for source in range(3) for support in range(3)
                if source != support]
    raise ValueError('Unknown scenario template: ' + template)


def _description(template, binding):
    if template == 'HOLDING':
        return dict(held=binding[0], source=None, goal=None, top=None)
    if template in ('SIT', 'CLIMB'):
        return dict(held=None, source=None, goal=None, top=binding[0])
    if template == 'HOLDING_AT':
        return dict(held=binding[0], source=binding[0], goal=binding[1], top=None)
    return dict(held=binding[0], source=binding[0], goal=None, top=binding[1])


def valid_binding_pair(templates, bindings):
    desc = [_description(t, b) for t, b in zip(templates, bindings)]
    held = [d['held'] for d in desc if d['held'] is not None]
    if len(held) != len(set(held)):
        return False
    sources = [d['source'] for d in desc if d['source'] is not None]
    if len(sources) != len(set(sources)):
        return False
    goals = [(d['source'], d['goal']) for d in desc if d['goal'] is not None]
    if len({g for _, g in goals}) != len(goals):
        return False
    tops = [d['top'] for d in desc if d['top'] is not None]
    if len(tops) != len(set(tops)):
        return False
    holding_only = {b[0] for t, b in zip(templates, bindings) if t == 'HOLDING'}
    if holding_only.intersection(tops):
        return False
    arcs = [(d['source'], d['top']) for d in desc
            if d['source'] is not None and d['top'] is not None]
    if len(arcs) == 2 and arcs[0] == arcs[1][::-1]:
        return False
    return True


def valid_bindings(templates):
    return [(a, b) for a in _options(templates[0]) for b in _options(templates[1])
            if valid_binding_pair(templates, (a, b))]


def validate_sampler(spec, m=2, o=3):
    expected = {'mode', 'sampler', 'semantic_only', 'edge_capacity',
                'max_edges_per_agent', 'template_probabilities',
                'random_binding', 'shuffle_edge_order'}
    independent = spec.get('sampler') == SCENARIO_INDEPENDENT_SAMPLER
    if set(spec) != (expected | ({'pair_sampling'} if independent else set())) or \
            spec.get('mode') != 'edge_composition' or \
            spec.get('sampler') not in (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
                                        SCENARIO_INDEPENDENT_SAMPLER):
        raise ValueError('Unsupported scenario sampler')
    if (m, o) != (2, 3) or spec['edge_capacity'] != 4 or \
            spec['max_edges_per_agent'] != 2 or spec['semantic_only'] is not True or \
            spec['random_binding'] != {'objects': 'all', 'goals': 'all'} or \
            type(spec['shuffle_edge_order']) is not bool:
        raise ValueError('Scenario requires 2 humans, 3 objects and semantic capacity 4')
    p = spec['template_probabilities']
    templates = scenario_templates(spec)
    if tuple(p) != templates or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v < 0 for v in p.values()) or \
            not math.isclose(sum(p.values()), 1., abs_tol=1e-8):
        raise ValueError('Scenario template probabilities must follow the contract')
    if independent and (spec['pair_sampling'] != 'balanced_without_double_ontop' or
            p != {'HOLDING': 0., 'SIT': 0., 'CLIMB': 1/3,
                  'HOLDING_AT': 1/3, 'HOLDING_ON_TOP': 1/3}):
        raise ValueError('Independent scenario requires balanced CLIMB/AT/ONTOP pairs')


INDEPENDENT_PAIRS = (
    ('CLIMB', 'CLIMB'), ('CLIMB', 'HOLDING_AT'),
    ('HOLDING_AT', 'CLIMB'), ('HOLDING_AT', 'HOLDING_AT'),
    ('CLIMB', 'HOLDING_ON_TOP'), ('HOLDING_ON_TOP', 'CLIMB'),
    ('HOLDING_AT', 'HOLDING_ON_TOP'), ('HOLDING_ON_TOP', 'HOLDING_AT'))
INDEPENDENT_PAIR_WEIGHTS = (1/12, 1/12, 1/12, 1/12, 1/6, 1/6, 1/6, 1/6)


def _independent_binding(templates, device, generator):
    roles = torch.rand(3, device=device, generator=generator).argsort().tolist()
    source = roles[:2]
    support = roles[2]
    at_agents = [a for a, template in enumerate(templates) if template == 'HOLDING_AT']
    goals = torch.rand(2, device=device, generator=generator).argsort().tolist()
    bindings = []
    for agent, template in enumerate(templates):
        if template == 'CLIMB':
            bindings.append((source[agent],))
        elif template == 'HOLDING_AT':
            bindings.append((source[agent], goals[at_agents.index(agent)]))
        else:
            bindings.append((source[agent], support))
    return tuple(bindings)


def _rows(agent, template, binding, m=2, o=3):
    obj = lambda i: m + i
    goal = lambda i: m + o + i
    if template == 'HOLDING':
        return [(agent, obj(binding[0]), HOLDING, True)]
    if template == 'SIT':
        return [(agent, obj(binding[0]), SIT, True)]
    if template == 'CLIMB':
        return [(agent, obj(binding[0]), CLIMB, True)]
    if template == 'HOLDING_AT':
        source, target = binding
        return [(agent, obj(source), HOLDING, False),
                (obj(source), goal(target), AT, True)]
    source, support = binding
    return [(agent, obj(source), HOLDING, False),
            (obj(source), obj(support), ON_TOP, True)]


def compose_graph(templates, bindings, shuffle=False, generator=None, device='cpu'):
    n = len(templates)
    src = torch.zeros(n, 4, dtype=torch.long, device=device)
    dst = torch.zeros_like(src); relation = torch.zeros_like(src); owner = torch.zeros_like(src)
    valid = torch.zeros(n, 4, dtype=torch.bool, device=device)
    required = torch.zeros_like(valid)
    for batch in range(n):
        if not valid_binding_pair(templates[batch], bindings[batch]):
            raise ValueError('Invalid scenario binding')
        offset = 0
        for agent in range(2):
            rows = _rows(agent, templates[batch][agent], bindings[batch][agent])
            for row, (s, d, r, req) in enumerate(rows, offset):
                src[batch, row], dst[batch, row], relation[batch, row] = s, d, r
                owner[batch, row], valid[batch, row], required[batch, row] = agent, True, req
            offset += len(rows)
    graph = EdgeContextGraph(('slot0', 'slot1', 'slot2', 'slot3'), 2, 3,
        src, dst, relation, owner, valid, required,
        torch.zeros(n, 4, 4, dtype=torch.bool, device=device),
        torch.full((n, 4), -1, dtype=torch.long, device=device))
    if shuffle:
        graph = permute_graph(graph, torch.rand(n, 4, device=device,
                                               generator=generator).argsort(-1))
    validate_graph(graph)
    return graph


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None):
    validate_sampler(spec)
    independent = spec['sampler'] == SCENARIO_INDEPENDENT_SAMPLER
    if preset not in (INDEPENDENT_PRESETS if independent else PRESETS):
        raise ValueError('Unknown no-CLIMB TASK_GRAPH preset: ' + preset)
    if spec['sampler'] == SCENARIO_SAMPLER and preset == 'climb':
        raise ValueError('CLIMB preset requires the with-CLIMB scenario sampler')
    templates_available = scenario_templates(spec)
    probs = torch.tensor([spec['template_probabilities'][k] for k in templates_available],
                         device=device)
    joint_probs = (torch.tensor(INDEPENDENT_PAIR_WEIGHTS, device=device)
                   if independent else None)
    template_rows, binding_rows = [], []
    fixed = {'holding': 'HOLDING', 'sit': 'SIT', 'climb': 'CLIMB',
             'holding_at': 'HOLDING_AT',
             'holding_ontop': 'HOLDING_ON_TOP'}
    for _ in range(n):
        if independent:
            if preset == 'random_scenario':
                pair = int(torch.multinomial(joint_probs, 1, generator=generator))
                templates = INDEPENDENT_PAIRS[pair]
            else:
                templates = {'climb': ('CLIMB', 'CLIMB'),
                    'holding_at': ('HOLDING_AT', 'CLIMB'),
                    'holding_ontop': ('HOLDING_ON_TOP', 'CLIMB'),
                    'climb_ontop': ('CLIMB', 'HOLDING_ON_TOP'),
                    'at_ontop': ('HOLDING_AT', 'HOLDING_ON_TOP')}[preset]
                if role_swap:
                    templates = templates[::-1]
            binding = _independent_binding(templates, device, generator)
            template_rows.append(templates); binding_rows.append(binding)
            continue
        if preset == 'random_scenario':
            ids = torch.multinomial(probs, 2, replacement=True, generator=generator).tolist()
            templates = (templates_available[ids[0]], templates_available[ids[1]])
        else:
            templates = (fixed[preset], 'HOLDING')
            if role_swap:
                templates = templates[::-1]
        choices = valid_bindings(templates)
        choice = int(torch.randint(len(choices), (), device=device, generator=generator))
        template_rows.append(templates); binding_rows.append(choices[choice])
    return compose_graph(template_rows, binding_rows,
        shuffle=spec['shuffle_edge_order'] if preset == 'random_scenario' else False,
        generator=generator, device=device)


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    preset = 'climb' if spec['sampler'] == SCENARIO_INDEPENDENT_SAMPLER else 'holding'
    return select_graph(sample_graph(1, spec, device or 'cpu', preset=preset), 0)


def validate_graph(graph):
    validate_base_graph(graph)
    if graph.edge_src.ndim != 1:
        for i in range(graph.edge_src.shape[0]):
            validate_graph(select_graph(graph, i))
        return
    if graph.prereq_mask.any() or (graph.term_index >= 0).any():
        raise ValueError('No-CLIMB scenarios cannot contain temporal context')
    for agent in range(2):
        ids = (graph.edge_valid & (graph.edge_owner == agent)).nonzero().flatten()
        if not 1 <= len(ids) <= 2:
            raise ValueError('Each scenario owner requires one or two edges')
        rel = graph.edge_relation[ids]
        hold = ids[rel == HOLDING]
        placement = ids[(rel == AT) | (rel == ON_TOP)]
        interaction = ids[(rel == SIT) | (rel == CLIMB)]
        if len(placement):
            if len(hold) != 1 or len(placement) != 1 or len(interaction):
                raise ValueError('Placement template must contain HOLDING plus one placement')
            if graph.edge_dst[hold[0]] != graph.edge_src[placement[0]] or \
                    graph.required_goal[hold[0]] or not graph.required_goal[placement[0]]:
                raise ValueError('Placement pair owner/source/required-goal mismatch')
        elif len(ids) != 1 or not graph.required_goal[ids[0]]:
            raise ValueError('Standalone template must be its required current goal')


def classify_templates(graph, slot_env, slot_agent, with_climb=False):
    owned = graph.edge_valid[slot_env] & (graph.edge_owner[slot_env] == slot_agent[:, None])
    rel = graph.edge_relation[slot_env]
    has = lambda value: ((rel == value) & owned).any(-1)
    if not with_climb and has(CLIMB).any():
        raise ValueError('CLIMB graph rejected by no-CLIMB RSI')
    if with_climb:
        return torch.where(has(SIT), 1, torch.where(has(CLIMB), 2,
            torch.where(has(AT), 3, torch.where(has(ON_TOP), 4,
            torch.zeros_like(slot_agent)))))
    return torch.where(has(SIT), 1, torch.where(has(AT), 2,
        torch.where(has(ON_TOP), 3, torch.zeros_like(slot_agent))))


def agent_object_indices(graph, slot_env, slot_agent):
    owned = graph.edge_valid[slot_env] & (graph.edge_owner[slot_env] == slot_agent[:, None])
    rel = graph.edge_relation[slot_env]
    target = torch.where((rel == HOLDING) | (rel == SIT) | (rel == CLIMB),
                         graph.edge_dst[slot_env], 0)
    return target.masked_fill(~owned, 0).amax(-1) - graph.num_agents


def independent_logical_box_order(graph, env_ids, assignment):
    """Map randomized logical source slots to each owner's physical box."""
    if assignment.shape != (len(env_ids), 2):
        raise ValueError('Independent scenario requires two physical owner boxes')
    slot_env = env_ids.repeat_interleave(2)
    slot_agent = torch.arange(2, device=env_ids.device).repeat(len(env_ids))
    source = agent_object_indices(graph, slot_env, slot_agent).view(-1, 2)
    if (source < 0).any() or (source >= 3).any() or (source[:, 0] == source[:, 1]).any():
        raise ValueError('Independent scenario source object binding is invalid')
    order = torch.full((len(env_ids), 3), -1, device=env_ids.device, dtype=torch.long)
    order.scatter_(1, source, assignment)
    free_logical = (order < 0).nonzero(as_tuple=True)
    all_physical = torch.arange(3, device=env_ids.device)[None].expand(len(env_ids), -1)
    assigned = torch.zeros_like(order, dtype=torch.bool)
    assigned.scatter_(1, assignment, True)
    free_physical = all_physical[~assigned]
    order[free_logical] = free_physical
    return order


def agent_goal_indices(graph, slot_env, slot_agent):
    owned = graph.edge_valid[slot_env] & (graph.edge_owner[slot_env] == slot_agent[:, None])
    target = torch.where((graph.edge_relation[slot_env] == AT) & owned,
                         graph.edge_dst[slot_env], graph.num_agents + graph.num_objects)
    return target.amin(-1) - graph.num_agents - graph.num_objects


def paired_placement_success(success, graph):
    valid = graph.edge_valid
    unbatched = valid.ndim == 1
    if unbatched:
        valid = valid.unsqueeze(0)
        relation = graph.edge_relation.unsqueeze(0)
        owner = graph.edge_owner.unsqueeze(0)
        src = graph.edge_src.unsqueeze(0)
        dst = graph.edge_dst.unsqueeze(0)
        success = success.unsqueeze(0) if success.ndim == 1 else success
    else:
        relation, owner, src, dst = (graph.edge_relation, graph.edge_owner,
                                     graph.edge_src, graph.edge_dst)
    holding = valid & (relation == HOLDING)
    placement = valid & ((relation == AT) | (relation == ON_TOP))
    paired = (holding[:, :, None] & placement[:, None, :] &
              (owner[:, :, None] == owner[:, None, :]) &
              (dst[:, :, None] == src[:, None, :]))
    result = (paired & success[:, None, :]).any(-1)
    return result[0] if unbatched else result
