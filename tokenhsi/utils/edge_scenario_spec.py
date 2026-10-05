"""Context-free joint scenarios with random object and goal bindings."""
import math

import torch

from utils.edge_context_spec import EdgeContextGraph, HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, permute_graph, select_graph, validate_graph as validate_base_graph
from utils.edge_interaction_spec import SIT, CLIMB


INDEPENDENT_CLIMB_SAMPLER = 'two_agent_three_object_independent_with_climb'
UNIFIED_SAMPLER = 'two_agent_four_object_independent_unified'
PAIRED_PLACEMENT_SAMPLER = 'two_agent_four_object_paired_placement'
CLIMB_TEMPLATES = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP')
PRESETS = ('random_scenario', 'holding', 'sit', 'climb', 'holding_at', 'holding_ontop')


def scenario_templates(spec):
    return CLIMB_TEMPLATES


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


def validate_sampler(spec, m=2, o=None):
    expected = {'mode', 'sampler', 'semantic_only', 'edge_capacity',
                'max_edges_per_agent', 'template_probabilities',
                'random_binding', 'shuffle_edge_order'}
    sampler = spec.get('sampler')
    if sampler == UNIFIED_SAMPLER:
        expected.add('shuffle_token_order')
        if spec.get('shuffle_token_order') is not True:
            raise ValueError('Unified sampler requires token permutation')
    if spec.get('policy_task_roles') is True:
        expected.add('policy_task_roles')
        if sampler != UNIFIED_SAMPLER or spec['template_probabilities']['HOLDING'] != 0:
            raise ValueError('Task-role policy requires unified sampling without standalone holding')
    if spec.get('owner_holding_state') is True:
        expected.add('owner_holding_state')
    if set(spec) != expected or spec.get('mode') != 'edge_composition' or \
            sampler not in (INDEPENDENT_CLIMB_SAMPLER, PAIRED_PLACEMENT_SAMPLER, UNIFIED_SAMPLER):
        raise ValueError('Unsupported scenario sampler')
    if spec.get('owner_holding_state', False) and sampler not in (PAIRED_PLACEMENT_SAMPLER, UNIFIED_SAMPLER):
        raise ValueError('Owner-HOLDING state requires paired placement')
    binding = ({'objects': 'canonical', 'goals': 'canonical'} if sampler == UNIFIED_SAMPLER
               else {'objects': 'disjoint', 'goals': 'disjoint'})
    required_objects = 4 if sampler in (PAIRED_PLACEMENT_SAMPLER, UNIFIED_SAMPLER) else 3
    if o is None:
        o = required_objects
    if (m, o) != (2, required_objects) or spec['edge_capacity'] != 4 or \
            spec['max_edges_per_agent'] != 2 or spec['semantic_only'] is not True or \
            spec['random_binding'] != binding or \
            type(spec['shuffle_edge_order']) is not bool:
        raise ValueError('Scenario requires 2 humans, {} objects and semantic capacity 4'.format(required_objects))
    p = spec['template_probabilities']
    templates = scenario_templates(spec)
    if tuple(p) != templates or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v < 0 for v in p.values()) or \
            not math.isclose(sum(p.values()), 1., abs_tol=1e-8):
        raise ValueError('Scenario template probabilities must follow the contract')
    if sampler == PAIRED_PLACEMENT_SAMPLER:
        if any(p[k] != 0 for k in ('HOLDING', 'SIT', 'CLIMB')):
            raise ValueError('Paired placement sampler permits AT and ON_TOP only')
    elif sampler != UNIFIED_SAMPLER and p['HOLDING_ON_TOP'] > .5:
        raise ValueError('Independent ON_TOP probability cannot exceed 0.5')


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


def compose_graph(templates, bindings, shuffle=False, generator=None, device='cpu', num_objects=3):
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
            rows = _rows(agent, templates[batch][agent], bindings[batch][agent], o=num_objects)
            for row, (s, d, r, req) in enumerate(rows, offset):
                src[batch, row], dst[batch, row], relation[batch, row] = s, d, r
                owner[batch, row], valid[batch, row], required[batch, row] = agent, True, req
            offset += len(rows)
    graph = EdgeContextGraph(('slot0', 'slot1', 'slot2', 'slot3'), 2, num_objects,
        src, dst, relation, owner, valid, required,
        torch.zeros(n, 4, 4, dtype=torch.bool, device=device),
        torch.full((n, 4), -1, dtype=torch.long, device=device))
    if shuffle:
        graph = permute_graph(graph, torch.rand(n, 4, device=device,
                                               generator=generator).argsort(-1))
    validate_graph(graph)
    return graph


def compose_canonical_graph(template_ids, shuffle=False):
    n = len(template_ids)
    device = template_ids.device
    owners = torch.arange(2, device=device)[None].expand(n, -1)
    placement = template_ids >= 3
    src = torch.stack((owners, owners + 2), -1).flatten(1)
    dst = torch.stack((owners + 2,
        torch.where(template_ids == 3, owners + 6, owners + 4)), -1).flatten(1)
    base_relations = torch.tensor([HOLDING, SIT, CLIMB, HOLDING, HOLDING], device=device)
    relation = torch.stack((base_relations[template_ids],
        torch.where(template_ids == 3, AT, ON_TOP)), -1).flatten(1)
    owner = owners[:, :, None].expand(-1, -1, 2).flatten(1)
    valid = torch.stack((torch.ones_like(placement), placement), -1).flatten(1)
    required = torch.stack((~placement, placement), -1).flatten(1)
    order = valid.long().argsort(dim=-1, descending=True, stable=True)
    values = [value.masked_fill(~valid, 0).gather(1, order)
              for value in (src, dst, relation, owner, valid, required)]
    graph = EdgeContextGraph(('slot0', 'slot1', 'slot2', 'slot3'), 2, 4, *values,
        torch.zeros(n, 4, 4, dtype=torch.bool, device=device),
        torch.full((n, 4), -1, dtype=torch.long, device=device))
    if shuffle:
        graph = permute_graph(graph, torch.rand(n, 4, device=device).argsort(-1))
    return graph


def sample_size_conditioned_graph(probabilities, spec, preset):
    if spec.get('policy_task_roles', False):
        from utils.task_role_spec import task_preset
        preset = task_preset(preset)
    n = len(probabilities)
    if preset == 'random_scenario':
        template_ids = torch.multinomial(probabilities.flatten(0, 1), 1).reshape(n, 2)
    else:
        template_ids = torch.full((n, 2), PRESETS.index(preset) - 1,
                                  dtype=torch.long, device=probabilities.device)
    return compose_canonical_graph(template_ids,
        shuffle=spec['shuffle_edge_order'] and preset == 'random_scenario')


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None):
    if spec.get('policy_task_roles', False):
        from utils.task_role_spec import task_preset
        preset = task_preset(preset)
    paired = spec['sampler'] == PAIRED_PLACEMENT_SAMPLER
    unified = spec['sampler'] == UNIFIED_SAMPLER
    num_objects = 4 if paired or unified else 3
    validate_sampler(spec, o=num_objects)
    if preset not in PRESETS:
        raise ValueError('Unknown scenario TASK_GRAPH preset: ' + preset)
    if paired and preset not in ('random_scenario', 'holding_at', 'holding_ontop'):
        raise ValueError('Paired placement permits AT and ON_TOP presets only')
    templates_available = scenario_templates(spec)
    probs = torch.tensor([spec['template_probabilities'][k] for k in templates_available],
                         device=device)
    template_rows, binding_rows = [], []
    fixed = {'holding': 'HOLDING', 'sit': 'SIT', 'climb': 'CLIMB',
             'holding_at': 'HOLDING_AT',
             'holding_ontop': 'HOLDING_ON_TOP'}
    if preset == 'random_scenario':
        if paired:
            pair_ids = torch.multinomial(probs, n, replacement=True,
                                         generator=generator).tolist()
        elif unified:
            pair_ids = torch.multinomial(probs, 2 * n, replacement=True,
                                         generator=generator).reshape(n, 2).tolist()
        else:
            top = probs[-1]
            pair_probs = torch.outer(probs, probs)
            pair_probs[-1, -1] = 0
            pair_probs[-1, :-1] /= 1 - top
            pair_probs[:-1, -1] /= 1 - top
            pair_probs[:-1, :-1] *= (1 - 2 * top) / (1 - top) ** 2
            pair_ids = torch.multinomial(pair_probs.flatten(), n, replacement=True,
                                         generator=generator).tolist()
    for row in range(n):
        if preset == 'random_scenario':
            if paired:
                template = templates_available[pair_ids[row]]
                templates = (template, template)
            else:
                ids = pair_ids[row] if unified else divmod(pair_ids[row], len(templates_available))
                templates = (templates_available[ids[0]], templates_available[ids[1]])
        else:
            templates = (fixed[preset], fixed[preset] if paired or unified else 'HOLDING')
            if role_swap and not (paired or unified):
                templates = templates[::-1]
        objects = list(range(num_objects)) if unified else torch.randperm(num_objects, device=device, generator=generator).tolist()
        goals = list(range(2)) if unified else torch.randperm(2, device=device, generator=generator).tolist()
        bindings = tuple((objects[agent], goals[agent]) if template == 'HOLDING_AT'
            else (objects[agent], objects[2 + agent] if paired or unified else objects[2]) if template == 'HOLDING_ON_TOP'
            else (objects[agent],) for agent, template in enumerate(templates))
        template_rows.append(templates); binding_rows.append(bindings)
    return compose_graph(template_rows, binding_rows,
        shuffle=spec['shuffle_edge_order'] if preset == 'random_scenario' else False,
        generator=generator, device=device, num_objects=num_objects)


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    preset = ('holding_at' if spec['sampler'] == PAIRED_PLACEMENT_SAMPLER
              or spec.get('policy_task_roles', False) else 'holding')
    return select_graph(sample_graph(1, spec, device or 'cpu', preset=preset), 0)


def validate_graph(graph):
    validate_base_graph(graph)
    if graph.edge_src.ndim != 1:
        for i in range(graph.edge_src.shape[0]):
            validate_graph(select_graph(graph, i))
        return
    if graph.prereq_mask.any() or (graph.term_index >= 0).any():
        raise ValueError('Independent scenarios cannot contain temporal context')
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


def agent_goal_indices(graph, slot_env, slot_agent):
    owned = graph.edge_valid[slot_env] & (graph.edge_owner[slot_env] == slot_agent[:, None])
    target = torch.where((graph.edge_relation[slot_env] == AT) & owned,
                         graph.edge_dst[slot_env], graph.num_agents + graph.num_objects)
    return target.amax(-1) - graph.num_agents - graph.num_objects


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
