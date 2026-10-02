"""Reset-wise Shared9 evaluation with disjoint pairs and immutable role-compatible assets."""
from copy import deepcopy

import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, permute_graph
from utils.edge_interaction_spec import SIT, CLIMB
from utils.stage2_shared_spec import PAIRS, ALIASES, PRESETS

SAMPLER = 'multi_agent_stage2_rescue_shared9'


def training_spec(spec):
    result = dict(spec, sampler='two_agent_four_object_stage2_rescue_shared9', edge_capacity=4)
    result.pop('num_agents', None)
    result.pop('num_objects', None)
    return result


def validate_sampler(spec, m, o):
    from utils.stage2_shared_spec import validate_sampler as validate_training
    if (type(m) is not int or type(o) is not int or not 2 <= m <= o or
            spec.get('num_agents') != m or spec.get('num_objects') != o or
            spec.get('edge_capacity') != 2 * m):
        raise ValueError('Shared9 evaluation requires objects >= humans >= 2 and 2*humans edge slots')
    validate_training(training_spec(spec), 2, 4)


def validate_env(env):
    from utils.stage2_shared_spec import validate_env as validate_training
    validate_sampler(env['relationGraph'], env['numAgents'], env['numObjects'])
    original = deepcopy(env)
    original.update(numAgents=2, numObjects=4, relationGraph=training_spec(env['relationGraph']))
    validate_training(original)


def sample_sizes(n, m, o, sizes, device='cpu', generator=None):
    """Support slots retain the training range; other slots fit ordinary AND payload roles.

    Their XY intersection is 40cm and Z is 40..50cm. This allows fresh AT/shared
    draws without resizing Isaac actors or pinning a viewer to one training pool.
    """
    result = torch.empty(n, o, 3, device=device)
    for axis in range(3):
        for columns, low, high in (
                (slice(0, m // 2), *sizes['support'][axis]),
                (slice(m // 2, o), max(sizes['ordinary'][axis][0], sizes['payload'][axis][0]),
                 min(sizes['ordinary'][axis][1], sizes['payload'][axis][1]))):
            if low > high:
                raise ValueError('Shared9 evaluation requires overlapping ordinary/payload size ranges')
            shape = result[:, columns, axis].shape
            count = round((high - low) / sizes['step']) + 1
            result[:, columns, axis] = low + sizes['step'] * torch.randint(
                count, shape, device=device, generator=generator)
    return result


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False, generator=None):
    from utils.edge_stage2_spec import _empty_graph, _put, validate_graph
    from utils.edge_scenario_spec import CLIMB_TEMPLATES, _rows
    m, o = spec['num_agents'], spec['num_objects']
    validate_sampler(spec, m, o)
    preset = ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError('Unknown Shared9 preset: ' + preset)
    graph = _empty_graph(n, m, o, 2 * m, device)
    names = tuple(PAIRS)
    weights = torch.tensor([spec['family_probabilities'][name] for name in names], device=device)
    independent_weights = torch.tensor(list(spec['independent_template_probabilities'].values()), device=device)

    def shuffle(values):
        return [values[i] for i in torch.randperm(len(values), device=device, generator=generator).tolist()]

    def draw(values):
        return int(torch.multinomial(values, 1, generator=generator))

    for batch in range(n):
        humans = shuffle(list(range(m))) if preset in ('random_scenario', 'independent') else list(range(m))
        if role_swap and preset not in ('random_scenario', 'independent'):
            humans.reverse()
        supports = shuffle(list(range(m // 2)))
        payloads = shuffle(list(range(m // 2, o)))
        goals = shuffle(list(range(m)))
        offset = 0

        def bundle(owner, template, binding):
            nonlocal offset
            for src, dst, relation, required in _rows(owner, template, binding, m, o):
                _put(graph, batch, offset, owner, src, dst, relation, required)
                offset += 1

        cooperative = preset not in ('random_scenario', 'independent') or (
            preset == 'random_scenario' and bool(torch.rand((), device=device, generator=generator)
                                                < spec['cooperative_probability']))
        if cooperative:
            while len(humans) >= 2:
                available = weights.clone()
                remaining = len(humans) - 2
                reserve = (remaining + 1) // 2
                for index, name in enumerate(names):
                    pair = PAIRS[name]
                    children = pair.count('HOLDING_ON_TOP')
                    needed_supports = int(not name.startswith('at_'))
                    needed_payloads = children + int(name.startswith('at_'))
                    if (len(supports) < needed_supports or len(payloads) < needed_payloads or
                            len(supports) + len(payloads) - needed_supports - needed_payloads < reserve):
                        available[index] = 0
                if preset != 'random_scenario':
                    index = names.index(preset)
                    if not available[index] > 0:
                        raise ValueError('Not enough role-compatible objects for Shared9 preset ' + preset)
                else:
                    if not available.sum() > 0:
                        raise ValueError('Not enough role-compatible objects for Shared9 evaluation')
                    index = draw(available)
                name = names[index]
                owners, humans = humans[:2], humans[2:]
                base = payloads.pop() if name.startswith('at_') else supports.pop()
                for owner, template in zip(owners, PAIRS[name]):
                    binding = ((base, goals.pop()) if template == 'HOLDING_AT' else
                               (payloads.pop(), base) if template == 'HOLDING_ON_TOP' else (base,))
                    bundle(owner, template, binding)

        objects = shuffle(payloads + supports)
        while humans:
            owner = humans.pop(0)
            available = independent_weights.clone()
            if len(objects) < len(humans) + 2:
                available[CLIMB_TEMPLATES.index('HOLDING_ON_TOP')] = 0
            movable = [index for index, obj in enumerate(objects) if obj >= m // 2]
            if not movable:
                for template in ('HOLDING', 'HOLDING_AT', 'HOLDING_ON_TOP'):
                    available[CLIMB_TEMPLATES.index(template)] = 0
            template = CLIMB_TEMPLATES[draw(available)]
            source = objects.pop(movable[0]) if movable else objects.pop()
            binding = ((source, goals.pop()) if template == 'HOLDING_AT' else
                       (source, objects.pop()) if template == 'HOLDING_ON_TOP' else (source,))
            bundle(owner, template, binding)
    if spec['shuffle_edge_order']:
        graph = permute_graph(graph, torch.rand(n, 2 * m, device=device, generator=generator).argsort(-1))
    validate_graph(graph, allow_shared_ontop=True)
    return graph


def family_counts(graph):
    """Count cooperative pairs individually, including scenes with several pair types."""
    n = graph.edge_src.shape[0]
    counts = torch.zeros(n, 10, device=graph.edge_src.device)
    template_names = {AT: 'HOLDING_AT', ON_TOP: 'HOLDING_ON_TOP', SIT: 'SIT', CLIMB: 'CLIMB'}
    lookup = {tuple(sorted(pair)): index + 1 for index, pair in enumerate(PAIRS.values())}
    for row in range(n):
        groups = {}
        for edge in (graph.edge_valid[row] & graph.required_goal[row]).nonzero().flatten().tolist():
            relation = int(graph.edge_relation[row, edge])
            if relation not in template_names:
                continue
            base = int((graph.edge_src if relation == AT else graph.edge_dst)[row, edge])
            groups.setdefault(base, []).append(template_names[relation])
        for group in groups.values():
            if len(group) == 2:
                counts[row, lookup[tuple(sorted(group))]] += 1
        if not counts[row].any():
            counts[row, 0] = 1
    return counts
