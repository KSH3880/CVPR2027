from copy import deepcopy

import torch

from utils import coupled_push_spec as coupled
from utils.before_spec import sample_roles, eligible_families
from utils.before_object_alias import alias_layout, replace_distractors
from utils.edge_ontop_spec import permute_graph
from utils.task_role_spec import task_graph_packet

SAMPLER = 'stage2_relations'
PRESETS = (*coupled.PRESETS, 'before', 'coupled', 'mixed',
           'before_sit', 'before_climb', 'before_stack')


def validate_env(env):
    from utils.experiments import native_config
    native = native_config('stage2')['env']
    if env['numAgents'] < 4 or env['numObjects'] < env['numAgents']:
        raise ValueError('stage2 requires at least four people and one object token per person')
    for key in ('stage2', 'before', 'relationGraph', 'relationReward'):
        if env[key] != native[key]:
            raise ValueError('stage2 contract differs: ' + key)
    base = deepcopy(env)
    base.pop('stage2')
    base.pop('before')
    base['relationGraph'] = deepcopy(native_config(
        'stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg')['env']['relationGraph'])
    coupled.validate_env(base)


def before_members(predecessor):
    members = predecessor >= 0
    rows, follower = members.nonzero(as_tuple=True)
    result = members.clone()
    result[rows, predecessor[rows, follower]] = True
    return result


def sample_episode(sizes, ranges, probabilities, preset='random_scenario', scene_weights=(1., 1., 1.), num_objects=None):
    count, people = sizes.shape[:2]
    objects = 2*people if num_objects is None else num_objects
    if objects < people:
        raise ValueError('stage2 requires numObjects >= numAgents')
    minimum = 2*people if preset == 'holding_ontop' else (
        people+people//2 if people == 4 and preset == 'before_stack' else people)
    if objects < minimum:
        raise ValueError(f'{preset} requires at least {minimum} object tokens')
    if preset == 'mixed' and objects == people:
        compatible = eligible_families(sizes, ranges)[..., :2].any(-1).any(-1)
        if not compatible.all():
            raise ValueError('mixed with M=N needs a SIT/CLIMB-compatible enable leader; increase M')
    result = _sample_episode(sizes, ranges, probabilities, preset, scene_weights)
    pending = (required_tokens(*result[1:]) > objects).nonzero(as_tuple=True)[0]
    for _ in range(1024):
        if not len(pending):
            return result
        fresh = _sample_episode(sizes[pending], ranges, probabilities, preset, scene_weights)
        accepted = required_tokens(*fresh[1:]) <= objects
        for dst, src in zip(result, fresh):
            dst[pending[accepted]] = src[accepted]
        pending = pending[~accepted]
    raise RuntimeError('stage2 object budget exhausted after 1024 scene resamples; increase M')


def required_tokens(groups, predecessor, templates):
    people = templates.shape[1]
    graph = compose_graph(templates, groups, predecessor, 2*people, shuffle=False)
    packet = task_graph_packet(graph, len(templates))
    dependent, _, _, used = alias_layout(packet, people, 2*people)
    return used+dependent.sum(-1)


def _sample_episode(sizes, ranges, probabilities, preset, scene_weights):
    count, people = sizes.shape[:2]
    device = sizes.device
    if preset not in PRESETS or people < 4:
        raise ValueError('Unsupported stage2 preset or person count')
    if people > 4:
        from utils.stage2_group_sampling import sample_groups
        return sample_groups(sizes, ranges, probabilities, preset)
    domain = torch.multinomial(sizes.new_tensor(scene_weights), count, replacement=True)
    forced = {'before': 0, 'coupled': 1, 'mixed': 2}
    if preset in forced:
        domain.fill_(forced[preset])
    elif preset.startswith('before_'):
        domain.zero_()
    elif preset != 'random_scenario':
        domain.fill_(1)
    groups = torch.arange(people, device=device)[None].expand(count, -1).clone()
    predecessor = torch.full_like(groups, -1)
    templates = torch.empty_like(groups)
    weights = sizes.new_tensor(list(probabilities.values()))
    for kind in range(3):
        rows = (domain == kind).nonzero(as_tuple=True)[0]
        if not len(rows):
            continue
        if kind == 0:
            pre, tasks = sample_roles(sizes[rows], ranges,
                preset if preset.startswith('before_') else 'random_scenario',
                solo_probabilities=list(probabilities.values())[1:], pairing_probability=.5)
            predecessor[rows], templates[rows] = pre, tasks
            groups[rows] = torch.where(pre >= 0, pre, groups[rows])
        elif kind == 1:
            groups[rows], templates[rows] = coupled.sample_episode(
                len(rows), people, 2*people, .5, weights, device,
                preset if preset in coupled.PRESETS else 'random_scenario')
        else:
            order = torch.rand(len(rows), people, device=device).argsort(-1)
            pair = order[:, :2]
            local_sizes = sizes[rows[:, None], pair]
            pre, tasks = sample_roles(local_sizes, ranges, 'random_scenario',
                                     solo_probabilities=list(probabilities.values())[1:], pairing_probability=1.)
            global_pre = pair.gather(1, pre.clamp_min(0))
            predecessor[rows[:, None], pair] = torch.where(pre >= 0, global_pre, -1)
            templates[rows] = 5
            templates[rows[:, None], pair] = tasks
            groups[rows[:, None], pair] = torch.where(pre >= 0, global_pre, pair)
            groups[rows[:, None], order[:, 2:]] = order[:, 2:3]
    return domain, groups, predecessor, templates


def compose_graph(templates, groups, predecessor, objects, shuffle=True):
    people = templates.shape[1]
    graph = coupled.compose_graph(templates, groups, False, individual_goals=True, num_objects=objects)
    rows, follower = (predecessor >= 0).nonzero(as_tuple=True)
    owner = graph.edge_owner[rows]
    dependent = ((owner == follower[:, None]) & graph.required_goal[rows]).long().argmax(-1)
    leading = ((owner == predecessor[rows, follower, None]) &
               (graph.edge_relation[rows] == 7) & graph.edge_valid[rows]).long().argmax(-1)
    graph.edge_dst[rows, dependent] = people + predecessor[rows, follower]
    graph.prereq_mask[rows, dependent, leading] = True
    if shuffle:
        graph = permute_graph(graph, torch.rand_like(graph.edge_src, dtype=torch.float).argsort(-1))
    return graph


def physical_mapping(templates, groups, predecessor, graph):
    people, objects = graph.num_agents, graph.num_objects
    primary = coupled.physical_assignment(templates, groups)
    primary = torch.where((predecessor >= 0) & (templates < 3), predecessor, primary)
    mapping = torch.cat((primary, coupled.spare_assignment(templates, objects)), -1)
    packet = task_graph_packet(graph, len(templates))
    mapping, _ = replace_distractors(mapping, packet, people)
    dependent, _, aliases, _ = alias_layout(packet, people, objects)
    valid, task, actor, payload, target = packet.reshape(len(packet), people, 5).long().unbind(-1)
    used = torch.stack((valid.bool() & (task >= 2), valid.bool()), -1)
    slots = torch.stack((payload, target), -1) - people
    used &= (slots >= 0) & (slots < objects)
    counts = torch.zeros_like(mapping)
    counts.scatter_add_(1, slots.clamp(0, objects-1).flatten(1), used.long().flatten(1))
    rows, record = dependent.nonzero(as_tuple=True)
    counts[rows, aliases[rows, record]-people] += 1
    return mapping, primary, counts > 0


def evaluation_groups(packet, groups):
    push = packet.reshape(len(packet), groups.shape[1], 5)[..., 1] == 4
    solo = torch.arange(groups.shape[1], device=groups.device)[None]
    return torch.where(push, groups, solo)
