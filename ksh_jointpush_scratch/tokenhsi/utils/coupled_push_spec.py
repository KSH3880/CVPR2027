from dataclasses import replace
import torch

from utils.edge_scenario_spec import compose_canonical_graph
from utils.jointpush_geometry import rotate
from utils.stage1_jointpush import PUSH

SAMPLER = 'stage2_jointpush'
PRESETS = ('random_scenario', 'all_solo', 'all_coupled', 'sit', 'climb', 'holding_at', 'holding_ontop', 'push')


def checkpoint_contract(env, inference=False):
    from copy import deepcopy
    from utils.before_push_spec import push_contract
    result = dict(policy=deepcopy(env['stage2Policy']),push=push_contract(env),
        coupled=deepcopy(env['stage2JointPush']),graph=deepcopy(env['relationGraph']),
        box=deepcopy(env['box']),people=env['numAgents'],objects=env['numObjects'])
    if inference and env['numAgents'] >= 1 and env['numObjects'] >= env['numAgents']:
        result.update(people=4, objects=8)
        result['coupled']['physical_boxes'] = physical_box_count(4)
    return result


def check_checkpoint(checkpoint,env,inference=False):
    from copy import deepcopy
    saved = deepcopy(checkpoint.get('coupled_push_contract'))
    if inference and saved is not None:
        saved.update(people=4, objects=8)
        saved['coupled']['physical_boxes'] = physical_box_count(4)
    if saved!=checkpoint_contract(env, inference=inference):
        raise ValueError('Coupled PUSH checkpoint settings differ from the environment')


def validate_env(env):
    from copy import deepcopy
    from utils.experiments import native_config
    from utils.before_push_spec import validate_env as validate_before
    people = env['numAgents']
    if people < 1 or env['numObjects'] < people:
        raise ValueError('Coupled PUSH requires at least one person and one object token per person')
    if env['relationGraph'].get('human_attention_count_normalization', 'none') not in ('none', 'coupled', 'coupled_none'):
        raise ValueError('Invalid human attention count normalization')
    peers = env['relationGraph'].get('coupled_peer_selection', 'all')
    fusion = env['relationGraph'].get('human_relation_fusion', 'none')
    if fusion not in ('none', 'learnedbg', 'zerobg'):
        raise ValueError('Invalid coupled relation fusion mode')
    if fusion != 'none' and (peers != 'all' or env['relationGraph'].get('human_attention_count_normalization', 'none') != 'none'):
        raise ValueError('Coupled relation fusion requires all peers and no count normalization')
    if peers not in ('all', 'keep1', 'shrink1'):
        raise ValueError('Invalid coupled peer selection')
    if peers != 'all' and env['stage2JointPush'].get('goal_mode') != 'individual':
        raise ValueError('Single-peer selection requires individual coupled PUSH goals')
    coupled = deepcopy(env['stage2JointPush'])
    penalty = coupled.pop('premature_motion_penalty', None)
    if penalty is not None:
        from utils.premature_push import validate_config
        validate_config(penalty)
        if coupled.get('goal_mode') != 'individual' or coupled.get('contact_face') != 'group_four':
            raise ValueError('Premature motion penalty requires individual four-face PUSH')
    native = native_config('stage2_coupled_push')['env']['stage2JointPush']
    if coupled.pop('physical_boxes') != physical_box_count(people, env['numObjects']):
        raise ValueError('Coupled PUSH physical bank does not match the people count')
    native.pop('physical_boxes')
    mode = (coupled.pop('goal_mode', 'shared'), coupled.pop('contact_face', 'rear'),
            coupled.pop('gate_alignment', True))
    if mode not in (('shared', 'rear', True), ('individual', 'group_four', False)):
        raise ValueError('Unsupported coupled goal/contact/gate combination')
    probability = coupled.pop('coupled_probability')
    native.pop('coupled_probability')
    if not 0 <= probability <= 1:
        raise ValueError('Invalid coupled PUSH probability')
    for key in ('solo_source_max_distance', 'solo_goal_max_distance'):
        if not 2 <= coupled.pop(key, 10.) <= 10:
            raise ValueError('Invalid solo PUSH distance maximum')
    for key in ('source_distance', 'goal_distance'):
        bounds = coupled.pop(key, [2., 10.])
        native.pop(key, None)
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
            raise ValueError('Invalid coupled PUSH distance range: ' + key)
    carry = coupled.pop('solo_carry_goal_distance', None)
    if carry is not None and (len(carry) != 2 or not 1 <= carry[0] <= carry[1]):
        raise ValueError('Invalid solo CARRY goal distance range')
    if coupled!=native:
        raise ValueError('Unsupported coupled PUSH sampling/physics contract')
    from utils.stage1_platforms import validate_stage1_platform_config
    validate_stage1_platform_config(env)
    base=deepcopy(env)
    base['numObjects'] = max(base['numObjects'], 2*people)
    base.pop('stage2JointPush')
    original=native_config('stage2_before_push')['env']
    base['before']=deepcopy(original['before'])
    base['box']['reset']=deepcopy(original['box']['reset'])
    base['relationGraph']=deepcopy(original['relationGraph'])
    if env['relationGraph']['template_probabilities']!=base['relationGraph']['template_probabilities']:
        raise ValueError('Coupled solo probabilities must follow stage1_push')
    validate_before(base)


def expected_solo_fraction(people,probability):
    expected=[0.,1.]
    for n in range(2,people+1):
        expected.append((1-probability)*(1+expected[n-1])+probability*sum(expected[:n-1])/(n-1))
    return expected[people]/people


def compile_graph(spec, people, objects, device=None):
    from utils.edge_ontop_spec import select_graph
    groups = torch.arange(people, device=device)[None]
    return select_graph(compose_graph(torch.full_like(groups, 3), groups, False, num_objects=objects), 0)


def physical_box_count(people, objects=None):
    extra = max(0, objects-2*people) if objects is not None else 0
    return 4*people+sum(people//size for size in range(1, people+1))+extra


def asset_bank(count, env, device):
    from utils.stage1_jointpush import box_profiles
    people = env['numAgents']
    push_end = physical_box_count(people)
    sizes = torch.empty(count, physical_box_count(people, env['numObjects']), 3, device=device)
    ranges = env['box']['build']['taskSizeRanges']
    for bank, name in enumerate(('HOLDING_AT','SIT','CLIMB','SUPPORT')):
        for axis, (low, high) in enumerate(ranges[name]):
            sizes[:,bank*people:(bank+1)*people,axis] = torch.randint(round(low*20),round(high*20)+1,(count,people),device=device)/20
    profiles = torch.tensor(box_profiles(), device=device)
    base = profiles[torch.randint(len(profiles),(count,people),device=device)]
    members = torch.cat([torch.full((people//size,), size, device=device) for size in range(1, people+1)])
    offset = 4*people
    for size in range(1, people+1):
        slots = people//size
        sizes[:,offset:offset+slots] = base[:,:slots]
        offset += slots
    sizes[:,4*people:push_end,0] *= members
    extra = sizes.shape[1]-push_end
    if extra:
        for axis, (low, high) in enumerate(ranges['SUPPORT']):
            sizes[:,push_end:,axis] = torch.randint(round(low*20),round(high*20)+1,(count,extra),device=device)/20
    masses = sizes.prod(-1)*100.
    physical = env['jointPush']['box']
    if 'density' in physical:
        masses[:,4*people:push_end] = sizes[:,4*people:push_end].prod(-1)*physical['density']
    else:
        masses[:,4*people:push_end] = physical['mass']*members
    return sizes, masses


def physical_assignment(templates, groups):
    n = groups.shape[1]
    people = torch.arange(n, device=groups.device)[None].expand_as(groups)
    counts = group_sizes(groups)
    push_slot = 4*n+people
    offset = 5*n
    for size in range(2, n+1):
        leader = (counts == size) & (groups == people)
        slot = (leader.long().cumsum(-1)-1).gather(1, groups)
        push_slot = torch.where(counts == size, offset+slot, push_slot)
        offset += n//size
    return torch.where(templates == 5, push_slot,
        torch.where(templates == 1, n+people, torch.where(templates == 2, 2*n+people, people)))


def sample_groups(count, people, probability, device, preset='random_scenario'):
    order = torch.rand(count, people, device=device).argsort(-1)
    groups = torch.full_like(order, -1)
    cursor = torch.zeros(count, dtype=torch.long, device=device)
    rows = torch.arange(count, device=device)
    rank = torch.arange(people, device=device)[None]
    for _ in range(people):
        remaining = people-cursor
        coupled = (remaining >= 2) & (torch.rand(count, device=device) < probability)
        if preset == 'all_solo':
            coupled.zero_()
        size = torch.where(coupled,
            2+(torch.rand(count, device=device)*(remaining-1).clamp_min(0)).long(), 1)
        if preset == 'all_coupled':
            size = remaining
        active = remaining > 0
        leader = order[rows, cursor.clamp_max(people-1)]
        selected = active[:, None] & (rank >= cursor[:, None]) & (rank < (cursor+size)[:, None])
        groups.scatter_(1, order, torch.where(selected, leader[:, None], groups.gather(1, order)))
        cursor += size*active
    return groups


def group_sizes(groups):
    return (groups[:, :, None] == groups[:, None, :]).sum(-1)


def sample_templates(groups, probabilities):
    solo = torch.multinomial(probabilities, groups.numel(), replacement=True).reshape_as(groups)
    return torch.where(group_sizes(groups) > 1, 5, solo)


def sample_episode(count, people, objects, probability, probabilities, device, preset):
    forced = dict(sit=1,climb=2,holding_at=3,holding_ontop=4,push=5)
    if preset == 'holding_ontop' and objects < 2*people:
        raise ValueError('All-person ON_TOP requires a distinct spare support per person')
    groups = torch.empty(count, people, dtype=torch.long, device=device)
    templates = torch.empty_like(groups)
    pending = torch.arange(count, device=device)
    while len(pending):
        sampled = sample_groups(len(pending), people, probability, device,
            preset if preset in ('random_scenario','all_coupled','all_solo') else 'all_solo')
        tasks = sample_templates(sampled, probabilities)
        if preset == 'all_coupled':
            tasks.fill_(5)
        if preset in forced:
            tasks.fill_(forced[preset])
        valid = (tasks == 4).sum(-1) <= objects-people
        groups[pending[valid]] = sampled[valid]
        templates[pending[valid]] = tasks[valid]
        pending = pending[~valid]
    return groups, templates


def spare_assignment(templates, objects):
    count, people = templates.shape
    spare = objects-people
    order = torch.arange(people, device=templates.device)[None].expand(count,-1)
    if spare < people:
        order = (templates == 4).long().argsort(dim=-1, descending=True, stable=True)
    assignment = 3*people+order[:,:min(spare,people)]
    if spare > people:
        extra = torch.arange(physical_box_count(people), physical_box_count(people, objects), device=templates.device)
        assignment = torch.cat((assignment, extra[None].expand(count,-1)), -1)
    return assignment


def assign_regions(groups):
    people = groups.shape[1]
    order = torch.rand_like(groups, dtype=torch.float).argsort(-1)
    same = groups[:, :, None] == groups[:, None, :]
    earlier = order[:, None, :] < order[:, :, None]
    return (same & earlier).sum(-1)


def compose_graph(templates, groups, shuffle=True, individual_goals=False, num_objects=None):
    people = groups.shape[1]
    objects = 2*people if num_objects is None else num_objects
    graph = compose_canonical_graph(torch.where(templates == 5, 3, templates), shuffle, objects)
    push = (templates == 5).gather(1, graph.edge_owner)
    leader = graph.edge_owner if individual_goals else groups.gather(1, graph.edge_owner)
    return replace(graph,
        edge_relation=torch.where(push & (graph.edge_relation == 6), PUSH, graph.edge_relation),
        edge_dst=torch.where(push & (graph.edge_relation == 7), people+objects+leader, graph.edge_dst))


def region_states(physical_states, physical_sizes, members, region):
    sizes = physical_sizes.clone()
    sizes[..., 0] /= members
    offset = torch.zeros_like(physical_sizes)
    offset[..., 0] = sizes[..., 0]*(region+.5)-physical_sizes[..., 0]/2
    delta = rotate(physical_states[..., 3:7], offset)
    states = physical_states.clone()
    states[..., :3] += delta
    states[..., 7:10] += torch.cross(states[..., 10:13], delta, dim=-1)
    return states, sizes


def inverse_rotate(q, vector):
    inverse = q.clone()
    inverse[..., :3] *= -1
    return rotate(inverse, vector)


def group_contact_gate(hands, boxes, sizes, goals, groups, tolerance=.18, success_distance=.2, z_tolerance=.001, goal_saturation=False):
    q = boxes[..., None, 3:7].expand(*hands.shape[:-1], 4)
    local = inverse_rotate(q, hands-boxes[..., None, :3])
    excess_x = (local[..., 0].abs()-sizes[..., None, 0]/2).clamp_min(0)
    excess_z = (local[..., 2].abs()-sizes[..., None, 2]/2).clamp_min(0)
    distance = (excess_x.square()+excess_z.square()+(local[..., 1]+sizes[..., None, 1]/2).square()).sqrt()
    normal = torch.zeros_like(sizes)
    normal[..., 1] = 1.
    normal = rotate(boxes[..., 3:7], normal)
    delta = goals-boxes[..., :3]
    at_position = (delta.norm(dim=-1) <= success_distance) & (delta[..., 2].abs() <= z_tolerance)
    aligned = (normal[..., :2]*delta[..., :2]).sum(-1) > 0
    on_face = (distance <= tolerance).all(-1) & (local[..., 1] <= 0).all(-1) & (aligned | at_position)
    same = groups[:, :, None] == groups[:, None, :]
    gate = (on_face[:, None, :] | ~same).all(-1)
    if goal_saturation:
        gate = gate | at_position
    return torch.where(group_sizes(groups) > 1, gate, True)

