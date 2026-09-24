"""Independent Stage-1 relation graphs with constant START/KEEP context."""
import math

import torch

from utils.edge_context_spec import EdgeContextGraph, HOLDING, AT
from utils.edge_ontop_spec import (ON_TOP, batched, permute_graph, select_graph,
    validate_graph as validate_ontop_graph)
from utils.edge_interaction_spec import SIT, CLIMB


STAGE1_CONTEXT_MODE = 'state_relation_edge_stage1_v1'
STAGE1_PACKET_FIELDS = ('valid', 'src', 'dst', 'relation', 'owner', 'start', 'keep')
STAGE1_SEMANTIC_FIELDS = STAGE1_PACKET_FIELDS[:5]
PATTERNS = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP',
            'HOLDING_SIT', 'HOLDING_CLIMB')
PRESETS = ('random_stage1', 'holding', 'sit', 'climb', 'holding_at',
           'holding_ontop', 'holding_sit', 'holding_climb')
PRIMITIVE_SAMPLER = 'two_agent_three_object_stage1_primitives'


def semantic_packet_size(capacity):
    return len(STAGE1_SEMANTIC_FIELDS) * capacity


def semantic_graph_packet(graph, batch_size):
    fields = [batched(getattr(graph, key), batch_size).float() for key in
        ('edge_valid', 'edge_src', 'edge_dst', 'edge_relation', 'edge_owner')]
    return (torch.stack(fields, -1) * fields[0][..., None]).flatten(1)


def parse_semantic_packet(packet):
    width = len(STAGE1_SEMANTIC_FIELDS)
    if packet.ndim != 2 or packet.shape[-1] % width:
        raise ValueError('Expected stored semantic graph packet [N,5*E]')
    fields = packet.reshape(packet.shape[0], -1, width)
    return (fields[..., 0].bool(), fields[..., 1].long(), fields[..., 2].long(),
        fields[..., 3].long(), fields[..., 4].long())


def validate_relation_rsi(spec, skills):
    if set(spec) != {'HOLDING', 'SIT', 'CLIMB'}:
        raise ValueError('relationRsi requires HOLDING/SIT/CLIMB rows')
    allowed = {
        'HOLDING': {'loco', 'pickUp', 'carryWith'},
        'SIT': {'loco', 'sit'},
        'CLIMB': {'loco', 'climb'},
    }
    for relation, values in spec.items():
        if len(values) != len(skills) or any(isinstance(v, bool) or
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-8):
            raise ValueError('Invalid relationRsi probability row')
        if any(v and skill not in allowed[relation] for skill, v in zip(skills, values)):
            raise ValueError('relationRsi skill does not match its relation')
        if values[skills.index('loco')] <= 0:
            raise ValueError('relationRsi must retain loco starts')


def standalone_interaction_owners(graph, env_ids, any_owned_object=False):
    """Return agents whose SIT/CLIMB target is their assigned physical box."""
    relation = graph.edge_relation[env_ids][:, None, :]
    destination = graph.edge_dst[env_ids][:, None, :]
    owner = graph.edge_owner[env_ids][:, None, :]
    valid = graph.edge_valid[env_ids][:, None, :]
    agents = torch.arange(graph.num_agents, device=relation.device)[None, :, None]
    own_slot = (destination == graph.num_agents + agents)
    return (valid & ((relation == SIT) | (relation == CLIMB))
            & (owner == agents) & (any_owned_object | own_slot)).any(-1)


def ground_standalone_interaction_targets(graph, env_ids, assignments,
                                          box_states, box_sizes,
                                          platform_pos=None, platform_default_pos=None,
                                          any_owned_object=False):
    """Floor only standalone SIT/CLIMB targets and deactivate their source platforms."""
    mask = standalone_interaction_owners(graph, env_ids, any_owned_object)
    local_env, agent = mask.nonzero(as_tuple=True)
    if not len(local_env):
        return
    world_env = env_ids[local_env]
    physical_box = assignments[world_env, agent]
    box_states[world_env, physical_box, 2] = box_sizes[world_env, physical_box, 2] / 2
    if platform_pos is not None:
        platform_pos[world_env, agent] = platform_default_pos[world_env, agent]


def max_stage1_stack_height(box_sizes):
    """Upper bound for the Stage-1 O_i -> O_X two-box stack in each environment."""
    return box_sizes[..., 2].topk(2, dim=-1).values.sum(-1)


DEFAULT_GRAPH = dict(
    mode='edge_composition', sampler='two_agent_three_object_stage1',
    edge_capacity=4, max_edges_per_agent=2,
    pattern_probabilities=dict(zip(PATTERNS, (.10, .10, .10, .25, .15, .15, .15))),
    target_binding={'standalone': 'own_object', 'composite_support': 'free_object',
                    'allow_teammate_object': False, 'max_free_object_users': 1},
    shuffle_edge_order=True)


def validate_stage1_context_config(config):
    schema = config.get('schema_version')
    semantic_only = schema in (7, 8, 9, 10)
    scenario = schema in (8, 9, 10)
    expected = {
        'mode', 'schema_version', 'state_reward_weight', 'progress_reward_weight',
        'success_reward_weight', 'satisfaction_threshold', 'holding', 'at', 'ontop',
        'sit', 'progress', 'success',
        'observation', 'diagnostics'}
    if schema != 8:
        expected.add('climb')
    if not semantic_only:
        expected.update(('context', 'contextReward'))
    if schema in (6, 7, 8, 9, 10):
        expected.add('stage1_variant')
    if scenario:
        expected.add('task_sharing')
    if set(config) != expected:
        raise ValueError('Unsupported Stage-1 relationReward fields')
    if config['mode'] != STAGE1_CONTEXT_MODE or schema not in (5, 6, 7, 8, 9, 10):
        raise ValueError('Expected Stage-1 schema 5 through 10')
    if config['schema_version'] == 6 and config['stage1_variant'] != 'primitive_relation_rsi':
        raise ValueError('Unsupported Stage-1 schema 6 variant')
    if schema == 7 and config['stage1_variant'] != 'climb_only_rsi_no_context':
        raise ValueError('Unsupported Stage-1 schema 7 variant')
    expected_variant = {8: 'scenario_no_climb', 9: 'scenario_with_climb',
                        10: 'independent_climb_placement_region'}.get(schema)
    if scenario and config['stage1_variant'] != expected_variant:
        raise ValueError('Unsupported Stage-1 scenario variant')
    if not semantic_only and config['context'] != {'kind': 'start_keep_constant'}:
        raise ValueError('Stage-1 requires constant START/KEEP context')
    if not semantic_only and config['contextReward'] != {'start_weight': 0., 'keep_weight': 0.}:
        raise ValueError('Stage-1 START/KEEP auxiliary reward must be disabled')
    expected_success = {'at_z_tolerance': .001,
        'saturation': 'paired_placement_current_success' if scenario else 'own_success',
        'terminate_when_all_subgoals_done': False}
    if config['success'] != expected_success:
        raise ValueError('Unsupported Stage-1 saturation contract')
    expected_sharing = {'self': 1., 'teammate': 0.} if schema == 10 else \
        {'self': .9, 'teammate': .1}
    if scenario and config['task_sharing'] != expected_sharing:
        raise ValueError('Scenario task sharing does not match its schema')
    expected_observation = ({'graph_packet_fields': list(STAGE1_SEMANTIC_FIELDS)}
        if semantic_only else {'edge_context_fields': ['start', 'keep'],
            'graph_packet_fields': list(STAGE1_PACKET_FIELDS)})
    if config['observation'] != expected_observation:
        raise ValueError('Stage-1 observation packet does not match its schema')
    if config['holding'] != {'hand_distance_scale': 10.}:
        raise ValueError('Unsupported Stage-1 HOLDING geometry')
    if config['at'] != {'state_definition': 'box_near', 'near_distance_scale': 10.}:
        raise ValueError('Unsupported Stage-1 AT geometry')
    expected_ontop = {'state_definition': 'centered_stack_world_z',
            'near_distance_scale': 10., 'z_tolerance': .001,
            'vertical_extent': 'rotated_bbox'}
    if schema == 10:
        expected_ontop['success_inner_margin_fraction'] = .1
    if config['ontop'] != expected_ontop:
        raise ValueError('Unsupported Stage-1 ON_TOP geometry')
    original_sit = {'state_definition': 'tokenhsi_tar_sit_pos',
                    'near_distance_scale': 10.,
                    'target_local_offset': [0., 0., 0.1381430834425038]}
    box_top_sit = config['sit'].get('state_definition') == 'box_top_plus_pelvis_clearance'
    if box_top_sit:
        sit = config['sit']
        clearance = sit.get('pelvis_clearance')
        if (set(sit) != {'state_definition', 'near_distance_scale', 'pelvis_clearance'}
                or sit['near_distance_scale'] != 10.
                or isinstance(clearance, bool) or not isinstance(clearance, (int, float))
                or not math.isfinite(clearance) or clearance <= 0):
            raise ValueError('Unsupported Stage-1 box-top SIT geometry')
    elif config['sit'] != original_sit:
        raise ValueError('Unsupported Stage-1 SIT geometry')
    if schema != 8:
        expected_climb = {'state_definition': 'root_target', 'near_distance_scale': 10.,
                          'target_height': 'rotated_bbox_top_plus_char_h'}
        climb = config['climb']
        climb_fields = set(expected_climb) | {'feet_height_tolerance'}
        if schema in (7, 9):
            climb_fields.add('success_phi_threshold')
        if schema == 10:
            climb_fields.update(('success_inner_margin_fraction', 'root_height_tolerance'))
        if (set(climb) != climb_fields
                or any(climb.get(k) != v for k, v in expected_climb.items())):
            raise ValueError('Unsupported Stage-1 CLIMB geometry')
        tolerance = climb['feet_height_tolerance']
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance <= 0):
            raise ValueError('Stage-1 CLIMB feet tolerance must be finite and positive')
        if schema in (7, 9) and (climb['success_phi_threshold'] != .6 or tolerance != .07):
            raise ValueError('CLIMB success must use phi=0.6 and feet=0.07m')
        if schema == 10 and (climb['success_inner_margin_fraction'] != .05 or
                climb['root_height_tolerance'] != .2 or tolerance != .07):
            raise ValueError('Independent CLIMB success requires 5% margin, 20cm root, 7cm feet')
    expected_progress = {'kind': 'distance', 'delta': .5, 'sigma': 1.}
    if schema in (7, 9, 10):
        expected_progress['climb_pinning'] = 'bbox_valid_radius'
    if config['progress'] != expected_progress:
        raise ValueError('Unsupported Stage-1 distance progress')
    for key in ('state_reward_weight', 'progress_reward_weight', 'success_reward_weight'):
        if config[key] != .2:
            raise ValueError('Stage-1 reward weights must all be 0.2')
    if config['satisfaction_threshold'] != .9:
        raise ValueError('Stage-1 satisfaction threshold must be 0.9')


def validate_sampler(spec, m=2, o=3):
    from utils.edge_scenario_spec import (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
        SCENARIO_INDEPENDENT_SAMPLER,
        validate_sampler as validate_scenario)
    if spec.get('sampler') in (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
                               SCENARIO_INDEPENDENT_SAMPLER):
        return validate_scenario(spec, m, o)
    if (m, o) != (2, 3):
        raise ValueError('Stage-1 sampler/presets require M=2, O=3')
    semantic_only = spec.get('semantic_only', False)
    if type(semantic_only) is not bool or set(spec) != set(DEFAULT_GRAPH) | ({'semantic_only'} if semantic_only else set()):
        raise ValueError('Unsupported Stage-1 edge composition fields')
    primitive = spec.get('sampler') == PRIMITIVE_SAMPLER
    contract = dict(DEFAULT_GRAPH)
    if primitive:
        contract.update(sampler=PRIMITIVE_SAMPLER, max_edges_per_agent=1)
    if semantic_only and (not primitive or spec['pattern_probabilities'] !=
            {'HOLDING': 0., 'SIT': 0., 'CLIMB': 1.}):
        raise ValueError('Semantic-only Stage-1 requires CLIMB-only primitive sampling')
    for key in ('mode', 'sampler', 'edge_capacity', 'max_edges_per_agent', 'target_binding'):
        if spec[key] != contract[key]:
            raise ValueError('Unsupported Stage-1 edge composition ' + key)
    p = spec['pattern_probabilities']
    expected_patterns = PATTERNS[:3] if primitive else PATTERNS
    if (tuple(p) != expected_patterns or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v < 0 for v in p.values())
            or not math.isclose(sum(p.values()), 1., abs_tol=1e-8)):
        raise ValueError('Stage-1 pattern probabilities must follow the contract and sum to one')
    ox_mass = sum(p.get(k, 0.) for k in ('HOLDING_ON_TOP', 'HOLDING_SIT', 'HOLDING_CLIMB'))
    if ox_mass > .5 + 1e-8:
        raise ValueError('Stage-1 O_X pattern mass must be <=0.5')
    if type(spec['shuffle_edge_order']) is not bool:
        raise ValueError('shuffle_edge_order must be boolean')


def _draw_patterns(n, probabilities, device, generator):
    """Keep exact per-agent marginals while allowing at most one O_X user."""
    p = torch.tensor([probabilities[k] for k in PATTERNS], device=device)
    ox_ids = torch.tensor([4, 5, 6], device=device)
    local_ids = torch.tensor([0, 1, 2, 3], device=device)
    q = p[ox_ids].sum()
    pattern = local_ids[torch.multinomial(
        p[local_ids], n * 2, replacement=True, generator=generator)].reshape(n, 2)
    joint = torch.rand(n, device=device, generator=generator)
    for agent, mask in ((0, joint < q), (1, (joint >= q) & (joint < 2 * q))):
        count = int(mask.sum())
        if count:
            choice = torch.multinomial(p[ox_ids], count, replacement=True,
                                       generator=generator)
            pattern[mask, agent] = ox_ids[choice]
    return pattern


def _draw_primitive_patterns(n, probabilities, device, generator):
    p = torch.tensor([probabilities[k] for k in PATTERNS[:3]], device=device)
    return torch.multinomial(p, n * 2, replacement=True, generator=generator).reshape(n, 2)


def compose_graph(pattern, shuffle=False, generator=None):
    n = pattern.shape[0]
    if pattern.shape != (n, 2) or ((pattern < 0) | (pattern >= len(PATTERNS))).any():
        raise ValueError('Expected Stage-1 pattern [N,2]')
    device = pattern.device
    agent = torch.arange(2, device=device)[None].expand(n, -1)
    composite = pattern >= 3
    rel0 = torch.where(pattern == 1, SIT,
        torch.where(pattern == 2, CLIMB, HOLDING))
    rel1_table = torch.tensor([0, 0, 0, AT, ON_TOP, SIT, CLIMB], device=device)
    rel1 = rel1_table[pattern]
    valid = torch.stack([torch.ones_like(pattern, dtype=torch.bool), composite], -1).flatten(1)
    relation = torch.stack([rel0, rel1], -1).flatten(1)
    owner = agent.repeat_interleave(2, -1)

    # O_i is logical object M+i, O_X is logical object M+2, G_i is M+O+i.
    src0 = agent
    dst0 = 2 + agent
    src1 = torch.where((rel1 == SIT) | (rel1 == CLIMB), agent, 2 + agent)
    dst1 = torch.where(rel1 == AT, 5 + agent, torch.full_like(agent, 4))
    src = torch.stack([src0, src1], -1).flatten(1)
    dst = torch.stack([dst0, dst1], -1).flatten(1)
    required = valid.clone()
    pre = torch.zeros(n, 4, 4, dtype=torch.bool, device=device)
    term = torch.full((n, 4), -1, dtype=torch.long, device=device)

    src = src.masked_fill(~valid, 0)
    dst = dst.masked_fill(~valid, 0)
    relation = relation.masked_fill(~valid, 0)
    owner = owner.masked_fill(~valid, 0)
    graph = EdgeContextGraph(('slot0', 'slot1', 'slot2', 'slot3'), 2, 3,
        src, dst, relation, owner, valid, required, pre, term)
    validate_graph(graph)
    if shuffle:
        graph = permute_graph(graph, torch.rand(
            n, 4, device=device, generator=generator).argsort(-1))
    return graph


def sample_graph(n, spec, device='cpu', preset='random_stage1', role_swap=False,
                 generator=None):
    from utils.edge_scenario_spec import (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
        SCENARIO_INDEPENDENT_SAMPLER,
        sample_graph as sample_scenario)
    if spec.get('sampler') in (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
                               SCENARIO_INDEPENDENT_SAMPLER):
        return sample_scenario(n, spec, device, preset, role_swap, generator)
    validate_sampler(spec)
    if preset not in PRESETS:
        raise ValueError('Unknown Stage-1 TASK_GRAPH preset: ' + preset)
    primitive = spec['sampler'] == PRIMITIVE_SAMPLER
    if primitive and preset not in PRESETS[:4]:
        raise ValueError('Primitive Stage-1 permits holding/sit/climb presets only')
    if spec.get('semantic_only', False) and preset not in ('random_stage1', 'climb'):
        raise ValueError('CLIMB-only Stage-1 permits climb preset only')
    if preset == 'random_stage1':
        draw = _draw_primitive_patterns if primitive else _draw_patterns
        pattern = draw(n, spec['pattern_probabilities'], device, generator)
    else:
        selected = {'holding': 0, 'sit': 1, 'climb': 2, 'holding_at': 3,
                    'holding_ontop': 4, 'holding_sit': 5,
                    'holding_climb': 6}[preset]
        pattern = torch.tensor((selected, 0), device=device).expand(n, -1).clone()
        if role_swap:
            pattern = pattern.flip(-1)
    return compose_graph(pattern,
        shuffle=spec['shuffle_edge_order'] if preset == 'random_stage1' else False,
        generator=generator)


def compile_stage1_graph(spec, m, o, device=None):
    from utils.edge_scenario_spec import (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
        SCENARIO_INDEPENDENT_SAMPLER,
        compile_graph)
    if spec.get('sampler') in (SCENARIO_SAMPLER, SCENARIO_CLIMB_SAMPLER,
                               SCENARIO_INDEPENDENT_SAMPLER):
        return compile_graph(spec, m, o, device)
    if spec.get('mode') != 'edge_composition':
        raise ValueError('Stage-1 accepts its conflict-free edge sampler only')
    validate_sampler(spec, m, o)
    preset = 'climb' if spec.get('semantic_only', False) else 'holding'
    return select_graph(sample_graph(1, spec, device=device or 'cpu', preset=preset), 0)


def validate_graph(graph):
    try:
        from utils.edge_scenario_spec import validate_graph as validate_scenario
        validate_scenario(graph)
        return
    except ValueError:
        pass
    validate_ontop_graph(graph)
    if graph.edge_src.ndim != 1:
        for i in range(graph.edge_src.shape[0]):
            validate_graph(select_graph(graph, i))
        return
    active = graph.edge_valid.nonzero(as_tuple=False).flatten()
    if graph.prereq_mask.any() or (graph.term_index >= 0).any():
        raise ValueError('Stage-1 graph cannot contain dependency or END links')
    if not torch.equal(graph.required_goal, graph.edge_valid):
        raise ValueError('Every active Stage-1 edge must be a required current goal')
    ox = graph.num_agents + 2
    users = set()
    for i in active.tolist():
        owner = int(graph.edge_owner[i])
        relation = int(graph.edge_relation[i])
        src, dst = int(graph.edge_src[i]), int(graph.edge_dst[i])
        own = graph.num_agents + owner
        if relation == HOLDING and (src, dst) != (owner, own):
            raise ValueError('Stage-1 HOLDING must bind H_i -> O_i')
        if relation == AT and (src, dst) != (own, graph.num_agents + graph.num_objects + owner):
            raise ValueError('Stage-1 AT must bind O_i -> G_i')
        if relation == ON_TOP and (src, dst) != (own, ox):
            raise ValueError('Stage-1 ON_TOP must bind O_i -> O_X')
        if relation in (SIT, CLIMB):
            expected = own if not any(int(graph.edge_relation[j]) == HOLDING and
                int(graph.edge_owner[j]) == owner for j in active.tolist()) else ox
            if (src, dst) != (owner, expected):
                raise ValueError('Stage-1 SIT/CLIMB binding violates own/O_X contract')
        if dst == ox:
            users.add(owner)
    if len(users) > 1:
        raise ValueError('At most one Stage-1 agent may use O_X')
