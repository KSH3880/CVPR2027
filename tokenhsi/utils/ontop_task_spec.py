"""Mixed carry/stack topology, shared by simulation and shuffled PPO batches."""
from dataclasses import replace
import torch

from utils.relation_task_spec import ONTOP_MODE, compile_carry_subgoal
REL_ONTOP = 8
SCENARIOS = ('carry_carry', 'carry_ontop_independent', 'carry_ontop_dependent')


def validate_ontop_reward(config):
    from utils.relation_task_spec import validate_relation_config, STATE_MODE
    base = dict(config)
    base['mode'] = STATE_MODE
    ontop = base.pop('ontop', {})
    pre = base.pop('prerequisite', {})
    validate_relation_config(base)
    expected = dict(state_definition='surface_center_distance', distance_scale=10.0,
                    source_state='source_box_bottom_face_center', target_state='support_box_top_face_center',
                    surface_offsets_frame='box_local_rotated_to_world', progress_source='source_box_center',
                    progress_target='support_box_center', require_support_owner_current_success=False)
    if ontop != expected:
        raise ValueError('Unsupported OnTop geometry/success configuration')
    if pre != dict(reduction='minimum', state_timing='previous_step', empty_value=1.0):
        raise ValueError('Unsupported OnTop prerequisite configuration')
    if base.get('progress', {}).get('kind') != 'distance':
        raise ValueError('Mixed OnTop requires distance progress')
    success = base.get('success', {})
    if (not success.get('saturate_edge_rewards_while_current')
            or success.get('require_current_target_prerequisites')):
        raise ValueError('Mixed OnTop requires local current success saturation')


def validate_mixture(config):
    """Reject unsupported graph semantics rather than silently ignoring YAML edits."""
    expected = dict(schedule='simultaneous_fixed', assignment='fixed_env_groups',
                    role_assignment='random_per_episode', role_assignment_fixed_within_episode=True,
                    reset_distribution='base_carry', initialize_stacked=False, mask_unused_goals=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Unsupported scenarioMixture.' + key)
    scenarios = config.get('scenarios', [])
    if [s.get('name') for s in scenarios] != list(SCENARIOS):
        raise ValueError('Expected carry, independent OnTop, dependent OnTop scenarios in order')
    for i, scenario in enumerate(scenarios):
        count = scenario.get('num_envs')
        if type(count) is not int or count <= 0:
            raise ValueError('Scenario num_envs must be a positive integer')
        terminal = 'at_b' if i == 0 else 'ontop_b'
        edges = [dict(id='holding_a', source='Ha', target='Oa', relation='holding', owner='a', prerequisites=[]),
                 dict(id='at_a', source='Oa', target='Ga', relation='at', owner='a', prerequisites=['holding_a']),
                 dict(id='holding_b', source='Hb', target='Ob', relation='holding', owner='b', prerequisites=[]),
                 dict(id=terminal, source='Ob', target=('Gb', 'Ox', 'Oa')[i],
                      relation='at' if i == 0 else 'ontop', owner='b',
                      prerequisites=['holding_b', 'at_a'] if i == 2 else ['holding_b'])]
        if scenario.get('edges') != edges or scenario.get('terminal_edges') != dict(a='at_a', b=terminal):
            raise ValueError('Unsupported scenario graph: ' + SCENARIOS[i])
        if i == 1 and scenario.get('support') != dict(node='Ox', fixed=True, resample_pose_on_reset=True):
            raise ValueError('Independent OnTop requires a fixed, resettable Ox')


def scenario_ids(config, num_envs, evaluation=False, selected='mixed', device=None):
    validate_mixture(config)
    counts = [s['num_envs'] for s in config['scenarios']]
    if selected != 'mixed':
        if not evaluation or selected not in SCENARIOS:
            raise ValueError('A single OnTop scenario can be selected only for evaluation')
        return torch.full((num_envs,), SCENARIOS.index(selected), dtype=torch.long, device=device)
    if not evaluation and num_envs != sum(counts):
        raise ValueError('Training numEnvs must equal the configured scenario counts')
    if evaluation:
        # Weighted systematic sampling also gives a meaningful scenario for N=1.
        positions = (torch.arange(num_envs, device=device) + .5) * sum(counts) / num_envs
        boundaries = torch.tensor(counts, device=device).cumsum(0)[:-1]
        return torch.bucketize(positions, boundaries)
    return torch.repeat_interleave(torch.arange(3, device=device), torch.tensor(counts, device=device))


def mixed_graph(scenario, base_agent):
    """Physical-agent edge order: H0,T0,H1,T1. Roles affect destinations/parents."""
    n, device = scenario.numel(), scenario.device
    graph = compile_carry_subgoal(2, 3, device)
    dst = graph.edge_dst.expand(n, -1).clone()
    relation = graph.edge_relation.expand(n, -1).clone()
    pre = graph.prereq_mask.expand(n, -1, -1).clone()
    row = torch.arange(n, device=device)
    b_terminal = 2 * (1 - base_agent) + 1
    active = scenario != 0
    support = torch.where(scenario == 1, 4, 2 + base_agent)
    dst[row[active], b_terminal[active]] = support[active]
    relation[row[active], b_terminal[active]] = REL_ONTOP
    dep = scenario == 2
    pre[row[dep], b_terminal[dep], 2 * base_agent[dep] + 1] = True
    return replace(graph, edge_dst=dst, edge_relation=relation, prereq_mask=pre)


def graph_for_envs(graph, env_ids):
    if graph.prereq_mask.ndim == 2:
        return graph
    return replace(graph, edge_dst=graph.edge_dst[env_ids],
                   edge_relation=graph.edge_relation[env_ids], prereq_mask=graph.prereq_mask[env_ids])


def mixed_policy_graph(scenario, base_agent):
    graph = mixed_graph(scenario, base_agent)
    n, device = scenario.numel(), scenario.device
    matrix = torch.eye(7, dtype=torch.long, device=device).expand(n, -1, -1).clone()
    row = torch.arange(n, device=device)
    matrix[row[:, None], graph.edge_src[None], graph.edge_dst] = graph.edge_relation
    valid_nodes = torch.ones(n, 7, dtype=torch.bool, device=device)
    active = scenario != 0
    valid_nodes[row[active], 5 + 1 - base_agent[active]] = False
    return graph, matrix, valid_nodes


def rotate_offset(quaternion, offset):
    """xyzw quaternion acting on a local box-space vector."""
    q = quaternion[..., :3]
    t = 2 * torch.cross(q, offset, dim=-1)
    return offset + quaternion[..., 3:] * t + torch.cross(q, t, dim=-1)


def terminal_geometry(states, sizes, goals, scenario, base_agent):
    """Logical box states [N,3,13]; return state endpoints and XY-progress target."""
    n = states.shape[0]
    row = torch.arange(n, device=states.device)
    support_id = torch.where(scenario == 1, 2, base_agent)
    support = states[row, support_id]
    offset = torch.zeros_like(states[:, :2, :3])
    offset[..., 2] = -sizes[:, :2, 2] / 2
    bottoms = states[:, :2, :3] + rotate_offset(states[:, :2, 3:7], offset)
    top_offset = torch.zeros_like(support[:, :3])
    top_offset[:, 2] = sizes[row, support_id, 2] / 2
    tops = support[:, :3] + rotate_offset(support[:, 3:7], top_offset)
    is_top = (scenario[:, None] != 0) & (torch.arange(2, device=states.device)[None] != base_agent[:, None])
    source = torch.where(is_top[..., None], bottoms, states[:, :2, :3])
    target = torch.where(is_top[..., None], tops[:, None], goals)
    progress_target = torch.where(is_top[..., None], support[:, None, :3], goals)
    return source, target, progress_target, is_top
