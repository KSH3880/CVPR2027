"""Shared, simulator-independent Carry relation schema and checkpoint contract."""
from dataclasses import dataclass
import math
import torch
from utils.edge_context_spec import CONTEXT_MODE, validate_edge_context_config

from utils.edge_ontop_spec import ONTOP_CONTEXT_MODE, PACKET_FIELDS, validate_ontop_context_config

LEGACY_MODE = 'legacy_tokenhsi'
STATE_MODE = 'state_relation_v0'
ONTOP_MODE = 'state_relation_ontop_mixed_v1'
REL_HOLDING, REL_AT = 6, 7
STATE_FIELDS = ('phi', 'gate', 'satisfied', 'achieved')


@dataclass(frozen=True)
class CarryGraph:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_relation: torch.Tensor
    edge_owner: torch.Tensor
    edge_terminal: torch.Tensor
    edge_mask: torch.Tensor
    prereq_mask: torch.Tensor
    subgoal_target: torch.Tensor


def compile_carry_subgoal(num_agents, num_objects, device=None):
    if num_agents < 1 or num_objects < num_agents:
        raise ValueError('Carry requires M>=1 and O>=M (logical assigned slots first)')
    h = torch.arange(num_agents, device=device)
    o, g = num_agents + h, num_agents + num_objects + h
    src = torch.stack([h, o], -1).flatten()
    dst = torch.stack([o, g], -1).flatten()
    relation = torch.tensor([REL_HOLDING, REL_AT], device=device).repeat(num_agents)
    owner = h.repeat_interleave(2)
    target = 2 * h + 1
    pre = torch.zeros(2 * num_agents, 2 * num_agents, device=device, dtype=torch.bool)
    pre[target, target - 1] = True
    return CarryGraph(src, dst, relation, owner, relation == REL_AT,
                      torch.ones_like(src, dtype=torch.bool), pre, target)


def build_state_relation_matrix(num_agents, num_objects, device=None):
    graph = compile_carry_subgoal(num_agents, num_objects, device)
    size = 2 * num_agents + num_objects
    matrix = torch.zeros(size, size, dtype=torch.long, device=device)
    matrix.fill_diagonal_(1)  # existing REL_SELF
    matrix[graph.edge_src, graph.edge_dst] = graph.edge_relation
    return matrix


def validate_relation_config(config):
    mode = config.get('mode', LEGACY_MODE)
    if mode == ONTOP_CONTEXT_MODE:
        validate_ontop_context_config(config)
        return
    if mode == CONTEXT_MODE:
        validate_edge_context_config(config)
        return
    if mode == ONTOP_MODE:
        from utils.ontop_task_spec import validate_ontop_reward
        validate_ontop_reward(config)
        return
    if mode not in (LEGACY_MODE, STATE_MODE):
        raise ValueError('Unsupported relationReward mode: ' + str(mode))
    if mode == LEGACY_MODE:
        return
    allowed = {'mode', 'schema_version', 'state_reward_weight', 'progress_reward_weight',
               'subgoal_success_bonus', 'satisfaction_threshold', 'soft_gate', 'progress',
               'holding', 'at', 'success', 'observation', 'diagnostics'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError('Unsupported relationReward fields/operators: ' + ', '.join(sorted(unknown)))
    nested = {
        'soft_gate': {'beta', 'center'},
        'progress': {'kind', 'normalization_epsilon', 'approach_radius', 'delta', 'sigma'},
        'holding': {'hand_distance_scale'},
        'at': {'state_definition', 'near_distance_scale'},
        'observation': {'include_relation_state', 'relation_state_fields', 'include_subgoal_done'},
        'success': {'require_achieved_target_prerequisites', 'once_per_subgoal',
                    'require_current_target_prerequisites', 'z_tolerance', 'saturate_edge_rewards',
                    'saturate_edge_rewards_while_current', 'current_saturation_z_tolerance',
                    'current_success_reward',
                    'seed_achieved_from_valid_reset_state', 'suppress_bonus_for_initial_success',
                    'terminate_when_all_subgoals_done'},
        'diagnostics': {'enabled', 'log_interval', 'sample_envs', 'validate_tensors',
                        'timeline_enabled', 'timeline_sample_envs',
                        'timeline_every_episodes', 'timeline_max_steps'}}
    for key, keys in nested.items():
        if not isinstance(config.get(key, {}), dict) or set(config.get(key, {})) - keys:
            raise ValueError('Unsupported relationReward.' + key + ' configuration')
    if config.get('schema_version', 1) != 1:
        raise ValueError('Unsupported relation schema_version')
    diagnostics = config.get('diagnostics', {})
    if type(diagnostics.get('timeline_enabled', True)) is not bool:
        raise ValueError('diagnostics.timeline_enabled must be boolean')
    for key, default, minimum in (('timeline_sample_envs', 1, 0),
                                 ('timeline_every_episodes', 100, 1),
                                 ('timeline_max_steps', 600, 1)):
        value = diagnostics.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError('diagnostics.' + key + ' has an invalid integer value')
    progress = config.get('progress', {})
    progress_kind = progress.get('kind', 'distance')
    if progress_kind not in ('direction', 'distance'):
        raise ValueError('Unsupported progress kind: ' + str(progress_kind))
    if progress_kind == 'distance':
        if set(progress) - {'kind', 'delta', 'sigma'}:
            raise ValueError('distance progress accepts only kind, delta and sigma; no direction/velocity/blending options')
        for key, default, allow_zero in (('delta', .5, True), ('sigma', 1., False)):
            value = progress.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
                raise ValueError('Invalid distance progress ' + key)
    elif set(progress) & {'delta', 'sigma'}:
        raise ValueError('delta/sigma require distance progress')
    if progress_kind == 'direction':
        radius = progress.get('approach_radius')
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
            raise ValueError('Approach radius must be finite and positive')
    success = config.get('success', {})
    current = success.get('require_current_target_prerequisites', False)
    historical = success.get('require_achieved_target_prerequisites', True)
    if type(current) is not bool or type(historical) is not bool or current == historical:
        raise ValueError('Success requires exactly one of current or achieved target prerequisites')
    if type(success.get('saturate_edge_rewards', False)) is not bool:
        raise ValueError('success.saturate_edge_rewards must be boolean')
    if type(success.get('saturate_edge_rewards_while_current', False)) is not bool:
        raise ValueError('success.saturate_edge_rewards_while_current must be boolean')
    if (success.get('saturate_edge_rewards', False)
            and success.get('saturate_edge_rewards_while_current', False)):
        raise ValueError('Latched and current edge saturation are mutually exclusive')
    if 'z_tolerance' in success:
        tolerance = success['z_tolerance']
        if not current:
            raise ValueError('success.z_tolerance requires current target prerequisites')
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('success.z_tolerance must be finite and positive')
    current_reward = success.get('current_success_reward', 0.)
    if (isinstance(current_reward, bool) or not isinstance(current_reward, (int, float))
            or not math.isfinite(current_reward) or current_reward < 0):
        raise ValueError('success.current_success_reward must be finite and nonnegative')
    current_success_enabled = success.get('saturate_edge_rewards_while_current', False) or current_reward > 0
    # Keep the existing tolerance key for checkpoint compatibility. Both the
    # optional saturation and the independent per-step bonus use this At/Z test.
    if 'current_saturation_z_tolerance' in success:
        tolerance = success['current_saturation_z_tolerance']
        if not current_success_enabled:
            raise ValueError('success.current_saturation_z_tolerance requires current edge saturation or current success reward')
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('success.current_saturation_z_tolerance must be finite and positive')
    elif current_success_enabled:
        raise ValueError('Current edge saturation or current success reward requires current_saturation_z_tolerance')
    for key in ('once_per_subgoal',
                'seed_achieved_from_valid_reset_state', 'suppress_bonus_for_initial_success'):
        if success.get(key, True) is not True:
            raise ValueError('state_relation_v0 requires success.' + key + '=true')
    if success.get('terminate_when_all_subgoals_done', False):
        raise ValueError('Agreed v0 retains timeout/fall termination; success termination is disabled')
    obs = config.get('observation', {})
    if not obs.get('include_relation_state', True) or not obs.get('include_subgoal_done', True):
        raise ValueError('state_relation_v0 requires relation state and done observations')
    if tuple(obs.get('relation_state_fields', STATE_FIELDS)) != STATE_FIELDS:
        raise ValueError('relation state fields must be phi,gate,satisfied,achieved')
    positive = [config.get('state_reward_weight', .2), config.get('progress_reward_weight', .2),
                config.get('soft_gate', {}).get('beta', 30.),
                config.get('progress', {}).get('normalization_epsilon', 1e-6),
                config.get('holding', {}).get('hand_distance_scale', 5.),
                config.get('at', {}).get('near_distance_scale', 10.)]
    if not all(math.isfinite(v) and v > 0 for v in positive):
        raise ValueError('relation reward weights, scales and tolerances must be finite and positive')
    bonus = config.get('subgoal_success_bonus', 0.)
    if isinstance(bonus, bool) or not isinstance(bonus, (int, float)) or not math.isfinite(bonus) or bonus < 0:
        raise ValueError('subgoal_success_bonus must be finite and nonnegative')
    for v in (config.get('satisfaction_threshold', .9), config.get('soft_gate', {}).get('center', .8)):
        if not math.isfinite(v) or not 0 < v < 1:
            raise ValueError('relation thresholds must be inside (0,1)')
    at = config.get('at', {})
    state_definition = at.get('state_definition', 'box_near')
    if state_definition != 'box_near':
        raise ValueError('Unsupported At state definition: ' + str(state_definition))


def checkpoint_metadata(config):
    mode = config.get('mode', LEGACY_MODE)
    if mode == ONTOP_CONTEXT_MODE:
        return {'reward_mode': mode, 'schema_version': 3,
                'relation_taxonomy': {'holding': 6, 'at': 7, 'ontop': 8},
                'suffix_fields': list(PACKET_FIELDS), 'packet_version': 1, 'graph_record_width': 7,
                'context_dim_per_edge': 2,
                'context_fusion': 'semantic64_context2x32x64_concat128x64x64',
                'relation_reward_config': config}
    if mode == CONTEXT_MODE:
        return {'reward_mode': mode, 'schema_version': 2,
                'relation_taxonomy': {'holding': REL_HOLDING, 'at': REL_AT},
                'suffix_fields': ['pre', 'term'], 'context_dim_per_edge': 2,
                'context_fusion': 'semantic64_context2x32x64_concat128x64x64',
                'relation_reward_config': config}
    if mode == ONTOP_MODE:
        return {'reward_mode': mode, 'schema_version': 1,
                'relation_taxonomy': {'holding': REL_HOLDING, 'at': REL_AT, 'ontop': 8},
                'suffix_fields': list(STATE_FIELDS) + ['subgoal_done', 'scenario', 'base_agent'],
                'relation_reward_config': config}
    return {'reward_mode': mode, 'schema_version': 1 if mode == STATE_MODE else 0,
            'relation_taxonomy': {'holding': REL_HOLDING, 'at': REL_AT} if mode == STATE_MODE else {},
            'suffix_fields': list(STATE_FIELDS) + ['subgoal_done'] if mode == STATE_MODE else [],
            'relation_reward_config': config}


def check_checkpoint_metadata(weights, expected):
    saved = weights.get('relation_metadata')
    if saved is None:
        if expected['reward_mode'] != LEGACY_MODE:
            raise ValueError('Legacy checkpoint cannot load a state relation mode; explicit conversion is required')
        return
    for key in ('reward_mode', 'schema_version', 'relation_taxonomy', 'suffix_fields'):
        if saved.get(key) != expected[key]:
            raise ValueError('Checkpoint relation schema mismatch: ' + key)
    if expected['reward_mode'] in (CONTEXT_MODE, ONTOP_CONTEXT_MODE):
        for key in ('context_dim_per_edge', 'context_fusion'):
            if saved.get(key) != expected[key]:
                raise ValueError('Checkpoint context architecture mismatch: ' + key)
    if expected['reward_mode'] == ONTOP_CONTEXT_MODE:
        for key in ('packet_version', 'graph_record_width'):
            if saved.get(key) != expected[key]:
                raise ValueError('Checkpoint graph packet mismatch: ' + key)
    # Training and evaluation must use the same reward definition. Diagnostics may vary.
    def objective(c):
        return {k: v for k, v in c.items() if k != 'diagnostics'}
    if expected['reward_mode'] in (STATE_MODE, ONTOP_MODE, CONTEXT_MODE, ONTOP_CONTEXT_MODE) and objective(saved.get('relation_reward_config', {})) != objective(expected['relation_reward_config']):
        raise ValueError('Checkpoint relation reward config differs (diagnostics-only overrides allowed)')
