"""Shared, simulator-independent Carry relation schema and checkpoint contract."""
from dataclasses import dataclass
import math
import torch

LEGACY_MODE = 'legacy_tokenhsi'
STATE_MODE = 'state_relation_v0'
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
    if mode not in (LEGACY_MODE, STATE_MODE):
        raise ValueError('Unsupported relationReward mode: ' + str(mode))
    if mode == LEGACY_MODE:
        return
    allowed = {'mode', 'schema_version', 'state_delta_weight', 'velocity_progress_weight',
               'subgoal_success_bonus', 'satisfaction_threshold', 'soft_gate', 'progress',
               'holding', 'at', 'success', 'observation', 'diagnostics'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError('Unsupported relationReward fields/operators: ' + ', '.join(sorted(unknown)))
    nested = {
        'soft_gate': {'beta', 'center'},
        'progress': {'mode', 'target_speed', 'velocity_scale', 'normalization_epsilon'},
        'holding': {'hand_distance_scale'},
        'at': {'near_distance_scale', 'near_fraction', 'putdown_xy_tolerance', 'putdown_z_tolerance'},
        'observation': {'include_relation_state', 'relation_state_fields', 'include_subgoal_done'},
        'success': {'require_achieved_target_prerequisites', 'once_per_subgoal',
                    'seed_achieved_from_valid_reset_state', 'suppress_bonus_for_initial_success',
                    'terminate_when_all_subgoals_done'},
        'diagnostics': {'enabled', 'log_interval', 'sample_envs', 'validate_tensors'}}
    for key, keys in nested.items():
        if not isinstance(config.get(key, {}), dict) or set(config.get(key, {})) - keys:
            raise ValueError('Unsupported relationReward.' + key + ' configuration')
    if config.get('progress', {}).get('mode', 'gaussian') not in ('gaussian', 'signed_linear'):
        raise ValueError('Unsupported relationReward.progress.mode')
    if config.get('schema_version', 1) != 1:
        raise ValueError('Unsupported relation schema_version')
    success = config.get('success', {})
    for key in ('require_achieved_target_prerequisites', 'once_per_subgoal',
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
    positive = [config.get('state_delta_weight', 1.), config.get('velocity_progress_weight', .2),
                config.get('subgoal_success_bonus', 5.), config.get('soft_gate', {}).get('beta', 30.),
                config.get('progress', {}).get('target_speed', 1.5),
                config.get('progress', {}).get('velocity_scale', 5.),
                config.get('progress', {}).get('normalization_epsilon', 1e-6),
                config.get('holding', {}).get('hand_distance_scale', 5.),
                config.get('at', {}).get('near_distance_scale', 10.),
                config.get('at', {}).get('putdown_xy_tolerance', .1),
                config.get('at', {}).get('putdown_z_tolerance', .001)]
    if not all(math.isfinite(v) and v > 0 for v in positive):
        raise ValueError('relation reward weights, scales and tolerances must be finite and positive')
    for v in (config.get('satisfaction_threshold', .9), config.get('soft_gate', {}).get('center', .8)):
        if not math.isfinite(v) or not 0 < v < 1:
            raise ValueError('relation thresholds must be inside (0,1)')
    alpha = config.get('at', {}).get('near_fraction', .5)
    if not 0 < alpha < config.get('satisfaction_threshold', .9):
        raise ValueError('near_fraction must be positive and below satisfaction_threshold')


def checkpoint_metadata(config):
    mode = config.get('mode', LEGACY_MODE)
    return {'reward_mode': mode, 'schema_version': 1 if mode == STATE_MODE else 0,
            'relation_taxonomy': {'holding': REL_HOLDING, 'at': REL_AT} if mode == STATE_MODE else {},
            'suffix_fields': list(STATE_FIELDS) + ['subgoal_done'] if mode == STATE_MODE else [],
            'relation_reward_config': config}


def check_checkpoint_metadata(weights, expected):
    saved = weights.get('relation_metadata')
    if saved is None:
        if expected['reward_mode'] != LEGACY_MODE:
            raise ValueError('Legacy checkpoint cannot resume state_relation_v0; explicit conversion is required')
        return
    for key in ('reward_mode', 'schema_version', 'relation_taxonomy', 'suffix_fields'):
        if saved.get(key) != expected[key]:
            raise ValueError('Checkpoint relation schema mismatch: ' + key)
    # Training and evaluation must use the same reward definition. Diagnostics may vary.
    def objective(c):
        return {k: v for k, v in c.items() if k != 'diagnostics'}
    if expected['reward_mode'] == STATE_MODE and objective(saved.get('relation_reward_config', {})) != objective(expected['relation_reward_config']):
        raise ValueError('Checkpoint relation reward config differs (diagnostics-only overrides allowed)')
