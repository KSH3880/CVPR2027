from copy import deepcopy
import hashlib
from pathlib import Path

import torch

PRESETS = ('random_scenario', 'sit', 'climb')
# PUSH scenes: 0 the pushed cube becomes the bridge, 1 the bridge is already there,
# 2 the bridge is already there and the pusher moves a decoy cube elsewhere.
SCENES = ('dependent', 'present', 'independent')
PUSH_PRESETS = ('random_scenario', 'bridge_push', 'bridge_present', 'decoy_push')
COMMON_KEYS = {'mode', 'stage_size_range', 'stage_size_step', 'spawn_margin', 'heading_jitter', 'far_size_range',
               'clearance', 'top_drop', 'landing_margin', 'object_band', 'spawn_depth', 'approach_distance',
               'stage_density', 'parked_height', 'near_start', 'task_probabilities'}
KEYS = {
    'scripted': COMMON_KEYS | {'height', 'gap_range', 'width_range', 'size_step', 'present_probability',
                               'arrival_time_range', 'speed_range', 'density', 'friction', 'tracking_frequency',
                               'damping_ratio', 'max_acceleration'},
    'push': COMMON_KEYS | {'scene_probabilities', 'floor_offset', 'decoy_distance', 'decoy_goal_distance',
                           'floor_task_distance', 'push'},
}
PUSH_KEYS = {'box', 'rsi', 'rsi_probability', 'approach_stand_off', 'approach_clearance', 'demo_family_probabilities'}


def presets(env):
    if env['bridge']['mode'] == 'push4':
        return ('random_scenario',)
    return PUSH_PRESETS if env['bridge']['mode'] == 'push' else PRESETS


def extra_physical_objects(env):
    if env['bridge']['mode'] == 'push4':
        return 4
    # Scripted: the bridge box. PUSH: the bridge cube and the decoy cube.
    return 2 if env['bridge']['mode'] == 'push' else 1


def validate_env(env):
    if env['bridge'].get('mode') == 'push4':
        from utils.bridge_push4_spec import validate_env as validate_push4
        return validate_push4(env)
    from utils.before_spec import validate_env as validate_before
    bridge = env['bridge']
    if bridge.get('mode') not in KEYS or set(bridge) != KEYS[bridge['mode']]:
        raise ValueError('Bridge configuration differs from the agreed contract')
    if (env['numAgents'], env['numObjects']) != (2, 4):
        raise ValueError('Bridge scenes use two humans and four object tokens')
    if env['box']['reset']['platformMode'] != 'ground_only':
        raise ValueError('Bridge scenes disable the Stage1 platforms (platformMode=ground_only)')
    if set(bridge['task_probabilities']) != {'SIT', 'CLIMB'}:
        raise ValueError('Bridge crossing tasks are SIT and CLIMB on the far stage')
    if bridge['clearance'] <= 0:
        raise ValueError('Bridge clearance must be positive')
    if bridge['mode'] == 'push':
        push = bridge['push']
        if set(push) != PUSH_KEYS or set(bridge['scene_probabilities']) != set(SCENES):
            raise ValueError('PUSH bridge configuration differs from the agreed contract')
        if abs(sum(bridge['scene_probabilities'].values()) - 1) > 1e-6 or not 0 <= push['rsi_probability'] <= 1:
            raise ValueError('PUSH bridge probabilities are invalid')
        if 'size_range' not in push['box']:
            raise ValueError('The PUSH bridge uses cubes')
    else:
        low, high = bridge['stage_size_range']
        reach = high / 2 + bridge['far_size_range'][1] + max(
            bridge['gap_range'][1], bridge['width_range'][1] / 2 + bridge['approach_distance'][1])
        if env['envSpacing'] < reach:
            raise ValueError('envSpacing is too small for two stages and the approaching bridge')
        if bridge['gap_range'][0] <= 2 * bridge['clearance']:
            raise ValueError('Bridge clearance must leave a bridge length')
    base = deepcopy(env)
    base.pop('bridge')
    validate_before(base)


def sample_templates(n, preset, probabilities, device):
    if preset not in PRESETS:
        raise ValueError('Unknown bridge preset: ' + preset)
    if preset != 'random_scenario':
        return torch.full((n, 2), PRESETS.index(preset), device=device, dtype=torch.long)
    weights = torch.tensor([probabilities['SIT'], probabilities['CLIMB']], device=device)
    return torch.multinomial(weights, 2 * n, replacement=True).reshape(n, 2) + 1


def sample_scenes(n, preset, probabilities, device):
    if preset not in PUSH_PRESETS:
        raise ValueError('Unknown PUSH bridge preset: ' + preset)
    if preset != 'random_scenario':
        return torch.full((n,), PUSH_PRESETS.index(preset) - 1, device=device, dtype=torch.long)
    weights = torch.tensor([probabilities[name] for name in SCENES], device=device)
    return torch.multinomial(weights, n, replacement=True)


def push_episode_graph(templates, crosser, scene, num_objects):
    """Canonical graph with the floor human's HOLDING edge turned into PUSH.

    In dependent scenes the crosser's goal edge requires the pusher's AT edge and the crosser's
    predecessor is the pusher (BEFORE relation); otherwise there is no dependency.
    """
    from utils.edge_scenario_spec import compose_canonical_graph
    from utils.stage1_push import PUSH
    k, m = templates.shape
    rows = torch.arange(k, device=templates.device)
    pusher = 1 - crosser
    graph = compose_canonical_graph(templates, num_objects=num_objects)
    graph.edge_relation[graph.edge_relation == 6] = PUSH
    dependent = scene == SCENES.index('dependent')
    follower = ((graph.edge_owner == crosser[:, None]) & graph.required_goal).long().argmax(-1)
    leader = ((graph.edge_owner == pusher[:, None]) & (graph.edge_relation == 7) & graph.edge_valid).long().argmax(-1)
    graph.prereq_mask[rows[dependent], follower[dependent], leader[dependent]] = True
    predecessor = torch.full((k, m), -1, device=templates.device, dtype=torch.long)
    predecessor[rows[dependent], crosser[dependent]] = pusher[dependent]
    return graph, predecessor


def token_mapping(primary, extra, bank_size):
    # Logical slots: each human's own task object first, then one extra object and one ordinary
    # distractor from the bank in random order. Task objects outside the bank are never distractors.
    n = len(primary)
    used = torch.zeros(n, bank_size + 1, device=primary.device, dtype=torch.bool)
    used.scatter_(1, primary.clamp_max(bank_size), True)
    score = torch.rand(n, bank_size, device=primary.device).masked_fill(used[:, :bank_size], 2.)
    distractor = score.argmin(-1, keepdim=True)
    extra = torch.as_tensor(extra, device=primary.device).expand(n)[:, None]
    tail = torch.cat((extra, distractor), -1)
    swap = torch.rand(n, 1, device=primary.device) < .5
    return torch.cat((primary, torch.where(swap, tail.flip(-1), tail)), -1)


def predecessor_relations(predecessor, num_agents):
    ids = torch.arange(num_agents, device=predecessor.device)
    before = predecessor.long()[..., None] == ids
    eye = torch.eye(num_agents, device=predecessor.device, dtype=torch.bool)
    return torch.where(eye, 1, torch.where(before, 2, 0))


def checkpoint_contract(env):
    contract = dict(version=1, bridge=env['bridge'], before=env['before'], template_rsi=env['templateRsi'],
                    sizes=env['box']['build']['taskSizeRanges'], reset=env['box']['reset'])
    if env['bridge']['mode'] in ('push', 'push4'):
        contract['rsi_bank_sha256'] = hashlib.sha256(Path(env['bridge']['push']['rsi']['bank']).read_bytes()).hexdigest()
    return contract
