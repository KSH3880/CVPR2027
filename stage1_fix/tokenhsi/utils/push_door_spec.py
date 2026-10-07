"""Stage-1 pushing/open-and-hold contracts, independent of Isaac Gym."""
import math
import torch

VARIANT = 'push_door_stage1_v1'
FAMILIES = ('push', 'door_open', 'door_hold')


def phase_update(holding, angle, enter, leave):
    """Hysteresis prevents AMP family flicker at the opening threshold."""
    return torch.where(holding, angle >= leave, angle >= enter)


def progress_reward(value, best, scale):
    """Reward only new progress, so opening/closing cycles cannot farm rewards."""
    improved = (value - best).clamp_min(0)
    return (improved / scale).clamp_max(1), torch.maximum(best, value)


def advance_success(valid, elapsed, done, dt, duration):
    elapsed = torch.where(valid, elapsed + dt, torch.zeros_like(elapsed))
    current = elapsed >= duration
    first = current & ~done
    return elapsed, done | current, first


def task_relation_matrix(tasks):
    """Canonical H[M], boxes[M], doors[M], goals[M]; bind actual active object."""
    B, M = tasks.shape
    L = 4 * M
    matrix = torch.zeros(B, L, L, dtype=torch.long, device=tasks.device)
    idx = torch.arange(L, device=tasks.device)
    matrix[:, idx, idx] = 1
    for owner in range(M):
        for other in range(M):
            if owner != other:
                matrix[:, owner, other] = 2
        obj = M + owner + tasks[:, owner].long() * M
        goal = 3 * M + owner
        rows = torch.arange(B, device=tasks.device)
        matrix[rows, owner, obj] = 3
        matrix[rows, obj, owner] = 3
        matrix[:, owner, goal] = matrix[:, goal, owner] = 4
        matrix[rows, obj, goal] = matrix[rows, goal, obj] = 5
    return matrix


def expert_time(lengths, history, phase_range, uniform=None):
    """Keep the entire expert AMP window inside the selected clip phase."""
    lo, hi = phase_range
    if not 0 <= lo < hi <= 1:
        raise ValueError('Invalid expert phase range')
    lower = lengths * lo + history
    upper = lengths * hi
    if torch.any(upper < lower):
        raise ValueError('Expert phase is shorter than AMP history')
    uniform = torch.rand_like(lengths) if uniform is None else uniform
    return lower + uniform * (upper - lower)


def interaction_metadata(env):
    """Checkpoint must describe physics, reward, AMP windows and task distribution."""
    metadata = {'variant': VARIANT, 'coordinate_frame': 'gym_env_local_v1', 'num_agents': env['numAgents'],
            'num_objects': env['numObjects'], 'amp_steps': env['numAMPObsSteps'],
            'interaction': env['interaction'], 'motion_file': env['motion_file'],
            'entity_sizes': [223, 34, 9], 'pose_size': 7}
    if 'startRandomization' in env:
        metadata['start_randomization'] = env['startRandomization']
    if env.get('agentCollisionPenalty', False):
        metadata['agent_collision_penalty'] = {
            'enabled': True, 'coefficient': env.get('agentCollisionCoeff', .5),
            'distance': env.get('agentCollisionDist', .7)}
    return metadata


def check_interaction_checkpoint(weights, expected):
    if weights.get('interaction_metadata') != expected:
        raise ValueError('Push/door checkpoint contract mismatch; use its dedicated checkpoint/config')


def hand_handle_contact(hands, handles, hand_forces, handle_forces, distance, force):
    """Conservative pair proxy: proximity AND contact force on both involved bodies."""
    separation = torch.linalg.vector_norm(hands[..., :, None, :] - handles[..., None, :, :], dim=-1)
    return ((separation < distance) & (hand_forces[..., :, None] > force)
            & (handle_forces[..., None, :] > force)).any(-1).any(-1)


def validate_interaction(config):
    if not 0 <= config['door_probability'] <= 1 or config['lane_spacing'] < 2.5:
        raise ValueError('Invalid task distribution or scene lane clearance')
    for key in ('progress_weight', 'maintain_weight', 'success_weight'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('Invalid reward weight: ' + key)
    push, door, amp = config['push'], config['door'], config['amp']
    if not 0 < door['reopen_degrees'] < door['open_degrees'] < 110:
        raise ValueError('Door phase requires reopen < open < joint limit')
    for data, keys in ((push, ('goal_tolerance','settle_speed','settle_seconds','progress_speed','ground_tolerance')),
                       (door, ('hold_seconds','hold_max_speed','contact_distance','contact_force','progress_speed'))):
        for key in keys:
            if not math.isfinite(data[key]) or data[key] <= 0:
                raise ValueError('Invalid task parameter: ' + key)
    if amp['hold_source'] not in ('door_tail', 'loco'):
        raise ValueError('AMP hold_source must be door_tail or loco')
    for name in ('push_phase','open_phase','hold_phase'):
        lo, hi = amp[name]
        if not 0 <= lo < hi <= 1:
            raise ValueError('Invalid AMP phase: ' + name)
    if amp['open_phase'][1] > amp['hold_phase'][0]:
        raise ValueError('Door open and holding reference phases must not overlap')


def sample_start_layout(tasks, config, box_size, lanes):
    """Collision-separated lane-local reset layout; doors stay closed and upright."""
    for name, bounds in config.items():
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Invalid start randomization bounds: ' + name)
    if config['push_distance'][0] < .6 or config['door_distance'][0] < .5:
        raise ValueError('Start distance must preserve human/object clearance')
    shape = tasks.shape
    device = tasks.device
    def uniform(bounds):
        lo, hi = bounds
        return lo + torch.rand(*shape, device=device) * (hi - lo)
    shift = torch.stack([uniform(config['scene_x']), uniform(config['scene_y'])], -1)
    shift[..., 1] += lanes
    boxes = torch.zeros(*shape, 3, device=device)
    boxes[..., :2] = shift
    boxes[..., 0] += torch.where(tasks.bool(), -2.2, -1.6) + uniform(config['box_x'])
    boxes[..., 1] += tasks * .85 + uniform(config['box_y'])
    boxes[..., 2] = box_size[2] / 2 + .005
    target = boxes.clone()
    direction = torch.deg2rad(uniform(config['target_yaw_degrees']))
    distance = uniform(config['target_distance'])
    target[..., 0] += distance * torch.cos(direction)
    target[..., 1] += distance * torch.sin(direction)
    human = boxes.clone()
    human[..., 0] -= uniform(config['push_distance'])
    human[..., 1] += uniform(config['human_lateral'])
    human[..., 0] = torch.where(tasks.bool(), shift[..., 0] - uniform(config['door_distance']), human[..., 0])
    human[..., 1] = torch.where(tasks.bool(), shift[..., 1] - .36 + uniform(config['human_lateral']), human[..., 1])
    facing = torch.where(tasks[..., None].bool(), torch.stack([shift[..., 0] - .0225, shift[..., 1] - .36], -1), boxes[..., :2])
    delta = facing - human[..., :2]
    yaw = torch.atan2(delta[..., 1], delta[..., 0]) + torch.deg2rad(uniform(config['human_yaw_degrees']))
    return boxes, target, human, yaw, torch.deg2rad(uniform(config['box_yaw_degrees'])), shift
