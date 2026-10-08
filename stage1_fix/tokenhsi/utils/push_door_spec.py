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


def door_open_amp_phase(previous, distance, contact, holding, enter, leave):
    """Distance hysteresis; actual contact/holding always keeps DOOR AMP active."""
    return torch.where(previous, distance<=leave, distance<=enter) | contact | holding


def door_amp_family(tasks, hand, holding, opening, count):
    if count>=5:
        family=1+hand+2*holding.long()
        if count==6:family=torch.where(opening | holding,family,5)
        return torch.where(tasks==0,0,family)
    return torch.where(tasks==0,0,torch.where(holding,2,1))


def door_shaping(angle, best_angle, previous_angle, hand_distance, best_distance, contact, dt, config):
    """Contact-gated opening, non-repeatable approach, and actual closing penalty."""
    target=math.radians(config['open_degrees'])
    opening,new_best=progress_reward(angle.clamp(0,target),best_angle,dt*config['progress_speed'])
    # Always consume the record, even without contact: touching later cannot
    # collect credit for an earlier body push or a close/reopen cycle.
    opening=opening*contact.float()
    approach,new_distance=progress_reward(-hand_distance,-best_distance,dt*config['approach_speed'])
    # Door movement toward a stationary hand must not create approach credit.
    stationary=(angle-previous_angle).abs()<=config['approach_angle_tolerance']
    approach=approach*stationary.float()*(~contact).float()
    closing=((previous_angle-angle-config['closing_deadband']).clamp_min(0)/(dt*config['closing_speed'])).clamp_max(1)
    return opening,new_best,approach,-new_distance,closing


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
    if door.get('reward_version') != 'handle_gated_v2':
        raise ValueError('Door reward requires handle_gated_v2 configuration')
    for key in ('approach_weight','closing_weight','approach_speed','closing_speed','approach_angle_tolerance','closing_deadband'):
        if not math.isfinite(door[key]) or door[key]<=0:
            raise ValueError('Invalid door reward parameter: '+key)
    if door.get('motion_direction','all') not in ('all','left_open'):
        raise ValueError('Invalid door motion direction')
    if not 0 < door['reopen_degrees'] < door['open_degrees'] < 110:
        raise ValueError('Door phase requires reopen < open < joint limit')
    for data, keys in ((push, ('goal_tolerance','settle_speed','settle_seconds','progress_speed','ground_tolerance')),
                       (door, ('hold_seconds','hold_max_speed','contact_distance','contact_force','progress_speed'))):
        for key in keys:
            if not math.isfinite(data[key]) or data[key] <= 0:
                raise ValueError('Invalid task parameter: ' + key)
    if push.get('direction', 'toward_door') not in ('toward_door', 'away_from_door'):
        raise ValueError('Invalid PUSH direction')
    if 'handle_height' in door and (not math.isfinite(door['handle_height']) or not 0 < door['handle_height'] < 2.1):
        raise ValueError('Invalid DOOR handle height')
    rsi=config.get('task_rsi')
    if rsi is not None:
        if 'door_rsi_probability' in rsi and (not math.isfinite(rsi['door_rsi_probability']) or not 0 <= rsi['door_rsi_probability'] <= 1):
            raise ValueError('Invalid DOOR RSI probability')
        for name in ('probability','late_fraction','door_early_fraction'):
            if not math.isfinite(rsi[name]) or not 0 <= rsi[name] <= 1:
                raise ValueError('Invalid task RSI probability: '+name)
        for name in ('push_phase','door_phase'):
            lo,hi=rsi[name]
            if not 0 <= lo < hi <= 1:raise ValueError('Invalid task RSI phase: '+name)
        lo,hi=rsi['door_angle_degrees']
        if not 0 <= lo < rsi['door_early_max_degrees'] < hi < door['open_degrees']:
            raise ValueError('Invalid task RSI door angle range')
        for name in ('frame_clearance','hand_gap','min_extension','door_height_tolerance','push_hand_radius'):
            if not math.isfinite(rsi[name]) or rsi[name]<=0:
                raise ValueError('Invalid task RSI clearance: '+name)
        if not math.isfinite(rsi['push_surface_gap']) or rsi['push_surface_gap']<0:
            raise ValueError('Invalid RSI push surface gap')
        if 'door_facing_degrees' in rsi:
            if not math.isfinite(rsi['door_facing_degrees']) or not 0 < rsi['door_facing_degrees'] < 90:
                raise ValueError('DOOR RSI facing limit must be below 90 degrees')
            if rsi.get('door_hand_binding') not in ('motion_extension_v1','user_paths_v1'):
                raise ValueError('Facing-filtered RSI requires motion hand binding')
        lo,hi=rsi['push_remaining']
        if not 0 < lo <= hi:raise ValueError('Invalid task RSI push distance')
        lo,hi=rsi['push_hand_height']
        if not 0 < lo < hi <= push['box']['size'][2]:raise ValueError('Invalid task RSI hand height')
    approach=amp.get('approach_loco')
    if approach is not None:
        if amp.get('hand_conditioning')!='user_hands_v1':
            raise ValueError('Approach loco requires hand-conditioned AMP')
        enter,leave=approach['enter_door_distance'],approach['return_loco_distance']
        if not math.isfinite(enter) or not math.isfinite(leave) or not 0 < enter < leave:
            raise ValueError('Invalid DOOR AMP distance hysteresis')
    if amp['hold_source'] not in ('door_tail', 'loco'):
        raise ValueError('AMP hold_source must be door_tail or loco')
    for name in ('push_phase','open_phase','hold_phase'):
        lo, hi = amp[name]
        if not 0 <= lo < hi <= 1:
            raise ValueError('Invalid AMP phase: ' + name)
    if amp['open_phase'][1] > amp['hold_phase'][0]:
        raise ValueError('Door open and holding reference phases must not overlap')


def redirect_push_away(boxes, goals, humans, rotations, tasks):
    """Half-turn PUSH about its box; preserve DOOR and all relative distances."""
    push = tasks == 0
    boxes, goals, humans, rotations = (x.clone() for x in (boxes, goals, humans, rotations))
    for value in (goals, humans):
        value[..., :2] = torch.where(push[..., None],
            2 * boxes[..., :2] - value[..., :2], value[..., :2])
    # Quaternion order xyzw: left-multiply a world-Z half-turn.
    for q in (boxes[..., 3:7], rotations):
        turned = torch.stack([-q[..., 1], q[..., 0], q[..., 3], -q[..., 2]], -1)
        q.copy_(torch.where(push[..., None], turned, q))
    return boxes, goals, humans, rotations


def push_box_start_x(box_size, maximum_target_distance, forward_jitter=0.):
    """Leave a 20cm gap to unused door geometry even at the farthest goal."""
    radius = float(torch.linalg.vector_norm(torch.as_tensor(box_size[:2]) / 2))
    return -(maximum_target_distance + radius + forward_jitter + .2)


def sample_start_layout(tasks, config, box_size, lanes):
    """Collision-separated lane-local reset layout; doors stay closed and upright."""
    for name, bounds in config.items():
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Invalid start randomization bounds: ' + name)
    radius = float(torch.linalg.vector_norm(torch.as_tensor(box_size[:2]) / 2))
    if config['push_distance'][0] < radius + .25 or config['door_distance'][0] < .5:
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
    start_x = push_box_start_x(box_size, config['target_distance'][1], config['box_x'][1])
    boxes[..., 0] += torch.where(tasks.bool(), -2.2, start_x) + uniform(config['box_x'])
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


def door_motion_hand(path):
    """User-confirmed working hand: 0=right, 1=left (keyBodies order)."""
    clip=str(path).replace('\\','/').split('/')[-3]
    if 'inside_door_handle_left_side_open_' in clip and not clip.endswith('_M'):
        return 0
    if 'inside_door_handle_right_side_open_' in clip and clip.endswith('_M'):
        return 1
    raise ValueError('No user-confirmed working-hand label for DOOR clip: '+clip)


def door_motion_matches(path, direction):
    """BONES left-opening references: left-side original or right-side mirror."""
    if direction == 'all':
        return True
    if direction != 'left_open':
        raise ValueError('door motion_direction must be all or left_open')
    clip = str(path).replace('\\','/').split('/')[-3]
    mirrored = clip.endswith('_M')
    return ('inside_door_handle_left_side_open_' in clip and not mirrored) or ('inside_door_handle_right_side_open_' in clip and mirrored)
