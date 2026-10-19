"""Stage-1 pushing/open-and-hold contracts, independent of Isaac Gym."""
import math
import torch

VARIANT = 'push_door_stage1_v1'
FAMILIES = ('push', 'door_open', 'door_hold')


def resize_push_box(env, size):
    """Resize the PUSH cube while preserving mass and safe reset clearance."""
    size = float(size)
    if not math.isfinite(size) or size <= 0:
        raise ValueError('PUSH box size must be finite and positive')
    box = env['interaction']['push']['box']
    mass = float(box['density']) * math.prod(box['size'])
    box.pop('size_range',None)
    box['size'] = [size, size, size]
    box['density'] = mass / size ** 3
    randomization = env.get('startRandomization')
    if randomization is not None:
        low, high = randomization['push_distance']
        safe_low = max(low, math.sqrt(2) * size / 2 + .25)
        randomization['push_distance'] = [safe_low, safe_low + high - low]


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


def push_surface_distance(roots, boxes, sizes):
    """XY root distance to the oriented box footprint, independent of box size."""
    delta=roots[...,:3]-boxes[...,:3]
    q=-boxes[...,3:6];w=boxes[...,6:7]
    local=delta+2*torch.cross(q,torch.cross(q,delta,dim=-1)+w*delta,dim=-1)
    return (local[...,:2].abs()-sizes[...,:2]/2).clamp_min(0).norm(dim=-1)


def nearest_door_hand_on_entry(previous_hand, previous_open, opening, tasks, hands, handles, contact):
    """Choose once on approach entry; preserve contact and RSI hand bindings."""
    distance=(hands[..., :, None, :]-handles[..., None, :, :]).norm(dim=-1).amin(-1)
    nearest=distance.argmin(-1)
    entering=(tasks==1) & opening & ~previous_open & ~contact
    return torch.where(entering, nearest, previous_hand)


def door_amp_family(tasks, hand, holding, opening, count, pushing=None):
    if count>=5:
        family=1+hand+2*holding.long()
        if count==6:family=torch.where(opening | holding,family,5)
        push_family=torch.where(pushing,0,5) if count==6 and pushing is not None else 0
        return torch.where(tasks==0,push_family,family)
    return torch.where(tasks==0,0,torch.where(holding,2,1))


def push_hand_alignment(local_hands, local_direction, half_size, config):
    """Both hands close to the trailing side face and below the box top."""
    axis=local_direction[...,:2].abs().argmax(-1)
    sign=torch.gather(local_direction[...,:2],-1,axis[...,None]).squeeze(-1).sign()
    sign=torch.where(sign==0,torch.ones_like(sign),sign)
    side=torch.nn.functional.one_hot(axis,3).to(local_hands.dtype)
    face=-sign[...,None]*side*(half_size+config['hand_radius'])
    bounds=half_size.unsqueeze(-2)
    desired=torch.maximum(torch.minimum(local_hands,bounds),-bounds)
    desired=torch.where(side[...,None,:].bool(),face[...,None,:],desired)
    low,high=config['hand_height_fraction']
    desired[...,2]=torch.maximum(torch.minimum(local_hands[...,2],(half_size[...,2]*(2*high-1)).unsqueeze(-1)),(half_size[...,2]*(2*low-1)).unsqueeze(-1))
    error=(local_hands-desired).norm(dim=-1)
    return torch.exp(-(error/config['hand_distance_scale']).square()).min(-1).values,error.mean(-1)


def stage2_push_hand_quality(local_hands, local_direction, half_size, config):
    """Stage2 contact-at rear-face XY quality, aligned to the actual goal face."""
    axis=local_direction[...,:2].abs().argmax(-1)
    sign=local_direction[...,:2].gather(-1,axis[...,None]).squeeze(-1).sign()
    sign=torch.where(sign==0,torch.ones_like(sign),sign)
    side=torch.nn.functional.one_hot(axis,2).to(local_hands.dtype)
    half=half_size[...,:2]
    point=torch.maximum(torch.minimum(local_hands[...,:2],half[...,None,:]),-half[...,None,:])
    face=-sign[...,None]*side*half
    point=torch.where(side[...,None,:].bool(),face[...,None,:],point)
    error=(local_hands[...,:2]-point).norm(dim=-1).amax(-1)
    return torch.exp(-config['hand_near_scale']*error.square())


def stage2_contact_at_rewards(quality, contact, root_distance, goal_distance, at_success, config):
    """Dense, recurring contact+AT terms from stage2 single-agent PUSH.

    No record-improvement or contact gate on approach/AT progress. The caller
    retains physical goal validity and independent tilt/lift penalties.
    """
    saturated=contact | at_success
    approach=1/(1+(root_distance-config['approach_delta']).clamp_min(0)/config['approach_sigma'])
    at_progress=1/(1+(goal_distance-config['approach_delta']).clamp_min(0)/config['approach_sigma'])
    at_phi=torch.exp(-config['at_near_scale']*goal_distance.square())
    one=torch.ones_like(quality)
    return torch.stack([
        config['hand_state_weight']*torch.where(saturated,one,quality),
        config['hand_progress_weight']*torch.where(saturated,one,approach),
        config['hand_result_weight']*saturated.float(),
        config['at_state_weight']*torch.where(at_success,one,at_phi),
        config['at_progress_weight']*torch.where(at_success,one,at_progress),
        config['at_result_weight']*at_success.float()],-1)


def push_stability_penalties(up_z, bottom, config):
    tilt=torch.acos(up_z.clamp(-1,1))
    free=math.radians(config['tilt_free_degrees'])
    scale=math.radians(config['max_tilt_degrees'])-free
    tilt_loss=((tilt-free).clamp_min(0)/scale).clamp_max(1)*config['tilt_penalty_weight']
    lift_loss=((bottom-config.get('lift_free_height',0.)).clamp_min(0)/config['lift_penalty_scale']).clamp_max(1)*config['lift_penalty_weight']
    return -tilt_loss,-lift_loss


def push_relation_rewards(quality, contact, distance, stable, at_progress, at_result,
                          best_quality, elapsed, achieved, dt, config):
    """Independent PUSH-hand and AT state/progress/result components.

    Progress rewards only a new record; results are once per episode. AT motion
    records are consumed outside this function even without hand contact.
    """
    hand_progress,new_best=progress_reward(quality,best_quality,dt*config['hand_progress_speed'])
    valid=contact & stable
    elapsed,achieved,first=advance_success(valid,elapsed,achieved,dt,config['hand_hold_seconds'])
    hand_state=quality*stable
    at_state=torch.exp(-(distance/config['at_distance_scale']).square())*stable
    components=torch.stack([
        config['hand_state_weight']*hand_state,
        config['hand_progress_weight']*hand_progress*stable,
        config['hand_result_weight']*first,
        config['at_state_weight']*at_state,
        config['at_progress_weight']*at_progress*valid,
        config['at_result_weight']*at_result],-1)
    return components,new_best,elapsed,achieved


def door_contact_open_rewards(hands, handles, hand_forces, handle_forces, angle, angular_velocity,
                              contact_elapsed, open_elapsed, dt, config, body_position=None, door_position=None):
    """Two independent SPR groups; current satisfaction, not record deltas.

    A single SAME hand/handle pair must satisfy distance and both force tests.
    Reference: HOLDING+AT dense progress, recurring success and saturation.
    """
    relation=config['relation_rewards']
    if config.get('working_hand', 'any') == 'right':
        hands=hands[..., :1, :]
        hand_forces=hand_forces[..., :1]
    distances=(hands[..., :, None, :]-handles[..., None, :, :]).norm(dim=-1)
    hand_distance=distances.amin(dim=(-1,-2))
    paired=(distances<config['contact_distance']) & (hand_forces[..., :, None]>config['contact_force']) & (handle_forces[..., None, :]>config['contact_force'])
    contact=paired.any(-1).any(-1)
    paired_phi=(torch.exp(-relation['hand_distance_scale']*distances.square())*paired.float()).amax(dim=(-1,-2))
    score_success='contact_success_threshold' in relation
    if score_success:
        contact_success=contact & (paired_phi>=relation['contact_success_threshold'])
        contact_elapsed=torch.zeros_like(contact_elapsed)  # No contact-duration requirement.
    else:
        contact_elapsed=torch.where(contact,contact_elapsed+dt,torch.zeros_like(contact_elapsed))
        contact_success=contact_elapsed>=relation['contact_hold_seconds']
    if relation.get('progress_reference', 'hand_handle') == 'body_door_xy':
        if body_position is None or door_position is None:
            raise ValueError('Body/door positions required for body_door_xy progress')
        progress_distance=(body_position[..., :2]-door_position[..., :2]).norm(dim=-1)
    else:
        progress_distance=hand_distance
    approach=1/(1+progress_distance/relation['approach_sigma'])
    target=angle.new_tensor(math.radians(config['open_degrees']))
    angle_progress=(angle/target).clamp(0,1)*contact.float()
    open_valid=contact & (angle>=target) & (angular_velocity.abs()<config['hold_max_speed'])
    open_elapsed=torch.where(open_valid,open_elapsed+dt,torch.zeros_like(open_elapsed))
    open_success=open_elapsed>=config['hold_seconds']
    # Like HOLDING+AT, saturation is current success, not a permanent episode latch.
    contact_satisfied=contact_success if score_success else (contact_success | open_success)
    one=torch.ones_like(hand_distance)
    components=torch.stack([
        relation['contact_state_weight']*torch.where(contact_satisfied,one,paired_phi),
        relation['contact_progress_weight']*torch.where(contact_satisfied,one,approach),
        relation['contact_success_weight']*contact_satisfied.float(),
        relation['open_state_weight']*(torch.exp(-relation['hand_distance_scale']*hand_distance.square())
            if relation.get('open_state_mode', 'contact')=='distance' else contact.float()),
        relation['open_progress_weight']*torch.where(open_success,one,angle_progress),
        relation['open_success_weight']*open_success.float()],-1)
    return dict(components=components,contact=contact,contact_elapsed=contact_elapsed,
                contact_success=contact_success,open_elapsed=open_elapsed,open_success=open_success,
                hand_distance=hand_distance,paired_phi=paired_phi)


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


def task_relation_matrix(tasks, holding=None, semantic=False, unified_door=False, generic_door_goal=False, hand_open=False):
    """Canonical H[M], boxes[M], doors[M], goals[M]; bind actual active object."""
    from utils.edge_context_spec import AT
    from utils.edge_interaction_spec import PUSH, HAND, OPEN
    legacy_open,legacy_hold=12,13  # Only used by historical relation schemas.
    if holding is None:
        holding=torch.zeros_like(tasks,dtype=torch.bool)
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
        interaction=torch.where(tasks[:,owner].bool(),torch.where(holding[:,owner] & (not unified_door),legacy_hold,legacy_open),PUSH) if semantic else 3
        if semantic and hand_open:
            interaction=torch.where(tasks[:,owner].bool(),HAND,PUSH)
        matrix[rows, owner, obj] = interaction
        matrix[rows, obj, owner] = interaction
        matrix[:, owner, goal] = matrix[:, goal, owner] = 4
        placement=torch.where(tasks[:,owner].bool(),5 if generic_door_goal else interaction,AT) if semantic else 5
        if semantic and hand_open:
            placement=torch.where(tasks[:,owner].bool(),OPEN,AT)
        matrix[rows, obj, goal] = matrix[rows, goal, obj] = placement
    return matrix


def amp_motion_sample(obs, family, family_count, conditioning=True):
    """Keep the expert mixture unchanged; optionally omit discriminator labels."""
    if not conditioning:
        return obs
    labels = torch.nn.functional.one_hot(family.long(), family_count).to(obs.dtype)
    return torch.cat([obs, labels], dim=-1)


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
    if env.get('relationGraph',{}).get('semantic_relations',False):
        metadata['relation_graph']=env['relationGraph']
        if env['relationGraph'].get('door_hand_open_relations',False):
            metadata['relation_taxonomy']={'AT':7,'PUSH':11,'HAND':14,'OPEN':15}
        else:
            metadata['relation_taxonomy']={'AT':7,'PUSH':11,'DOOR_OPEN':12,'DOOR_HOLD':13}
    if 'startRandomization' in env:
        metadata['start_randomization'] = env['startRandomization']
    if env.get('agentCollisionPenalty', False):
        metadata['agent_collision_penalty'] = {
            'enabled': True, 'coefficient': env.get('agentCollisionCoeff', .5),
            'distance': env.get('agentCollisionDist', .7)}
    return metadata


def check_interaction_checkpoint(weights, expected):
    actual=weights.get('interaction_metadata')
    def active_metadata(value):
        if value is not None and value.get('relation_graph',{}).get('door_hand_open_relations',False):
            value=dict(value)
            value['relation_taxonomy']={k:v for k,v in value.get('relation_taxonomy',{}).items() if k not in ('DOOR_OPEN','DOOR_HOLD')}
        return value
    if active_metadata(actual) != active_metadata(expected):
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
    mix=amp.get('task_expert_mix')
    if mix is not None:
        if amp.get('task_conditioning',True) is not True or set(mix)!= {'push','door'}:
            raise ValueError('Task expert mix requires conditional PUSH/OPEN labels')
        for name,forbidden in (('push',2),('door',1)):
            weights=mix[name]
            if len(weights)!=3 or any(not math.isfinite(v) or v<0 for v in weights) or abs(sum(weights)-1)>1e-6 or weights[forbidden]!=0:
                raise ValueError('Invalid task expert mix: '+name)
    if not isinstance(amp.get('task_conditioning', True), bool):
        raise ValueError('AMP task_conditioning must be boolean')
    bounds=push['box'].get('size_range')
    if bounds is not None:
        lo,hi=bounds
        if not all(math.isfinite(x) for x in bounds) or not 0<lo<=hi:
            raise ValueError('Invalid PUSH box size range')

    if door.get('reward_version') not in ('handle_gated_v2','contact_open_spr_v1'):
        raise ValueError('Unsupported Door reward version')
    if door['reward_version']=='contact_open_spr_v1':
        relation=door.get('relation_rewards',{})
        for key in ('contact_state_weight','contact_progress_weight','contact_success_weight','open_state_weight','open_progress_weight','open_success_weight'):
            value=relation.get(key,-1)
            if not math.isfinite(value) or value<0:raise ValueError('Invalid Door relation weight: '+key)
        if 'contact_success_threshold' in relation:
            threshold=relation['contact_success_threshold']
            if not math.isfinite(threshold) or not 0<threshold<=1:
                raise ValueError('Invalid Door contact success threshold')
        if relation.get('open_state_mode', 'contact') not in ('contact', 'distance'):
            raise ValueError('Invalid Door open state mode')
        if relation.get('progress_reference', 'hand_handle') not in ('hand_handle', 'body_door_xy'):
            raise ValueError('Invalid Door progress reference')
        parameters=('hand_distance_scale','approach_sigma') if 'contact_success_threshold' in relation else ('hand_distance_scale','approach_sigma','contact_hold_seconds')
        for key in parameters:
            value=relation.get(key,0)
            if not math.isfinite(value) or value<=0:raise ValueError('Invalid Door relation parameter: '+key)
    for key in ('approach_weight','closing_weight','approach_speed','closing_speed','approach_angle_tolerance','closing_deadband'):
        if not math.isfinite(door[key]) or door[key]<=0:
            raise ValueError('Invalid door reward parameter: '+key)
    if door.get('working_hand', 'any') not in ('any', 'right'):
        raise ValueError('Door working_hand must be any or right')
    if door.get('working_hand', 'any')=='right' and door.get('motion_direction')!='left_open':
        raise ValueError('Fixed right hand requires left_open labeled motions')
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
    if 'max_tilt_degrees' in push:
        if not 0 <= push['tilt_free_degrees'] < push['max_tilt_degrees'] < 90:
            raise ValueError('Invalid PUSH tilt penalty bounds')
        for key in ('tilt_penalty_weight','lift_penalty_weight','lift_penalty_scale'):
            if not math.isfinite(push[key]) or push[key]<=0:
                raise ValueError('Invalid PUSH stability parameter: '+key)
    split=push.get('relation_rewards')
    if split is not None:
        if split.get('version') not in ('push_at_state_progress_result_v1','stage2_contact_at_dense_v1'):
            raise ValueError('Unsupported PUSH relation reward version')
        for key in ('hand_state_weight','hand_progress_weight','hand_result_weight','at_state_weight','at_progress_weight','at_result_weight'):
            if not math.isfinite(split[key]) or split[key]<0:
                raise ValueError('Invalid PUSH relation reward weight: '+key)
        parameters=('hand_contact_distance','contact_force','hand_near_scale','at_near_scale','approach_sigma') if split['version']=='stage2_contact_at_dense_v1' else ('hand_progress_speed','hand_hold_seconds','hand_contact_distance','contact_force','at_distance_scale')
        if split['version']=='stage2_contact_at_dense_v1' and (not math.isfinite(split['approach_delta']) or split['approach_delta']<0):
            raise ValueError('Invalid PUSH approach_delta')
        for key in parameters:
            if not math.isfinite(split[key]) or split[key]<=0:
                raise ValueError('Invalid PUSH relation reward parameter: '+key)
    if 'hand_reward_weight' in push or split is not None:
        for key in (('hand_reward_weight',) if 'hand_reward_weight' in push else ()) + ('hand_radius','hand_distance_scale'):
            if not math.isfinite(push[key]) or push[key]<=0:
                raise ValueError('Invalid PUSH hand parameter: '+key)
        lo,hi=push['hand_height_fraction']
        if not 0 < lo < hi < 1:raise ValueError('Invalid PUSH hand height band')
    if 'handle_height' in door and (not math.isfinite(door['handle_height']) or not 0 < door['handle_height'] < 2.1):
        raise ValueError('Invalid DOOR handle height')
    rsi=config.get('task_rsi')
    if rsi is not None:
        offsets=rsi.get('door_alignment_yaw_degrees',[0.])
        if not offsets or any(not math.isfinite(v) or abs(v)>90 for v in offsets):
            raise ValueError('Invalid Door RSI alignment yaw offsets')
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
        if 'push_goal_yaw_degrees' in rsi:
            bounds=rsi['push_goal_yaw_degrees']
            if len(bounds)!=2 or not all(math.isfinite(v) for v in bounds) or not -90 < bounds[0] <= bounds[1] < 90:
                raise ValueError('Invalid RSI push goal yaw range')
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
    if approach is not None and 'enter_push_surface_distance' in approach:
        enter,leave=approach['enter_push_surface_distance'],approach['return_push_surface_distance']
        if not all(math.isfinite(v) for v in (enter,leave)) or not 0 < enter < leave:
            raise ValueError('Invalid PUSH AMP distance hysteresis')
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


def rotate_push_layout(boxes, goals, humans, rotations, tasks, angles):
    """Rotate each PUSH scene about its box; leave DOOR scenes untouched."""
    push=tasks==0
    boxes,goals,humans,rotations=(value.clone() for value in (boxes,goals,humans,rotations))
    c,s=torch.cos(angles),torch.sin(angles)
    for value in (goals,humans):
        delta=value[...,:2]-boxes[...,:2]
        turned=torch.stack([c*delta[...,0]-s*delta[...,1],s*delta[...,0]+c*delta[...,1]],-1)
        value[...,:2]=torch.where(push[...,None],boxes[...,:2]+turned,value[...,:2])
    c,s=torch.cos(angles/2),torch.sin(angles/2)
    for q in (boxes[...,3:7],rotations):
        x,y,z,w=q.unbind(-1)
        turned=torch.stack([c*x-s*y,s*x+c*y,c*z+s*w,c*w-s*z],-1)
        q.copy_(torch.where(push[...,None],turned,q))
    return boxes,goals,humans,rotations


def push_box_start_x(box_size, maximum_target_distance, forward_jitter=0.):
    """Leave a 20cm gap to unused door geometry even at the farthest goal."""
    sizes=torch.as_tensor(box_size)
    radius=torch.linalg.vector_norm(sizes[..., :2]/2,dim=-1)
    return -(maximum_target_distance + radius + forward_jitter + .2)


def sample_start_layout(tasks, config, box_size, lanes):
    """Collision-separated lane-local reset layout; doors stay closed and upright."""
    for name, bounds in config.items():
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Invalid start randomization bounds: ' + name)
    box_size=torch.as_tensor(box_size,device=tasks.device,dtype=torch.float32)
    radius=torch.linalg.vector_norm(box_size[..., :2]/2,dim=-1)
    if torch.any(config['push_distance'][0] < radius + .25) or config['door_distance'][0] < .5:
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
    boxes[..., 2] = box_size[..., 2] / 2 + .005
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


def door_motion_matches(path, direction, working_hand="any"):
    """BONES left-opening references: left-side original or right-side mirror."""
    if working_hand not in ('any', 'right'):
        raise ValueError('Door working_hand must be any or right')
    if direction == 'all':
        if working_hand != 'any':
            raise ValueError('Fixed hand requires left_open labeled motions')
        return True
    if direction != 'left_open':
        raise ValueError('door motion_direction must be all or left_open')
    clip = str(path).replace('\\','/').split('/')[-3]
    mirrored = clip.endswith('_M')
    direction_matches = ('inside_door_handle_left_side_open_' in clip and not mirrored) or ('inside_door_handle_right_side_open_' in clip and mirrored)
    return direction_matches and (working_hand=='any' or door_motion_hand(path)==0)
