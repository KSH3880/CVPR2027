"""Grounded box pushing and existing continuous door support in one runtime."""
import torch
from utils.edge_ontop_spec import expand_graph, copy_graph_rows
from utils.edge_stage1_spec import semantic_graph_packet
from utils.open_push_spec import compile_graph, graph_for_tasks
from utils.door_hold import update_hold
from env.tasks.multi_agent.edge_context_reward import EdgeContextRuntime
from env.tasks.multi_agent.edge_door_reward import door_reward
from utils.edge_door_spec import compile_graph as door_graph



def midpoint_xy_control(spec):
    return spec.get('control_mode') == 'midpoint_xy_net_credit'


def push_local_xy(hands, boxes):
    """Ignore box pitch/roll and hand Z; follow the box's yaw only."""
    from utils.door_reference import yaw_from_quat
    yaw=yaw_from_quat(boxes[...,3:7])
    delta=hands[...,:2]-boxes[...,:2][:,:,None]
    c,s=torch.cos(yaw)[:,:,None],torch.sin(yaw)[:,:,None]
    return torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)


def box_sizes(boxes, size):
    return torch.as_tensor(size, device=boxes.device, dtype=boxes.dtype).expand(*boxes.shape[:-1], 3)


def push_face_geometry(hands, boxes, goals, size):
    """Select one finite yaw-only face, within 60 degrees of the goal direction."""
    from utils.door_reference import yaw_from_quat
    yaw=yaw_from_quat(boxes[...,3:7]);c,s=torch.cos(yaw),torch.sin(yaw)
    local_normals=boxes.new_tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    normals=torch.stack((c[...,None]*local_normals[:,0]-s[...,None]*local_normals[:,1],
                         s[...,None]*local_normals[:,0]+c[...,None]*local_normals[:,1]),-1)
    dims=box_sizes(boxes,size)/2
    half=dims[..., [0,0,1,1]]
    centers=boxes[...,:2].unsqueeze(-2)-normals*half[...,None]
    tangents=torch.stack((normals[...,1],-normals[...,0]),-1)
    widths=dims[..., [1,1,0,0]]
    delta=goals[...,:2]-boxes[...,:2]
    fallback=torch.stack((-s,c),-1)
    direction=torch.where((delta.norm(dim=-1)>1e-6)[...,None],
        delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6),fallback)
    allowed=(normals*direction[...,None,:]).sum(-1)>=.5-1e-6
    offset=hands[...,:2].unsqueeze(-3)-centers.unsqueeze(-2)
    along=(offset*tangents.unsqueeze(-2)).sum(-1)
    across=(offset*normals.unsqueeze(-2)).sum(-1)
    over=along-along.clamp(-widths[...,None],widths[...,None])
    worst=(over.square()+across.square()).amax(-1)
    face=worst.masked_fill(~allowed,float('inf')).argmin(-1)
    def select(value):
        return value.gather(-2,face[...,None,None].expand(*face.shape,1,2)).squeeze(-2)
    return select(centers),select(normals),select(tangents),widths.gather(-1,face[...,None]).squeeze(-1)


def push_rear_xy(hands, boxes, goals, size):
    center,normal,tangent,half=push_face_geometry(hands,boxes,goals,size)
    offset=hands[...,:2]-center.unsqueeze(-2)
    along=(offset*tangent.unsqueeze(-2)).sum(-1)
    across=(offset*normal.unsqueeze(-2)).sum(-1)
    over=along-along.clamp(-half[...,None],half[...,None])
    return (over.square()+across.square()).sqrt()


def push_rear_xy_distance(hands, boxes, size):
    local=push_local_xy(hands,boxes)
    nearest=local.clone()
    nearest[...,0]=nearest[...,0].clamp(-size[0]/2,size[0]/2)
    nearest[...,1]=-size[1]/2
    return (local-nearest).norm(dim=-1)


def push_midpoint_xy_distance(hands, boxes, size):
    middle=push_local_xy(hands,boxes).mean(-2)
    target=middle.new_tensor([0.,-size[1]/2])
    return (middle-target).norm(dim=-1)


def push_goal_opposite_xy_target(boxes, goals, size):
    """Intersect the center-to-goal opposite ray with the yaw-only box rectangle."""
    from utils.door_reference import yaw_from_quat
    yaw=yaw_from_quat(boxes[...,3:7])
    c,s=torch.cos(yaw),torch.sin(yaw)
    delta=goals[...,:2]-boxes[...,:2]
    local=torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)
    fallback=torch.zeros_like(local);fallback[...,1]=1.
    local=torch.where((local.norm(dim=-1)>1e-6)[...,None],local,fallback)
    half=box_sizes(boxes,size)[...,:2]/2
    rear=-local/(local.abs()/half).amax(-1,keepdim=True)
    return boxes[...,:2]+torch.stack((c*rear[...,0]-s*rear[...,1],s*rear[...,0]+c*rear[...,1]),-1)


def push_contact_distance(hands, boxes, size, spec, goals=None, config=None):
    if goals is not None and (config or {}).get('contact_at'):
        return push_rear_xy(hands,boxes,goals,size)
    if goals is not None and (config or {}).get('push_at',{}).get('hand_reference')=='goal_opposite_surface_xy':
        # Use the rectangle face hit by the reward's opposite-goal ray.
        # Each hand may slide along that face; hand Z remains unrestricted.
        rear=push_goal_opposite_xy_target(boxes,goals,size)
        local=push_local_xy(hands,boxes)
        local_rear=push_local_xy(torch.cat((rear,boxes[...,2:3]),-1).unsqueeze(-2),boxes).squeeze(-2)
        half=box_sizes(boxes,size)[...,:2]/2
        axis=(local_rear.abs()/half).argmax(-1)
        nearest=torch.maximum(torch.minimum(local,half[...,None,:]),-half[...,None,:])
        index=axis[...,None,None].expand(*local.shape[:-1],1)
        face=local_rear.gather(-1,axis[...,None]).unsqueeze(-2).expand(*local.shape[:-1],1)
        nearest=nearest.scatter(-1,index,face)
        return (local-nearest).norm(dim=-1)
    if midpoint_xy_control(spec):
        return push_rear_xy_distance(hands,boxes,size)
    return push_hand_point_distance(hands,boxes,size,spec)


def push_forward_normal(boxes, spec, goals=None, config=None):
    if goals is not None and (config or {}).get('push_at',{}).get('hand_reference')=='goal_opposite_surface_xy':
        delta=goals[...,:2]-boxes[...,:2]
        fallback=push_forward_normal(boxes,spec)[...,:2]
        normal=torch.where((delta.norm(dim=-1)>1e-6)[...,None],
            delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6),fallback)
        return torch.cat((normal,torch.zeros_like(normal[...,:1])),-1)
    from utils.door_reference import rotate, yaw_from_quat
    if midpoint_xy_control(spec):
        yaw=yaw_from_quat(boxes[...,3:7])
        return torch.stack((-torch.sin(yaw),torch.cos(yaw),torch.zeros_like(yaw)),-1)
    return rotate(boxes[...,3:7],boxes.new_tensor([0.,1.,0.]).expand_as(boxes[...,:3]))


def push_surface_distance(hands,boxes,size):
    from utils.door_reference import rotate
    q=boxes[...,3:7].clone();q[...,:3]*=-1
    local=rotate(q[:,:,None].expand(*hands.shape[:-1],4),hands-boxes[:,:,:3][:,:,None])
    half=hands.new_tensor(size)/2
    # Nearest point on the rear face, which the forward-facing expert pushes.
    point=local.clone()
    point[...,0]=point[...,0].clamp(-half[0],half[0])
    point[...,1]=-half[1]
    point[...,2]=point[...,2].clamp(-half[2],half[2])
    return (local-point).norm(dim=-1)


def push_hand_targets(boxes, size, spec):
    """RSI right/left alignment points; also reward targets in the legacy mode."""
    from utils.door_reference import rotate
    offsets=boxes.new_tensor([[spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2],
                             [-spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2]])
    return boxes[...,:3][:,:,None]+rotate(boxes[...,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        offsets.expand(*boxes.shape[:2],2,3))


def push_rsi_hand_targets(boxes, goals, size, config):
    """Align RSI hand midpoint and facing direction to the reward reference."""
    spec=config['push']
    if config.get('contact_at'):
        # RSI has no placed hands yet: use the opposite-goal ray as a seed.
        rear=push_goal_opposite_xy_target(boxes,goals,size)
        seed=torch.cat((rear,boxes[...,2:3]),-1).unsqueeze(-2).expand(*boxes.shape[:2],2,3)
        center,normal,tangent,_=push_face_geometry(seed,boxes,goals,size)
        targets=boxes.new_zeros(*boxes.shape[:2],2,3)
        targets[...,:2]=center.unsqueeze(-2)+tangent.unsqueeze(-2)*boxes.new_tensor(
            [spec['hand_lateral_offset'],-spec['hand_lateral_offset']])[...,None]
        targets[...,2]=(boxes[...,2]+spec['hand_target_height']-box_sizes(boxes,size)[...,2]/2)[...,None]
        return targets
    if config.get('push_at',{}).get('hand_reference')!='goal_opposite_surface_xy':
        return push_hand_targets(boxes,size,spec)
    rear=push_goal_opposite_xy_target(boxes,goals,size)
    normal=boxes[...,:2]-rear
    normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    right=torch.stack((normal[...,1],-normal[...,0]),-1)
    targets=boxes.new_zeros(*boxes.shape[:2],2,3)
    targets[...,:2]=rear.unsqueeze(-2)+right.unsqueeze(-2)*boxes.new_tensor(
        [spec['hand_lateral_offset'],-spec['hand_lateral_offset']])[...,None]
    targets[...,2]=spec['hand_target_height']
    return targets


def push_hand_point_distance(hands, boxes, size, spec):
    return (hands-push_hand_targets(boxes,size,spec)).norm(dim=-1)


def push_contact_near(distance, spec, config=None):
    """Goal-relative CONTACT state before reward saturation; legacy keeps its radius."""
    hand=(config or {}).get('push_at',{})
    if (config or {}).get('contact_at'):
        cfg=config['contact_at']
        return torch.exp(-cfg['near_distance_scale']*distance.amax(-1).square())>cfg['satisfaction_threshold']
    if hand.get('hand_reference')=='goal_opposite_surface_xy':
        scores=torch.exp(-hand['hand_distance_scale']*distance.square())
        right,left=scores.unbind(-1)
        state=2*right*left/(right+left).clamp_min(1e-12)
        return state>(config or {})['satisfaction_threshold']
    return (distance<=spec['hand_distance']).all(-1)


def push_debug_metrics(hands,forces,boxes,box_forces,start,size,spec,contact,eligible,episode_start=None,episode_direction=None,goals=None,config=None):
    """Per-owner physical diagnostics; never modify rewards or contact history."""
    from utils.door_reference import rotate
    target=push_rsi_hand_targets(boxes,goals,size,config) if goals is not None and config is not None else push_hand_targets(boxes,size,spec)
    distance=push_contact_distance(hands,boxes,size,spec,goals,config)
    normal=(push_face_geometry(hands,boxes,goals,size)[1] if goals is not None and
        (config or {}).get('contact_at') else push_forward_normal(boxes,spec,goals,config))
    if normal.shape[-1]==2:normal=torch.cat((normal,torch.zeros_like(normal[...,:1])),-1)
    reaction=-(forces*normal[:,:,None]).sum(-1)
    tilt=torch.acos((1-2*(boxes[...,3].square()+boxes[...,4].square())).clamp(-1,1))
    near=push_contact_near(distance,spec,config if goals is not None else None)
    upright=(tilt<=spec['max_tilt']) & ((boxes[...,2]-start[...,2]).abs()<=spec['ground_tolerance'])
    # Tracking bottom edges removes center motion caused purely by forward rocking.
    dims=box_sizes(boxes,size)
    edge=boxes.new_zeros(*boxes.shape[:2],2,3)
    edge[...,0,1]=-dims[...,1]/2;edge[...,1,1]=dims[...,1]/2
    edge[...,2]=-dims[...,2,None]/2
    world=boxes[:,:,:3][:,:,None]+rotate(boxes[:,:,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        edge)
    low=world[...,2].argmin(-1,keepdim=True)
    edge_y=world[...,1].gather(-1,low).squeeze(-1)
    initial_y=start[...,1]+edge[...,1].gather(-1,low).squeeze(-1)
    episode_start=start if episode_start is None else episode_start
    episode_edge_y=episode_start[...,1]+edge[...,1].gather(-1,low).squeeze(-1)
    box_forward=boxes[...,1]-start[...,1]
    reset_forward=boxes[...,1]-episode_start[...,1]
    bottom_forward=edge_y-initial_y
    reset_bottom_forward=edge_y-episode_edge_y
    if episode_direction is not None:
        box_forward=((boxes[...,:3]-start)*episode_direction).sum(-1)
        reset_forward=((boxes[...,:3]-episode_start)*episode_direction).sum(-1)
        projection=(world*episode_direction[:,:,None]).sum(-1).gather(-1,low).squeeze(-1)
        bottom_forward=projection-(start*episode_direction).sum(-1)-edge[...,1].gather(-1,low).squeeze(-1)
        reset_bottom_forward=projection-(episode_start*episode_direction).sum(-1)-edge[...,1].gather(-1,low).squeeze(-1)
    return dict(right_distance=distance[...,0],left_distance=distance[...,1],
        worst_hand_distance=distance.amax(-1),right_height_error=hands[...,0,2]-target[...,0,2],
        left_height_error=hands[...,1,2]-target[...,1,2],both_hands_near=near.float(),
        right_reaction_N=reaction[...,0],left_reaction_N=reaction[...,1],
        box_forward_force_N=(box_forces*normal).sum(-1),paired_contact=contact.float(),
        eligible=eligible.float(),tilt_rad=tilt,tilt_violation=(tilt>spec['max_tilt']).float(),
        grounded=upright.float(),box_forward_m=box_forward,
        bottom_edge_forward_m=bottom_forward,
        since_reset_forward_m=reset_forward,
        since_reset_bottom_edge_forward_m=reset_bottom_forward,
        box_speed=boxes[...,7:10].norm(dim=-1))


def approach_xy_diagnostics(roots, handles, boxes, config):
    """Reward-independent XY approach scores; larger is closer (0..1)."""
    door = config['door']
    open_distance = (roots[..., :2] - handles[..., :2]).norm(dim=-1)
    push_distance = (roots[..., :2] - boxes[..., :2]).norm(dim=-1)
    return {
        'open': dict(distance_xy=open_distance, xy_progress=1 / (1 +
            (open_distance - door['approach_delta']).clamp_min(0) / door['approach_sigma'])),
        'push': dict(distance_xy=push_distance, xy_progress=1 / (1 + (push_distance - .5).clamp_min(0))),
    }


def push_hand_contact(hands,forces,boxes,box_forces,size,spec,goals=None,config=None):
    from utils.door_reference import rotate
    normal=(push_face_geometry(hands,boxes,goals,size)[1] if goals is not None and
        (config or {}).get('contact_at') else push_forward_normal(boxes,spec,goals,config))
    if normal.shape[-1]==2:normal=torch.cat((normal,torch.zeros_like(normal[...,:1])),-1)
    distance=push_contact_distance(hands,boxes,size,spec,goals,config)
    near=push_contact_near(distance,spec,config if goals is not None else None)
    reaction=-(forces*normal[:,:,None]).sum(-1)
    push=(box_forces*normal).sum(-1)
    return near & (reaction>=spec['min_hand_force']).all(-1) & (push>=spec['min_box_force'])

def push_reward(roots,hands,boxes,goals,start,eligible,config,size=None,manipulation=None):
    p=config['push'];pos=boxes[...,:3];q=boxes[...,3:7]
    distance=(pos[...,:2]-goals[...,:2]).norm(dim=-1)
    # Box center stays close to its reset ground height, with an upright orientation.
    up_z=1-2*(q[...,0].square()+q[...,1].square())
    grounded=((pos[...,2]-start[...,2]).abs()<=p['ground_tolerance']) & (up_z>=torch.cos(pos.new_tensor(p['max_tilt'])))
    stable=(boxes[...,7:10].norm(dim=-1)<=p['max_speed']) & (boxes[...,10:13].norm(dim=-1)<=p['max_angular_speed'])
    success=(distance<=p['goal_tolerance']) & grounded & stable & eligible
    initial=(start[...,:2]-goals[...,:2]).norm(dim=-1).clamp_min(1e-6)
    progress=(1-distance/initial).clamp(0,1)*grounded
    approach=1/(1+((roots[...,:2]-pos[...,:2]).norm(dim=-1)-.5).clamp_min(0))
    if size is None:
        raise ValueError('Two-hand PUSH requires the actual box size')
    if midpoint_xy_control(p):
        hand_distance=push_midpoint_xy_distance(hands,boxes,size)
    else:
        hand_distance=push_hand_point_distance(hands,boxes,size,p).amax(-1)
    interaction=torch.exp(-10*hand_distance.square())*grounded
    state=(approach+interaction)*.5*config['state_reward_weight']
    state=torch.where(success,torch.full_like(state,config['state_reward_weight']),state)
    prog=torch.where(success,torch.ones_like(progress),progress)*config['progress_reward_weight']
    bonus=success.float()*config['success_reward_weight']
    if manipulation is not None:
        # Keep actual hand-distance shaping, including after completion.
        state=(approach+interaction)*.5*config['state_reward_weight']
        completed=manipulation['completed'] & success & ~manipulation['body_contact']
        progress=torch.minimum(progress,manipulation['credited_progress'])
        prog=torch.where(completed,torch.ones_like(progress),
                         progress*manipulation['valid'].float())*config['progress_reward_weight']
        success=completed
        bonus=completed.float()*config['success_reward_weight']
    return dict(state=state,progress=prog,bonus=bonus,total=state+prog+bonus,success=success,
        phi=progress,error=distance,grounded=grounded)


def push_contact_shaping(roots,hands,boxes,goals,size,config):
    """Two-hand XY face proximity and goal-opposite root positioning, without forces."""
    h=config['push_at'];s=h['contact_shaping']
    distances=push_contact_distance(hands,boxes,size,config['push'],goals,config)
    scores=torch.exp(-h['hand_distance_scale']*distances.square())
    left,right=scores.unbind(-1)
    proximity=2*left*right/(left+right).clamp_min(1e-12)
    offset=roots[...,:2]-boxes[...,:2]
    distance=offset.norm(dim=-1)
    direction=push_forward_normal(boxes,config['push'],goals,config)[...,:2]
    dot=((offset/distance.clamp_min(1e-6)[...,None])*direction).sum(-1).clamp(-1.,1.)
    error=1-torch.acos(dot)/torch.pi  # zero when root and goal are on opposite sides
    alignment=torch.exp(-error.square()/s['alignment_sigma2'])
    # A root at the box center has no meaningful positioning direction.
    alignment=torch.where(distance>1e-6,alignment,torch.zeros_like(alignment))
    near=((s['gate_start_distance']-distance)/
          (s['gate_start_distance']-s['gate_full_distance'])).clamp(0.,1.)
    gate=1-near*(1-alignment)
    approach=1/(1+(distance-h['approach_delta']).clamp_min(0)/h['approach_sigma'])
    weight=s['alignment_weight']
    progress=(1-weight)*approach*gate+weight*alignment
    return proximity,progress,alignment,gate,scores


def contact_at_reward(roots,hands,boxes,goals,contact,size,config):
    """Independent CONTACT/AT edges; force contact or AT arrival saturates CONTACT."""
    a=config['contact_at']
    distances=push_rear_xy(hands,boxes,goals,size)
    push_phi=torch.exp(-a['near_distance_scale']*distances.amax(-1).square())
    delta=boxes[...,:3]-goals
    at_phi=torch.exp(-a['near_distance_scale']*delta.square().sum(-1))
    at_success=(at_phi>=a['satisfaction_threshold']) & (delta[...,2].abs()<=a['z_tolerance'])
    if 'max_tilt' in a:
        up=1-2*(boxes[...,3].square()+boxes[...,4].square())
        at_success &= up>=torch.cos(boxes.new_tensor(a['max_tilt']))
    root_distance=(roots[...,:2]-boxes[...,:2]).norm(dim=-1)
    push_progress=1/(1+(root_distance-a['progress_delta']).clamp_min(0)/a['progress_sigma'])
    at_progress=1/(1+(delta[...,:2].norm(dim=-1)-a['at_progress_delta']).clamp_min(0)/a['progress_sigma'])
    phi=torch.stack((push_phi,at_phi),-1).flatten(1)
    progress=torch.stack((push_progress,at_progress),-1).flatten(1)
    own_success=torch.stack((contact,at_success),-1).flatten(1)
    saturated=torch.stack((contact|at_success,at_success),-1).flatten(1)
    state=torch.where(saturated,1.,phi)*config['state_reward_weight']
    prog=torch.where(saturated,1.,progress)*config['progress_reward_weight']
    bonus=saturated.float()*config['success_reward_weight']
    if 'tilt_full_angle' in a:
        angle=torch.acos((1-2*(boxes[...,3].square()+boxes[...,4].square())).clamp(-1,1))
        scale=((a['tilt_zero_angle']-angle)/(a['tilt_zero_angle']-a['tilt_full_angle'])).clamp(0.,1.)
        # Apply after saturation: arriving at the goal cannot bypass shaping.
        edge_scale=torch.stack((torch.ones_like(scale),scale),-1).flatten(1)
        state=state*edge_scale
        prog=prog*edge_scale
    local=(state+prog+bonus).reshape(-1,2,2).sum(-1)
    reward=config['task_sharing']['self']*local+config['task_sharing']['teammate']*local.flip(-1)/2
    return dict(state=state.reshape(-1,2,2).sum(-1),progress=prog.reshape(-1,2,2).sum(-1),
        bonus=bonus.reshape(-1,2,2).sum(-1),reward=reward,success=at_success,
        push_success=contact,phi=phi,own_success=own_success,edge_state=state,
        edge_progress=prog,edge_bonus=bonus,error=delta[...,:2].norm(dim=-1))


def push_at_edge_reward(roots,hands,boxes,goals,start,size,config):
    """HOLDING+AT-style independent edges, without contact/credit gating of AT."""
    p=config['push'];a=config['at'];h=config['push_at']
    shaping=h.get('contact_shaping')
    if shaping is not None:
        hand_phi,shaped_progress,_,_,_=push_contact_shaping(
            roots,hands,boxes,goals,size,config)
    elif h.get('hand_reference')=='goal_opposite_surface_xy':
        target=push_goal_opposite_xy_target(boxes,goals,size)
        hand_distance=(hands[...,:2].mean(-2)-target).norm(dim=-1)
    else:
        hand_distance=push_midpoint_xy_distance(hands,boxes,size)
    if shaping is None:
        hand_phi=torch.exp(-h['hand_distance_scale']*hand_distance.square())
    delta=boxes[...,:3]-goals
    at_phi=torch.exp(-a['near_distance_scale']*delta.square().sum(-1))
    up_z=1-2*(boxes[...,3].square()+boxes[...,4].square())
    upright=up_z>=torch.cos(boxes.new_tensor(p['max_tilt']))
    height_valid=(boxes[...,2]-start[...,2]).abs()<=p['ground_tolerance']
    grounded=height_valid & upright
    hand_valid=grounded if h.get('require_tilt',True) else height_valid
    hand_phi=hand_phi*hand_valid
    hand_success=hand_phi>=config['satisfaction_threshold']
    at_success=(at_phi>=config['satisfaction_threshold']) & (delta[...,2].abs()<=a['z_tolerance'])
    at_valid=grounded if a.get('require_grounded',False) else torch.ones_like(grounded)
    at_phi=at_phi*at_valid
    at_success &= at_valid
    if 'success_max_tilt' in a:
        at_success &= up_z>=torch.cos(boxes.new_tensor(a['success_max_tilt']))
    tilt=a.get('tilt_shaping')
    tilt_scale=None
    if tilt is not None:
        angle=torch.acos(up_z.clamp(-1.,1.))
        tilt_scale=((tilt['zero_angle']-angle)/(tilt['zero_angle']-tilt['full_angle'])).clamp(0.,1.)
        at_success &= up_z>=torch.cos(boxes.new_tensor(tilt['success_angle']))
    approach=(roots[...,:2]-boxes[...,:2]).norm(dim=-1)
    hand_progress=1/(1+(approach-h['approach_delta']).clamp_min(0)/h['approach_sigma'])
    if shaping is not None:
        hand_progress=shaped_progress
    at_progress=1/(1+(delta[...,:2].norm(dim=-1)-a['progress_delta']).clamp_min(0)/a['progress_sigma'])
    at_progress=at_progress*at_valid
    phi=torch.stack((hand_phi,at_phi),-1).flatten(1)
    progress=torch.stack((hand_progress*hand_valid,at_progress),-1).flatten(1)
    own_success=torch.stack((hand_success,at_success),-1).flatten(1)
    # Like paired HOLDING+AT: placement also saturates the matching hand edge.
    saturated=torch.stack((hand_success|at_success,at_success),-1).flatten(1)
    state=torch.where(saturated,1.,phi)*config['state_reward_weight']
    if a.get('state_require_upright',False):
        # Apply after saturation so arrival cannot bypass the AT state tilt gate.
        state=state*torch.stack((torch.ones_like(upright),upright),-1).flatten(1)
    prog=torch.where(saturated,1.,progress)*config['progress_reward_weight']
    if tilt_scale is not None:
        # Shaping also applies after arrival; completion must not bypass it.
        scale=torch.stack((torch.ones_like(tilt_scale),tilt_scale),-1).flatten(1)
        state=state*scale
        prog=prog*scale
    bonus=saturated.float()*config['success_reward_weight']
    local=(state+prog+bonus).reshape(-1,2,2).sum(-1)
    teammate=local.flip(-1)/2
    reward=config['task_sharing']['self']*local+config['task_sharing']['teammate']*teammate
    return dict(state=state.reshape(-1,2,2).sum(-1),
        progress=prog.reshape(-1,2,2).sum(-1),bonus=bonus.reshape(-1,2,2).sum(-1),
        reward=reward,success=at_success,phi=phi,own_success=own_success,
        edge_state=state,edge_progress=prog,edge_bonus=bonus,error=delta[...,:2].norm(dim=-1))


class OpenPushRuntime(EdgeContextRuntime):
    def __init__(self,n,spec,config,hold,dt,device,size,manipulation=None):
        super().__init__(n,expand_graph(compile_graph(spec,2,2 if spec.get('push_only',False) else 4,device),n),config,device)
        self.spec,self.hold,self.dt,self.size=spec,hold,dt,size
        self.tasks=torch.zeros(n,2,dtype=torch.long,device=device)
        self.elapsed=torch.zeros(n,2,device=device)
        self.contact_age=torch.full_like(self.elapsed,hold['contact_grace']+dt)
        self.contact=torch.zeros(n,2,dtype=torch.bool,device=device)
        self.posture=torch.zeros_like(self.contact)
        self.push_contact=torch.zeros_like(self.contact)
        self.push_eligible=torch.zeros_like(self.contact)
        self.start=torch.zeros(n,2,3,device=device)
        self.error=torch.zeros(n,2,device=device)
        self.manipulation=manipulation
        self.body_contact=torch.zeros_like(self.contact)
        self.clean_open_contact=torch.zeros_like(self.contact)
        self.clean_push_contact=torch.zeros_like(self.contact)
        self.push_completed=torch.zeros_like(self.contact)
        self.push_frontier=torch.zeros_like(self.elapsed)
        self.push_credited=torch.zeros_like(self.elapsed)
        self.push_hand_distance=torch.zeros_like(self.elapsed)
        self.open_frontier=torch.zeros_like(self.elapsed)
        self.open_credited=torch.zeros_like(self.elapsed)
        self.push_reset_remaining=torch.zeros_like(self.elapsed)
        self.push_previous_distance=torch.zeros_like(self.elapsed)
        self.push_rsi_progress=torch.zeros_like(self.elapsed)
        if config.get('edge_aggregation'):
            self.graph.required_goal[:,::2]=False

    def reset(self,ids,tasks,boxes):
        self.tasks[ids]=tasks
        copy_graph_rows(self.graph,ids,graph_for_tasks(self.spec,tasks))
        if self.config.get('edge_aggregation'):
            self.graph.required_goal[ids,::2]=False
        for x in (self.phi,self.own_success,self.achieved,self.done,self.elapsed,self.contact,self.posture,self.push_contact,self.push_eligible,self.error):x[ids]=0
        self.contact_age[ids]=self.hold['contact_grace']+self.dt
        self.start[ids]=boxes[...,:3]
        for x in (self.body_contact,self.clean_open_contact,self.clean_push_contact,self.push_completed,
                  self.push_frontier,self.push_credited,self.push_hand_distance,self.open_frontier,self.open_credited,
                  self.push_reset_remaining,self.push_previous_distance,self.push_rsi_progress):x[ids]=0

    def initialize_manipulation(self,ids,boxes,goals,angles):
        """RSI geometry remains credited; no hand movement is fabricated by reset."""
        initial=(self.start[ids,...,:2]-goals[...,:2]).norm(dim=-1)
        current=(boxes[...,:2]-goals[...,:2]).norm(dim=-1)
        advanced=(initial-current).clamp_min(0)
        self.push_frontier[ids]=advanced
        self.push_credited[ids]=advanced
        self.push_reset_remaining[ids]=current
        self.push_previous_distance[ids]=current
        self.push_rsi_progress[ids]=advanced
        self.push_hand_distance[ids]=0
        self.open_frontier[ids]=angles
        self.open_credited[ids]=angles

    def collect_manipulation(self,boxes,goals,angles,open_valid,push_valid):
        """Collect same-tick valid motion: net balance or legacy clean frontiers."""
        initial=(self.start[...,:2]-goals[...,:2]).norm(dim=-1)
        current=(boxes[...,:2]-goals[...,:2]).norm(dim=-1)
        advanced=(initial-current).clamp_min(0)
        if midpoint_xy_control(self.config['push']):
            # Backward motion cancels credit even without hand contact. Cap the
            # balance by net movement since RSI so cycles cannot accumulate it.
            delta=self.push_previous_distance-current
            balance=(self.push_hand_distance+delta.clamp_min(0)*push_valid+delta.clamp_max(0)).clamp_min(0)
            net=(self.push_reset_remaining-current).clamp_min(0)
            self.push_hand_distance.copy_(torch.minimum(balance,net))
            self.push_credited.copy_(self.push_rsi_progress+self.push_hand_distance)
            self.push_previous_distance.copy_(current)
        else:
            gain=(advanced-self.push_frontier).clamp_min(0)*push_valid
            self.push_credited+=gain
            self.push_hand_distance+=gain
            self.push_frontier.copy_(torch.maximum(self.push_frontier,advanced))
        self.open_credited+=(angles-self.open_frontier).clamp_min(0)*open_valid
        self.open_frontier.copy_(torch.maximum(self.open_frontier,angles))

    def step(self,roots,hands,handles,angles,boxes,goals):
        if self.config.get('edge_aggregation')=='self_sum_teammate_mean':
            if self.config.get('contact_at'):
                result=contact_at_reward(roots,hands,boxes,goals,self.push_contact,self.size,self.config)
            else:
                result=push_at_edge_reward(roots,hands,boxes,goals,self.start,self.size,self.config)
            self.phi.copy_(result['phi']);self.own_success.copy_(result['own_success'])
            self.achieved|=self.own_success
            self.done.copy_(result['success']);self.error.copy_(result['error'])
            self.push_eligible.copy_(self.own_success[:,::2])
            self.push_completed.copy_(self.done)
            return result
        if self.spec.get('push_only',False):
            ds=dp=db=door_progress=torch.zeros_like(angles)
            opened=torch.zeros_like(self.contact)
        else:
            support=self.contact & self.posture
            if self.manipulation is not None:
                support &= ~self.body_contact & (self.open_credited >=
                    self.config['door']['target_angle']-self.manipulation['open_credit_tolerance'])
            self.elapsed[:],opened=update_hold(self.elapsed,angles,support,
                self.config['door']['target_angle']-self.config['door']['angle_tolerance'],self.hold['duration'],self.dt)
            d=door_reward(roots,hands,handles,angles,
                door_graph({'sampler':'two_agent_two_door_open','semantic_only':True,'bindings':[0,1]},2,2,roots.device),self.config,success_override=opened)
            d['interaction_component']*=(.5+.5*self.contact.float())*self.posture.float()
            door_progress=.5*(angles/self.config['door']['target_angle']).clamp(0,1)*self.posture.float()+.5*self.elapsed/self.hold['duration']
            if self.manipulation is not None:
                valid=self.clean_open_contact & self.posture & ~self.body_contact
                angle_progress=(torch.minimum(angles,self.open_credited)/self.config['door']['target_angle']).clamp(0,1)
                door_progress=.5*angle_progress*valid+.5*self.elapsed/self.hold['duration']
            ds=d['approach_component']+d['interaction_component']
            dp=torch.where(opened,torch.ones_like(angles),door_progress)*self.config['progress_reward_weight']
            db=d['success_component']
        # Eligibility requires own-hand contact while the grounded box has advanced toward its goal.
        initial=(self.start[...,:2]-goals[...,:2]).norm(dim=-1)
        current=(boxes[...,:2]-goals[...,:2]).norm(dim=-1)
        candidate=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size)
        manipulation=None
        if midpoint_xy_control(self.config['push']):
            required=(self.push_reset_remaining-self.config['push']['goal_tolerance']).clamp_min(
                self.config['push']['min_contact_progress'])
            self.push_eligible.copy_((self.push_hand_distance>=required) & (self.tasks==1))
            credited=self.push_credited/initial.clamp_min(1e-6)
            ready=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size)['success']
            self.push_completed.copy_(ready)
            manipulation=dict(valid=self.clean_push_contact & candidate['grounded'],credited_progress=credited,
                completed=ready,body_contact=torch.zeros_like(self.body_contact))
        elif self.manipulation is None:
            self.push_eligible|=self.push_contact & candidate['grounded'] & ((initial-current)>=self.config['push']['min_contact_progress']) & (self.tasks==1)
        else:
            self.push_eligible.copy_((self.push_hand_distance>=self.config['push']['min_contact_progress']) & (self.tasks==1))
            valid=self.clean_push_contact & ~self.body_contact & candidate['grounded']
            credited=self.push_credited/initial.clamp_min(1e-6)
            qualified=credited>=1-self.config['push']['goal_tolerance']/initial.clamp_min(1e-6)
            ready=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size)['success']
            self.push_completed.copy_(ready & ~self.body_contact &
                (self.push_completed | (qualified & valid)))
            manipulation=dict(valid=valid,credited_progress=credited,
                completed=self.push_completed,body_contact=self.body_contact)
        p=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size,manipulation)
        active=self.tasks==1
        state=torch.where(active,p['state'],ds)
        progress=torch.where(active,p['progress'],dp)
        bonus=torch.where(active,p['bonus'],db)
        success=torch.where(active,p['success'],opened)
        local=state+progress+bonus
        reward=.9*local+.1*local.flip(-1)
        self.phi[:,::2]=torch.where(active,p['phi'],door_progress)
        self.phi[:,1::2]=p['phi']*active
        self.own_success[:,::2]=success
        self.own_success[:,1::2]=p['success'] & active
        self.done.copy_(success);self.achieved|=self.own_success
        self.error.copy_(torch.where(active,p['error'],(self.config['door']['target_angle']-angles).clamp_min(0)))
        return dict(state=state,progress=progress,bonus=bonus,reward=reward,success=success)

    def suffix(self,ids=None):
        from utils.edge_ontop_spec import select_graph
        g=self.graph if ids is None else select_graph(self.graph,ids)
        return semantic_graph_packet(g,self.phi.shape[0] if ids is None else len(ids))
