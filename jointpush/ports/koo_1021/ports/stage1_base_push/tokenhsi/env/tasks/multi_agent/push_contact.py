"""Current PUSH contact geometry, copied without OPEN runtime dependencies."""
import torch

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

