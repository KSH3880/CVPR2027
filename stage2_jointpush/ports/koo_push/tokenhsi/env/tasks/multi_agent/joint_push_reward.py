"""Two-person PUSH+AT reward, following JointCarryRuntime's unsaturated edge sum.

Model/reward building block only: caller must supply ONE shared box and goal.
No simulator or MPS dependency; contact gate is geometric, not force-certified.
"""
import math
import torch

DEFAULT_CONFIG = dict(state_reward_weight=.2, progress_reward_weight=.2,
    success_reward_weight=.2, satisfaction_threshold=.9, hand_distance_scale=10.,
    at_distance_scale=10., at_z_tolerance=.001, progress_sigma=1.,
    push_bbox_buffer=.2, at_bbox_buffer=.1, min_separation_fraction=.25,
    symmetry_distance_scale=10.,
    hand_target_height=.8)


def validate_config(config):
    if set(config)-{'at_gate_mode','contact_mode'} != set(DEFAULT_CONFIG):
        raise ValueError('Unexpected joint PUSH reward fields')
    if config.get('at_gate_mode','hard') not in ('hard','soft_min'):
        raise ValueError('Unsupported AT gate mode')
    if config.get('contact_mode','fixed_face') not in ('fixed_face','directional_surface','assigned_faces'):
        raise ValueError('Unsupported contact mode')
    for name, value in config.items():
        if name in ('at_gate_mode','contact_mode'):continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Reward fields must be finite numbers')
        if value < 0:
            raise ValueError('Reward fields must be nonnegative')
    if not 0 < config['min_separation_fraction'] < 1 or not 0 < config['satisfaction_threshold'] <= 1:
        raise ValueError('Invalid anchor fraction/threshold')
    if min(config['progress_sigma'], config['hand_distance_scale'], config['at_distance_scale'], config['symmetry_distance_scale']) <= 0:
        raise ValueError('Reward scales must be positive')


def push_face_geometry(boxes, goals, sizes, config, local_normal=None):
    """Select a real rear face; return its center, normal, tangent and half width.

Yaw-frame geometry, hand height measured above the box's bottom. At zero
remaining XY distance use local +Y as forward. Face ties prefer local Y.
"""
    validate_config(config)
    n = boxes.shape[0]
    if boxes.shape != (n, 13) or goals.shape != (n, 3):
        raise ValueError('Expected shared box [N,13] and goal [N,3]')
    sizes = torch.as_tensor(sizes, device=boxes.device, dtype=boxes.dtype)
    if sizes.shape == (3,): sizes = sizes.expand(n, -1)
    if sizes.shape != (n, 3) or (sizes <= 0).any() or not torch.isfinite(sizes).all():
        raise ValueError('Expected positive actual size [N,3]')
    if not torch.isfinite(boxes).all() or not torch.isfinite(goals).all():
        raise ValueError('Box/goal must be finite')
    q = boxes[:, 3:7]
    if not torch.allclose(q.norm(dim=-1), torch.ones(n, device=q.device, dtype=q.dtype), atol=1e-4):
        raise ValueError('Box quaternion must be normalized')
    x,y,z,w = q.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c,s = yaw.cos(),yaw.sin()
    delta = goals[:, :2]-boxes[:, :2]
    local_direction = torch.stack((c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1]), -1)
    zero = delta.norm(dim=-1) <= 1e-6
    local_direction = torch.where(zero[:,None], boxes.new_tensor([0.,1.]), local_direction)
    axis = (local_direction[:,1].abs() >= local_direction[:,0].abs()).long()
    selected = local_direction.gather(1,axis[:,None]).squeeze(1)
    sign = torch.where(selected>=0,1.,-1.)
    normal = torch.zeros_like(local_direction).scatter(1,axis[:,None],sign[:,None])
    if local_normal is not None:
        if local_normal.shape != (n,2):
            raise ValueError("Expected saved local face normal [N,2]")
        normal = local_normal
        axis = normal.abs().argmax(-1)
    tangent = torch.stack((-normal[:,1],normal[:,0]),-1)
    half = sizes[:,:2]/2
    center = -normal*half
    lateral_half = half.gather(1,(1-axis)[:,None]).squeeze(1)
    world_center = torch.stack((c*center[:,0]-s*center[:,1],
                                s*center[:,0]+c*center[:,1]),-1)+boxes[:,:2]
    height = boxes[:,2]-sizes[:,2]/2+config['hand_target_height']
    face_center = torch.cat((world_center,height[:,None]),-1)
    world_normal = torch.stack((c*normal[:,0]-s*normal[:,1],s*normal[:,0]+c*normal[:,1]),-1)
    world_tangent = torch.stack((-world_normal[:,1],world_normal[:,0]),-1)
    return face_center, world_normal, world_tangent, lateral_half, sizes



def directional_surface_contact(roots, hands, boxes, goals, sizes, config):
    """Nearest vertical box surface for each hand; no assigned face or agent role.

    Uses the upright box yaw frame, as in the original contact geometry.
    Both hands must have valid distance, height and inward push direction.
    """
    q=boxes[:,3:7];x,y,z,w=q.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));c=yaw.cos();s=yaw.sin()
    def local_xy(points):
        d=points-boxes[:,None,None,:2]
        return torch.stack((c[:,None,None]*d[...,0]+s[:,None,None]*d[...,1],
                            -s[:,None,None]*d[...,0]+c[:,None,None]*d[...,1]),-1)
    xy=local_xy(hands[...,:2]);half=sizes[:,:2]/2
    # Face order: -X,+X,-Y,+Y. Distance includes finite face extents.
    normals=boxes.new_tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    hx=half[:,0,None,None];hy=half[:,1,None,None]
    ex=(xy[...,0].abs()-hx).clamp_min(0);ey=(xy[...,1].abs()-hy).clamp_min(0)
    errors=torch.stack(((xy[...,0]+hx).square()+ey.square(),
        (xy[...,0]-hx).square()+ey.square(),
        (xy[...,1]+hy).square()+ex.square(),
        (xy[...,1]-hy).square()+ex.square()),-1)
    face=errors.argmin(-1)
    surface=torch.exp(-config['hand_distance_scale']*errors.amin(-1).amax(-1))
    height=boxes[:,2]-sizes[:,2]/2+config['hand_target_height']
    height_phi=torch.exp(-config['hand_distance_scale']*(hands[...,2]-height[:,None,None]).square().amax(-1))
    delta=goals[:,:2]-boxes[:,:2]
    direction=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),-1)
    length=direction.norm(dim=-1,keepdim=True)
    direction=torch.where(length>1e-6,direction/length.clamp_min(1e-6),boxes.new_tensor([0.,1.]))
    direction_phi=((normals[face]*direction[:,None,None]).sum(-1)/.7).clamp(0,1).amin(-1)
    mid=hands.mean(-2)
    pair=(mid[:,:,None,:2]-mid[:,None,:,:2]).norm(dim=-1)
    diagonal=torch.eye(roots.shape[1],device=boxes.device,dtype=torch.bool)[None]
    nearest=pair.masked_fill(diagonal,float('inf')).amin(-1)
    minimum=sizes[:,0]*config['min_separation_fraction']
    spacing=(nearest/minimum[:,None]).clamp(0,1)
    phi=surface*height_phi*direction_phi*spacing
    root=local_xy(roots[...,:2].unsqueeze(2)).squeeze(2)
    distance=(root.abs()-half[:,None]).clamp_min(0).norm(dim=-1)
    progress=1/(1+(distance-config['push_bbox_buffer']).clamp_min(0)/config['progress_sigma'])
    return dict(phi=phi,surface=surface,height=height_phi,direction=direction_phi,
                spacing=spacing,separation=nearest.amin(-1),minimum=minimum,progress=progress,
                faces=face)


def joint_push_reward(roots, hands, boxes, goals, sizes, config=None, face_geometry=None, assigned_targets=None):
    config = dict(DEFAULT_CONFIG) if config is None else config
    center,normal,tangent,half_width,sizes = push_face_geometry(boxes,goals,sizes,config) if face_geometry is None else face_geometry
    n=boxes.shape[0]
    agents=roots.shape[1]
    if agents not in (2,3,4) or roots.shape != (n,agents,13) or hands.shape != (n,agents,2,3):
        raise ValueError('Expected 2-4 roots [N,A,13] and hands [N,A,2,3]')
    if not torch.isfinite(roots).all() or not torch.isfinite(hands).all():
        raise ValueError('Human states must be finite')
    midpoints=hands.mean(-2)
    relative=midpoints-center[:,None]
    lateral=(relative[...,:2]*tangent[:,None]).sum(-1)
    depth=(relative[...,:2]*normal[:,None]).sum(-1)
    outside=(lateral.abs()-half_width[:,None]).clamp_min(0)
    surface_error_sq=depth.square()+outside.square()+relative[...,2].square()
    surface_phi=torch.exp(-config['hand_distance_scale']*surface_error_sq)
    symmetry_error=(midpoints.mean(1)-center).norm(dim=-1)
    symmetry_phi=torch.exp(-config['symmetry_distance_scale']*symmetry_error.square())
    ordered=lateral.sort(-1).values
    separation=(ordered[:,1:]-ordered[:,:-1]).amin(-1)
    minimum_separation=2*half_width*config['min_separation_fraction']
    separation_phi=(separation/minimum_separation).clamp(0,1)
    coordination_phi=symmetry_phi*separation_phi
    phi_push=surface_phi*coordination_phi[:,None]
    opposite=(lateral.amin(-1)<0)&(lateral.amax(-1)>0)
    push_success=phi_push>=config['satisfaction_threshold']
    gate=push_success.all(-1)&opposite&(separation>=minimum_separation)
    directional=config.get('contact_mode','fixed_face')=='directional_surface'
    contact=None
    if directional:
        contact=directional_surface_contact(roots,hands,boxes,goals,sizes,config)
        phi_push=contact['phi'];surface_phi=contact['surface']
        separation=contact['separation'];minimum_separation=contact['minimum']
        symmetry_phi=torch.ones_like(symmetry_phi)
        coordination_phi=contact['spacing'].amin(-1)
        push_success=phi_push>=config['satisfaction_threshold']
        gate=push_success.all(-1)

    assigned=config.get('contact_mode','fixed_face')=='assigned_faces'
    if assigned:
        if assigned_targets is None:
            raise ValueError('Assigned-face reward requires reset-time hand/approach targets')
        hand_targets,approach_targets=assigned_targets
        if hand_targets.shape!=hands.shape or approach_targets.shape!=(n,agents,3):
            raise ValueError('Invalid assigned target shapes')
        phi_push=torch.exp(-config['hand_distance_scale']*(hands-hand_targets).square().sum(-1).amax(-1))
        surface_phi=phi_push
        symmetry_phi=torch.ones_like(symmetry_phi)
        coordination_phi=torch.ones_like(coordination_phi)
        push_success=phi_push>=config['satisfaction_threshold']
        gate=push_success.all(-1)
    delta=boxes[:,:3]-goals
    phi_at=torch.exp(-config['at_distance_scale']*delta.square().sum(-1))
    at_success=(phi_at>=config['satisfaction_threshold']) & (delta[:,2].abs()<=config['at_z_tolerance']) & gate
    # Same distance pinning recipe as Carry: payload/support bbox + buffer.
    radius=sizes[:,:2].norm(dim=-1)/2
    push_distance=(roots[...,:2]-boxes[:,None,:2]).norm(dim=-1)
    push_progress=1/(1+(push_distance-radius[:,None]-config['push_bbox_buffer']).clamp_min(0)/config['progress_sigma'])
    if directional: push_progress=contact['progress']
    if assigned:
        push_progress=1/(1+((roots[...,:2]-approach_targets[...,:2]).norm(dim=-1)-config['push_bbox_buffer']).clamp_min(0)/config['progress_sigma'])
    at_distance=delta[:,:2].norm(dim=-1)
    at_progress=1/(1+(at_distance-config['at_bbox_buffer']).clamp_min(0)/config['progress_sigma'])
    phi=torch.stack((phi_push,phi_at[:,None].expand(-1,agents)),-1)
    progress=torch.stack((push_progress,at_progress[:,None].expand(-1,agents)),-1)
    active=torch.stack((torch.ones_like(push_success),gate[:,None].expand(-1,agents)),-1)
    success=torch.stack((push_success,at_success[:,None].expand(-1,agents)),-1)
    # Only dense AT rewards change; result/done retain the strict joint gate.
    at_gate=phi_push.amin(-1)*(torch.ones_like(opposite) if directional or assigned else opposite).to(phi.dtype) if config.get('at_gate_mode','hard')=='soft_min' else gate.to(phi.dtype)
    dense_active=torch.stack((torch.ones_like(at_gate),at_gate),-1)[:,None].expand(-1,agents,-1)
    state=phi*dense_active*config['state_reward_weight']
    prog=progress*dense_active*config['progress_reward_weight']
    bonus=success.to(phi.dtype)*config['success_reward_weight']
    total=state+prog+bonus
    return dict(hand_targets=hand_targets if assigned else None,approach_targets=approach_targets if assigned else None,contact_direction_phi=contact['direction'] if directional else torch.ones_like(phi_push),
        contact_height_phi=contact['height'] if directional else torch.ones_like(phi_push),
        contact_face_ids=contact['faces'] if directional else None,face_center=center,push_normal=normal,hand_midpoints=midpoints,
        lateral_offsets=lateral,surface_phi=surface_phi,symmetry_phi=symmetry_phi,
        separation=separation,minimum_separation=minimum_separation,coordination_phi=coordination_phi,
        dense_at_gate=at_gate,coupled_contact=gate,phi_raw=phi,progress_raw=progress,active_edges=active,own_success=success,
        state_component=state,progress_component=prog,success_component=bonus,total=total,
        local_task_reward=total.sum(-1),agent_task_reward=total.sum(-1),
        teammate_task_reward=torch.zeros_like(phi_push),done=at_success[:,None].expand(-1,agents))
