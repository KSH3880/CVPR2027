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
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError('Unexpected joint PUSH reward fields')
    for name, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Reward fields must be finite numbers')
        if value < 0:
            raise ValueError('Reward fields must be nonnegative')
    if not 0 < config['min_separation_fraction'] < 1 or not 0 < config['satisfaction_threshold'] <= 1:
        raise ValueError('Invalid anchor fraction/threshold')
    if min(config['progress_sigma'], config['hand_distance_scale'], config['at_distance_scale'], config['symmetry_distance_scale']) <= 0:
        raise ValueError('Reward scales must be positive')


def push_face_geometry(boxes, goals, sizes, config):
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


def joint_push_reward(roots, hands, boxes, goals, sizes, config=None):
    config = dict(DEFAULT_CONFIG) if config is None else config
    center,normal,tangent,half_width,sizes = push_face_geometry(boxes,goals,sizes,config)
    n=boxes.shape[0]
    if roots.shape != (n,2,13) or hands.shape != (n,2,2,3):
        raise ValueError('Expected exactly two roots [N,2,13] and hands [N,2,2,3]')
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
    separation=(lateral[:,0]-lateral[:,1]).abs()
    minimum_separation=2*half_width*config['min_separation_fraction']
    separation_phi=(separation/minimum_separation).clamp(0,1)
    coordination_phi=symmetry_phi*separation_phi
    phi_push=surface_phi*coordination_phi[:,None]
    opposite=lateral[:,0]*lateral[:,1]<0
    push_success=phi_push>=config['satisfaction_threshold']
    gate=push_success.all(-1)&opposite&(separation>=minimum_separation)
    delta=boxes[:,:3]-goals
    phi_at=torch.exp(-config['at_distance_scale']*delta.square().sum(-1))
    at_success=(phi_at>=config['satisfaction_threshold']) & (delta[:,2].abs()<=config['at_z_tolerance']) & gate
    # Same distance pinning recipe as Carry: payload/support bbox + buffer.
    radius=sizes[:,:2].norm(dim=-1)/2
    push_distance=(roots[...,:2]-boxes[:,None,:2]).norm(dim=-1)
    push_progress=1/(1+(push_distance-radius[:,None]-config['push_bbox_buffer']).clamp_min(0)/config['progress_sigma'])
    at_distance=delta[:,:2].norm(dim=-1)
    at_progress=1/(1+(at_distance-config['at_bbox_buffer']).clamp_min(0)/config['progress_sigma'])
    phi=torch.stack((phi_push,phi_at[:,None].expand(-1,2)),-1)
    progress=torch.stack((push_progress,at_progress[:,None].expand(-1,2)),-1)
    active=torch.stack((torch.ones_like(push_success),gate[:,None].expand(-1,2)),-1)
    success=torch.stack((push_success,at_success[:,None].expand(-1,2)),-1)
    state=phi*active*config['state_reward_weight']
    prog=progress*active*config['progress_reward_weight']
    bonus=success.to(phi.dtype)*config['success_reward_weight']
    total=state+prog+bonus
    return dict(face_center=center,push_normal=normal,hand_midpoints=midpoints,
        lateral_offsets=lateral,surface_phi=surface_phi,symmetry_phi=symmetry_phi,
        separation=separation,minimum_separation=minimum_separation,coordination_phi=coordination_phi,
        coupled_contact=gate,phi_raw=phi,progress_raw=progress,active_edges=active,own_success=success,
        state_component=state,progress_component=prog,success_component=bonus,total=total,
        local_task_reward=total.sum(-1),agent_task_reward=total.sum(-1),
        teammate_task_reward=torch.zeros_like(phi_push),done=at_success[:,None].expand(-1,2))
