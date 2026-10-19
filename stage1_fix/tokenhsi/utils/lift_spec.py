"""Independent SUPPORT + LIFT geometry, rewards and AMP feature contract."""
import math
import torch

TEMPLATE = 'support_lift_v1'
SUPPORT, LIFT = 6, 7
# Native 129-D AMP: root13 + joint6D72 + velocities32 + key positions12.
# Shoulder/elbow rotations+velocities and both hand positions are excluded.
ARM_FEATURES = tuple(range(25,49))+tuple(range(91,99))+tuple(range(117,123))


def mask_lift_amp(obs):
    if obs.shape[-1] != 129:
        raise ValueError('Masked lift requires native 129-D AMP features')
    mask = obs.new_ones(129)
    mask[list(ARM_FEATURES)] = 0
    return obs * mask


def relation_matrix(agents, objects, device=None):
    if agents != 2 or objects != 2:
        raise ValueError('Independent lift requires 2 agents and 2 boxes')
    matrix=torch.zeros(6,6,dtype=torch.long,device=device)
    matrix.fill_diagonal_(1)
    matrix[0,1]=matrix[1,0]=2
    for a in range(2):
        o,g=2+a,4+a
        matrix[a,o]=matrix[o,a]=SUPPORT
        matrix[o,g]=matrix[g,o]=LIFT
        matrix[a,g]=matrix[g,a]=4
    return matrix


def inverse_rotate(q, v):
    # Unit xyzw quaternion, with arbitrary leading batch dimensions.
    xyz=-q[...,:3]
    t=2*torch.cross(xyz,v,dim=-1)
    return v+q[...,3:]*t+torch.cross(xyz,t,dim=-1)


def support_geometry(hands, elbows, box, size, forces, config):
    """Four bottom-surface contact proxies in box-local coordinates.

    Ordering: right hand, left hand, left forearm, right forearm. Forearm
    capsule distal axis endpoint uses the actual asset's local segment.
    forces are magnitudes ordered as the corresponding four body proxies.
    Net force + proximity is a contact proxy, not a measured pairwise force.
    """
    centre=box[...,:3];q=box[...,3:7]
    forearms=elbows+config['forearm_fraction']*(hands-elbows)
    points=torch.stack((hands[...,0,:],hands[...,1,:],forearms[...,1,:],forearms[...,0,:]),-2)
    local=inverse_rotate(q[...,None,:].expand(*points.shape[:-1],4),points-centre[...,None,:])
    # Closest underside-facing sphere/capsule surface, radius=.04m in asset.
    local=local.clone();local[...,2]+=config['contact_radius']
    half=size/2
    xy_excess=(local[...,:2].abs()-half[...,None,:2]+config['inside_margin']).clamp_min(0)
    z_error=local[...,2]+half[...,None,2]
    surface_error=torch.sqrt(xy_excess.square().sum(-1)+z_error.square())
    proximity=torch.exp(-config['surface_scale']*surface_error.square().amax(-1))
    xy=local[...,:2]
    polygon_centre=xy.mean(-2)
    centring=torch.exp(-config['centering_scale']*polygon_centre.square().sum(-1))
    nxt=xy.roll(-1,-2)
    edges=nxt-xy
    cross=edges[...,0]*(-xy[...,1])-edges[...,1]*(-xy[...,0])
    # Convex winding + nonzero margins. A degenerate/crossed quadrilateral
    # cannot earn a support score merely by averaging points at the COM.
    edge_norm=edges.norm(dim=-1).clamp_min(1e-6)
    signed=cross/edge_norm
    inside_margin=torch.maximum(signed.amin(-1),(-signed).amin(-1))
    coverage=(inside_margin/config['com_margin']).clamp(0,1)
    area=.5*torch.abs((xy[...,0]*nxt[...,1]-xy[...,1]*nxt[...,0]).sum(-1))
    area_score=(area/config['min_area']).clamp(0,1)
    contact=((forces/config['contact_force']).clamp(0,1)*
             (surface_error<=config['contact_distance']).to(forces.dtype)).amin(-1)
    state=(proximity*centring*coverage*area_score*contact).clamp(0,1)
    return dict(state=state,contact=contact,proximity=proximity,centring=centring,
                coverage=coverage,area=area,points_local=local,surface_error=surface_error)


def lift_scores(roots, hands, elbows, box, size, forces, targets, initial_rot, cfg):
    s=cfg['support'];l=cfg['lift']
    support=support_geometry(hands,elbows,box,size,forces,s)
    raw=support['state']
    support_success=raw>s['success_threshold']
    support_gate=torch.where(support_success,torch.ones_like(raw),raw)
    root_xy=(roots[...,:2]-box[...,:2]).norm(dim=-1)
    # Exactly the HOLDING root-XY distance potential, not its temporal delta.
    support_progress=1/(1+(root_xy-s['progress_delta']).clamp_min(0)/s['progress_sigma'])
    support_progress=torch.where(support_success,torch.ones_like(raw),support_progress)
    # Quaternion geodesic relative to reset orientation (yaw included).
    dot=(box[...,3:7]*initial_rot).sum(-1).abs().clamp(0,1)
    # atan2 avoids acos' ill-conditioned derivative near identity.
    theta=2*torch.atan2(torch.sqrt((1-dot.square()).clamp_min(0)),dot)
    omega=box[...,10:13].norm(dim=-1)
    angle_score=torch.exp(-(theta/l['angle_scale_rad']).square()-(omega/l['angular_speed_scale']).square())
    distance=(box[...,:3]-targets).norm(dim=-1)
    gate=support_gate*angle_score
    lift_progress=gate/(1+distance/l['progress_sigma'])
    lift_state=gate*torch.exp(-l['fine_distance_scale']*distance.square())
    hold_valid=(distance<=l['success_distance']) & support_success & (theta<=l['success_angle_rad']) \
        & (omega<=l['success_angular_speed']) & (box[...,7:10].norm(dim=-1)<=l['success_linear_speed'])
    return dict(support_state_raw=raw,support_state=support_gate,support_progress=support_progress,
        support_success=support_success,lift_state=lift_state,lift_progress=lift_progress,
        angle_score=angle_score,angle_rad=theta,distance=distance,hold_valid=hold_valid,
        support_contact=support['contact'],support_area=support['area'])


def reward_step(scores, hold_time, achieved, dt, cfg):
    elapsed=torch.where(scores['hold_valid'],hold_time+dt,torch.zeros_like(hold_time))
    success=elapsed>=cfg['lift']['hold_seconds']
    bonus=success & ~achieved
    w=cfg['weights']
    terms=torch.stack((w['support_state']*scores['support_state'],
        w['support_progress']*scores['support_progress'],
        w['support_success']*scores['support_success'].float(),
        w['lift_state']*scores['lift_state'],w['lift_progress']*scores['lift_progress'],
        w['lift_success']*bonus.float()),-1)
    return terms,elapsed,achieved|success,success


def validate_config(cfg):
    if cfg['target_height']<=cfg['stand_height']+cfg['box_size'][2]/2:
        raise ValueError('Lift goal must be above starting box centre')
    if not 0<cfg['support']['success_threshold']<1 or cfg['lift']['hold_seconds']<=0:
        raise ValueError('Invalid lift success conditions')
    if cfg['lane_spacing']<2.5 or min(cfg['start_distance'])<1:
        raise ValueError('Insufficient approach/agent clearance')
    for group in ('support','lift','weights'):
        for name,value in cfg[group].items():
            if isinstance(value,(int,float)) and (not math.isfinite(value) or value<=0):
                raise ValueError('Expected positive '+group+'.'+name)
