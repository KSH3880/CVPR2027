"""Grounded box pushing and existing continuous door support in one runtime."""
import torch
from utils.edge_ontop_spec import expand_graph, copy_graph_rows
from utils.edge_stage1_spec import semantic_graph_packet
from utils.open_push_spec import compile_graph, graph_for_tasks
from utils.door_hold import update_hold
from env.tasks.multi_agent.edge_context_reward import EdgeContextRuntime
from env.tasks.multi_agent.edge_door_reward import door_reward
from utils.edge_door_spec import compile_graph as door_graph



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


def push_any_surface_geometry(hands,boxes,size):
    """Distance and outward normal of the nearest point on the full box surface."""
    from utils.door_reference import rotate
    from utils.open_push_sizes import size_tensor
    q=boxes[...,3:7]
    inverse=q.clone();inverse[...,:3]*=-1
    local=rotate(inverse[:,:,None].expand(*hands.shape[:-1],4),hands-boxes[:,:,None,:3])
    half=size_tensor(boxes,size)[:,:,None]/2
    point=torch.maximum(torch.minimum(local,half),-half)
    inside=(local.abs()<=half).all(-1)
    axis=(half-local.abs()).argmin(-1,keepdim=True)
    sign=torch.where(local.gather(-1,axis)>=0,1.,-1.)
    surface=point.scatter(-1,axis,sign*half.expand_as(local).gather(-1,axis))
    point=torch.where(inside[...,None],surface,point)
    delta=local-point
    normal=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    face_normal=torch.zeros_like(local).scatter(-1,axis,sign)
    normal=torch.where((inside | (delta.norm(dim=-1)<1e-8))[...,None],face_normal,normal)
    return delta.norm(dim=-1),rotate(q[:,:,None].expand(*hands.shape[:-1],4),normal)


def push_face_geometry(boxes,size,goals=None,hands=None,face_side=None):
    """Face opposite the goal: dominant axis for RSI, nearest eligible face for hands.

    Legacy mode keeps local -Y. Goal mode uses the same yaw-frame geometry as
    the contact reward, with a horizontal inward normal toward the goal.
    face_side freezes the reset-selected local face; its yaw and position still
    follow the box. Height remains unrestricted by XY contact geometry.
    """
    from utils.door_reference import rotate,yaw_from_quat
    if goals is None:
        normal=rotate(boxes[...,3:7],boxes.new_tensor([0.,1.,0.]).expand_as(boxes[...,:3]))
        tangent=rotate(boxes[...,3:7],boxes.new_tensor([1.,0.,0.]).expand_as(boxes[...,:3]))
        center=boxes[...,:3]-normal*size[1]/2
        return center,normal,tangent,boxes.new_full(boxes.shape[:2],size[0]/2),torch.full(boxes.shape[:2],2,device=boxes.device,dtype=torch.long)
    from utils.open_push_sizes import size_tensor
    size=size_tensor(boxes,size)
    yaw=yaw_from_quat(boxes[...,3:7]);c,s=yaw.cos(),yaw.sin()
    delta=goals[...,:2]-boxes[...,:2]
    x=c*delta[...,0]+s*delta[...,1];y=-s*delta[...,0]+c*delta[...,1]
    axis_x=x.abs()>y.abs()
    component=torch.where(axis_x,x,y)
    sign=torch.where(component>=0,1.,-1.)
    if face_side is not None:
        # Material face identity is chosen at goal creation, never from hands.
        axis_x=face_side<2
        sign=torch.where(face_side.remainder(2)==0,1.,-1.)
    nx=torch.where(axis_x,sign,0.);ny=torch.where(axis_x,0.,sign)
    normal=torch.stack((c*nx-s*ny,s*nx+c*ny,torch.zeros_like(x)),-1)
    tangent=torch.stack((normal[...,1],-normal[...,0],torch.zeros_like(x)),-1)
    half=torch.where(axis_x,size[...,0]/2,size[...,1]/2)
    lateral_half=torch.where(axis_x,size[...,1]/2,size[...,0]/2)
    center=boxes[...,:3]-normal*half[...,None]
    side=torch.where(axis_x,0,2)+(sign<0).long()
    if hands is not None and face_side is None:
        # Adjacent opposite faces are both valid near a diagonal goal. Picking
        # the nearest valid face avoids a 90deg reward jump at the dominant-axis
        # boundary while excluding faces more than60deg from the goal direction.
        local=boxes.new_tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        cx=c[...,None]*local[:,0]-s[...,None]*local[:,1]
        cy=s[...,None]*local[:,0]+c[...,None]*local[:,1]
        normals=torch.stack((cx,cy,torch.zeros_like(cx)),-1)
        tangents=torch.stack((cy,-cx,torch.zeros_like(cx)),-1)
        halves=size[...,[0,0,1,1]]/2
        lateral=size[...,[1,1,0,0]]/2
        centers=boxes[:,:,None,:3]-normals*halves[...,None]
        delta_h=hands[:,:,None,:,:2]-centers[:,:,:,None,:2]
        a=(delta_h*tangents[:,:,:,None,:2]).sum(-1)
        b=(delta_h*normals[:,:,:,None,:2]).sum(-1)
        cost=torch.stack((a-a.clamp(-lateral[...,None],lateral[...,None]),b),-1).norm(dim=-1).amax(-1)
        length=delta.norm(dim=-1)
        eligible=(normals[...,:2]*delta[:,:,None]).sum(-1)>=.5*length[...,None]
        eligible=torch.where((length<1e-6)[...,None],torch.arange(4,device=boxes.device)[None,None]==side[...,None],eligible)
        side=cost.masked_fill(~eligible,float('inf')).argmin(-1)
        index=side[...,None,None].expand(-1,-1,1,3)
        center=centers.gather(2,index).squeeze(2)
        normal=normals.gather(2,index).squeeze(2)
        tangent=tangents.gather(2,index).squeeze(2)
        lateral_half=lateral.gather(-1,side[...,None]).squeeze(-1)
    return center,normal,tangent,lateral_half,side


def push_hand_targets(boxes, size, spec, goals=None,hands=None,face_side=None):
    """Right/left targets on the rear face, rotated with the box."""
    from utils.door_reference import rotate
    if goals is not None:
        center,normal,tangent,_,_=push_face_geometry(boxes,size,goals,hands,face_side)
        target=center[:,:,None]+tangent[:,:,None]*boxes.new_tensor([1.,-1.])[None,None,:,None]*spec['hand_lateral_offset']
        from utils.open_push_sizes import size_tensor
        height=torch.as_tensor(spec['hand_target_height'],device=boxes.device,dtype=boxes.dtype)
        if height.ndim==1:height=height[:,None]
        height=height.expand(boxes.shape[:2])[...,None]
        target[...,2]=boxes[...,None,2]+height-size_tensor(boxes,size)[...,None,2]/2
        return target
    offsets=boxes.new_tensor([[spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2],
                             [-spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2]])
    return boxes[...,:3][:,:,None]+rotate(boxes[...,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        offsets.expand(*boxes.shape[:2],2,3))


def push_hand_point_distance(hands, boxes, size, spec, goals=None,face_side=None):
    return (hands-push_hand_targets(boxes,size,spec,goals,hands,face_side)).norm(dim=-1)


def push_rear_xy(hands, boxes, size, goals=None,face_side=None):
    """Yaw-frame rear face geometry; height and hand assignment are unrestricted."""
    if goals is not None:
        center,normal,tangent,half,_=push_face_geometry(boxes,size,goals,hands,face_side)
        delta=hands[...,:2]-center[:,:,None,:2]
        along=(delta*tangent[:,:,None,:2]).sum(-1)
        across=(delta*normal[:,:,None,:2]).sum(-1)
        error=torch.stack((along-along.clamp(-half[:,:,None],half[:,:,None]),across),-1)
        return delta.mean(-2).norm(dim=-1),error.norm(dim=-1)
    from utils.door_reference import yaw_from_quat
    yaw = yaw_from_quat(boxes[..., 3:7])
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    delta = hands[..., :2] - boxes[..., None, :2]
    x = c * delta[..., 0] + s * delta[..., 1]
    y = -s * delta[..., 0] + c * delta[..., 1]
    local = torch.stack((x, y), -1)
    midpoint_error = local.mean(-2) - local.new_tensor([0., -size[1]/2])
    surface_error = torch.stack((x - x.clamp(-size[0]/2, size[0]/2), y + size[1]/2), -1)
    return midpoint_error.norm(dim=-1), surface_error.norm(dim=-1)


def push_debug_metrics(hands,forces,boxes,box_forces,start,size,spec,contact,eligible,episode_start=None,forward_direction=None,goals=None,reset_states=None,face_side=None):
    """Per-owner physical diagnostics; never modify rewards or contact history."""
    from utils.door_reference import rotate
    target=push_hand_targets(boxes,size,spec,goals,hands,face_side)
    midpoint_distance,distance=push_rear_xy(hands,boxes,size,goals,face_side)
    _,normal,_,_,_=push_face_geometry(boxes,size,goals,hands,face_side)
    reaction=-(forces*normal[:,:,None]).sum(-1)
    if spec.get('contact_mode')=='any_surface':
        distance,surface_normal=push_any_surface_geometry(hands,boxes,size)
        reaction=(forces*surface_normal).sum(-1)
    tilt=torch.acos((1-2*(boxes[...,3].square()+boxes[...,4].square())).clamp(-1,1))
    near=distance<=spec['hand_distance']
    if spec.get("contact_mode")=="fixed_face_midpoint":
        near=near & (midpoint_distance<=spec["hand_distance"])[...,None]
    upright=(tilt<=spec['max_tilt']) & ((boxes[...,2]-start[...,2]).abs()<=spec['ground_tolerance'])
    # Tracking bottom edges removes center motion caused purely by forward rocking.
    legacy_size=[1.3]*3 if torch.is_tensor(size) else size
    edge=boxes.new_tensor([[0.,-legacy_size[1]/2,-legacy_size[2]/2],[0.,legacy_size[1]/2,-legacy_size[2]/2]])
    world=boxes[:,:,:3][:,:,None]+rotate(boxes[:,:,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        edge.expand(*boxes.shape[:2],2,3))
    low=world[...,2].argmin(-1,keepdim=True)
    axis=boxes.new_tensor([0.,1.,0.]).expand_as(start) if forward_direction is None else forward_direction
    edge_y=(world*axis[:,:,None]).sum(-1).gather(-1,low).squeeze(-1)
    initial_y=(start*axis).sum(-1)+edge[:,1][low.squeeze(-1)]
    episode_start=start if episode_start is None else episode_start
    episode_edge_y=(episode_start*axis).sum(-1)+edge[:,1][low.squeeze(-1)]
    bottom_forward=edge_y-initial_y;bottom_episode=edge_y-episode_edge_y
    if goals is not None:
        if reset_states is None:raise ValueError('Goal-face diagnostics require reset quaternion')
        from utils.open_push_sizes import size_tensor
        corner=boxes.new_tensor([[-1.,-1.,-1.],[-1.,1.,-1.],[1.,-1.,-1.],[1.,1.,-1.]])*size_tensor(boxes,size)[:,:,None]/2
        world=boxes[:,:,None,:3]+rotate(boxes[:,:,None,3:7].expand(-1,-1,4,-1),corner.expand(*boxes.shape[:2],4,3))
        initial=rotate(reset_states[:,:,None,3:7].expand(-1,-1,4,-1),corner.expand(*boxes.shape[:2],4,3))
        bottom_forward=((world-(start[:,:,None]+initial))*axis[:,:,None]).sum(-1).amin(-1)
        bottom_episode=((world-(episode_start[:,:,None]+initial))*axis[:,:,None]).sum(-1).amin(-1)
    return dict(midpoint_xy_distance=midpoint_distance,right_height=hands[...,0,2],
        left_height=hands[...,1,2],hand_height_difference=(hands[...,0,2]-hands[...,1,2]).abs(),
        right_distance=distance[...,0],left_distance=distance[...,1],
        worst_hand_distance=distance.amax(-1),right_height_error=hands[...,0,2]-target[...,0,2],
        left_height_error=hands[...,1,2]-target[...,1,2],both_hands_near=near.all(-1).float(),
        right_reaction_N=reaction[...,0],left_reaction_N=reaction[...,1],
        box_forward_force_N=(box_forces*normal).sum(-1),paired_contact=contact.float(),
        eligible=eligible.float(),tilt_rad=tilt,tilt_violation=(tilt>spec['max_tilt']).float(),
        grounded=upright.float(),box_forward_m=((boxes[...,:3]-start)*axis).sum(-1),
        bottom_edge_forward_m=bottom_forward,
        since_reset_forward_m=((boxes[...,:3]-episode_start)*axis).sum(-1),
        since_reset_bottom_edge_forward_m=bottom_episode,
        box_speed=boxes[...,7:10].norm(dim=-1))


def approach_xy_diagnostics(roots, handles, boxes, config):
    """Reward-independent XY approach scores; larger is closer (0..1)."""
    if 'door' not in config:
        distance=(roots[...,:2]-boxes[...,:2]).norm(dim=-1)
        return {'push':dict(distance_xy=distance,xy_progress=1/(1+(distance-.5).clamp_min(0)))}
    door = config['door']
    open_distance = (roots[..., :2] - handles[..., :2]).norm(dim=-1)
    push_distance = (roots[..., :2] - boxes[..., :2]).norm(dim=-1)
    return {
        'open': dict(distance_xy=open_distance, xy_progress=1 / (1 +
            (open_distance - door['approach_delta']).clamp_min(0) / door['approach_sigma'])),
        'push': dict(distance_xy=push_distance, xy_progress=1 / (1 + (push_distance - .5).clamp_min(0))),
    }


def push_hand_contact(hands,forces,boxes,box_forces,size,spec,goals=None,face_side=None):
    if spec.get('contact_mode')=='any_surface':
        distance,normal=push_any_surface_geometry(hands,boxes,size)
        reaction=(forces*normal).sum(-1)
        return ((distance<=spec['hand_distance']) & (reaction>=spec['min_hand_force'])).all(-1)
    if spec.get("contact_mode")=="fixed_face_midpoint" and face_side is None:
        raise ValueError("Fixed face must be initialized at goal creation")
    _,normal,_,_,_=push_face_geometry(boxes,size,goals,hands,face_side)
    midpoint,distance=push_rear_xy(hands,boxes,size,goals,face_side)
    if spec.get("contact_mode")=="fixed_face_midpoint":
        distance=torch.maximum(distance,midpoint[...,None])
    near=distance<=spec['hand_distance']
    reaction=-(forces*normal[:,:,None]).sum(-1)
    push=(box_forces*normal).sum(-1)
    return (near & (reaction>=spec['min_hand_force'])).all(-1) & (push>=spec['min_box_force'])

def push_reward(roots,hands,boxes,goals,start,eligible,config,size=None,contact=None,credited_distance=None,episode_distance=None,goal_contact=False,face_side=None):
    if 'contact_at' in config:
        return contact_at_reward(roots,hands,boxes,goals,config,size,contact,goal_contact,face_side)
    p=config['push'];pos=boxes[...,:3];q=boxes[...,3:7]
    distance=(pos[...,:2]-goals[...,:2]).norm(dim=-1)
    # Box center stays close to its reset ground height, with an upright orientation.
    up_z=1-2*(q[...,0].square()+q[...,1].square())
    grounded=((pos[...,2]-start[...,2]).abs()<=p['ground_tolerance']) & (up_z>=torch.cos(pos.new_tensor(p['max_tilt'])))
    stable=(boxes[...,7:10].norm(dim=-1)<=p['max_speed']) & (boxes[...,10:13].norm(dim=-1)<=p['max_angular_speed'])
    success=(distance<=p['goal_tolerance']) & grounded & stable & eligible
    # Preserve the RSI starting progress, but authenticate all later improvement.
    contact=torch.zeros_like(eligible) if contact is None else contact
    credited_distance=torch.zeros_like(distance) if credited_distance is None else credited_distance
    initial=(start[...,:2]-goals[...,:2]).norm(dim=-1).clamp_min(1e-6)
    reset_remaining=initial if episode_distance is None else episode_distance
    authenticated=(1-reset_remaining/initial+credited_distance/initial).clamp(0,1)
    geometric=(1-distance/initial).clamp(0,1)
    progress=torch.minimum(geometric,authenticated)*grounded*contact
    approach=1/(1+((roots[...,:2]-pos[...,:2]).norm(dim=-1)-.5).clamp_min(0))
    if size is None:
        raise ValueError('Two-hand PUSH requires the actual box size')
    midpoint_distance,_=push_rear_xy(hands,boxes,size,goals if goal_contact else None,face_side)
    interaction=torch.exp(-10*midpoint_distance.square())*grounded
    state=(approach+interaction)*.5*config['state_reward_weight']
    prog=torch.where(success,torch.ones_like(progress),progress)*config['progress_reward_weight']
    bonus=success.float()*config['success_reward_weight']
    return dict(state=state,progress=prog,bonus=bonus,total=state+prog+bonus,success=success,
        phi=progress,error=distance,grounded=grounded)


def contact_at_reward(roots,hands,boxes,goals,config,size,contact,goal_contact=False,face_side=None):
    """Independent rear-face contact and legacy AT; sum both edge rewards."""
    if size is None:
        raise ValueError('Contact PUSH requires the actual box size')
    a=config['contact_at']
    contact=torch.zeros(boxes.shape[:2],dtype=torch.bool,device=boxes.device) if contact is None else contact
    # Both hands must approach the rear face; averaging cannot hide a far hand.
    if config['push'].get('contact_mode')=='any_surface':
        hand_distances,_=push_any_surface_geometry(hands,boxes,size)
    else:
        midpoint,hand_distances=push_rear_xy(hands,boxes,size,goals if goal_contact else None,face_side)
    hand_distance=hand_distances.amax(-1)
    if config["push"].get("contact_mode")=="fixed_face_midpoint":
        if face_side is None:raise ValueError("Fixed face must be initialized at goal creation")
        hand_distance=torch.maximum(hand_distance,midpoint)
    push_phi=torch.exp(-10*hand_distance.square())
    approach=(roots[...,:2]-boxes[...,:2]).norm(dim=-1)
    push_progress=1/(1+(approach-a['progress_delta']).clamp_min(0)/a['progress_sigma'])
    delta=boxes[...,:3]-goals
    distance=delta[...,:2].norm(dim=-1)
    at_phi=torch.exp(-a['near_distance_scale']*delta.square().sum(-1))
    at_progress=1/(1+(distance-a.get('at_progress_delta',a['progress_delta'])).clamp_min(0)/a['progress_sigma'])
    # Judge raw state before success/contact saturation, avoiding self-triggering.
    state_score=.5*(push_phi+at_phi)
    if a.get('success_mode','state_and_height')=='state_only':
        at_success=state_score>=a['satisfaction_threshold']
        if config['push'].get('contact_mode')=='any_surface':
            at_success=at_success & contact
    else:
        at_success=(at_phi>=a['satisfaction_threshold']) & (delta[...,2].abs()<=a['z_tolerance'])
    placement_reached=at_success
    if 'max_tilt' in a:
        up_z=1-2*(boxes[...,3].square()+boxes[...,4].square())
        at_success=at_success & (up_z>=torch.cos(boxes.new_tensor(a['max_tilt'])))
    # Like paired HOLDING+AT, placement saturates the contact edge on release.
    # The optional fade affects AT only, including after arrival. Preserve the
    # original PUSH arrival saturation based on distance/height in this mode.
    fade_enabled='tilt_fade_start' in a
    push_saturated=contact | (placement_reached if fade_enabled else at_success)
    at_state=torch.where(at_success,1.,at_phi)
    at_motion=torch.where(at_success,1.,at_progress)
    if fade_enabled:
        tilt=torch.acos((1-2*(boxes[...,3].square()+boxes[...,4].square())).clamp(-1,1))
        factor=((a['tilt_fade_end']-tilt)/(a['tilt_fade_end']-a['tilt_fade_start'])).clamp(0,1)
        at_state=at_state*factor
        at_motion=at_motion*factor
    state=.5*config['state_reward_weight']*(torch.where(push_saturated,1.,push_phi)+at_state)
    progress=config['progress_reward_weight']*(torch.where(push_saturated,1.,push_progress)+at_motion)
    bonus=config['success_reward_weight']*(push_saturated.float()+at_success.float())
    return dict(state=state,progress=progress,bonus=bonus,total=state+progress+bonus,
        success=at_success,phi=at_phi,state_score=state_score,error=distance,push_phi=push_phi,push_success=contact,
        grounded=delta[...,2].abs()<=a['z_tolerance'])


class OpenPushRuntime(EdgeContextRuntime):
    def __init__(self,n,spec,config,hold,dt,device,size,goal_contact=False):
        super().__init__(n,expand_graph(compile_graph(spec,2,4,device),n),config,device)
        self.spec,self.hold,self.dt,self.size=spec,hold,dt,size
        self.goal_contact=goal_contact
        self.face_side=None
        self.tasks=torch.zeros(n,2,dtype=torch.long,device=device)
        self.elapsed=torch.zeros(n,2,device=device)
        self.contact_age=torch.full_like(self.elapsed,hold['contact_grace']+dt)
        self.contact=torch.zeros(n,2,dtype=torch.bool,device=device)
        self.posture=torch.zeros_like(self.contact)
        self.push_contact=torch.zeros_like(self.contact)
        self.push_eligible=torch.zeros_like(self.contact)
        self.open_credit=torch.zeros_like(self.elapsed)
        self.open_start_angle=torch.zeros_like(self.elapsed)
        self.open_previous_angle=torch.zeros_like(self.elapsed)
        self.open_eligible=torch.zeros_like(self.contact)
        self.push_credit=torch.zeros_like(self.elapsed)
        self.push_reset_distance=torch.full_like(self.elapsed,config['push']['target_distance'])
        self.push_previous_distance=self.push_reset_distance.clone()
        self.start=torch.zeros(n,2,3,device=device)
        self.error=torch.zeros(n,2,device=device)

    def reset(self,ids,tasks,boxes):
        self.tasks[ids]=tasks
        copy_graph_rows(self.graph,ids,graph_for_tasks(self.spec,tasks))
        for x in (self.phi,self.own_success,self.achieved,self.done,self.elapsed,self.contact,self.posture,self.push_contact,self.push_eligible,self.push_credit,self.open_credit,self.open_eligible,self.open_start_angle,self.open_previous_angle,self.error):x[ids]=0
        self.contact_age[ids]=self.hold['contact_grace']+self.dt
        self.start[ids]=boxes[...,:3]
        self.push_reset_distance[ids]=self.config['push']['target_distance']
        self.push_previous_distance[ids]=self.push_reset_distance[ids]

    def reset_open_credit(self,ids,angles):
        self.open_start_angle[ids]=angles
        self.open_previous_angle[ids]=angles
        self.open_credit[ids]=0
        self.open_eligible[ids]=False
        self.elapsed[ids]=0
        self.contact_age[ids]=self.hold['contact_grace']+self.dt
        self.contact[ids]=False

    def accumulate_open_credit(self,angles,contact,posture):
        delta=angles-self.open_previous_angle
        valid=contact & posture & (self.tasks==0)
        increment=torch.where(delta<0,delta,delta*valid)
        self.open_credit.copy_(torch.minimum((self.open_credit+increment).clamp_min(0),
            (angles-self.open_start_angle).clamp_min(0)))
        self.open_previous_angle.copy_(angles)
        threshold=self.config['door']['target_angle']-self.config['door']['angle_tolerance']
        required=(threshold-self.open_start_angle).clamp_min(1e-6)
        self.open_eligible.copy_((self.open_credit>=required-1e-6) & (self.open_credit>0) & (self.tasks==0))

    def reset_push_credit(self,ids,boxes,goals):
        """Call after RSI placement; pre-positioned boxes earn no credit."""
        distance=(boxes[...,:2]-goals[...,:2]).norm(dim=-1)
        self.push_reset_distance[ids]=distance
        self.push_previous_distance[ids]=distance
        self.push_credit[ids]=0
        self.push_eligible[ids]=False

    def accumulate_push_credit(self,boxes,goals,contact):
        """Authenticate displacement in the same physics tick as both-hand contact."""
        p=self.config['push']
        current=(boxes[...,:2]-goals[...,:2]).norm(dim=-1)
        q=boxes[...,3:7]
        grounded=((boxes[...,2]-self.start[...,2]).abs()<=p['ground_tolerance']) & (
            1-2*(q[...,0].square()+q[...,1].square())>=torch.cos(boxes.new_tensor(p['max_tilt'])))
        delta=self.push_previous_distance-current
        valid=contact & grounded & (self.tasks==1)
        # Retreat removes credit even without contact, preventing oscillation farming.
        credited_delta=torch.where(delta<0,delta,delta*valid)
        net_advance=(self.push_reset_distance-current).clamp_min(0)
        self.push_credit.copy_(torch.minimum((self.push_credit+credited_delta).clamp_min(0),net_advance))
        self.push_previous_distance.copy_(current)
        required=(self.push_reset_distance-p['goal_tolerance']).clamp_min(p['min_contact_progress'])
        self.push_eligible.copy_((self.push_credit>=required-1e-6) & (self.tasks==1))

    def step(self,roots,hands,handles,angles,boxes,goals):
        self.elapsed[:],opened=update_hold(self.elapsed,angles,self.contact & self.posture & self.open_eligible,
            self.config['door']['target_angle']-self.config['door']['angle_tolerance'],self.hold['duration'],self.dt)
        d=door_reward(roots,hands,handles,angles,
            door_graph({'sampler':'two_agent_two_door_open','semantic_only':True,'bindings':[0,1]},2,2,roots.device),self.config,success_override=torch.zeros_like(opened))
        d['interaction_component']*=(.5+.5*self.contact.float())*self.posture.float()
        # RSI baseline is rewarded; body-only opening above it earns no extra progress.
        authenticated_angle=torch.minimum(angles,self.open_start_angle+self.open_credit)
        door_progress=.5*(authenticated_angle/self.config['door']['target_angle']).clamp(0,1)*self.contact*self.posture+.5*self.elapsed/self.hold['duration']
        ds=d['approach_component']+d['interaction_component']
        dp=torch.where(opened,torch.ones_like(angles),door_progress)*self.config['progress_reward_weight']
        db=opened.float()*self.config['success_reward_weight']
        p=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size,
            self.push_contact,self.push_credit,self.push_reset_distance,self.goal_contact,self.face_side)
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
        if 'contact_at' in self.config:
            self.phi[:,::2]=torch.where(active,p['push_phi'],door_progress)
            self.own_success[:,::2]=torch.where(active,p['push_success'],opened)
        self.done.copy_(success);self.achieved|=self.own_success
        self.error.copy_(torch.where(active,p['error'],(self.config['door']['target_angle']-angles).clamp_min(0)))
        return dict(state=state,progress=progress,bonus=bonus,reward=reward,success=success)

    def suffix(self,ids=None):
        from utils.edge_ontop_spec import select_graph
        g=self.graph if ids is None else select_graph(self.graph,ids)
        return semantic_graph_packet(g,self.phi.shape[0] if ids is None else len(ids))


class PushOnlyRuntime(OpenPushRuntime):
    """PUSH/AT only: no OPEN reward, posture, hold or hinge state."""
    def __init__(self,n,spec,config,dt,device,size,goal_contact):
        EdgeContextRuntime.__init__(self,n,expand_graph(compile_graph(spec,2,4,device),n),config,device)
        self.spec,self.dt,self.size,self.goal_contact=spec,dt,size,goal_contact
        self.face_side=None
        self.tasks=torch.ones(n,2,dtype=torch.long,device=device)
        self.push_contact=torch.zeros(n,2,dtype=torch.bool,device=device)
        self.push_eligible=torch.zeros_like(self.push_contact)
        self.push_credit=torch.zeros(n,2,device=device)
        self.push_reset_distance=torch.full_like(self.push_credit,config['push']['target_distance'])
        self.push_previous_distance=self.push_reset_distance.clone()
        self.start=torch.zeros(n,2,3,device=device)
        self.error=torch.zeros(n,2,device=device)

    def reset(self,ids,tasks,boxes):
        if not (tasks==1).all():raise ValueError('PUSH-only runtime cannot run another task')
        self.tasks[ids]=tasks
        copy_graph_rows(self.graph,ids,graph_for_tasks(self.spec,tasks))
        for value in (self.phi,self.own_success,self.achieved,self.done,self.push_contact,self.push_eligible,self.push_credit,self.error):value[ids]=0
        self.start[ids]=boxes[...,:3]

    def step(self,roots,hands,handles,angles,boxes,goals):
        p=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size,
            self.push_contact,self.push_credit,self.push_reset_distance,self.goal_contact,self.face_side)
        self.phi[:,::2]=p['push_phi'];self.phi[:,1::2]=p['phi']
        self.own_success[:,::2]=p['push_success'];self.own_success[:,1::2]=p['success']
        self.done.copy_(p['success']);self.achieved|=self.own_success;self.error.copy_(p['error'])
        return dict(state=p['state'],progress=p['progress'],bonus=p['bonus'],
            reward=.9*p['total']+.1*p['total'].flip(-1),success=p['success'])
