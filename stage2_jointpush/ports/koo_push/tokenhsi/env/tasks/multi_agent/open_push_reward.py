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


def push_hand_targets(boxes, size, spec):
    """Right/left targets on the rear face, rotated with the box."""
    from utils.door_reference import rotate
    offsets=boxes.new_tensor([[spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2],
                             [-spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2]])
    return boxes[...,:3][:,:,None]+rotate(boxes[...,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        offsets.expand(*boxes.shape[:2],2,3))


def push_hand_point_distance(hands, boxes, size, spec):
    return (hands-push_hand_targets(boxes,size,spec)).norm(dim=-1)


def push_rear_xy(hands, boxes, size):
    """Yaw-frame rear face geometry; height and hand assignment are unrestricted."""
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


def push_debug_metrics(hands,forces,boxes,box_forces,start,size,spec,contact,eligible,episode_start=None,forward_direction=None):
    """Per-owner physical diagnostics; never modify rewards or contact history."""
    from utils.door_reference import rotate
    target=push_hand_targets(boxes,size,spec)
    midpoint_distance,distance=push_rear_xy(hands,boxes,size)
    normal=rotate(boxes[...,3:7],boxes.new_tensor([0.,1.,0.]).expand_as(boxes[...,:3]))
    reaction=-(forces*normal[:,:,None]).sum(-1)
    tilt=torch.acos((1-2*(boxes[...,3].square()+boxes[...,4].square())).clamp(-1,1))
    near=distance<=spec['hand_distance']
    upright=(tilt<=spec['max_tilt']) & ((boxes[...,2]-start[...,2]).abs()<=spec['ground_tolerance'])
    # Tracking bottom edges removes center motion caused purely by forward rocking.
    edge=boxes.new_tensor([[0.,-size[1]/2,-size[2]/2],[0.,size[1]/2,-size[2]/2]])
    world=boxes[:,:,:3][:,:,None]+rotate(boxes[:,:,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        edge.expand(*boxes.shape[:2],2,3))
    low=world[...,2].argmin(-1,keepdim=True)
    axis=boxes.new_tensor([0.,1.,0.]).expand_as(start) if forward_direction is None else forward_direction
    edge_y=(world*axis[:,:,None]).sum(-1).gather(-1,low).squeeze(-1)
    initial_y=(start*axis).sum(-1)+edge[:,1][low.squeeze(-1)]
    episode_start=start if episode_start is None else episode_start
    episode_edge_y=(episode_start*axis).sum(-1)+edge[:,1][low.squeeze(-1)]
    return dict(midpoint_xy_distance=midpoint_distance,right_height=hands[...,0,2],
        left_height=hands[...,1,2],hand_height_difference=(hands[...,0,2]-hands[...,1,2]).abs(),
        right_distance=distance[...,0],left_distance=distance[...,1],
        worst_hand_distance=distance.amax(-1),right_height_error=hands[...,0,2]-target[...,0,2],
        left_height_error=hands[...,1,2]-target[...,1,2],both_hands_near=near.all(-1).float(),
        right_reaction_N=reaction[...,0],left_reaction_N=reaction[...,1],
        box_forward_force_N=(box_forces*normal).sum(-1),paired_contact=contact.float(),
        eligible=eligible.float(),tilt_rad=tilt,tilt_violation=(tilt>spec['max_tilt']).float(),
        grounded=upright.float(),box_forward_m=((boxes[...,:3]-start)*axis).sum(-1),
        bottom_edge_forward_m=edge_y-initial_y,
        since_reset_forward_m=((boxes[...,:3]-episode_start)*axis).sum(-1),
        since_reset_bottom_edge_forward_m=edge_y-episode_edge_y,
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


def push_hand_contact(hands,forces,boxes,box_forces,size,spec):
    from utils.door_reference import rotate
    normal=rotate(boxes[...,3:7],boxes.new_tensor([0.,1.,0.]).expand_as(boxes[...,:3]))
    _,distance=push_rear_xy(hands,boxes,size)
    near=distance<=spec['hand_distance']
    reaction=-(forces*normal[:,:,None]).sum(-1)
    push=(box_forces*normal).sum(-1)
    return (near & (reaction>=spec['min_hand_force'])).all(-1) & (push>=spec['min_box_force'])

def push_reward(roots,hands,boxes,goals,start,eligible,config,size=None,contact=None,credited_distance=None,episode_distance=None):
    if 'contact_at' in config:
        return contact_at_reward(roots,hands,boxes,goals,config,size,contact)
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
    midpoint_distance,_=push_rear_xy(hands,boxes,size)
    interaction=torch.exp(-10*midpoint_distance.square())*grounded
    state=(approach+interaction)*.5*config['state_reward_weight']
    prog=torch.where(success,torch.ones_like(progress),progress)*config['progress_reward_weight']
    bonus=success.float()*config['success_reward_weight']
    return dict(state=state,progress=prog,bonus=bonus,total=state+prog+bonus,success=success,
        phi=progress,error=distance,grounded=grounded)


def contact_at_reward(roots,hands,boxes,goals,config,size,contact):
    """Independent rear-face contact and legacy AT; sum both edge rewards."""
    if size is None:
        raise ValueError('Contact PUSH requires the actual box size')
    a=config['contact_at']
    contact=torch.zeros(boxes.shape[:2],dtype=torch.bool,device=boxes.device) if contact is None else contact
    # Both hands must approach the rear face; averaging cannot hide a far hand.
    _,hand_distances=push_rear_xy(hands,boxes,size)
    hand_distance=hand_distances.amax(-1)
    push_phi=torch.exp(-10*hand_distance.square())
    approach=(roots[...,:2]-boxes[...,:2]).norm(dim=-1)
    push_progress=1/(1+(approach-a['progress_delta']).clamp_min(0)/a['progress_sigma'])
    delta=boxes[...,:3]-goals
    distance=delta[...,:2].norm(dim=-1)
    at_phi=torch.exp(-a['near_distance_scale']*delta.square().sum(-1))
    at_progress=1/(1+(distance-a['progress_delta']).clamp_min(0)/a['progress_sigma'])
    at_success=(at_phi>=a['satisfaction_threshold']) & (delta[...,2].abs()<=a['z_tolerance'])
    # Like paired HOLDING+AT, placement saturates the contact edge on release.
    push_saturated=contact | at_success
    state=.5*config['state_reward_weight']*(torch.where(push_saturated,1.,push_phi)+torch.where(at_success,1.,at_phi))
    progress=config['progress_reward_weight']*(torch.where(push_saturated,1.,push_progress)+torch.where(at_success,1.,at_progress))
    bonus=config['success_reward_weight']*(push_saturated.float()+at_success.float())
    return dict(state=state,progress=progress,bonus=bonus,total=state+progress+bonus,
        success=at_success,phi=at_phi,error=distance,push_phi=push_phi,push_success=contact,
        grounded=delta[...,2].abs()<=a['z_tolerance'])


class OpenPushRuntime(EdgeContextRuntime):
    def __init__(self,n,spec,config,hold,dt,device,size,num_agents=2):
        super().__init__(n,expand_graph(compile_graph(spec,num_agents,4,device),n),config,device)
        self.spec,self.hold,self.dt,self.size=spec,hold,dt,size
        self.tasks=torch.zeros(n,num_agents,dtype=torch.long,device=device)
        self.elapsed=torch.zeros(n,num_agents,device=device)
        self.contact_age=torch.full_like(self.elapsed,hold['contact_grace']+dt)
        self.contact=torch.zeros(n,num_agents,dtype=torch.bool,device=device)
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
        self.start=torch.zeros(n,num_agents,3,device=device)
        self.error=torch.zeros(n,num_agents,device=device)

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
            self.push_contact,self.push_credit,self.push_reset_distance)
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
