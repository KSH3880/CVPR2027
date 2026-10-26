"""Fixed-size owner-relative PUSH layout and yaw-consistent episode geometry."""
import math
import torch


def validate_scene(env):
    spec=env.get('pushScene')
    if spec is None:return
    random_goal=spec.get('mode')=='owner_loco_random_at'
    expected={'mode','owner_anchor_xy','path_separation'}|({'random_at_goal'} if random_goal else set())
    if set(spec)!=expected or spec['mode'] not in ('owner_loco_random','owner_loco_random_at'):
        raise ValueError('Invalid randomized PUSH scene')
    if random_goal:
        goal=spec['random_at_goal']
        if not env['relationGraph'].get('push_only') or not env['relationReward'].get('edge_aggregation'):
            raise ValueError('Random AT goals require dedicated PUSH+AT')
        radial=goal.get('sampling') in ('box_radial','box_forward')
        fields={'sampling','min_distance','max_distance','agent_clearance'} if radial else {'reach','min_distance','agent_clearance'}
        if goal.get('sampling') == 'stage1_square_cone':
            fields = {'sampling', 'reach', 'min_distance', 'agent_clearance', 'half_angle_degrees'}
        numbers={k:v for k,v in goal.items() if k!='sampling'}
        if set(goal)!=fields or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in numbers.values()):
            raise ValueError('Invalid random AT goal bounds')
        if goal['min_distance']>=goal['max_distance' if radial else 'reach'] or goal['agent_clearance']<.5:
            raise ValueError('Random AT goals need usable bounds and humanoid clearance')
    if env['relationGraph']['task_probabilities']!={'OPEN':0.,'PUSH':1.}:
        raise ValueError('Owner-relative PUSH scenes require PUSH-only task sampling')
    reset=env['box']['reset']
    if reset['randomRot'] is not True or reset['randomAssignment'] is not True:
        raise ValueError('Random PUSH scenes require random rotation and assignment')
    bounds=reset['ownerLocoDistanceRange']
    if len(bounds)!=2 or not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in bounds) or not 0<bounds[0]<bounds[1]:
        raise ValueError('Invalid owner-relative distance range')
    anchors=torch.as_tensor(spec['owner_anchor_xy'])
    if anchors.shape!=(2,2) or not torch.isfinite(anchors).all():
        raise ValueError('Two finite owner anchors required')
    separation=spec['path_separation']
    diameter=math.hypot(*env['pushBox']['size'][:2])
    if isinstance(separation,bool) or not isinstance(separation,(float,int)) or not math.isfinite(separation) or separation<diameter+.3:
        raise ValueError('Path separation must include both box footprints')
    if not math.isclose(env.get('zoneAssignment',{}).get('swap_probability',0),.5):
        raise ValueError('Two physical PUSH objects must have unbiased owner assignment')


def forward_xy(quaternions):
    from utils.door_reference import yaw_from_quat
    yaw=yaw_from_quat(quaternions)
    return torch.stack((-torch.sin(yaw),torch.cos(yaw)),-1)


def layout_valid(positions,yaw,anchors,door_centers,size,distance,separation,remove_doors=False):
    """Conservative swept footprints prevent initial and intended-path overlap."""
    direction=torch.stack((-torch.sin(yaw),torch.cos(yaw)),-1)
    # Include space behind each box for RSI bodies as well as the whole goal path.
    t=torch.linspace(-1.5,distance,17,device=positions.device)
    path=positions[:,:,None]+direction[:,:,None]*t[None,None,:,None]
    gap=(path[:,0,:,None]-path[:,1,None,:]).norm(dim=-1).amin(dim=(1,2))
    cross=gap>=separation
    # Doors stay at their physical positions. Their frames/panel/handle fit this disc.
    clear_doors=torch.ones(len(positions),device=positions.device,dtype=torch.bool)
    if not remove_doors:
        door_gap=(path[:,:,:,None]-door_centers[:,None,None]).norm(dim=-1).amin(dim=(1,2,3))
        clear_doors=door_gap>=math.hypot(size[0]/2,size[1]/2)+1.1+.15
    delta=anchors-positions;c,s=torch.cos(yaw),torch.sin(yaw)
    local=torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)
    outside=(local.abs()-local.new_tensor(size[:2])/2).clamp_min(0).norm(dim=-1)
    clear_owner=(outside>=.3).all(-1)
    return cross & clear_doors & clear_owner


def sample_layout(count,env,device,generator):
    spec=env['pushScene'];anchors=torch.tensor(spec['owner_anchor_xy'],device=device,dtype=torch.float)
    anchors=anchors.expand(count,2,2)
    # Door origins are hinges; use their conservative frame centres.
    doors=anchors.new_tensor([[1.45,-2.5],[1.45,2.5]]).expand(count,2,2)
    positions=torch.empty_like(anchors);yaw=torch.empty(count,2,device=device)
    pending=torch.arange(count,device=device)
    lo,hi=env['box']['reset']['ownerLocoDistanceRange']
    for _ in range(128):
        if not len(pending):break
        draw=torch.rand(len(pending),2,3,device=device,generator=generator)
        angle=draw[...,0]*2*math.pi;radius=lo+(hi-lo)*draw[...,1]
        candidate=anchors[pending]+radius[...,None]*torch.stack((torch.cos(angle),torch.sin(angle)),-1)
        rotation=draw[...,2]*2*math.pi
        valid=layout_valid(candidate,rotation,anchors[pending],doors[pending],env['pushBox']['size'],
            0. if spec['mode']=='owner_loco_random_at' else env['relationReward']['push']['target_distance'],spec['path_separation'],
            env['relationGraph'].get('push_only',False))
        positions[pending[valid]]=candidate[valid];yaw[pending[valid]]=rotation[valid]
        pending=pending[~valid]
    if len(pending):raise RuntimeError('Unable to sample collision-separated PUSH paths')
    quaternion=torch.zeros(count,2,4,device=device)
    quaternion[...,2]=torch.sin(yaw/2);quaternion[...,3]=torch.cos(yaw/2)
    return positions,quaternion


def segment_distance(a, b, c, d):
    """Exact XY distance between closed segments, including crossings."""
    def point_distance(p, x, y):
        v=y-x
        t=((p-x)*v).sum(-1)/v.square().sum(-1).clamp_min(1e-12)
        return (p-x-t.clamp(0,1)[...,None]*v).norm(dim=-1)
    def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    ab=b-a;cd=d-c
    s1=cross(ab,c-a);s2=cross(ab,d-a)
    s3=cross(cd,a-c);s4=cross(cd,b-c)
    distance=torch.stack((point_distance(a,c,d),point_distance(b,c,d),
                          point_distance(c,a,b),point_distance(d,a,b)),-1).amin(-1)
    # Collinear disjoint segments must retain their endpoint distance.
    intersect=(s1*s2<=0)&(s3*s4<=0)&(cross(ab,cd).abs()>1e-8)
    return torch.where(intersect,torch.zeros_like(distance),distance)


def random_at_goal_valid(positions,quaternions,goals,env,uniform_rsi=None,distance_range=None):
    spec=env['pushScene'];g=spec['random_at_goal']
    anchors=positions.new_tensor(spec['owner_anchor_xy']).expand_as(positions)
    distance=(goals-positions).norm(dim=-1)
    if g.get('sampling') in ('box_radial','box_forward'):
        near=((goals-anchors).norm(dim=-1)>=1.) & (distance>=g['min_distance']-1e-6)
        within=distance<=g['max_distance']+1e-6
        if g.get('sampling')=='box_forward':
            delta=goals-positions
            forward=forward_xy(quaternions)
            along=(delta*forward).sum(-1)
            sideways=delta-along[...,None]*forward
            within &= (sideways.norm(dim=-1)<=1e-5)&(along>=g['min_distance']-1e-6)
    else:
        near=((goals-anchors).norm(dim=-1)>=g['min_distance']) & (distance>=g['min_distance'])
        within=((goals-anchors).abs()<=g['reach']).all(-1)
        if g.get('sampling') == 'stage1_square_cone':
            # Original Stage1 square is centred on the environment, not each owner.
            within = (goals.abs() <= g['reach']).all(-1)
            direction = (goals-positions)/distance.clamp_min(1e-8)[...,None]
            within &= (direction*forward_xy(quaternions)).sum(-1) >= math.cos(math.radians(g['half_angle_degrees']))-1e-6
    if uniform_rsi is not None:
        lo, hi = distance_range
        direction = (goals-positions)/distance.clamp_min(1e-8)[...,None]
        cone = (direction*forward_xy(quaternions)).sum(-1) >= math.cos(math.radians(g['half_angle_degrees']))-1e-6
        within = torch.where(uniform_rsi, (distance>=lo-1e-5)&(distance<=hi+1e-5)&cone, within)
    # Include the 1.5m RSI/person corridor behind each box and the full AT path.
    direction=forward_xy(quaternions)
    if env['relationReward'].get('push_at',{}).get('hand_reference')=='goal_opposite_surface_xy':
        delta=goals-positions
        direction=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    rear=positions-1.5*direction
    goal_rear=goals-1.5*direction
    starts=torch.stack((rear,positions,rear,goals),2)
    ends=torch.stack((positions,goals,goal_rear,goal_rear),2)
    gap=segment_distance(starts[:,0,:,None],ends[:,0,:,None],starts[:,1,None,:],ends[:,1,None,:]).amin((1,2))
    # Other owner's locomotion start is also protected from each swept box disc.
    other=anchors.flip(1)
    anchor_gap=segment_distance(starts,ends,other[:,:,None],other[:,:,None]).amin((1,2))
    radius=math.hypot(*env['pushBox']['size'][:2])/2
    return near.all(-1)&within.all(-1)&(gap>=spec['path_separation'])&(anchor_gap>=radius+g['agent_clearance'])


def sample_random_at_scene(count,env,device,generator):
    """Owner-square, radial, or local +Y goals with strict joint path rejection."""
    positions=torch.empty(count,2,2,device=device)
    quaternions=torch.empty(count,2,4,device=device)
    goals=torch.empty_like(positions)
    pending=torch.arange(count,device=device)
    spec=env['pushScene'];g=spec['random_at_goal']
    anchors=positions.new_tensor(spec['owner_anchor_xy'])
    # Joint square/cone/path rejection accepts fewer draws than a radial goal.
    attempts = 2048 if g.get('sampling') == 'stage1_square_cone' else 128
    for _ in range(attempts):
        if not len(pending):break
        xy,q=sample_layout(len(pending),env,device,generator)
        if g.get('sampling') in ('box_radial','box_forward'):
            draw=torch.rand(len(pending),2,2,device=device,generator=generator)
            angle=draw[...,0]*2*math.pi
            radius=g['min_distance']+(g['max_distance']-g['min_distance'])*draw[...,1]
            direction=forward_xy(q) if g.get('sampling')=='box_forward' else torch.stack((torch.cos(angle),torch.sin(angle)),-1)
            target=xy+radius[...,None]*direction
        else:
            target=anchors+(2*torch.rand(len(pending),2,2,device=device,generator=generator)-1)*g['reach']
            if g.get('sampling') == 'stage1_square_cone':
                target = target - anchors
        valid=random_at_goal_valid(xy,q,target,env)
        accepted=pending[valid]
        positions[accepted]=xy[valid];quaternions[accepted]=q[valid];goals[accepted]=target[valid]
        pending=pending[~valid]
    if len(pending):raise RuntimeError('Unable to sample separated random AT goals')
    return positions,quaternions,goals
