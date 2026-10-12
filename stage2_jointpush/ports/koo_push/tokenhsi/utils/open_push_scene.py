"""Random physical PUSH scenes with unchanged cube physics and local RSI geometry."""
import math
import torch
from utils.door_reference import rotate


def validate_scene(env):
    spec=env.get('pushSceneRandomization')
    if spec is None:return
    if set(spec)!={'arena_center','arena_radius','min_station_distance','owner_loco_distance_range'}:
        raise ValueError('Invalid PUSH random scene fields')
    numbers=spec['arena_center']+[spec['arena_radius'],spec['min_station_distance']]+spec['owner_loco_distance_range']
    if len(spec['arena_center'])!=2 or len(spec['owner_loco_distance_range'])!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in numbers):
        raise ValueError('Invalid PUSH scene numbers')
    lo,hi=spec['owner_loco_distance_range'];radius=spec['arena_radius'];separation=spec['min_station_distance']
    if not 0<lo<hi or not 0<separation<2*radius or env.get('openPushRandomizeLanes') is not True:
        raise ValueError('PUSH scenes require random binding and feasible positive ranges')
    # Circular workspaces cover arbitrary box heading, approach and 2m swept path.
    goal=env.get('pushGoalRandomization')
    if goal is not None:
        if set(goal)!={'half_extent','min_distance'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in goal.values()) or not goal['min_distance']<goal['half_extent']:
            raise ValueError('Invalid PUSH random goal square/range')
    travel=math.sqrt(2)*goal['half_extent'] if goal else env['relationReward']['push']['target_distance']
    extent=max(hi+.5,travel+math.hypot(*env['pushBox']['size'][:2])/2)
    if separation<2*extent:
        raise ValueError('PUSH workspaces overlap')
    layout=env['openPushLayout'];cx,cy=spec['arena_center']
    for y in (-layout['lane_spacing']/2,layout['lane_spacing']/2):
        if math.hypot(cx-layout['door_x'],cy-y)<radius+extent+1.:
            raise ValueError('PUSH random workspace approaches a fixed door')
    if max(abs(cx),abs(cy))+radius+extent>env['envSpacing']:
        raise ValueError('PUSH random workspace exceeds environment extent')


def sample_boxes(count,device,spec,size):
    positions=torch.empty(count,2,2,device=device)
    pending=torch.ones(count,dtype=torch.bool,device=device)
    for _ in range(128):
        n=int(pending.sum())
        if not n:break
        angle=torch.rand(n,2,device=device)*2*math.pi
        radius=torch.rand(n,2,device=device).sqrt()*spec['arena_radius']
        xy=torch.stack((angle.cos(),angle.sin()),-1)*radius[...,None]
        xy+=xy.new_tensor(spec['arena_center'])
        positions[pending]=xy
        pending[pending.clone()]=(xy[:,0]-xy[:,1]).norm(dim=-1)<spec['min_station_distance']
    if pending.any():raise RuntimeError('Cannot sample disjoint PUSH workspaces')
    states=torch.zeros(count,2,13,device=device);states[...,:2]=positions;states[...,2]=size[2]/2
    yaw=torch.rand(count,2,device=device)*2*math.pi
    states[...,5]=(yaw/2).sin();states[...,6]=(yaw/2).cos()
    direction=rotate(states[...,3:7],states.new_tensor([0.,1.,0.]).expand(count,2,3))
    return states,direction


def sample_goal_offsets(boxes,spec):
    """Ksh-style uniform XY square, rejecting goals near the initial box."""
    offsets=torch.zeros_like(boxes[...,:3]);pending=torch.ones(boxes.shape[:2],dtype=torch.bool,device=boxes.device)
    for _ in range(64):
        n=int(pending.sum())
        if not n:break
        xy=(torch.rand(n,2,device=boxes.device)*2-1)*spec['half_extent']
        offsets[pending]=torch.cat((xy,torch.zeros_like(xy[:,:1])),-1)
        pending[pending.clone()]=xy.norm(dim=-1)<spec['min_distance']
    if pending.any():raise RuntimeError('Cannot sample PUSH goals outside box exclusion')
    return offsets


def sample_loco_positions(boxes,relative_body,root_height,size,distance_range,goals=None,min_goal_distance=1.):
    """Uniform bearings/ranges, rejecting initial body-center penetration."""
    count=len(boxes);target=boxes[:,:3].clone();target[:,2]=root_height
    pending=torch.ones(count,dtype=torch.bool,device=boxes.device)
    inv=boxes[:,3:7].clone();inv[:,:3]*=-1
    half=boxes.new_tensor(size)/2
    lo,hi=distance_range
    for _ in range(64):
        n=int(pending.sum())
        if not n:break
        angle=torch.rand(n,device=boxes.device)*2*math.pi
        distance=lo+(hi-lo)*torch.rand(n,device=boxes.device)
        xy=boxes[pending,:2]+torch.stack((angle.cos(),angle.sin()),-1)*distance[:,None]
        target[pending,:2]=xy
        body=relative_body[pending]+target[pending,None]
        local=rotate(inv[pending,None].expand(-1,body.shape[1],-1),body-boxes[pending,None,:3])
        penetration=(local.abs()<(half+.12)).all(-1).any(-1)
        if goals is not None:
            penetration|=(xy-goals[pending,:2]).norm(dim=-1)<min_goal_distance
        pending[pending.clone()]=penetration
    if pending.any():raise RuntimeError('Cannot place PUSH locomotion bodies without penetration')
    return target
