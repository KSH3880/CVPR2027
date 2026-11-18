"""Solo PUSH corridors in a BEFORE scene with arbitrary human/object banks."""
import math
import torch
from utils.jointpush_scene import radial_xy, corridors, segment_distance, goal_angles_valid


class PushLayoutError(RuntimeError):
    def __init__(self, boxes, goals, pending):
        super().__init__('BEFORE solo PUSH corridor sampler exhausted')
        self.boxes, self.goals, self.pending = boxes, goals, pending


def paths_valid(boxes,sizes,goals,roots,push,primary,active,spec):
    n,m=push.shape
    row=torch.arange(n,device=boxes.device)[:,None]
    payload=boxes[row,primary]
    starts,ends=corridors(payload,goals,spec['rear_clearance'])
    radius=sizes[...,:2].norm(dim=-1)/2
    own_radius=radius.gather(1,primary)
    push_objects=torch.zeros_like(active)
    push_objects.scatter_(1,primary,push)
    valid=torch.ones(n,dtype=torch.bool,device=boxes.device)
    for a in range(m):
        starts_a,ends_a=starts[:,a],ends[:,a]
        gap=segment_distance(starts_a[:,:,None],ends_a[:,:,None],boxes[:,None,:,:2],boxes[:,None,:,:2]).amin(1)
        blocked=(gap < own_radius[:,a,None]+radius+spec['obstacle_clearance']) & active & ~push_objects
        valid &= ~(push[:,a]&blocked.any(-1))
        for b in range(m):
            if a==b:continue
            human_gap=segment_distance(starts_a,ends_a,roots[:,b,None,:2],roots[:,b,None,:2]).amin(-1)
            valid &= ~(push[:,a]&(human_gap < own_radius[:,a]+spec['agent_clearance']))
            if b>a:
                path_gap=segment_distance(starts_a[:,:,None],ends_a[:,:,None],starts[:,b,None],ends[:,b,None]).amin((1,2))
                valid &= ~(push[:,a]&push[:,b]&(path_gap < spec['path_separation']))
    return valid


def sample_layout(boxes,sizes,roots,push,primary,active,spec,max_attempts=4096):
    n,m=push.shape
    rows=torch.arange(n,device=boxes.device)[:,None]
    result=boxes.clone()
    goals=boxes[rows,primary,:3].clone()
    pending=push.any(-1).nonzero().flatten()
    for _ in range(max_attempts):
        if not len(pending):return result,goals
        local=push[pending]
        xy=radial_xy(roots[pending,:,:2],spec['source_distance'])
        target=radial_xy(xy,spec['goal_distance'])
        # Draw yaw conditioned on the same uniform local +Y cone. Avoid the
        # exponentially low joint acceptance of independent yaw for four pushes.
        direction=target-xy
        cone=math.radians(spec['goal_angle_degrees'])
        yaw=torch.atan2(direction[...,1],direction[...,0])-math.pi/2
        yaw += (torch.rand(len(pending),m,device=boxes.device)-.5)*cone
        q=torch.zeros(len(pending),m,4,device=boxes.device)
        q[...,2],q[...,3]=(yaw/2).sin(),(yaw/2).cos()
        source=boxes[pending[:,None],primary[pending]].clone()
        source[...,:2]=xy
        source[...,2]=sizes[pending[:,None],primary[pending],2]/2
        source[...,3:7]=q
        source[...,7:]=0
        candidate=boxes[pending].clone()
        candidate[torch.arange(len(pending),device=boxes.device)[:,None],primary[pending]]=torch.where(local[...,None],source,candidate[torch.arange(len(pending),device=boxes.device)[:,None],primary[pending]])
        goal=source[...,:3].clone();goal[...,:2]=target
        valid=goal_angles_valid(source,goal,local,spec)
        valid &= (((target-roots[pending,:,:2]).norm(dim=-1)>=spec['goal_min_human_distance'])|~local).all(-1)
        valid &= paths_valid(candidate,sizes[pending],goal,roots[pending],local,primary[pending],active[pending],spec)
        accepted=pending[valid]
        result[accepted]=candidate[valid];goals[accepted]=goal[valid]
        pending=pending[~valid]
    if not len(pending):return result,goals
    raise PushLayoutError(result,goals,pending)
