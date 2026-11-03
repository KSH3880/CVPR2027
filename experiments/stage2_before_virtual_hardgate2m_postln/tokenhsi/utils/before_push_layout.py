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


def sample_batched_layout(boxes, sizes, roots, push, primary, active, spec,
                          max_attempts=128, candidate_batch=16):
    n, m = push.shape
    result = boxes.clone()
    rows = torch.arange(n, device=boxes.device)[:, None]
    goals = boxes[rows, primary, :3].clone()
    pending = push.any(-1).nonzero().flatten()
    for offset in range(0, max_attempts, candidate_batch):
        if not len(pending):
            return result, goals
        count = min(candidate_batch, max_attempts-offset)
        args = [value[pending].repeat_interleave(count, 0)
                for value in (boxes, sizes, roots, push, primary, active)]
        valid = torch.ones(len(pending)*count, device=boxes.device, dtype=torch.bool)
        try:
            candidate, drawn = sample_layout(*args, spec, max_attempts=1)
        except PushLayoutError as error:
            candidate, drawn = error.boxes, error.goals
            valid[error.pending] = False
        valid = valid.reshape(len(pending), count)
        good = valid.any(-1)
        choice = torch.arange(len(pending), device=boxes.device)*count + valid.long().argmax(-1)
        accepted = pending[good]
        result[accepted] = candidate[choice[good]]
        goals[accepted] = drawn[choice[good]]
        pending = pending[~good]
    if len(pending):
        raise PushLayoutError(result, goals, pending)
    return result, goals


def sample_sequential_layout(boxes, sizes, roots, push, primary, active, spec,
                             max_restarts=32, candidate_batch=128, max_attempts=16):
    n, m = push.shape
    device = boxes.device
    row = torch.arange(n, device=device)[:, None]
    result = boxes.clone()
    goals = boxes[row, primary, :3].clone()
    push_objects = torch.zeros_like(active)
    push_objects.scatter_(1, primary, push)
    obstacles = active & ~push_objects
    pending = push.any(-1).nonzero().flatten()
    for _ in range(max_restarts):
        if not len(pending):
            return result, goals
        current = boxes[pending].clone()
        targets = goals[pending].clone()
        settled = torch.zeros_like(push[pending])
        blocked = torch.zeros(len(pending), dtype=torch.bool, device=device)
        order = torch.rand(len(pending), m, device=device).masked_fill(~push[pending], 2.).argsort(-1)
        for stage in range(m):
            actor = order[:, stage]
            selected = (torch.arange(m, device=device)[None] == actor[:, None]) & push[pending]
            retry = (selected.any(-1) & ~blocked).nonzero().flatten()
            for _ in range(max_attempts):
                if not len(retry):
                    break
                ids = pending[retry]
                local = selected[retry].repeat_interleave(candidate_batch, 0)
                actors = actor[retry].repeat_interleave(candidate_batch)
                batch = torch.arange(len(actors), device=device)
                batch_sizes = sizes[ids].repeat_interleave(candidate_batch, 0)
                batch_roots = roots[ids].repeat_interleave(candidate_batch, 0)
                batch_primary = primary[ids].repeat_interleave(candidate_batch, 0)
                batch_active = obstacles[ids].repeat_interleave(candidate_batch, 0)
                accepted = torch.ones(len(actors), dtype=torch.bool, device=device)
                try:
                    candidate, drawn = sample_layout(current[retry].repeat_interleave(candidate_batch, 0),
                        batch_sizes, batch_roots, local, batch_primary, batch_active, spec, max_attempts=1)
                except PushLayoutError as error:
                    candidate, drawn = error.boxes, error.goals
                    accepted[error.pending] = False
                target = targets[retry].repeat_interleave(candidate_batch, 0)
                target[batch, actors] = drawn[batch, actors]
                checking = (settled[retry] | selected[retry]).repeat_interleave(candidate_batch, 0)
                valid = accepted & paths_valid(candidate, batch_sizes, target, batch_roots,
                    checking, batch_primary, batch_active, spec)
                valid = valid.reshape(len(retry), candidate_batch)
                good = valid.any(-1)
                choice = torch.arange(len(retry), device=device) * candidate_batch + valid.long().argmax(-1)
                current[retry[good]] = candidate[choice[good]]
                targets[retry[good]] = target[choice[good]]
                settled[retry[good]] |= selected[retry[good]]
                retry = retry[~good]
            blocked[retry] = True
        valid = (settled == push[pending]).all(-1)
        valid &= paths_valid(current, sizes[pending], targets, roots[pending],
            push[pending], primary[pending], active[pending], spec)
        accepted = pending[valid]
        result[accepted], goals[accepted] = current[valid], targets[valid]
        pending = pending[~valid]
    if len(pending):
        raise PushLayoutError(result, goals, pending)
    return result, goals
