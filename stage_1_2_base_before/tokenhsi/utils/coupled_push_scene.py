import torch


def axes(yaw):
    x = torch.stack((yaw.cos(), yaw.sin()), -1)
    return torch.stack((x, torch.stack((-x[..., 1], x[..., 0]), -1)), -2)


def swept_overlap(start, end, yaw, half, other_start, other_end, other_yaw, other_half, margin=0.):
    basis, other_basis = axes(yaw), axes(other_yaw)
    delta, other_delta = end-start, other_end-other_start
    normal = torch.stack((-delta[..., 1], delta[..., 0]), -1)
    normal /= normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    other_normal = torch.stack((-other_delta[..., 1], other_delta[..., 0]), -1)
    other_normal /= other_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    basis, other_basis = torch.broadcast_tensors(basis, other_basis)
    normal, other_normal = torch.broadcast_tensors(normal, other_normal)
    test = torch.cat((basis, other_basis, normal.unsqueeze(-2), other_normal.unsqueeze(-2)), -2)
    radius = ((test.unsqueeze(-2)*basis.unsqueeze(-3)).sum(-1).abs()*half.unsqueeze(-2)).sum(-1)
    other_radius = ((test.unsqueeze(-2)*other_basis.unsqueeze(-3)).sum(-1).abs()*other_half.unsqueeze(-2)).sum(-1)
    a, b = (start.unsqueeze(-2)*test).sum(-1), (end.unsqueeze(-2)*test).sum(-1)
    c, d = (other_start.unsqueeze(-2)*test).sum(-1), (other_end.unsqueeze(-2)*test).sum(-1)
    separated = (torch.maximum(a,b)+radius+margin < torch.minimum(c,d)-other_radius)
    separated |= (torch.maximum(c,d)+other_radius+margin < torch.minimum(a,b)-radius)
    return ~separated.any(-1)


def box_yaw(states):
    q = states[..., 3:7]
    return torch.atan2(2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1]), 1-2*(q[..., 1].square()+q[..., 2].square()))


def solo_corridors(center, goal):
    unit = goal-center
    unit = unit/unit.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    rear, end_rear = center-1.5*unit, goal-1.5*unit
    return torch.stack((rear,center,rear,goal),-2),torch.stack((center,goal,end_rear,end_rear),-2)


def solo_path_clear(center, goal, size, boxes, sizes, goals, roots, other_people, obstacles, other_push):
    from utils.jointpush_scene import segment_distance
    start,end = solo_corridors(center,goal)
    radius = size[...,:2].norm(dim=-1)/2
    human_gap = segment_distance(start[...,None,:],end[...,None,:],roots[...,None,:,:],roots[...,None,:,:]).amin(-2)
    valid = ((human_gap >= radius[...,None]+.5) | ~other_people).all(-1)
    gap = segment_distance(start[...,None,:],end[...,None,:],boxes[...,None,:,:2],boxes[...,None,:,:2]).amin(-2)
    required = radius[...,None]+sizes[...,:2].norm(dim=-1)/2+.5
    valid &= ((gap >= required) | ~obstacles | other_push).all(-1)
    other_start,other_end = solo_corridors(boxes[...,:2],goals[...,:2])
    gap = segment_distance(start[...,None,:,None,:],end[...,None,:,None,:],
                           other_start[...,None,:,:],other_end[...,None,:,:]).amin((-1,-2))
    valid &= ((gap >= 3.5) | ~obstacles | ~other_push).all(-1)
    return valid


def path_goals(task, ids):
    return task._center_tar_pos[ids] if getattr(task, '_individual_push_goals', False) else task._tar_pos[ids]


def paths(task, ids):
    boxes = task._logical_box_values(task._box_states, ids).clone()
    # Paths use whole physical boxes; region centres are only policy/contact tokens.
    boxes[:, :4] = task._assigned_box_values(task._box_states, ids)
    sizes = task._gather_entity_slots(task._box_size[ids], task._logical_box_order[ids])
    start = boxes[..., :2].clone()
    end = start.clone()
    push = task._templates_by_owner[ids] == 5
    goals = path_goals(task, ids)
    delta = goals[..., :2]-start[:, :4]
    unit = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    start[:, :4] -= torch.where(push[..., None], 1.5*unit, 0.)
    end[:, :4] = torch.where(push[..., None], goals[..., :2], end[:, :4])
    return start, end, box_yaw(boxes), sizes[..., :2]/2


def sample_push(task, ids):
    spec = task.cfg['env']['stage2JointPush']
    solo = task.cfg['env']['jointPush']['scene']
    source_bounds = torch.tensor([[solo['source_distance'][0], spec.get('solo_source_max_distance', solo['source_distance'][1])],
                                 spec.get('source_distance', [2., 10.])], device=task.device)
    goal_bounds = torch.tensor([[solo['goal_distance'][0], spec.get('solo_goal_max_distance', solo['goal_distance'][1])],
                               spec.get('goal_distance', [2., 10.])], device=task.device)
    failed = torch.zeros(len(ids), dtype=torch.bool, device=task.device)
    people = torch.arange(4, device=task.device)
    placed = task._templates_by_owner[ids] != 5
    for leader in range(4):
        pending = ((task._groups[ids, leader] == leader) & (task._templates_by_owner[ids, leader] == 5)).nonzero().flatten()
        for _ in range(spec['candidate_rounds']):
            if not len(pending):
                break
            es = ids[pending]
            n, c = len(es), spec['candidate_batch']
            roots = task._humanoid_root_states[es, ..., :2]
            physical = task._agent_box_assignment[es, leader]
            size = task._box_size[es, physical]
            draws = torch.rand(n, c, 5, device=task.device)
            shared = task._members[es, leader] > 1
            source_min, source_max = source_bounds[shared.long()].unbind(-1)
            goal_min, goal_max = goal_bounds[shared.long()].unbind(-1)
            radius = source_min[:, None]+(source_max-source_min)[:, None]*draws[..., 1]
            angle, yaw = draws[..., 0]*2*torch.pi, draws[..., 2]*2*torch.pi
            center = roots[:, leader, None]+radius[..., None]*torch.stack((angle.cos(),angle.sin()), -1)
            cone = shared | (task.cfg['env']['jointPush']['scene']['goal_angle_degrees'] == 90.)
            heading = torch.where(cone[:, None], yaw+torch.pi/2+(draws[..., 3]-.5)*torch.pi/2,
                                  draws[..., 3]*2*torch.pi)
            if spec['goal_half_angle_degrees'] == 0.:
                heading = torch.where(shared[:, None], yaw+torch.pi/2, heading)
            direction = torch.stack((heading.cos(), heading.sin()), -1)
            goal_distance = goal_min[:, None]+(goal_max-goal_min)[:, None]*draws[..., 4]
            goal = center+goal_distance[..., None]*direction
            relative = roots[:, None]-center[:, :, None]
            distances = relative.norm(dim=-1)
            member = task._groups[es] == leader
            rear = (relative*axes(yaw)[:, :, None, 1]).sum(-1) < 0
            valid = (((distances >= source_min[:,None,None])&(distances <= source_max[:,None,None])
                &(rear | ~shared[:,None,None])) | ~member[:,None]).all(-1)
            local = (relative.unsqueeze(-2)*axes(yaw)[:, :, None]).sum(-1)
            clearance = (local.abs()-size[:, None, None, :2]/2).clamp_min(0).norm(dim=-1)
            valid &= (clearance >= spec['human_clearance']).all(-1)
            valid &= (((goal[:,:,None]-roots[:,None]).norm(dim=-1) >= 1.) | ~member[:,None]).all(-1)
            starts, ends, oyaw, half = paths(task, es)
            relevant = torch.cat((placed[pending], task._templates_by_owner[es] == 4), -1)
            hit = swept_overlap((center-1.5*direction)[:,:,None], goal[:,:,None], yaw[:,:,None], size[:,None,None,:2]/2,
                starts[:,None], ends[:,None], oyaw[:,None], half[:,None], .5)
            valid &= ~(hit & relevant[:,None]).any(-1)
            # Other groups must also have a clear approach/path, as in Stage1 PUSH.
            human_hit = swept_overlap((center-1.5*direction)[:,:,None],goal[:,:,None],yaw[:,:,None],size[:,None,None,:2]/2,
                roots[:,None],roots[:,None],torch.zeros(n,1,4,device=task.device),torch.full((n,1,4,2),.5,device=task.device))
            valid &= ~(human_hit & ~member[:,None]).any(-1)
            boxes = task._gather_entity_slots(task._box_states[es],task._logical_box_order[es])
            sizes = task._gather_entity_slots(task._box_size[es],task._logical_box_order[es])
            goals = torch.cat((path_goals(task, es),boxes[:,4:,:3]),1)
            other_push = torch.cat((task._templates_by_owner[es] == 5,torch.zeros(n,4,device=task.device,dtype=torch.bool)),-1)
            solo_valid = solo_path_clear(center,goal,size[:,None],boxes[:,None],sizes[:,None],goals[:,None],roots[:,None],
                ~member[:,None],relevant[:,None],other_push[:,None])
            valid &= solo_valid | shared[:,None]
            accepted = valid.any(-1)
            choice = valid.long().argmax(-1)
            rows = torch.arange(n, device=task.device)
            es_ok, parent = es[accepted], physical[accepted]
            xy, target, turn = center[rows,choice][accepted], goal[rows,choice][accepted], yaw[rows,choice][accepted]
            task._box_states[es_ok,parent,:2] = xy
            task._box_states[es_ok,parent,2] = size[accepted,2]/2
            task._box_states[es_ok,parent,3:7] = torch.stack((turn*0,turn*0,(turn/2).sin(),(turn/2).cos()),-1)
            task._box_states[es_ok,parent,7:] = 0.
            target3 = torch.cat((target,size[accepted,2,None]/2),-1)
            task._tar_pos[es_ok] = torch.where(member[accepted,:,None],target3[:,None],task._tar_pos[es_ok])
            if getattr(task, '_individual_push_goals', False):
                task._center_tar_pos[es_ok] = task._tar_pos[es_ok]
            placed[pending[accepted]] |= member[accepted]
            pending = pending[~accepted]
        failed[pending] = True
    return failed


def place_distractors(task, ids):
    for owner in range(4):
        pending = ids[task._templates_by_owner[ids, owner] != 4]
        for _ in range(128):
            if not len(pending):
                break
            starts, ends, yaw, half = paths(task, pending)
            slot = owner+4
            point = task._box_states[pending,12+owner,:2]
            hit = swept_overlap(point[:,None],point[:,None],yaw[:,slot,None],half[:,slot,None],starts,ends,yaw,half,.5)
            hit[:,slot] = False
            roots = task._humanoid_root_states[pending,...,:2]
            near_human = ((point[:,None]-roots).norm(dim=-1) < (half[:,slot].norm(dim=-1)+.35).clamp_min(1.)[:,None]).any(-1)
            pending = pending[hit.any(-1)|near_human]
            task._box_states[pending,12+owner,:2] = task._sample_arena_xy(task._env_origins[pending,:2])


def scene_infeasible(task, ids):
    starts, ends, yaw, half = paths(task, ids)
    mapping = task._logical_box_order[ids]
    push = torch.cat((task._templates_by_owner[ids] == 5, torch.zeros(len(ids),4,device=task.device,dtype=torch.bool)),-1)
    # Sweeps for PUSH, static footprints for the remaining objects.
    hits = swept_overlap(starts[:,:,None],ends[:,:,None],yaw[:,:,None],half[:,:,None],
                         starts[:,None],ends[:,None],yaw[:,None],half[:,None],.05)
    different = mapping[:,:,None] != mapping[:,None]
    bad = (hits & different).any((1,2))
    states = task._gather_entity_slots(task._box_states[ids],mapping)
    roots = task._humanoid_root_states[ids,...,:2]
    relative = roots[:,:,None]-states[:,None,:,:2]
    local = (relative.unsqueeze(-2)*axes(yaw)[:,None]).sum(-1)
    near = (local.abs()-half[:,None]).clamp_min(0).norm(dim=-1) < .35
    bad |= near.any((1,2))
    sizes = task._gather_entity_slots(task._box_size[ids],mapping)
    radius = half.norm(dim=-1)
    gap = (states[:,:,None,:2]-states[:,None,:,:2]).norm(dim=-1)
    height_overlap = (states[:,:,None,2]-states[:,None,:,2]).abs() < (sizes[:,:,None,2]+sizes[:,None,:,2])/2+.02
    legacy_pair = ~push[:,:,None] & ~push[:,None]
    bad |= ((gap < radius[:,:,None]+radius[:,None]+.05) & different & height_overlap & legacy_pair).any((1,2))
    bodies = task._kinematic_humanoid_rigid_body_states[ids,...,:3]
    from utils.coupled_push_spec import inverse_rotate
    relative = bodies[:,:,None]-states[:,None,:,None,:3]
    local_body = inverse_rotate(states[:,None,:,None,3:7].expand(*relative.shape[:-1],4),relative)
    inside = (local_body.abs() < sizes[:,None,:,None]/2-.06).all(-1).any(-1)
    exempt = torch.arange(4,device=task.device)[None,:,None] == torch.arange(8,device=task.device)[None,None]
    bad |= (inside & (~exempt | push[:,None])).any((1,2))
    goal_gap = (task._tar_pos[ids,:,None,:2]-states[:,None,:,:2]).norm(dim=-1)
    legacy_at = task._templates_by_owner[ids] == 3
    goal_other = mapping[:,:4,None] != mapping[:,None]
    bad |= ((goal_gap < radius[:,:4,None]+radius[:,None]+.25) & goal_other & legacy_at[...,None]).any((1,2))
    goals = torch.cat((path_goals(task, ids),states[:,4:,:3]),1)
    for owner in range(4):
        solo = (task._templates_by_owner[ids,owner] == 5)&(task._members[ids,owner] == 1)
        other_people = torch.arange(4,device=task.device)[None] != owner
        other_boxes = mapping != mapping[:,owner,None]
        clear = solo_path_clear(states[:,owner,:2],goals[:,owner,:2],sizes[:,owner],states,sizes,goals,roots,other_people,other_boxes,push)
        bad |= solo & ~clear
    return bad

