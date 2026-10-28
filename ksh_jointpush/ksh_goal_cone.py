"""Local Stage1/Stage2 solo PUSH goal-cone fix; samplers copied from frozen KSH source.
Only the marked acceptance tests are added. Invalid candidates follow original retries.
"""
import math
import torch
from utils.jointpush_scene import radial_xy, paths_clear
from utils.stage2_jointpush_scene import axes, paths, solo_path_clear, swept_overlap
from utils.jointpush_geometry import rotate


def yaw_goal_in_cone(yaw, delta, half_angle_degrees=45.):
    normal=torch.stack((-yaw.sin(),yaw.cos()),-1)
    distance=delta.norm(dim=-1)
    return (distance>1e-8)&((normal*delta).sum(-1)>=distance*(math.cos(math.radians(half_angle_degrees))-1e-6))


def goal_in_cone(boxes, goals, half_angle_degrees=45.):
    normal=torch.zeros_like(boxes[...,:3]);normal[...,1]=1
    normal=rotate(boxes[...,3:7],normal)[...,:2]
    delta=goals[...,:2]-boxes[...,:2]
    distance=delta.norm(dim=-1)
    return (distance>1e-8)&((normal*delta).sum(-1)>=distance*(math.cos(math.radians(half_angle_degrees))-1e-6))


def sample_layout(roots, boxes, sizes, push, spec):
    result = boxes.clone()
    goals = boxes[:, :2, :3].clone()
    pending = push.any(-1).nonzero().flatten()
    for _ in range(512):
        if not len(pending):
            return result, goals
        active = push[pending]
        xy = radial_xy(roots[pending, :, :2], spec['source_distance'])
        target = radial_xy(xy, spec['goal_distance'])
        candidate = boxes[pending].clone()
        candidate[:, :2, :2] = torch.where(active[..., None], xy, candidate[:, :2, :2])
        candidate[:, :2, 2] = torch.where(active, sizes[pending, :2, 2]/2, candidate[:, :2, 2])
        yaw = torch.rand(len(pending), 2, device=boxes.device)*2*torch.pi
        q = torch.zeros(len(pending), 2, 4, device=boxes.device)
        q[..., 2], q[..., 3] = (yaw/2).sin(), (yaw/2).cos()
        candidate[:, :2, 3:7] = torch.where(active[..., None], q, candidate[:, :2, 3:7])
        candidate[:, :2, 7:] = torch.where(active[..., None], 0., candidate[:, :2, 7:])
        if spec.get('solo_goal_half_angle_degrees',spec.get('goal_half_angle_degrees')) == 0.:
            distance=(target-xy).norm(dim=-1)
            target=xy+distance[...,None]*torch.stack((-yaw.sin(),yaw.cos()),-1)
        target3 = candidate[:, :2, :3].clone()
        target3[..., :2] = target
        valid = paths_clear(candidate, sizes[pending], target3, roots[pending], active, spec)
        valid &= (((target-roots[pending, :, :2]).norm(dim=-1) >= spec['goal_min_human_distance']) | ~active).all(-1)
        valid &= (goal_in_cone(candidate[:, :2], target3, spec.get('solo_goal_half_angle_degrees',spec.get('goal_half_angle_degrees',45.))) | ~active).all(-1)
        accepted = pending[valid]
        result[accepted] = candidate[valid]
        goals[accepted] = target3[valid]
        pending = pending[~valid]
    raise RuntimeError('stage1_jointpush could not sample separated PUSH paths')

def sample_push(task, ids):
    spec = task.cfg['env']['stage2JointPush']
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
            angle, radius, yaw = draws[..., 0]*2*torch.pi, 2+8*draws[..., 1], draws[..., 2]*2*torch.pi
            center = roots[:, leader, None]+radius[..., None]*torch.stack((angle.cos(),angle.sin()), -1)
            shared = task._members[es, leader] > 1
            heading = torch.where(shared[:, None], yaw+torch.pi/2+(draws[..., 3]-.5)*torch.pi/2,
                                  draws[..., 3]*2*torch.pi)
            scene = task.cfg['env']['jointPush']['scene']
            coupled_angle = scene.get('goal_half_angle_degrees',45.)
            solo_angle = scene.get('solo_goal_half_angle_degrees',coupled_angle)
            if coupled_angle == 0.:
                heading = torch.where(shared[:,None],yaw+torch.pi/2,heading)
            if solo_angle == 0.:
                heading = torch.where(~shared[:,None],yaw+torch.pi/2,heading)
            direction = torch.stack((heading.cos(), heading.sin()), -1)
            goal = center+(2+8*draws[..., 4, None])*direction
            relative = roots[:, None]-center[:, :, None]
            distances = relative.norm(dim=-1)
            member = task._groups[es] == leader
            rear = (relative*axes(yaw)[:, :, None, 1]).sum(-1) < 0
            valid = (((distances >= 2)&(distances <= 10)&(rear | ~shared[:,None,None])) | ~member[:,None]).all(-1)
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
            goals = torch.cat((task._tar_pos[es],boxes[:,4:,:3]),1)
            other_push = torch.cat((task._templates_by_owner[es] == 5,torch.zeros(n,4,device=task.device,dtype=torch.bool)),-1)
            solo_valid = solo_path_clear(center,goal,size[:,None],boxes[:,None],sizes[:,None],goals[:,None],roots[:,None],
                ~member[:,None],relevant[:,None],other_push[:,None])
            valid &= solo_valid | shared[:,None]
            valid &= shared[:, None] | yaw_goal_in_cone(yaw, goal-center,solo_angle)
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
            placed[pending[accepted]] |= member[accepted]
            pending = pending[~accepted]
        failed[pending] = True
    return failed

def install(env):
    spec=env.get('jointPush',{}).get('scene',{})
    angle=spec.get('goal_half_angle_degrees')
    if angle is None:return
    if angle not in (0.,45.):raise ValueError('Supported goal half angles: 0 or 45 degrees')
    solo_angle=spec.get('solo_goal_half_angle_degrees',angle)
    if solo_angle not in (0.,45.):raise ValueError('Supported solo goal half angles: 0 or 45 degrees')
    if angle == 0 and 'stage2JointPush' in env:
        from copy import deepcopy
        from utils import stage2_jointpush as stage2_spec
        original_validate=stage2_spec.validate_env
        def validate(config):
            if config['stage2JointPush']['goal_half_angle_degrees']!=0.:raise ValueError('Aligned experiment requires zero goal angle')
            original=deepcopy(config)
            original['stage2JointPush']['goal_half_angle_degrees']=45.
            return original_validate(original)
        stage2_spec.validate_env=validate
    import utils.jointpush_scene as stage1_scene
    import utils.stage2_jointpush_scene as stage2_scene
    import env.tasks.multi_agent.humanoid_ma_stage1_jointpush as stage1_task
    stage1_scene.sample_layout=sample_layout
    stage1_task.sample_layout=sample_layout
    stage2_scene.sample_push=sample_push
    print('KSH_PUSH_GOAL_CONE',dict(stage1_solo=solo_angle,stage2_solo=solo_angle,stage2_coupled=angle),flush=True)
