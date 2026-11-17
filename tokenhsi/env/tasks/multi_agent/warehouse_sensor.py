import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.before_spec import task_objects
from utils.sensor_stage2_spec import PUSH, WAREHOUSE
from utils.warehouse_sensor import opening_clearance, sample_layout, walls


def add_geometry(task):
    cfg = task._warehouse
    n = task.num_envs
    half = task._surface_halves(cfg,(n,2))
    opening = task._box_size[:,task._bridge_bank,1]-2*cfg['opening_overlap']
    if task._warehouse_coop:
        spec = task._warehouse_coop
        half = half.new_tensor(spec['room_size']).expand(n,2)/2
        variant = getattr(task,'_warehouse_geometry_variant',0)
        width = spec['box_size'][1] if variant < 3 else spec['box_size'][0]*(variant%3+1)
        opening = opening.new_full((n,),width+(-2*cfg['opening_overlap'] if variant < 3 else spec['frontal_opening_gap']))
    centers, dimensions, starts, ends = walls(half,cfg['wall_thickness'],cfg['wall_height'],opening)
    centers[...,2] += cfg['floor_height']
    bounds = torch.stack((-half[:,0]-cfg['wall_thickness'],half[:,0]+cfg['wall_thickness'],
        -half[:,1]-cfg['wall_thickness'],half[:,1]+cfg['wall_thickness']),-1)
    if task._warehouse_coop:
        bounds[:,0] = -half[:,0]-spec['apron_length']
        bounds[:,2:] = half[:,1,None].expand(-1,2)*half.new_tensor([-1.,1.])+half.new_tensor([-.5,.5])
    placed = scene.place_structure(bounds,task._scene_cfg['floor_clearance'],task._scene_cfg['translation_range'])
    if task._scene_spacing < float(placed['reach'].max())+20.:
        raise ValueError('sceneSpacing is too small for the warehouse')
    task._structure_bounds = torch.cat((task._structure_bounds,bounds[:,None]),1)
    for name in ('yaw','shift','axis','direction','near'):
        key = '_structure_'+name
        setattr(task,key,torch.cat((getattr(task,key),placed[name][:,None]),1))
    center = half.new_zeros(n,6,3)
    size = half.new_zeros(n,6,3)
    center[:,0,2] = cfg['floor_height']/2
    size[:,0,:2],size[:,0,2] = half,cfg['floor_height']/2
    if task._warehouse_coop:
        center[:,0,0] = -spec['apron_length']/2
        size[:,0,0] += spec['apron_length']/2
        size[:,0,1] += .5
    center[:,1:],size[:,1:] = centers,dimensions
    yaw,shift = placed['yaw'],placed['shift']
    local = torch.cat((scene.rotate_xy(center[...,:2],yaw[:,None])+shift[:,None],center[...,2:]),-1)
    task._static_local = torch.cat((task._static_local,local[:,None]),1)
    task._static_half = torch.cat((task._static_half,size[:,None]),1)
    quat = scene.yaw_quat(yaw)[:,None,None].expand(-1,1,6,-1)
    task._static_quat = torch.cat((task._static_quat,quat),1)
    surface = half.new_zeros(n,1,2,2)
    surface[:,0,0] = half
    task._surface_half = torch.cat((task._surface_half,surface),1)
    task._surface_center = torch.cat((task._surface_center,torch.zeros_like(surface)),1)
    task._surface_height = torch.cat((task._surface_height,half.new_full((n,1),cfg['floor_height'])),1)
    offset = half.new_tensor([[-task._scene_spacing/2,1.5*task._scene_spacing,0.]])
    task._scene_offsets = torch.cat((task._scene_offsets,offset),0)
    task._warehouse_wall_local = (scene.rotate_xy(starts,yaw[:,None])+shift[:,None],
                                 scene.rotate_xy(ends,yaw[:,None])+shift[:,None])


def allocate(task):
    origin = task._scene_origins[:,WAREHOUSE,None,:2]
    task._warehouse_wall_starts,task._warehouse_wall_ends = [value+origin for value in task._warehouse_wall_local]
    task._warehouse_landing = torch.zeros(task.num_envs,3,device=task.device)
    task._warehouse_opened = torch.zeros(task.num_envs,dtype=torch.bool,device=task.device)
    task._warehouse_arrived = torch.zeros_like(task._warehouse_opened)
    task._warehouse_entered = torch.zeros(task.num_envs,task.num_agents,dtype=torch.bool,device=task.device)
    task._warehouse_outcomes = torch.zeros(4,device=task.device)
    task._warehouse_start_mode = torch.zeros(task.num_envs,dtype=torch.long,device=task.device)
    task._warehouse_start_samples = torch.zeros(3,device=task.device)
    task._warehouse_start_outcomes = torch.zeros(3,4,device=task.device)
    task._warehouse_worker_episodes = torch.zeros((),device=task.device)
    task._warehouse_start_worker_episodes = torch.zeros(3,device=task.device)
    if task._warehouse_coop:
        task._warehouse_direction = torch.zeros(task.num_envs,dtype=torch.long,device=task.device)
        task._warehouse_coop_samples = torch.zeros(36,device=task.device)
        task._warehouse_coop_outcomes = torch.zeros(36,5,device=task.device)


def sample_starts(task, ids):
    if not len(ids):
        return
    choice = task._warehouse_start_choice
    if choice == 'mixed':
        mode = torch.multinomial(torch.tensor(task._warehouse_start['probabilities'],device=task.device),len(ids),True)
    else:
        mode = torch.full((len(ids),),('closed','partial','open').index(choice),dtype=torch.long,device=task.device)
    task._warehouse_start_mode[ids] = mode
    task._warehouse_start_samples += torch.bincount(mode,minlength=3)
    if task._warehouse_coop:
        key = ((task._bridge_count[ids]-1)*4+task._warehouse_direction[ids])*3+mode
        task._warehouse_coop_samples += torch.bincount(key,minlength=36)


def place(task, ids):
    cfg = task._warehouse
    n,m = len(ids),task.num_agents
    rows = torch.arange(n,device=task.device)
    people = torch.arange(m,device=task.device)[None].expand(n,-1)
    templates,predecessor,roles = task._templates[ids],task._before_predecessor[ids],task._roles[ids]
    primary = task._agent_box_assignment[ids].clone()
    support = 3*m+people
    if task._warehouse_coop:
        from utils.warehouse_coop import sample_layout as cooperative_layout
        result = cooperative_layout(templates,predecessor,roles,task._box_size[ids[:,None],primary],
            task._box_size[ids[:,None],support],task._before_near_pairs[ids],task._bridge_count[ids],
            task._warehouse_direction[ids],task._warehouse_start_mode[ids],cfg,task._scene_cfg['placement_tries'])
        push = templates == 5
        original = (push.long().cumsum(-1)-1).clamp_min(0)
        tangent = torch.stack((result['box_yaw'].cos(),result['box_yaw'].sin()),-1)
        delta = (task._regions[ids]-original)*task._warehouse_coop['box_size'][0]
        result['root'] += torch.where(push[...,None],delta[...,None]*tangent[:,None],0.)
    else:
        result = sample_layout(templates,predecessor,roles,task._box_size[ids[:,None],primary],
            task._box_size[ids[:,None],support],task._surface_half[ids,WAREHOUSE,0],
            task._box_size[ids,task._bridge_slot[ids]],task._before_near_pairs[ids],cfg,task._scene_cfg['placement_tries'],
            task._warehouse_start_mode[ids] if task._warehouse_start else None)
    task._place_failed[ids] = ~result['valid']
    accepted = result['valid'].nonzero().flatten()
    if not len(accepted):
        return
    env = ids[accepted]
    rows = torch.arange(len(env),device=task.device)
    primary,support = primary[accepted],support[accepted]
    templates,predecessor,roles = templates[accepted],predecessor[accepted],roles[accepted]
    result = {key:value[accepted] for key,value in result.items()}
    half = task._surface_half[env,WAREHOUSE,0]
    yaw,shift,origin = task._structure_yaw[env,WAREHOUSE],task._structure_shift[env,WAREHOUSE],task._env_origins[env]

    def world(xy):
        local = xy.clone()
        local[...,0] -= half[:,0].reshape(len(env),*(1,)*(xy.ndim-2))
        return scene.to_world(local,yaw,shift,origin)

    valid = task_objects(predecessor,templates)
    used = torch.zeros(len(env),task.num_physical_objects,device=task.device,dtype=torch.bool)
    for values,slots,mask in ((result['source'],primary,valid[:,:m]),(result['support'],support,valid[:,m:])):
        row,person = mask.nonzero(as_tuple=True)
        e,slot = env[row],slots[row,person]
        task._box_states[e,slot,:2] = world(values)[row,person]
        lift = torch.ones(len(row),device=task.device) if task._warehouse_coop else values[row,person,0] >= 0
        task._box_states[e,slot,2] = task._env_origins[e,2]+task._box_size[e,slot,2]/2+cfg['floor_height']*lift
        task._box_states[e,slot,3:7] = scene.yaw_quat(yaw[row])
        task._box_states[e,slot,7:] = 0
        used[row,slot] = True
    if task._warehouse_coop:
        task._box_states[env,task._bridge_slot[env],3:7] = scene.yaw_quat(yaw+result['box_yaw'])
    roots = world(result['root'])
    targets = world(result['source'])
    actual = torch.where((predecessor >= 0) & (templates < 3),predecessor,people[:len(env)])
    targets = targets.gather(1,actual[...,None].expand(-1,-1,2))
    heading = torch.atan2(targets[...,1]-roots[...,1],targets[...,0]-roots[...,0])
    if task._warehouse_coop:
        heading = torch.where(templates == 5,yaw[:,None]+result['push_heading'][:,None],heading)
    for person in range(m):
        lift = cfg['floor_height'] if task._warehouse_coop else 0.
        task._place_person(env,person,roots[:,person],heading[:,person],torch.full((len(env),),lift,device=task.device))
    task._tar_pos[env,:,:2] = world(result['goal'])
    task._tar_pos[env,:,2] = task._env_origins[env,None,2]+task._box_size[env[:,None],primary,2]/2
    if task._warehouse_coop:
        task._tar_pos[env,:,2] += cfg['floor_height']
        row,person = (templates == 5).nonzero(as_tuple=True)
        task._push_goals[env[row],person] = task._tar_pos[env[row],person]
    pusher = roles[:,0]
    task._push_goals[env,pusher] = task._tar_pos[env,pusher]
    task._warehouse_landing[env] = task._tar_pos[env,pusher]
    active = task._active_physical(env)
    parked = active & ~used
    row,slot = parked.nonzero(as_tuple=True)
    if len(row):
        local = task._box_states.new_zeros(len(env),task.num_physical_objects,2)
        local[...,0] = -10.-3*(parked.long().cumsum(-1)-1)
        local[...,1] = 10.
        e = env[row]
        task._box_states[e,slot,:2] = world(local)[row,slot]
        task._box_states[e,slot,2] = task._env_origins[e,2]+task._box_size[e,slot,2]/2
        task._box_states[e,slot,7:] = 0
    if task.viewer is not None:
        from isaacgym import gymapi
        for e in env.tolist():
            handle = task._box_handles[e*task.num_physical_objects+int(task._bridge_slot[e])]
            task.gym.set_rigid_body_color(task.envs[e],handle,0,gymapi.MESH_VISUAL,gymapi.Vec3(.9,.08,.06))


def wall_box_cpa(task, ids, root_pos, min_dist):
    """CPA against the five solid wall boxes (static slots 1-5), so outer faces, corners and door jambs count."""
    centers = task._static_centers[ids,WAREHOUSE,1:]
    quats,halves = task._static_quat[ids,WAREHOUSE,1:],task._static_half[ids,WAREHOUSE,1:]
    velocity,still = task._humanoid_root_states[ids,:,7:10],torch.zeros_like(centers[:,0])
    return torch.stack([scene.compute_box_cpa_collision_penalty(root_pos,velocity,centers[:,k],quats[:,k],halves[:,k],
        still,min_dist,task._warehouse['wall_ttc_discount'],task.dt) for k in range(centers.shape[1])]).amax(0)


def collision(task, agents, root_pos, min_dist):
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    if len(ids):
        if task._warehouse.get('wall_cpa') == 'box':
            walls = wall_box_cpa(task,ids,root_pos[ids],min_dist)
        else:
            walls = scene.compute_wall_cpa_collision_penalty(root_pos[ids],task._humanoid_root_states[ids,:,7:10],
                task._warehouse_wall_starts[ids],task._warehouse_wall_ends[ids],min_dist,
                task._warehouse['wall_ttc_discount'],task.dt)
        agents[ids] = torch.maximum(agents[ids],walls*task._wall_collision_scale)
        if task._push_box_collision_scale:
            slot = task._bridge_slot[ids]
            box = task._box_states[ids,slot]
            risk = scene.compute_box_cpa_collision_penalty(root_pos[ids],task._humanoid_root_states[ids,:,7:10],
                box[:,:3],box[:,3:7],task._box_size[ids,slot]/2,box[:,7:10],min_dist,
                task._warehouse['wall_ttc_discount'],task.dt)
            others = task._templates[ids] != PUSH
            agents[ids] = torch.maximum(agents[ids],risk*others*task._push_box_collision_scale)
    return agents


def push_box_contact_step(task):
    """Count consecutive contact of non-pushers with the PUSH box; return people over the configured limit."""
    from env.tasks.multi_agent.sensor_stage2_observation import WALL_CONTACT_TOLERANCE
    touching = torch.zeros_like(task._push_box_contact_steps,dtype=torch.bool)
    chosen = task._scene_ids == WAREHOUSE
    if getattr(task, '_warehouse_before_probability', 0.):
        chosen &= ~task._warehouse_before_pairs
    others = task._templates != PUSH
    ids = chosen.nonzero().flatten()
    if len(ids):
        slot = task._bridge_slot[ids]
        box = task._box_states[ids,slot]
        bodies = task._wall_contact_body_ids
        touching[ids] = scene.body_box_contact(task._rigid_body_pos[ids][:,:,bodies],
            task._rigid_body_rot[ids][:,:,bodies],task._contact_forces[ids][:,:,bodies],task._wall_contact_shapes,
            box[:,None,:3],box[:,None,3:7],task._box_size[ids,slot][:,None]/2,WALL_CONTACT_TOLERANCE)
    touching &= others & (task.progress_buf[:,None] > 1)
    steps = torch.where(touching,task._push_box_contact_steps+1,torch.zeros_like(task._push_box_contact_steps))
    task._push_box_contact_steps[:] = steps
    limit = task._push_box_contact_limit if task._enable_early_termination else None
    ended = steps >= limit if limit else torch.zeros_like(touching)
    crossed = steps == limit if limit else ended
    task._push_box_outcomes += torch.stack((touching[chosen].sum(),(others & chosen[:,None]).sum(),
        crossed[chosen].sum())).float()
    return ended


def record(task, ending):
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    if getattr(task, '_warehouse_before_probability', 0.):
        from env.tasks.multi_agent import warehouse_before
        warehouse_before.record(task, ending)
        ids = ids[~task._warehouse_before_pairs[ids]]
    if getattr(task, '_warehouse_done_probability', 0.):
        from env.tasks.multi_agent import warehouse_done
        warehouse_done.record(task, ending)
        ids = ids[~task._warehouse_leader_done[ids]]
    if not len(ids):
        return
    cfg = task._warehouse
    roots = task._rigid_body_pos[ids,:,0]
    local = scene.rotate_xy(roots[...,:2]-task._env_origins[ids,None,:2]-task._structure_shift[ids,WAREHOUSE,None],
        -task._structure_yaw[ids,WAREHOUSE,None])
    half = task._surface_half[ids,WAREHOUSE,0]
    entered = (local.abs() < half[:,None]-.35).all(-1)
    workers = task._templates[ids] != 5
    task._warehouse_entered[ids] |= entered & workers
    box = task._box_states[ids,task._bridge_slot[ids]]
    xy = scene.rotate_xy(box[:,:2]-task._env_origins[ids,:2]-task._structure_shift[ids,WAREHOUSE],
        -task._structure_yaw[ids,WAREHOUSE])
    width = task._box_size[ids,task._bridge_slot[ids],1]
    clearance = opening_clearance(xy[:,1],box[:,3:7],task._box_size[ids,task._bridge_slot[ids]],
        task._structure_yaw[ids,WAREHOUSE],width-2*cfg['opening_overlap'])
    if task._warehouse_coop:
        from env.tasks.multi_agent.warehouse_coop import opened
        # Frontal motion is measured along the doorway normal, rather than its width.
        task._warehouse_opened[ids] |= opened(task,ids,xy,box,task._box_size[ids,task._bridge_slot[ids]])
    else:
        task._warehouse_opened[ids] |= clearance >= cfg['open_clearance']-.001
    task._warehouse_arrived[ids] |= (box[:,:3]-task._warehouse_landing[ids]).norm(dim=-1) < task._relation_cfg['at']['success_distance']
    ended = ids[ending[ids]]
    worker_count = (task._templates[ended] != 5).sum(-1)
    task._warehouse_worker_episodes += worker_count.sum()
    task._warehouse_outcomes += torch.stack((ending[ids].sum(),task._warehouse_opened[ended].sum(),
        task._warehouse_entered[ended].sum(),task._warehouse_arrived[ended].sum())).float()

    if task._warehouse_start:
        for mode in range(3):
            selected = ended[task._warehouse_start_mode[ended] == mode]
            task._warehouse_start_outcomes[mode] += torch.stack((ending[selected].sum(),
                task._warehouse_opened[selected].sum(),task._warehouse_entered[selected].sum(),
                task._warehouse_arrived[selected].sum())).float()
            task._warehouse_start_worker_episodes[mode] += (task._templates[selected] != 5).sum()
    if task._warehouse_coop:
        key = ((task._bridge_count[ended]-1)*4+task._warehouse_direction[ended])*3+task._warehouse_start_mode[ended]
        values = torch.stack((torch.ones_like(worker_count),task._warehouse_opened[ended],
            task._warehouse_entered[ended].sum(-1),task._warehouse_arrived[ended],worker_count),-1).float()
        task._warehouse_coop_outcomes.index_add_(0,key,values)
