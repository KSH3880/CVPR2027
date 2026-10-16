import torch

from utils.joint_carry_rsi import transform_states
from utils.task_role_spec import task_graph_packet
from utils.group_carry_spec import owner_tasks, entity_validity
from utils.size_rsi import resolve_skills


def reset_groups(t, ids):
    t._sample_episode_graph(ids)
    tasks = owner_tasks(task_graph_packet(t.relation_runtime.graph, t.num_envs)[ids], 4).long()
    joint = t._group_joint[ids]
    leaders = t._group_leader[ids]
    ontop = tasks[..., 1] == 3
    if t._is_eval:
        weights = t._skill_init_prob.expand(len(ids), 4, -1).clone()
        if 'putDown' in t._skill:
            weights[..., t._skill.index('putDown')].masked_fill_(joint & ontop, 0.)
        if (weights.sum(-1) == 0).any():
            raise ValueError('Requested RSI skills cannot initialize joint ON_TOP')
    else:
        weights = t._template_rsi_weights[torch.where(ontop, 4, 3)].clone()
        weights[~joint] = weights.new_tensor(t.cfg['env']['independentCarryRsi'])
    skills = torch.multinomial(weights.reshape(-1, len(t._skill)), 1).reshape(len(ids), 4)
    solo_row, solo_human = (~joint).nonzero(as_tuple=True)
    if len(solo_row) and not t._is_eval:
        template = torch.where(ontop[solo_row, solo_human], 4, 3)
        requested = skills[solo_row, solo_human]
        selected, fallback = resolve_skills(requested, weights[solo_row, solo_human],
            t._size_rsi_cache.availability(ids[solo_row], solo_human, template), t._skill.index('loco'))
        skills[solo_row, solo_human] = selected
        t._size_rsi_requested.index_put_((template, requested), torch.ones_like(template, dtype=torch.float), accumulate=True)
        t._size_rsi_fallbacks.index_add_(0, template, fallback.float())
        t._size_rsi_executed.index_put_((template, selected), torch.ones_like(template, dtype=torch.float), accumulate=True)
    skills = skills.gather(1, leaders)
    t._group_skill[ids] = skills
    t._group_frame[ids] = -1
    t._group_source_role[ids] = -1
    t._reset_default_slots = None
    t._reset_ref_slots = {}
    t._reset_ref_motion_ids = {}
    t._reset_ref_motion_times = {}
    t._box_states[ids] = 0.
    t._box_states[ids, :, 6] = 1.
    t._box_states[ids, :, 2] = -20.
    t._box_states[ids, :, :2] = t._env_origins[ids, None, :2]
    t._tar_pos[ids] = t._env_origins[ids, None]
    for leader in range(4):
        for paired in (False, True):
            group_rows = ((leaders[:, leader] == leader) & (joint[:, leader] == paired)).nonzero().flatten()
            for skill_index, name in enumerate(t._skill):
                rows = group_rows[skills[group_rows, leader] == skill_index]
                if not len(rows):
                    continue
                envs = ids[rows]
                humans = torch.full((len(rows), 1), leader, device=t.device, dtype=torch.long)
                if paired:
                    partner = ((leaders[rows] == leader) & (torch.arange(4, device=t.device)[None] != leader)).long().argmax(-1)
                    humans = torch.cat((humans, partner[:, None]), -1)
                is_ontop = ontop[rows, leader]
                root, dof, vel, body, box, terminal = local_state(t, envs, humans, name, is_ontop, paired)
                payload = t._logical_box_order[envs, leader]
                support = t._logical_box_order[envs, leader+4]
                size = t._box_size[envs, payload]
                goal = sample_local_goals(box, t._box_size[envs, support], body, terminal, name, is_ontop)
                goal[:, 2] = size[:, 2]/2 + is_ontop*t._box_size[envs, support, 2]
                t._box_states[envs, payload] = box
                t._box_states[envs[is_ontop], support[is_ontop], :2] = goal[is_ontop, :2]
                t._box_states[envs[is_ontop], support[is_ontop], 2] = t._box_size[envs[is_ontop], support[is_ontop], 2]/2
                t._humanoid_root_states[envs[:, None], humans] = root
                t._dof_pos[envs[:, None], humans] = dof
                t._dof_vel[envs[:, None], humans] = vel
                t._kinematic_humanoid_rigid_body_states[envs[:, None], humans] = body
                t._every_env_init_dof_pos[envs[:, None], humans] = dof
                t._tar_pos[envs, leader] = goal
                t._stage2_rsi_counts[skill_index] += len(envs)*humans.shape[1]
    t._joint_loco_platform.reset(ids)
    place_groups(t, ids)
    park_unused_boxes(t, ids)
    t._joint_reset.commit(ids)
    t._refresh_sim_tensors()
    t._reset_relation_history(ids)
    t._compute_observations(ids)
    t._init_amp_obs(ids)
    r, h = joint.nonzero(as_tuple=True)
    t._hist_amp_obs_buf[ids[r], h] = t._curr_amp_obs_buf[ids[r], h, None]


def park_unused_boxes(t, ids):
    packet = task_graph_packet(t.relation_runtime.graph, t.num_envs)[ids]
    valid = entity_validity(packet, 4, 8)[:, 4:12]
    active = torch.zeros(len(ids), 12, device=t.device, dtype=torch.bool)
    active.scatter_(1, t._logical_box_order[ids], valid)
    row, physical = (~active).nonzero(as_tuple=True)
    envs = ids[row]
    parked = torch.zeros(len(row), 13, device=t.device)
    parked[:, :2] = t._env_origins[envs, :2]+1000.
    parked[:, 0] += physical*2.
    parked[:, 2] = t._box_size[envs, physical, 2]/2
    parked[:, 6] = 1.
    t._box_states[envs, physical] = parked


def local_state(t, envs, humans, name, ontop, paired):
    n, count = humans.shape
    box = torch.zeros(n, 13, device=t.device)
    box[:, 6] = 1.
    payload = t._agent_box_assignment[envs, humans[:, 0]]
    box[:, 2] = t._box_size[envs, payload, 2]/2
    terminal = torch.zeros(n, 2, device=t.device)
    if paired:
        if name == 'loco':
            root, dof, vel, body = t._joint_reset.locomotion(n)
        else:
            pool = t._joint_reset.pools[name]
            indices = torch.randint(len(pool['root_state']), (n,), device=t.device)
            root, dof, vel, body = [pool[key][indices].clone() for key in
                ('root_state', 'dof_position', 'dof_velocity', 'body_state')]
            box = pool['box_state'][indices, 0].clone()
            if name == 'putDown':
                terminal = pool['terminal_box_xy'][indices].clone()
            t._group_frame[envs[:, None], humans] = pool['source_frame'][indices, None]
        permutation = torch.rand(n, 2, device=t.device).argsort(-1)
        row = torch.arange(n, device=t.device)[:, None]
        root, dof, vel, body = [value[row, permutation] for value in (root, dof, vel, body)]
        t._group_source_role[envs[:, None], humans] = permutation
    else:
        lib = t._motion_lib[name]
        template = torch.where(ontop, 4, 3)
        if t._is_eval:
            clips = lib.sample_motions(n)
            times = lib.sample_time_rsi(clips)
        else:
            clips, times = t._size_rsi_cache.sample(name, envs, humans[:, 0], template)
        pos, rot, dof, lin, ang, vel, _ = lib.get_motion_state(clips, times)
        root = torch.cat((pos, rot, lin, ang), -1)[:, None]
        body = torch.cat(lib.get_motion_state_max(clips, times), -1)[:, None]
        dof, vel = dof[:, None], vel[:, None]
        if name != 'loco':
            box[:, :3], box[:, 3:7] = lib.get_obj_motion_state(clips, times)
            box[:, 2] = torch.maximum(box[:, 2], t._box_size[envs, payload, 2]/2)
        else:
            delta = -root[:, 0, :3].clone()
            delta[:, 0] -= 1.+torch.rand(n, device=t.device)
            delta[:, 2] = 0.
            root[..., :3] += delta[:, None]
            body[..., :3] += delta[:, None, None]
        if name == 'putDown':
            terminal = lib.get_obj_motion_state(clips, lib.get_motion_length(clips))[0][:, :2]
        t._group_frame[envs, humans[:, 0]] = (times/lib._motion_dt[clips]).long()
        slots = (envs, humans[:, 0])
        if name in t._reset_ref_slots:
            slots = tuple(torch.cat((a, b)) for a, b in zip(t._reset_ref_slots[name], slots))
            clips = torch.cat((t._reset_ref_motion_ids[name], clips))
            times = torch.cat((t._reset_ref_motion_times[name], times))
        t._reset_ref_slots[name] = slots
        t._reset_ref_motion_ids[name] = clips
        t._reset_ref_motion_times[name] = times
    center = box[:, :3].clone()
    center[:, 2] = 0.
    root[..., :3] -= center[:, None]
    body[..., :3] -= center[:, None, None]
    box[:, :3] -= center
    terminal -= center[:, :2]
    return root, dof, vel, body, box, terminal


def sample_local_goals(box, support_size, body, terminal, skill, ontop):
    goal = torch.zeros(len(box), 3, device=box.device)
    if skill == 'putDown':
        goal[:, :2] = terminal
        return goal
    pending = torch.arange(len(box), device=box.device)
    for _ in range(64):
        angle = torch.rand(len(pending), device=box.device)*2*torch.pi
        radius = 2.2+.6*torch.rand(len(pending), device=box.device)
        goal[pending, :2] = box[pending, :2] + radius[:, None]*torch.stack((angle.cos(), angle.sin()), -1)
        bad = ((goal[pending, None, :2]-body[pending, :, 0, :2]).norm(dim=-1) < 1.).any(-1)
        support_center = torch.cat((goal[pending, :2], support_size[pending, 2:3]/2), -1)
        inside = ((body[pending, ..., :3]-support_center[:, None, None]).abs() <
            support_size[pending, None, None]/2+.1).all(-1).flatten(1).any(-1)
        bad |= ontop[pending] & inside
        pending = pending[bad]
        if not len(pending):
            return goal
    raise RuntimeError('No clear group-local Goal360 placement')


def place_groups(t, ids):
    roots = t._humanoid_root_states[ids].clone()
    bodies = t._kinematic_humanoid_rigid_body_states[ids].clone()
    boxes = t._box_states[ids].clone()
    goals = t._tar_pos[ids].clone()
    platforms = t._joint_loco_platform.states[ids].clone()
    pending = torch.arange(len(ids), device=t.device)
    for _ in range(128):
        envs = ids[pending]
        leaders = t._group_leader[envs]
        yaw = torch.rand(len(envs), 4, device=t.device)*2*torch.pi
        shift = torch.zeros(len(envs), 4, 3, device=t.device)
        shift[..., :2] = (torch.rand(len(envs), 4, 2, device=t.device)-.5)*8.
        shift += t._env_origins[envs, None]
        human_yaw = yaw.gather(1, leaders)
        human_shift = shift.gather(1, leaders[..., None].expand(-1, -1, 3))
        t._humanoid_root_states[envs] = transform_states(roots[pending], human_yaw, human_shift)
        t._kinematic_humanoid_rigid_body_states[envs] = transform_states(bodies[pending], human_yaw, human_shift)
        from utils.joint_carry_rsi import rotate_vectors
        t._tar_pos[envs] = rotate_vectors(goals[pending], yaw)+shift
        for leader in range(4):
            active = leaders[:, leader] == leader
            rows = pending[active]
            es = envs[active]
            for logical in (leader, leader+4):
                physical = t._logical_box_order[es, logical]
                t._box_states[es, physical] = transform_states(boxes[rows, physical], yaw[active, leader], shift[active, leader])
            paired = active & t._group_joint[envs, leader]
            pe = envs[paired]
            slot = t._logical_box_order[pe, leader]-8
            on = t._joint_loco_platform.active[pe, slot]
            pe, slot = pe[on], slot[on]
            t._joint_loco_platform.states[pe, slot] = transform_states(
                platforms[pending[paired][on], slot], yaw[paired, leader][on], shift[paired, leader][on])
        bad = scene_invalid(t, envs)
        t._sampling_retries += bad.sum()
        pending = pending[bad]
        if not len(pending):
            return
    t._sampling_failures += len(pending)
    raise RuntimeError('Four-agent group placement failed after 128 attempts')


def scene_invalid(t, ids):
    packet = task_graph_packet(t.relation_runtime.graph, t.num_envs)[ids]
    valid = entity_validity(packet, 4, 8)[:, 4:12]
    boxes = t._logical_box_values(t._box_states, ids)
    sizes = t._logical_box_values(t._box_size, ids)
    bodies = t._kinematic_humanoid_rigid_body_states[ids, ..., :3]
    groups = t._group_leader[ids]
    object_group = torch.arange(8, device=t.device)%4
    cross = groups[:, :, None] != groups[:, None, :]
    distance = (bodies[:, :, None, 0, :2]-bodies[:, None, :, 0, :2]).norm(dim=-1)
    bad = ((distance < .8) & cross).flatten(1).any(-1)
    relative = bodies[:, :, None]-boxes[:, None, :, None, :3]
    from env.tasks.multi_agent.edge_interaction_reward import quat_rotate
    q = boxes[:, None, :, None, 3:7].expand(*relative.shape[:-1], 4).clone()
    q[..., :3] *= -1
    local = quat_rotate(q, relative)
    inside = (local.abs() < sizes[:, None, :, None]/2+.03).all(-1).any(-1)
    unrelated = groups[..., None] != object_group
    bad |= (inside & unrelated & valid[:, None]).flatten(1).any(-1)
    radius = sizes[..., :2].norm(dim=-1)/2
    distance = (boxes[:, :, None, :2]-boxes[:, None, :, :2]).norm(dim=-1)
    cross_objects = object_group[:, None] != object_group[None]
    occupied = valid[:, :, None] & valid[:, None, :] & cross_objects
    bad |= ((distance < radius[:, :, None]+radius[:, None]+.1) & occupied).flatten(1).any(-1)
    tasks = owner_tasks(packet, 4).long()
    row = torch.arange(len(ids), device=t.device)[:, None]
    goals = t._tar_pos[ids][row, groups]
    target_radius = radius.gather(1, torch.where(tasks[..., 1] == 3, groups+4, groups))
    distance = (goals[:, :, None, :2]-boxes[:, None, :, :2]).norm(dim=-1)
    bad |= ((distance < target_radius[..., None]+radius[:, None]+.1) & unrelated & valid[:, None]).flatten(1).any(-1)
    distance = (goals[:, :, None, :2]-bodies[:, None, :, 0, :2]).norm(dim=-1)
    bad |= ((distance < 1.) & cross).flatten(1).any(-1)
    distance = (goals[:, :, None, :2]-goals[:, None, :, :2]).norm(dim=-1)
    bad |= ((distance < target_radius[:, :, None]+target_radius[:, None]+.1) & cross).flatten(1).any(-1)
    bounds = (bodies[..., :2]-t._env_origins[ids, None, None, :2]).abs().amax(-1).amax(-1).amax(-1)
    bad |= bounds > t.cfg['env']['envSpacing']-.5
    return bad
