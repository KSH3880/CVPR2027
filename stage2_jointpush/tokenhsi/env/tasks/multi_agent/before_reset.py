import torch

from env.tasks.multi_agent.edge_interaction_reward import quat_rotate
from utils.edge_ontop_spec import ON_TOP


def quat_multiply(a, b):
    return torch.cat((a[..., 3:]*b[..., :3] + b[..., 3:]*a[..., :3]
        + torch.cross(a[..., :3], b[..., :3], dim=-1),
        a[..., 3:]*b[..., 3:] - (a[..., :3]*b[..., :3]).sum(-1, keepdim=True)), -1)


def align_transform(source, target):
    conjugate = source[..., 3:7] * source.new_tensor([-1., -1., -1., 1.])
    rotation = quat_multiply(target[..., 3:7], conjugate)
    return rotation, target[..., :3] - quat_rotate(rotation, source[..., :3])


def transform_states(states, rotation, translation):
    while rotation.ndim < states.ndim:
        rotation = rotation.unsqueeze(-2)
        translation = translation.unsqueeze(-2)
    rotation = rotation.expand(*states.shape[:-1], 4)
    result = states.clone()
    result[..., :3] = quat_rotate(rotation, states[..., :3]) + translation
    result[..., 3:7] = quat_multiply(rotation, states[..., 3:7])
    result[..., 7:10] = quat_rotate(rotation, states[..., 7:10])
    result[..., 10:13] = quat_rotate(rotation, states[..., 10:13])
    return result


def reset_boxes(t, ids):
    t._reset_all_boxes_random_arena(ids)
    t._before_amp_rotation[ids] = t._before_amp_rotation.new_tensor([0., 0., 0., 1.])
    t._before_amp_translation[ids] = t._env_origins[ids, None] + t._agent_spawn_offsets[None]
    agents = torch.arange(2, device=t.device)[None].expand(len(ids), -1)
    physical = t._scenario_physical_object(ids.repeat_interleave(2), agents.flatten()).reshape(-1, 2)
    source = t._box_states[ids[:, None], physical].clone()
    support_reference = source.clone()
    skills = torch.zeros(len(ids), 2, dtype=torch.long, device=t.device)
    for skill, slots in t._reset_ref_slots.items():
        envs, owners = slots
        rows = torch.searchsorted(ids, envs)
        skills[rows, owners] = t._skill.index(skill)
        offset = t._agent_spawn_offsets[owners] + t._env_origins[envs]
        box = source[rows, owners].clone()
        size = t._box_size[envs, physical[rows, owners]]
        if skill == 'loco':
            angle = torch.rand(len(rows), device=t.device)*2*torch.pi
            lo, hi = t.cfg['env']['box']['reset']['ownerLocoDistanceRange']
            distance = lo + (hi-lo)*torch.rand(len(rows), device=t.device)
            box[:, :2] = t._humanoid_root_states[envs, owners, :2] + distance[:, None]*torch.stack((angle.cos(), angle.sin()), -1)
            box[:, 2] = size[:, 2]/2
        else:
            lib = t._motion_lib[skill]
            motion = t._reset_ref_motion_ids[skill]
            times = t._reset_ref_motion_times[skill]
            if skill in ('sit', 'climb'):
                pos, rot = lib.get_obj_motion_state_single_frame(motion)
            else:
                pos, rot = lib.get_obj_motion_state(motion, times)
            box[:, :3], box[:, 3:7] = pos+offset, rot
            box[:, 2] = size[:, 2]/2 if skill in ('sit', 'climb') else torch.maximum(box[:, 2], size[:, 2]/2)
            if skill == 'putDown':
                graph = t.relation_runtime.graph
                mask = graph.edge_valid[envs] & (graph.edge_owner[envs] == owners[:, None]) & (graph.edge_relation[envs] == ON_TOP)
                selected = mask.any(-1)
                if selected.any():
                    end, _ = lib.get_obj_motion_state(motion, lib.get_motion_length(motion))
                    target = (graph.edge_dst[envs].masked_fill(~mask, 0).amax(-1)-2).clamp_min(0)
                    pivot = support_reference[rows, owners].clone()
                    pivot[:, :3] = end+offset
                    pivot[:, 2] = t._box_size[envs, target, 2]/2
                    pivot[:, 3:7] = pivot.new_tensor([0., 0., 0., 1.])
                    pivot[:, 7:] = 0.
                    support_reference[rows, owners] = pivot
                    independent = selected & (t._before_families[envs] == 0)
                    t._box_states[envs[independent], target[independent]] = pivot[independent]
        box[:, 7:] = 0.
        source[rows, owners] = box

    dependent = t._before_families[ids] > 0
    shared = dependent & (t._before_families[ids] != 3)
    advanced = skills[:, 1] != t._skill.index('loco') if 'loco' in t._skill else torch.ones_like(shared)
    stack_putdown = dependent & (t._before_families[ids] == 3)
    stack_putdown &= (skills[:, 1] == t._skill.index('putDown')) if 'putDown' in t._skill else False
    rows = ((shared & advanced) | stack_putdown).nonzero().flatten()
    if len(rows):
        pivot = torch.where(stack_putdown[rows, None], support_reference[rows, 1], source[rows, 1])
        rotation, translation = align_transform(pivot, source[rows, 0])
        envs = ids[rows]
        t._humanoid_root_states[envs, 1] = transform_states(t._humanoid_root_states[envs, 1], rotation, translation)
        t._kinematic_humanoid_rigid_body_states[envs, 1] = transform_states(
            t._kinematic_humanoid_rigid_body_states[envs, 1], rotation, translation)
        source[rows, 1] = transform_states(source[rows, 1], rotation, translation)
        t._before_amp_rotation[envs, 1] = rotation
        t._before_amp_translation[envs, 1] = quat_rotate(rotation,
            t._agent_spawn_offsets[1]+t._env_origins[envs]) + translation
    t._box_states[ids, physical[:, 0]] = source[:, 0]
    # A owns the one shared physical state; B's locomotion reference never overwrites it.
    rows = (~shared).nonzero().flatten()
    t._box_states[ids[rows], physical[rows, 1]] = source[rows, 1]


def transform_amp(t, slots, root_pos, root_rot, root_vel, root_ang_vel, key_pos):
    count = t._num_amp_obs_steps-1
    rotation = t._before_amp_rotation[slots].repeat_interleave(count, 0)
    translation = t._before_amp_translation[slots].repeat_interleave(count, 0)
    return (quat_rotate(rotation, root_pos)+translation,
        quat_multiply(rotation, root_rot), quat_rotate(rotation, root_vel),
        quat_rotate(rotation, root_ang_vel),
        quat_rotate(rotation[:, None].expand(-1, key_pos.shape[1], -1), key_pos)+translation[:, None])


def commit(t, ids):
    from isaacgym import gymtorch
    from utils.edge_context_spec import AT
    actors = [t._humanoid_actor_ids[ids].flatten(), t._box_actor_ids[ids].flatten()]
    if t._enable_markers:
        t._marker_pos[ids] = t._tar_pos[ids]
        graph = t.relation_runtime.graph
        at = graph.edge_valid[ids] & (graph.edge_relation[ids] == AT)
        for goal in range(t.num_agents):
            visible = (at & (graph.edge_dst[ids] == t.num_agents+t.num_objects+goal)).any(-1)
            t._marker_pos[ids[~visible], goal, 2] = 20.
        actors.append(t._marker_actor_ids[ids].flatten())
    actors = torch.cat(actors).contiguous()
    humans = t._humanoid_actor_ids[ids].flatten().contiguous()
    t.gym.set_actor_root_state_tensor_indexed(t.sim, gymtorch.unwrap_tensor(t._root_states),
                                            gymtorch.unwrap_tensor(actors), len(actors))
    t.gym.set_dof_state_tensor_indexed(t.sim, gymtorch.unwrap_tensor(t._dof_state),
                                     gymtorch.unwrap_tensor(humans), len(humans))
    t.progress_buf[ids] = 0
    t.reset_buf[ids] = 0
    t._terminate_buf[t._flat_slot_ids(ids)] = 0
    if t._is_eval:
        t._success_buf[t._flat_slot_ids(ids)] = 0
        t._precision_buf[t._flat_slot_ids(ids)] = float('inf')
