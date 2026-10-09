import torch

from utils.edge_ontop_spec import ON_TOP
from utils.joint_carry_rsi import load_paired_snapshots, rotate_vectors, transform_states


class JointCarryReset:
    def __init__(self, task):
        self.task = task
        self.pools = load_paired_snapshots(task.cfg['env']['jointCarryRsi']['directory'], task.device)
        self.source_frame = torch.full((task.num_envs,), -1, device=task.device, dtype=torch.long)
        self.skill_index = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)

    def reset(self, ids, finalize=True):
        if not len(ids):
            return
        t = self.task
        t._sample_episode_graph(ids)
        t._sample_box_assignments(ids)
        graph = t.relation_runtime.graph
        ontop = ((graph.edge_relation[ids] == ON_TOP) & graph.edge_valid[ids]).any(-1)
        names = t._skill
        if t._is_eval:
            if any(name not in ('loco', 'pickUp', 'carryWith', 'putDown') for name in names):
                raise ValueError('Joint RSI evaluation supports loco/pickUp/carryWith/putDown')
            weights = t._skill_init_prob.expand(len(ids), -1)
            if 'putDown' in names and weights[:, names.index('putDown')].any() and ontop.any():
                raise ValueError('Floor putDown RSI cannot initialize joint ON_TOP')
        else:
            weights = t._template_rsi_weights[torch.where(ontop, 4, 3)]
        skills = torch.multinomial(weights, 1).flatten()
        self.skill_index[ids] = skills
        self.source_frame[ids] = -1
        yaw = torch.rand(len(ids), device=t.device) * (2 * torch.pi)
        shift = t._env_origins[ids]
        boxes = torch.zeros(len(ids), t.num_objects, 13, device=t.device)
        boxes[..., 6] = 1.
        boxes[..., 2] = t._box_size[ids, :, 2] / 2
        boxes[:, 1, :2] = boxes.new_tensor([-3., -2.])
        boxes[:, 3, :2] = boxes.new_tensor([-3., 2.])
        destination = torch.zeros(len(ids), 3, device=t.device)
        destination[:, 0] = 2.2 + .6 * torch.rand(len(ids), device=t.device)
        destination[:, 1] = .6 * (torch.rand(len(ids), device=t.device) - .5)
        destination[:, 2] = t._box_size[ids, 0, 2] / 2
        boxes[:, 2, :2] = destination[:, :2]
        # In AT scenes the unused support must not occupy the goal.
        boxes[~ontop, 2, :2] = boxes.new_tensor([0., 3.5])
        for index, name in enumerate(names):
            selected = (skills == index).nonzero().flatten()
            if not len(selected):
                continue
            envs = ids[selected]
            if name == 'loco':
                root, dof, vel, body = self.locomotion(len(selected))
            else:
                pool = self.pools[name]
                row = torch.randint(len(pool['root_state']), (len(selected),), device=t.device)
                root, dof, vel, body = [pool[key][row] for key in
                    ('root_state', 'dof_position', 'dof_velocity', 'body_state')]
                box = pool['box_state'][row, 0].clone()
                center = box[:, :3].clone()
                center[:, 2] = 0.
                root = root.clone(); body = body.clone()
                root[..., :3] -= center[:, None]
                body[..., :3] -= center[:, None, None]
                box[:, :3] -= center
                boxes[selected, 0] = box
                self.source_frame[envs] = pool['source_frame'][row]
            t._humanoid_root_states[envs] = transform_states(root, yaw[selected], shift[selected])
            t._dof_pos[envs], t._dof_vel[envs] = dof, vel
            t._kinematic_humanoid_rigid_body_states[envs] = transform_states(body, yaw[selected], shift[selected])
            t._every_env_init_dof_pos[envs] = dof
            t._stage2_rsi_counts[index] += 2 * len(envs)
        t._box_states[ids] = transform_states(boxes, yaw, shift)
        goal = rotate_vectors(destination, yaw) + shift
        t._tar_pos[ids] = goal[:, None]
        if not finalize:
            return
        self.commit(ids)
        t._refresh_sim_tensors()
        t._reset_relation_history(ids)
        t._compute_observations(ids)
        t._compute_amp_observations(ids)
        t._hist_amp_obs_buf[ids] = t._curr_amp_obs_buf[ids, :, None]

    def locomotion(self, count):
        t = self.task
        lib = t._motion_lib['loco']
        motion = lib.sample_motions(count * 2)
        times = lib.sample_time_rsi(motion)
        pos, rot, dof, vel, ang, dof_vel, _ = lib.get_motion_state(motion, times)
        root = torch.cat((pos, rot, vel, ang), -1).reshape(count, 2, 13)
        body = torch.cat(lib.get_motion_state_max(motion, times), -1).reshape(count, 2, t.num_bodies, 13)
        x, y, z, w = root[..., 3:7].unbind(-1)
        heading = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        yaw = root.new_tensor([torch.pi/2, -torch.pi/2])[None] - heading
        desired = torch.zeros(count, 2, 3, device=t.device)
        desired[..., 1] = (1.0 + torch.rand(count, 2, device=t.device)) * root.new_tensor([-1., 1.])
        center = root[..., :3].clone(); center[..., 2] = 0.
        translation = desired - rotate_vectors(center, yaw)
        return (transform_states(root, yaw, translation), dof.reshape(count, 2, -1),
                dof_vel.reshape(count, 2, -1), transform_states(body, yaw, translation))

    def commit(self, ids):
        from isaacgym import gymtorch
        t = self.task
        actors = [t._humanoid_actor_ids[ids].flatten(), t._box_actor_ids[ids].flatten()]
        if t._enable_markers:
            t._marker_pos[ids] = t._tar_pos[ids]
            graph = t.relation_runtime.graph
            from utils.edge_context_spec import AT
            at = graph.edge_valid[ids] & (graph.edge_relation[ids] == AT)
            for goal in range(t.num_agents):
                visible = (at & (graph.edge_dst[ids] == t.num_agents + t.num_objects + goal)).any(-1)
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
