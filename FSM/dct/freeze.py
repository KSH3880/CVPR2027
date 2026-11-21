import torch
from isaacgym import gymtorch

from controller import StateFreeze
from state import CoordinationState
from utils.evaluation_protocol import predecessors, spatial_success
from utils.task_role_spec import task_graph_packet
from table2_metrics import surface_distance


def delay_people(parent):
    n, m = parent.shape
    selected = torch.arange(n, device=parent.device) % m
    rows, followers = (parent >= 0).nonzero(as_tuple=True)
    selected[rows] = parent[rows, followers]
    return selected


def normalize_initial(task):
    task._humanoid_root_states[..., 7:] = 0
    task._dof_vel[:] = 0
    task._kinematic_humanoid_rigid_body_states[..., 7:] = 0
    task._rigid_body_pos[:] = task._kinematic_humanoid_rigid_body_states[..., :3]
    task._rigid_body_rot[:] = task._kinematic_humanoid_rigid_body_states[..., 3:7]
    task._rigid_body_vel[:] = 0
    task._rigid_body_ang_vel[:] = 0
    actors = torch.arange(len(task._root_states), device=task.device, dtype=torch.int32)
    humans = task._humanoid_actor_ids.flatten().contiguous()
    task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
        gymtorch.unwrap_tensor(actors), len(actors))
    task.gym.set_dof_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._dof_state),
        gymtorch.unwrap_tensor(humans), len(humans))


class DelayFreeze(StateFreeze):
    def __init__(self, task, delay, use_fsm):
        self.task = task
        self.delay_steps = round(delay/task.dt)
        self.use_fsm = use_fsm
        n, m = task.num_envs, task.num_agents
        self.packet = task_graph_packet(task.relation_runtime.graph, n).reshape(n, m, 5).long().clone()
        parent = predecessors(task.relation_runtime.graph, n, task.device)
        self.selected = delay_people(parent)
        self.target = torch.arange(m, device=task.device)[None] == self.selected[:, None]
        self.state = CoordinationState(parent, task._groups.clone())
        self.alive = self.packet[..., 0].bool()
        self.score_alive = self.alive.clone()
        self.saved_root = task._humanoid_root_states.clone()
        self.saved_dof = task._dof_pos.clone()
        self.saved_body = task._kinematic_humanoid_rigid_body_states.clone()
        self.saved_root[..., 7:] = 0
        self.saved_body[..., 7:] = 0
        self.frozen = (self.target & (self.delay_steps > 0)) | (self.state.frozen & use_fsm)
        self.frame = self.projections = 0
        self.max_root_drift = self.max_dof_drift = 0.
        self.max_frozen_root_error = self.max_frozen_dof_error = 0.
        self.in_physics = False
        self.events = []
        self.trace = {key: [] for key in ('frozen', 'delayed', 'fsm_frozen', 'released')}
        self.original_refresh = task._refresh_sim_tensors
        self.original_physics = task._physics_step
        self.original_targets = task._action_to_pd_targets

        def refresh():
            self.original_refresh()
            self.project(submit=self.in_physics)

        def physics():
            self.in_physics = True
            try:
                return self.original_physics()
            finally:
                self.in_physics = False

        def targets(actions):
            return torch.where(self.frozen[..., None], self.saved_dof, self.original_targets(actions))

        task._refresh_sim_tensors = refresh
        task._physics_step = physics
        task._action_to_pd_targets = targets
        self.project(submit=True)
        normalize_initial(task)

    def before_action(self):
        task = self.task
        if self.use_fsm and self.frame:
            boxes, sizes = task._evaluation_box_values(task._box_states), task._evaluation_box_values(task._box_size)
            success = spatial_success(self.packet, task._humanoid_root_states[..., :3], boxes, sizes, task._tar_pos)[..., 1]
            n, m = task.num_envs, task.num_agents
            hands = task._rigid_body_pos[:, :, task._key_body_ids[:2]]
            physical = task._assigned_box_values(task._box_states)[..., :7]
            size = task._assigned_box_values(task._box_size)
            distance = surface_distance(hands.reshape(n*m, 1, 2, 3).double(),
                physical.reshape(n*m, 7).double(), size.reshape(n*m, 3).double()).reshape(n, m, 2)
            self.state.update(success & self.score_alive, (distance < .12).all(-1), self.alive)
        delayed = self.target & (self.frame < self.delay_steps) & self.alive
        fsm = self.state.frozen & self.use_fsm
        frozen = delayed | fsm
        added = frozen & ~self.frozen
        if added.any():
            self.saved_root[added] = task._humanoid_root_states[added]
            self.saved_root[..., 7:][added] = 0
            self.saved_dof[added] = task._dof_pos[added]
            body = torch.cat((task._rigid_body_pos, task._rigid_body_rot,
                torch.zeros_like(task._rigid_body_vel), torch.zeros_like(task._rigid_body_ang_vel)), -1)
            self.saved_body[added] = body[added]
        changed = not torch.equal(frozen, self.frozen)
        self.frozen = frozen
        if changed:
            self.project(submit=True)
        for key, value in dict(frozen=frozen, delayed=delayed, fsm_frozen=fsm, released=self.state.released).items():
            self.trace[key].append(value.cpu().numpy().copy())
        return changed

    def finish(self):
        proof = super().finish()
        proof.update(delay_steps=self.delay_steps, selected=self.selected.cpu().tolist(), use_fsm=self.use_fsm,
            common_initial_velocities_zero=True, release_before_action_step=self.delay_steps+1)
        return proof
