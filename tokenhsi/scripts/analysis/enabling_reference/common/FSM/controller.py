from isaacgym import gymapi, gymtorch
import torch

from utils.evaluation_protocol import spatial_success, predecessors
from utils.task_role_spec import task_graph_packet
from table2_metrics import surface_distance
from state import CoordinationState


class StateFreeze:
    def __init__(self, task):
        self.task = task
        n,m = task.num_envs, task.num_agents
        self.packet = task_graph_packet(task.relation_runtime.graph,n).reshape(n,m,5).long().clone()
        parent = predecessors(task.relation_runtime.graph,n,task.device)
        self.state = CoordinationState(parent,task._groups.clone())
        self.alive = self.packet[...,0].bool().clone()
        self.score_alive = self.alive.clone()
        self.saved_root = task._humanoid_root_states.clone()
        self.saved_dof = task._dof_pos.clone()
        self.saved_body = task._kinematic_humanoid_rigid_body_states.clone()
        self.frozen = self.state.frozen.clone()
        self.saved_root[...,7:] = 0
        self.saved_body[...,7:] = 0
        self.events = []
        self.trace = {key:[] for key in ('frame','leader_success','touching','alive','frozen','released','hand_distance')}
        self.frame = 0
        self.projections = 0
        self.max_root_drift = self.max_dof_drift = 0.
        self.max_frozen_root_error = self.max_frozen_dof_error = 0.
        self.in_physics = False
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
            result = self.original_targets(actions)
            return torch.where(self.frozen[...,None],self.saved_dof,result)

        task._refresh_sim_tensors = refresh
        task._physics_step = physics
        task._action_to_pd_targets = targets
        self.log('freeze', self.frozen)
        self.color()
        self.project(submit=True)
        if self.frozen.any():
            # Preserve the pending full reset; Gym uses the last submitted actor indices.
            actors = torch.arange(len(task._root_states),device=task.device,dtype=torch.int32)
            task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),
                                                        gymtorch.unwrap_tensor(actors),len(actors))
            humans = task._humanoid_actor_ids.flatten().contiguous()
            task.gym.set_dof_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._dof_state),
                                                 gymtorch.unwrap_tensor(humans),len(humans))

    def log(self, event, mask):
        if not mask.any():
            return
        people = mask.nonzero(as_tuple=False).cpu().tolist()
        self.events.append(dict(frame=self.frame,event=event,people=people))
        print(f'[FSM] frame={self.frame} {event} env,person={people[:12]} count={len(people)}',flush=True)

    def color(self):
        task = self.task
        if task.viewer is None:
            return
        for env in range(task.num_envs):
            for person,handle in enumerate(task.humanoid_handles_all[env]):
                color = (.15,.65,1.) if self.frozen[env,person] else task._design_colors[env][person]
                for body in range(task.num_bodies):
                    task.gym.set_rigid_body_color(task.envs[env],handle,body,gymapi.MESH_VISUAL,gymapi.Vec3(*color))

    def update(self):
        task = self.task
        boxes = task._evaluation_box_values(task._box_states)
        sizes = task._evaluation_box_values(task._box_size)
        current = spatial_success(self.packet,task._humanoid_root_states[...,:3],boxes,sizes,task._tar_pos)[...,1]
        current &= self.score_alive
        box = task._assigned_box_values(task._box_states)[...,:7]
        size = task._assigned_box_values(task._box_size)
        hands = task._rigid_body_pos[:,:,task._key_body_ids[:2]]
        n,m = task.num_envs,task.num_agents
        distance = surface_distance(hands.reshape(n*m,1,2,3).double(),
                                    box.reshape(n*m,7).double(),size.reshape(n*m,3).double()).reshape(n,m,2)
        touching = (distance<.12).all(-1)
        old_released = self.state.released.clone()
        frozen = self.state.update(current,touching,self.alive)
        added = frozen & ~self.frozen
        removed = self.frozen & ~frozen
        if added.any():
            self.saved_root[added] = task._humanoid_root_states[added]
            self.saved_root[...,7:][added] = 0
            self.saved_dof[added] = task._dof_pos[added]
            body = torch.cat((task._rigid_body_pos,task._rigid_body_rot,
                              torch.zeros_like(task._rigid_body_vel),torch.zeros_like(task._rigid_body_ang_vel)),-1)
            self.saved_body[added] = body[added]
        self.frozen = frozen.clone()
        self.trace['frame'].append(self.frame)
        for name,value in dict(leader_success=current,touching=touching,alive=self.alive,
                               frozen=self.frozen,released=self.state.released,hand_distance=distance).items():
            self.trace[name].append(value.cpu().numpy().copy())
        self.log('freeze',added)
        self.log('release',self.state.released & ~old_released)
        if added.any() or removed.any():
            self.color()
            self.project(submit=True)
        return bool(added.any() or removed.any())

    def project(self, submit):
        task, mask = self.task, self.frozen
        if not mask.any():
            return
        self.max_root_drift = max(self.max_root_drift,float((task._humanoid_root_states[mask][...,:7]-self.saved_root[mask][...,:7]).abs().max()))
        self.max_dof_drift = max(self.max_dof_drift,float((task._dof_pos[mask]-self.saved_dof[mask]).abs().max()))
        task._humanoid_root_states[mask] = self.saved_root[mask]
        task._dof_pos[mask] = self.saved_dof[mask]
        task._dof_vel[mask] = 0
        task._rigid_body_pos[mask] = self.saved_body[mask][...,:3]
        task._rigid_body_rot[mask] = self.saved_body[mask][...,3:7]
        task._rigid_body_vel[mask] = 0
        task._rigid_body_ang_vel[mask] = 0
        if submit:
            actors = task._humanoid_actor_ids[mask].contiguous()
            task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),gymtorch.unwrap_tensor(actors),len(actors))
            task.gym.set_dof_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._dof_state),gymtorch.unwrap_tensor(actors),len(actors))
            self.projections += 1
        self.max_frozen_root_error = max(self.max_frozen_root_error,float((task._humanoid_root_states[mask]-self.saved_root[mask]).abs().max()))
        self.max_frozen_dof_error = max(self.max_frozen_dof_error,float((task._dof_pos[mask]-self.saved_dof[mask]).abs().max()))

    def finish(self):
        task = self.task
        task._refresh_sim_tensors = self.original_refresh
        task._physics_step = self.original_physics
        task._action_to_pd_targets = self.original_targets
        return dict(events=self.events,frames=self.frame,projections=self.projections,
            frozen=self.frozen.cpu().tolist(),released=self.state.released.cpu().tolist(),
            max_substep_root_drift_before_projection=self.max_root_drift,
            max_substep_dof_drift_before_projection=self.max_dof_drift,
            max_frozen_root_error=self.max_frozen_root_error,max_frozen_dof_error=self.max_frozen_dof_error,
            freeze='root and DOF state projection each physics substep; humanoid observation pose/velocity projected too',
            true_kinematic_actor=False,collisions_retained=True,box_state_not_projected=True,
            one_shot_release=True,contact_distance_m=.12,coupled_release='all original participants arrived at least once',
            contact_box='each participant assigned physical box',success_rule='unchanged formal XY100')
