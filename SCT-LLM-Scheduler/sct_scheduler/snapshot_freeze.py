"""Initial suspension with one-time restoration of the sampled motion state.

No FSM state, success detector, or contact gate. Dynamic actor collisions remain
enabled; suspension uses simulator tensor writes, not a kinematic actor flag.
"""
from .core import require, validate_response
from .executable import validate_executable
from .isaac_adapter import check_gpu_mapping
from .runtime import ReleaseClock


class SnapshotFreeze:
    def __init__(self, task, requests, responses, agent_order, *, physical_gpu,
                 execution_only=False):
        check_gpu_mapping(physical_gpu)
        import torch
        from isaacgym import gymtorch
        require(not hasattr(task, '_sct_initial_release_adapter'), 'Adapter already installed')
        require(len(requests) == len(responses) == len(agent_order) == task.num_envs,
                'Environment count mismatch')
        self.task, self.torch, self.gymtorch = task, torch, gymtorch
        self.agent_order = agent_order
        self.clocks = []
        for request, response, order in zip(requests, responses, agent_order):
            (validate_executable if execution_only else validate_response)(request, response)
            times = {r['agent_id']: r['release_time_s'] for r in response['schedule']}
            require(len(order) == task.num_agents and len(set(order)) == len(order)
                    and set(order) == set(times), 'Agent mapping mismatch')
            self.clocks.append(ReleaseClock(times, task.dt))
        self.root = task._humanoid_root_states.clone()
        self.position = task._dof_pos.clone()
        self.velocity = task._dof_vel.clone()
        self.body = task._kinematic_humanoid_rigid_body_states.clone()
        self.saved_root = self.root.clone()
        self.saved_root[..., 7:] = 0
        self.frame, self.closed, self.inside_step = 0, False, False
        self.restore_error = 0.0
        self.restored = torch.zeros(task.num_envs, task.num_agents, dtype=torch.bool,
                                    device=task.device)
        self.frozen = self._mask(0)
        self.refresh = task._refresh_sim_tensors
        self.physics = task._physics_step
        self.pd = task._action_to_pd_targets

        def refresh():
            self.refresh()
            self._hold()
            if self.inside_step:
                self._submit(self.frozen)

        def physics():
            self.inside_step = True
            try:
                return self.physics()
            finally:
                self.inside_step = False

        def pd(actions):
            return torch.where(self.frozen[..., None], self.position, self.pd(actions))

        task._refresh_sim_tensors = refresh
        task._physics_step = physics
        task._action_to_pd_targets = pd
        task._sct_initial_release_adapter = self
        # Make the first policy observation agree with the sampled reset pose.
        all_agents = torch.ones_like(self.frozen)
        self._body_state(all_agents, resume=True)
        self._hold()
        self._submit(all_agents, full_reset=True)

    def _mask(self, frame):
        masks = [c.advance(frame) for c in self.clocks]
        return self.torch.tensor([[mask[a] for a in order]
                                 for mask, order in zip(masks, self.agent_order)],
                                device=self.task.device, dtype=self.torch.bool)

    def _body_state(self, mask, *, resume):
        t = self.task
        t._rigid_body_pos[mask] = self.body[mask][..., :3]
        t._rigid_body_rot[mask] = self.body[mask][..., 3:7]
        t._rigid_body_vel[mask] = self.body[mask][..., 7:10] if resume else 0
        t._rigid_body_ang_vel[mask] = self.body[mask][..., 10:13] if resume else 0

    def _hold(self):
        t, m = self.task, self.frozen
        t._humanoid_root_states[m] = self.saved_root[m]
        t._dof_pos[m] = self.position[m]
        t._dof_vel[m] = 0
        self._body_state(m, resume=False)

    def _submit(self, humans, *, full_reset=False):
        if not full_reset and not bool(humans.any()):
            return
        t, g = self.task, self.gymtorch
        ids = t._humanoid_actor_ids[humans].contiguous()
        roots = (self.torch.arange(len(t._root_states), device=t.device,
                                 dtype=self.torch.int32) if full_reset else ids)
        t.gym.set_actor_root_state_tensor_indexed(t.sim, g.unwrap_tensor(t._root_states),
                                                 g.unwrap_tensor(roots), len(roots))
        t.gym.set_dof_state_tensor_indexed(t.sim, g.unwrap_tensor(t._dof_state),
                                          g.unwrap_tensor(ids), len(ids))

    def before_action(self, frame):
        require(not self.closed, 'Adapter closed')
        new_mask = self._mask(frame)
        leaving = self.frozen & ~new_mask
        require(not bool((new_mask & ~self.frozen).any()), 'Refreeze forbidden')
        t = self.task
        if bool(leaving.any()):
            require(not bool(self.restored[leaving].any()), 'Repeated restoration')
            t._humanoid_root_states[leaving] = self.root[leaving]
            t._dof_pos[leaving] = self.position[leaving]
            t._dof_vel[leaving] = self.velocity[leaving]
            self._body_state(leaving, resume=True)
            self.restored |= leaving
            self.restore_error = max(self.restore_error,
                float((t._humanoid_root_states[leaving]-self.root[leaving]).abs().max()),
                float((t._dof_vel[leaving]-self.velocity[leaving]).abs().max()))
        self.frozen, self.frame = new_mask, frame
        self._hold()
        self._submit(self.torch.ones_like(new_mask) if frame == 0 else new_mask | leaving,
                     full_reset=frame == 0)

    def mask_actions(self, actions):
        return actions.reshape(self.task.num_envs, self.task.num_agents, -1).masked_fill(
            self.frozen[..., None], 0).reshape_as(actions)

    def close(self):
        if not self.closed:
            self.task._refresh_sim_tensors = self.refresh
            self.task._physics_step = self.physics
            self.task._action_to_pd_targets = self.pd
            del self.task._sct_initial_release_adapter
            self.closed = True
        return dict(implementation='snapshot_resume_v1',
                    release_events=[c.events for c in self.clocks], frame=self.frame,
                    quantized_release_times_s=[c.quantized_release_times() for c in self.clocks],
                    restored_agents=self.restored.cpu().tolist(),
                    max_restoration_error=self.restore_error,
                    restore_initial_velocities=True, true_kinematic_actor=False,
                    collisions_retained=True, box_state_projected=False, runtime_gate=False)
