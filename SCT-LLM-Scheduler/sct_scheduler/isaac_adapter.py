"""Stage1/2 tensor adapter, adapted from FSM/controller.py (see provenance.json).

Import only inside an Isaac Gym process, after isaacgym and before torch elsewhere.
No source imports, writes, launcher, process control or MPS management.
"""
import os
import subprocess

from .core import require, validate_response
from .runtime import ReleaseClock


def check_gpu_mapping(physical_gpu):
    require(type(physical_gpu) is int and physical_gpu in (6, 7), 'Only physical GPU 6 or 7 permitted')
    # UUID visibility avoids ambiguous ordinal remapping under an existing MPS server.
    uuid = subprocess.check_output(
        ['nvidia-smi', '-i', str(physical_gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    require(uuid.startswith('GPU-') and os.environ.get('CUDA_VISIBLE_DEVICES') == uuid,
            'Set CUDA_VISIBLE_DEVICES to the selected physical GPU UUID before starting Python')


class InitialReleaseFreeze:
    def __init__(self, task, requests, responses, agent_order, *, physical_gpu):
        check_gpu_mapping(physical_gpu)
        from isaacgym import gymtorch
        import torch
        require(str(task.device) in ('cuda:0', 'cuda'), 'Use logical cuda:0 with one visible GPU UUID')
        require(not hasattr(task, '_sct_initial_release_adapter'), 'Freeze adapter already installed')
        require(len(requests) == len(responses) == len(agent_order) == task.num_envs, 'Batch/environment mismatch')
        self.clocks = []
        self.agent_order = [list(row) for row in agent_order]
        for request, response, order in zip(requests, responses, self.agent_order):
            validate_response(request, response)
            require(response['status'] == 'ok', 'Non-ok schedule cannot run')
            releases = {r['agent_id']: r['release_time_s'] for r in response['schedule']}
            require(len(order) == task.num_agents and len(set(order)) == len(order) and set(order) == set(releases),
                    'Explicit actor-slot to agent-ID bijection required')
            self.clocks.append(ReleaseClock(releases, task.dt))
        self.task, self.torch, self.gymtorch = task, torch, gymtorch
        ids = task._humanoid_actor_ids
        require(ids.dtype == torch.int32 and tuple(ids.shape) == (task.num_envs, task.num_agents), 'Invalid actor IDs')
        require(bool((ids >= 0).all()) and bool((ids < len(task._root_states)).all()), 'Actor IDs out of range')
        require(ids.unique().numel() == ids.numel(), 'Actor IDs must be unique')
        self.saved_root = task._humanoid_root_states.clone()
        self.saved_dof = task._dof_pos.clone()
        self.saved_body = task._kinematic_humanoid_rigid_body_states.clone()
        self.saved_root[..., 7:] = 0
        self.saved_body[..., 7:] = 0
        self.frame = 0
        self.in_physics = False
        self.closed = False
        self.projections = 0
        self.max_root_drift = self.max_dof_drift = 0.0
        self.original_refresh = task._refresh_sim_tensors
        self.original_physics = task._physics_step
        self.original_targets = task._action_to_pd_targets
        self.frozen = self._mask(0)

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
            return torch.where(self.frozen[..., None], self.saved_dof, result)

        task._refresh_sim_tensors = refresh
        task._physics_step = physics
        task._action_to_pd_targets = targets
        task._sct_initial_release_adapter = self
        try:
            self.project(submit=True)
            if self.frozen.any():
                # Gym keeps the last submitted actor indices: retain pending full reset.
                actors = torch.arange(len(task._root_states), device=task.device, dtype=torch.int32)
                humans = ids.flatten().contiguous()
                task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                                                            gymtorch.unwrap_tensor(actors), len(actors))
                task.gym.set_dof_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._dof_state),
                                                     gymtorch.unwrap_tensor(humans), len(humans))
        except Exception:
            self.close()
            raise

    def _mask(self, frame):
        masks = [clock.advance(frame) for clock in self.clocks]
        return self.torch.tensor([[mask[a] for a in order] for mask, order in zip(masks, self.agent_order)],
                                 dtype=self.torch.bool, device=self.task.device)

    def before_action(self, frame):
        """Call BEFORE policy inference, using episode control-step index (not wall time)."""
        require(not self.closed, 'Adapter closed')
        self.frozen = self._mask(frame)
        self.frame = frame
        self.project(submit=True)

    def mask_actions(self, actions):
        return actions.reshape(self.task.num_envs, self.task.num_agents, -1).masked_fill(
            self.frozen[..., None], 0).reshape_as(actions)

    def project(self, submit):
        task, mask, gymtorch = self.task, self.frozen, self.gymtorch
        if not mask.any():
            return
        self.max_root_drift = max(self.max_root_drift, float((task._humanoid_root_states[mask][..., :7] - self.saved_root[mask][..., :7]).abs().max()))
        self.max_dof_drift = max(self.max_dof_drift, float((task._dof_pos[mask] - self.saved_dof[mask]).abs().max()))
        task._humanoid_root_states[mask] = self.saved_root[mask]
        task._dof_pos[mask] = self.saved_dof[mask]
        task._dof_vel[mask] = 0
        task._rigid_body_pos[mask] = self.saved_body[mask][..., :3]
        task._rigid_body_rot[mask] = self.saved_body[mask][..., 3:7]
        task._rigid_body_vel[mask] = 0
        task._rigid_body_ang_vel[mask] = 0
        if submit:
            actors = task._humanoid_actor_ids[mask].contiguous()
            task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                                                        gymtorch.unwrap_tensor(actors), len(actors))
            task.gym.set_dof_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._dof_state),
                                                 gymtorch.unwrap_tensor(actors), len(actors))
            self.projections += 1

    def close(self):
        """Restore Python hooks only. Does not terminate/reset any process, GPU or MPS server."""
        if not self.closed:
            self.task._refresh_sim_tensors = self.original_refresh
            self.task._physics_step = self.original_physics
            self.task._action_to_pd_targets = self.original_targets
            del self.task._sct_initial_release_adapter
            self.closed = True
        return {'release_events': [clock.events for clock in self.clocks], 'frame': self.frame,
                'quantized_release_times_s': [clock.quantized_release_times() for clock in self.clocks],
                'projections': self.projections, 'max_root_drift_before_projection': self.max_root_drift,
                'max_dof_drift_before_projection': self.max_dof_drift,
                'true_kinematic_actor': False, 'collisions_retained': True, 'box_state_projected': False,
                'runtime_gate': False, 'release_quantization': 'ceil to control dt; never early'}
