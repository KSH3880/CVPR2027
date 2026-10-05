"""Random two-box delivery with explicit straight routes and frozen ms18."""
import os
import torch
from carry_box_swap.ms18_env import HumanoidMACarryBoxSwapMS18
from carry_box_swap.run import snapshot
from task_allocation.core import sample_layout


class HumanoidTaskAllocationMS18(HumanoidMACarryBoxSwapMS18):
    def _ensure_delivery(self):
        if not hasattr(self, 'allocation_delivered'):
            self.allocation_delivered = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
            self.allocation_locked = torch.zeros_like(self.allocation_delivered)
            self.allocation_stable = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
            self.allocation_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_envs(self, env_ids):
        self._ensure_delivery()
        self.allocation_delivered[env_ids] = False
        self.allocation_locked[env_ids] = False
        self.allocation_stable[env_ids] = 0
        self.allocation_initialized[env_ids] = False
        super()._reset_envs(env_ids)
        if hasattr(self, '_carry_timeout_deadline'):
            self._carry_timeout_deadline[env_ids] = self.max_episode_length

    def apply_layout(self, env_ids):
        # loco_carry only: boxes are unheld on the floor. No carryWith reset.
        if not len(env_ids):
            return
        initial = self.agent_axis(self._initial_humanoid_root_states)
        center = initial[env_ids, :, :2].mean(1)
        rows = self.agent_rows(env_ids)
        sizes = self._box_lib._box_size[rows].reshape(-1, 2, 3)
        clearance = max(float(os.environ.get('ALLOC_CLEARANCE', '1.2')),
                        float(sizes[..., :2].norm(dim=-1).max()) + .3)
        points = sample_layout(len(env_ids), self.device,
                               float(os.environ.get('ALLOC_EXTENT', '3')), clearance) + center[:, None]
        self.agent_axis(self._humanoid_root_states)[env_ids, :, :2] = points[:, :2]
        boxes = self.agent_axis(self._box_states)
        boxes[env_ids, :, :2] = points[:, 2:4]
        boxes[env_ids, :, 2] = sizes[..., 2]*.5 + .05
        boxes[env_ids, :, 7:13] = 0
        self._box_tar_pos[rows, :2] = points[:, 4:6].reshape(-1, 2)
        self._box_tar_pos[rows, 2] = sizes[..., 2].reshape(-1)*.5
        if hasattr(self, '_platform_states'):
            self.agent_axis(self._platform_states)[env_ids, :, :2] = points[:, 2:4]
            self.agent_axis(self._tar_platform_states)[env_ids, :, :2] = points[:, 4:6]

    def allocation_observation(self):
        self._ensure_delivery()
        s = snapshot(self)
        center = self.agent_axis(self._initial_humanoid_root_states)[..., :2].mean(1)
        root, box, goal = s['root'].clone(), s['box'].clone(), s['goal'].clone()
        root[..., :2] -= center[:, None]
        box[..., :2] -= center[:, None]
        goal[..., :2] -= center[:, None]
        state = self.planner_state()
        remaining = (1-self.progress_buf.float()/self.max_episode_length)[:, None, None].expand(-1, 2, 1)
        agent = torch.cat((root, s['assigned_held'][..., None].float(), state.phase[..., None].float()/3, remaining), -1)
        # root=13 + held=1 + phase=1 + remaining=1.
        owner = torch.nn.functional.one_hot(self.box_assignment, 2).transpose(1, 2).float()
        # Convert held flags from agent order to physical box order.
        held = torch.zeros_like(self.allocation_delivered).scatter(1, self.box_assignment, s['assigned_held'])
        box_features = torch.cat((box, self._box_lib._box_size.reshape(self.num_envs, 2, 3),
                                  owner, held[..., None].float(), self.allocation_delivered[..., None].float()), -1)
        return dict(agent=agent, box=box_features, goal=goal,
                    assignment=self.box_assignment.clone(), locked=self.allocation_locked.clone(),
                    delivered=self.allocation_delivered.clone())

    def update_delivery(self):
        s = snapshot(self)
        held = torch.zeros_like(self.allocation_delivered).scatter(1, self.box_assignment, s['assigned_held'])
        self.allocation_locked |= held
        placed = ((s['box'][..., :3]-s['goal']).norm(dim=-1) < .3) & (
            s['box'][..., 7:10].norm(dim=-1) < .2) & ~held & self.allocation_locked
        self.allocation_stable = torch.where(placed, self.allocation_stable+1, torch.zeros_like(self.allocation_stable))
        new = (self.allocation_stable >= 10) & ~self.allocation_delivered
        self.allocation_delivered |= new
        return new.sum(-1), self.allocation_delivered.all(-1)
