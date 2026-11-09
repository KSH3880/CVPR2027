"""Evaluation-only single-person scenes. Never selected by the training path."""
import numpy as np
import torch
from push_reference.env.tasks.multi_agent.humanoid_ma_stage1_jointpush import HumanoidMAStage1JointPush

from env.cvpr2027_scene import PRESETS, sample_scene


class CVPRSingleTask(HumanoidMAStage1JointPush):
    def __init__(self, cfg, *args, **kwargs):
        self.cvpr_task = cfg['cvpr_task']
        self.cvpr_rng = np.random.RandomState(int(cfg['seed']))
        super().__init__(cfg, *args, **kwargs)
        if self.num_agents != 1 or self.num_objects != 2 or not self._is_eval:
            raise ValueError('CVPRSingleTask is evaluation-only: one person, two object slots')
        if abs(self.dt - 1/30) > 1e-6:
            raise ValueError('Protocol requires 30 Hz policy control')

    def _reset_boxes(self, ids):
        roots = self._humanoid_root_states[ids, 0, :3].cpu().numpy()
        sizes = self._box_size[ids].cpu().numpy()
        boxes, goals = sample_scene(self.cvpr_rng, roots, sizes, self.cvpr_task)
        if not hasattr(self, 'cvpr_observed_initial_boxes'):
            self.cvpr_observed_initial_boxes = self._box_states.clone()
            self.cvpr_goals = self._tar_pos.clone()
        sampled = torch.as_tensor(boxes, device=self.device)
        self.cvpr_observed_initial_boxes[ids] = sampled
        self._box_states[ids] = sampled
        self.cvpr_goals[ids, 0] = torch.as_tensor(goals, device=self.device)
        if self.cvpr_task != 'on_top':
            # Keep actor/shape allocation and token metadata; remove only collision
            # participation by parking the non-working object outside the scene.
            self._box_states[ids, 1, 0] = self._env_origins[ids, 0]+10000
            self._box_states[ids, 1, 2] = -10000
        self._platform_pos[ids] = self._platform_default_pos[ids]

    def _reset_task(self, ids):
        self._tar_pos[ids] = self.cvpr_goals[ids]
        self._tar_platform_pos[ids] = self._tar_platform_default_pos[ids]

    def _configure_ontop_targets(self, ids):
        pass  # sample_scene already placed the actual support and checked layout.

    def _ontop_scene_infeasible(self, ids):
        boxes = self.cvpr_observed_initial_boxes[ids]
        sizes = self._box_size[ids]
        body = self._kinematic_humanoid_rigid_body_states[ids, 0, :, :3]
        delta = body[:, None]-boxes[:, :, None, :3]
        yaw = 2*torch.atan2(boxes[..., 5], boxes[..., 6])
        x = yaw.cos()[..., None]*delta[..., 0]+yaw.sin()[..., None]*delta[..., 1]
        y = -yaw.sin()[..., None]*delta[..., 0]+yaw.cos()[..., None]*delta[..., 1]
        margin = .05
        collision = ((x.abs() < sizes[..., 0, None]/2+margin)
                     & (y.abs() < sizes[..., 1, None]/2+margin)
                     & (delta[..., 2].abs() < sizes[..., 2, None]/2+margin))
        return collision.any(-1).any(-1)
