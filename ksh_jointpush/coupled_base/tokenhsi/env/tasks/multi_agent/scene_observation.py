"""SceneObservationMixin: shared fixed Stage1/BEFORE environment behavior."""
import torch
from env.tasks.multi_agent.scene_features import build_gta_pose_records
from isaacgym.torch_utils import quat_rotate
from utils import torch_utils

class SceneObservationMixin:
    def _task_goal_positions(self, env_ids=None):
        return self._tar_pos if env_ids is None else self._tar_pos[env_ids]

    def get_relation_suffix_size(self):
        return self._context_suffix_width

    def get_object_obs_size(self):
        return 3 + 3 + 3 + 6 + 24

    def get_goal_obs_size(self):
        return 3 + 3

    def get_clean_object_obs_size(self):
        return 3 + 3 + 24

    def get_clean_goal_obs_size(self):
        return 1

    def get_scene_kinematic_size(self):
        return 3 + 4

    def get_scene_arena_scale(self):
        """Scene spacing used by runner configuration."""
        return self._arena_scale

    def get_geometry_obs_size(self):
        return 3 + 6 + 3 + 3

    def get_scene_entity_sizes(self):
        return [
            self.get_clean_humanoid_obs_size(),
            self.get_clean_object_obs_size(),
            self.get_clean_goal_obs_size(),
        ]

    def get_scene_normalized_entity_sizes(self):
        """Normalize intrinsic H/O nodes; constant Target and GTA poses pass through."""
        return [
            self.get_clean_humanoid_self_obs_size(),
            self.get_clean_object_obs_size(),
            0,
        ]

    def get_scene_num_tokens(self):
        return 2 * self.num_agents + self.num_objects

    def get_task_obs_size(self):
        if not self._enable_task_obs:
            return 0
        return (
            self.num_objects * self.get_object_obs_size()
            + self.num_agents * self.get_goal_obs_size()
        )

    def get_obs_size(self):
        node_width = (
            self.num_agents * self.get_clean_humanoid_obs_size()
            + self.num_objects * self.get_clean_object_obs_size()
            + self.num_agents * self.get_clean_goal_obs_size()
        )
        kinematic_width = self.get_scene_num_tokens() * self.get_scene_kinematic_size()
        return node_width + kinematic_width + self.get_relation_suffix_size()

    def _compute_observations(self, env_ids=None):
        obs = self._compute_clean_scene_obs(env_ids)
        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_clean_scene_obs(self, env_ids=None):
        """Build one clean, compact scene observation per environment.

        Layout: [M*H223 | O*O30 | M*T1 | (2M+O)*pose7].  Pose records are
        env-local position plus local-to-env orientation and bypass node RMS.
        """
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            box_states = self._logical_box_values(self._box_states)
            box_bps = self._logical_box_values(self._box_bps)
            tar_pos = self._task_goal_positions()
            origins = self._env_origins
        else:
            kin = self._kinematic_humanoid_rigid_body_states[env_ids]
            body_pos = kin[..., 0:3]
            body_rot = kin[..., 3:7]
            body_vel = kin[..., 7:10]
            body_ang_vel = kin[..., 10:13]
            box_states = self._logical_box_values(self._box_states, env_ids)
            box_bps = self._logical_box_values(self._box_bps, env_ids)
            tar_pos = self._task_goal_positions(env_ids)
            origins = self._env_origins[env_ids]
        B, M, O = (body_pos.shape[0], self.num_agents, self.num_objects)
        human_nodes = self._compute_clean_humanoid_nodes(
            body_pos, body_rot, body_vel, body_ang_vel, origins
        )
        box_rot = box_states[..., 3:7]
        box_rot_inv = box_rot.clone()
        box_rot_inv[..., 0:3] *= -1.0
        object_local = torch.cat(
            [
                quat_rotate(
                    box_rot_inv.reshape(-1, 4), box_states[..., 7:10].reshape(-1, 3)
                ).view(B, O, 3),
                quat_rotate(
                    box_rot_inv.reshape(-1, 4), box_states[..., 10:13].reshape(-1, 3)
                ).view(B, O, 3),
                box_bps.reshape(B, O, 24),
            ],
            dim=-1,
        )
        object_nodes = object_local
        goal_nodes = torch.ones(B, M, 1, device=self.device, dtype=body_pos.dtype)
        assert human_nodes.shape == (B, M, self.get_clean_humanoid_obs_size())
        assert object_nodes.shape == (B, O, self.get_clean_object_obs_size())
        assert goal_nodes.shape == (B, M, self.get_clean_goal_obs_size())
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        human_heading = torch_utils.calc_heading_quat(root_rot.reshape(-1, 4)).view(
            B, M, 4
        )
        nodes = torch.cat(
            [
                human_nodes.reshape(B, -1),
                object_nodes.reshape(B, -1),
                goal_nodes.reshape(B, -1),
            ],
            dim=-1,
        )
        entity_poses = build_gta_pose_records(
            root_pos,
            human_heading,
            box_states[..., 0:3],
            box_rot,
            tar_pos,
            origins,
            goal_rotation=self.cfg["env"].get("goalRotation", "human_heading"),
        )
        parts = [nodes, entity_poses.reshape(B, -1)]
        suffix = self.relation_runtime.suffix(env_ids)
        parts.append(suffix)
        return torch.cat(parts, dim=-1)

