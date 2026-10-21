import json
from pathlib import Path
import numpy as np

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_angle_axis
import torch

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_cliff import HumanoidMACliff
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from env.tasks.multi_agent.cliff_geometry import stage_height


class ProbedCliff(HumanoidMACliff):
    def _create_envs(self, *args):
        self._probe_asset = self.gym.create_sphere(self.sim, .04, gymapi.AssetOptions())
        self._probe_handles = []
        super()._create_envs(*args)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        origin = self.gym.get_env_origin(env_ptr)
        width, length = self._stage_dimensions[env_id] / 2
        handles = []
        for i, x in enumerate((width - .2, width + .2, -width + .2, -width - .2)):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(origin.x + x, origin.y + length * .55, 4.)
            handles.append(self.gym.create_actor(env_ptr, self._probe_asset, pose,
                                                 'height_probe_' + str(i), env_id, 0))
        self._probe_handles.append(handles)


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(29)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['seed'] = 29
    task = ProbedCliff(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                       args.device, args.device_id, True)
    env = VecTaskPythonWrapperMA(task, args.rl_device)
    ids = torch.arange(task.num_envs, device=task.device)
    stage_ids = torch.tensor([task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                             for env_ptr, handle in zip(task.envs, task._stage_handles)], device=task.device)
    probe_ids = torch.tensor([[task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                              for handle in handles] for env_ptr, handles in zip(task.envs, task._probe_handles)], device=task.device)
    probe_initial = task._root_states[probe_ids].clone()
    report = {}
    try:
        obs = env.reset()
        assert obs.shape == (task.num_envs, 2655) and torch.isfinite(obs).all()
        assert set(task._reset_ref_slots) == {'loco'}
        body = task._kinematic_humanoid_rigid_body_states
        inside = (body[..., :2] - task._env_origins[:, None, None, :2]).abs() <= task._stage_half_size[:, None, None]
        assert inside.all()
        box_xy = task._box_states[..., :2] - task._env_origins[:, None, :2]
        radius = task._box_size[..., :2].norm(dim=-1) / 2
        gap = (box_xy.abs() - task._stage_half_size[:, None]).amax(-1) - radius
        assert (gap >= .79).all()
        report['observation_shape'] = list(obs.shape)
        report['minimum_box_clearance'] = gap.min().item()
        report['stage_sizes'] = (task._stage_half_size * 2).cpu().tolist()
        report['rsi'] = list(task._reset_ref_slots)
        poses = obs[:, 538:587].reshape(task.num_envs, 7, 7)
        torch.testing.assert_close(poses[:, :2, :3], body[:, :, 0, :3] - task._env_origins[:, None])
        logical_boxes = task._logical_box_values(task._box_states)
        torch.testing.assert_close(poses[:, 2:5, :3], logical_boxes[..., :3] - task._env_origins[:, None])
        report['logical_physical_object_pose_match'] = True
        q = body[:, :, 0, 3:7].cpu().numpy()
        yaw = np.arctan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                         1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))
        grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 32), np.linspace(-2, 2, 32), indexing='ij')
        local_root = (body[:, :, 0, :3] - task._env_origins[:, None]).cpu().numpy()
        x = local_root[..., 0, None] + np.cos(yaw)[..., None] * grid_x.ravel() - np.sin(yaw)[..., None] * grid_y.ravel()
        y = local_root[..., 1, None] + np.sin(yaw)[..., None] * grid_x.ravel() + np.cos(yaw)[..., None] * grid_y.ravel()
        half = task._stage_half_size.cpu().numpy()
        inside_sensor = (np.abs(x) <= half[:, 0, None, None]) & (np.abs(y) <= half[:, 1, None, None])
        expected_sensor = np.clip(inside_sensor * 1.5 - local_root[..., 2, None], -3, 3) / 3
        sensor_error = np.abs(obs[:, -2048:].reshape(-1, 2, 1024).cpu().numpy() - expected_sensor)
        report['reset_sensor_reference_max_error'] = float(sensor_error.max())
        assert sensor_error.max() < 1e-6
        report['origins'] = task._env_origins.cpu().tolist()
        report['stage_positions'] = task._root_states[stage_ids, :3].cpu().tolist()
        expected_stage = task._env_origins.clone(); expected_stage[:, 2] += .75
        torch.testing.assert_close(task._root_states[stage_ids, :3], expected_stage, atol=1e-5, rtol=0)
        report['human_positions_before'] = task._humanoid_root_states[..., :3].cpu().tolist()

        task._humanoid_root_states[..., 2] = task._initial_humanoid_root_states[..., 2] + task._cliff['height']
        task._humanoid_root_states[..., 3:7] = task._initial_humanoid_root_states[..., 3:7]
        task._humanoid_root_states[..., 7:13] = 0
        task._dof_pos[:] = task._initial_dof_pos
        task._dof_vel.zero_()
        task._reset_env_tensors(ids)
        actions = ((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1)
        foot_contact = torch.zeros(task.num_envs, 2, dtype=torch.bool, device=task.device)
        for _ in range(24):
            obs, _, _, _ = env.step(actions)
            foot_contact |= task._contact_forces[:, :, task._contact_body_ids].abs().sum((-1, -2)) > .1
        supported = task._rigid_body_pos[:, :, 0, 2] > task._cliff['height'] + .5
        report['foot_contact_agents'] = foot_contact.sum().item()
        report['supported_agents_after_24_steps'] = supported.sum().item()
        report['support_root_height'] = task._rigid_body_pos[:, :, 0, 2].cpu().tolist()
        report['human_positions_after'] = task._humanoid_root_states[..., :3].cpu().tolist()
        report['stage_positions_after'] = task._root_states[stage_ids, :3].cpu().tolist()
        report['support_foot_height'] = task._rigid_body_pos[:, :, task._contact_body_ids, 2].cpu().tolist()
        report['stage_shape_filters'] = [p.filter for p in task.gym.get_actor_rigid_shape_properties(task.envs[0], task._stage_handles[0])]
        report['humanoid_shape_filters'] = [p.filter for p in task.gym.get_actor_rigid_shape_properties(task.envs[0], task.humanoid_handles_all[0][0])]
        out = Path(args.output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'support_probe.json').write_text(json.dumps(report, indent=2))
        assert foot_contact.all() and supported.all()

        env.reset()
        task._humanoid_root_states[:, 0, :2] = task._env_origins[:, :2]
        task._humanoid_root_states[:, 0, 0] += task._stage_half_size[:, 0] + 5
        task._humanoid_root_states[:, 0, 2] = task._initial_humanoid_root_states[:, 0, 2] + .25
        task._humanoid_root_states[:, 0, 3:7] = task._initial_humanoid_root_states[:, 0, 3:7]
        task._humanoid_root_states[..., 7:13] = 0
        task._dof_pos[:] = task._initial_dof_pos
        task._dof_vel.zero_()
        task._reset_env_tensors(ids)
        task.progress_buf[:] = 10
        actions = ((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1)
        landed = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        for _ in range(24):
            _, _, _, info = env.step(actions)
            failed = info['terminate'].view(-1, 2)[:, 0].bool()
            feet = task._contact_forces[:, 0, task._contact_body_ids].abs().sum((-1, -2)) > .1
            upright = task._rigid_body_pos[:, 0, 0, 2] > .5
            landed |= failed & feet & upright
        report['upright_floor_landings_failed'] = int(landed.sum())
        assert landed.all()

        env.reset()
        # Rotate agent 0 sideways, first on the stage, then in other scenes on the floor.
        stage_cases = ids % 2 == 0
        task._humanoid_root_states[:, 0, :2] = task._env_origins[:, :2]
        task._humanoid_root_states[~stage_cases, 0, 0] += task._stage_half_size[~stage_cases, 0] + 2
        task._humanoid_root_states[:, 0, 2] = torch.where(stage_cases, 1.9, .4)
        axis = torch.zeros(task.num_envs, 3, device=task.device); axis[:, 1] = 1
        task._humanoid_root_states[:, 0, 3:7] = quat_from_angle_axis(
            torch.full((task.num_envs,), torch.pi / 2, device=task.device), axis)
        task._humanoid_root_states[..., 7:13] = 0
        task._dof_vel.zero_()
        task._reset_env_tensors(ids)
        actions = ((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1)
        observed = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        teammate_ok = False
        for _ in range(40):
            obs, _, done, info = env.step(actions)
            fallen = info['terminate'].view(-1, 2).bool()
            observed |= fallen[:, 0]
            teammate_ok |= bool((fallen[:, 0] & ~fallen[:, 1] & done.view(-1, 2)[:, 1].bool()).any())
        report['stage_falls_detected'] = int(observed[stage_cases].sum())
        report['ground_falls_detected'] = int(observed[~stage_cases].sum())
        report['normal_teammate_truncation'] = teammate_ok
        assert observed.all() and teammate_ok

        env.reset()
        # Moving outside the stage while still airborne must not terminate by itself.
        task._humanoid_root_states[:, 0, 0] = task._env_origins[:, 0] + task._stage_half_size[:, 0] + 2
        task._humanoid_root_states[:, 0, 2] = 5
        task._humanoid_root_states[:, 0, 7:13] = 0
        task._reset_env_tensors(ids)
        task.progress_buf[:] = 10
        for _ in range(3):
            _, _, _, info = env.step(actions)
            assert not info['terminate'].view(-1, 2)[:, 0].any()
        report['airborne_outside_is_not_terminal'] = True
        task._humanoid_root_states[..., 0] = task._env_origins[:, None, 0] + task._stage_half_size[:, None, 0] + 3
        task._humanoid_root_states[..., 1] = task._env_origins[:, None, 1] + torch.tensor([-1., 1.], device=task.device)
        task._humanoid_root_states[..., 2] = 1.
        task._humanoid_root_states[..., 7:13] = 0
        task._root_states[probe_ids] = probe_initial
        flat_ids = torch.cat((probe_ids.flatten().to(torch.int32), task._humanoid_actor_ids.flatten()))
        task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                                                     gymtorch.unwrap_tensor(flat_ids), len(flat_ids))
        for _ in range(90):
            env.step(actions)
        probes = task._root_states[probe_ids, :3]
        expected_height = stage_height(probes[..., :2], task._env_origins,
                                       task._stage_half_size, task._cliff['height']) + .04
        height_error = (probes[..., 2] - expected_height).abs()
        drift = (probes[..., :2] - probe_initial[..., :2]).norm(dim=-1)
        report['physical_height_probes'] = probes.shape[0] * probes.shape[1]
        report['maximum_probe_height_error_m'] = height_error.max().item()
        report['maximum_probe_xy_drift_m'] = drift.max().item()
        report['probe_positions'] = probes.cpu().tolist()
        report['probe_height_errors'] = height_error.cpu().tolist()
        (out / 'height_probe.json').write_text(json.dumps(report, indent=2))
        assert height_error.max() < .035 and drift.max() < .05
        torch.testing.assert_close(task._root_states[stage_ids, :3], expected_stage, atol=1e-5, rtol=0)
        report['finite_observations'] = bool(torch.isfinite(task.obs_buf).all())
        out = Path(args.output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'physics_smoke.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if task.viewer:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
