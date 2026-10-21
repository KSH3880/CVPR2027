import json
from pathlib import Path

import isaacgym
import numpy as np
import torch

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.edge_scenario_spec import agent_object_indices
from env.tasks.multi_agent.humanoid_ma_bridge import HumanoidMABridge, LIFT
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from env.tasks.multi_agent.cliff_geometry import terrain_fall
from env.tasks.multi_agent.vision_sensor import uniform_fan_grid
from env.tasks.multi_agent.bridge_geometry import bridge_trajectory


def stand(task, xy, heights):
    for agent, z in enumerate(heights):
        task._humanoid_root_states[:, agent, :2] = xy[:, agent]
        task._humanoid_root_states[:, agent, 2] = task._initial_humanoid_root_states[:, agent, 2] + z
        task._humanoid_root_states[:, agent, 3:7] = task._initial_humanoid_root_states[:, agent, 3:7]
    task._humanoid_root_states[..., 7:13] = 0
    task._dof_pos[:] = task._initial_dof_pos
    task._dof_vel.zero_()
    task._reset_env_tensors(torch.arange(task.num_envs, device=task.device))
    return ((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1)


def quat_matrix(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack((np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
                     np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
                     np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1)), -2)


def reference_sensor(task, body_pos, body_rot):
    # Independent NumPy recomputation of the policy sensor from simulator-frame states.
    c = task._cliff
    height, clip = c['height'], c['sensor_clip']
    pos, rot = body_pos.cpu().numpy(), body_rot.cpu().numpy()
    head = task._sensor_head_id
    eye = pos[..., head, :] + np.einsum('eaij,j->eai', quat_matrix(rot[..., head, :]), np.array(c['sensor_head_offset']))
    q = rot[..., 0, :]
    yaw = np.arctan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]), 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))
    grid = uniform_fan_grid(c['sensor_resolution'], c['sensor_range'], c['sensor_fov'], 'cpu')[0].numpy()
    cos, sin = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    x = eye[..., 0, None] + cos * grid[:, 0] - sin * grid[:, 1]
    y = eye[..., 1, None] + sin * grid[:, 0] + cos * grid[:, 1]
    origin = task._env_origins.cpu().numpy()
    n = len(origin)
    identity = np.tile([0., 0., 0., 1.], (n, 1))
    centers = [origin + [0, 0, height / 2], task._far_stage_states[:, :3].cpu().numpy(),
               task._bridge_states[:, :3].cpu().numpy()]
    quats = [identity, task._far_stage_states[:, 3:7].cpu().numpy(), task._bridge_states[:, 3:7].cpu().numpy()]
    halves = [np.concatenate((task._stage_half_size.cpu().numpy(), np.full((n, 1), height / 2)), -1),
              np.concatenate((task._far_stage_half_size.cpu().numpy(), np.full((n, 1), height / 2)), -1),
              task._bridge_half_size.cpu().numpy()]
    active = task._bridge_active.cpu().numpy()
    for k in range(task.num_objects):
        centers.append(task._box_states[:, k, :3].cpu().numpy())
        quats.append(task._box_states[:, k, 3:7].cpu().numpy())
        halves.append(task._box_size[:, k].cpu().numpy() / 2)
    surface = np.broadcast_to(origin[:, 2, None, None], x.shape).copy()
    boundary = np.full(x.shape, np.inf)
    for k, (center, quat, half) in enumerate(zip(centers, quats, halves)):
        if k in (1, 2) and not active.all():
            raise ValueError('Reference sensor audit expects bridge episodes only')
        matrix = quat_matrix(quat)[:, None, None]
        dx, dy = x - center[:, 0, None, None], y - center[:, 1, None, None]
        local_x = matrix[..., 0, 0] * dx + matrix[..., 1, 0] * dy
        local_y = matrix[..., 0, 1] * dx + matrix[..., 1, 1] * dy
        inside = (np.abs(local_x) <= half[:, 0, None, None]) & (np.abs(local_y) <= half[:, 1, None, None])
        outside = np.maximum(np.abs(local_x) - half[:, 0, None, None], np.abs(local_y) - half[:, 1, None, None])
        boundary = np.minimum(boundary, np.abs(outside))
        top = center[:, 2] + (np.abs(quat_matrix(quat)[:, 2]) * half).sum(-1)
        surface = np.where(inside, np.maximum(surface, top[:, None, None]), surface)
    return np.clip((surface - pos[..., 0, 2, None]) / clip, -1, 1), boundary


def compare_sensor(observed, reference):
    # float32 world coordinates can flip inside/outside only for samples on a box edge.
    expected, boundary = reference
    error = np.abs(observed - expected)
    mismatch = error > 1e-5
    assert not (mismatch & (boundary > 1e-3)).any()
    return {'max_error_off_edges': float(error[boundary > 1e-3].max()),
            'edge_ambiguous_mismatches': int(mismatch.sum()), 'samples': int(error.size)}


def close(actual, expected):
    # World coordinates of large env grids carry float32 rounding of about 1e-5 m.
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def audit_coordinates(task, obs):
    # Bridge episodes right after reset; kinematic states are what the reset observation used.
    b, height, n = task._bridge, task._cliff['height'], task.num_envs
    origin = task._env_origins
    half_a, half_b = task._stage_half_size, task._far_stage_half_size
    gap, bridge_half = task._bridge_gap, task._bridge_half_size
    report = {}
    far = task._far_stage_states[:, :3] - origin
    close(task._far_stage_states[:, 3:7], torch.tensor([[0., 0., 0., 1.]], device=task.device).expand(n, -1))
    close(far[:, 0] - half_b[:, 0], half_a[:, 0] + gap)
    close(far[:, 2], torch.full_like(far[:, 2], height / 2 + LIFT))
    goal = task._bridge_goal - origin
    close(goal[:, 0], half_a[:, 0] + gap / 2)
    close(goal[:, 2], bridge_half[:, 2] + LIFT)
    assert (goal[:, 0] - bridge_half[:, 0] >= half_a[:, 0] + b['clearance'] - 1e-5).all()
    assert (goal[:, 0] + bridge_half[:, 0] <= half_a[:, 0] + gap - b['clearance'] + 1e-5).all()
    overlap_low = torch.maximum(-half_a[:, 1], far[:, 1] - half_b[:, 1])
    overlap_high = torch.minimum(half_a[:, 1], far[:, 1] + half_b[:, 1])
    assert ((goal[:, 1] - bridge_half[:, 1] >= overlap_low - 1e-5) & (goal[:, 1] + bridge_half[:, 1] <= overlap_high + 1e-5)).all()
    start = task._bridge_start - origin
    close(start[:, 0], goal[:, 0])
    initial, velocity = bridge_trajectory(task._bridge_start, task._bridge_goal, -task._bridge_wait, task._bridge_duration)
    close(task._bridge_states[:, :3], initial)
    close(task._bridge_states[:, 7:10], velocity)
    report['bridge_initial_phase'] = ((initial - task._bridge_start).norm(dim=-1)
        / (task._bridge_goal - task._bridge_start).norm(dim=-1).clamp_min(1e-6)).cpu().tolist()

    body = task._kinematic_humanoid_rigid_body_states
    root = body[..., 0, :3] - origin[:, None]
    assert (root[..., :2].abs() <= half_a[:, None] - task._cliff['spawn_margin'] + 1e-5).all()
    report['root_height_above_stage'] = [(root[..., 2] - height).min().item(), (root[..., 2] - height).max().item()]
    assert ((root[..., 2] - height > .5) & (root[..., 2] - height < 1.2)).all()
    report['lowest_body_above_stage_m'] = (body[..., 2] - origin[:, None, None, 2] - height).min().item()
    assert report['lowest_body_above_stage_m'] > -.05

    boxes = task._box_states[..., :3] - origin[:, None]
    radius = task._box_size[..., :2].norm(dim=-1) / 2
    relative = boxes[..., :2] - far[:, None, :2]
    assert ((relative.abs() + radius[..., None]) <= half_b[:, None] + 1e-5).all()
    bottom = boxes[..., 2] - task._box_size[..., 2] / 2 - (height + LIFT)
    report['box_bottom_above_far_top_m'] = [bottom.min().item(), bottom.max().item()]
    assert ((bottom >= 0) & (bottom <= .02)).all()
    band = relative[..., 0] + half_b[:, None, 0] - radius
    assert ((band >= b['object_band'][0] - 1e-5) & (band <= b['object_band'][1] + 1e-5)).all()
    expected_goal = torch.stack((far[:, 0] - half_b[:, 0] + sum(b['object_band']) / 2, far[:, 1],
                                 torch.full_like(far[:, 1], height + LIFT)), -1)
    close(task._tar_pos - origin[:, None], expected_goal[:, None].expand(-1, 2, -1))

    poses = obs[:, 538:587].reshape(n, 7, 7)
    close(poses[:, :2, :3], root)
    close(poses[:, 2:5, :3], task._logical_box_values(task._box_states)[..., :3] - origin[:, None])
    close(poses[:, 5:7, :3], task._tar_pos - origin[:, None])

    ids = torch.arange(n, device=task.device)
    slot_env, slot_agent = task._expand_slots(ids)
    index = agent_object_indices(task.relation_runtime.graph, slot_env, slot_agent).view(n, 2)
    own = task._logical_box_values(task._box_states).gather(1, index[..., None].expand(-1, -1, 13))[..., :2]
    direction = torch.atan2(own[..., 1] - body[..., 0, 1], own[..., 0] - body[..., 0, 0])
    q = body[..., 0, 3:7]
    yaw = torch.atan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]), 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))
    error = torch.remainder(yaw - direction + torch.pi, 2 * torch.pi) - torch.pi
    report['max_heading_error_rad'] = error.abs().max().item()
    assert error.abs().max() <= task._cliff['heading_jitter'] + 1e-3

    probes = torch.stack((torch.zeros_like(goal[:, :2]), far[:, :2],
                          task._bridge_states[:, :2] - origin[:, :2],
                          torch.stack((goal[:, 0], goal[:, 1] + bridge_half[:, 1] + .3), -1)), 1)
    measured = task._terrain_height(probes[:, None] + origin[:, None, None, :2])[:, 0] - origin[:, None, 2]
    bridge_top = task._bridge_states[:, 2] - origin[:, 2] + bridge_half[:, 2]
    beside = ((probes[:, 3] - (task._bridge_states[:, :2] - origin[:, :2])).abs() <= bridge_half[:, :2]).all(-1)
    expected = torch.stack((torch.full_like(gap, height), torch.full_like(gap, height + LIFT), bridge_top,
                            torch.where(beside, bridge_top, torch.zeros_like(gap))), -1)
    close(measured, expected)
    report['gap_probe_on_moving_bridge'] = int(beside.sum())
    arrival = task._bridge_wait + task._bridge_duration
    present = (task._bridge_start == task._bridge_goal).all(-1)
    low, high = b['arrival_time_range']
    assert ((arrival[~present] >= low - 1e-4) & (arrival[~present] <= high + 1e-4)).all()
    report['arrival_time_s'] = arrival[~present].cpu().tolist()

    sensor = obs[:, 607:].reshape(n, 2, task._sensor_points).cpu().numpy()
    report['reset_sensor'] = compare_sensor(sensor, reference_sensor(task, body[..., :3], body[..., 3:7]))
    return report


def fallen(task):
    ground = task._terrain_height(task._rigid_body_pos[..., :2])
    return terrain_fall(task._contact_forces, task._rigid_body_pos, task._contact_body_ids, ground,
                        task._termination_heights, task.progress_buf, task._env_origins[:, 2])


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(29)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['seed'] = 29
    cfg['env']['enableEarlyTermination'] = False
    cfg['env']['episodeLength'] = 100000
    task = HumanoidMABridge(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                            args.device, args.device_id, True)
    env = VecTaskPythonWrapperMA(task, args.rl_device)
    b, height = task._bridge, task._cliff['height']
    report = {}
    try:
        obs = env.reset()
        assert obs.shape == (task.num_envs, 607 + 2 * task._sensor_points) and torch.isfinite(obs).all()
        report['observation_shape'] = list(obs.shape)
        report['bridge_env_fraction_first_reset'] = task._bridge_active.float().mean().item()
        parked = ~task._bridge_active
        torch.testing.assert_close(task._tar_pos[parked], task._env_origins[parked, None].expand(-1, 2, -1))
        parked_before = torch.stack((task._far_stage_states, task._bridge_states), 1)[parked, :, :3].clone()
        actions = torch.zeros(task.num_envs * 2, task._dof_pos.shape[-1], device=task.device)
        for _ in range(30):
            env.step(actions)
        parked_after = torch.stack((task._far_stage_states, task._bridge_states), 1)[parked, :, :3]
        report['parked_envs'] = int(parked.sum())
        report['parked_max_drift_m'] = (parked_after - parked_before).norm(dim=-1).max().item() if parked.any() else 0.
        assert report['parked_max_drift_m'] < 1e-3

        task._bridge_probability = 1.
        obs = env.reset()
        assert task._bridge_active.all()
        report['coordinate_audit'] = audit_coordinates(task, obs)
        spawn = task._kinematic_humanoid_rigid_body_states[..., 0, :3].clone()
        env.step(((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1))
        simulated = task._rigid_body_pos[..., 0, :]
        report['spawn_first_step_max_xy_shift_m'] = (simulated[..., :2] - spawn[..., :2]).norm(dim=-1).max().item()
        report['spawn_first_step_min_root_above_stage_m'] = (simulated[..., 2] - task._env_origins[:, None, 2] - height).min().item()
        assert report['spawn_first_step_max_xy_shift_m'] < .1 and report['spawn_first_step_min_root_above_stage_m'] > .5
        report['present_at_reset'] = int((task._bridge_start == task._bridge_goal).all(-1).sum())
        task._bridge_wait.zero_()
        task.progress_buf.zero_()
        task._bridge_states[:, :3] = task._bridge_start
        task._bridge_states[:, 7:13] = 0
        far_start = task._far_stage_states[:, :3].clone()
        xy = task._env_origins[:, None, :2].repeat(1, 2, 1)
        xy[..., 0] -= task._stage_half_size[:, None, 0] - 1
        xy[..., 1] += torch.tensor([-1., 1.], device=task.device)
        actions = stand(task, xy, (height, height))
        steps = int((task._bridge_duration.max() / task.dt).ceil()) + 30
        lateral = torch.zeros(task.num_envs, device=task.device)
        vertical = torch.zeros(task.num_envs, device=task.device)
        tilt = torch.zeros(task.num_envs, device=task.device)
        for _ in range(steps):
            env.step(actions)
            state = task._bridge_states
            lateral = torch.maximum(lateral, (state[:, 0] - task._bridge_goal[:, 0]).abs())
            vertical = torch.maximum(vertical, (state[:, 2] - task._bridge_goal[:, 2]).abs())
            tilt = torch.maximum(tilt, state[:, 3:6].norm(dim=-1))
        final = (task._bridge_states[:, :3] - task._bridge_goal).norm(dim=-1)
        far_drift = (task._far_stage_states[:, :3] - far_start).norm(dim=-1)
        report.update(steps=steps, clearance_m=b['clearance'],
                      max_lateral_error_m=lateral.max().item(), max_vertical_error_m=vertical.max().item(),
                      max_quaternion_xyz=tilt.max().item(), final_position_error_m=final.max().item(),
                      far_stage_max_drift_m=far_drift.max().item(),
                      bridge_travel_m=(task._bridge_goal - task._bridge_start).norm(dim=-1).cpu().tolist())
        assert lateral.max() < b['clearance'] and final.max() < .02 and far_drift.max() < 1e-3
        boxes = task._box_states[..., :3] - task._far_stage_states[:, None, :3]
        on_far = (boxes[..., :2].abs() <= task._far_stage_half_size[:, None]).all(-1) & (boxes[..., 2] > 0)
        report['task_boxes_on_far_stage'] = int(on_far.sum())
        assert on_far.all()

        data = task._fan_sensor_data(live=True)
        state = task._bridge_states
        local = data['xy'] - state[:, None, None, :2]
        on_bridge = (local.abs() < task._bridge_half_size[:, None, None, :2] - .02).all(-1)
        top = (state[:, 2] + task._bridge_half_size[:, 2])[:, None, None].expand_as(on_bridge)
        report['sensor_samples_on_bridge'] = int(on_bridge.sum())
        report['sensor_bridge_height_max_error_m'] = (data['surface'][on_bridge] - top[on_bridge]).abs().max().item()
        assert on_bridge.any() and report['sensor_bridge_height_max_error_m'] < 1e-4
        report['live_sensor'] = compare_sensor(task.obs_buf[:, 607:].reshape(task.num_envs, 2, -1).cpu().numpy(),
                                               reference_sensor(task, task._rigid_body_pos, task._rigid_body_rot))

        # Agent 0 stands on the arrived bridge, agent 1 on the gap floor beside it.
        task._bridge_start[:] = task._bridge_goal
        xy = task._bridge_goal[:, None, :2].repeat(1, 2, 1)
        xy[:, 1, 1] += task._bridge_half_size[:, 1] + .6
        loaded_start = task._bridge_states[:, :3].clone()
        actions = stand(task, xy, (height - b['top_drop'] + .002, 0.))
        task.progress_buf[:] = 10
        floor_fall = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        for _ in range(24):
            env.step(actions)
            floor_fall |= fallen(task)[:, 1]
        on_bridge_fall = fallen(task)[:, 0]
        supported = task._rigid_body_pos[:, 0, 0, 2] > task._env_origins[:, 2] + height + .5
        load_drift = (task._bridge_states[:, :3] - loaded_start).norm(dim=-1)
        report.update(standing_on_bridge_supported=int(supported.sum()),
                      standing_on_bridge_fall_flags=int(on_bridge_fall.sum()),
                      bridge_drift_under_load_m=load_drift.max().item(),
                      gap_floor_contact_failures=int(floor_fall.sum()))
        assert supported.all() and not on_bridge_fall.any() and floor_fall.all() and load_drift.max() < .01
        report['finite_observations'] = bool(torch.isfinite(task.obs_buf).all())
        out = Path(args.output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'physics_smoke.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'bridge_travel_m'}, indent=2))
    finally:
        if task.viewer:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
