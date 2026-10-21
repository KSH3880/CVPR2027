import json
from pathlib import Path

import isaacgym
from isaacgym import gymapi
import numpy as np
import torch

from env.tasks.multi_agent.humanoid_ma_cliff import HumanoidMACliff
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from env.tasks.multi_agent.vision_sensor_debug import draw_fan_sensor
from utils.config import get_args, load_cfg, parse_sim_params, set_seed


class CaptureCliff(HumanoidMACliff):
    def _load_box_asset(self):
        sizes = [[1., 2., .8], [1., 6., 5.], [.7, 1.2, .6]]
        self._box_size = torch.tensor([sizes], device=self.device)
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)
        self._box_density = torch.full((3,), 100., device=self.device)
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.density = 1e8
        self._box_assets = [self.gym.create_box(self.sim, *size, options) for size in sizes]


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(29)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['seed'] = 29
    cfg['env']['numEnvs'] = 1
    cfg['env']['cliff']['size_range'] = [8., 8.]
    cfg['env']['enableDebugVis'] = True
    task = CaptureCliff(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                        args.device, args.device_id, args.headless)
    env = VecTaskPythonWrapperMA(task, args.rl_device)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    ids = torch.tensor([0], device=task.device)
    try:
        env.reset()
        for name in ('cliff', 'low_box', 'tall_wall', 'carried_height_box', 'heading_90'):
            task._humanoid_root_states[:] = task._initial_humanoid_root_states
            task._humanoid_root_states[..., :2] = torch.tensor([[[0., 0.], [-3., -2.]]], device=task.device)
            task._humanoid_root_states[..., 2] += 1.5
            task._humanoid_root_states[..., 7:13] = 0
            task._dof_pos[:] = task._initial_dof_pos
            task._dof_vel.zero_()
            task._box_states.zero_()
            task._box_states[..., 6] = 1
            task._box_states[..., 0] = -200
            task._box_states[..., 1] = torch.tensor([0., 10., 20.], device=task.device)
            task._box_states[..., 2] = task._box_size[..., 2] / 2
            if name in ('cliff', 'heading_90'):
                axis = 1 if name == 'heading_90' else 0
                task._humanoid_root_states[0, 0, axis] = 2.
            if name == 'heading_90':
                task._humanoid_root_states[0, 0, 3:7] = torch.tensor(
                    [0., 0., 2**-.5, 2**-.5], device=task.device)
            if name == 'low_box':
                task._box_states[0, 0, :3] = torch.tensor([2., 0., 1.9], device=task.device)
                task._box_states[0, 0, 5:7] = torch.tensor([.258819, .965926], device=task.device)
            if name == 'tall_wall':
                task._box_states[0, 1, :3] = torch.tensor([3., 0., 4.], device=task.device)
            if name == 'carried_height_box':
                task._box_states[0, 2, :3] = torch.tensor([.6, 0., 2.6], device=task.device)
            task._reset_env_tensors(ids)
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim, True)
            task._refresh_sim_tensors()
            task._compute_observations()
            data = task._fan_sensor_data()
            sensor = data['observation']
            assert sensor.shape == (1, 2, task._sensor_points) and torch.isfinite(sensor).all()
            torch.testing.assert_close(task.obs_buf[:, -2 * task._sensor_points:], sensor.flatten(1))
            if task.viewer is not None:
                for key in ('xy', 'filled', 'status', 'eye'):
                    torch.testing.assert_close(task._fan_view_data[key], data[key][0])
                calculate = task._fan_sensor_data
                def unexpected_recompute(*args, **kwargs):
                    raise RuntimeError('Viewer recomputed the sensor')
                task._fan_sensor_data = unexpected_recompute
                task._draw_fan_sensor()
                task._fan_sensor_data = calculate
            status = data['status'][0, 0]
            counts = {label: int((status == value).sum()) for value, label in
                      ((1, 'visible'), (2, 'inferred'), (0, 'unknown'))}
            report[name] = {**counts, 'head': data['eye'][0, 0].cpu().tolist(),
                            'observation_shape': list(task.obs_buf.shape)}
            np.savez_compressed(out / (name + '.npz'),
                **{key: value.cpu().numpy() for key, value in data.items()},
                box_states=task._box_states.cpu().numpy(), box_sizes=task._box_size.cpu().numpy())
            if task.viewer is not None:
                for view, position, target in (
                    ('wide', (10., -24., 30.), (8., 0., .5)),
                    ('detail', (6., -9., 7.), (2., 0., 1.5))):
                    if name == 'heading_90':
                        position = (-position[1], position[0], position[2])
                        target = (-target[1], target[0], target[2])
                    task.gym.viewer_camera_look_at(task.viewer, task.envs[0],
                                                  gymapi.Vec3(*position), gymapi.Vec3(*target))
                    task.gym.clear_lines(task.viewer)
                    draw_fan_sensor(task.gym, task.viewer, task.envs[0],
                        {key: value[0, :1] for key, value in data.items()}, task._env_origins[0], task._cliff['sensor_resolution'], uniform_rings=True)
                    for _ in range(3):
                        task.gym.step_graphics(task.sim)
                        task.gym.draw_viewer(task.viewer, task.sim, True)
                    task.gym.write_viewer_image_to_file(task.viewer, str(out / (name + '_' + view + '.png')))
        assert report['cliff']['inferred'] > 0 and report['cliff']['unknown'] == 0
        assert report['tall_wall']['unknown'] > 0
        assert report['low_box']['visible'] > 0
        (out / 'scenarios.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
