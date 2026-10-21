import json
import copy
from pathlib import Path

import isaacgym
from isaacgym import gymapi
import numpy as np
import torch

from env.tasks.multi_agent.humanoid_ma_room import HumanoidMARoom
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from env.tasks.multi_agent.vision_sensor_debug import draw_fan_sensor
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.edge_scenario_spec import agent_goal_indices, classify_templates


def capture(task, output, name, camera, target, sensor=False):
    task.gym.viewer_camera_look_at(task.viewer, task.envs[0], gymapi.Vec3(*camera), gymapi.Vec3(*target))
    task.gym.clear_lines(task.viewer)
    if task._supports_holding_at:
        task._draw_room_goals()
    if sensor:
        data = task._fan_sensor_data()
        draw_fan_sensor(task.gym, task.viewer, task.envs[0],
            {key: value[0, :1] for key, value in data.items()}, task._env_origins[0],
            task._cliff['sensor_resolution'], uniform_rings=True)
    for _ in range(3):
        task.gym.step_graphics(task.sim)
        task.gym.draw_viewer(task.viewer, task.sim, True)
    task.gym.write_viewer_image_to_file(task.viewer, str(output / (name + '.png')))


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(29)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['seed'] = 29
    cfg['env']['numEnvs'] = 1
    cfg['env']['enableDebugVis'] = False
    task = HumanoidMARoom(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                          args.device, args.device_id, False)
    env = VecTaskPythonWrapperMA(task, args.rl_device)
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    report = {'room_inner_size_m': (task._stage_half_size[0] * 2).cpu().tolist(),
              'wall_height_above_floor_m': task._room['wall_height'],
              'wall_thickness_m': task._room['wall_thickness'], 'captures': {}}
    partition = task._supports_holding_at
    if partition:
        report['partition_size_m'] = (task._wall_half_size[0, 4] * 2).cpu().tolist()
        report['passage_width_m'] = task._passage_width
    probabilities = copy.deepcopy(task._relation_graph_spec['template_probabilities'])
    cases = ('mixed_layout', 'both_holding_at', 'partition_sensor') if partition else ('layout_1', 'layout_2', 'wall_sensor')
    try:
        for name in cases:
            if partition:
                task._relation_graph_spec['template_probabilities'] = (probabilities if name == 'mixed_layout' else
                    {key: float(key == 'HOLDING_AT') for key in probabilities})
            env.reset()
            if name == 'both_holding_at':
                for _ in range(32):
                    separation = (task._tar_pos[0, :, None, :2] - task._humanoid_root_states[0, None, :, :2]).norm(dim=-1)
                    if separation.min() > 1.:
                        break
                    env.reset()
            if name in ('wall_sensor', 'partition_sensor'):
                candidates = torch.zeros(9, 2, device=task.device)
                candidates[:, 0] = -1. if partition else task._stage_half_size[0, 0] - .8
                extent = task._wall_half_size[0, 4, 1].item() - .5 if partition else 1.5
                candidates[:, 1] = torch.linspace(-extent, extent, 9, device=task.device)
                occupied = torch.cat((task._box_states[0, :, :2], task._humanoid_root_states[0, 1:, :2]))
                clearance = (candidates[:, None] - occupied[None]).norm(dim=-1).min(-1)[0]
                xy = candidates[clearance.argmax()]
                task._humanoid_root_states[0, 0] = task._initial_humanoid_root_states[0, 0]
                task._humanoid_root_states[0, 0, :3] = torch.tensor(
                    [xy[0], xy[1], task._char_h + task._cliff['height']], device=task.device)
                task._humanoid_root_states[0, 0, 7:13] = 0
                task._dof_pos[0, 0] = task._initial_dof_pos[0, 0]
                task._dof_vel[0, 0] = 0
                task._reset_env_tensors(torch.tensor([0], device=task.device))
            actions = ((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1)
            env.step(actions)
            data = task._fan_sensor_data()
            sensor = data['observation']
            torch.testing.assert_close(task.obs_buf[:, 607:], sensor.flatten(1))
            roots = task._humanoid_root_states[0, :, :3] - task._env_origins[0]
            boxes = task._box_states[0, :, :3] - task._env_origins[0]
            assert (roots[:, :2].abs() < task._stage_half_size[0]).all()
            assert (boxes[:, :2].abs() < task._stage_half_size[0]).all()
            report['captures'][name] = {'agents_xyz': roots.cpu().tolist(), 'objects_xyz': boxes.cpu().tolist(),
                'object_sizes': task._box_size[0].cpu().tolist(),
                'wall_samples': (sensor > .4).sum(-1)[0].cpu().tolist(),
                'observation_shape': list(task.obs_buf.shape)}
            if partition:
                agents = torch.arange(2, device=task.device)
                envs = torch.zeros_like(agents)
                kinds = classify_templates(task.relation_runtime.graph, envs, agents, True)
                goal_ids = agent_goal_indices(task.relation_runtime.graph, envs, agents)
                active = kinds == 3
                for slot in range(2):
                    visible = ((goal_ids == slot) & active).any()
                    if visible:
                        torch.testing.assert_close(task._marker_pos[0, slot], task._tar_pos[0, slot])
                    else:
                        assert task._marker_pos[0, slot, 2] == 20.
                report['captures'][name].update(templates=kinds.cpu().tolist(),
                    goals_xyz=(task._tar_pos[0] - task._env_origins[0]).cpu().tolist(),
                    goal_slots=goal_ids.cpu().tolist(), logical_box_order=task._logical_box_order[0].cpu().tolist(),
                    graph={key: getattr(task.relation_runtime.graph, key)[0].cpu().tolist()
                           for key in ('edge_src', 'edge_dst', 'edge_relation', 'edge_owner', 'edge_valid')})
            np.savez_compressed(output / (name + '.npz'),
                **{key: value.cpu().numpy() for key, value in data.items()})
            capture(task, output, name + '_top', (0., -.05, 12.5), (0., 0., 0.))
            capture(task, output, name + '_oblique', (6., -8., 11.), (0., 0., .5))
            if name in ('wall_sensor', 'partition_sensor'):
                capture(task, output, name + '_observed', (6., -8., 11.), (0., 0., .5), sensor=True)
        (output / 'captures.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
