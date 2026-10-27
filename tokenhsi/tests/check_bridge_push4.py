import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaacgym import gymapi, gymtorch
import torch
from torch import nn

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from learning.multi_agent.bridge_network import BridgeBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def insertion_check(task, env):
    task.reset_buf[:] = 1
    env.reset()
    ids = torch.arange(task.num_envs, device=task.device)
    task._humanoid_root_states[..., :2] += 100.
    task._reset_env_tensors(ids)
    force = torch.zeros_like(task._bridge_forces)
    initial = (task._bridge_states[:, :2]-task._bridge_goal[:, :2]).norm(dim=-1)
    best = initial.clone()
    for _ in range(600):
        delta = task._bridge_goal[:, 1]-task._bridge_states[:, 1]
        vy = task._bridge_states[:, 8]
        target_v = delta.sign()
        target_v = torch.where(delta.abs() < .08, 0., target_v)
        force[task._bridge_body_ids, 1] = (500*(target_v-vy)).clamp(-400, 400)
        task.gym.apply_rigid_body_force_tensors(task.sim, gymtorch.unwrap_tensor(force), None, gymapi.ENV_SPACE)
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()
        distance = (task._bridge_states[:, :2]-task._bridge_goal[:, :2]).norm(dim=-1)
        best = torch.minimum(best, distance)
    return dict(envs=task.num_envs, steps=600, external_force_only=True,
        gap_minus_cube_width_m=(task._bridge_gap-task._bridge_size[:, 0]).tolist(),
        initial_distance_m=initial.tolist(), minimum_distance_m=best.tolist(),
        reached_20cm_fraction=float((best < .2).float().mean()))


def capture(task, output, count):
    from PIL import Image, ImageDraw
    images, metadata = [], []
    for number in range(min(count, task.num_envs)):
        index = number
        task._reset_envs(torch.tensor([index], device=task.device))
        for human in range(task.num_agents):
            physical = int(task._agent_box_assignment[index, human])
            color = task._agent_color(human)
            if physical == task._bridge_slot:
                color = gymapi.Vec3(.62, .45, .25)
            task.gym.set_rigid_body_color(task.envs[index],
                task._box_handles[index*task.num_physical_objects+physical], 0, gymapi.MESH_VISUAL, color)
        props = gymapi.CameraProperties()
        props.width, props.height, props.horizontal_fov = 1440, 1080, 65.
        camera = task.gym.create_camera_sensor(task.envs[index], props)
        origin = task._env_origins[index]
        near = task._near_stage_states[index, :3]-origin
        far = task._far_stage_states[index, :3]-origin
        points = torch.cat((task._humanoid_root_states[index, :, :3],
                            task._box_states[index, task._logical_box_order[index, :4], :3]), 0)-origin
        low, high = points.amin(0), points.amax(0)
        low[:2] = torch.minimum(low[:2], torch.minimum(near[:2]-task._stage_half_size[index],
                                                      far[:2]-task._far_stage_half_size[index]))
        high[:2] = torch.maximum(high[:2], torch.maximum(near[:2]+task._stage_half_size[index],
                                                        far[:2]+task._far_stage_half_size[index]))
        center = (low+high)/2
        span = max(float((high-low)[:2].max()), 10.)
        # Render each environment at the origin, inside the finite ground visual mesh.
        actors = task.get_num_actors_per_env()
        roots = task._root_states.view(task.num_envs, actors, 13)
        original = roots[index, :, :3].clone()
        roots[index, :, :3] -= origin
        ids = torch.arange(index*actors, (index+1)*actors, device=task.device, dtype=torch.int32)
        task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                                                     gymtorch.unwrap_tensor(ids), len(ids))
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()
        target = gymapi.Vec3(float(center[0]), float(center[1]), 0.)
        position = gymapi.Vec3(float(center[0])-span*.15, float(center[1])-span*.25, span*1.1)
        task.gym.set_camera_location(camera, task.envs[index], position, target)
        task.gym.step_graphics(task.sim)
        task.gym.render_all_camera_sensors(task.sim)
        data = task.gym.get_camera_image(task.sim, task.envs[index], camera, gymapi.IMAGE_COLOR)
        image = Image.fromarray(data.reshape(props.height, props.width, 4)[..., :3])
        roots[index, :, :3] = original
        task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                                                     gymtorch.unwrap_tensor(ids), len(ids))
        filename = 'scene_{:02d}.png'.format(number+1)
        image.save(output/filename)
        images.append(image)
        metadata.append(dict(image=filename, env=index, roles=task._roles[index].tolist(),
            tasks=task._tasks4[index].tolist(), surfaces=task._task_surface[index].tolist(),
            start_surfaces=task._start_surface[index].tolist(),
            stage_a_size=task._stage_dimensions[index].tolist(), stage_b_size=task._far_stage_dimensions[index].tolist(),
            bridge_size=task._bridge_size[index].tolist(),
            stage_a_position=(task._near_stage_states[index, :3]-origin).tolist(),
            stage_b_position=(task._far_stage_states[index, :3]-origin).tolist(),
            human_positions=(task._humanoid_root_states[index, :, :3]-origin).tolist(),
            box_positions=(task._box_states[index, task._agent_box_assignment[index], :3]-origin).tolist(),
            bridge_goal=(task._bridge_goal[index]-origin).tolist()))
        task.gym.destroy_camera_sensor(task.sim, task.envs[index], camera)
    sheet = Image.new('RGB', (960, 5*388), '#20242c')
    draw = ImageDraw.Draw(sheet)
    for index, im in enumerate(images):
        x, y = (index % 2)*480, (index//2)*388
        im.thumbnail((480, 360))
        sheet.paste(im, (x, y+24))
        draw.text((x+8, y+6), 'Scene {:02d} | roles={} tasks={}'.format(index+1,
            metadata[index]['roles'], metadata[index]['tasks']), fill='white')
    sheet.save(output/'overview.jpg', quality=92)
    (output/'scenes.json').write_text(json.dumps(metadata, indent=2)+'\n')
    cards = ''.join('<figure><a href="{0}"><img src="{0}"></a><figcaption>Scene {1:02d}</figcaption></figure>'.format(
        row['image'], index+1) for index, row in enumerate(metadata))
    (output/'index.html').write_text('<meta charset="utf-8"><title>Bridge PUSH4 scenes</title>'
        '<style>body{background:#20242c;color:white;font:18px sans-serif}main{display:grid;grid-template-columns:1fr 1fr}'
        'img{width:100%}figure{margin:12px}</style><h1>Bridge PUSH4: 10 initial scenes</h1>'
        '<p>Tasks: 1 SIT, 2 CLIMB, 3 CARRY, 4 PUSH. Roles: bridge pusher, crosser, independent, independent.</p>'
        '<main>'+cards+'</main>')


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test, args.eval = True, True
    args.task_graph = 'random_scenario'
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    cfg['env']['video']['enable'] = os.environ.get('CAPTURE_SCENES') == '1'
    set_seed(int(os.environ.get('SEED', '42')))
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    builder = BridgeBuilder()
    builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents,
        num_objects=task.num_objects, observation_mode='clean_scene',
        scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=7,
        scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device).to(task.device)
    model = nn.Module()
    model.a2c_network = net
    checkpoint = torch.load(cfg['experiment']['transfer']['checkpoint'], map_location=task.device, weights_only=False)
    report = transfer_stage1(model, checkpoint)
    rms = SceneRunningMeanStd([223, 30, 1], [task.num_agents, task.num_objects, task.num_agents],
        normalized_sizes=[223, 30, 0], kinematic_size=7,
        extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    if args.checkpoint != 'Base':
        weights = torch.load(args.checkpoint, map_location=task.device, weights_only=False)
        from utils.stage2_spec import check_checkpoint
        check_checkpoint(weights, cfg['env'])
        model.load_state_dict(weights['model'], strict=True)
        rms.load_state_dict(weights['running_mean_std'], strict=True)
    rms.eval()
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        obs = env.reset()
        torch.testing.assert_close(task._bridge_gap, task._bridge_size[:, 0]+.1)
        roots = task._kinematic_humanoid_rigid_body_states[..., 0, :]
        terrain = task._terrain_height(roots[..., :2])-task._env_origins[:, None, 2]
        expected = (task._start_surface > 0)*task._stage_height[:, None]
        torch.testing.assert_close(terrain, expected, atol=1e-3, rtol=0)
        from env.tasks.multi_agent.push_reward import face_xy_distances
        ids, human = task._pushing.nonzero(as_tuple=True)
        payload = task._push_payload[ids, human]
        hands = task._kinematic_humanoid_rigid_body_states[ids, human][:, task._key_body_ids[:2], :3]
        distance = face_xy_distances(hands, task._box_states[ids, payload], task._box_size[ids, payload])
        assert distance.max() < .18
        report['maximum_hand_face_distance_m'] = float(distance.max())
        graph = task.relation_runtime.graph
        assert graph.prereq_mask.sum((1, 2)).eq(1).all()
        phi, diag = task._evaluate_relations()
        reward = task.relation_runtime.step(phi, diag['progress'], diag['z_error'],
            diag['feet_height_error'], diag['region_error'])
        mixture = cfg['env']['bridge'].get('start_mixture')
        if mixture is None:
            assert not reward['prerequisite_gate'][graph.prereq_mask.any(-1)].any()
        else:
            rows = torch.arange(task.num_envs, device=task.device)
            near = task._bridge_near_start.clone()
            aligned = task._bridge_aligned_start.clone()
            initial_distance = (task._bridge_start[:, :2]-task._bridge_goal[:, :2]).norm(dim=-1)
            low, high = mixture['near_distance']
            assert ((initial_distance[near] >= low-1e-4) & (initial_distance[near] <= high+1e-4)).all()
            assert (initial_distance[~near] > 2.).all()
            follower = graph.prereq_mask.any(-1).long().argmax(-1)
            leader = graph.prereq_mask[rows, follower].long().argmax(-1)
            assert torch.equal(reward['prerequisite_gate'][rows, follower], reward['raw_own_success'][rows, leader])
            arrived = task._bridge_arrived()
            assert not (arrived & ~near).any()
            report['start_mixture'] = dict(near_count=int(near.sum()), original_count=int((~near).sum()),
                near_aligned_count=int(aligned.sum()), near_random_count=int((near & ~aligned).sum()),
                near_fraction=float(near.float().mean()),
                near_distance_range_m=[float(initial_distance[near].min()), float(initial_distance[near].max())] if near.any() else None,
                original_distance_range_m=[float(initial_distance[~near].min()), float(initial_distance[~near].max())] if (~near).any() else None,
                initial_connected_count=int(arrived.sum()),
                initial_gate_open_count=int(reward['prerequisite_gate'][rows, follower].sum()))
            xy = task._humanoid_root_states[rows, task._roles[:, 1], :2]
            box = task._box_states[rows, task._agent_box_assignment[rows, task._roles[:, 1]], :2]
            direction = box-xy
            delta = task._bridge_goal[:, :2]-xy
            error = (direction[:, 0]*delta[:, 1]-direction[:, 1]*delta[:, 0]).abs()/direction.norm(dim=-1)
            assert not (aligned & ~near).any()
            if aligned.any():
                assert error[aligned].max() < .003
                report['start_mixture']['maximum_alignment_error_m'] = float(error[aligned].max())
                slope = direction[:, 1]/direction[:, 0]
                margin = task._bridge_size[:, 1]/2-slope.abs()*task._bridge_gap/2
                assert margin[aligned].min() >= .198
                report['start_mixture']['minimum_aligned_bridge_margin_m'] = float(margin[aligned].min())
        demo = task.fetch_amp_obs_demo(512)
        assert demo.shape[-1] == 1330 and torch.isfinite(demo).all()
        roles = task._roles.clone()
        tasks = task._tasks4.clone()
        for _ in range(int(os.environ.get('BRIDGE_STEPS', '64'))):
            mu, _ = net.eval_actor(rms(obs))
            obs, rew, done, _ = env.step(mu)
            if not all(torch.isfinite(x).all() for x in (mu, obs, rew)):
                raise RuntimeError('Non-finite PUSH4 rollout')
        other = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
        if mixture is not None:
            counts = task._bridge_start_counts.clone()
            checks = []
            original_check = task._ontop_scene_infeasible

            def reject_once(ids):
                checks.append(torch.stack((task._bridge_near_start[ids], task._bridge_aligned_start[ids])))
                rejected = original_check(ids)
                return torch.ones_like(rejected) if len(checks) == 1 else rejected

            task._ontop_scene_infeasible = reject_once
        task._reset_envs(torch.tensor([0], device=task.device))
        if mixture is not None:
            del task._ontop_scene_infeasible
            assert len(checks) >= 2 and all(torch.equal(x, checks[0]) for x in checks)
            assert (task._bridge_start_counts-counts).sum() == 1
            report['physical_retry_mode_preserved'] = True
        assert torch.equal(other, task._root_states.view(task.num_envs, -1, 13)[1:])
        if cfg['env']['video']['enable']:
            task.reset_buf[:] = 1
            obs = env.reset()
            capture(task, output, 10)
        if os.environ.get('CHECK_INSERTION') == '1':
            report['insertion'] = insertion_check(task, env)
    assert all(torch.equal(value, snapshots[key]) for key, value in model.state_dict().items())
    report.update(envs=task.num_envs, obs_size=task.get_obs_size(), amp_obs_size=task.get_num_amp_obs(),
        physical_objects=task.num_physical_objects, tasks=tasks.tolist(), roles=roles.tolist(),
        partial_reset=True, finite_rollout=True, weights_unchanged=True)
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Bridge PUSH4 validation complete:', output/'validation.json')


if __name__ == '__main__':
    main()
