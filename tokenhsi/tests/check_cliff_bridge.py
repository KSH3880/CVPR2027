import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import capture_bridge_chamfer as capture
import check_sensor_stage2 as checks
from isaacgym import gymapi
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import yaml

from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder, transfer
from utils.coupled_push_spec import checkpoint_contract
from utils.sensor_stage2_spec import paired_people


def inspect_surface(task, obs):
    expected = task._kinematic_humanoid_rigid_body_states[:, :, 0, 2]-task._env_origins[:, 2, None]-(
        task._start_surface > 0)*task._surface_height[:, 1, None]
    bridge = task._scene_ids == 1
    humans = obs[:, :4*223].reshape(task.num_envs, 4, 223)
    torch.testing.assert_close(humans[bridge, :, 0], expected[bridge], atol=2e-3, rtol=0)
    records = 0
    for env in range(task.num_envs):
        for bank, density in ((task._bridge_bank, task._joint['box']['density']),
                              (task._controlled_bank, task._scene_cfg['bridge']['controller']['density'])):
            for slot in range(bank, bank+3):
                handle = task._box_handles[env*task.num_physical_objects+slot]
                props = task.gym.get_actor_rigid_shape_properties(task.envs[env], handle)
                assert len(props) == 2
                np.testing.assert_allclose([p.friction for p in props], [.4, 1.], atol=1e-6)
                mass = task.gym.get_actor_rigid_body_properties(task.envs[env], handle)[0].mass
                expected_mass = float(task._box_size[env, slot].prod())*density
                assert abs(mass-expected_mass) < expected_mass*2e-5
                records += 1
    return records


def inspect(task, obs, stats):
    ids = torch.arange(task.num_envs, device=task.device)
    cliff = task._scene_ids == 0
    assert not (task._scene_ids == 2).any()
    assert task._unreachable[cliff].all()
    assert task._start_surface[cliff].eq(1).all()
    assert task._before_predecessor[cliff].eq(-1).all()
    bridge = task._scene_ids == 1
    branch = task._bridge_branch
    for kind, count in ((0, 4), (1, 3), (2, 1)):
        chosen = bridge & (branch == kind)
        assert task._crosser_mask[chosen].sum(-1).eq(count).all()
    paired = paired_people(task._before_predecessor)
    assert paired[bridge & (branch == 1)].sum(-1).eq(2).all()
    controlled = ids[bridge & (branch == 0)]
    if len(controlled):
        mapped = task._logical_box_order[controlled]
        assert (mapped == task._bridge_slot[controlled, None]).sum(-1).eq(1).all()
        assert not task._bridge_pushers[controlled].any()
        error = (task._box_states[controlled, task._bridge_slot[controlled], :3]-task._controlled_goal[controlled]).norm(dim=-1)
        assert (error < 30).all(), ('controlled reset', error.tolist())
    violation = checks.check_sensor(task, obs, stats)
    stats['sensor_tolerance_violation_max'] = max(stats.get('sensor_tolerance_violation_max', 0.), violation)
    assert torch.isfinite(obs).all()
    if task._scene_cfg['bridge'].get('top_surface'):
        stats['surface_actor_checks'] = stats.get('surface_actor_checks', 0)+inspect_surface(task, obs)
    if hasattr(task, '_compact_actor_ids'):
        states = task._root_states[task._compact_actor_ids]
        active = task._compact_active
        assert active.sum(-1).eq(torch.where(cliff,1,2)).all()
        for slot in range(9):
            chosen = active[:,slot]
            surface,item = (0,0) if slot == 0 else (1,int(slot >= 5))
            torch.testing.assert_close(states[chosen,slot,:3],task._static_centers[chosen,surface,item],atol=1e-4,rtol=0)
        gap = task._bridge_gap-task._box_size[ids,task._bridge_slot,0]
        torch.testing.assert_close(gap,torch.full_like(gap,.1),atol=1e-5,rtol=0)
        stats['compact_requested'] = torch.bincount(task._compact_requested.flatten(),minlength=4).tolist()
        stats['compact_executed'] = torch.bincount(task._compact_kind.flatten(),minlength=4).tolist()
    return dict(cliff=int(cliff.sum()), branches=torch.bincount(branch[bridge], minlength=3).tolist(),
        before_pairs=int((task._before_predecessor >= 0).sum()), retries=int(task._sampling_retries),
        **(dict(boundary_executed=torch.bincount(task._compact_kind.flatten(),minlength=4).tolist(),
                boundary_fallback=task._compact_failures.tolist()) if hasattr(task,'_compact_kind') else {}))


@torch.no_grad()
def main():
    args = capture.get_args()
    if not args.eval:
        raise ValueError('Branch checks require --test --eval and an explicit --cfg_env')
    cfg, train, _ = capture.load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    cfg['env']['video'].update(enable=True, width=1280, height=720, fov=65.)
    torch.set_num_threads(4)
    capture.set_seed(42)
    task, env = capture.parse_task(args, cfg, train['params']['config'], capture.parse_sim_params(args, cfg, train))
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    stats = dict(resets=[], graphics_device=task.graphics_device_id, rendered=[])
    builder = SensorStage2Builder()
    builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=4, num_objects=8,
        observation_mode='clean_scene', scene_entity_sizes=task.get_scene_entity_sizes(),
        scene_kinematic_size=7, scene_arena_scale=task.get_scene_arena_scale(),
        relation_reward_mode=task._relation_cfg['mode'], relation_graph_spec=task._relation_graph_spec,
        device=task.device).to(task.device)
    model = nn.Module()
    model.a2c_network = net
    source = torch.load(cfg['experiment']['transfer']['checkpoint'], map_location='cpu', weights_only=False)
    transfer(model, source)
    model.eval()
    rms = SceneRunningMeanStd([223, 30, 1], [4, 8, 4], normalized_sizes=[223, 30, 0],
        kinematic_size=7, extra_passthrough_size=task.get_relation_suffix_size()).to(task.device).eval()
    rms.load_state_dict(source['running_mean_std'])
    torch.save(dict(model=model.cpu().state_dict(), running_mean_std=source['running_mean_std'],
        amp_input_mean_std=source['amp_input_mean_std'], relation_metadata=source['relation_metadata'],
        agent_collision_config=source['agent_collision_config'], coupled_push_contract=checkpoint_contract(cfg['env']),
        relation_experiment_config=cfg, initialization_only=True), output/'initialization_only.pth')
    model.to(task.device)
    with (output/'relation_config.yaml').open('w') as stream:
        yaml.safe_dump({k: cfg[k] for k in ('env', 'sim', 'experiment')}, stream, sort_keys=False)
    compact = hasattr(task, '_compact_kind')
    steps = 16 if compact else 64
    modes = ['cliff','controlled','free','cooperative']
    if compact:
        modes += ['entry_before','entry_after','exit_before']
    sheet = Image.new('RGB', (1280, 360*((len(modes)+1)//2)))
    handle = task._video_cam
    for iteration, name in enumerate(modes):
        task._scene_probabilities[:] = task._scene_probabilities.new_tensor([1., 0., 0.] if name == 'cliff' else [0., 1., 0.])
        task._branch_mode = 'training' if name == 'cliff' else 'free' if name.startswith(('entry','exit')) else name
        if compact:
            task._compact_test_kind = {'entry_before':1,'entry_after':2,'exit_before':3}.get(name)
        obs = env.reset()
        row = inspect(task, obs, stats)
        if compact and name.startswith(('entry','exit')):
            mode = task._compact_test_kind
            assert (task._compact_kind == mode).any(), ('no valid boundary RSI',name,row)
        assert torch.isfinite(net.eval_critic(rms(obs))).all()
        row['mode'] = name
        stats['resets'].append(row)
        obs, _, _, _ = env.step(net.eval_actor(rms(obs))[0].clamp(-1, 1))
        before = task._root_states.clone()
        branches = task._bridge_branch.clone()
        env.reset(torch.tensor([0], device=task.device))
        actors = task.get_num_actors_per_env()
        torch.testing.assert_close(before[actors:], task._root_states[actors:], atol=0, rtol=0)
        assert torch.equal(branches[1:], task._bridge_branch[1:])
        obs = task.obs_buf
        current = 0 if name == 'cliff' else 1
        target = task._static_centers[0, current, :2].mean(0).cpu().numpy() if current else task._static_centers[0, 0, 0].cpu().numpy()
        base = task.gym.get_env_origin(task.envs[0])
        target -= np.array([base.x, base.y, base.z])
        radius = float(2*task._surface_half[0, current, :, 0].sum()+task._bridge_gap[0])*1.2 if current else 20.
        offset = capture.scene.rotate_xy(torch.tensor([[.1*radius, .75*radius]]), task._structure_yaw[0, current].cpu())[0].numpy()
        task.gym.set_camera_location(handle, task.envs[0], gymapi.Vec3(*(target+np.array([*offset, radius]))), gymapi.Vec3(*target))
        for step in range(steps):
            action = net.eval_actor(rms(obs))[0].clamp(-1, 1)
            obs, reward, done, info = env.step(action)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if name == 'controlled':
                ids = torch.arange(task.num_envs, device=task.device)
                error = (task._box_states[ids, task._bridge_slot, :3]-task._controlled_goal).norm(dim=-1)
                assert (error < 30).all(), ('controlled step', step, error.tolist(), task._box_states[ids, task._bridge_slot, :3].tolist(), task._controlled_goal.tolist())
            if step == 0:
                task.gym.fetch_results(task.sim, True)
                task.gym.step_graphics(task.sim)
                task.gym.render_all_camera_sensors(task.sim)
                frame = capture.pixels(task, 0, handle, 1280, 720)
                stats['rendered'].append(float(np.asarray(frame).std()) > 2)
                ImageDraw.Draw(frame).text((12, 12), name, fill='white')
                frame.save(output/(name+'.png'))
                sheet.paste(frame.resize((640, 360)), ((iteration%2)*640, (iteration//2)*360))
            pending = done.nonzero(as_tuple=False)[::4, 0]
            obs = env.reset(pending)
    sheet.save(output/'starts_overview.png')
    if compact:
        stats['distance_resets'] = []
        task._compact_test_kind = None
        for alpha in (0.,1.):
            task._compact_test_alpha = alpha
            for mode in ('controlled','free','cooperative'):
                task._branch_mode = mode
                for _ in range(2):
                    obs = env.reset()
                    row = inspect(task,obs,stats)
                    row.update(alpha=alpha,mode=mode)
                    stats['distance_resets'].append(row)
                    obs,_,_,_ = env.step(net.eval_actor(rms(obs))[0].clamp(-1,1))
        task._compact_test_alpha = None
    diagnostics = task._consume_sampling_diagnostics()
    assert all(torch.isfinite(v).all() for v in diagnostics.values())
    stats.update(partial_reset=True, finite=True, screenshots=len(modes), simulation_steps=steps*len(modes),
        diagnostics={k: float(v) for k, v in diagnostics.items() if k.startswith(('sampling/before', 'controller/', 'cliff/', 'compact/'))})
    (output/'physics.json').write_text(json.dumps(stats, indent=2))
    task.gym.destroy_sim(task.sim)
    print(json.dumps(stats), flush=True)
    assert all(stats['rendered'])


if __name__ == '__main__':
    main()
