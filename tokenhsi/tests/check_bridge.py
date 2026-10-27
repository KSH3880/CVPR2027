import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from torch import nn

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from env.tasks.multi_agent.bridge_geometry import bridge_trajectory
from learning.multi_agent.bridge_network import BridgeBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def check_layout(task):
    b, m, slot = task._bridge_cfg, task.num_agents, task._bridge_slot
    origin = task._env_origins
    height = task._stage_height[:, None]
    mapping = task._logical_box_order
    assert ((mapping == slot).sum(-1) == 1).all()
    root = task._humanoid_root_states[..., :3] - origin[:, None]
    assert (root[..., :2].abs() <= task._stage_half_size[:, None] - b['spawn_margin'] + 1e-4).all()
    assert (root[..., 2] > height + .5).all()
    boxes = task._box_states[torch.arange(task.num_envs, device=task.device)[:, None], mapping[:, :m], :3] - origin[:, None]
    far = task._far_stage_states[:, :3] - origin
    assert ((boxes[..., :2] - far[:, None, :2]).abs() <= task._far_stage_half_size[:, None] + 1e-4).all()
    torch.testing.assert_close(far[:, 2], height[:, 0] / 2 + 1e-3)
    present = task._bridge_present
    bridge = task._bridge_states[:, :3]
    torch.testing.assert_close(bridge[present], task._bridge_goal[present])
    assert (task._bridge_size[:, 0] - task._bridge_gap + 2 * b['clearance']).abs().max() < 1e-4
    return dict(present_fraction=float(present.float().mean()), gap=task._bridge_gap.tolist(),
                bridge_size=task._bridge_size.tolist())


def check_observation(task, obs):
    m, o = task.num_agents, task.num_objects
    root_h = obs[:, :m * 223].reshape(-1, m, 223)[..., 0]
    # Freshly reset rows were observed from the kinematic reference state, the rest from simulation.
    fresh = (task.progress_buf == 0)[:, None, None]
    root = torch.where(fresh, task._kinematic_humanoid_rigid_body_states[..., 0, :3], task._rigid_body_pos[..., 0, :])
    support = task._terrain_height(root[..., :2]) - task._env_origins[:, None, 2]
    torch.testing.assert_close(root_h, root[..., 2] - task._env_origins[:, None, 2] - support)
    poses = obs[:, m * 223 + o * 30 + m:][:, :(2 * m + o) * 7].reshape(-1, 2 * m + o, 7)
    slot = (task._logical_box_order == task._bridge_slot).long().argmax(-1)
    bridge_pose = poses[torch.arange(len(obs), device=obs.device), m + slot, :3]
    torch.testing.assert_close(bridge_pose, task._bridge_states[:, :3] - task._env_origins, atol=1e-4, rtol=0)
    torch.testing.assert_close(obs[:, -m:], task._before_predecessor.float())
    return dict(root_height_min=float(root_h.min()), root_height_max=float(root_h.max()))


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test, args.eval = True, os.environ.get('BRIDGE_TRAIN_INIT') != '1'
    args.task_graph = 'random_scenario'
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    builder = BridgeBuilder(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents,
        num_objects=task.num_objects, observation_mode='clean_scene',
        scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=7,
        scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device).to(task.device)
    model = nn.Module(); model.a2c_network = net
    checkpoint = torch.load(cfg['experiment']['transfer']['checkpoint'], map_location=task.device, weights_only=False)
    report = transfer_stage1(model, checkpoint)
    rms = SceneRunningMeanStd([223, 30, 1], [task.num_agents, task.num_objects, task.num_agents],
        normalized_sizes=[223, 30, 0], kinematic_size=7,
        extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True); rms.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    results = []
    steps = int(os.environ.get('BRIDGE_STEPS', '64'))
    with torch.no_grad():
        for preset in os.environ.get('BRIDGE_PRESETS', 'random_scenario,sit,climb').split(','):
            task._task_graph_preset = preset
            task.reset_buf[:] = 1
            obs = env.reset()
            layout = check_layout(task)
            heights = check_observation(task, obs)
            demo = task.fetch_amp_obs_demo(128)
            if demo.shape[-1] != 10 * 133 or not torch.isfinite(demo).all():
                raise RuntimeError('AMP demos must be finite with four families')
            tracking, lowest, falls = [], [], 0
            for step in range(steps):
                mu, _ = net.eval_actor(rms(obs)); value = net.eval_critic(rms(obs))
                obs, reward, done, info = env.step(mu)
                if not all(torch.isfinite(x).all() for x in (mu, value, obs, reward)):
                    raise RuntimeError('Non-finite bridge rollout')
                fresh = task.progress_buf > 0
                elapsed = task.progress_buf * task.dt - task._bridge_wait
                target, _ = bridge_trajectory(task._bridge_start, task._bridge_goal, elapsed, task._bridge_duration)
                tracking.append(float((task._bridge_states[fresh, :3] - target[fresh]).norm(dim=-1).max()) if fresh.any() else 0.)
                standing = ~task._terminate_buf.view(task.num_envs, -1).bool() & fresh[:, None]
                ground = task._terrain_height(task._rigid_body_pos[..., 0, :2])
                if standing.any():
                    lowest.append(float((task._rigid_body_pos[..., 0, 2] - ground)[standing].min()))
                falls += int(task._terminate_buf.sum())
            check_observation(task, obs)
            untouched = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
            task._reset_envs(torch.tensor([0], device=task.device))
            if not torch.equal(untouched, task._root_states.view(task.num_envs, -1, 13)[1:]):
                raise RuntimeError('Partial reset changed another environment')
            results.append(dict(preset=preset, templates=task._templates.tolist(),
                                bridge_tracking_max_m=max(tracking), lowest_root_above_terrain_m=min(lowest),
                                terminations=falls, reward=float(reward.mean()), **layout, **heights))
        crossing = cross_present_bridge(task, env, net, rms)
        gap_fall = drop_into_gap(task, env, net, rms)
    if any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()):
        raise RuntimeError('Validation changed weights')
    output = Path(args.output_path); output.mkdir(parents=True, exist_ok=True)
    report.update(physics=results, crossing=crossing, gap_fall=gap_fall, training_executed=False, agents=task.num_agents,
                  physical_objects=task.num_physical_objects, obs_size=task.get_obs_size(),
                  amp_obs_size=task.get_num_amp_obs())
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Bridge verification complete:', output / 'validation.json')


def shift_humans(task, xy):
    delta = xy - task._humanoid_root_states[:, :, :2]
    task._humanoid_root_states[:, :, :2] += delta
    task._kinematic_humanoid_rigid_body_states[:, :, :, :2] += delta[:, :, None]


def cross_present_bridge(task, env, net, rms):
    # Connected bridges only; human 0 and its task box are aligned with the bridge centre line.
    task._bridge_cfg['present_probability'] = 1.
    task._task_graph_preset = 'climb'
    task.reset_buf[:] = 1
    obs = env.reset()
    ids = torch.arange(task.num_envs, device=task.device)
    landing, edge = task._bridge_goal, task._env_origins[:, 0] + task._stage_half_size[:, 0]
    xy = task._humanoid_root_states[:, :, :2].clone()
    xy[:, 0] = torch.stack((edge - 1.5, landing[:, 1]), -1)
    shift_humans(task, xy)
    box = task._logical_box_order[:, 0]
    task._box_states[ids, box, 0] = edge + task._bridge_gap + .8
    task._box_states[ids, box, 1] = landing[:, 1]
    task._reset_env_tensors(ids)
    task._refresh_sim_tensors()
    crossed = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    fallen = torch.zeros_like(crossed)
    for _ in range(int(os.environ.get('BRIDGE_CROSS_STEPS', '300'))):
        mu, _ = net.eval_actor(rms(obs))
        obs, _, _, _ = env.step(mu)
        active = ~(crossed | fallen)
        fallen |= active & task._terminate_buf.view(task.num_envs, -1)[:, 0].bool()
        crossed |= active & task._bridge_crossed[:, 0]
    if not crossed.any():
        raise RuntimeError('No human crossed a connected bridge')
    return dict(crossed_fraction=float(crossed.float().mean()), fall_before_crossing_fraction=float(fallen.float().mean()))


def drop_into_gap(task, env, net, rms):
    # Hold every bridge outside the channel span, then place human 0 above the empty gap.
    task._task_graph_preset = 'random_scenario'
    task.reset_buf[:] = 1
    obs = env.reset()
    landing = task._bridge_goal.clone()
    task._bridge_goal[:, 1] += 12.
    task._bridge_start[:] = task._bridge_goal
    ids = torch.arange(task.num_envs, device=task.device)
    task._bridge_states[:, :3] = task._bridge_goal
    task._bridge_states[:, 7:] = 0
    xy = task._humanoid_root_states[:, :, :2].clone()
    xy[:, 0] = landing[:, :2]
    shift_humans(task, xy)
    task._reset_env_tensors(ids)
    task._refresh_sim_tensors()
    terminated = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    for step in range(60):
        mu, _ = net.eval_actor(rms(obs))
        obs, _, done, _ = env.step(mu)
        terminated |= task._terminate_buf.view(task.num_envs, -1)[:, 0].bool()
        if terminated.all():
            break
    if not terminated.all():
        raise RuntimeError('A human dropped into the gap without terminating')
    return dict(steps=step + 1, terminated_fraction=float(terminated.float().mean()))


if __name__ == '__main__':
    main()
