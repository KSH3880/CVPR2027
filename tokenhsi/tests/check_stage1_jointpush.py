import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from torch import nn
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from learning.multi_agent.stage1_push_network import Stage1PushBuilder


def check(player):
    task = player.env.task
    report = dict(reset_count=32, loco=0, contact_rsi=0, policy_steps=16)
    report['preset'] = task._task_graph_preset
    source_distances, goal_distances = [], []
    for _ in range(report['reset_count']):
        obs = player.env_reset()
        player.get_batch_size(obs['obs'], 1)
        push = task._templates(torch.arange(task.num_envs, device=task.device)) == 5
        assert set(task._reset_ref_slots) == {'loco'}
        assert not task._interaction.any()
        assert not hasattr(task, '_size_rsi_cache')
        loco = push & ~task._interaction
        report['loco'] += len(task._reset_ref_slots['loco'][0])
        report['contact_rsi'] += int(task._interaction.sum())
        distance = (task._box_states[:, :2, :2]-task._humanoid_root_states[..., :2]).norm(dim=-1)[loco]
        source_distances.extend(distance.tolist())
        goals = (task._tar_pos[..., :2]-task._box_states[:, :2, :2]).norm(dim=-1)[push]
        goal_distances.extend(goals.flatten().tolist())
        assert ((distance >= 2.-1e-4) & (distance <= 10.+1e-4)).all()
        assert ((goals >= 2.-1e-4) & (goals <= 10.+1e-4)).all()
        torch.testing.assert_close(task._tar_pos[..., 2][push], task._box_size[:, :2, 2][push]/2)
        assert not task._ontop_scene_infeasible(torch.arange(task.num_envs, device=task.device)).any()
        slots = task._reset_ref_slots.get('loco')
        if slots is not None:
            mids, times = task._reset_ref_motion_ids['loco'], task._reset_ref_motion_times['loco']
            for mask, name in ((push[slots], 'push_loco'), (~push[slots], 'loco')):
                if mask.any():
                    q = task._motion_lib[name].get_motion_state(mids[mask], times[mask])[1]
                    torch.testing.assert_close(task._humanoid_root_states[slots[0][mask], slots[1][mask], 3:7], q)
    report['source_distance'] = [min(source_distances), max(source_distances)] if source_distances else []
    report['goal_distance'] = [min(goal_distances), max(goal_distances)] if goal_distances else []
    report['physical_boxes'] = []
    for owner in range(2):
        if not push[0, owner]:
            continue
        handle = task._box_handles[owner]
        mass = sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[0], handle))
        shapes = task.gym.get_actor_rigid_shape_properties(task.envs[0], handle)
        assert abs(mass-20.) < 1e-4
        assert len(shapes) == 1 and abs(shapes[0].friction-.4) < 1e-6
        report['physical_boxes'].append(dict(mass=mass, friction=shapes[0].friction, shapes=len(shapes)))
    goal = task._tar_pos.clone()
    for _ in range(report['policy_steps']):
        action = player.get_action(obs if isinstance(obs, dict) else {'obs': obs}, player.is_determenistic)
        obs, reward, _, _ = player.env_step(player.env, action)
        state = obs['obs'] if isinstance(obs, dict) else obs
        assert torch.isfinite(state).all() and torch.isfinite(reward).all()
        torch.testing.assert_close(task._tar_pos, goal)
    demo = task._fetch_conditioned_amp_demo(512)
    assert demo.shape == (512, 1330) and torch.isfinite(demo).all()
    families = demo.reshape(512, 10, 133)[..., -4:]
    assert (families.sum(-1) == 1).all()
    assert len(families[:, 0].argmax(-1).unique()) == 4
    builder = Stage1PushBuilder()
    builder.load(entry.cfg_train['params']['network'])
    model = nn.Module()
    model.a2c_network = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1330,),
        value_size=1, num_agents=2, num_objects=4, observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=7, scene_arena_scale=5., relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device).to(task.device)
    before = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    with torch.enable_grad():
        state = player._preproc_obs(state).detach()
        mu, _ = model.a2c_network.eval_actor(state)
        value = model.a2c_network.eval_critic(state)
        disc = model.a2c_network.eval_disc(player._preproc_amp_obs(demo).detach())
        loss = mu.square().mean()+value.square().mean()+disc.square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    report['scratch_gradient_loss'] = float(loss.detach())
    report['scratch_updated_tensors'] = sum(not torch.equal(before[k], p) for k, p in model.named_parameters() if k in before)
    assert report['scratch_updated_tensors'] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report['amp_demo_shape'] = list(demo.shape)
    directory = Path(task.cfg['args'].output_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'simulator_verification.json').write_text(json.dumps(report, indent=2))
    print('PASS stage1_jointpush:', json.dumps(report), flush=True)


if __name__ == '__main__':
    MAPlayerContinuous.run = check
    MAPlayerContinuous.run_eval = check
    entry.main()
