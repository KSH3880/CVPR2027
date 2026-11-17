"""Run with run.py arguments: restored-policy resets, CPA, and live platform contact."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # Must precede torch.
from isaacgym import gymtorch
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.task_role_spec import task_graph_packet
from env.tasks.multi_agent.collision_reward import compute_agent_cpa_collision_penalty
from utils import stage1_platforms

REPORT = dict(reset_calls=0, source_slots=0, target_slots=0, elevated_sources=0,
              elevated_targets=0, cpa_checks=0, cpa_nonzero=0, scenarios=[],
              distance_checks=0)
ORIGINAL_RESET = stage1_platforms.reset_stage1_platforms


def check_distances(task):
    from utils.stage1_sampling import uses_tokenhsi_sampling
    if not uses_tokenhsi_sampling(task.cfg['env']):
        return
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(
        task.num_envs, task.num_agents, 5).long()
    for skill in ('loco', 'pickUp', 'carryWith'):
        slots = task._collect_random_slots([skill])
        if slots is None:
            continue
        env, owner = slots
        records = packet[env, owner]
        kind = records[:, 1]
        source = task._scenario_physical_object(env, owner)
        human = task._humanoid_root_states[env, owner, :2]
        payload = task._box_states[env, source, :2]
        if skill == 'loco':
            distance = (human - payload).norm(dim=-1)
            maximum = torch.where(kind == 0, 5., 10.)
            assert ((distance >= .9999) & (distance <= maximum + .0001)).all()
            REPORT['distance_checks'] += len(env)
        for category in (2, 3):
            mask = kind == category
            e, a = env[mask], owner[mask]
            if not len(e):
                continue
            if category == 2:
                goal = task._scenario_goal(e, a)
                xy = task._tar_pos[e, goal, :2]
            else:
                support = task._logical_box_order[e, records[mask, 4] - task.num_agents]
                xy = task._box_states[e, support, :2]
            assert ((xy - task._env_origins[e, :2]).abs() <= 4.5001).all()
            assert ((xy - human[mask]).norm(dim=-1) >= .9999).all()
            assert ((xy - payload[mask]).norm(dim=-1) >= .9999).all()
            REPORT['distance_checks'] += len(e)


def checked_reset(task, ids):
    boxes, goals = task._box_states.clone(), task._tar_pos.clone()
    platforms, targets = task._platform_pos.clone(), task._tar_platform_pos.clone()
    ORIGINAL_RESET(task, ids)
    expected_boxes, expected_goals = boxes.clone(), goals.clone()
    expected_source = task._platform_default_pos[ids].clone()
    expected_target = task._tar_platform_default_pos[ids].clone()
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(-1, 2, 5).long()
    for source, skills in ((True, ['loco']), (False, ['loco', 'pickUp', 'carryWith'])):
        slots = task._collect_random_slots(skills)
        if slots is None:
            continue
        env, owner = slots
        category = packet[env, owner, 1]
        keep = category >= 2 if source else category == 2
        env, owner = env[keep], owner[keep]
        logical = packet[env, owner, 3] - task.num_agents
        payload = task._logical_box_order[env, logical]
        size = task._box_size[env, payload]
        rows = torch.searchsorted(ids, env)
        if source:
            position = task._box_states[env, payload, :3]
            z = position[:, 2] - .05
            expected_boxes[env, payload, 2] = position[:, 2]
            plate = expected_source
            key = 'source'
        else:
            goal = packet[env, owner, 4] - task.num_agents - task.num_objects
            position = task._tar_pos[env, goal]
            z = position[:, 2]
            expected_goals[env, goal, 2] = z
            plate = expected_target
            key = 'target'
        assert (z >= size[:, 2] / 2 - 1e-6).all()
        assert (z + size[:, 2] / 2 <= 1.2 + 1e-6).all()
        plate[rows, owner, :2] = position[:, :2]
        plate[rows, owner, 2] = z - size[:, 2] / 2 - .01
        REPORT[key + '_slots'] += len(env)
        REPORT['elevated_' + key + 's'] += int((z > size[:, 2] / 2 + 1e-5).sum())
    torch.testing.assert_close(task._box_states, expected_boxes)
    torch.testing.assert_close(task._tar_pos, expected_goals)
    torch.testing.assert_close(task._platform_pos[ids], expected_source)
    torch.testing.assert_close(task._tar_platform_pos[ids], expected_target)
    untouched = torch.ones(task.num_envs, device=task.device, dtype=torch.bool)
    untouched[ids] = False
    torch.testing.assert_close(task._platform_pos[untouched], platforms[untouched])
    torch.testing.assert_close(task._tar_platform_pos[untouched], targets[untouched])
    REPORT['reset_calls'] += 1


@torch.no_grad()
def check_simulator(player):
    task = player.env.task
    assert task._stage1_platforms and task.num_envs >= 4
    ids = torch.arange(task.num_envs, device=task.device)
    task._skill = ['loco', 'sit', 'climb', 'pickUp', 'carryWith', 'putDown']
    net = player.model.a2c_network
    for enc in (net.actor_encoder, net.critic_encoder):
        assert enc.edge_encoder.bias_projection.shape == (4, 2, 64)
        table = enc.edge_encoder.bias_table()
        torch.testing.assert_close(table, table[:, :1, :1].expand_as(table), atol=0, rtol=0)
        assert enc.build_relation_bias().count_nonzero() == 0
    scenarios = [('sit', 'loco'), ('sit', 'sit'), ('climb', 'loco'), ('climb', 'climb')]
    scenarios += [(p, s) for p in ('holding_at', 'holding_ontop')
                  for s in ('loco', 'pickUp', 'carryWith', 'putDown')]
    scenarios += [('random_scenario', 'loco')]
    if os.environ.get('STAGE1_FIXTURE_ONLY'):
        scenarios = []
    for preset, skill in scenarios:
        task._task_graph_preset = preset
        task._task_role_swap = preset == 'holding_ontop'
        task._skill_init_prob = torch.tensor([float(s == skill) for s in task._skill], device=task.device)
        obs = player.env_reset()
        check_distances(task)
        player.get_batch_size(obs['obs'], 1)
        # Actual partial simulator reset must leave the other environments intact.
        before = task._root_states.view(task.num_envs, -1, 13)[2:].clone()
        obs = player.env_reset(ids[:2])
        check_distances(task)
        torch.testing.assert_close(task._root_states.view(task.num_envs, -1, 13)[2:], before)
        resets = []
        for _ in range(32):
            obs = player.env_reset(resets)
            action = player.get_action(obs, True)
            assert torch.isfinite(action).all() and torch.isfinite(obs['obs']).all()
            obs, reward, done, info = player.env_step(player.env, action)
            assert torch.isfinite(reward).all() and torch.isfinite(task._root_states).all()
            expected = -.5 * compute_agent_cpa_collision_penalty(task._humanoid_root_states[..., :3],
                task._humanoid_root_states[..., 7:10], .7, .99, task.dt)
            torch.testing.assert_close(info['reward_terms'][:, 4].reshape_as(expected), expected)
            REPORT['cpa_checks'] += 1
            REPORT['cpa_nonzero'] += int((expected < 0).sum())
            resets = done.nonzero(as_tuple=False).flatten()[::task.num_agents]
        REPORT['scenarios'].append(dict(task=preset, skill=skill, steps=32))
        print('PASS stage1', preset, skill, flush=True)

    # Move the same GPU collision actors twice: catches stale fixed-base shapes.
    fixtures = []
    for height in (.25, .75):
        # Isolate support physics from the preceding untrained-policy rollout:
        # its fallen humans can otherwise land on or push a fixture box.
        for owner in range(task.num_agents):
            task._humanoid_root_states[:, owner, :3] = task._env_origins + torch.tensor(
                [6. + 2*owner, -6., 1.], device=task.device)
        task._humanoid_root_states[..., 7:13] = 0
        task._platform_pos[ids] = task._platform_default_pos[ids]
        task._tar_platform_pos[ids] = task._tar_platform_default_pos[ids]
        for b in range(task.num_objects):
            task._box_states[:, b, :3] = task._env_origins + torch.tensor([2. + b, 4., .2], device=task.device)
        task._box_states[..., 3:6] = 0
        task._box_states[..., 6] = 1
        task._box_states[..., 7:13] = 0
        for b, plate in ((0, task._platform_pos), (1, task._tar_platform_pos)):
            xy = task._env_origins[:, :2] + torch.tensor([-3. + 1.5*b, -3.], device=task.device)
            plate[:, 0, :2] = xy
            plate[:, 0, 2] = height - .01
            task._box_states[:, b, :2] = xy
            task._box_states[:, b, 2] = height + task._box_size[:, b, 2] / 2 + .05
        task._reset_env_tensors(ids)
        force = gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim))
        force = force.view(task.num_envs, -1, 3)[:, task.num_agents * task.num_bodies:][:, :2]
        peak_vertical_contact = torch.zeros(task.num_envs, 2, device=task.device)
        for _ in range(60):
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim, True)
            task.gym.refresh_net_contact_force_tensor(task.sim)
            peak_vertical_contact = torch.maximum(peak_vertical_contact, force[..., 2])
        task._refresh_sim_tensors()
        bottom = task._box_states[:, :2, 2] - task._box_size[:, :2, 2] / 2
        error = (bottom - height).abs()
        if error.max() >= .012:
            print('FIXTURE DEBUG', dict(height=height, bottom=bottom.tolist(),
                box=task._box_states[:3,:2].tolist(),
                source=task._platform_states[:3].tolist(),
                target=task._tar_platform_states[:3].tolist()), flush=True)
        assert error.max() < .012, (height, error.max().item())
        assert task._box_states[:, :2, 7:10].norm(dim=-1).max() < .08
        assert (task._platform_pos[:, 0, 2] - height + .01).abs().max() < 1e-5
        assert (task._tar_platform_pos[:, 0, 2] - height + .01).abs().max() < 1e-5
        # Contact must occur during landing. A sleeping settled box can report
        # zero force in the final step; height and velocity above check support.
        assert (peak_vertical_contact > .01).all()
        fixtures.append(dict(height=height, max_bottom_error=float(error.max()),
                             min_peak_contact_force=float(peak_vertical_contact.min()),
                             min_final_contact_force=float(force[..., 2].min())))
    REPORT['physics_fixtures'] = fixtures
    if not os.environ.get('STAGE1_FIXTURE_ONLY'):
        assert REPORT['elevated_sources'] > 0 and REPORT['elevated_targets'] > 0
    output = Path(task.cfg['args'].output_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'simulator_verification.json').write_text(json.dumps(REPORT, indent=2))
    print('PASS Stage1 simulator:', json.dumps(REPORT), flush=True)


if __name__ == '__main__':
    stage1_platforms.reset_stage1_platforms = checked_reset
    MAPlayerContinuous.run = check_simulator
    MAPlayerContinuous.run_eval = check_simulator
    entry.main()
