"""Bounded actual-simulator assertions, run via the ordinary run.py CLI arguments.

Example: source runtime_env.sh; PYTHONPATH=tokenhsi python this_file.py --task ... --test ...
This is not a pytest module: IsaacGym MUST be imported before torch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # noqa: F401
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.relation_reward import relation_step, relation_progress


@torch.no_grad()
def check_simulator(self):
    task = self.env.task
    self._reward_debug = False
    N, M = task.num_envs, task.num_agents
    obs = self.env_reset()
    self.get_batch_size(obs['obs'], 1)
    if not task._state_relation:
        # Strict legacy checkpoint restore already happened before entering here.
        for _ in range(5):
            obs, reward, done, info = self.env_step(self.env, self.get_action(obs, True))
            assert torch.isfinite(obs['obs'] if isinstance(obs, dict) else obs).all()
            assert torch.isfinite(reward).all()
            obs = self.env_reset(done.nonzero().flatten()[::M])
        print('PASS legacy strict-load / finite simulator rollout', flush=True)
        return
    runtime = task.relation_runtime
    mixed = getattr(task, '_ontop_mixed', False)
    suffix_width = task.get_relation_suffix_size()
    if mixed:
        assert task._ontop_scenario.bincount(minlength=3).min() > 0
        independent = task._ontop_scenario == 1
        assert not (task._agent_box_assignment[independent] == 2).any()
        fixed_pose = task._box_states[independent, 2, :7].clone()
        _, _, _, is_top = task._ontop_geometry()
        assert not task._ontop_reset_rejected(torch.arange(N, device=task.device)).any()
    cfg = task._relation_cfg
    progress_cfg = cfg.get('progress', {})
    ids = torch.arange(N, device=task.device)
    torch.testing.assert_close(runtime.phi, task._evaluate_relations(ids)[0])
    assert not runtime.done.all(-1).any()
    if mixed:
        # Commit the initial all-env reset before exercising a subset reset:
        # Isaac Gym permits only one indexed root-state setter per physics step.
        obs, _, _, _ = self.env_step(self.env, self.get_action(obs, True))
        torch.testing.assert_close(task._box_states[independent, 2, :7], fixed_pose, atol=1e-6, rtol=0)
    before = runtime.suffix().clone()
    self.env_reset(torch.tensor([0], device=task.device))
    torch.testing.assert_close(runtime.suffix()[1:], before[1:])
    torch.testing.assert_close(runtime.phi[:1], task._evaluate_relations(ids[:1])[0])
    # Optional observation clipping cannot touch poses or semantic state.
    old_clip = self.env.clip_obs
    self.env.clip_obs = .1
    clipped = self.env._policy_observation()
    bypass_offset = M * 223 + task.num_objects * 30
    torch.testing.assert_close(clipped[:, bypass_offset:], task.obs_buf[:, bypass_offset:])
    self.env.clip_obs = old_clip
    task.relation_runtime.done[0] = True
    task._compute_reset()
    assert not task.reset_buf[0]  # success alone never resets the scene
    self.env_reset(torch.tensor([0], device=task.device))
    prev_done_rows = []
    for step in range(64):
        obs = self.env_reset(prev_done_rows)
        if mixed:
            fixed_pose = task._box_states[independent, 2, :7].clone()
        before_phi = runtime.phi.clone()
        before_a, before_done = runtime.achieved.clone(), runtime.done.clone()
        saved_obs = (obs['obs'] if isinstance(obs, dict) else obs).clone()
        action = self.get_action(obs, True)
        obs, reward, done, info = self.env_step(self.env, action)
        phi, relation_diag = task._evaluate_relations()
        objects = task._assigned_box_values(task._box_states)[..., :3]
        root = task._humanoid_root_states[..., :3]
        ph = relation_progress(task._prev_root_pos, root, objects, task.dt, progress_cfg)
        targets = task._ontop_geometry()[2] if mixed else task._tar_pos
        pa = relation_progress(task._prev_box_pos, objects, targets, task.dt, progress_cfg)
        expected = relation_step(before_phi, phi, torch.stack([ph, pa], -1).flatten(1),
                                 before_a, before_done, runtime.graph,
                                 state_weight=cfg.get('state_reward_weight', .2),
                                 progress_weight=cfg.get('progress_reward_weight', .2),
                                 success_bonus=cfg.get('subgoal_success_bonus', 0.),
                                 beta=cfg.get('soft_gate', {}).get('beta', 30.),
                                 gate_center=cfg.get('soft_gate', {}).get('center', .8),
                                 satisfaction_threshold=cfg.get('satisfaction_threshold', .9),
                                 require_current_target_prerequisites=cfg.get('success', {}).get(
                                     'require_current_target_prerequisites', False),
                                 at_z_error=relation_diag.get('goal_z_error'),
                                 success_z_tolerance=cfg.get('success', {}).get('z_tolerance'),
                                 saturate_edge_rewards=cfg.get('success', {}).get(
                                     'saturate_edge_rewards', False),
                                 progress_kind=progress_cfg.get('kind', 'distance'),
                                 approach_radius=progress_cfg.get('approach_radius'),
                                 edge_distance_xy=torch.stack([
                                     (objects[..., :2] - root[..., :2]).norm(dim=-1),
                                     relation_diag['goal_xy_error']], -1).flatten(1),
                                 saturate_edge_rewards_while_current=cfg.get('success', {}).get(
                                     'saturate_edge_rewards_while_current', False),
                                 current_saturation_z_tolerance=cfg.get('success', {}).get(
                                     'current_saturation_z_tolerance'),
                                 current_success_reward=cfg.get('success', {}).get(
                                     'current_success_reward', 0.))
        torch.testing.assert_close(runtime.phi, phi)
        torch.testing.assert_close(runtime.achieved, expected['achieved_next'])
        torch.testing.assert_close(runtime.done, expected['done_next'])
        expected_suffix = runtime.suffix()
        if mixed:
            expected_suffix = torch.cat([expected_suffix, task._ontop_scenario[:, None].float(),
                                         task._ontop_base_agent[:, None].float()], -1)
        torch.testing.assert_close(task.obs_buf[:, -suffix_width:], expected_suffix)
        torch.testing.assert_close(info['policy_obs'][:, -suffix_width:], expected_suffix)
        terms = info['reward_terms'].reshape(N, M, -1)
        state = terms[..., :2].clone()
        progress = terms[..., 2:4].clone()
        task_reward = terms[..., :5].sum(-1)
        if mixed:
            state[..., 1] += terms[..., 8]
            progress[..., 1] += terms[..., 9]
            task_reward += terms[..., 8:10].sum(-1)
        torch.testing.assert_close(state, expected['state_component'].reshape(N, M, 2))
        torch.testing.assert_close(progress, expected['progress_component'].reshape(N, M, 2))
        torch.testing.assert_close(task_reward, expected['agent_task_reward'], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(terms[..., -1].flatten(), reward)
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(reward).all()
        if mixed:
            torch.testing.assert_close(task._box_states[independent, 2, :7], fixed_pose, atol=1e-6, rtol=0)
            indices = torch.arange(N - 1, -1, -1, device=task.device)
            network = self.model.a2c_network
            for encoder in ((network.actor_encoder, network.critic_encoder) if step == 0 else ()):
                # PPO minibatch reordering must reconstruct the same scene graph.
                normalized = self._preproc_obs(saved_obs)
                torch.testing.assert_close(encoder(normalized[indices]), encoder(normalized)[indices], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(saved_obs[:, -suffix_width:][:, :4],
                                  torch.stack([before_phi[:, 0], torch.sigmoid(30 * (before_phi[:, 0] - .8)),
                                               (before_phi[:, 0] >= .9).float(), before_a[:, 0].float()], -1))
        prev_done_rows = done.nonzero().flatten()[::M]
    print('PASS relation simulator: M={} O={}, 64 transitions, reset subset, stored suffix, '
          'reward/history agreement, clipping bypass, no success termination'.format(
              M, task.num_objects), flush=True)


if __name__ == '__main__':
    MAPlayerContinuous.run = check_simulator
    entry.main()
