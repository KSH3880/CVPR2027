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
from env.tasks.multi_agent.relation_reward import relation_step, velocity_progress
from env.tasks.multi_agent.humanoid_ma_carry import compute_walk_reward, compute_carry_reward, compute_handheld_reward


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
    cfg = task._relation_cfg
    progress_cfg = cfg.get('progress', {})
    progress_mode = progress_cfg.get('mode', 'gaussian')
    progress_args = (task.dt, progress_cfg.get('target_speed', 1.5),
                     progress_cfg.get('velocity_scale', 5.),
                     progress_cfg.get('normalization_epsilon', 1e-6), progress_mode)
    ids = torch.arange(N, device=task.device)
    torch.testing.assert_close(runtime.phi, task._evaluate_relations(ids)[0])
    assert not runtime.done.all(-1).any()
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
    negative_progress_count = torch.zeros((), device=task.device)
    for step in range(64):
        obs = self.env_reset(prev_done_rows)
        before_phi = runtime.phi.clone()
        before_a, before_done = runtime.achieved.clone(), runtime.done.clone()
        saved_obs = (obs['obs'] if isinstance(obs, dict) else obs).clone()
        action = self.get_action(obs, True)
        obs, reward, done, info = self.env_step(self.env, action)
        phi, _ = task._evaluate_relations()
        objects = task._assigned_box_values(task._box_states)[..., :3]
        root = task._humanoid_root_states[..., :3]
        ph = velocity_progress(task._prev_root_pos, root, objects, *progress_args)
        pa = velocity_progress(task._prev_box_pos, objects, task._tar_pos, *progress_args)
        negative_progress_count += (ph < 0).sum() + (pa < 0).sum()
        expected = relation_step(before_phi, phi, torch.stack([ph, pa], -1).flatten(1),
                                 before_a, before_done, runtime.graph,
                                 state_weight=cfg.get('state_delta_weight', 1.),
                                 velocity_weight=cfg.get('velocity_progress_weight', .2),
                                 success_bonus=cfg.get('subgoal_success_bonus', 5.),
                                 beta=cfg.get('soft_gate', {}).get('beta', 30.),
                                 gate_center=cfg.get('soft_gate', {}).get('center', .8),
                                 satisfaction_threshold=cfg.get('satisfaction_threshold', .9),
                                 progress_mode=progress_mode)
        torch.testing.assert_close(runtime.phi, phi)
        torch.testing.assert_close(runtime.achieved, expected['achieved_next'])
        torch.testing.assert_close(runtime.done, expected['done_next'])
        torch.testing.assert_close(task.obs_buf[:, -9 * M:], runtime.suffix())
        torch.testing.assert_close(info['policy_obs'][:, -9 * M:], runtime.suffix())
        terms = info['reward_terms'].reshape(N, M, -1)
        torch.testing.assert_close(terms[..., :2], expected['state_component'].reshape(N, M, 2))
        torch.testing.assert_close(terms[..., 2:4], expected['velocity_component'].reshape(N, M, 2))
        torch.testing.assert_close(terms[..., :5].sum(-1), expected['agent_task_reward'], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(terms[..., -1].flatten(), reward)
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(reward).all()
        torch.testing.assert_close(saved_obs[:, -9 * M:-9 * M + 4] if M > 1 else saved_obs[:, -9:-5],
                                  torch.stack([before_phi[:, 0], torch.sigmoid(30 * (before_phi[:, 0] - .8)),
                                               (before_phi[:, 0] >= .9).float(), before_a[:, 0].float()], -1))
        prev_done_rows = done.nonzero().flatten()[::M]
    if progress_mode == 'signed_linear':
        assert negative_progress_count > 0, 'Signed simulation must exercise negative progress'
    # Compare only original velocity terms outside pinning and anti-kick masks.
    prev = torch.tensor([[0., 0., 1.]], device=task.device)
    cur = torch.tensor([[.05, 0., 1.]], device=task.device)
    target = torch.tensor([[2., 0., 1.]], device=task.device)
    p = velocity_progress(prev, cur, target, task.dt)
    torch.testing.assert_close(compute_walk_reward(cur, prev, target, task.dt, 1.5, True), .2 * p)
    near = torch.exp(-10 * (cur - target).square().sum(-1))
    old_carry = compute_carry_reward(cur, prev, target, task.dt, 1.5,
                                    torch.full_like(cur, .4), True, False, 1., 2.5)
    torch.testing.assert_close(old_carry, .2 * p + .2 * near)
    bodies = torch.zeros(1, 3, 3, device=task.device)
    bodies[:, 1:3] = cur[:, None]
    bodies[:, 0] = cur
    assert compute_handheld_reward(bodies, cur, torch.tensor([1, 2], device=task.device), False).item() > .199
    print('PASS relation simulator: M={} O={}, 64 transitions, reset subset, stored suffix, '
          'reward/history agreement, clipping bypass, no success termination, legacy scalar shadow; '
          'mode={} negative_progress_samples={}'.format(
              M, task.num_objects, progress_mode, int(negative_progress_count)), flush=True)


if __name__ == '__main__':
    MAPlayerContinuous.run = check_simulator
    entry.main()
