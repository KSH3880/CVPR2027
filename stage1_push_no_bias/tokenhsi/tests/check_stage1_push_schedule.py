"""Exercise curriculum boundaries/resume on live training scenes, then a short PPO check."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # Must precede torch.
import torch
import run as entry
from learning.multi_agent.ma_agent import MAAgent
from utils.stage1_push_schedule import schedule_state, check_schedule_checkpoint
from utils.unified_training import amp_family_ids

original_hook = MAAgent._update_training_curriculum


@torch.no_grad()
def check_boundaries(agent):
    original_hook(agent)
    if getattr(agent, '_schedule_boundary_check_done', False):
        return
    agent._schedule_boundary_check_done = True
    task = agent.vec_env.env.task
    assert task.num_envs == 2048 and task.num_agents == 2 and task.num_objects == 4
    ids = torch.arange(task.num_envs, device=task.device)
    sizes, push = task._box_size.clone(), task._push_assets.clone()
    original_epoch = agent.epoch_num
    records = []
    for epoch in (0, 30000, 35000, 40000, 1000000):
        agent.epoch_num = epoch
        original_hook(agent)
        state = schedule_state(task.cfg['env'], epoch)
        expected = torch.tensor(state['probabilities'], device=task.device)
        torch.testing.assert_close(task._joint_probabilities.mean((0, 1)), expected, atol=1e-6, rtol=0)
        obs = agent.env_reset(ids)
        assert torch.isfinite(obs['obs']).all() and torch.isfinite(task._root_states).all()
        assert torch.equal(sizes, task._box_size) and torch.equal(push, task._push_assets)
        templates = task._templates(ids)
        assert torch.equal(templates == 5, push)
        bounds = task._joint_probabilities.gather(-1, templates[..., None]).squeeze(-1)
        assert (bounds > 0).all()
        demo = task.fetch_amp_obs_demo(12000)
        assert torch.isfinite(demo).all()
        family = amp_family_ids(demo, task._num_amp_obs_steps, 4)
        actual_amp = family.bincount(minlength=4)/float(len(family))
        expected_amp = expected[[3, 1, 2, 5]].clone()
        expected_amp[0] += expected[4]
        torch.testing.assert_close(actual_amp, expected_amp, atol=.02, rtol=0)
        weights = agent.get_full_state_weights()
        check_schedule_checkpoint(weights, task.cfg['env'])
        records.append(dict(epoch=epoch, expected=expected.cpu().tolist(),
                            actual_amp=actual_amp.cpu().tolist(), reset_finite=True,
                            sizes_and_push_unchanged=True))
    agent.epoch_num = original_epoch
    original_hook(agent)
    snapshot = agent.get_full_state_weights()
    restored = dict(snapshot, epoch=35000,
                    stage1_push_sampling_state=schedule_state(task.cfg['env'], 35000))
    agent.set_full_state_weights(restored)
    original_hook(agent)
    assert task._sampling_epoch == 35000
    assert task._task_sampling_schedule.state['fraction'] == .5
    agent.set_full_state_weights(snapshot)
    original_hook(agent)
    assert task._sampling_epoch == original_epoch
    out = Path(task.cfg['args'].output_path)
    (out/'schedule_validation.json').write_text(json.dumps(dict(
        phases=records, restored_global_epoch=True, restored_original_epoch=original_epoch,
        no_policy_evaluation=True), indent=2)+'\n')
    print('PASS live schedule boundaries and resume', flush=True)


if __name__ == '__main__':
    MAAgent._update_training_curriculum = check_boundaries
    entry.main()
