"""Live 2048-env PPO/rebuild check; synthetic eval scores never measure policy.

Run with the adaptive native YAML and output_etc/check. The transition probe
advances the epoch counter after two real updates, so no 2000-epoch job is needed.
Resume the emitted handoff using the normal managed train wrapper afterwards.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # Must precede torch.
import torch
import run as entry
from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent import adaptive_sampling as runtime
from utils.stage1_push_adaptive import TASKS, state_for_env, template_probabilities
from utils.unified_training import amp_family_ids

original_curriculum = MAAgent._update_training_curriculum
original_after = MAAgent._after_training_epoch


@torch.no_grad()
def check_assets(agent):
    original_curriculum(agent)
    if getattr(agent, '_adaptive_assets_checked', False):
        return
    agent._adaptive_assets_checked = True
    task = agent.vec_env.env.task
    assert (task.num_envs, task.num_agents, task.num_objects) == (2048, 2, 4)
    expected = template_probabilities(state_for_env(task.cfg['env']))
    names = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP', 'PUSH_AT')
    actual = task._joint_probabilities.mean((0, 1))
    for index, name in enumerate(names):
        assert abs(float(actual[index])-expected[name]) <= 1/4096
        if index:
            assert actual[index] >= .05
    ids = torch.arange(task.num_envs, device=task.device)
    obs = agent.env_reset(ids)
    assert torch.isfinite(obs['obs']).all() and torch.isfinite(task._root_states).all()
    assert torch.equal(task._templates(ids), task._joint_probabilities.argmax(-1))
    assert torch.equal(task._push_assets, task._templates(ids) == 5)
    demo = task.fetch_amp_obs_demo(12000)
    assert torch.isfinite(demo).all()
    family = amp_family_ids(demo, task._num_amp_obs_steps, 4)
    amp_actual = family.bincount(minlength=4)/float(len(family))
    amp_expected = actual[[3, 1, 2, 5]].clone()
    amp_expected[0] += actual[4]
    torch.testing.assert_close(amp_actual, amp_expected, atol=.02, rtol=0)
    Path(task.cfg['args'].output_path, 'adaptive_assets_validation.json').write_text(json.dumps(dict(
        epoch=agent.epoch_num, actual=actual.cpu().tolist(), desired=expected,
        actual_amp=amp_actual.cpu().tolist(), expected_amp=amp_expected.cpu().tolist(),
        reset_finite=True, physical_push_slots_match=True), indent=2)+'\n')
    print('PASS adaptive physical slots, reset and AMP families', flush=True)


def synthetic_evaluation(checkpoint, directory, spec):
    print('CHECK ONLY: synthetic success rates, no policy evaluation', flush=True)
    return dict.fromkeys(TASKS, .0)


def transition_after_two_updates(agent):
    if agent.epoch_num == 2:
        agent.epoch_num = 2000
        return original_after(agent)
    return original_after(agent)


if __name__ == '__main__':
    MAAgent._update_training_curriculum = check_assets
    if '--resume' not in sys.argv:
        runtime.evaluate = synthetic_evaluation
        MAAgent._after_training_epoch = transition_after_two_updates
    entry.main()
