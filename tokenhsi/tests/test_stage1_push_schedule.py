"""Fixed-asset task marginals and checkpoint epoch semantics for PUSH schedules."""
from copy import deepcopy
import pytest
import torch

from utils.experiments import native_config, validate_experiment
from utils.stage1_jointpush import sample_assets, sample_templates, contract
from utils.stage1_push_schedule import (TaskSamplingSchedule, schedule_state,
                                       check_schedule_checkpoint, fit_task_weights)

NAME = 'stage1_push_schedule'


@pytest.fixture(scope='module')
def assets():
    env = native_config(NAME)['env']
    torch.manual_seed(42)
    sizes, base = sample_assets(2048, env, 'cpu')
    return env, sizes, base


@pytest.mark.parametrize('epoch,expected', [
    (0, [0., .10, .30, .20, .15, .25]),
    (29999, [0., .10, .30, .20, .15, .25]),
    (30000, [0., .10, .30, .20, .15, .25]),
    (35000, [0., .075, .275, .225, .175, .25]),
    (40000, [0., .05, .25, .25, .20, .25]),
    (40001, [0., .05, .25, .25, .20, .25]),
    (2000000, [0., .05, .25, .25, .20, .25]),
])
def test_fixed_asset_population_matches_schedule_and_stays_eligible(assets, epoch, expected):
    env, sizes, base = assets
    unchanged = sizes.clone()
    rng = torch.random.get_rng_state()
    schedule = TaskSamplingSchedule(env, base)
    weights = schedule.update(epoch)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(sizes, unchanged)
    assert torch.equal(weights[..., 5], base[..., 5])
    assert (weights[base == 0] == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2048, 2))
    torch.testing.assert_close(weights.mean((0, 1)), torch.tensor(expected), atol=1e-6, rtol=0)
    draws = torch.stack([sample_templates(weights, 4) for _ in range(64)])
    actual = draws.flatten().bincount(minlength=6)/draws.numel()
    torch.testing.assert_close(actual, torch.tensor(expected), atol=.004, rtol=0)


def test_final_weights_need_population_balance_not_only_prior_reweighting(assets):
    env, _, base = assets
    initial = torch.tensor(schedule_state(env, 0)['probabilities'])
    final = torch.tensor(schedule_state(env, 40000)['probabilities'])
    naive = base*(final/initial.clamp_min(1e-9))
    naive /= naive.sum(-1, keepdim=True)
    assert (naive.mean((0, 1))-final).abs().max() > .01
    weights = TaskSamplingSchedule(env, base).update(40000)
    torch.testing.assert_close(weights.mean((0, 1)), final, atol=1e-6, rtol=0)


def test_resume_uses_saved_global_epoch_and_rejects_changed_schedule(assets):
    env, _, base = assets
    checkpoint = dict(epoch=35000, stage1_push_sampling_state=schedule_state(env, 35000))
    check_schedule_checkpoint(checkpoint, env)
    restored = TaskSamplingSchedule(env, base).update(checkpoint['epoch'])
    uninterrupted = TaskSamplingSchedule(env, base)
    uninterrupted.update(34999)
    assert torch.equal(restored, uninterrupted.update(35000))
    changed = deepcopy(env)
    changed['samplingSchedule']['end_epoch'] = 50000
    with pytest.raises(ValueError):
        check_schedule_checkpoint(checkpoint, changed)
    assert contract(changed) != contract(env)
    with pytest.raises(ValueError):
        check_schedule_checkpoint(dict(epoch=35000), env)
    with pytest.raises(ValueError):
        check_schedule_checkpoint(checkpoint, native_config('stage1_push')['env'])


def test_scheduled_and_legacy_experiments_cannot_be_cross_loaded():
    new = native_config(NAME)['env']
    old = native_config('stage1_push')['env']
    validate_experiment(NAME, new, native_config(NAME, training=True))
    assert native_config(NAME, training=True)['params']['config']['max_epochs'] == -1
    assert contract(old)['revision'] == 1 and 'sampling_schedule' not in contract(old)
    assert contract(new)['revision'] == 2
    for name, env in [(NAME, old), ('stage1_push', new)]:
        with pytest.raises(ValueError):
            validate_experiment(name, env, native_config(name, training=True))


def test_invalid_moving_push_slots_are_rejected():
    env = native_config(NAME)['env']
    env['samplingSchedule']['final_template_probabilities']['PUSH_AT'] = .30
    env['samplingSchedule']['final_template_probabilities']['HOLDING_AT'] = .20
    with pytest.raises(ValueError, match='slots'):
        validate_experiment(NAME, env, native_config(NAME, training=True))


def test_tiny_viewer_falls_back_without_assigning_incompatible_tasks():
    base = torch.tensor([[[0., 0., 1., 0., 0., 0.]]])
    target = [0., .05, .25, .25, .20, .25]
    with pytest.raises(ValueError):
        fit_task_weights(base, target)
    assert torch.equal(fit_task_weights(base, target, strict=False), base)


def test_one_person_random_viewer_can_still_draw_push_or_other_assets():
    env = native_config(NAME)['env']
    env.update(numAgents=1, numObjects=1)
    push_flags = []
    for seed in range(32):
        torch.manual_seed(seed)
        _, weights = sample_assets(1, env, 'cpu')
        push_flags.append(bool(weights[0, 0, 5]))
    assert any(push_flags) and not all(push_flags)


@pytest.mark.parametrize('limit,start', [(-1, 40000), (1, 0)])
def test_training_loop_has_no_epoch_exit_when_unlimited_but_checks_remain_bounded(limit, start):
    import ast
    import os
    import time
    from pathlib import Path
    from types import SimpleNamespace
    # Run the actual loop with fake rollouts, without loading Isaac Gym on CPU.
    path = Path('tokenhsi/learning/common_agent.py')
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'CommonAgent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    namespace = dict(time=time, os=os)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    calls, saves = [], []
    class ProbeComplete(Exception):
        pass
    agent = SimpleNamespace(epoch_num=start, init_tensors=lambda: None,
        _update_training_curriculum=lambda: None, env_reset=lambda: None,
        _after_training_epoch=lambda: False,
        batch_size_envs=1, nn_dir='output_etc/check/stage1_push_schedule',
        config={'name': 'limit_probe'}, multi_gpu=False, _init_train=lambda: None,
        rank=0, frame=0, print_stats=False, writer=SimpleNamespace(add_scalar=lambda *a: None),
        _log_train_info=lambda *a: None,
        algo_observer=SimpleNamespace(after_print_stats=lambda *a: None),
        game_rewards=SimpleNamespace(current_size=0), save_freq=0, max_epochs=limit,
        save=saves.append, last_mean_rewards=0)
    def update():
        agent.epoch_num += 1
        return agent.epoch_num
    def rollout():
        calls.append(agent.epoch_num)
        if len(calls) == 3:
            raise ProbeComplete()
        return dict(total_time=.1, play_time=.1)
    agent.update_epoch, agent.train_epoch = update, rollout
    if limit < 0:
        with pytest.raises(ProbeComplete):
            namespace['train'](agent)
        assert calls == [40001, 40002, 40003] and not saves
    else:
        namespace['train'](agent)
        assert calls == [1, 2] and len(saves) == 1


@pytest.mark.parametrize('epoch', [5000, 35000, 50000])
def test_manual_player_restore_applies_checkpoint_epoch_before_first_reset(epoch):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    path = Path('tokenhsi/learning/multi_agent/ma_players.py')
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'MAPlayerContinuous')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'restore']
    applied, loaded = [], []
    class Parent:
        def restore(self, filename):
            loaded.append(filename)
    env = native_config(NAME)['env']
    checkpoint = dict(epoch=epoch, stage1_push_sampling_state=schedule_state(env, epoch))
    namespace = dict(amp_players=SimpleNamespace(AMPPlayerContinuous=Parent),
        load_checkpoint=lambda *a: checkpoint, check_checkpoint_metadata=lambda *a: None,
        checkpoint_metadata=lambda *a: {}, check_agent_collision_checkpoint=lambda *a: None)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    player_cls = namespace['MAPlayerContinuous']
    player = player_cls.__new__(player_cls)
    player.model = SimpleNamespace(a2c_network=SimpleNamespace(actor_encoder=SimpleNamespace(use_relation_bias=True)))
    player.env = SimpleNamespace(task=SimpleNamespace(cfg={'env': env}, _relation_cfg={},
                                                     set_sampling_epoch=applied.append))
    player.device = 'cpu'
    player.restore('saved.pth')
    assert applied == [epoch] and loaded == ['saved.pth']
