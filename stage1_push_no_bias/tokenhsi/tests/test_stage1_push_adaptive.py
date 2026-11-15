"""Adaptive block transitions, physical task budgets and checkpoint hand-offs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.experiments import native_config, validate_experiment, load_environment
from utils.stage1_jointpush import sample_assets, contract, sample_templates
from utils.stage1_push_adaptive import (TASKS, STATE_KEY, initial_state, advance,
    validate_state, focus_probabilities, template_probabilities, check_checkpoint)

NAME = 'stage1_push_adaptive'


@pytest.fixture
def env():
    return native_config(NAME)['env']


def evaluate_until(state, spec, stop, rates):
    for epoch in range(state['last_eval_epoch']+spec['eval_interval'], stop+1, spec['eval_interval']):
        state = advance(state, spec, epoch, rates)
    return state


def test_uniform_then_focus_is_held_until_maximum_without_breakthrough(env):
    spec = env['adaptiveSampling']
    state = initial_state(spec)
    assert state['probabilities'] == dict.fromkeys(TASKS, .2)
    rates = dict.fromkeys(TASKS, .1)
    state = advance(state, spec, 2000, rates)
    assert state['focus'] == 'CLIMB'
    assert state['probabilities'] == pytest.approx(dict(AT=.125, ON_TOP=.125, SIT=.125, CLIMB=.5, PUSH=.125))
    p = deepcopy(state['probabilities'])
    state = evaluate_until(state, spec, 12000, dict(rates, PUSH=0.))
    assert state['focus'] == 'CLIMB' and state['probabilities'] == p
    state = advance(state, spec, 14000, dict(rates, PUSH=0.))
    assert state['focus'] == 'PUSH' and state['focus_since'] == 14000


def test_confirmed_breakthrough_waits_for_minimum_focus_and_does_not_oscillate(env):
    spec = env['adaptiveSampling']
    rates = dict.fromkeys(TASKS, .1)
    state = advance(initial_state(spec), spec, 2000, rates)
    state = advance(state, spec, 4000, dict(rates, CLIMB=.9))
    assert not state['mastered']['CLIMB']
    state = advance(state, spec, 6000, dict(rates, CLIMB=.9))
    assert state['mastered']['CLIMB'] and state['focus'] == 'CLIMB'
    state = advance(state, spec, 8000, dict(rates, CLIMB=.85))
    assert state['focus'] == 'PUSH'
    assert state['probabilities']['CLIMB'] == .05
    assert state['probabilities']['PUSH'] == .5
    for t in ('AT', 'ON_TOP', 'SIT'):
        assert state['probabilities'][t] == pytest.approx(.15)


def test_all_mastered_keeps_training_uniform_then_confirmed_forgetting_reopens_focus(env):
    spec = env['adaptiveSampling']
    state = evaluate_until(initial_state(spec), spec, 4000, dict.fromkeys(TASKS, .95))
    assert state['focus'] is None and all(state['mastered'].values())
    assert state['probabilities'] == dict.fromkeys(TASKS, .2)
    # One dip never interrupts maintenance; the threshold is strictly <80%.
    state = advance(state, spec, 6000, dict.fromkeys(TASKS, .8))
    assert all(state['mastered'].values())
    state = advance(state, spec, 8000, dict.fromkeys(TASKS, .79))
    assert state['focus'] is None
    state = advance(state, spec, 10000, dict.fromkeys(TASKS, .79))
    assert state['focus'] == 'PUSH' and all(state['forgotten'].values())
    assert not any(state['mastered'].values())


def test_forgetting_does_not_interrupt_existing_block(env):
    spec = env['adaptiveSampling']
    rates = dict(AT=.95, ON_TOP=.2, SIT=.2, CLIMB=.1, PUSH=.2)
    state = evaluate_until(initial_state(spec), spec, 4000, rates)
    assert state['mastered']['AT']
    p = deepcopy(state['probabilities'])
    state = evaluate_until(state, spec, 8000, dict(rates, AT=.1))
    assert state['forgotten']['AT'] and state['focus'] == 'CLIMB'
    assert state['probabilities'] == p
    state = evaluate_until(state, spec, 14000, dict(rates, AT=.1))
    assert state['focus'] == 'AT'


@pytest.mark.parametrize('mask', range(32))
def test_every_mastery_combination_has_exact_total_and_floor(env, mask):
    spec = env['adaptiveSampling']
    mastered = {t: bool(mask & (1 << i)) for i, t in enumerate(TASKS)}
    for focus in TASKS:
        p = focus_probabilities(spec, focus, mastered)
        assert sum(p.values()) == pytest.approx(1.)
        assert min(p.values()) >= .05 and p[focus] == .5


@pytest.mark.parametrize('focus', [None, *TASKS])
def test_actual_4096_slots_preserve_frame_floor_and_task_size_ranges(env, focus):
    spec = env['adaptiveSampling']
    state = initial_state(spec)
    if focus:
        state['probabilities'] = focus_probabilities(spec, focus, dict.fromkeys(TASKS, True))
    env['_adaptiveSamplingState'] = state
    torch.manual_seed(42)
    sizes, weights = sample_assets(2048, env, 'cpu')
    templates = weights.argmax(-1)
    # Each slot remains with its physical task through repeated episode resets.
    assert torch.equal(sample_templates(weights, 4), templates)
    actual = weights.mean((0, 1))
    desired = template_probabilities(state)
    names = ('HOLDING', 'SIT', 'CLIMB', 'HOLDING_AT', 'HOLDING_ON_TOP', 'PUSH_AT')
    for index, task in enumerate(names):
        assert abs(float(actual[index])-desired[task]) <= 1/4096
        if index:
            assert actual[index] >= .05
    assert actual[0] == 0
    from utils.size_rsi import STAGE1_RANGES
    profiles = {tuple(v) for v in env['jointPush']['box']['profiles']}
    for task, name in enumerate(names):
        selected = sizes[:, :2][templates == task]
        if not len(selected):
            continue
        if task == 5:
            assert all(any(torch.allclose(v, torch.tensor(p)) for p in profiles) for v in selected.unique(dim=0))
        else:
            bounds = torch.tensor(STAGE1_RANGES[name])
            assert ((selected >= bounds[:, 0]-1e-6) & (selected <= bounds[:, 1]+1e-6)).all()


def test_resume_restores_decision_before_assets_and_rejects_other_experiments(env, tmp_path):
    state = advance(initial_state(env['adaptiveSampling']), env['adaptiveSampling'], 2000, dict.fromkeys(TASKS, .1))
    checkpoint = tmp_path/'model.pth'
    torch.save(dict(epoch=2000, **{STATE_KEY: state}, relation_experiment_config=native_config(NAME)), checkpoint)
    cfg, _ = load_environment(NAME, '', str(checkpoint), saved=True)
    assert cfg['env']['_adaptiveSamplingState'] == state
    weights = torch.load(checkpoint, weights_only=False)
    check_checkpoint(weights, cfg['env'])
    wrong_criterion = dict(weights, stage1_push_adaptive_eval_criterion='point')
    with pytest.raises(ValueError, match='criterion'):
        check_checkpoint(wrong_criterion, cfg['env'])
    with pytest.raises(ValueError, match='built'):
        check_checkpoint(weights, env)
    for other in ('stage1_push', 'stage1_push_schedule'):
        with pytest.raises(ValueError):
            validate_experiment(other, env, native_config(other, True))
        with pytest.raises(ValueError):
            check_checkpoint(weights, native_config(other)['env'])
    assert contract(env)['revision'] == 3
    changed = deepcopy(env)
    changed['adaptiveSampling']['mastery_threshold'] = .95
    assert contract(changed) != contract(env)


def test_single_player_validates_saved_contract_before_viewer_spacing_changes(env, tmp_path):
    import ast
    weights = dict(epoch=0, **{STATE_KEY: initial_state(env['adaptiveSampling'])},
                   stage1_jointpush_contract=contract(env), relation_experiment_config=native_config(NAME))
    checkpoint = tmp_path/'model.pth'
    torch.save(weights, checkpoint)
    cfg, _ = load_environment(NAME, '', str(checkpoint), saved=True)
    cfg['env'].update(numAgents=1, numObjects=1, envSpacing=5.5, episodeLength=32)
    assert cfg['_adaptiveTrainingContract'] == weights['stage1_jointpush_contract']
    assert contract(cfg['env']) != weights['stage1_jointpush_contract']
    path = Path('tokenhsi/learning/multi_agent/ma_players.py')
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'restore']
    class Parent:
        def restore(self, fn):
            return fn
    namespace = dict(amp_players=SimpleNamespace(AMPPlayerContinuous=Parent),
        load_checkpoint=lambda *a: weights, check_checkpoint_metadata=lambda *a: None,
        checkpoint_metadata=lambda *a: {}, check_agent_collision_checkpoint=lambda *a: None)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    player_cls = namespace['MAPlayerContinuous']
    player = player_cls.__new__(player_cls)
    player.model = SimpleNamespace(a2c_network=SimpleNamespace(actor_encoder=SimpleNamespace(use_relation_bias=True)))
    player.env = SimpleNamespace(task=SimpleNamespace(cfg=cfg, _relation_cfg={}))
    player.device = 'cpu'
    assert player.restore(str(checkpoint)) == str(checkpoint)
    cfg['_adaptiveTrainingContract']['revision'] = 0
    with pytest.raises(ValueError, match='contract'):
        player.restore(str(checkpoint))


def test_new_experiment_preserves_rewards_physics_rsi_and_ppo(env):
    original = native_config('stage1_push')['env']
    restored = deepcopy(env)
    del restored['adaptiveSampling']
    restored['relationGraph']['template_probabilities'] = original['relationGraph']['template_probabilities']
    assert restored == original
    new = native_config(NAME, True)['params']
    old = native_config('stage1_push', True)['params']
    assert new['load_checkpoint'] is False
    assert new['config']['max_epochs'] == -1
    new['config']['max_epochs'] = old['config']['max_epochs']
    new['config']['name'] = old['config']['name']
    assert new == old


@pytest.mark.parametrize('bad', [float('nan'), -.01, 1.01])
def test_invalid_eval_cannot_modify_state(env, bad):
    spec = env['adaptiveSampling']
    state = initial_state(spec)
    before = deepcopy(state)
    with pytest.raises(ValueError):
        advance(state, spec, 2000, dict.fromkeys(TASKS, bad))
    assert state == before


def test_replay_roundtrip_preserves_samples_across_rebuild():
    from learning.replay_buffer import ReplayBuffer
    from learning.multi_agent.adaptive_sampling import replay_state, restore_replay
    original = ReplayBuffer(20, 'cpu')
    original.store({'amp_obs': torch.arange(42).reshape(14, 3).float()})
    original.sample(5)
    state = replay_state(original)
    restored = ReplayBuffer(20, 'cpu')
    restore_replay(restored, state)
    assert restored._head == original._head and restored._total_count == original._total_count
    assert torch.equal(original.sample(10)['amp_obs'], restored.sample(10)['amp_obs'])


def test_random_evaluations_never_change_focus_early_or_lose_the_floor(env):
    spec = env['adaptiveSampling']
    state = initial_state(spec)
    rng = torch.Generator().manual_seed(142)
    for epoch in range(2000, 500001, 2000):
        previous = deepcopy(state)
        rates = dict(zip(TASKS, torch.rand(5, generator=rng).tolist()))
        state = advance(state, spec, epoch, rates)
        assert sum(state['probabilities'].values()) == pytest.approx(1.)
        assert min(state['probabilities'].values()) >= .05
        if previous['focus'] and state['focus'] != previous['focus'] and state['focus'] is not None:
            assert epoch-previous['focus_since'] >= spec['minimum_focus_epochs']


def test_adaptive_viewer_with_no_spare_supports_resamples_ontop(env):
    env.update(numAgents=4, numObjects=4, _adaptiveEvaluation=True)
    for seed in range(8):
        torch.manual_seed(seed)
        _, weights = sample_assets(4, env, 'cpu')
        assert not (sample_templates(weights, 4) == 4).any()


def test_supervisor_relaunches_only_a_verified_successful_handoff(monkeypatch, tmp_path):
    import adaptive_train as supervisor
    import hashlib
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor.sys, 'argv', ['adaptive_train.py', '2', '2048', '4'])
    monkeypatch.setenv('OUTPUT_PATH', 'output_etc/check/adaptive_supervisor')
    monkeypatch.delenv('RESUME_CHECKPOINT', raising=False)
    calls = []
    def child(command, cwd, env):
        calls.append(dict(env))
        assert command[-3:] == ['2', '2048', '4']
        if len(calls) == 1:
            checkpoint = tmp_path/env['OUTPUT_PATH']/env['TOKENHSI_ADAPTIVE_RUN_NAME']/'nn/model.pth'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'verified handoff')
            marker = Path(env['TOKENHSI_ADAPTIVE_RESTART_FILE'])
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps(dict(checkpoint=str(checkpoint), epoch=2000,
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
        else:
            assert env['RESUME_CHECKPOINT'].endswith('/nn/model.pth')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(supervisor.subprocess, 'run', child)
    supervisor.main()
    assert len(calls) == 2
    assert calls[0]['TOKENHSI_ADAPTIVE_RUN_NAME'] == calls[1]['TOKENHSI_ADAPTIVE_RUN_NAME']


def test_supervisor_does_not_restart_failed_training(monkeypatch, tmp_path):
    import adaptive_train as supervisor
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    monkeypatch.setenv('OUTPUT_PATH', 'output_etc/check/adaptive_failure')
    monkeypatch.delenv('RESUME_CHECKPOINT', raising=False)
    calls = []
    def child(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(supervisor.subprocess, 'run', child)
    with pytest.raises(SystemExit) as error:
        supervisor.main()
    assert error.value.code == 7 and len(calls) == 1


def test_training_hook_saves_transition_and_requests_only_its_own_rebuild(env, monkeypatch, tmp_path):
    from learning.multi_agent import adaptive_sampling as runtime
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'evaluate', lambda *a: dict.fromkeys(TASKS, .1))
    marker = tmp_path/'restart.json'
    monkeypatch.setenv('TOKENHSI_ADAPTIVE_RESTART_FILE', str(marker))
    task = SimpleNamespace(cfg=dict(env=env, args=SimpleNamespace(checkpoint='Base')))
    logs, saves = [], []
    agent = SimpleNamespace(vec_env=SimpleNamespace(env=SimpleNamespace(task=task)),
        _adaptive_sampling_state=initial_state(env['adaptiveSampling']), epoch_num=1999,
        frame=123, experiment_dir=str(tmp_path/'run'), nn_dir=str(tmp_path/'run/nn'),
        config=dict(name='Stage1PushAdaptive'), writer=SimpleNamespace(add_scalar=lambda *a: logs.append(a), flush=lambda: None))
    def save(path):
        saves.append((path, deepcopy(agent._adaptive_sampling_state), getattr(agent, '_adaptive_save_replay', False)))
        target = Path(path+'.pth')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'fake_checkpoint')
    agent.save = save
    assert not runtime.after_epoch(agent) and not saves
    agent.epoch_num = 2000
    assert runtime.after_epoch(agent)
    assert len(saves) == 2 and not saves[0][2] and saves[1][2]
    assert saves[0][1]['focus'] is None and saves[1][1]['focus'] == 'CLIMB'
    assert json.loads(marker.read_text())['epoch'] == 2000
    decision = next((tmp_path/'output_etc/check/stage1_push_adaptive/run').glob('epoch_00002000/attempt_*/decision.json'))
    assert json.loads(decision.read_text())['criterion'] == 'region'
    assert any(tag == 'adaptive/eval_region_climb' for tag, *_ in logs)
    assert not any(tag.startswith('adaptive/eval_point_') for tag, *_ in logs)
    marker.unlink()
    agent.epoch_num = 4000
    assert not runtime.after_epoch(agent) and not marker.exists()
    assert not saves[-1][2]


def test_auto_evaluation_checks_complete_results_and_source_hash(env, monkeypatch, tmp_path):
    from learning.multi_agent import adaptive_sampling as runtime
    import hashlib
    checkpoint = tmp_path/'snapshot.pth'
    checkpoint.write_bytes(b'checkpoint')
    directory = tmp_path/'evaluation'
    spec = env['adaptiveSampling']
    def child(command, **kwargs):
        assert '--profile' in command and command[command.index('--profile')+1] == 'stage1_single'
        assert command[command.index('--envs')+1] == '512'
        assert command[command.index('--seeds')+1] == '0'
        (directory/'summary.json').write_text(json.dumps(dict(runs=5, scores=dict(
            point={t: dict(trials=512, successes=51, mean=51/512) for t in TASKS},
            region={t: dict(trials=512, successes=461, mean=461/512) for t in TASKS}))))
        (directory/'manifest.json').write_text(json.dumps(dict(horizon=900, envs=512, seeds=[0],
            profile='stage1_single', tasks=list(TASKS),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
    monkeypatch.setattr(runtime.subprocess, 'run', child)
    assert runtime.evaluate(checkpoint, directory, spec) == dict.fromkeys(TASKS, 461/512)
    # Missing tasks/partial results must not count as a successful evaluation.
    def incomplete(*args, **kwargs):
        child(*args, **kwargs)
        summary = json.loads((directory/'summary.json').read_text())
        summary['runs'] = 4
        (directory/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(runtime.subprocess, 'run', incomplete)
    with pytest.raises(ValueError, match='Incomplete'):
        runtime.evaluate(checkpoint, directory, spec)


def test_region_feedback_controls_mastery_instead_of_point(env, monkeypatch, tmp_path):
    from learning.multi_agent import adaptive_sampling as runtime
    checkpoint = tmp_path/'model.pth'
    checkpoint.write_bytes(b'checkpoint')
    directory = tmp_path/'evaluation'
    spec = env['adaptiveSampling']
    successes = dict(AT=51, ON_TOP=461, SIT=461, CLIMB=461, PUSH=51)
    def child(command, **kwargs):
        import hashlib
        (directory/'summary.json').write_text(json.dumps(dict(runs=5, scores={
            'point': {t: dict(trials=512, successes=51) for t in TASKS},
            'region': {t: dict(trials=512, successes=successes[t]) for t in TASKS}})))
        (directory/'manifest.json').write_text(json.dumps(dict(horizon=900, envs=512, seeds=[0],
            profile='stage1_single', tasks=list(TASKS),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
    monkeypatch.setattr(runtime.subprocess, 'run', child)
    rates = runtime.evaluate(checkpoint, directory, spec)
    state = advance(initial_state(spec), spec, 2000, rates)
    state = advance(state, spec, 4000, rates)
    assert {t for t in TASKS if state['mastered'][t]} == {'ON_TOP', 'SIT', 'CLIMB'}
