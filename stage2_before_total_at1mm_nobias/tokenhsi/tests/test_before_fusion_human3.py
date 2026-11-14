import pytest
import torch

from test_before_wait import setup
from utils.before_fusion_resume import prepare
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import BEFORE_PUSH_RELFUSION as BASE, BEFORE_PUSH_RELFUSION_HUMAN3 as NAME
from utils.experiments import load_environment, native_config, validate_experiment
from utils.unified_training import validate_unified_env


def test_profile_only_changes_ontop_entry_and_names():
    cfg, base = native_config(NAME), native_config(BASE)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert cfg['env']['before']['waiting_penalty'] == dict(radius_buffer=3., coefficient=1., ontop_distance_source='human')
    del cfg['env']['before']['waiting_penalty']['ontop_distance_source']
    cfg['experiment']['name'] = BASE
    assert cfg == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = native_config(BASE, True)['params']['config']['name']
    assert train == native_config(BASE, True)


def test_continuation_preserves_training_state_and_uses_new_saved_environment(tmp_path):
    cfg = native_config(BASE)
    saved = dict(cfg, train=native_config(BASE, True))
    weights = dict(model={'weight': torch.arange(4.)}, optimizer={'state': {0: {'step': torch.tensor(123.)}}},
                   epoch=21000, frame=900000, running_mean_std={'mean': torch.arange(3.)},
                   before_contract=checkpoint_contract(cfg['env']), relation_experiment_config=saved)
    source = tmp_path / 'source.pth'
    torch.save(weights, source)
    original = source.read_bytes()
    prepared = prepare(source, tmp_path / 'prepared')
    result = torch.load(prepared, weights_only=False)
    for key in ('model', 'optimizer', 'running_mean_std'):
        assert str(result[key]) == str(weights[key])
    assert result['epoch'] == 21000 and result['frame'] == weights['frame']
    loaded, _ = load_environment(NAME, '', str(prepared), saved=True)
    validate_experiment(NAME, loaded['env'], native_config(NAME, True))
    assert loaded['env'] == native_config(NAME)['env']
    check_checkpoint(result, loaded['env'])
    with pytest.raises(ValueError, match='contract differs'):
        check_checkpoint(weights, loaded['env'])
    assert prepare(source, tmp_path / 'prepared') == prepared
    assert prepare(prepared, tmp_path / 'unused') == prepared
    assert source.read_bytes() == original


@pytest.mark.parametrize('epoch,profile', [(20000, BASE), (21000, BASE + '_v2')])
def test_wrong_initial_epoch_or_profile_is_rejected(tmp_path, epoch, profile):
    source = tmp_path / 'wrong.pth'
    torch.save(dict(epoch=epoch, before_contract=checkpoint_contract(native_config(profile)['env'])), source)
    with pytest.raises(ValueError):
        prepare(source, tmp_path / 'prepared')


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_three_meter_human_entry_xyz_penalty_and_ever_release(family):
    config = native_config(NAME)['env']['before']['waiting_penalty']
    state, values, graph, rows, edges = setup(family, config=config)
    roots, velocity, boxes, sizes, goals, raw = values
    center = goals[rows, state.goal, :2].clone()
    radius = sizes[rows, state.support, :2].norm(dim=-1) / 2 + 3.
    roots[rows, state.owners, :2] = center
    roots[rows, state.owners, 0] += radius + .01
    boxes[rows, (state.source - 4).clamp_min(0), :2] = center
    assert not state.penalty(*values).any()
    roots[rows, state.owners, 0] -= .02
    velocity.zero_()
    velocity[..., 2] = .5
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = .5
    torch.testing.assert_close(state.penalty(*values), expected)
    roots[rows, state.owners, :2] += 100
    torch.testing.assert_close(state.penalty(*values), expected)
    raw[rows, state.predecessor] = True
    assert not state.penalty(*values).any()
    raw.zero_()
    assert not state.penalty(*values).any()
