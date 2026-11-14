from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.before_total_finetune import initialize
from utils.experiments import BEFORE_PUSH_RELFUSION, BEFORE_TOTAL, BEFORE_TOTAL_FT21K
from utils.experiments import experiment_horizon, native_config, validate_experiment


def fixture():
    model = torch.nn.Linear(3, 2)
    agent = SimpleNamespace(model=model, optimizer=torch.optim.Adam(model.parameters()),
                            running_mean_std=torch.nn.BatchNorm1d(3),
                            _amp_input_mean_std=torch.nn.BatchNorm1d(4),
                            value_mean_std=torch.nn.BatchNorm1d(1),
                            normalize_value=True, epoch_num=0, frame=0)
    cfg = native_config(BEFORE_TOTAL_FT21K)
    source = native_config(BEFORE_PUSH_RELFUSION)
    weights = dict(model={k: torch.ones_like(v) * .7 for k, v in model.state_dict().items()},
                   running_mean_std=deepcopy(agent.running_mean_std.state_dict()),
                   amp_input_mean_std=deepcopy(agent._amp_input_mean_std.state_dict()),
                   reward_mean_std=deepcopy(agent.value_mean_std.state_dict()),
                   optimizer={'state': {0: {'step': torch.tensor(999.)}}}, epoch=21000,
                   frame=123456, before_contract=checkpoint_contract(source['env']),
                   relation_experiment_config=source)
    for key in ('running_mean_std', 'amp_input_mean_std', 'reward_mean_std'):
        weights[key]['running_mean'].fill_(3.)
        weights[key]['running_var'].fill_(5.)
    return agent, weights, cfg


def test_total_environment_and_training_settings_are_preserved():
    cfg, base = native_config(BEFORE_TOTAL_FT21K), native_config(BEFORE_TOTAL)
    assert cfg['env'] == base['env'] and cfg['sim'] == base['sim']
    assert cfg['env']['episodeLength'] == experiment_horizon(BEFORE_TOTAL_FT21K) == 900
    assert cfg['experiment']['transfer']['mode'] == 'stage2_weights'
    assert cfg['experiment']['transfer']['checkpoint'].endswith('_00021000.pth')
    train = native_config(BEFORE_TOTAL_FT21K, True)
    validate_experiment(BEFORE_TOTAL_FT21K, cfg['env'], deepcopy(train))
    assert not train['params']['load_checkpoint']
    train['params']['config']['name'] = native_config(BEFORE_TOTAL, True)['params']['config']['name']
    assert train == native_config(BEFORE_TOTAL, True)


def test_only_model_and_normalizers_are_loaded_without_mutating_source():
    agent, weights, cfg = fixture()
    original = deepcopy(weights)
    optimizer = deepcopy(agent.optimizer.state_dict())
    report = initialize(agent, weights, cfg)
    for key, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, weights['model'][key], rtol=0, atol=0)
    for module, key in ((agent.running_mean_std, 'running_mean_std'),
                        (agent._amp_input_mean_std, 'amp_input_mean_std'),
                        (agent.value_mean_std, 'reward_mean_std')):
        for name, value in module.state_dict().items():
            assert torch.equal(value, weights[key][name])
    assert agent.optimizer.state_dict() == optimizer
    assert agent.epoch_num == 0 and agent.frame == 0
    assert report['optimizer_state_entries'] == 0 and not report['optimizer_restored']
    assert report['source_epoch'] == 21000
    assert str(weights) == str(original)
    with pytest.raises(ValueError, match='contract differs'):
        check_checkpoint(weights, cfg['env'])


@pytest.mark.parametrize('change', ['epoch', 'source_contract', 'source_graph', 'target_name', 'target_horizon', 'target_contract'])
def test_incorrect_sources_and_targets_are_rejected_before_loading(change):
    agent, weights, cfg = fixture()
    original = deepcopy(agent.model.state_dict())
    if change == 'epoch':
        weights['epoch'] = 20000
    elif change == 'source_contract':
        weights['before_contract'] = checkpoint_contract(cfg['env'])
    elif change == 'source_graph':
        weights['relation_experiment_config']['env']['relationGraph']['human_relation_fusion'] = 'none'
    elif change == 'target_name':
        cfg['experiment']['name'] = BEFORE_TOTAL
    elif change == 'target_horizon':
        cfg['env']['episodeLength'] = 1200
    else:
        cfg['env']['before']['waiting_penalty']['radius_buffer'] = 2.
    with pytest.raises(ValueError):
        initialize(agent, weights, cfg)
    assert all(torch.equal(value, original[key]) for key, value in agent.model.state_dict().items())


def test_model_architecture_must_match_exactly():
    agent, weights, cfg = fixture()
    del weights['model']['bias']
    with pytest.raises(RuntimeError, match='Missing key'):
        initialize(agent, weights, cfg)
