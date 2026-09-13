"""The success10 experiment changes only the one-time success bonus."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.relation_reward import RelationRuntime
from utils.relation_task_spec import (
    compile_carry_subgoal, validate_relation_config,
    checkpoint_metadata, check_checkpoint_metadata)


def test_success10_config_and_one_time_bonus():
    directory = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    stem = 'amp_humanoid_ma_carry_relation_state02_near_dir'
    base = yaml.safe_load((directory / (stem + '.yaml')).read_text())
    variant = yaml.safe_load((directory / (stem + '_success10.yaml')).read_text())
    expected = copy.deepcopy(base)
    expected['env']['relationReward']['subgoal_success_bonus'] = 10.
    assert variant == expected
    cfg = variant['env']['relationReward']
    validate_relation_config(cfg)
    runtime = RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    runtime.reset(torch.tensor([0]), torch.tensor([[1., 0.]]))
    first = runtime.step(torch.ones(1, 2), torch.ones(1, 2))
    second = runtime.step(torch.ones(1, 2), torch.ones(1, 2))
    assert first['success_bonus'].item() == 10.
    assert second['success_bonus'].item() == 0.
    assert second['edge_reward'].sum().item() > 0.
    metadata = checkpoint_metadata(cfg)
    check_checkpoint_metadata({'relation_metadata': metadata}, metadata)
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(base['env']['relationReward'])}, metadata)
