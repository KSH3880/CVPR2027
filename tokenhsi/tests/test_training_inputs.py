from copy import deepcopy

import pytest
import torch

from sa_helpers import configs
from utils.joint_carry_amp import sample_amp_sources, validate_training, check_training_checkpoint
from utils.joint_carry_rsi import load_paired_snapshots
from utils.edge_context_spec import task_instance, check_task_resume


def test_full_pickup_rsi_and_terminal_targets():
    env, _ = configs()
    spec = env['jointCarryRsi']
    pools = load_paired_snapshots(spec['directory'], 'cpu', terminal_targets=True,
                                 expected_digest=spec['dataset_sha256'])
    assert {name: len(pool['root_state']) for name, pool in pools.items()} == {
        'pickUp': 100, 'carryWith': 129, 'putDown': 135}
    assert 'terminal_box_xy' in pools['putDown']
    for pool in pools.values():
        assert all(torch.isfinite(value).all() for value in pool.values())


def test_locomotion_amp_source_distribution():
    torch.manual_seed(53)
    sources = sample_amp_sources(torch.zeros(100000, dtype=torch.long))
    fractions = torch.bincount(sources, minlength=3) / len(sources)
    torch.testing.assert_close(fractions, torch.tensor([.8, .1, .1]), atol=.005, rtol=0)
    assert sample_amp_sources(torch.ones(100, dtype=torch.long)).count_nonzero() == 0


def test_regular_training_contract_accepts_existing_sa_checkpoints():
    env, train = configs()
    weights = dict(joint_carry_loco_platform=env['jointCarryRsi']['locoPlatform'],
        joint_carry_rsi_dataset=env['jointCarryRsi']['dataset_sha256'],
        joint_carry_goal_sampling=env['jointCarryRsi']['goalSampling'],
        joint_carry_training=env['jointCarryTraining'])
    check_training_checkpoint(weights, env)
    legacy = deepcopy(weights)
    legacy['joint_carry_ablation'] = legacy.pop('joint_carry_training')
    check_training_checkpoint(legacy, env)
    bad = deepcopy(env)
    bad['jointCarryTraining']['amp']['probabilities']['original'] = .7
    with pytest.raises(ValueError, match='80/10/10'):
        validate_training(bad, train)


def test_resume_requires_the_same_graph_and_entity_counts():
    env, _ = configs()
    weights = {'relation_task_instance': task_instance(env['relationGraph'], 2, 4)}
    check_task_resume(weights, env['relationGraph'], 2, 4)
    with pytest.raises(ValueError, match='identical task graph'):
        check_task_resume(weights, env['relationGraph'], 2, 5)
