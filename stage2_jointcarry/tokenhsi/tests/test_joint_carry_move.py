from copy import deepcopy

import pytest
import torch
import yaml

from env.tasks.multi_agent.joint_carry_reward import shared_alignment_reward
from utils.edge_ontop_spec import permute_graph
from utils.joint_carry_amp import validate_ablation
from utils.joint_carry_rsi import transform_states
from utils.joint_carry_spec import validate_env
from utils.mixed_carry_spec import sample_graph
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from test_joint_carry_alignment import scene, ROOT
from test_joint_carry_amp_ablation import configs

ENV, TRAIN = configs('locoamp_ca4_move_task_embedding')
CONFIG = ENV['relationReward']['joint_carry']['alignment']


def test_move_config_changes_only_reward_and_name():
    base, train = configs('locoamp_ca4_task_embedding')
    restored = deepcopy(ENV)
    restored['relationReward']['joint_carry'].pop('alignment')
    assert restored == base
    restored = deepcopy(TRAIN)
    restored['params']['config']['name'] = train['params']['config']['name']
    assert restored == train
    validate_env(ENV); validate_ablation(ENV, TRAIN)
    validate_relation_config(ENV['relationReward'])
    for key, value in [('weight', .1), ('buffer', .5), ('direction', 'heading'), ('buffer', float('nan'))]:
        bad = deepcopy(ENV['relationReward']); bad['joint_carry']['alignment'][key] = value
        with pytest.raises(ValueError, match='alignment'):
            validate_relation_config(bad)
    align = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding.yaml').read_text())['env']
    for old in (base, align):
        for saved, target in ((old, ENV), (ENV, old)):
            with pytest.raises(ValueError, match='reward config differs'):
                check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(saved['relationReward'])},
                                          checkpoint_metadata(target['relationReward']))


@pytest.mark.parametrize('preset', ['joint_carry_at', 'joint_carry_ontop'])
@pytest.mark.parametrize('velocities,expected', [
    ([[1., 0.], [2., 0.]], .2),
    ([[.1, 0.], [3., 4.]], .12),
    ([[1., 0.], [-1., 0.]], 0.),
    ([[1., 0.], [0., 1.]], 0.),
    ([[1., 0.], [0., 0.]], 0.),
    ([[0., 0.], [0., 0.]], 0.),
])
def test_minimum_actual_motion_not_heading_or_speed(preset, velocities, expected):
    roots, objects, sizes, goals, graph = scene(preset)
    roots[0, :, 7:9] = torch.tensor(velocities)
    gate = torch.tensor([True])
    reward, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    torch.testing.assert_close(reward, torch.tensor([expected]))
    roots[..., 3:7] = torch.tensor([0., 0., 1., 0.])
    roots[0, 0, 7:9] *= .3; roots[0, 1, 7:9] *= 7.
    actual, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    torch.testing.assert_close(actual, reward)


@pytest.mark.parametrize('preset,radius', [('joint_carry_at', .8),
    ('joint_carry_ontop', (.52**2+.8**2)**.5/2+.8)])
def test_move_pinning_gate_and_zero_distance(preset, radius):
    roots, objects, sizes, goals, graph = scene(preset)
    for distance, expected in [(radius+.001, 0.), (radius, .2), (radius-.001, .2), (0., .2)]:
        objects[0, 2, 0] = distance; goals[0, 0, 0] = distance
        reward, diag = shared_alignment_reward(roots, objects, sizes, goals, graph, torch.tensor([True]), CONFIG)
        torch.testing.assert_close(reward, torch.tensor([expected]))
        torch.testing.assert_close(diag['radius'], torch.tensor([radius]))
        reward, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, torch.tensor([False]), CONFIG)
        assert reward.count_nonzero() == 0


@pytest.mark.parametrize('preset', ['joint_carry_at', 'joint_carry_ontop'])
def test_move_shuffle_bindings_human_swap_world_transform(preset):
    roots, objects, sizes, goals, graph = scene(preset)
    roots[0, :, 7:9] = torch.tensor([[1., 0.], [3., 4.]])
    gate = torch.tensor([True])
    expected, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    shuffled = permute_graph(graph, torch.tensor([[3, 1, 0, 2]]))
    actual, _ = shared_alignment_reward(roots.flip(1), objects, sizes, goals, shuffled, gate, CONFIG)
    torch.testing.assert_close(actual, expected)
    remapped = deepcopy(shuffled)
    order = torch.tensor([0, 1, 3, 2, 5, 4, 7, 6])
    remapped.edge_src.copy_(order[remapped.edge_src]); remapped.edge_dst.copy_(order[remapped.edge_dst])
    actual, _ = shared_alignment_reward(roots, objects[:, [1, 0, 3, 2]], sizes[:, [1, 0, 3, 2]],
        goals.flip(1), remapped, gate, CONFIG)
    torch.testing.assert_close(actual, expected)
    angle = torch.tensor([1.7]); shift = torch.tensor([[20., -10., 0.]])
    targets = torch.zeros(1, 2, 13); targets[..., :3] = goals; targets[..., 6] = 1.
    actual, _ = shared_alignment_reward(transform_states(roots, angle, shift),
        transform_states(objects, angle, shift), sizes,
        transform_states(targets, angle, shift)[..., :3], graph, gate, CONFIG)
    torch.testing.assert_close(actual, expected)
    independent = sample_graph(1, ENV['relationGraph'], preset='independent_at_at')
    actual, _ = shared_alignment_reward(roots, objects, sizes, goals, independent, gate, CONFIG)
    assert actual.count_nonzero() == 0
