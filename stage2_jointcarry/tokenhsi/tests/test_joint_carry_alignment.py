from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.joint_carry_reward import shared_alignment_reward
from utils.joint_carry_amp import validate_ablation
from utils.joint_carry_rsi import transform_states
from utils.joint_carry_spec import validate_env
from utils.mixed_carry_spec import sample_graph
from utils.edge_ontop_spec import permute_graph
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata

ROOT = Path(__file__).resolve().parents[1]
NAME = 'approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding'
ENV = yaml.safe_load((ROOT/'data/cfg/multi_agent'/(NAME+'.yaml')).read_text())['env']
CONFIG = ENV['relationReward']['joint_carry']['alignment']


def scene(preset='joint_carry_at', distance=3.):
    graph = sample_graph(1, ENV['relationGraph'], preset=preset)
    roots = torch.zeros(1, 2, 13); roots[..., 6] = 1.
    roots[0, :, 0] = torch.tensor([-.5, .5])
    roots[0, 1, 5:7] = torch.tensor([1., 0.])
    objects = torch.zeros(1, 4, 13); objects[..., 6] = 1.
    objects[0, 2, 0] = distance
    sizes = torch.tensor([[[.52, .8, .4]]]).expand(1, 4, -1)
    goals = torch.tensor([[[distance, 0., .2], [99., 99., .2]]])
    return roots, objects, sizes, goals, graph


def test_config_isolated_and_checkpoint_contract():
    base = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_task_embedding.yaml').read_text())['env']
    changed = deepcopy(ENV); changed['relationReward']['joint_carry'].pop('alignment')
    assert changed == base
    train = yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_align_task_embedding.yaml').read_text())
    validate_env(ENV); validate_ablation(ENV, train); validate_relation_config(ENV['relationReward'])
    for value in (.5, -1., float('nan')):
        bad = deepcopy(ENV['relationReward']); bad['joint_carry']['alignment']['buffer'] = value
        with pytest.raises(ValueError, match='alignment'):
            validate_relation_config(bad)
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base['relationReward'])},
                                  checkpoint_metadata(ENV['relationReward']))


@pytest.mark.parametrize('preset', ['joint_carry_at', 'joint_carry_ontop'])
def test_farthest_changes_with_target_and_heading_score(preset):
    roots, objects, sizes, goals, graph = scene(preset)
    gate = torch.tensor([True])
    reward, diag = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    torch.testing.assert_close(reward, torch.tensor([.1])); assert diag['leader'].item() == 0
    roots[0, 0, 5:7] = torch.tensor([3**.5/2, .5])
    reward, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    torch.testing.assert_close(reward, torch.zeros(1))
    objects[0, 2, 0] = -3.; goals[0, 0, 0] = -3.
    reward, diag = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    torch.testing.assert_close(reward, torch.tensor([.1])); assert diag['leader'].item() == 1


@pytest.mark.parametrize('preset,radius', [('joint_carry_at', .8),
    ('joint_carry_ontop', (.52**2+.8**2)**.5/2+.8)])
def test_radius_hard_saturation_zero_distance_and_holding_gate(preset, radius):
    roots, objects, sizes, goals, graph = scene(preset, radius+.01)
    roots[..., 5:7] = torch.tensor([1., 0.])
    reward, diag = shared_alignment_reward(roots, objects, sizes, goals, graph, torch.tensor([True]), CONFIG)
    torch.testing.assert_close(reward, torch.zeros(1))
    torch.testing.assert_close(diag['radius'], torch.tensor([radius]))
    for distance in (radius-.01, 0.):
        objects[0, 2, 0] = distance; goals[0, 0, 0] = distance
        reward, diag = shared_alignment_reward(roots, objects, sizes, goals, graph, torch.tensor([True]), CONFIG)
        torch.testing.assert_close(reward, torch.tensor([.1])); assert diag['saturated'].all()
        off, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, torch.tensor([False]), CONFIG)
        torch.testing.assert_close(off, torch.zeros(1))


def test_edge_shuffle_human_swap_world_transform_and_independent_exclusion():
    roots, objects, sizes, goals, graph = scene('joint_carry_ontop')
    gate = torch.tensor([True])
    expected, _ = shared_alignment_reward(roots, objects, sizes, goals, graph, gate, CONFIG)
    shuffled = permute_graph(graph, torch.tensor([[3, 1, 0, 2]]))
    actual, _ = shared_alignment_reward(roots.flip(1), objects, sizes, goals, shuffled, gate, CONFIG)
    torch.testing.assert_close(actual, expected)
    angle = torch.tensor([1.7]); shift = torch.tensor([[20., -10., 0.]])
    target = torch.zeros(1, 2, 13); target[..., :3] = goals; target[..., 6] = 1.
    actual, _ = shared_alignment_reward(transform_states(roots, angle, shift),
        transform_states(objects, angle, shift), sizes,
        transform_states(target, angle, shift)[..., :3], graph, gate, CONFIG)
    torch.testing.assert_close(actual, expected)
    independent = sample_graph(1, ENV['relationGraph'], preset='independent_at_at')
    actual, _ = shared_alignment_reward(roots, objects, sizes, goals, independent, gate, CONFIG)
    torch.testing.assert_close(actual, torch.zeros(1))
