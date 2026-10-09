from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.task_coordination import task_relations
from utils.edge_context_spec import HOLDING
from utils.task_role_spec import task_graph_packet
from utils.mixed_carry_spec import MIXED_SAMPLER, sample_graph, joint_scenes, environment_mask
from env.tasks.multi_agent.joint_carry_reward import JointCarryRuntime, apply_joint_geometry
from test_joint_carry_runtime import ENV


SPEC = dict(ENV['relationGraph'], sampler=MIXED_SAMPLER,
    family_probabilities={'joint_carry_at': .4, 'joint_carry_ontop': .4, 'independent': .2})


def test_mixed_sampler_distribution_disjoint_bindings_and_relations():
    graph = sample_graph(20000, SPEC, generator=torch.Generator().manual_seed(18))
    packet = task_graph_packet(graph, 20000).reshape(-1, 2, 5).long()
    joint = joint_scenes(graph)
    assert abs(joint.float().mean() - .8) < .015
    assert abs(((packet[:, 0, 1] == 2) & joint).float().mean() - .4) < .015
    assert (packet[joint, 0, 3:] == packet[joint, 1, 3:]).all()
    solo = packet[~joint]
    assert (solo[:, :, 3] == torch.tensor([2, 3])).all()
    assert (solo[:, 0, 4] != solo[:, 1, 4]).all()
    for a in (2, 3):
        for b in (2, 3):
            assert abs(((solo[:, 0, 1] == a) & (solo[:, 1, 1] == b)).float().mean() - .25) < .03
    relations = task_relations(packet, 2)
    assert (relations[joint] == torch.tensor([[1, 2], [2, 1]])).all()
    assert (relations[~joint] == torch.tensor([[1, 0], [0, 1]])).all()


def test_independent_reward_only_needs_own_holding_and_keeps_center_geometry():
    graph = sample_graph(2, SPEC, preset='independent_at_ontop')
    runtime = JointCarryRuntime(2, graph, deepcopy(ENV['relationReward']), 'cpu')
    phi = torch.ones(2, 4)
    holding = graph.edge_relation == HOLDING
    phi[holding & (graph.edge_owner == 1)] = 0.
    zero = torch.zeros_like(phi)
    result = runtime.step(phi, phi, zero, zero, zero)
    place_a = ~holding & (graph.edge_owner == 0)
    place_b = ~holding & (graph.edge_owner == 1)
    assert (result['total'][place_a] > 0.).all()
    assert not result['total'][place_b].any()
    phi[holding & (graph.edge_owner == 0)] = 0.
    phi[holding & (graph.edge_owner == 1)] = 1.
    result = runtime.step(phi, phi, zero, zero, zero)
    assert not result['total'][place_a].any()
    assert (result['total'][place_b] > 0.).all()
    objects = torch.zeros(2, 4, 13); objects[..., 6] = 1.
    sizes = torch.ones(2, 4, 3) * .4
    hands = torch.zeros(2, 2, 2, 3)
    diag = {'target': torch.randn(2, 4, 3), 'distance': torch.ones(2, 4)}
    old = deepcopy(diag)
    adjusted, gate = apply_joint_geometry(phi, diag, hands, objects, sizes, graph, ENV['relationReward'])
    torch.testing.assert_close(adjusted, phi)
    torch.testing.assert_close(diag['target'], old['target'])
    assert not gate.any()


def test_requested_scene_mask_is_preserved_on_partial_resets():
    mask = torch.tensor([True, False, False, True, False])
    graph = sample_graph(5, SPEC, joint_mask=mask)
    assert torch.equal(joint_scenes(graph), mask)


def test_fixed_environment_allocation_and_config_isolation():
    from utils.joint_carry_spec import validate_env
    from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
    from utils.edge_stage1_spec import compile_stage1_graph
    root = Path(__file__).resolve().parents[2] / 'tokenhsi/data/cfg/multi_agent'
    env = yaml.safe_load((root / 'approach_stage2_joint_carry_mixed80_task_embedding.yaml').read_text())['env']
    stage1 = yaml.safe_load((root / 'approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml').read_text())['env']
    validate_env(env)
    validate_relation_config(env['relationReward'])
    compile_stage1_graph(env['relationGraph'], 2, 4)
    assert env['sizeAwareRsi'] == stage1['sizeAwareRsi']
    assert env['independentCarryRsi'] == stage1['templateRsi']['HOLDING_AT'] == stage1['templateRsi']['HOLDING_ON_TOP']
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(ENV['relationReward'])},
                                  checkpoint_metadata(env['relationReward']))
    mask = environment_mask(2048, 'cpu')
    assert mask.sum() == 1638
    for ids in (torch.arange(2048), torch.arange(0, 2048, 3)):
        graph = sample_graph(len(ids), SPEC, joint_mask=mask[ids])
        assert torch.equal(joint_scenes(graph), mask[ids])
    assert environment_mask(16, 'cpu', 'joint_carry_at').all()
    assert not environment_mask(16, 'cpu', 'independent_at_ontop').any()
