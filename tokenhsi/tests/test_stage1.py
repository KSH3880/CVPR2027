"""Stage1 shared bias, unchanged Action behavior, and carry-only platform resets."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import test_task_role_message as reference
import test_task_type_embedding as embedding
import test_task_embedding_action as action
from utils.task_role_spec import task_graph_packet
from utils.stage1_platforms import reset_stage1_platforms, sample_payload_height
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from utils.unified_training import validate_unified_env, validate_typed_bias_config

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/stage1.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'data/cfg/train/rlg/stage1.yaml').read_text())


@pytest.fixture(autouse=True)
def use_config(monkeypatch):
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)


def test_config_parity_and_checkpoint_isolation():
    expected = deepcopy(action.CFG)
    expected['relationReward']['stage1_variant'] = 'stage1'
    expected.update(agentCollisionMode='cpa', agentCollisionTTCDiscount=.99)
    expected['box']['reset'].update(randomHeight=True, randomHeightProb=.5,
        minPlatformHeight=0., platformMode='carry_payload_tokenhsi',
        ownerLocoDistanceRange=[1., 10.], samplingMode='tokenhsi')
    for name in ('HOLDING_AT', 'HOLDING_ON_TOP'):
        expected['templateRsi'][name] = [.5, 0., 0., 0., 0., .1, .3, .1]
    assert CFG == expected
    expected = deepcopy(action.TRAIN)
    expected['params']['network']['transformer']['relation_bias_mode'] = 'task_shared_embedding_action'
    assert TRAIN == expected
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for new, old in ((CFG, action.CFG), (action.CFG, CFG)):
        with pytest.raises(ValueError, match='mismatch'):
            check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(new['relationReward'])},
                                      checkpoint_metadata(old['relationReward']))
    for env, train in ((CFG, action.TRAIN), (action.CFG, TRAIN)):
        with pytest.raises(ValueError):
            validate_typed_bias_config(env, train)


@pytest.mark.parametrize('key,value', [('randomHeight', False), ('randomHeightProb', .3),
    ('platformMode', None), ('minPlatformHeight', .05), ('maxTopSurfaceHeight', 1.3),
    ('testRandomArenaSpawn', True)])
def test_platform_contract_rejects_silent_changes(key, value):
    env = deepcopy(CFG)
    env['box']['reset'][key] = value
    with pytest.raises(ValueError):
        validate_unified_env(env)


def test_bias_independent_of_all_type_pairs_and_roles_with_manual_oracle():
    enc = reference.network().actor_encoder
    assert enc.edge_encoder.bias_projection.shape == (4, 2, 64)
    assert sum(p.numel() for p in enc.edge_encoder.parameters()) == 768
    with torch.no_grad():
        enc.edge_encoder.category_embed.weight.normal_()
        enc.edge_encoder.bias_projection.normal_()
    packet = task_graph_packet(reference.all_pairs(), 16)
    actual = embedding.fused(enc, packet)
    expected = torch.zeros_like(actual)
    for b, records in enumerate(packet.reshape(16, 2, 5).long()):
        for valid, task, actor, payload, target in records:
            pairs = [(actor, target)] + ([(actor, payload), (payload, target)] if task >= 2 else [])
            bias = (enc.edge_encoder.category_embed.weight[task] * enc.edge_encoder.bias_projection).sum(-1)
            for src, dst in pairs:
                expected[:, b, :, src, dst] = bias
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, embedding.fused(enc, packet, enc.entity_types.roll(3)), atol=0, rtol=0)
    assert enc.build_relation_bias().count_nonzero() == 0
    assert embedding.fused(enc, torch.zeros_like(packet)).count_nonzero() == 0


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_action_and_shared_bias_token_edge_task_permutation_gradients(branch):
    action.test_token_edge_task_shuffle_outputs_and_all_gradients(branch)


@pytest.mark.parametrize('check', [reference.test_all_pairs_packet_binding_owner_and_edge_shuffle,
    reference.test_size_sampling_no_standalone_holding_preserves_family_rsi_bindings,
    reference.test_rewards_equal_old_local_sum_no_teammate_dependency_and_partial_suffix,
    reference.test_carry_ontop_payload_above_support_and_at_goal_match_packet,
    reference.test_checkpoint_roundtrip_rebuild_and_actor_critic_separation,
    action.test_zero_start_two_step_learning_shared_layers_and_independent_branches,
    action.test_post_gta_pre_output_projection_placement_and_attention_independence,
    embedding.test_arbitrary_endpoint_and_type_permutation_remaps_bias])
def test_runtime_contracts(check):
    check()


def test_original_height_distribution_top_clamp_and_ground_probability():
    torch.manual_seed(5)
    size = torch.full((20000, 3), .6)
    z = sample_payload_height(size)
    assert ((z >= .3) & (z <= .900001)).all()
    assert .48 < float((z == .3).float().mean()) < .52
    # Clamping U(0,1) creates a point mass at top=1.2; rescaling would not.
    assert .18 < float(torch.isclose(z, torch.tensor(.9)).float().mean()) < .22


@pytest.mark.parametrize('skill', ['loco', 'pickUp', 'carryWith', 'putDown'])
def test_platform_scope_rsi_exceptions_binding_and_partial_reset(skill):
    graph = reference.all_pairs()
    records = task_graph_packet(graph, 16).reshape(16, 2, 5).long()
    # Swap physical boxes and goals to catch accidental owner-index assumptions.
    object_order = torch.tensor([3, 2, 1, 0])
    goal_order = torch.tensor([1, 0])
    ids = torch.tensor([1, 3, 6, 10, 11, 14, 15])
    env = ids.repeat_interleave(2)
    owner = torch.arange(2).repeat(len(ids))
    payload = lambda e, a: object_order[records[e, a, 3] - 2]
    goal = lambda e, a: goal_order[records[e, a, 4] - 6]
    task = SimpleNamespace(_box_states=torch.randn(16, 4, 13), _box_size=torch.full((16,4,3), .4),
        _tar_pos=torch.randn(16,2,3), _platform_pos=torch.full((16,2,3), -99.),
        _tar_platform_pos=torch.full((16,2,3), -98.),
        _platform_default_pos=torch.full((16,2,3), 20.),
        _tar_platform_default_pos=torch.full((16,2,3), 20.5),
        _reset_random_height_prob=1., _reset_max_top_surface_height=1.2, _platform_height=.02,
        relation_runtime=SimpleNamespace(graph=graph),
        _scenario_physical_object=payload, _scenario_goal=goal,
        _collect_random_slots=lambda skills: (env, owner) if skill in skills else None)
    old = deepcopy(task)
    reset_stage1_platforms(task, ids)
    for e in range(16):
        expected_boxes = old._box_states[e].clone()
        expected_goals = old._tar_pos[e].clone()
        for a in range(2):
            category = records[e,a,1]
            source = e in ids and skill == 'loco' and category >= 2
            target = e in ids and skill != 'putDown' and category == 2
            if source:
                p = payload(e,a)
                z = task._box_states[e,p,2]
                assert .249999 <= z <= 1.050001
                expected_boxes[p,2] = z
                torch.testing.assert_close(task._platform_pos[e,a,:2], old._box_states[e,p,:2])
                torch.testing.assert_close(task._platform_pos[e,a,2], z - .25 - .01)
            else:
                torch.testing.assert_close(task._platform_pos[e,a],
                    task._platform_default_pos[e,a] if e in ids else old._platform_pos[e,a])
            if target:
                g = goal(e,a)
                z = task._tar_pos[e,g,2]
                expected_goals[g,2] = z
                torch.testing.assert_close(task._tar_platform_pos[e,a,:2], old._tar_pos[e,g,:2])
                torch.testing.assert_close(task._tar_platform_pos[e,a,2], z - .2 - .01)
            else:
                torch.testing.assert_close(task._tar_platform_pos[e,a],
                    task._tar_platform_default_pos[e,a] if e in ids else old._tar_platform_pos[e,a])
        torch.testing.assert_close(task._box_states[e], expected_boxes)
        torch.testing.assert_close(task._tar_pos[e], expected_goals)
