from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from utils.edge_scenario_spec import sample_graph, classify_templates, agent_object_indices, validate_sampler
from utils.edge_stage1_spec import semantic_graph_packet, owner_holding_graph_packet
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from utils.unified_training import (append_family, amp_family_ids, family_from_templates,
    sample_family_matched, preserve_amp_labels, validate_unified_env, EXPERT_PROBS, FAMILY_PROBS)
from env.tasks.multi_agent.scene_features import build_gta_pose_records
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder

ROOT = Path(__file__).resolve().parents[1]


def config(owner=False):
    suffix = '_owner_holding' if owner else ''
    return yaml.safe_load((ROOT / ('data/cfg/multi_agent/approach_scenario_stage1_unified'
                                   + suffix + '.yaml')).read_text())['env']


def test_two_variants_differ_only_in_edge_state_and_contract():
    base, owner = config(), config(True)
    expected = deepcopy(base)
    expected['relationGraph']['owner_holding_state'] = True
    expected['relationReward']['stage1_variant'] += '_owner_holding'
    expected['relationReward']['observation']['graph_packet_fields'].append('owner_holding_state')
    assert owner == expected
    for cfg in (base, owner):
        validate_sampler(cfg['relationGraph'], 2, 4)
        validate_relation_config(cfg['relationReward'])
        validate_unified_env(cfg)
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base['relationReward'])},
                                   checkpoint_metadata(owner['relationReward']))
    expected['ampTaskConditioning'] = False
    with pytest.raises(ValueError, match='must be paired'):
        validate_unified_env(expected)


def test_shared_edge_encoder_config_and_gradients():
    torch.manual_seed(117)
    torch.set_num_threads(1)
    base = config()
    shared = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder.yaml').read_text())['env']
    expected = deepcopy(base)
    expected['relationReward']['stage1_variant'] += '_shared_edge_encoder'
    assert shared == expected
    validate_sampler(shared['relationGraph'], 2, 4)
    validate_relation_config(shared['relationReward'])
    validate_unified_env(shared)
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base['relationReward'])},
                                  checkpoint_metadata(shared['relationReward']))

    base_train = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())
    shared_train = yaml.safe_load((ROOT / 'data/cfg/train/rlg/amp_ma_carry_relation_unified_shared_edge_encoder.yaml').read_text())
    expected_train = deepcopy(base_train)
    expected_train['params']['network']['transformer']['share_edge_encoder'] = True
    assert shared_train == expected_train
    builder = AMPMultiAgentBuilder()
    builder.load(shared_train['params']['network'])
    network = builder.build('amp', actions_num=32, input_shape=(644,),
        amp_input_shape=(1320,), value_size=1, num_agents=2, num_objects=4,
        humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
        observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7,
        scene_arena_scale=5., relation_reward_mode=shared['relationReward']['mode'],
        relation_graph_spec=shared['relationGraph'], device='cpu')
    edge = network.actor_encoder.edge_encoder
    assert network.critic_encoder.edge_encoder is edge
    assert network.actor_encoder.layers[0] is not network.critic_encoder.layers[0]
    assert sum(p is edge.bias_projection for p in network.parameters()) == 1
    for encoder in (network.actor_encoder, network.critic_encoder):
        encoder.gta_diagnostics_first_forward = False
    graph = sample_graph(4, shared['relationGraph'])
    packet = semantic_graph_packet(graph, 4)
    kin = torch.randn(4, 8, 7)
    kin[..., 3:] = torch.nn.functional.normalize(kin[..., 3:], dim=-1)
    obs = torch.cat([torch.randn(4, 568), kin.flatten(1), packet], -1)
    for encoder, head in ((network.actor_encoder, network.action_head),
                          (network.critic_encoder, network.value_head)):
        network.zero_grad()
        head(encoder(obs)).square().mean().backward()
        assert edge.bias_projection.grad is not None
        assert edge.bias_projection.grad.abs().sum() > 0


def test_independent_distribution_and_canonical_endpoints():
    n = 5000
    cfg = config()
    graph = sample_graph(n, cfg['relationGraph'], generator=torch.Generator().manual_seed(845))
    env = torch.arange(n).repeat_interleave(2)
    agent = torch.arange(2).repeat(n)
    templates = classify_templates(graph, env, agent, with_climb=True).reshape(n, 2)
    assert torch.equal(agent_object_indices(graph, env, agent), agent)
    p = torch.tensor([.05, .05, .20, .35, .35])
    for a in range(2):
        torch.testing.assert_close(torch.bincount(templates[:, a], minlength=5) / n, p, atol=.025, rtol=0)
    joint = torch.bincount(templates[:, 0]*5+templates[:, 1], minlength=25).reshape(5, 5)/n
    assert (joint > 0).all()  # includes SIT+CLIMB, HOLDING+AT and two ON_TOP
    torch.testing.assert_close(joint, p[:, None]*p[None], atol=.018, rtol=0)
    top = graph.edge_valid & (graph.edge_relation == 8)
    at = graph.edge_valid & (graph.edge_relation == 7)
    assert torch.equal(graph.edge_src[top], 2+graph.edge_owner[top])
    assert torch.equal(graph.edge_dst[top], 4+graph.edge_owner[top])
    assert torch.equal(graph.edge_src[at], 2+graph.edge_owner[at])
    assert torch.equal(graph.edge_dst[at], 6+graph.edge_owner[at])


def test_amp_labels_follow_task_not_reset_skill_and_match_replay():
    family = family_from_templates(torch.tensor([0, 1, 2, 3, 4]))
    assert family.tolist() == [0, 1, 2, 0, 0]
    reference = append_family(torch.randn(5, 10, 4), family[:, None].expand(-1, 10)).flatten(1)
    # Marker body features identify the true source family; matching must not relabel.
    source_family = torch.tensor([2, 0, 2, 0])
    pool = append_family(source_family[:, None, None].expand(-1, 10, 4).float(),
                         source_family[:, None].expand(-1, 10)).flatten(1)
    fallback = append_family(family[:, None, None].expand(-1, 10, 4).float(),
                             family[:, None].expand(-1, 10)).flatten(1)
    sampled = sample_family_matched(pool, reference, 10, fallback)
    assert torch.equal(amp_family_ids(sampled, 10), family)
    assert torch.equal(sampled[:, 0], family.float())
    assert torch.equal(sampled.view(5, 10, 7)[..., -3:], reference.view(5, 10, 7)[..., -3:])
    with pytest.raises(ValueError, match='sit'):
        sample_family_matched(pool, reference, 10)
    normalized = preserve_amp_labels(reference, reference * 7 - 3, 10).view(5, 10, 7)
    assert torch.equal(normalized[..., -3:], reference.view(5, 10, 7)[..., -3:])
    assert torch.equal(normalized[..., :-3], (reference*7-3).view(5,10,7)[..., :-3])
    torch.testing.assert_close(torch.tensor(EXPERT_PROBS).sum(-1), torch.ones(3))
    nominal = torch.tensor(FAMILY_PROBS) @ torch.tensor(EXPERT_PROBS)
    torch.testing.assert_close(nominal, torch.tensor(config()['skillDiscProb']))
    assert nominal[6] == 0  # carryWith is RSI-only in the unified baseline


def test_goal_identity_rotation_does_not_follow_human_heading():
    hp=torch.randn(2,2,3);op=torch.randn(2,4,3);gp=torch.randn(2,2,3)
    hq=torch.nn.functional.normalize(torch.randn(2,2,4),dim=-1)
    oq=torch.nn.functional.normalize(torch.randn(2,4,4),dim=-1)
    origin=torch.randn(2,3)
    poses=build_gta_pose_records(hp,hq,op,oq,gp,origin,goal_rotation='identity')
    torch.testing.assert_close(poses[:,-2:,:3],gp-origin[:,None])
    torch.testing.assert_close(poses[:,-2:,3:],torch.tensor([0.,0.,0.,1.]).expand(2,2,4))
    legacy=build_gta_pose_records(hp,hq,op,oq,gp,origin)
    torch.testing.assert_close(legacy[:,-2:,3:],hq)
    torch.testing.assert_close(legacy[:,:6],poses[:,:6])


@pytest.mark.parametrize('owner', [False, True])
def test_token_permutation_preserves_actions_values_and_gradients(owner):
    torch.manual_seed(117)
    torch.set_num_threads(1)
    cfg=config(owner)
    params=yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())['params']['network']
    builder=AMPMultiAgentBuilder();builder.load(params)
    width=24 if owner else 20
    network=builder.build('amp', actions_num=32, input_shape=(624+width,),
        amp_input_shape=(1080,),value_size=1,num_agents=2,num_objects=4,
        humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,
        observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,
        scene_arena_scale=5.,relation_reward_mode=cfg['relationReward']['mode'],
        relation_graph_spec=cfg['relationGraph'],device='cpu')
    assert network.actor_encoder.edge_encoder is not network.critic_encoder.edge_encoder
    network.eval()
    graph=sample_graph(8,cfg['relationGraph'])
    phi=torch.rand(8,4)
    packet=owner_holding_graph_packet(graph,phi) if owner else semantic_graph_packet(graph,8)
    kin=torch.randn(8,8,7);kin[...,3:]=torch.nn.functional.normalize(kin[...,3:],dim=-1)
    obs=torch.cat([torch.randn(8,568),kin.flatten(1),packet],-1)
    for encoder,head in [(network.actor_encoder,network.action_head),
                         (network.critic_encoder,network.value_head)]:
        encoder.gta_diagnostics_first_forward=False
        if owner:
            # Exercise nonzero state residual, not just its zero-initialised path.
            torch.nn.init.normal_(encoder.context_fusion.state_mlp[-1].weight,std=.1)
        identity=torch.arange(8)
        permutation=torch.tensor([6,4,1,3,7,0,5,2])
        y=head(encoder(obs,token_order=identity))
        y.square().mean().backward()
        grads={n:p.grad.clone() for n,p in encoder.named_parameters() if p.grad is not None}
        encoder.zero_grad();head.zero_grad()
        z=head(encoder(obs,token_order=permutation))
        torch.testing.assert_close(y,z,atol=2e-5,rtol=2e-5)
        z.square().mean().backward()
        for n,p in encoder.named_parameters():
            if n in grads:
                torch.testing.assert_close(grads[n],p.grad,atol=5e-5,rtol=1e-3)
        assert z.shape == (8,2,32 if head is network.action_head else 1)
