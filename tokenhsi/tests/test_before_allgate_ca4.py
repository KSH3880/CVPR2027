from copy import deepcopy

import pytest
import torch
import yaml

from env.tasks.multi_agent.before_reward import BeforeRuntime
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from learning.multi_agent.task_coordination import TaskCoordination
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias
from utils.before_spec import sample_graph, validate_env
from utils.edge_context_spec import AT, HOLDING
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from test_before import ROOT, ENV, SPEC
from test_joint_carry import build_network as source_network
from test_stage2_coordination import Wrapper
from test_task_role_message import observations


NEW_ENV = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_stage2_before_allgate_ca4_task_embedding.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_stage2_before_allgate_ca4_task_embedding.yaml').read_text())


def network():
    builder = AMPMultiAgentBuilder()
    builder.load(TRAIN['params']['network'])
    return builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1320,),
        value_size=1, num_agents=2, num_objects=4, humanoid_obs_size=230,
        object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=NEW_ENV['relationReward']['mode'], relation_graph_spec=SPEC, device='cpu')


def test_config_scope_and_checkpoint_isolation():
    validate_env(NEW_ENV)
    validate_relation_config(NEW_ENV['relationReward'])
    restored = deepcopy(NEW_ENV)
    restored['relationReward']['before']['progress'] = 'always'
    assert restored == ENV
    restored = deepcopy(TRAIN)
    restored['params']['config']['name'] = 'Stage2BeforeTaskEmbedding'
    coordination = restored['params']['network']['coordination']
    assert coordination.pop('num_layers') == 4
    assert coordination.pop('feedforward_dim') == 128
    old_train = yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_stage2_before_task_embedding.yaml').read_text())
    assert restored == old_train
    old, new = [checkpoint_metadata(e['relationReward']) for e in (ENV, NEW_ENV)]
    for saved, target in ((old, new), (new, old)):
        with pytest.raises(ValueError):
            check_checkpoint_metadata({'relation_metadata': saved}, target)
    invalid = deepcopy(NEW_ENV['relationReward'])
    invalid['before']['progress'] = 'unknown'
    with pytest.raises(ValueError):
        validate_relation_config(invalid)


@pytest.mark.parametrize('preset', ['before_sit', 'before_climb', 'before_stack'])
def test_allgate_current_success_reclosure_and_holding(preset):
    graph = sample_graph(8, SPEC, preset=preset)
    runtime = BeforeRuntime(8, graph, NEW_ENV['relationReward'], 'cpu')
    baseline = BeforeRuntime(8, graph, ENV['relationReward'], 'cpu')
    phi = torch.ones(8, 4)
    progress = torch.full_like(phi, .37)
    zero = torch.zeros_like(phi)
    dependent = graph.prereq_mask.any(-1)
    at = (graph.edge_relation == AT) & graph.edge_valid
    holding_b = (graph.edge_relation == HOLDING) & (graph.edge_owner == 1) & graph.edge_valid
    for delivered in (False, True, False):
        phi[at] = float(delivered)
        new = runtime.step(phi, progress, zero, zero, zero)
        old = baseline.step(phi, progress, zero, zero, zero)
        for key in ('state_component', 'progress_component', 'success_component'):
            torch.testing.assert_close(new[key][dependent], torch.full((8,), .2 if delivered else 0.))
        torch.testing.assert_close(new['total'][~dependent], old['total'][~dependent])
        assert (new['total'][holding_b] > 0).all()
        assert (runtime.done[:, 1] == delivered).all()
        torch.testing.assert_close(new['agent_task_reward'].sum(-1), new['total'].sum(-1))
    ids = torch.tensor([1, 5])
    runtime.reset(ids, phi[ids], zero[ids], zero[ids], zero[ids])
    assert not runtime.done.any()


def test_allgate_independent_reward_remains_stage1():
    graph = sample_graph(40, SPEC, preset='independent')
    phi, progress = torch.rand(40, 4), torch.rand(40, 4)
    zero = torch.zeros_like(phi)
    new = BeforeRuntime(40, graph, NEW_ENV['relationReward'], 'cpu').step(phi, progress, zero, zero, zero)
    old = Stage1ContextRuntime(40, graph, NEW_ENV['relationReward'], 'cpu').step(phi, progress, zero, zero, zero)
    for key in ('total', 'own_success', 'agent_task_reward', 'reward_saturated'):
        torch.testing.assert_close(new[key], old[key])


def test_ca4_before_padding_permutation_and_gradients():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    module = TaskCoordination(before=True, num_layers=4, feedforward_dim=128)
    embed = TaskTypeEmbeddingBias(4, 2).requires_grad_(False)
    nodes = torch.randn(2, 9, 64)
    packet = torch.tensor([[[1, 2, 0, 3, 7], [1, 1, 1, 0, 3],
                           [1, 3, 2, 4, 6], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]]*4])
    output = module(nodes[:, :3], nodes, packet, embed)
    assert output[1].count_nonzero() == 0 and torch.isfinite(output).all()
    shuffled = packet[:, [2, 0, 3, 1]]
    order = torch.tensor([2, 0, 1, 5, 3, 4, 8, 6, 7])
    remapped = packet.clone()
    remapped[..., 2:] = order.argsort()[packet[..., 2:]]
    torch.testing.assert_close(output[:, order[:3]], module(nodes[:, order[:3]], nodes[:, order], remapped, embed))
    torch.testing.assert_close(output, module(nodes[:, :3], nodes, shuffled, embed))
    weight = torch.randn_like(output)
    (output*weight).sum().backward()
    gradients = {name:p.grad.clone() for name,p in module.named_parameters()}
    for block in module.blocks:
        assert (block.relation_bias.grad[:, [0, 1, 3]].abs().sum(0) > 0).all()
        assert block.relation_bias.grad[:, 2].count_nonzero() == 0
    module.zero_grad()
    (module(nodes[:, :3], nodes, shuffled, embed)*weight).sum().backward()
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-5, rtol=2e-4)
    assert all(p.grad is None for p in embed.parameters())


def test_ca4_stage1_transfer_freeze_updates_and_reload():
    torch.manual_seed(43)
    torch.set_num_threads(1)
    source, target = Wrapper(source_network(False)), Wrapper(network())
    source_reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml').read_text())['env']['relationReward']
    report = transfer_stage1_weights(target, dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(source_reward)))
    net = target.a2c_network
    assert len(report['frozen_parameters']) == 69 and len(net.coordination.blocks) == 4
    assert net.action_head[0][0].weight[:, 64:].count_nonzero() == 0
    obs = observations(sample_graph(16, SPEC, families=torch.tensor([0, 1, 2, 3]*4)))
    torch.testing.assert_close(net.eval_actor(obs)[0], source.a2c_network.action_head(net.actor_encoder(obs)).flatten(0, 1))
    frozen = {k:v.clone() for k,v in net.actor_encoder.state_dict().items()}
    initial = {k:v.clone() for k,v in net.coordination.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        net.eval_actor(obs)[0].square().mean().backward()
        optimizer.step()
    assert all(torch.equal(v, net.actor_encoder.state_dict()[k]) for k,v in frozen.items())
    assert all(not p.requires_grad and p.grad is None for p in net.actor_encoder.parameters())
    for i in range(4):
        for key in ('attention.in_proj_weight', 'query_norm.weight', 'ffn.0.weight', 'relation_bias'):
            name = 'blocks.{}.{}'.format(i, key)
            assert not torch.equal(initial[name], net.coordination.state_dict()[name])
    restored = network()
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(net.eval_actor(obs)[0], restored.eval_actor(obs)[0])
