"""Four-object Stage-2 preserves a three-object, unconditioned rescue policy."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.stage2_transfer import transfer_stage1_weights, load_stage1_for_evaluation
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from utils.edge_stage1_spec import semantic_graph_packet, compile_stage1_graph, parse_semantic_packet
from utils.edge_ontop_spec import ON_TOP, permute_graph
from utils.edge_context_spec import AT
from utils.edge_stage2_spec import (sample_graph, classify_family, validate_sampler,
    validate_graph, configure_evaluation_graph, general_rescue_spec,
    STAGE2_GENERAL_RESCUE_SAMPLER, STAGE2_PRESETS)
from utils.edge_scenario_spec import classify_templates
from utils.relation_task_spec import checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env

ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_rescue_klclimb50.yaml').read_text())['env']
SOURCE = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml').read_text())['env']


class Wrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.a2c_network = network


def network(stage2, objects=4, agents=2, graph_spec=None):
    name = 'amp_ma_stage2_rescue_klclimb50.yaml' if stage2 else 'amp_ma_carry_relation.yaml'
    params = yaml.safe_load((ROOT / 'data/cfg/train/rlg' / name).read_text())['params']['network']
    builder = AMPMultiAgentBuilder(); builder.load(params)
    graph = graph_spec or (SOURCE['relationGraph'] if objects == 3 else ENV['relationGraph'])
    width = (223 * agents + 30 * objects + agents + 7 * (2 * agents + objects) +
             5 * graph['edge_capacity'])
    return builder.build('rescue_test', actions_num=32, input_shape=(width,),
        amp_input_shape=(1290,), value_size=1, num_agents=agents, num_objects=objects,
        humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
        observation_mode='clean_scene', scene_entity_sizes=[223,30,1],
        scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=(ENV if stage2 else SOURCE)['relationReward']['mode'],
        relation_graph_spec=graph, device='cpu')


def observation(graph):
    batch = graph.edge_valid.shape[0]
    poses = torch.randn(batch,8,7)
    poses[...,3:] = torch.nn.functional.normalize(poses[...,3:],dim=-1)
    return torch.cat((torch.randn(batch,568), poses.flatten(1), semantic_graph_packet(graph,batch)), -1)


@pytest.mark.parametrize('agents,objects', [(2, 2), (2, 7), (3, 3), (4, 5), (5, 8), (8, 16)])
@pytest.mark.parametrize('preset', STAGE2_PRESETS)
def test_general_sampling_covers_every_human_with_valid_resource_bindings(agents, objects, preset):
    spec = general_rescue_spec(ENV['relationGraph'], agents, objects)
    graph = sample_graph(32, spec, preset=preset, generator=torch.Generator().manual_seed(31))
    validate_graph(graph)
    assert graph.edge_valid.shape == (32, 2 * agents)
    slots = torch.arange(32).repeat_interleave(agents)
    humans = torch.arange(agents).repeat(32)
    from utils.edge_scenario_spec import agent_object_indices, agent_goal_indices
    primary = agent_object_indices(graph, slots, humans)
    assert ((primary >= 0) & (primary < objects)).all()
    goals = agent_goal_indices(graph, slots, humans)
    assert ((goals >= 0) & (goals < agents)).all()
    for row in range(32):
        valid = graph.edge_valid[row]
        owners = graph.edge_owner[row, valid]
        assert set(owners.tolist()) == set(range(agents))
        assert ((owners.bincount(minlength=agents) >= 1) &
                (owners.bincount(minlength=agents) <= 2)).all()
        assert graph.required_goal[row, valid].sum() == agents
        # Independent scenes allocate distinct objects, including ON_TOP supports.
        if preset == 'independent':
            assert len(set(primary.view(32, agents)[row].tolist())) == agents
            on_top = valid & (graph.edge_relation[row] == ON_TOP)
            supports = graph.edge_dst[row, on_top]
            assert not set(supports.tolist()) & set((primary.view(32, agents)[row] + agents).tolist())
    if preset.startswith('place_'):
        expected = {'place_climb': 1, 'place_sit': 2, 'place_stack': 3}[preset]
        assert (classify_family(graph) == expected).all()


def test_general_sampling_is_reproducible_and_includes_odd_shared_groups():
    spec = general_rescue_spec(ENV['relationGraph'], 5, 7)
    a = sample_graph(512, spec, generator=torch.Generator().manual_seed(67))
    b = sample_graph(512, spec, generator=torch.Generator().manual_seed(67))
    for field in ('edge_src', 'edge_dst', 'edge_owner', 'edge_relation', 'edge_valid', 'required_goal'):
        torch.testing.assert_close(getattr(a, field), getattr(b, field))
    assert .85 < float((classify_family(a) > 0).float().mean()) < .95
    from utils.edge_scenario_spec import agent_object_indices
    primary = agent_object_indices(a, torch.arange(512).repeat_interleave(5),
                                   torch.arange(5).repeat(512)).view(512, 5)
    assert any(int(row.bincount(minlength=7).max()) >= 3 for row in primary)
    assert set(primary.flatten().tolist()) == set(range(7))


def test_general_sampling_conversion_is_evaluation_only_and_preserves_legacy_and_explicit():
    original = deepcopy(ENV)
    configure_evaluation_graph(original, True)
    assert original == ENV
    expanded = deepcopy(ENV)
    expanded.update(numAgents=4, numObjects=5)
    configure_evaluation_graph(expanded, False)
    assert expanded['relationGraph'] == ENV['relationGraph']
    configure_evaluation_graph(expanded, True)
    assert expanded['relationGraph']['sampler'] == STAGE2_GENERAL_RESCUE_SAMPLER
    validate_unified_env(expanded, evaluation=True)
    with pytest.raises(ValueError, match='evaluation only'):
        validate_unified_env(expanded)
    explicit = deepcopy(expanded)
    explicit['relationGraph'] = {'mode': 'stage2_explicit', 'edges': []}
    configure_evaluation_graph(explicit, True)
    assert explicit['relationGraph'] == {'mode': 'stage2_explicit', 'edges': []}
    for m, o in ((1, 4), (4, 3), (3, 0)):
        with pytest.raises(ValueError, match='objects >= agents'):
            general_rescue_spec(ENV['relationGraph'], m, o)
    bad = deepcopy(expanded['relationGraph'])
    bad['edge_capacity'] = 4
    with pytest.raises(ValueError, match='edge slots'):
        validate_sampler(bad, 4, 5)


def test_general_layout_clears_actual_box_radii_and_multiple_at_goals():
    from utils.stage2_evaluation_layout import sample_clear_xy, sample_at_goals
    torch.manual_seed(17)
    agents, objects, count = 8, 16, 16
    centers = torch.randn(count, 2) * 20
    angle = torch.arange(agents) * (2 * torch.pi / agents)
    roots = torch.zeros(count, agents, 3)
    roots[..., :2] = centers[:, None] + 2.5 * torch.stack((angle.cos(), angle.sin()), -1)
    sizes = .4 + .2 * torch.rand(count, objects, 3)
    radii = sizes[..., :2].norm(dim=-1) / 2
    boxes = torch.zeros(count, objects, 3)
    for obj in range(objects):
        occupied = torch.cat((roots[..., :2], boxes[:, :obj, :2]), 1)
        clearance = torch.cat((torch.ones(count, agents),
            (radii[:, obj, None] + radii[:, :obj] + .05).clamp_min(.7)), 1)
        boxes[:, obj, :2] = sample_clear_xy(centers, 4.5, occupied, clearance)
        assert ((boxes[:, obj, None, :2] - occupied).norm(dim=-1) >= clearance - 1e-5).all()
    graph = sample_graph(count, general_rescue_spec(ENV['relationGraph'], agents, objects),
                         preset='place_stack')
    goals = sample_at_goals(graph, boxes, sizes, roots, torch.zeros(count, agents, 3), centers, 4.5)
    for row in range(count):
        previous = []
        for edge in (graph.edge_valid[row] & (graph.edge_relation[row] == AT)).nonzero().flatten():
            src = graph.edge_src[row, edge] - agents
            dst = graph.edge_dst[row, edge] - agents - objects
            owner = graph.edge_owner[row, edge]
            xy = goals[row, dst, :2]
            clearance = radii[row, src] + radii[row] + .25
            clearance[src] = 1.
            assert ((xy - boxes[row, :, :2]).norm(dim=-1) >= clearance - 1e-5).all()
            assert (xy - roots[row, owner, :2]).norm() >= 1. - 1e-5
            for prior_xy, prior_radius in previous:
                assert (xy - prior_xy).norm() >= radii[row, src] + prior_radius + .35 - 1e-5
            assert (xy - centers[row]).norm() <= 4.5 + 1e-5
            previous.append((xy, radii[row, src]))


@pytest.mark.parametrize('agents,objects', [(3, 3), (4, 5), (5, 8), (8, 16)])
def test_general_sampling_model_and_rms_strict_loading_and_edge_permutation(agents, objects):
    trained = network(True).eval()
    spec = general_rescue_spec(ENV['relationGraph'], agents, objects)
    expanded = network(True, objects=objects, agents=agents, graph_spec=spec).eval()
    expanded.load_state_dict(trained.state_dict(), strict=True)
    # Activate the context branch so endpoint/edge errors affect the actions.
    with torch.no_grad():
        expanded.action_head[0][0].weight[:, 64:].normal_(0, .02)
    graph = sample_graph(2, spec, generator=torch.Generator().manual_seed(81))
    intrinsic = torch.randn(2, 224 * agents + 30 * objects)
    poses = torch.zeros(2, 2 * agents + objects, 7)
    poses[..., 6] = 1
    source_rms = SceneRunningMeanStd([223, 30, 1], [2, 4, 2],
        normalized_sizes=[223, 30, 0], kinematic_size=7, extra_passthrough_size=20).eval()
    target_rms = SceneRunningMeanStd([223, 30, 1], [agents, objects, agents],
        normalized_sizes=[223, 30, 0], kinematic_size=7,
        extra_passthrough_size=10 * agents).eval()
    target_rms.load_state_dict(source_rms.state_dict(), strict=True)
    def obs(g):
        return target_rms(torch.cat((intrinsic, poses.flatten(1), semantic_graph_packet(g, 2)), -1))
    reordered = permute_graph(graph, torch.arange(2 * agents - 1, -1, -1).repeat(2, 1))
    with torch.no_grad():
        mu, sigma = expanded.eval_actor(obs(graph))
        other_mu, other_sigma = expanded.eval_actor(obs(reordered))
        value = expanded.eval_critic(obs(graph))
    assert mu.shape == sigma.shape == (2 * agents, 32)
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all() and torch.isfinite(value).all()
    torch.testing.assert_close(mu, other_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sigma, other_sigma)


def test_rescue_three_human_double_climb_graph_is_evaluation_only():
    from utils.edge_stage2_spec import compile_explicit_graph
    spec = yaml.safe_load((ROOT / 'data/cfg/multi_agent/graphs/stage2_three_agent_place_two_climb.yaml').read_text())
    env = deepcopy(ENV)
    env['relationGraph'] = spec
    env['numAgents'] = 3
    with pytest.raises(ValueError, match='Rescue Stage-2 sampler'):
        validate_unified_env(env)
    validate_unified_env(env, evaluation=True)
    for field, value in (('ampTaskConditioning', True), ('numAMPObsSteps', 9)):
        bad = deepcopy(env)
        bad[field] = value
        with pytest.raises(ValueError, match='unconditioned ten-frame'):
            validate_unified_env(bad, evaluation=True)
    graph = compile_explicit_graph(spec, 3, 4)
    assert graph.edge_src.tolist() == [0, 3, 1, 2]
    assert graph.edge_dst.tolist() == [3, 7, 3, 3]
    assert graph.edge_owner.tolist() == [0, 0, 1, 2]
    assert graph.required_goal.tolist() == [False, True, True, True]


def test_rescue_trained_state_and_rms_load_for_three_humans():
    from utils.edge_stage2_spec import expand_explicit
    spec = yaml.safe_load((ROOT / 'data/cfg/multi_agent/graphs/stage2_three_agent_place_two_climb.yaml').read_text())
    trained = network(True).eval()
    expanded = network(True, agents=3, graph_spec=spec).eval()
    expanded.load_state_dict(trained.state_dict(), strict=True)
    graph = expand_explicit(spec, 2, 3, 4)
    intrinsic = torch.randn(2, 792)
    poses = torch.zeros(2, 10, 7)
    poses[..., 6] = 1
    obs = torch.cat([intrinsic, poses.flatten(1), semantic_graph_packet(graph, 2)], -1)
    source_rms = SceneRunningMeanStd([223, 30, 1], [2, 4, 2],
        normalized_sizes=[223, 30, 0], kinematic_size=7, extra_passthrough_size=20).eval()
    expanded_rms = SceneRunningMeanStd([223, 30, 1], [3, 4, 3],
        normalized_sizes=[223, 30, 0], kinematic_size=7, extra_passthrough_size=20).eval()
    expanded_rms.load_state_dict(source_rms.state_dict(), strict=True)
    with torch.no_grad():
        mu, sigma = expanded.eval_actor(expanded_rms(obs))
    assert obs.shape == (2, 882)
    assert mu.shape == sigma.shape == (6, 32)
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()


def test_rescue_contract_and_conditioned_amp_rejection():
    validate_unified_env(ENV)
    validate_relation_config(ENV['relationReward'])
    validate_sampler(ENV['relationGraph'],2,4)
    assert compile_stage1_graph(ENV['relationGraph'],2,4).num_objects == 4
    meta = checkpoint_metadata(ENV['relationReward'])
    assert (meta['packet_version'],meta['graph_record_width']) == (3,5)
    for field,value in (('ampTaskConditioning',True),('numAMPObsSteps',9)):
        bad = deepcopy(ENV); bad[field] = value
        with pytest.raises(ValueError,match='unconditioned ten-frame'):
            validate_unified_env(bad)
    bad = deepcopy(ENV)
    bad['relationReward']['stage1_variant'] = 'scenario_stage2_unified'
    with pytest.raises(ValueError,match='own reward variant'):
        validate_unified_env(bad)
    with pytest.raises(ValueError,match='matching sampler'):
        validate_sampler(ENV['relationGraph'],2,3)


def test_rescue_sampling_and_canonical_cooperation():
    graph = sample_graph(1000,ENV['relationGraph'],generator=torch.Generator().manual_seed(29))
    fractions = torch.bincount(classify_family(graph),minlength=4).float()/1000
    torch.testing.assert_close(fractions,torch.tensor([.1,.3,.3,.3]),atol=.04,rtol=0)
    unified = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_stage2_unified.yaml').read_text())['env']
    for preset in ('place_climb','place_sit','place_stack'):
        for swapped in (False,True):
            a = sample_graph(8,ENV['relationGraph'],preset=preset,role_swap=swapped)
            b = sample_graph(8,unified['relationGraph'],preset=preset,role_swap=swapped)
            for key in ('edge_src','edge_dst','edge_owner','edge_relation','edge_valid','required_goal'):
                torch.testing.assert_close(getattr(a,key),getattr(b,key))
    graph = sample_graph(2000,ENV['relationGraph'],preset='independent',generator=torch.Generator().manual_seed(5))
    templates = classify_templates(graph,torch.arange(2000).repeat_interleave(2),torch.arange(2).repeat(2000),True)
    torch.testing.assert_close(torch.bincount(templates,minlength=5).float()/4000,
                              torch.tensor([.05,.05,.5,.3,.1]),atol=.025,rtol=0)


def test_three_to_four_object_transfer_identity_gradients_and_rejections():
    torch.manual_seed(7); torch.set_num_threads(1)
    source = network(False,3)
    reward = deepcopy(SOURCE['relationReward'])
    reward['stage1_variant'] = 'scenario_independent_stage1_self_sum_rescue'
    checkpoint = {'model':Wrapper(source).state_dict(),
                  'relation_metadata':checkpoint_metadata(reward),'epoch':7000}
    # Entity counts do not add or reshape actor/critic parameters.
    reference = network(False,4)
    Wrapper(reference).load_state_dict(checkpoint['model'],strict=True)
    target = network(True,4)
    report = transfer_stage1_weights(Wrapper(target),checkpoint)
    assert (len(report['copied_tensors']),len(report['new_tensors'])) == (169,8)
    assert all(not p.requires_grad for p in target.actor_encoder.parameters())
    assert target._disc_mlp[0].weight.shape == (1024,1290)
    reference.eval(); target.eval()
    for preset in ('place_climb','place_sit','place_stack','independent'):
        obs = observation(sample_graph(2,ENV['relationGraph'],preset=preset))
        for a,b in zip(reference.eval_actor(obs),target.eval_actor(obs)):
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(reference.eval_critic(obs),target.eval_critic(obs),atol=1e-5,rtol=1e-5)
    frozen = {k:v.clone() for k,v in target.actor_encoder.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in target.parameters() if p.requires_grad),lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(); target.eval_actor(obs)[0].square().mean().backward(); optimizer.step()
    assert target.coordination.grounding[0].weight.grad.abs().sum() > 0
    for k,v in frozen.items():
        torch.testing.assert_close(target.actor_encoder.state_dict()[k],v,atol=0,rtol=0)
    bad = deepcopy(checkpoint)
    bad['relation_metadata']['relation_reward_config']['stage1_variant'] = 'scenario_independent_stage1_unified'
    with pytest.raises(ValueError,match='matching Stage-1'):
        transfer_stage1_weights(Wrapper(network(True)),bad)
    bad = deepcopy(checkpoint)
    bad['model']['a2c_network._disc_mlp.0.weight'] = torch.zeros(1024,1320)
    with pytest.raises(ValueError,match='tensor shape mismatch'):
        transfer_stage1_weights(Wrapper(network(True)),bad)


def test_stage1_only_evaluation_preserves_normalized_actions_without_training():
    torch.manual_seed(31); torch.set_num_threads(1)
    def actor_rms(objects):
        return SceneRunningMeanStd([223,30,1],[2,objects,2],
            normalized_sizes=[223,30,0],kinematic_size=7,extra_passthrough_size=20)
    source = network(False,3)
    source_rms = actor_rms(3)
    source_amp_rms = RunningMeanStd((129,))
    with torch.no_grad():
        for rms in (*source_rms.running_mean_std,source_amp_rms):
            rms.running_mean.uniform_(-.5,.5)
            rms.running_var.uniform_(.5,2.)
            rms.count.fill_(10000)
    reward = deepcopy(SOURCE['relationReward'])
    reward['stage1_variant'] = 'scenario_independent_stage1_self_sum_rescue'
    checkpoint = {'model':Wrapper(source).state_dict(),
        'relation_metadata':checkpoint_metadata(reward),'epoch':7000,
        'running_mean_std':source_rms.state_dict(),
        'amp_input_mean_std':source_amp_rms.state_dict()}
    reference = network(False,4); reference.eval()
    Wrapper(reference).load_state_dict(checkpoint['model'],strict=True)
    reference_rms = actor_rms(4); reference_rms.eval()
    reference_rms.load_state_dict(checkpoint['running_mean_std'])
    target = Wrapper(network(True)); target_rms = actor_rms(4)
    target_amp_rms = RunningMeanStd((129,))
    report = load_stage1_for_evaluation(target,checkpoint,target_rms,target_amp_rms)
    assert report['stage2_training_steps'] == 0
    assert report['context_action_weight_max_abs'] == 0
    assert not target.training and not target_rms.training and not target_amp_rms.training
    assert all(not p.requires_grad for p in target.parameters())
    original = {k:v.clone() for k,v in target.state_dict().items()}
    original_rms = {k:v.clone() for k,v in target_rms.state_dict().items()}
    for preset in ('place_climb','place_sit','place_stack','independent'):
        obs = observation(sample_graph(4,ENV['relationGraph'],preset=preset))
        with torch.no_grad():
            expected = reference.eval_actor(reference_rms(obs))
            actual = target.a2c_network.eval_actor(target_rms(obs))
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    for key,value in target.state_dict().items():
        torch.testing.assert_close(value,original[key],atol=0,rtol=0)
    for key,value in target_rms.state_dict().items():
        torch.testing.assert_close(value,original_rms[key],atol=0,rtol=0)
    for key,value in target_amp_rms.state_dict().items():
        torch.testing.assert_close(value,checkpoint['amp_input_mean_std'][key],atol=0,rtol=0)
    bad = dict(checkpoint); bad.pop('running_mean_std')
    with pytest.raises(ValueError,match='actor observation RMS'):
        load_stage1_for_evaluation(Wrapper(network(True)),bad,actor_rms(4))
    bad = dict(checkpoint); bad.pop('amp_input_mean_std')
    with pytest.raises(ValueError,match='AMP observation RMS'):
        load_stage1_for_evaluation(Wrapper(network(True)),bad,actor_rms(4),target_amp_rms)
    with pytest.raises(ValueError,match='Stage-2 model'):
        load_stage1_for_evaluation(Wrapper(source),checkpoint,actor_rms(3))


@pytest.mark.parametrize('preset', ('place_climb','place_sit','place_stack','independent'))
@pytest.mark.parametrize('swapped', (False, True))
def test_active_coordination_preserves_human_rows_and_grounded_edges(preset, swapped):
    """Exercise learned context, not the transfer's initially zero context columns."""
    torch.manual_seed(41); torch.set_num_threads(1)
    target = network(True); target.eval()
    target.actor_encoder.requires_grad_(False)
    with torch.no_grad():
        target.action_head[0][0].weight[:,64:].normal_(std=.1)
    graph = sample_graph(4,ENV['relationGraph'],preset=preset,role_swap=swapped)
    obs = observation(graph)
    original_forward = target.actor_encoder.forward
    captured = {}
    def force_order(order):
        def forward(value, **kwargs):
            humans,nodes = original_forward(value,token_order=order,**kwargs)
            captured['humans'] = humans.detach().clone()
            captured['nodes'] = nodes.detach().clone()
            return humans,nodes
        return forward
    def capture_grounding(module, inputs):
        captured['grounding'] = inputs[0].detach().clone()
    handle = target.coordination.grounding.register_forward_pre_hook(capture_grounding)
    identity = torch.arange(8)
    permutation = torch.tensor([6,4,1,3,7,0,5,2])
    try:
        with patch.object(target.actor_encoder,'forward',side_effect=force_order(identity)):
            expected_mu,expected_sigma = target.eval_actor(obs)
        nodes = captured['nodes']; humans = captured['humans']
        torch.testing.assert_close(humans,nodes[:,:2],atol=0,rtol=0)
        valid,src,dst,_,_ = parse_semantic_packet(obs[:,-20:])
        rows = torch.arange(4)[:,None]
        torch.testing.assert_close(captured['grounding'][...,64:128][valid],
                                   nodes[rows,src][valid],atol=0,rtol=0)
        torch.testing.assert_close(captured['grounding'][...,128:][valid],
                                   nodes[rows,dst][valid],atol=0,rtol=0)
        context = target.coordination(humans,nodes,obs[:,-20:],
            target.actor_encoder.edge_encoder,target.actor_encoder.entity_types)
        reversed_context = target.coordination(humans.flip(1),nodes,obs[:,-20:],
            target.actor_encoder.edge_encoder,target.actor_encoder.entity_types)
        torch.testing.assert_close(reversed_context,context.flip(1),atol=1e-6,rtol=1e-6)
        zero_context_mu = target.action_head(torch.cat((humans,torch.zeros_like(context)),-1))
        assert (expected_mu.reshape(4,2,32)-zero_context_mu).abs().max()>1e-4
        with patch.object(target.actor_encoder,'forward',side_effect=force_order(permutation)):
            permuted_mu,permuted_sigma = target.eval_actor(obs)
        torch.testing.assert_close(captured['humans'],humans,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(captured['nodes'],nodes,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(permuted_mu,expected_mu,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(permuted_sigma,expected_sigma,atol=0,rtol=0)
        weights = torch.randn_like(expected_mu)
        (expected_mu*weights).sum().backward()
        gradients = {name:p.grad.clone() for name,p in target.named_parameters() if p.grad is not None}
        target.zero_grad()
        (permuted_mu*weights).sum().backward()
        for name,p in target.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad,gradients[name],atol=5e-5,rtol=1e-3)
        edge_order = torch.tensor([[2,0,3,1],[1,3,0,2],[3,2,1,0],[0,2,1,3]])
        shuffled = permute_graph(graph,edge_order)
        edge_obs = torch.cat((obs[:,:-20],semantic_graph_packet(shuffled,4)),-1)
        with patch.object(target.actor_encoder,'forward',side_effect=force_order(identity)):
            edge_mu,edge_sigma = target.eval_actor(edge_obs)
        torch.testing.assert_close(edge_mu,expected_mu,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(edge_sigma,expected_sigma,atol=0,rtol=0)
    finally:
        handle.remove()


def test_human_relabeling_relabels_action_rows_with_active_coordination():
    torch.manual_seed(57); torch.set_num_threads(1)
    target = network(True); target.eval()
    with torch.no_grad():
        target.action_head[0][0].weight[:,64:].normal_(std=.1)
    order = torch.tensor([1,0,3,2,5,4,7,6])
    for preset in ('place_climb','place_sit','place_stack','independent'):
        graph = sample_graph(4,ENV['relationGraph'],preset=preset)
        obs = observation(graph)
        swapped = replace(graph,edge_src=order[graph.edge_src],
            edge_dst=order[graph.edge_dst],edge_owner=1-graph.edge_owner)
        swapped_obs = torch.cat((
            obs[:,:446].reshape(4,2,223).flip(1).flatten(1),
            obs[:,446:566].reshape(4,4,30)[:,[1,0,3,2]].flatten(1),
            obs[:,566:568].flip(1),
            obs[:,568:624].reshape(4,8,7)[:,order].flatten(1),
            semantic_graph_packet(swapped,4)), -1)
        with torch.no_grad():
            mu,sigma = target.eval_actor(obs)
            swapped_mu,swapped_sigma = target.eval_actor(swapped_obs)
        torch.testing.assert_close(swapped_mu.reshape(4,2,32),mu.reshape(4,2,32).flip(1),
                                   atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(swapped_sigma.reshape(4,2,32),sigma.reshape(4,2,32).flip(1),
                                   atol=0,rtol=0)
