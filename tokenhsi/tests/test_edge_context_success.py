"""Acceptance tests for raw PRE/TERM context and edge-local current saturation."""
import copy
import io
from pathlib import Path
import pytest
import torch
from torch import nn
import yaml
from utils.edge_context_spec import (CONTEXT_MODE, HOLDING, AT, compile_edge_context_graph,
                                     context_suffix_size, validate_edge_context_config)
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from env.tasks.multi_agent.edge_context_reward import (edge_context, own_success, edge_context_reward,
    EdgeContextRuntime, evaluate_edge_geometry, goal_success, scene_success)
from learning.multi_agent.amp_network_builder_ma import RelationEncoder, AMPMultiAgentBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd

ROOT = Path(__file__).parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_distance_edge_context_success.yaml').read_text())['env']['relationReward']


def specs(e=4):
    edges = []
    for i in range(e):
        h = i % 2 == 0; a = i % 2
        edges.append(dict(id=str(i), owner=a, src=('H_' if h else 'O_') + str(a),
            dst=('O_' if h else 'G_') + str(a), relation='HOLDING' if h else 'AT', pre=[],
            term=None, required_goal=not h))
    if e >= 2:
        edges[0]['term'] = '1'; edges[1]['pre'] = ['0']
    return {'edges': edges}


def reward(phi, progress, z=None, graph=None):
    g = graph or compile_edge_context_graph(None, 1, 1)
    phi = torch.tensor(phi, dtype=torch.float).reshape(1, -1)
    p = torch.tensor(progress, dtype=torch.float).reshape_as(phi)
    z = torch.zeros_like(phi) if z is None else torch.tensor(z).reshape_as(phi)
    return edge_context_reward(phi, p, own_success(phi, z, g), g, CFG)


@pytest.mark.parametrize('phi,z,p,expected', [
    ([.8,.8],[0,0],[.6,.6],[.28,.28]),
    ([.9,0],[0,0],[0,0],[.6,0]),
    ([0,.9],[0,.001],[0,0],[.6,.6]),
    ([1,1],[0,0],[0,0],[.6,.6]),
    ([.8999,.8999],[0,0],[0,0],[.17998,.17998]),
    ([0,.9],[0,.00101],[0,0],[0,.18]),
    ([0,.9],[0,-.001],[0,0],[.6,.6]),
    ([0,.9],[0,-.00101],[0,0],[0,.18])])
def test_numeric_success_boundaries(phi,z,p,expected):
    result=reward(phi,p,z)
    torch.testing.assert_close(result['total'][0],torch.tensor(expected))
    assert result['agent_task_reward'].item() == pytest.approx(sum(expected))


def test_raw_context_distinct_from_strict_success_and_reward():
    g=compile_edge_context_graph(None,1,1)
    phi=torch.tensor([[.2,.95]])
    ctx=edge_context(phi,g)
    torch.testing.assert_close(ctx,torch.tensor([[[1,.95],[.2,0]]]))
    r=reward([.2,.95],[.7,.7],[0,.002])
    assert not r['reward_saturated'].any()
    torch.testing.assert_close(phi,torch.tensor([[.2,.95]]))
    s=specs(4);s['edges'][2]['pre']=['0','1'];s['edges'][0]['term']=None
    g=compile_edge_context_graph(s,2,3)
    ctx=edge_context(torch.tensor([[.8,.3,0.,0.]]),g)
    assert ctx[0,2,0] == .3
    # Holding 0 is a prerequisite of edge 1, but never gates edge 1's reward.
    a=reward([0,.8,0,0],[0,.6,0,0],graph=g)
    b=reward([1,.8,0,0],[0,.6,0,0],graph=g)
    assert a['total'][0,1] == b['total'][0,1]


def test_live_saturation_and_nonrecursive_term():
    g=compile_edge_context_graph(None,1,1);r=EdgeContextRuntime(1,g,CFG,'cpu')
    r.reset(torch.tensor([0]),torch.tensor([[1.,0.]]),torch.zeros(1,2))
    for _ in range(5):
        assert r.step(torch.tensor([[1.,0.]]),torch.zeros(1,2),torch.zeros(1,2))['total'][0,0] == .6
    assert not r.step(torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1,2))['reward_saturated'].any()
    out=r.step(torch.tensor([[0.,1.]]),torch.zeros(1,2),torch.zeros(1,2))
    assert out['agent_task_reward'].item()==pytest.approx(1.2)
    assert r.achieved.all()
    assert not r.step(torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1,2))['reward_saturated'].any()
    s=specs(3);s['edges'][1]['term']='2'
    g=compile_edge_context_graph(s,2,3)
    out=reward([0,0,1],[0,0,0],graph=g)
    assert out['reward_saturated'].tolist()==[[False,True,True]]


@pytest.mark.parametrize('e',[2,4,5,8,10])
def test_variable_edges_owners_permutation_and_padding(e):
    s=specs(e);g=compile_edge_context_graph(s,2,3)
    phi=torch.rand(3,e);p=torch.rand_like(phi);z=torch.zeros_like(phi)
    success=own_success(phi,z,g)
    r=edge_context_reward(phi,p,success,g,CFG)
    perm=torch.randperm(e);shuffled={'edges':[s['edges'][i] for i in perm.tolist()]}
    other=compile_edge_context_graph(shuffled,2,3)
    r2=edge_context_reward(phi[:,perm],p[:,perm],success[:,perm],other,CFG)
    torch.testing.assert_close(r['total'][:,perm],r2['total'])
    torch.testing.assert_close(r['agent_task_reward'],r2['agent_task_reward'])
    torch.testing.assert_close(edge_context(phi,g)[:,perm],edge_context(phi[:,perm],other))
    all_success=edge_context_reward(torch.ones_like(phi),p,torch.ones_like(success),g,CFG)
    assert all_success['agent_task_reward'].sum().item()==pytest.approx(3*e*.6)
    if e>=4:
        s['edges'][-1].update(valid=False,required_goal=False)
        g=compile_edge_context_graph(s,2,3)
        out=reward([1]*e,[1]*e,graph=g)
        assert out['total'][0,-1]==0
        assert not out['own_success'][0,-1]


def test_goal_set_not_last_edge_or_all_holdings():
    g=compile_edge_context_graph(None,2,3)
    s=torch.tensor([[False,True,False,True],[True,True,True,False]])
    assert goal_success(s,g).tolist()==[[True,True],[True,False]]
    assert scene_success(s,g).tolist()==[True,False]
    s=specs(5)
    for e in s['edges']:e['owner']=0
    g=compile_edge_context_graph(s,2,3)
    result=reward([1]*5,[0]*5,graph=g)
    torch.testing.assert_close(result['agent_task_reward'],torch.tensor([[3.,0.]]))
    # Unrelated successful edge never retires another owner's unreferenced edge.
    result=reward([0,0,1,0,0],[0]*5,graph=g)
    assert result['total'][0,0]==0


@pytest.mark.parametrize('mutation',[
    lambda s:s['edges'][0].update(src='O_0'),lambda s:s['edges'][0].update(dst='O_9'),
    lambda s:s['edges'][0].update(term='missing'),lambda s:s['edges'][0].update(term='0'),
    lambda s:s['edges'][0].update(term=['1','2']),lambda s:s['edges'][0].update(pre=['missing']),
    lambda s:s['edges'][0].update(relation='ONTOP'),lambda s:s['edges'][0].update(owner=9),
    lambda s:s['edges'][1].update(valid=False,required_goal=False),
    lambda s:s.update(edges=[]),lambda s:[e.update(valid=False,required_goal=False) for e in s['edges']]])
def test_bad_graphs_rejected(mutation):
    s=specs();mutation(s)
    with pytest.raises(ValueError):compile_edge_context_graph(s,2,3)


def test_geometry_binding_and_analytic_state():
    s=specs(2);s['edges'][0].update(src='H_1',dst='O_2');s['edges'][1].update(src='O_2',dst='G_0')
    g=compile_edge_context_graph(s,2,3)
    hands=torch.zeros(1,2,2,3);hands[:,1,:,0]=.1
    roots=torch.zeros(1,2,3);roots[:,1,0]=2.
    boxes=torch.zeros(1,3,3);goals=torch.zeros(1,2,3);goals[:,0,0]=1.
    phi,d=evaluate_edge_geometry(hands,roots,boxes,goals,g,CFG)
    torch.testing.assert_close(phi,torch.tensor([[torch.exp(torch.tensor(-.1)),torch.exp(torch.tensor(-10.))]]))
    torch.testing.assert_close(d['progress'],torch.tensor([[.4,2/3]]))


def encoder(spec=None,m=2,o=3):
    return RelationEncoder([223,30,1],m,o,16,2,2,32,
        lambda size:nn.Sequential(nn.Linear(size,16),nn.ReLU()),observation_mode='clean_scene',
        kinematic_size=7,relation_bias_mode='edge_mlp',gta_cfg={'enable':True},
        relation_reward_mode=CONTEXT_MODE,relation_graph_spec=spec)


def scene(e=4,m=2,o=3,n=3):
    width=224*m+30*o;L=2*m+o
    x=torch.randn(n,width+7*L+2*e)
    x[:,width:width+7*L].reshape(n,L,7)[...,3:7]=torch.tensor([0.,0.,0.,1.])
    x[:,-2*e:]=torch.rand(n,2*e)
    return x


def test_policy_variable_edges_strict_weights_and_reordering():
    torch.manual_seed(22);net=encoder(specs())
    with torch.no_grad():net.edge_encoder.bias_projection.normal_(std=.1)
    count=sum(p.numel() for p in net.parameters());state=copy.deepcopy(net.state_dict())
    assert not any('edge_src' in k or 'edge_relation' in k or 'rel_matrix' in k for k in state)
    x=scene();before=net(x)
    s=specs();perm=[3,1,0,2];net.set_task_graph({'edges':[s['edges'][i] for i in perm]})
    y=x.clone();y[:,-8:]=x[:,-8:].reshape(3,4,2)[:,perm].flatten(1)
    torch.testing.assert_close(net(y),before,atol=2e-6,rtol=2e-6)
    for e in [2,5,8,10]:
        net.set_task_graph(specs(e));net.load_state_dict(state,strict=True)
        assert net(scene(e)).shape==(3,2,16)
        assert count==sum(p.numel() for p in net.parameters())
    other=encoder(None,3,4);other.load_state_dict(state,strict=True)
    assert other(scene(6,3,4)).shape==(3,3,16)
    rms=SceneRunningMeanStd([223,30,1],[2,3,2],[223,30,0],7,8)
    torch.testing.assert_close(rms(x)[:,536:],x[:,536:],rtol=0,atol=0)


def test_context_gradient_stored_observation_and_pair_sum():
    torch.manual_seed(24);net=encoder(specs());x=scene();opt=torch.optim.Adam(net.parameters(),lr=.01)
    for step in range(2):
        opt.zero_grad();(net(x)*torch.randn(3,2,16)).sum().backward()
        assert net.edge_encoder.bias_projection.grad.abs().sum()>0
        for module in [net.context_fusion.context_encoder,net.context_fusion.fusion_mlp,net.edge_encoder.edge_mlp]:
            grad=module[0].weight.grad
            assert torch.isfinite(grad).all()
            assert (grad.abs().sum()>0)==bool(step)
        opt.step()
    before=net(x).detach();changed=x.clone();changed[0,-8:]=1-changed[0,-8:]
    after=net(changed).detach();torch.testing.assert_close(before[1:],after[1:]);assert not torch.allclose(before[0],after[0])
    runtime=EdgeContextRuntime(3,compile_edge_context_graph(specs(),2,3),CFG,'cpu')
    runtime.phi.uniform_();torch.testing.assert_close(net(x),before)
    sem=net.edge_encoder;types=net.entity_types
    s={'edges':[dict(id='x',owner=0,src='H_0',dst='O_0',relation='HOLDING')]}
    net.set_task_graph(s);ctx=torch.rand(3,2);zero=torch.zeros(2,2,7,7)
    single=net.context_fusion(ctx,sem,types,zero)
    s['edges'].append(dict(s['edges'][0],id='y'));net.set_task_graph(s)
    duplicate=net.context_fusion(ctx.repeat(1,2),sem,types,zero)
    torch.testing.assert_close(duplicate,2*single)
    s['edges'][1]['valid']=False;net.set_task_graph(s)
    torch.testing.assert_close(net.context_fusion(ctx.repeat(1,2),sem,types,zero),single)


def test_schema_config_and_baseline_settings():
    validate_edge_context_config(CFG)
    saved={'relation_metadata':checkpoint_metadata(CFG)}
    check_checkpoint_metadata(saved,checkpoint_metadata(CFG))
    bad=copy.deepcopy(CFG);bad['soft_gate']={'beta':30}
    with pytest.raises(ValueError):validate_edge_context_config(bad)
    old=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_distance_success.yaml').read_text())
    with pytest.raises(ValueError):check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(old['env']['relationReward'])},checkpoint_metadata(CFG))
    new=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_distance_edge_context_success.yaml').read_text())
    del new['env']['relationReward'];del new['env']['relationGraph'];del old['env']['relationReward']
    assert new==old


def full_network(m=2,o=3,spec=None):
    params=yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())['params']['network']
    builder=AMPMultiAgentBuilder();builder.load(params)
    e=len(compile_edge_context_graph(spec,m,o).ids)
    return builder.build('amp',actions_num=28,input_shape=(238*m+37*o+2*e,),amp_input_shape=(64,),
        value_size=1,num_agents=m,num_objects=o,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,
        observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,
        scene_arena_scale=5.,relation_reward_mode=CONTEXT_MODE,relation_graph_spec=spec,device='cpu')


def test_full_actor_critic_checkpoint_other_e_and_counts():
    net=full_network();x=scene(n=2)
    assert net.eval_actor(x)[0].shape==(4,28)
    assert net.eval_critic(x).shape==(4,1)
    assert net.actor_encoder.context_fusion is not net.critic_encoder.context_fusion
    assert not hasattr(net.actor_encoder,'dynamic_edge_mlp')
    weights=dict(model=net.state_dict(),relation_metadata=checkpoint_metadata(CFG))
    buffer=io.BytesIO();torch.save(weights,buffer);buffer.seek(0);loaded=torch.load(buffer,weights_only=False)
    for m,o,spec,e in [(2,3,specs(5),5),(3,4,None,6)]:
        other=full_network(m,o,spec);other.load_state_dict(loaded['model'],strict=True)
        check_checkpoint_metadata(loaded,checkpoint_metadata(CFG))
        obs=scene(e,m,o,n=2)
        assert torch.isfinite(other.eval_actor(obs)[0]).all()
        assert torch.isfinite(other.eval_critic(obs)).all()
        assert sum(p.numel() for p in net.parameters())==sum(p.numel() for p in other.parameters())


def test_task_mixin_penalties_survive_saturation(tmp_path):
    from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
    from types import SimpleNamespace
    class Task(EdgeContextTaskMixin):
        def _evaluate_relations(self,ids=None):
            return torch.ones(1,4),dict(progress=torch.zeros(1,4),z_error=torch.zeros(1,4),
                distance=torch.zeros(1,4),distance_xy=torch.zeros(1,4))
        def _assigned_box_values(self,value,env_ids=None):return value
    task=Task();task._edge_context=True;task._relation_cfg=copy.deepcopy(CFG)
    task._relation_cfg['diagnostics']['sample_envs']=0
    task._relation_graph_spec={'template':'independent_carry'}
    task.num_envs=1;task.num_agents=2;task.num_objects=3;task.device='cpu'
    task._init_relation_runtime()
    task._humanoid_root_states=torch.zeros(1,2,13);task._box_states=torch.ones(1,2,13)
    task._prev_box_pos=torch.zeros(1,2,3);task.dt=1.
    task._power_reward=True;task._power_coefficient=.1
    task.dof_force_tensor=torch.ones(1,2,1);task._dof_vel=torch.ones(1,2,1)
    task._agent_collision_penalty=True;task._agent_collision_coeff=.2;task._agent_collision_dist=.7
    task._box_vel_penalty=True;task._box_vel_pen_coeff=.3;task._box_vel_pen_thre=.5
    task.rew_buf=torch.zeros(2);task.extras={};task._reward_term_sums=torch.zeros(7);task._reward_term_count=0
    task._compute_relation_reward(lambda roots,dist:torch.ones(1,2))
    terms=task.extras['reward_terms']
    torch.testing.assert_close(terms[:,:3].sum(-1),torch.full((2,),1.2))
    assert (terms[:,3:6]<0).all()
    torch.testing.assert_close(terms[:,:-1].sum(-1),task.rew_buf)
    assert task.rew_buf.max()<1.2


def test_training_resume_graph_contract_is_separate_from_evaluation():
    from utils.edge_context_spec import task_instance,check_task_resume
    weights={'relation_metadata':checkpoint_metadata(CFG),'relation_task_instance':task_instance(specs(4),2,3)}
    check_task_resume(weights,specs(4),2,3)
    with pytest.raises(ValueError):check_task_resume(weights,specs(5),2,3)
    with pytest.raises(ValueError):check_task_resume(weights,specs(4),3,4)
    check_checkpoint_metadata(weights,checkpoint_metadata(CFG))  # evaluation may supply another graph
