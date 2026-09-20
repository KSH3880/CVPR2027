"""Acceptance of sampled graphs, real OnTop geometry and replayable policy bindings."""
import copy
from pathlib import Path
from collections import Counter
import itertools
import pytest
import torch
from torch import nn
import yaml
from utils.edge_ontop_spec import *
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
from env.tasks.multi_agent.edge_context_reward import *
from env.tasks.multi_agent.edge_ontop_reward import *
from learning.multi_agent.amp_network_builder_ma import RelationEncoder, AMPMultiAgentBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd

torch.set_num_threads(1)
ROOT=Path(__file__).parents[1]
CFG=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_distance_edge_context_ontop.yaml').read_text())['env']['relationReward']
SPEC=copy.deepcopy(DEFAULT_GRAPH)


def test_sampler_100000_resets_distribution_and_structure():
    spec=copy.deepcopy(SPEC);spec['shuffle_edge_order']=False
    g=sample_graph(100000,spec,generator=torch.Generator().manual_seed(723))
    valid=g.edge_valid;rel=g.edge_relation
    second=torch.where(valid[:,[1,3]],rel[:,[1,3]]-6,0)
    torch.testing.assert_close(torch.stack([(second==i).float().mean() for i in range(3)]),torch.tensor([.2,.5,.3]),atol=.004,rtol=0)
    torch.testing.assert_close(torch.stack([(valid.sum(-1)==i).float().mean() for i in (2,3,4)]),torch.tensor([.04,.32,.64]),atol=.004,rtol=0)
    assert (rel[:,[0,2]]==HOLDING).all() and (valid.sum(-1)>=2).all() and (valid.sum(-1)<=4).all()
    target=g.edge_dst[:,[1,3]]-2
    outcomes=Counter(tuple(x) for x in torch.cat([second,target],-1).tolist())
    assert len(outcomes)==12
    expected={ (0,0,-2,-2):.04,(0,1,-2,4):.10,(1,0,3,-2):.10,(0,2,-2,2):.06,(2,0,2,-2):.06,
              (1,1,3,4):.25,(1,2,3,2):.075,(2,1,2,4):.075,(1,2,3,0):.075,(2,1,1,4):.075,
              (2,2,2,0):.045,(2,2,1,2):.045 }
    assert sum(expected.values())==pytest.approx(1.)
    assert set(outcomes)==set(expected)
    for key,p in expected.items():assert outcomes[key]/100000==pytest.approx(p,abs=.004)
    for a in (0,1):
        other=1-a
        top=second[:,a]==2
        assert (target[top&(second[:,other]==0),a]==2).all()
        assert not (target[top,a]==a).any()
    both=(second==2).all(-1)
    assert ((target[both]==2).sum(-1)==1).all()
    assert not ((g.prereq_mask & ~valid[:,None]).any())
    # PRE + TERM intentionally contains H <-> placement; independent DAG validation accepts it.
    validate_graph(select_graph(g,torch.arange(100)))


def test_sampler_rng_partial_reset_presets_and_references():
    a=sample_graph(30,SPEC,generator=torch.Generator().manual_seed(20))
    b=sample_graph(30,SPEC,generator=torch.Generator().manual_seed(20))
    for f in fields(a):
        if isinstance(getattr(a,f.name),torch.Tensor):torch.testing.assert_close(getattr(a,f.name),getattr(b,f.name))
    ids=torch.tensor([2,7]);old=a.edge_dst.clone();copy_graph_rows(a,ids,sample_graph(2,SPEC,preset='ontop_chain'))
    keep=torch.ones(30,dtype=torch.bool);keep[ids]=False
    torch.testing.assert_close(a.edge_dst[keep],old[keep])
    for preset in PRESETS[1:]:
        g=sample_graph(3,SPEC,preset=preset);validate_graph(g)
        swapped=sample_graph(3,SPEC,preset=preset,role_swap=True);validate_graph(swapped)
    g=sample_graph(1,SPEC,preset='at_ontop');phi=torch.tensor([[.8,.3,.7,.95]])
    torch.testing.assert_close(edge_context(phi,g),torch.tensor([[[1,.3],[.8,0],[1,.95],[.3,0]]]))
    g=sample_graph(1,SPEC,preset='ontop_chain')
    assert g.prereq_mask[0,3].tolist()==[False,True,True,False]
    assert g.term_index[0].tolist()==[1,-1,3,-1]
    assert not scene_success(torch.tensor([[True,False,True,True]]),g).item()
    g=compose_graph(torch.tensor([[0,0]]),torch.tensor([[2,2]]))
    assert g.required_goal.tolist()==[[True,False,True,False]]
    assert edge_context(torch.ones(1,4),g)[0,0].tolist()==[1,0]


def fixture():
    source=torch.zeros(2,13,dtype=torch.float64);source[:,6]=1;source[:,2]=.6
    support=source.clone();support[:,2]=.2
    size=torch.full((2,3),.4,dtype=torch.float64)
    return source,support,size


def test_geometry_center_extent_and_signed_gap():
    source,support,size=fixture()
    delta,d=ontop_geometry(source,support,size,size)
    torch.testing.assert_close(delta,torch.zeros_like(delta),atol=1e-15,rtol=0)
    assert torch.exp(-10*delta.square().sum(-1)).tolist()==[1.,1.]
    source[:,2]=.2
    assert (ontop_geometry(source,support,size,size)[1]['signed_gap']<-.39).all()
    source,support,size=fixture();size[0,2]=.2
    assert ontop_geometry(source,support,size,torch.full_like(size,.4))[1]['target'][0,2]==pytest.approx(.5)
    for gap in [.001,-.001,.00101,-.00101]:
        source,support,size=fixture();source[:,2]+=(gap+1e-12*(1 if gap>0 else -1))
        delta,d=ontop_geometry(source,support,size,size)
        assert d['signed_gap'][0]==pytest.approx(gap,abs=2e-12)
    q=torch.randn(100,4,dtype=torch.float64);q/=q.norm(dim=-1,keepdim=True)
    sizes=torch.rand(100,3,dtype=torch.float64)
    corners=torch.tensor(list(itertools.product([-1.,1.],repeat=3)),dtype=q.dtype)[None]*sizes[:,None]/2
    xyz=q[:,:3][:,None];w=q[:,3:][:,None]
    rotated=corners+2*(w*torch.cross(xyz.expand_as(corners),corners,dim=-1)+torch.cross(xyz.expand_as(corners),torch.cross(xyz.expand_as(corners),corners,dim=-1),dim=-1))
    torch.testing.assert_close(vertical_extent(q,sizes/2),rotated[...,2].abs().amax(-1))


def test_geometry_binding_translation_yaw_progress_and_success():
    g=sample_graph(1,SPEC,preset='independent_ontop');objects=torch.zeros(1,3,13);objects[...,6]=1
    objects[0,0,2]=.6;objects[0,2,2]=.2;sizes=torch.full((1,3,3),.4)
    hands=torch.zeros(1,2,2,3);roots=torch.zeros(1,2,3);goals=torch.zeros(1,2,3)
    phi,d=evaluate_ontop_edges(hands,roots,objects,sizes,goals,g,CFG)
    assert phi[0,1]==1 and own_success(phi,d['z_error'],g)[0,1]
    for z in [.001,-.001,.00101,-.00101]:
        # Exact boundary is covered directly in own_success (no geometry subtraction rounding).
        zz=d['z_error'].clone();zz[0,1]=z
        assert own_success(phi,zz,g)[0,1].item()==(abs(z)<=.001)
    for translation in [torch.tensor([2.,-3.,0.]),torch.tensor([0.,0.,1.])]:
        other=objects.clone();other[...,:3]+=translation
        p,dd=evaluate_ontop_edges(hands+translation,roots+translation,other,sizes,goals+translation,g,CFG)
        torch.testing.assert_close(p,phi);torch.testing.assert_close(dd['progress'],d['progress'])
    other=objects.clone();other[:,2,0]+=1
    p,dd=evaluate_ontop_edges(hands,roots,other,sizes,goals,g,CFG)
    assert p[0,1]<phi[0,1] and dd['target'][0,1,0]==1
    other=objects.clone();other[:,0,2]+=1
    p,dd=evaluate_ontop_edges(hands,roots,other,sizes,goals,g,CFG)
    torch.testing.assert_close(dd['progress'][0,1],d['progress'][0,1])


def test_reward_mixing_all_graphs_saturation_and_permutation():
    g=sample_graph(200,SPEC,generator=torch.Generator().manual_seed(8))
    phi=torch.rand(200,4);progress=torch.rand_like(phi);z=torch.zeros_like(phi)
    runtime=OnTopContextRuntime(200,g,CFG,'cpu');r=runtime.step(phi,progress,z)
    torch.testing.assert_close(r['agent_task_reward'],mix_task_reward(r['local_task_reward']))
    torch.testing.assert_close(r['agent_task_reward'].sum(-1),r['local_task_reward'].sum(-1))
    torch.testing.assert_close(mix_task_reward(torch.tensor([[.6,1.2],[6.,12.]])),torch.tensor([[.66,1.14],[6.6,11.4]]))
    permutation=torch.rand(200,4).argsort(-1);other=permute_graph(g,permutation)
    r2=edge_context_reward(phi.gather(1,permutation),progress.gather(1,permutation),own_success(phi,z,g).gather(1,permutation),other,CFG)
    torch.testing.assert_close(r['local_task_reward'],r2['agent_task_reward'])
    torch.testing.assert_close(edge_context(phi,g).gather(1,permutation[...,None].expand(-1,-1,2)),edge_context(phi.gather(1,permutation),other))
    out=runtime.step(torch.ones_like(phi),progress,z)
    for key in ('state_component','progress_component','success_component'):
        torch.testing.assert_close(out[key],g.edge_valid.float()*.2)
    assert not runtime.step(torch.zeros_like(phi),progress,z)['reward_saturated'].any()
    g=sample_graph(1,SPEC,preset='at_ontop');r=OnTopContextRuntime(1,g,CFG,'cpu')
    phi=torch.tensor([[0.,0.,0.,.95]]);z=torch.tensor([[0.,0.,0.,.002]])
    out=r.step(phi,torch.zeros_like(phi),z)
    assert edge_context(phi,g)[0,2,1]==.95 and not out['term_success'].any()
    phi[0,3]=1;z.zero_();out=r.step(phi,torch.zeros_like(phi),z)
    assert out['total'][0,2]==.6 and out['phi_raw'][0,2]==0
    # PRE does not gate the top reward (AT phi=0).
    assert out['own_success'][0,3]


def encoder(spec=SPEC,m=2,o=3):
    return RelationEncoder([223,30,1],m,o,16,2,2,32,lambda sz:nn.Sequential(nn.Linear(sz,16),nn.ReLU()),
        observation_mode='clean_scene',kinematic_size=7,relation_bias_mode='edge_mlp',gta_cfg={'enable':True},
        relation_reward_mode=ONTOP_CONTEXT_MODE,relation_graph_spec=spec)


def scene(g,phi=None):
    n=g.edge_src.shape[0];m=g.num_agents;o=g.num_objects;e=len(g.ids);width=224*m+30*o;l=2*m+o
    x=torch.randn(n,width+7*l+7*e)
    x[:,width:width+7*l].reshape(n,l,7)[...,3:7]=torch.tensor([0.,0.,0.,1.])
    phi=torch.rand(n,e) if phi is None else phi
    x[:,-7*e:]=graph_packet(g,edge_context(phi,g))
    return x


def test_policy_stored_bindings_padding_permutation_gradients_and_rms():
    torch.manual_seed(7);g=sample_graph(20,SPEC);net=encoder();x=scene(g)
    with torch.no_grad():net.edge_encoder.bias_projection.normal_(std=.1)
    before=net(x).detach().clone()
    copy_graph_rows(g,torch.arange(20),sample_graph(20,SPEC,preset='ontop_chain'))
    torch.testing.assert_close(net(x),before) # reset cannot rewrite saved bindings
    perm=torch.rand(20,4).argsort(-1);y=x.clone();y[:,-28:]=x[:,-28:].reshape(20,4,7).gather(1,perm[...,None].expand(-1,-1,7)).flatten(1)
    torch.testing.assert_close(net(y),net(x),atol=2e-6,rtol=2e-6)
    order=torch.randperm(20);torch.testing.assert_close(net(x[order]),net(x)[order],atol=2e-6,rtol=2e-6)
    rms=SceneRunningMeanStd([223,30,1],[2,3,2],[223,30,0],7,28)
    torch.testing.assert_close(rms(x)[:,536:],x[:,536:],rtol=0,atol=0)
    net(x).square().sum().backward()
    for module in [net.context_fusion.context_encoder,net.context_fusion.fusion_mlp,net.edge_encoder.edge_mlp,net.tokenizers]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters())>0
    assert net.edge_encoder.relation_embed.weight.grad[ON_TOP].abs().sum()>0
    # Same physical scene and context with changed target changes semantic scatter.
    fusion=net.context_fusion;sem=net.edge_encoder;zero=torch.zeros(2,2,7,7)
    p=x[:1,-28:].clone();valid,src,dst,rel,owner,ctx=parse_packet(p)
    fused=fusion(p,sem,net.entity_types,zero)
    idx=valid[0].nonzero()[0,0];p.reshape(1,4,7)[0,idx,2]=(dst[0,idx]+1)%7
    assert not torch.allclose(fused,fusion(p,sem,net.entity_types,zero))


def test_capacity_counts_actor_critic_and_strict_checkpoint():
    params=yaml.safe_load((ROOT/'data/cfg/train/rlg/amp_ma_carry_relation.yaml').read_text())['params']['network']
    def build(m,o,spec):
        builder=AMPMultiAgentBuilder();builder.load(params);e=len(compile_ontop_graph(spec,m,o).ids)
        return builder.build('amp',actions_num=28,input_shape=(238*m+37*o+7*e,),amp_input_shape=(64,),
            value_size=1,num_agents=m,num_objects=o,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,
            observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=5.,
            relation_reward_mode=ONTOP_CONTEXT_MODE,relation_graph_spec=spec,device='cpu')
    model=build(2,3,SPEC);x=scene(sample_graph(4,SPEC))
    assert model.eval_actor(x)[0].shape==(8,28) and model.eval_critic(x).shape==(8,1)
    loss=model.eval_actor(x)[0].square().mean()+model.eval_critic(x).square().mean();loss.backward()
    for enc in [model.actor_encoder,model.critic_encoder]:
        assert torch.isfinite(enc.edge_encoder.bias_projection.grad).all()
    weights=copy.deepcopy(model.state_dict())
    for m,o,e in [(2,3,2),(3,4,6),(4,5,10)]:
        from utils.edge_context_spec import compile_edge_context_graph
        base=compile_edge_context_graph(None,m,o)
        edges=[]
        for i in range(e):
            a=i%m
            edges.append(dict(id=str(i),owner=a,src='H_'+str(a),dst='O_'+str(a),relation='HOLDING',required_goal=True))
        spec={'edges':edges};g=expand_graph(compile_ontop_graph(spec,m,o),2)
        other=build(m,o,spec);other.load_state_dict(weights,strict=True)
        obs=scene(g);assert torch.isfinite(other.eval_actor(obs)[0]).all() and torch.isfinite(other.eval_critic(obs)).all()
        assert sum(p.numel() for p in other.parameters())==sum(p.numel() for p in model.parameters())
    validate_relation_config(CFG);meta=checkpoint_metadata(CFG);check_checkpoint_metadata({'relation_metadata':meta},meta)
    for key in ('schema_version','graph_record_width','packet_version','context_dim_per_edge'):
        bad=copy.deepcopy(meta);bad[key]=99
        with pytest.raises(ValueError):check_checkpoint_metadata({'relation_metadata':bad},meta)
    bad=copy.deepcopy(CFG);bad['sharing']['other_weight']=.2
    with pytest.raises(ValueError):validate_relation_config(bad)
    bad=copy.deepcopy(SPEC);bad['second_edge_probabilities']['AT']=float('nan')
    with pytest.raises(ValueError):validate_sampler(bad)
    with pytest.raises(ValueError):sample_graph(1,SPEC,preset='unknown')
    with pytest.raises(ValueError):validate_sampler(SPEC,3,4)


def test_validator_rejects_support_cycles_and_holding_only_targets():
    from dataclasses import replace
    for second,target in [([[2,2]],[[1,0]]),([[2,0]],[[1,2]]),([[2,2]],[[2,2]]),([[2,0]],[[0,2]])]:
        with pytest.raises(ValueError):validate_graph(compose_graph(torch.tensor(second),torch.tensor(target)))
    g=sample_graph(1,SPEC,preset='at_ontop');g.prereq_mask[0,0,1]=True
    with pytest.raises(ValueError):validate_graph(g)


def test_world_yaw_and_logical_physical_assignment_invariance():
    g=sample_graph(2,SPEC,preset='ontop_chain')
    objects=torch.randn(2,3,13);objects[...,3:7]=torch.tensor([0.,0.,0.,1.])
    sizes=torch.rand(2,3,3)*.3+.1
    roots=torch.randn(2,2,3);hands=torch.randn(2,2,2,3);goals=torch.randn(2,2,3)
    phi,diag=evaluate_ontop_edges(hands,roots,objects,sizes,goals,g,CFG)
    angle=torch.tensor(.7);c=angle.cos();s=angle.sin()
    rotation=torch.tensor([[c,-s,0],[s,c,0],[0,0,1.]])
    other=objects.clone();other[...,:3]=objects[...,:3]@rotation.T
    other[...,3:7]=torch.tensor([0.,0.,torch.sin(angle/2),torch.cos(angle/2)])
    p,d=evaluate_ontop_edges(hands@rotation.T,roots@rotation.T,other,sizes,goals@rotation.T,g,CFG)
    torch.testing.assert_close(phi,p,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(diag['progress'],d['progress'])
    physical_order=torch.tensor([[2,0,1],[1,2,0]])
    physical=torch.zeros_like(objects);physical_size=torch.zeros_like(sizes)
    batch=torch.arange(2)[:,None]
    physical[batch,physical_order]=objects;physical_size[batch,physical_order]=sizes
    p,d=evaluate_ontop_edges(hands,roots,physical[batch,physical_order],physical_size[batch,physical_order],goals,g,CFG)
    torch.testing.assert_close(phi,p)
    torch.testing.assert_close(diag['target'],d['target'])


def test_masked_pair_sum_and_padding_bias():
    from learning.multi_agent.edge_context_encoder import PackedEdgeContextFusion
    net=encoder();fusion=net.context_fusion;sem=net.edge_encoder
    with torch.no_grad():sem.bias_projection.normal_()
    zero=torch.zeros(2,2,7,7)
    edge=torch.tensor([[1.,0.,2.,6.,0.,.8,.4]])
    a=fusion(edge,sem,net.entity_types,zero)
    b=fusion(edge.repeat(1,2),sem,net.entity_types,zero)
    torch.testing.assert_close(b,2*a)
    padded=torch.cat([edge,torch.zeros_like(edge)],-1)
    torch.testing.assert_close(fusion(padded,sem,net.entity_types,zero),a)


def test_relation_metric_denominator_does_not_count_absent_edges():
    from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
    class Task(EdgeContextTaskMixin):pass
    task=Task();task._edge_context=True;task._edge_metric_steps=4
    task._edge_metric_fields=['phi_raw']
    task._edge_metric_sums={'edge/ontop':torch.tensor([1.6])}
    task._edge_metric_denominators={'edge/ontop':torch.tensor(2)}
    task._edge_episode_sums=torch.zeros(6)
    assert task.consume_relation_diagnostics()['edge/ontop/phi_raw']==.8
