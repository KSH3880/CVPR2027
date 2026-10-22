from copy import deepcopy
from pathlib import Path
import torch
import yaml
import pytest
from utils.stage1_push_spec import compose_graph, validate_env, sample_asset_sizes, original_spec
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet
from env.tasks.multi_agent.stage1_push_reward import Stage1PushRuntime
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from utils.unified_training import append_family, preserve_amp_labels, sample_family_matched

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'data/cfg/multi_agent/stage1_base_push.yaml').read_text())['env']


def test_all_old_graphs_and_task_ids_unchanged():
    ids=torch.cartesian_prod(torch.arange(1,5),torch.arange(1,5))
    old,new=compose_canonical_graph(ids),compose_graph(ids)
    for name in ('edge_src','edge_dst','edge_relation','edge_owner','edge_valid','required_goal'):
        assert torch.equal(getattr(old,name),getattr(new,name))
    assert torch.equal(task_graph_packet(old,len(ids)),task_graph_packet(new,len(ids)))
    graph=compose_graph(torch.tensor([[1,5],[2,5],[3,5],[4,5],[5,5]]))
    packet=task_graph_packet(graph,5).reshape(5,2,5)
    assert packet[:,0,1].tolist()==[0,1,2,3,4]
    assert packet[:,1,1].tolist()==[4]*5
    assert packet[-1,:,3:].tolist()==[[2,6],[3,7]]


def test_original_reward_and_rsi_configuration_preserved():
    old=yaml.safe_load((ROOT/'data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    assert CFG['relationReward']==old['relationReward']
    for key,row in old['templateRsi'].items():
        assert CFG['templateRsi'][key]==row+[0.]
    assert original_spec(CFG['relationGraph'])==old['relationGraph']
    validate_env(CFG)


def test_assets_and_sampling_keep_four_current_push_boxes():
    torch.manual_seed(7)
    sizes,prob=sample_asset_sizes(2048,CFG,'cpu')
    assert sizes.shape==(2048,4,3)
    assert (sizes[...,1]==2).all() and (sizes[...,2]==1.3).all()
    assert sizes[...,0].min()>=1 and sizes[...,0].max()<=1.5
    assert (prob[...,5]==1).all() and not prob[...,:5].any()
    mixed=deepcopy(CFG)
    mixed['relationGraph']['template_probabilities']=dict(zip(('HOLDING','SIT','CLIMB','HOLDING_AT','HOLDING_ON_TOP','PUSH_AT'),(0,.1,.1,.2,.2,.4)))
    validate_env(mixed)
    sizes,prob=sample_asset_sizes(2048,mixed,'cpu')
    assert torch.isfinite(prob).all() and (prob.sum(-1)-1).abs().max()<1e-6
    assert (prob[...,1:]>0).flatten(0,1).any(0).all()


def geometry(graph):
    n=graph.edge_src.shape[0]
    roots=torch.zeros(n,2,3)
    objects=torch.zeros(n,4,13);objects[...,6]=1.;objects[...,2]=.65
    sizes=torch.tensor([1.5,2.,1.3]).expand(n,4,3)
    goals=objects[:,:2,:3].clone()
    phi,diag=evaluate_interaction_edges(torch.zeros(n,2,2,3),torch.zeros(n,2,2,3),roots,
                                      objects,sizes,goals,graph,CFG['relationReward'],.94)
    return phi,diag,objects,sizes,goals,roots


def test_legacy_rewards_bitwise_equal_for_all_sixteen_old_task_pairs():
    graph=compose_graph(torch.cartesian_prod(torch.arange(1,5),torch.arange(1,5)))
    phi,diag,*_=geometry(graph)
    old=Stage1ContextRuntime(16,graph,CFG['relationReward'],'cpu')
    new=Stage1PushRuntime(16,graph,CFG['relationReward'],'cpu',contact_weights=[.4,.2,.2])
    a=old.step(phi,diag['progress'],diag['z_error'],diag['feet_height_error'],diag['region_error'])
    b=new.step(phi,diag['progress'],diag['z_error'],diag['feet_height_error'],diag['region_error'])
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


def test_at_uses_original_twenty_cm_one_mm_and_has_no_tilt_gate():
    graph=compose_graph(torch.full((5,2),5))
    phi,diag,objects,sizes,goals,roots=geometry(graph)
    goals[:,0,0]=torch.tensor([.199,.201,.0,.0,.0])
    goals[:,0,2]+=torch.tensor([0.,0.,.0009,.0011,0.])
    objects[4,0,3]=torch.sin(torch.tensor(.4));objects[4,0,6]=torch.cos(torch.tensor(.4))
    phi,diag=evaluate_interaction_edges(torch.zeros(5,2,2,3),torch.zeros(5,2,2,3),roots,objects,sizes,goals,graph,CFG['relationReward'],.94)
    runtime=Stage1PushRuntime(5,graph,CFG['relationReward'],'cpu',contact_weights=[.4,.2,.2])
    result=runtime.step(phi,diag['progress'],diag['z_error'],diag['feet_height_error'],diag['region_error'])
    assert runtime.done[:,0].tolist()==[True,False,True,False,True]
    # Completed CONTACT(.8) + original AT(.6), no teammate contribution.
    assert result['agent_task_reward'][0,0].item()==pytest.approx(1.4)


def test_contact_success_requires_force_flag_and_resets_history():
    graph=compose_graph(torch.full((1,2),5))
    phi,diag,*_=geometry(graph)
    phi[:,::2]=1.;phi[:,1::2]=0.
    runtime=Stage1PushRuntime(1,graph,CFG['relationReward'],'cpu',contact_weights=[.4,.2,.2])
    args=(phi,diag['progress'],torch.zeros_like(phi),diag['feet_height_error'],diag['region_error'])
    assert not runtime.step(*args)['own_success'][:,::2].any()
    runtime.contact[:]=True
    assert runtime.step(*args)['own_success'][:,::2].all()
    runtime.reset(torch.tensor([0]),phi,args[2],args[3],args[4])
    assert not runtime.contact.any()


def test_four_amp_families_keep_original_ids_and_add_push():
    raw=append_family(torch.randn(4,10,129),torch.arange(4)[:,None].expand(-1,10),4).flatten(1)
    assert raw.shape==(4,1330)
    normalized=preserve_amp_labels(raw,raw*2,10,4).reshape(4,10,-1)
    assert torch.equal(normalized[...,-4:],raw.reshape(4,10,-1)[...,-4:])
    reference=raw.flip(0)
    matched=sample_family_matched(raw,reference,10,num_families=4)
    assert torch.equal(matched.reshape(4,10,-1)[...,-4:],reference.reshape(4,10,-1)[...,-4:])


def test_network_changes_only_task_embedding_rows_and_amp_input_width():
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    original=yaml.safe_load((ROOT/'data/cfg/train/rlg/stage1_v2_fixed.yaml').read_text())['params']['network']
    extension=yaml.safe_load((ROOT/'data/cfg/train/rlg/stage1_base_push.yaml').read_text())['params']['network']
    assert original==extension
    from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias,TaskActionFusion
    assert TaskTypeEmbeddingBias(4,2).category_embed.weight.shape==(4,64)
    assert TaskTypeEmbeddingBias(4,2,5).category_embed.weight.shape==(5,64)
    assert TaskActionFusion().mlp[0].in_features==128
    def build(spec,amp_width):
        builder=AMPMultiAgentBuilder()
        builder.load(original)
        return builder.build('amp',actions_num=32,input_shape=(634,),amp_input_shape=(amp_width,),
            value_size=1,num_agents=2,num_objects=4,humanoid_obs_size=230,
            object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',
            scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=5.,
            relation_reward_mode=CFG['relationReward']['mode'],relation_graph_spec=spec,device='cpu')
    old=build(original_spec(CFG['relationGraph']),1320)
    new=build(CFG['relationGraph'],1330)
    a,b=old.state_dict(),new.state_dict()
    assert a.keys()==b.keys()
    changed={key for key in a if a[key].shape!=b[key].shape}
    embeddings={'actor_encoder.edge_encoder.category_embed.weight',
                'critic_encoder.edge_encoder.category_embed.weight'}
    assert embeddings<=changed
    remaining=changed-embeddings
    assert len(remaining)==1
    disc=remaining.pop()
    assert 'disc' in disc and a[disc].shape[-1]==1320 and b[disc].shape[-1]==1330
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==10368


def test_contact_functions_are_identical_to_current_push_implementation():
    import ast
    from env.tasks.multi_agent import push_contact
    source=Path(__file__).resolve().parents[4]/'tokenhsi/env/tasks/multi_agent/open_push_reward.py'
    current={f.name:ast.dump(f,include_attributes=False) for f in ast.parse(source.read_text()).body if isinstance(f,ast.FunctionDef)}
    copied={f.name:ast.dump(f,include_attributes=False) for f in ast.parse(Path(push_contact.__file__).read_text()).body if isinstance(f,ast.FunctionDef)}
    for name in ('push_rear_xy','push_face_geometry','push_hand_contact','push_contact_near','push_rsi_hand_targets'):
        assert current[name]==copied[name]


def test_incompatible_checkpoint_rejected_before_simulation(monkeypatch):
    from utils.stage1_push_spec import check_checkpoint,checkpoint_contract
    monkeypatch.chdir(ROOT.parent)
    env=deepcopy(CFG);env['motion_file']='tokenhsi/data/dataset_stage1_base_push.yaml'
    with pytest.raises(ValueError,match='570D/6000'):
        check_checkpoint({},env)
    check_checkpoint({'stage1_push_contract':checkpoint_contract(env)},env)
    with pytest.raises(ValueError,match='validation fixtures'):
        check_checkpoint({'validation_only':True},env,for_training=True)


def test_four_box_layout_protects_current_goal_and_rsi_corridors():
    from utils.stage1_push_scene import sample_four_box_scene
    from utils.push_scene import random_at_goal_valid,forward_xy,segment_distance
    source=yaml.safe_load((ROOT/'data/cfg/multi_agent/push_native_phase_y2m.yaml').read_text())['env']
    gen=torch.Generator().manual_seed(17)
    xy,q,goals=sample_four_box_scene(2048,source,'cpu',gen)
    assert random_at_goal_valid(xy[:,:2],q[:,:2],goals,source).all()
    radius=torch.tensor(source['pushBox']['size'][:2]).norm()/2
    rear=xy[:,:2]-1.5*forward_xy(q[:,:2])
    for box in (2,3):
        gap=segment_distance(rear,goals,xy[:,box,None],xy[:,box,None])
        assert (gap>=2*radius+.5-1e-5).all()
    assert ((xy[:,2]-xy[:,3]).norm(dim=-1)>=2*radius+.5-1e-5).all()
