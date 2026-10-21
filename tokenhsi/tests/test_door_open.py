"""CPU contracts for independent OPEN, rewards, shared network and AMP toggle."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
import torch
import yaml
from utils.edge_door_spec import OPEN, compile_graph, validate_env
from utils.edge_stage1_spec import compile_stage1_graph, semantic_graph_packet, parse_semantic_packet
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from env.tasks.multi_agent.edge_door_reward import door_reward, DoorRuntime

ROOT=Path(__file__).resolve().parents[1]

def env_config(mode='on'):
    return yaml.safe_load((ROOT/('data/cfg/multi_agent/door_open_amp_'+mode+'.yaml')).read_text())['env']

def train_config(mode='on'):
    return yaml.safe_load((ROOT/('data/cfg/train/rlg/door_open_amp_'+mode+'.yaml')).read_text())

@pytest.mark.parametrize('mode',['on','off'])
def test_config(mode):
    env=env_config(mode)
    validate_env(env);validate_relation_config(env['relationReward'])
    assert env['numEnvs']==2048
    assert train_config(mode)['params']['algo']['name']=='ma'
    assert train_config(mode)['params']['config']['amp_enabled']==env['ampEnabled']

def test_only_amp_toggle_differs():
    a,b=env_config('off'),env_config('on')
    a['ampEnabled']=True
    assert a==b
    a,b=train_config('off'),train_config('on')
    a['params']['config']['amp_enabled']=True
    assert a==b

@pytest.mark.parametrize('binding',[[0,0],[1,0],[0,2]])
def test_reject_shared_or_wrong_binding(binding):
    env=env_config();env['relationGraph']['bindings']=binding
    with pytest.raises(ValueError):validate_env(env)

def test_open_packet():
    e=env_config();g=compile_stage1_graph(e['relationGraph'],2,2)
    assert g.edge_relation.tolist()==[OPEN,OPEN]
    assert g.edge_src.tolist()==[0,1] and g.edge_dst.tolist()==[2,3]
    assert g.edge_owner.tolist()==[0,1]
    assert not g.prereq_mask.any() and (g.term_index==-1).all()
    packet=semantic_graph_packet(g,3)
    assert packet.shape==(3,10)
    valid,src,dst,relation,owner=parse_semantic_packet(packet)
    assert valid.all() and torch.equal(src,owner) and (relation==OPEN).all()
    assert torch.equal(dst,src+2)

def geometry():
    handles=torch.tensor([[[0.,0.,1.],[0.,5.,1.]]])
    roots=handles.clone();roots[...,2]=0.
    hands=handles[:,:,None].repeat(1,1,2,1)
    return roots,hands,handles,torch.zeros(1,2)

def evaluate(values):
    e=env_config();g=compile_graph(e['relationGraph'],2,2)
    return door_reward(*values,g,e['relationReward'])

def test_monotonic_terms_and_bounds():
    roots,hands,handles,angles=geometry()
    near=evaluate((roots,hands,handles,angles))
    far=evaluate((roots+torch.tensor([3.,0.,0.]),hands+3,handles,angles))
    assert (near['approach_component']>far['approach_component']).all()
    assert (near['interaction_component']>far['interaction_component']).all()
    mid=evaluate((roots,hands,handles,angles+.6))
    assert (mid['progress_component']>near['progress_component']).all()
    opened=evaluate((roots+100,hands+100,handles,angles+1.2))
    torch.testing.assert_close(opened['total'],torch.full((1,2),.6))
    for angle in (-1.,0.,.6,1.2,1.7):
        r=evaluate((roots,hands,handles,angles+angle))
        assert torch.isfinite(r['total']).all() and (r['total']>=0).all() and (r['total']<=.600001).all()

def test_owner_only_geometry_and_conventional_team_sharing():
    values=geometry();base=evaluate(values)
    roots,hands,handles,angles=[x.clone() for x in values]
    handles[:,1]+=100;angles[:,1]=1.2
    changed=evaluate((roots,hands,handles,angles))
    for key in ('approach_component','interaction_component','progress_component','success_component','own_success'):
        torch.testing.assert_close(base[key][:,0],changed[key][:,0])
    local=changed['total']
    torch.testing.assert_close(changed['agent_task_reward'],.9*local+.1*local.flip(-1))
    assert changed['own_success'].tolist()==[[False,True]]

def test_success_is_current_and_partial_reset_preserves_other_env():
    e=env_config();g=compile_graph(e['relationGraph'],2,2)
    rt=DoorRuntime(3,g,e['relationReward'],'cpu')
    roots,hands,handles,angles=geometry()
    rt.step(roots.expand(3,-1,-1),hands.expand(3,-1,-1,-1),handles.expand(3,-1,-1),angles.expand(3,-1)+1.2)
    assert rt.done.all()
    rt.reset(torch.tensor([1]),torch.zeros(1,2))
    assert rt.done[[0,2]].all() and not rt.done[1].any()
    assert rt.suffix(torch.tensor([1])).shape==(1,10)
    rt.step(roots.expand(3,-1,-1),hands.expand(3,-1,-1,-1),handles.expand(3,-1,-1),angles.expand(3,-1))
    assert not rt.done.any() and rt.achieved[[0,2]].all()

def test_new_checkpoint_contract_rejects_old_relations():
    door=checkpoint_metadata(env_config()['relationReward'])
    assert door['relation_taxonomy']=={'open':11} and door['graph_record_width']==5
    old=yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder.yaml').read_text())['env']
    with pytest.raises(ValueError):check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(old['relationReward'])},door)

def test_asset_has_passive_hinge_and_collision_handle():
    asset=ET.parse(ROOT/'assets/door/door.urdf').getroot()
    hinge=asset.find("joint[@name='hinge']")
    assert hinge.attrib['type']=='revolute' and hinge.find('axis').attrib['xyz']=='0 0 1'
    assert float(hinge.find('limit').attrib['upper'])>env_config()['relationReward']['door']['target_angle']
    assert asset.find("link[@name='handle']/collision") is not None

def test_shared_encoder_open_gradient_and_finite_actor_critic():
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    torch.set_num_threads(1)
    e=env_config();builder=AMPMultiAgentBuilder();builder.load(train_config()['params']['network'])
    # H223*2 + O35*2 + G1*2 + pose7*6 + packet5*2 = 570
    network=builder.build('amp',actions_num=32,input_shape=(570,),amp_input_shape=(1290,),
        value_size=1,num_agents=2,num_objects=2,humanoid_obs_size=230,object_obs_size=39,
        goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,35,1],
        scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode=e['relationReward']['mode'],
        relation_graph_spec=e['relationGraph'],device='cpu')
    edge=network.actor_encoder.edge_encoder
    assert edge is network.critic_encoder.edge_encoder
    assert edge.relation_embed.num_embeddings==12
    poses=torch.zeros(2,6,7);poses[...,6]=1.
    obs=torch.cat((torch.randn(2,518),poses.flatten(1),semantic_graph_packet(compile_graph(e['relationGraph'],2,2),2)),-1)
    for enc,head in ((network.actor_encoder,network.action_head),(network.critic_encoder,network.value_head)):
        enc.gta_diagnostics_first_forward=False
        network.zero_grad()
        out=head(enc(obs));assert torch.isfinite(out).all()
        out.square().mean().backward()
        # Existing shared encoder starts with a zero bias projection. First PPO
        # gradient updates that projection; semantic embeddings receive gradients next.
        assert edge.bias_projection.grad.abs().sum()>0
        with torch.no_grad():
            edge.bias_projection.add_(-.01*edge.bias_projection.grad)
        network.zero_grad()
        head(enc(obs)).square().mean().backward()
        assert edge.relation_embed.weight.grad[OPEN].abs().sum()>0

def test_amp_off_skips_discriminator_and_style_reward():
    from learning.multi_agent.ma_agent import MAAgent
    agent=object.__new__(MAAgent);agent._amp_enabled=False;agent._task_reward_w=.5
    obs=torch.randn(3,2,1290)
    amp=agent._calc_amp_rewards(obs)
    assert amp['disc_rewards'].shape==(3,2,1) and not amp['disc_rewards'].any()
    task=torch.rand(3,2,1)
    torch.testing.assert_close(agent._combine_rewards(task,{'disc_rewards':torch.full_like(task,float('nan'))}),.5*task)

def test_model_amp_off_skips_disc_forward(monkeypatch):
    from learning.amp_models import ModelAMPContinuous, ModelA2CContinuousLogStd
    model=object.__new__(ModelAMPContinuous.Network)
    torch.nn.Module.__init__(model)
    class RejectDisc(torch.nn.Module):
        def eval_disc(self,obs):raise AssertionError('AMP OFF called discriminator')
    model.a2c_network=RejectDisc()
    monkeypatch.setattr(ModelA2CContinuousLogStd.Network,'forward',lambda self,inputs:{'values':torch.ones(1)})
    assert 'values' in model({'is_train':True,'amp_enabled':False})


@pytest.mark.parametrize('direction', ['push', 'pull'])
def test_partial_hinge_reset_and_object_observation(monkeypatch, direction):
    from env.tasks.multi_agent import humanoid_ma_door as module
    task=object.__new__(module.HumanoidMADoor)
    task.device='cpu';task.sim=None;task.num_objects=2
    task._door_cfg={'resetAngleRange':[0.,0.], 'openingDirection':direction}
    task._dof_state=torch.zeros(3*6,2)
    task._door_dof_state=task._dof_state.view(3,6,2)[:,4:]
    task._box_states=torch.zeros(3,2,13);task._box_states[...,6]=1.
    task._door_panel_states=task._box_states.clone()
    task._door_handle_states=task._box_states.clone()
    task._box_actor_ids=torch.tensor([[2,3],[6,7],[10,11]],dtype=torch.int32)
    calls=[]
    task.gym=SimpleNamespace(set_dof_state_tensor_indexed=lambda sim,states,ids,n:calls.append(ids.clone()))
    monkeypatch.setattr(module.gymtorch,'unwrap_tensor',lambda value:value)
    task._door_dof_state[:]=.2
    untouched=task._door_dof_state[[0,2]].clone()
    task.reset_door_angles(torch.tensor([1]),torch.tensor([[.3,.7]]))
    torch.testing.assert_close(task._door_dof_state[[0,2]],untouched)
    torch.testing.assert_close(task.door_hinge_angle[1],torch.tensor([.3,.7]))
    assert not task.door_hinge_velocity[1].any()
    assert calls[0].tolist()==[6,7]
    features=task._clean_object_features(torch.zeros(1,2,30),torch.tensor([1]))
    assert features.shape==(1,2,35)
    torch.testing.assert_close(features[...,30],torch.tensor([[.3,.7]]))
    torch.testing.assert_close(features[...,32:],torch.tensor([[task.door_local_handle_offset]*2]))
    # Positive opening must rotate toward the human (-Y) in pull mode.
    sign = -1 if direction == 'pull' else 1
    torch.testing.assert_close(task._door_panel_states[1, :, 5], sign*torch.sin(torch.tensor([.3,.7])/2))
    expected_y = sign*torch.sin(torch.tensor([.3,.7]))*.78 + torch.cos(torch.tensor([.3,.7]))*task.door_local_handle_offset[1]
    torch.testing.assert_close(task.door_handle_position[1, :, 1], expected_y)
    for invalid in (torch.tensor([[float('nan'),0.]]),torch.tensor([[-.1,0.]]),torch.zeros(1,3)):
        with pytest.raises(ValueError):task.reset_door_angles(torch.tensor([1]),invalid)


@pytest.mark.parametrize('pd',[False,True])
@pytest.mark.parametrize('extra',[0,2])
def test_passive_object_dofs_not_actuated(monkeypatch,pd,extra):
    from env.tasks.multi_agent import humanoid_ma as module
    task=object.__new__(module.HumanoidMA)
    task.num_envs=2;task.num_agents=2;task.num_dof=3
    task.device='cpu';task.sim=None;task._extra_object_dofs=extra
    task.cfg={'env':{'enableTrackInitState':False}}
    task._pd_control=pd;task._pd_action_offset=torch.ones(3);task._pd_action_scale=torch.ones(3)
    task.motor_efforts=torch.ones(2,3);task.power_scale=1.
    outputs=[]
    task.gym=SimpleNamespace(set_dof_position_target_tensor=lambda sim,value:outputs.append(value.clone()),
        set_dof_actuation_force_tensor=lambda sim,value:outputs.append(value.clone()))
    monkeypatch.setattr(module.gymtorch,'unwrap_tensor',lambda value:value)
    module.HumanoidMA.pre_physics_step(task,torch.ones(4,3))
    assert outputs[0].shape==(2,6+extra)
    assert (outputs[0][:,:6]==(2 if pd else 1)).all()
    if extra:assert not outputs[0][:,-extra:].any()


def test_amp_off_skips_demo_replay_and_normalization():
    from learning.multi_agent.ma_agent import MAAgent
    agent=object.__new__(MAAgent);agent._amp_enabled=False
    # No env, normalization or buffer objects: OFF must never access them.
    agent._init_amp_demo_buf();agent._update_amp_demos()
    obs=torch.randn(2,1290)
    agent._store_replay_amp_obs(obs)
    assert agent._preproc_amp_obs(obs) is obs


def test_amp_expert_mapping_independent_from_eval_reset_skills():
    from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
    task=object.__new__(HumanoidMADoor)
    task.device='cpu';task.dt=1/30;task.cfg={'env':{'ampEnabled':True}}
    task._amp_task_conditioning=False
    task._skill=['loco']  # evaluation reset skills
    task._amp_expert_skills=['loco','doorRef']
    task._skill_disc_prob=torch.tensor([0.,1.])
    task._num_amp_obs_steps=10;task._num_amp_obs_per_step=129
    task._amp_obs_demo_buf=None
    task._motion_lib={'doorRef':SimpleNamespace(sample_motions=lambda n:torch.zeros(n,dtype=torch.long),
        sample_time=lambda ids,truncate_time:torch.zeros(len(ids)))}
    task.build_amp_obs_demo=lambda ids,times,lib:torch.ones(len(ids),1290)
    demo=task.fetch_amp_obs_demo(4)
    assert demo.shape==(4,1290) and (demo==1).all()
