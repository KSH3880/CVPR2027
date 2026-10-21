"""Door-free graph contracts and equivalence with existing PUSH rewards."""
from copy import deepcopy
from pathlib import Path
from isaacgym import gymapi
import pytest
import torch
import yaml
from utils.open_push_spec import compile_graph,graph_for_tasks,validate_env,PUSH,AT
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime,push_hand_targets

ROOT=Path(__file__).resolve().parents[2]

def env_config(name):
    env=yaml.safe_load((ROOT/('tokenhsi/data/cfg/multi_agent/'+name+'.yaml')).read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


def test_box_physics_only_observation_and_actor_critic():
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    from utils.edge_stage1_spec import semantic_graph_packet

    task=SimpleNamespace(_push_only=True)
    task.get_clean_object_obs_size=lambda:HumanoidMAOpenPush.get_clean_object_obs_size(task)
    local=torch.randn(2,2,30)
    assert HumanoidMAOpenPush.get_clean_object_obs_size(task)==30
    torch.testing.assert_close(HumanoidMAOpenPush._clean_object_features(task,local,None),local)
    e=env_config('push_toy_random_at_10deg_at_tilt_soft')
    params=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/push_toy_random_at_10deg_at_tilt_soft.yaml').read_text())['params']['network']
    builder=AMPMultiAgentBuilder();builder.load(params)
    network=builder.build('amp',actions_num=32,input_shape=(570,),amp_input_shape=(1310,),value_size=1,
        num_agents=2,num_objects=2,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,
        observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,
        scene_arena_scale=7.,relation_reward_mode=e['relationReward']['mode'],
        relation_graph_spec=e['relationGraph'],device='cpu')
    graph=graph_for_tasks(e['relationGraph'],torch.ones(2,2,dtype=torch.long))
    kin=torch.randn(2,6,7);kin[...,3:]=torch.nn.functional.normalize(kin[...,3:],dim=-1)
    obs=torch.cat((torch.randn(2,446),local.flatten(1),torch.ones(2,2),kin.flatten(1),semantic_graph_packet(graph,2)),-1)
    assert obs.shape==(2,570)
    with torch.no_grad():
        actor=network.eval_actor(obs);critic=network.eval_critic(obs)
    assert all(x.shape==(4,32) and torch.isfinite(x).all() for x in actor)
    assert critic.shape==(4,1) and torch.isfinite(critic).all()


def test_graph_has_six_entities_and_no_open_edges():
    env=env_config('push_toy_only_random_scene');validate_env(env)
    graph=compile_graph(env['relationGraph'],2,2)
    assert graph.edge_src.tolist()==[0,2,1,3]
    assert graph.edge_dst.tolist()==[2,4,3,5]
    assert graph.edge_relation.tolist()==[PUSH,AT,PUSH,AT]
    assert graph.edge_valid.all()
    graph=graph_for_tasks(env['relationGraph'],torch.ones(8,2,dtype=torch.long))
    assert graph.num_objects==2 and graph.edge_dst.max()==5
    with pytest.raises(ValueError):graph_for_tasks(env['relationGraph'],torch.zeros(1,2,dtype=torch.long))
    with pytest.raises(ValueError):compile_graph(env['relationGraph'],2,4)


def test_legacy_box_features_and_bank_relocation():
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
    from utils.open_push_spec import checkpoint_contracts_match
    env=env_config('push_soft_legacy40_at05');validate_env(env)
    rt=SimpleNamespace(push_eligible=torch.ones(2,2,dtype=torch.bool),push_credited=torch.full((2,2),.5),
        push_hand_distance=torch.full((2,2),.025),push_completed=torch.zeros(2,2,dtype=torch.bool),body_contact=torch.ones(2,2,dtype=torch.bool))
    task=SimpleNamespace(_push_only=True,_push_object_obs_size=40,relation_runtime=rt,_relation_cfg=env['relationReward'])
    task.get_clean_object_obs_size=lambda:HumanoidMAOpenPush.get_clean_object_obs_size(task)
    local=torch.randn(2,2,30);obs=HumanoidMAOpenPush._clean_object_features(task,local,None)
    assert obs.shape==(2,2,40)
    torch.testing.assert_close(obs[...,:30],local)
    torch.testing.assert_close(obs[...,30:],torch.tensor([.25,.5,0,1,0,0,0,0,0,1]).expand(2,2,10))
    a={'rsi_bank_sha256':'same','object_obs_size':40,'rsi':{'bank_file':'old','interaction_probability':.5}}
    b=deepcopy(a);b['rsi']['bank_file']='new';assert checkpoint_contracts_match(a,b)
    b['object_obs_size']=30;assert not checkpoint_contracts_match(a,b)
    b=deepcopy(a);b['rsi_bank_sha256']='different';assert not checkpoint_contracts_match(a,b)


def test_physics_reward_bank_and_amp_configuration_preserved():
    new=env_config('push_toy_only_random_scene');old=env_config('push_toy_midpoint_2m_random_scene')
    assert bank_contract(new)==bank_contract(old)
    changed=deepcopy(new);changed['relationGraph'].pop('push_only');changed['numObjects']=4
    assert changed==old
    train=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/push_toy_only_random_scene.yaml').read_text())
    assert train['params']['config']['max_epochs']==1000000


def test_open_kernel_never_called_and_push_rewards_identical(monkeypatch):
    import env.tasks.multi_agent.open_push_reward as module
    old=env_config('push_toy_midpoint_2m_random_scene');new=env_config('push_toy_only_random_scene')
    boxes=torch.zeros(1,2,13);boxes[...,2]=.65;boxes[...,6]=1
    angles=torch.zeros(1,2);handles=torch.zeros(1,2,3);goals=boxes[...,:3].clone();goals[...,1]+=2
    runtimes=[OpenPushRuntime(1,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',e['pushBox']['size'],e['manipulationContact']) for e in (old,new)]
    for rt in runtimes:
        rt.reset(torch.arange(1),torch.ones(1,2,dtype=torch.long),boxes)
        rt.initialize_manipulation(torch.arange(1),boxes,goals,angles)
    original=module.door_reward
    def forbidden(*args,**kwargs):raise AssertionError('Door-free runtime evaluated OPEN reward')
    for offset,valid in ((.2,True),(.5,False),(.1,False),(-.3,False),(1.5,True)):
        boxes[...,1]+=offset
        hands=push_hand_targets(boxes,new['pushBox']['size'],new['relationReward']['push'])
        results=[]
        for index,rt in enumerate(runtimes):
            monkeypatch.setattr(module,'door_reward',forbidden if index else original)
            rt.clean_push_contact[:]=valid;rt.push_contact[:]=valid
            rt.collect_manipulation(boxes,goals,angles,torch.zeros_like(rt.contact),rt.clean_push_contact)
            results.append(rt.step(hands.mean(-2),hands,handles,angles,boxes,goals))
        for key in results[0]:torch.testing.assert_close(results[0][key],results[1][key])
        torch.testing.assert_close(runtimes[0].push_hand_distance,runtimes[1].push_hand_distance)
