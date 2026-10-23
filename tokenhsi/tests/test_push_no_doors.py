"""Actual deletion, graph remapping and real policy transfer for PUSH-only inputs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
import torch
import yaml
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.open_push_transfer import transfer_open_push
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.open_push_spec import compile_graph, validate_env
from utils.push_policy_observation import (PUSH_OBJECT_COLUMNS, remove_door_tokens,
                                           project_source_model, project_source_normalizer)

torch.set_num_threads(1)
CONFIG=Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml')


def full_observation():
    obs=torch.randn(2,644)*.1
    poses=obs[:,568:624].reshape(2,8,7)
    poses[...,3:7]=0;poses[...,6]=1
    obs[:,-20:]=torch.tensor([[1,0,4,12,0],[1,4,6,7,0],[1,1,5,12,1],[1,5,7,7,1]]).flatten()
    return obs


def test_door_values_and_poses_are_deleted_not_padded():
    raw=full_observation();compact=remove_door_tokens(raw,2,[223,30,1])
    assert compact.shape==(2,570)
    changed=raw.clone();changed[:,446:506]=float('nan');changed[:,582:596]=float('nan')
    assert torch.equal(compact,remove_door_tokens(changed,2,[223,30,1]))
    torch.testing.assert_close(compact[:,446:506],raw[:,506:566])
    spec=yaml.safe_load(CONFIG.read_text())['env']['relationGraph']
    graph=compile_graph(spec,2,2)
    packet=compact[:,-20:].reshape(2,4,5)
    assert torch.equal(packet[0,:,1].long(),graph.edge_src)
    assert torch.equal(packet[0,:,2].long(),graph.edge_dst)
    # Arbitrary physical box bindings survive remapping.
    raw[:,-20:]=torch.tensor([[1,0,5,12,0],[1,5,6,7,0],[1,1,4,12,1],[1,4,7,7,1]]).flatten()
    remapped=remove_door_tokens(raw,2,[223,30,1])[:,-20:].reshape(2,4,5)
    assert remapped[0,:,1].tolist()==[0,3,1,2]
    assert remapped[0,:,2].tolist()==[3,4,2,5]


def test_reject_door_graph_and_mixed_tasks():
    obs=full_observation();obs[:,-18]=2
    with pytest.raises(ValueError,match='reference a door'):
        remove_door_tokens(obs,2,[223,30,1])
    env=yaml.safe_load(CONFIG.read_text())['env'];validate_env(env)
    env['relationGraph']['task_probabilities']={'OPEN':.5,'PUSH':.5}
    with pytest.raises(ValueError,match='PUSH-only'):
        validate_env(env)


def test_no_door_or_eligibility_columns_or_reads():
    # Neither door tensors nor a reward runtime are needed for intrinsic features.
    task=SimpleNamespace(_no_door_policy=True)
    intrinsic=torch.randn(2,4,30)
    features=HumanoidMAOpenPush._clean_object_features(task,intrinsic,None)
    assert features.shape==(2,4,30)
    torch.testing.assert_close(features,intrinsic)
    task.relation_runtime=SimpleNamespace(push_eligible=torch.ones(2,2,dtype=torch.bool))
    torch.testing.assert_close(HumanoidMAOpenPush._clean_object_features(task,intrinsic,None),features)
    partial=HumanoidMAOpenPush._clean_object_features(task,intrinsic[1:],torch.tensor([1]))
    torch.testing.assert_close(partial,features[1:])


def test_real_source_transfer_normalization_six_tokens_and_gradients(tmp_path):
    cfg=yaml.safe_load(CONFIG.read_text())
    # Optional projection utility remains covered; the runnable profile is scratch.
    original=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01.yaml').read_text())
    cfg=deepcopy(original)
    cfg['env']['pushNoDoorPolicy']=True
    cfg['experiment']=deepcopy(original['experiment'])
    source_path=Path(cfg['experiment']['transfer']['checkpoint'])
    weights=torch.load(source_path,map_location='cpu',weights_only=False)
    params=deepcopy(weights['relation_experiment_config']['train']['params']['network'])
    builder=AMPMultiAgentBuilder();builder.load(params)
    amp_width=weights['model']['a2c_network._disc_mlp.0.weight'].shape[1]
    env=cfg['env']
    net=builder.build('push_no_doors',actions_num=32,input_shape=(570,),amp_input_shape=(amp_width,),value_size=1,
        num_agents=2,num_objects=2,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,
        scene_arena_scale=14.,relation_reward_mode=env['relationReward']['mode'],relation_graph_spec=env['relationGraph'],device='cpu')
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.a2c_network=net
    model=Model()
    rms=SceneRunningMeanStd(entity_sizes=[223,30,1],entity_counts=[2,2,2],normalized_sizes=[223,30,0],kinematic_size=7,extra_passthrough_size=20)
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    amp_rms=RunningMeanStd(tuple(weights['amp_input_mean_std']['running_mean'].shape))
    contract=deepcopy(weights['open_push_contract'])
    contract.update(policy_observation='push_no_doors_30d_v2',object_obs_size=30,
        contact_at=env['relationReward']['contact_at'],push_goal=env['pushGoalRandomization'],
        rsi=env['openPushRSI'],rsi_bank_sha256=hashlib.sha256(Path(env['openPushRSI']['bank_file']).read_bytes()).hexdigest())
    task=SimpleNamespace(cfg=cfg,_relation_cfg=env['relationReward'],_no_door_policy=True,
        get_obs_size=lambda:570,get_open_push_contract=lambda:deepcopy(contract))
    agent=SimpleNamespace(model=model,optimizer=torch.optim.Adam(model.parameters()),running_mean_std=rms,
        _amp_input_mean_std=amp_rms,vec_env=SimpleNamespace(env=SimpleNamespace(task=task)),
        _relation_experiment_config=dict(train=dict(params=dict(network=params))))
    before={k:v.clone() for k,v in model.state_dict().items()}
    report=transfer_open_push(agent,weights,source_path)
    assert report['removed_observation_width']==114 and report['source_epoch']==10000
    projected=project_source_model(weights['model'])
    for key in report['copied_tensors']:torch.testing.assert_close(model.state_dict()[key],projected[key])
    for key in report['reset_tensors']:torch.testing.assert_close(model.state_dict()[key],before[key])
    for key,value in rms.state_dict().items():torch.testing.assert_close(value,project_source_normalizer(weights['running_mean_std'])[key])
    net.eval();rms.eval()
    obs=rms(remove_door_tokens(full_observation(),2,[223,30,1]))
    actor=net.eval_actor(obs)[0];critic=net.eval_critic(obs)
    assert actor.shape==(4,32) and critic.shape==(4,1)
    assert torch.isfinite(actor).all() and torch.isfinite(critic).all()
    for encoder in (net.actor_encoder,net.critic_encoder):
        assert encoder.last_shape_flow['tokens']==(2,6,64)
        assert encoder.last_shape_flow['gta_g']==(2,6,4,4)
    with torch.no_grad():
        normal=net.actor_encoder(obs,return_all=True,token_order=torch.arange(6))[1]
        swapped=net.actor_encoder(obs,return_all=True,token_order=torch.tensor([4,1,5,0,3,2]))[1]
    torch.testing.assert_close(normal,swapped,atol=2e-5,rtol=1e-5)
    (actor.square().mean()+critic.square().mean()).backward()
    assert net.actor_encoder.tokenizers[1][0].weight.grad.abs().sum()>0
    assert net.critic_encoder.tokenizers[1][0].weight.grad.abs().sum()>0
    assert not agent.optimizer.state
