from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.joint_carry_amp import validate_ablation, check_ablation_checkpoint, sample_amp_sources
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata
from test_joint_carry import build_network, joint_observations
from test_stage2_coordination import Wrapper

ROOT = Path(__file__).resolve().parents[1]
BASE = 'approach_stage2_joint_carry_mixed80_'


def configs(suffix):
    env = yaml.safe_load((ROOT/'data/cfg/multi_agent'/(BASE+suffix+'.yaml')).read_text())['env']
    train = yaml.safe_load((ROOT/'data/cfg/train/rlg'/('amp_ma_stage2_joint_carry_mixed80_'+suffix+'.yaml')).read_text())
    return env, train


def network(suffix):
    env, train = configs(suffix)
    builder = AMPMultiAgentBuilder();builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1320,),
        value_size=1, num_agents=2, num_objects=4, humanoid_obs_size=230,
        object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=env['relationReward']['mode'], relation_graph_spec=env['relationGraph'], device='cpu')
    net.actor_encoder.gta_diagnostics_first_forward = False
    net.critic_encoder.gta_diagnostics_first_forward = False
    return Wrapper(net)


def test_configs_change_only_amp_and_policy():
    baseline, _ = configs('task_embedding')
    ca, ca_train = configs('locoamp_task_embedding')
    head, head_train = configs('locoamp_head_only')
    for env, train in ((ca, ca_train), (head, head_train)):
        validate_env(env);validate_ablation(env, train)
        assert {k:v for k,v in env.items() if k != 'jointCarryAblation'} == baseline
    assert ca['jointCarryAblation']['amp'] == head['jointCarryAblation']['amp']
    bad = deepcopy(head_train);bad['params']['network']['coordination'] = ca_train['params']['network']['coordination']
    with pytest.raises(ValueError, match='coordination'):
        validate_ablation(head, bad)


def test_head_only_transfer_and_policy_gradients():
    torch.manual_seed(19)
    source = Wrapper(build_network(False))
    reward = yaml.safe_load((ROOT/'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']['relationReward']
    checkpoint = dict(model=source.state_dict(), relation_metadata=checkpoint_metadata(reward))
    ca, head = network('locoamp_task_embedding'), network('locoamp_head_only')
    transfer_stage1_weights(ca, checkpoint)
    report = transfer_stage1_weights(head, checkpoint)
    net = head.a2c_network
    assert not hasattr(net, 'coordination')
    assert net.action_head[0][0].in_features == 64
    assert report['head_expansion'] == {'source_columns':64, 'context_columns':0}
    assert report['new_tensors'] == []
    for name, value in net.action_head.state_dict().items():
        torch.testing.assert_close(value, source.a2c_network.action_head.state_dict()[name], rtol=0, atol=0)
    obs = joint_observations()
    with torch.no_grad():
        expected = source.a2c_network.action_head(net.actor_encoder(obs)).flatten(0, 1)
        torch.testing.assert_close(net.eval_actor(obs)[0], expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(net.eval_actor(obs)[0], ca.a2c_network.eval_actor(obs)[0], atol=1e-6, rtol=1e-5)
    encoder = {k:v.clone() for k,v in net.actor_encoder.state_dict().items()}
    head_weight = net.action_head[0][0].weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    mu, _ = net.eval_actor(obs)
    (mu.square().mean()+net.eval_critic(obs).square().mean()+net.eval_disc(torch.randn(8,1320)).square().mean()).backward()
    assert not net.sigma.requires_grad
    assert all(p.grad is None and not p.requires_grad for p in net.actor_encoder.parameters())
    assert net.action_head[0][0].weight.grad is not None
    optimizer.step()
    assert not torch.equal(head_weight, net.action_head[0][0].weight)
    assert all(torch.equal(value, net.actor_encoder.state_dict()[key]) for key,value in encoder.items())


def test_amp_source_distribution_and_checkpoint_isolation():
    torch.manual_seed(11)
    family = torch.zeros(100000, dtype=torch.long)
    fraction = torch.bincount(sample_amp_sources(family), minlength=3)/len(family)
    torch.testing.assert_close(fraction, torch.tensor([.8,.1,.1]), atol=.006, rtol=0)
    assert sample_amp_sources(torch.ones(1000, dtype=torch.long)).count_nonzero() == 0
    ca, _ = configs('locoamp_task_embedding');head, _ = configs('locoamp_head_only')
    weights = {'joint_carry_ablation':deepcopy(ca['jointCarryAblation'])}
    check_ablation_checkpoint(weights, ca)
    for saved, env in ((weights,head), ({},ca), (weights,{})):
        with pytest.raises(ValueError, match='ablation differs'):
            check_ablation_checkpoint(saved, env)
    previous_dataset = deepcopy(weights)
    previous_dataset['joint_carry_ablation']['amp']['dataset_sha256'] = '0'*64
    with pytest.raises(ValueError, match='ablation differs'):
        check_ablation_checkpoint(previous_dataset, ca)


def test_actual_demo_path_mixes_only_experts_not_rsi():
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    env, _ = configs('locoamp_task_embedding')

    class Library:
        def __init__(self, marker):
            self.marker = marker
        def sample_motions(self, n):
            return torch.zeros(n, dtype=torch.long)
        def sample_time(self, ids, truncate_time):
            return torch.ones(len(ids))

    original = {name:Library(i+1) for i,name in enumerate(env['skill'])}
    extra = dict(backward=Library(101), sideways=Library(102))
    task = SimpleNamespace(_carry_only=False, _joint_carry=True, _before=False,
        cfg={'env':env}, device='cpu', dt=1/30, _num_amp_obs_steps=10,
        _motion_lib=original, _amp_locomotion_libs=extra)
    task.get_num_amp_obs = lambda:1320
    task.build_amp_obs_demo = lambda ids,times,lib:torch.full((len(ids)*10,129), float(lib.marker))
    result = HumanoidMACarry._fetch_conditioned_amp_demo(task, 30000).reshape(30000,10,132)
    assert torch.isfinite(result).all()
    assert (result[:,:,129:] == torch.tensor([1.,0.,0.])).all()
    assert (result[:,:,0] == result[:,0,0,None]).all()
    assert abs((result[:,0,0] == 101).float().mean()-.1) < .01
    assert abs((result[:,0,0] == 102).float().mean()-.1) < .01
    assert task._motion_lib is original and set(original) == set(env['skill'])
