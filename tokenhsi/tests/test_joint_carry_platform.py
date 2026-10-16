from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.joint_carry_platform import JointCarryPlatform
from utils.joint_carry_platform import LOCO_PLATFORM, platform_heights
from utils.joint_carry_spec import validate_env
from utils.joint_carry_amp import validate_ablation, check_ablation_checkpoint
from test_joint_carry_amp_ablation import configs


BASE = 'locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding'
NEW = 'locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding'


def test_only_loco_platform_changes_and_checkpoint_contract():
    env,train=configs(NEW)
    base,base_train=configs(BASE)
    validate_env(env);validate_ablation(env,train)
    restored=deepcopy(env)
    assert restored['jointCarryRsi'].pop('locoPlatform')==LOCO_PLATFORM
    assert restored==base
    restored=deepcopy(train)
    restored['params']['config']['name']=base_train['params']['config']['name']
    assert restored==base_train
    weights=dict(joint_carry_loco_platform=LOCO_PLATFORM,
        joint_carry_rsi_dataset=env['jointCarryRsi']['dataset_sha256'],
        joint_carry_goal_sampling=env['jointCarryRsi']['goalSampling'],
        joint_carry_ablation=env['jointCarryAblation'])
    check_ablation_checkpoint(weights,env)
    with pytest.raises(ValueError,match='loco platform'):
        check_ablation_checkpoint(weights,base)
    weights.pop('joint_carry_loco_platform')
    check_ablation_checkpoint(weights,base)
    with pytest.raises(ValueError,match='loco platform'):
        check_ablation_checkpoint(weights,env)
    env['jointCarryRsi']['locoPlatform']['probability']=1.
    with pytest.raises(ValueError,match='platform contract'):
        validate_env(env)


def test_size_dependent_bounds_and_original_clipped_distribution():
    h=platform_heights(torch.tensor([.3,.4,.6,.4]),torch.tensor([.99,.99,.99,0.]),LOCO_PLATFORM)
    torch.testing.assert_close(h,torch.tensor([.9,.8,.6,.02]))
    with pytest.raises(ValueError,match='too tall'):
        platform_heights(torch.tensor([1.19]),torch.tensor([.5]),LOCO_PLATFORM)
    torch.manual_seed(42)
    values=platform_heights(torch.full((100000,),.4),torch.rand(100000),LOCO_PLATFORM)
    assert abs(float(torch.isclose(values,values.new_tensor(.8)).float().mean())-.2)<.005
    assert abs(float((values==.02).float().mean())-.02)<.003


def task_for_test(n=10000):
    root=torch.zeros(n,7,13);root[...,6]=1.
    root[:,-1,2]=20.
    sizes=torch.full((n,4,3),.4)
    boxes=root[:,2:6]
    boxes[...,2]=sizes[...,2]/2
    order=torch.rand(n,4).argsort(-1)
    return SimpleNamespace(num_envs=n,device='cpu',_root_states=root.reshape(-1,13),
        get_num_actors_per_env=lambda:7,_joint_loco_platform_cfg=deepcopy(LOCO_PLATFORM),
        _box_states=boxes,_box_size=sizes,_logical_box_order=order,
        _platform_inactive_height=20.,_env_origins=torch.zeros(n,3),
        _skill=['loco','pickUp','carryWith','putDown'],
        _joint_reset=SimpleNamespace(skill_index=torch.arange(n)%4))


def test_loco_only_payload_binding_probability_and_partial_reset():
    torch.manual_seed(123)
    t=task_for_test()
    platform=JointCarryPlatform(t)
    ids=torch.arange(t.num_envs)
    old=t._box_states.clone()
    platform.reset(ids)
    loco=t._joint_reset.skill_index==0
    assert not platform.active[~loco].any()
    assert abs(float(platform.active[loco].float().mean())-.5)<.04
    changed=torch.zeros(t.num_envs,4,dtype=torch.bool)
    chosen=ids[platform.active]
    changed[chosen,t._logical_box_order[chosen,0]]=True
    torch.testing.assert_close(t._box_states[~changed],old[~changed],atol=0,rtol=0)
    delta=t._box_states[chosen,t._logical_box_order[chosen,0],2]-.2
    torch.testing.assert_close(delta,platform.height[chosen]+.05)
    saved=t._root_states.view(t.num_envs,-1,13)[1::2].clone()
    t._joint_reset.skill_index[::2]=1
    platform.reset(ids[::2])
    assert not platform.active[::2].any()
    assert (platform.states[::2,2]==20.).all()
    torch.testing.assert_close(saved,t._root_states.view(t.num_envs,-1,13)[1::2],atol=0,rtol=0)


def test_pose_and_scene_translation_follow_payload():
    t=task_for_test(4)
    t._joint_reset.skill_index[:]=0
    t._joint_loco_platform_cfg['probability']=1.
    ids=torch.arange(4)
    payload=t._logical_box_order[:,0]
    yaw=torch.tensor([0.,.7,1.5,3.])
    t._box_states[ids,payload,:2]=torch.tensor([[1.,2.],[4.,5.],[-2.,3.],[8.,9.]])
    t._box_states[ids,payload,5]=torch.sin(yaw/2)
    t._box_states[ids,payload,6]=torch.cos(yaw/2)
    platform=JointCarryPlatform(t)
    platform.reset(ids)
    torch.testing.assert_close(platform.states[:,:2],t._box_states[ids,payload,:2])
    torch.testing.assert_close(platform.states[:,3:7],t._box_states[ids,payload,3:7])
    torch.testing.assert_close(platform.states[:,2]+.01,platform.height)
