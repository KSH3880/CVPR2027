"""Y1.8m height comparison keeps state>0.9 success without separate distance gates."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
import torch
import yaml
from utils.push_point import local_points, world_points, midpoint_reward_phi, midpoint_position_contact, midpoint_force_contact
from utils.stage1_push import validate_env, check_checkpoint
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push


def config(tag):
    return yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe_{tag}_relaxed.yaml').read_text())['env']


@pytest.mark.parametrize('tag',['h065','hamp','h100'])
def test_config_and_checkpoint_contract(tag):
    env=config(tag);validate_env(env)
    assert env['pushBox']['size_y_m']==1.8 and env['pushBox']['size_z_m']==1.3
    assert env['pushBox']['mass']==20 and env['pushTask']['require_contact_force'] is True
    if tag=='h100':
        assert env['pushTask']['height_m']==1.0 and env['pushTask'].get('height_mode','fixed')=='fixed'
    assert env['pushTask']['min_hand_force_n']==env['pushTask']['min_box_force_n']==.5
    assert set(env['pushPointReward'])=={'xy_reward_scale','z_reward_sigma_m'}
    old=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_depth180_safe.yaml').read_text())['env']
    for key in ['pushBox','relationReward','relationGraph','pushGoalRandomization']:
        assert env[key]==old[key]
    assert env['pushTask']['rsi_cache_file']==old['pushTask']['rsi_cache_file']
    assert env['pushTask']['rsi_cache_sha256']==old['pushTask']['rsi_cache_sha256']
    assert env['pushTask']['rsi_retarget']=='none'
    check_checkpoint({'relation_experiment_config':{'env':env}},env)
    for other in [old,config('hamp' if tag=='h065' else 'h065')]:
        with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':other}},env)
    bad=deepcopy(env);bad['pushPointReward']['xy_tolerance_m']=.15
    with pytest.raises(ValueError):validate_env(bad)


@pytest.mark.parametrize('xy,z,passed',[(.149,0,True),(0,.049,True),(.151,0,False),(0,.051,False),(.14,.04,False)])
def test_only_state_threshold_and_force_decide_success(xy,z,passed):
    spec=config('h065')['pushPointReward'];boxes=torch.zeros(1,1,13);boxes[...,2]=.65;boxes[...,6]=1
    size=torch.tensor([[[1.3,1.8,1.3]]]);side=torch.full((1,1),2)
    local=local_points(side,size,torch.full((1,1),.65))
    hands=world_points(boxes,local)[...,None,:].expand(-1,-1,2,-1).clone()
    hands[...,0]+=xy;hands[...,2]+=z
    phi=midpoint_reward_phi(hands,boxes,local,spec)
    assert phi.item()==pytest.approx(math.exp(-spec['xy_reward_scale']*xy**2-(z/spec['z_reward_sigma_m'])**2),abs=1e-6)
    assert midpoint_position_contact(hands,boxes,local,spec).item()==passed
    forces=torch.zeros_like(hands);forces[...,1]=-1
    box_forces=torch.tensor([[[0.,1.,0.]]])
    push=config('h065')['pushTask']
    assert midpoint_force_contact(hands,forces,boxes,box_forces,size,side,local,spec,push).item()==passed
    assert not midpoint_force_contact(hands,forces*0,boxes,box_forces,size,side,local,spec,push).any()


def test_amp_target_uses_reference_height_preserves_other_slots_and_does_not_follow_policy():
    task=SimpleNamespace(_key_body_ids=torch.tensor([0,1]),_box_size=torch.full((3,2,3),1.3),
                         _push_local=torch.zeros(3,2,3))
    body=torch.zeros(2,2,3);body[:,:,2]=torch.tensor([[.9,1.1],[.7,.9]])
    rows=torch.tensor([2,0]);owners=torch.tensor([0,1])
    before=task._push_local.clone()
    HumanoidMAStage1Push._set_amp_midpoint_height(task,rows,owners,body)
    torch.testing.assert_close(task._push_local[rows,owners,2]+.65,torch.tensor([1.,.8]))
    assert torch.equal(task._push_local[1],before[1])
    saved=task._push_local.clone();body[:,:,2]+=.1
    assert torch.equal(task._push_local,saved)
    body[:,:,2]=1.4
    with pytest.raises(ValueError,match='outside'):HumanoidMAStage1Push._set_amp_midpoint_height(task,rows,owners,body)
