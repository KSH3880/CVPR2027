from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from utils.stage1_push import validate_env, sample_box_sizes, sample_boxes_and_goals, check_checkpoint, box_tilt_degrees, load_bank
from utils.push_point import local_points, world_points, point_reward_phi, point_reward_contact


def config():
    return yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_rect_point.yaml').read_text())['env']


def test_config_preserves_native_at_and_architecture():
    env=config();validate_env(env)
    old=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml').read_text())['env']
    for key in ('relationReward','relationGraph','skillDiscProb','numAgents','numObjects','pushGoalRandomization'):
        assert env[key]==old[key]
    assert env['pushTask']['height_m']==.65 and not env['pushTask']['require_contact_force']
    root=Path('tokenhsi/data/cfg/train/rlg')
    a,b=[yaml.safe_load((root/f'{name}.yaml').read_text()) for name in ('stage1_push_no_force','stage1_push_rect_point')]
    assert a['params']['network']==b['params']['network']


def test_independent_xyz_grid_and_physical_size_reset():
    env=config();torch.manual_seed(13)
    sizes=sample_box_sizes(10000,'cpu',env)
    assert sizes.shape==(10000,4,3)
    assert sizes.min()>=1.2 and sizes.max()<=1.5
    assert len(torch.unique(sizes.reshape(-1,3),dim=0))==343
    assert (sizes[...,0]!=sizes[...,1]).float().mean()>.8
    corr=torch.corrcoef(sizes.reshape(-1,3).T)
    assert (corr-torch.eye(3)).abs().max()<.02
    states,goals,fwd,distance=sample_boxes_and_goals(10000,'cpu',torch.zeros(10000,3),env,sizes)
    torch.testing.assert_close(states[...,2],sizes[...,2]/2)
    torch.testing.assert_close(goals[...,2],sizes[:,:2,2]/2)
    torch.testing.assert_close(goals-states[:,:2,:3],fwd*distance[...,None],atol=1e-6,rtol=0)
    assert distance.min()>=1 and distance.max()<=5
    with pytest.raises(ValueError):sample_boxes_and_goals(1,'cpu',torch.zeros(1,3),env)


@pytest.mark.parametrize('size',[[1.2,1.2,1.2],[1.2,1.5,1.5],[1.5,1.2,1.35]])
def test_point_is_rear_centerline_at_absolute_initial_height_and_follows_box(size):
    box=torch.zeros(1,1,13);box[...,2]=size[2]/2;box[...,6]=1
    side=torch.full((1,1),2);local=local_points(side,size,torch.full((1,1),.65))
    target=world_points(box,local)
    torch.testing.assert_close(target,torch.tensor([[[0.,-size[1]/2,.65]]]))
    box[...,:2]+=torch.tensor([2.,3.])
    torch.testing.assert_close(world_points(box,local),target+torch.tensor([2.,3.,0.]))
    box[...,3:7]=torch.tensor([0.,0.,2**-.5,2**-.5])
    torch.testing.assert_close(world_points(box,local),torch.tensor([[[2+size[1]/2,3.,.65]]]))
    assert box_tilt_degrees(box[...,3:7]).item()==0


def test_midpoint_objective_allows_different_hand_heights_and_spacing_without_force():
    env=config();box=torch.tensor([[[0.,0.,.6,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]]])
    size=torch.tensor([[[1.2,1.2,1.2]]]);side=torch.full((1,1),2)
    local=local_points(side,size,torch.full((1,1),.65));limits=dict(hand_distance=.18,require_contact_force=False)
    hands=torch.tensor([[[[-.4,-.6,.50],[.4,-.6,.80]]]])
    def score(h):return point_reward_phi(h,box,size,side,local,env['pushPointReward']).item()
    def contact(h):return point_reward_contact(h,torch.zeros_like(h),box,torch.zeros(1,1,3),size,limits,side,local,env['pushPointReward']).item()
    assert contact(hands) and score(hands)>.99999
    other=hands.clone();other[...,0]*=.5
    assert contact(other) and abs(score(other)-score(hands))<1e-6
    high=hands.clone();high[...,2]+=.04
    assert not contact(high) and score(high)<score(hands)
    far=hands.clone();far[...,0,1]-=.25;far[...,1,1]+=.25
    assert not contact(far)  # Correct midpoint alone cannot qualify hands far from the rear face.


def test_checkpoint_separates_all_previous_profiles_and_physical_contract():
    env=config();saved={'relation_experiment_config':{'env':deepcopy(env)}}
    check_checkpoint(saved,env)
    for name in ('stage1_v2_fixed','stage1_push','stage1_push_no_force'):
        old=yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())['env']
        with pytest.raises(ValueError):check_checkpoint(saved,old)
        with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':old}},env)
    wrong=deepcopy(env);wrong['pushBox']['mass']=25
    with pytest.raises(ValueError):validate_env(wrong)
    with pytest.raises(ValueError):check_checkpoint(saved,wrong)


def test_only_new_screened_bank_is_accepted():
    env=config();bank=load_bank(env)
    assert len(bank['candidates'])==3 and bank['screen']['shapes']==343 and bank['screen']['owners']==2
    assert bank['screen_status']=='passed_all_rectangular_sizes'
    wrong=deepcopy(env);wrong['pushTask']['bank_sha256']='0'*64
    with pytest.raises(ValueError):load_bank(wrong)
    wrong=deepcopy(env);wrong['pushTask']['bank_file']='tokenhsi/data/stage1_push_bank.json'
    with pytest.raises(ValueError):load_bank(wrong)
