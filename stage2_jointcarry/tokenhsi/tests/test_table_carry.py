import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from learning.multi_agent.grasp_view import make_grasp_views,scene_parts
from env.tasks.multi_agent.joint_carry_reward import anchor_scores,opposite_gate
from utils.joint_carry_amp import check_ablation_checkpoint,validate_ablation
from utils.table_carry import verify_files
from test_joint_carry_grasp_view import observations

ROOT=Path(__file__).resolve().parents[1]

def config():
    return yaml.safe_load((ROOT/'data/cfg/multi_agent/table_undergrasp_carry.yaml').read_text())['env']

def test_rotated_under_table_anchor_is_same_in_observation_and_reward():
    obs=observations()[:1].clone();sides=torch.tensor([[-1.,1.]])
    objects,poses=scene_parts(obs,2)
    objects[:,0,3:6]=torch.tensor([0.,0.,2.])
    views=make_grasp_views(obs,sides,.15,anchor_z=.351).reshape(1,2,-1)
    _,original=scene_parts(obs,2);_,shifted=scene_parts(views,2)
    delta=shifted[:,:,2,:3]-original[:,None,2,:3]
    # Z offset is body-local, not a fixed world displacement.
    box=original[:,2].clone();q=box[:,3:7]
    local=torch.zeros(1,2,3);local[:,:,1]=sides*(objects[:,0,6:].reshape(1,8,3)[...,1].abs().amax(-1)[:,None]-.15);local[:,:,2]=.351
    from env.tasks.multi_agent.edge_interaction_reward import quat_rotate
    torch.testing.assert_close(delta,quat_rotate(q[:,None].expand(-1,2,-1),local))
    size=torch.tensor([[.52,.8,.4]])
    hands=shifted[:,:,2,:3][:,:,None].expand(-1,-1,2,-1)
    scores,_=anchor_scores(hands,box,size,.15,anchor_z=.351)
    assert opposite_gate(scores,.9).all()
    old,_=anchor_scores(hands,box,size,.15)
    assert not opposite_gate(old,.9).any()
    assert torch.equal(obs[:,-10:],views[:,0,-10:])

def test_table_checkpoint_cannot_mix_with_box_policy():
    env=config()
    with pytest.raises(ValueError,match='table-carry'):
        check_ablation_checkpoint({},env)
    with pytest.raises(ValueError,match='table-carry'):
        check_ablation_checkpoint({'table_carry':env['tableCarry']},{})

def test_table_files_and_source_network_contract():
    env=config();verify_files(env['tableCarry'])
    from utils.relation_task_spec import validate_relation_config
    validate_relation_config(env['relationReward'])
    train=yaml.safe_load((ROOT/'data/cfg/train/rlg/table_undergrasp_carry.yaml').read_text())
    validate_ablation(env,train)
    damaged=copy.deepcopy(env['tableCarry']);damaged['asset_sha256']='0'*64
    with pytest.raises(ValueError,match='input changed'):verify_files(damaged)

def test_initial_hands_below_slab_and_feet_stay_grounded():
    env=config();p=np.load(env['tableCarry']['initial_states']);b=p['body_state']
    assert len(b)>=4
    assert np.isfinite(b).all() and np.isfinite(p['dof_position']).all()
    hand=b[:,:,[5,8],:3]
    # Slab underside = local .39 + actor Z; hand spheres have radius .04.
    underside=p['box_state'][:,None,None,2]+.39
    assert np.abs(hand[...,2]+.04-underside).max()<.03
    assert (p['box_state'][:,2]-.43>.29).all()
    np.testing.assert_array_equal(p['dof_velocity'],0.)
