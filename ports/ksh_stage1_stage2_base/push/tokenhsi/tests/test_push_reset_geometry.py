from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
import torch
import yaml
import hashlib
import json
import numpy as np
from types import SimpleNamespace
from utils.push_reset_geometry import PushResetGeometry,reset_geometry_contract
from utils.stage1_push import validate_env,check_checkpoint
from utils.push_native_rsi import PushNativeRsiCache


def env(name='stage1_push_midpoint_safe'):
    return yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())['env']


def fixture():
    cfg=env();xml=Path(cfg['asset']['assetRoot'])/cfg['asset']['assetFileName']
    names=[b.get('name') for b in ET.parse(xml).getroot().findall('.//worldbody//body')]
    geometry=PushResetGeometry(cfg,names,'cpu')
    body=torch.full((1,len(names),3),-5.);body[...,2]=5.
    rot=torch.zeros(1,len(names),4);rot[...,3]=1
    box=torch.zeros(1,13);box[:,2]=.65;box[:,6]=1
    return geometry,names,body,rot,box,torch.full((1,3),1.3)


def test_foot_mesh_extent_is_checked_even_when_ankle_center_is_clear():
    g,names,body,rot,box,size=fixture();i=names.index('left_foot')
    body[:,i]=torch.tensor([0.,-.8,.05]);rot[:,i]=torch.tensor([0.,0.,2**-.5,2**-.5])
    assert not ((body[:,i]-box[:,:3]).abs()<size/2+.04).all(-1).any()
    assert g.invalid(body,rot,box,size).all()
    body[:,i,1]-=.4
    assert not g.invalid(body,rot,box,size).any()


def test_torso_radius_and_local_offset_are_checked():
    g,names,body,rot,box,size=fixture();i=names.index('torso')
    body[:,i]=torch.tensor([0.,-.74,.8])
    assert not ((body[:,i]-box[:,:3]).abs()<size/2+.04).all(-1).any()
    assert g.invalid(body,rot,box,size).all()
    body[:,i,1]-=.3  # The same torso body also owns both clavicle capsules.
    assert not g.invalid(body,rot,box,size).any()


def test_native_clearance_moves_whole_pose_without_joint_changes():
    g,names,body,rot,box,size=fixture();i=names.index('left_foot')
    body[:,i]=torch.tensor([0.,-.8,.05]);rot[:,i]=torch.tensor([0.,0.,2**-.5,2**-.5])
    shift=g.rear_shift(body,rot,box,size)
    assert 0<shift.item()<.4
    shifted=body.clone();shifted[...,1]-=shift[:,None]
    assert not g.invalid(shifted,rot,box,size).any()
    torch.testing.assert_close(shifted[:,1:]-shifted[:,:1],body[:,1:]-body[:,:1])


def test_floor_clearance_covers_mesh_below_an_ankle_above_ground():
    g,names,body,rot,box,size=fixture();i=names.index('left_foot')
    body[:,i]=torch.tensor([0.,-2.,.005])
    assert body[:,i,2].min()>0
    shift=g.floor_shift(body,rot)
    assert shift.min()>0
    raised=body.clone();raised[...,2]+=shift[:,None]
    assert g.floor_shift(raised,rot).max()<1e-5
    torch.testing.assert_close(raised[:,1:]-raised[:,:1],body[:,1:]-body[:,:1])


@pytest.mark.parametrize('name,old,distance',[
    ('stage1_push_midpoint_safe','stage1_push_midpoint_native_rsi',[1.,2.]),
    ('stage1_push_depth180_safe','stage1_push_depth180_native_rsi',[1.,5.])])
def test_configs_only_change_reset_contract_and_preserve_distance_reward(name,old,distance):
    cfg=env(name);prior=env(old);validate_env(cfg)
    assert cfg['pushTask']['loco_distance_range_m']==distance
    for key in ('pushBox','relationReward','pushPointReward','pushGoalRandomization','relationGraph'):
        assert cfg[key]==prior[key]
    assert reset_geometry_contract(cfg)['reset_geometry_filter']=='collider_bounds_v1'
    with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':prior}},cfg)
    check_checkpoint({'relation_experiment_config':{'env':deepcopy(cfg)}},cfg)
    wrong=deepcopy(cfg);wrong['pushTask']['reset_geometry_sha256']='0'*64
    with pytest.raises(ValueError):PushResetGeometry(wrong,fixture()[1],'cpu')


def test_native_cache_rejects_unscreened_reset_contract(tmp_path):
    cfg=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe.yaml').read_text())
    spec=cfg['env']['pushTask'];path=tmp_path/'cache.npz'
    metadata=dict(motion_hashes=spec['motion_hashes'],box=cfg['env']['pushBox'],sim=cfg['sim'])
    lib=SimpleNamespace(num_motions=lambda:1,_motion_weights=torch.ones(1))
    task=SimpleNamespace(cfg=cfg,device='cpu',_motion_lib={'push':lib})
    def write():
        np.savez(path,valid=np.ones((1,4),dtype=np.bool_),sizes=np.array([[1.3,1.3,1.3]]),clip_offsets=np.array([0,4]),metadata=np.array(json.dumps(metadata)))
        spec.update(rsi_cache_file=str(path),rsi_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write()
    with pytest.raises(ValueError,match='reset-collider contract'):
        PushNativeRsiCache(task)
    metadata['reset_geometry']=reset_geometry_contract(cfg['env']);write()
    assert PushNativeRsiCache(task).available(torch.tensor([[1.3,1.3,1.3]])).all()
