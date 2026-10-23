"""Native RSI height-only frame selection and two isolated training contracts."""
from isaacgym import gymapi
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from utils.stage1_push import validate_env,check_checkpoint
from utils.push_native_rsi import PushNativeRsiCache

NAMES=['stage1_push_depth180_h100_rsi1m_at5deg','stage1_push_depth180_h080_rsi1m']


def config(name=NAMES[0]):
    return yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())


@pytest.mark.parametrize('name,height,angle',[(NAMES[0],1.,5.),(NAMES[1],.8,None)])
def test_experiment_contracts(name,height,angle):
    cfg=config(name);env=cfg['env'];validate_env(env);spec=env['pushTask']
    assert env['pushBox']==dict(size_range_m=[1.2,1.5],size_interval_m=.05,size_z_m=1.3,mass=20.,friction=.6,restitution=0.,size_y_m=1.8)
    assert env['pushGoalRandomization']['distance_range']==[1.,4.]
    assert spec['loco_distance_range_m']==[1.,5.]
    assert spec['height_m']==height and spec.get('at_success_max_tilt_deg')==angle
    assert spec['rsi_max_hand_midpoint_height_m']==1. and spec['rsi_retarget']=='none'
    assert spec['interaction_probability']==.5 and spec['late_probability']==.7
    assert spec['require_contact_force'] and spec['min_hand_force_n']==spec['min_box_force_n']==.5
    check_checkpoint({'relation_experiment_config':{'env':env}},env)
    other=config(NAMES[1] if name==NAMES[0] else NAMES[0])['env']
    with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':other}},env)
    bad=deepcopy(env);bad['pushTask'].pop('rsi_max_hand_midpoint_height_m')
    with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':bad}},env)


@pytest.mark.parametrize('limit',[0,-1,True,None,float('nan'),float('inf'),2.])
def test_invalid_selection_limit(limit):
    env=config()['env'];env['pushTask']['rsi_max_hand_midpoint_height_m']=limit
    with pytest.raises(ValueError,match='height selection'):validate_env(env)


def fake_cache(tmp_path,change=None):
    cfg=config();env=cfg['env'];spec=env['pushTask']
    meta=dict(motion_hashes=spec['motion_hashes'],box=env['pushBox'],sim=cfg['sim'],
        reset_geometry={k:spec[k] for k in ['reset_geometry_filter','reset_geometry_margin_m','reset_floor_clearance_m','reset_geometry_sha256']},
        height_selection=dict(kind='native_hand_midpoint_z',max_m=1.,include_safe_floor_shift=True))
    payload=dict(valid=np.array([[1,0,1,0],[1,0,1,0]],dtype=np.bool_),sizes=np.array([[1.2,1.8,1.3],[1.5,1.8,1.3]]),
        clip_offsets=np.array([0,2,4]),reference_hand_midpoint_z_m=np.array([.9,1.1,.99,1.2]),
        placed_hand_midpoint_z_m=np.array([.92,1.1,1.,1.2]))
    if change:change(meta,payload)
    payload['metadata']=np.array(json.dumps(meta));path=tmp_path/'cache.npz';np.savez(path,**payload)
    spec.update(rsi_cache_file=str(path),rsi_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    lib=SimpleNamespace(num_motions=lambda:2,_motion_weights=torch.tensor([.25,.75]),_motion_dt=torch.tensor([.1,.1]),_motion_lengths=torch.tensor([.1,.1]))
    return PushNativeRsiCache(SimpleNamespace(cfg=cfg,device='cpu',_motion_lib={'push':lib}))


def test_filtered_pool_samples_only_allowed_frames_and_keeps_motion_weights(tmp_path):
    torch.manual_seed(21);cache=fake_cache(tmp_path)
    assert cache.available(torch.tensor([[1.2,1.8,1.3],[1.5,1.8,1.3]])).all()
    sample=cache.sample(torch.tensor([[1.2,1.8,1.3]]).repeat(10000,1))
    assert (sample['time']==0).all()
    assert sample['motion_id'].float().mean().item()==pytest.approx(.75,abs=.02)
    assert sample['late_requested'].float().mean().item()==pytest.approx(.7,abs=.02)


@pytest.mark.parametrize('change',[
    lambda meta,payload:meta.pop('height_selection'),
    lambda meta,payload:payload['placed_hand_midpoint_z_m'].__setitem__(0,1.001),
    lambda meta,payload:payload['reference_hand_midpoint_z_m'].__setitem__(0,1.001),
    lambda meta,payload:payload['valid'].__setitem__(1,False),
    lambda meta,payload:payload['placed_hand_midpoint_z_m'].__setitem__(0,float('nan'))])
def test_reject_unfiltered_leaking_or_empty_profiles(tmp_path,change):
    with pytest.raises(ValueError):fake_cache(tmp_path,change)


def test_real_cache_is_a_subset_and_both_height_measurements_pass():
    cfg=config();spec=cfg['env']['pushTask']
    with np.load(spec['rsi_cache_file'],allow_pickle=False) as data:
        mask=data['valid'];selected=mask.any(axis=0)
        assert mask.any(axis=1).all()
        assert (data['reference_hand_midpoint_z_m'][selected]<=1.).all()
        assert (data['placed_hand_midpoint_z_m'][selected]<=1.).all()
        with np.load('tokenhsi/data/stage1_push_depth180_safe_cache.npz',allow_pickle=False) as original:
            assert not (mask & ~original['valid']).any()
            assert np.array_equal(data['sizes'],original['sizes'])
            assert np.array_equal(data['clip_offsets'],original['clip_offsets'])
