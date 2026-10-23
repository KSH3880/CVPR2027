"""Physical/cache contracts for the two user-requested common-face PUSH runs."""
from isaacgym import gymapi
from copy import deepcopy
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from utils.stage1_push import validate_env,check_checkpoint
from utils.push_native_rsi import PushNativeRsiCache

DATA=Path('tokenhsi/data')


def env_for(variant):
    return yaml.safe_load((DATA/'cfg/multi_agent'/f'stage1_push_face_xy_{variant}.yaml').read_text())['env']


@pytest.mark.parametrize('variant,depth,mass,friction',[('depth180_20kg',1.8,20.,.6),('depth200_25kg',2.,25.,.6),('depth200_20kg_friction04',2.,20.,.4)])
def test_physics_and_rescreen_contract(variant,depth,mass,friction):
    cfg=yaml.safe_load((DATA/'cfg/multi_agent'/f'stage1_push_face_xy_{variant}.yaml').read_text())
    env=cfg['env'];validate_env(env)
    assert env['pushBox']['friction']==friction
    assert env['pushBox']['mass']==mass and env['pushBox']['size_y_m']==depth
    assert env['pushBox']['size_range_m']==[1.2,1.5] and env['pushBox']['size_z_m']==1.3
    assert env['pushTask']['contact_objective']=='common_face_xy_v1'
    assert env['pushPointReward']=={}
    assert env['pushTask']['rsi_retarget']=='none'
    assert 'at_success_max_tilt_deg' not in env['pushTask']
    assert env['pushTask']['interaction_probability']==.5
    path=Path(env['pushTask']['rsi_cache_file'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==env['pushTask']['rsi_cache_sha256']
    with np.load(path) as data,np.load(DATA/'stage1_push_depth180_rsi_le1m_push_phase_cache.npz') as source:
        meta=json.loads(str(data['metadata']))
        assert meta['box']==env['pushBox'] and meta['sim']==cfg['sim']
        assert meta['rescreen']['seconds']==.1 and meta['rescreen']['steps']==6
        assert np.all(data['sizes'][:,1]==depth)
        assert not (data['valid'] & ~source['valid']).any()
        assert data['valid'].any(1).all()
        selected=data['valid'].any(0)
        assert (data['reference_hand_midpoint_z_m'][selected]<=1).all()
        assert (data['placed_hand_midpoint_z_m'][selected]<=1).all()


def test_only_depth_mass_and_cache_differ_and_checkpoint_rejects_cross_run():
    a,b=env_for('depth180_20kg'),env_for('depth200_25kg')
    normalized=deepcopy(b)
    normalized['pushBox'].update(size_y_m=a['pushBox']['size_y_m'],mass=a['pushBox']['mass'])
    for key in ('rsi_cache_file','rsi_cache_sha256'):normalized['pushTask'][key]=a['pushTask'][key]
    assert normalized==a
    with pytest.raises(ValueError,match='checkpoint objective/config differs'):
        check_checkpoint({'relation_experiment_config':{'env':a}},b)


@pytest.mark.parametrize('variant',['depth180_20kg','depth200_25kg','depth200_20kg_friction04'])
def test_sampler_both_full_and_late_pool_stay_in_verified_frames(variant):
    cfg=yaml.safe_load((DATA/'cfg/multi_agent'/f'stage1_push_face_xy_{variant}.yaml').read_text())
    env=cfg['env']
    with np.load(env['pushTask']['rsi_cache_file']) as data:
        offsets=data['clip_offsets'];valid=data['valid'];sizes=data['sizes']
    lib=SimpleNamespace(num_motions=lambda:21,_motion_weights=torch.full((21,),1/21),_motion_dt=torch.full((21,),1/30),_motion_lengths=torch.tensor(np.diff(offsets)-1,dtype=torch.float32)/30)
    cache=PushNativeRsiCache(SimpleNamespace(cfg=cfg,device='cpu',_motion_lib={'push':lib}))
    torch.manual_seed(1942);sample=cache.sample(torch.tensor(sizes,dtype=torch.float32).repeat_interleave(2000,0))
    clips=sample['motion_id'].numpy();frames=np.rint(sample['time'].numpy()*30).astype(int)
    assert valid[np.repeat(np.arange(7),2000),offsets[clips]+frames].all()
    assert set(clips)=={0,3,4,7,8}
    assert .67<float(sample['late_requested'].float().mean())<.73


def test_friction04_changes_only_mass_friction_and_cache():
    previous=env_for('depth200_25kg');current=env_for('depth200_20kg_friction04')
    normalized=deepcopy(current)
    normalized['pushBox'].update(mass=previous['pushBox']['mass'],friction=previous['pushBox']['friction'])
    for key in ('rsi_cache_file','rsi_cache_sha256'):normalized['pushTask'][key]=previous['pushTask'][key]
    assert normalized==previous
    with pytest.raises(ValueError,match='checkpoint objective/config differs'):
        check_checkpoint({'relation_experiment_config':{'env':previous}},current)
