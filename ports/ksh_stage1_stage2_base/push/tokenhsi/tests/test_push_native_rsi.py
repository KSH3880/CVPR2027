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
from utils.push_native_rsi import PushNativeRsiCache
from utils.stage1_push import validate_env,check_checkpoint,load_bank,sample_box_sizes
from utils.size_rsi import late_frames
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push


def config():
    return yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml').read_text())


def test_native_config_and_old_checkpoint_separation():
    cfg=config();env=cfg['env'];validate_env(env)
    old=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_rect_point.yaml').read_text())['env']
    for key in ('pushPointReward','relationReward','relationGraph','pushGoalRandomization'):
        assert env[key]==old[key]
    assert env['pushBox']==dict(size_range_m=[1.2,1.5],size_interval_m=.05,size_z_m=1.3,
                                mass=25.,friction=.6,restitution=0.)
    sizes=sample_box_sizes(512,'cpu',env)
    assert sizes[...,0:2].min()>=1.2 and sizes[...,0:2].max()<=1.5
    assert torch.all(sizes[...,2]==1.3)
    assert load_bank(env) is None
    assert env['pushTask']['late_probability']==.7 and env['pushTask']['late_fraction']==.3
    with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':old}},env)
    wrong=deepcopy(env);wrong['pushTask']['late_fraction']=.2
    with pytest.raises(ValueError):validate_env(wrong)


def test_no_arm_or_velocity_retarget():
    task=SimpleNamespace(cfg=config())
    values=tuple(torch.randn(4,3) for _ in range(6))
    result=HumanoidMAStage1Push._retarget_push_arms(task,*values,None)
    assert all(a is b for a,b in zip(values,result))


def test_frame_pools_use_native_valid_tail_and_clip_weights(tmp_path):
    torch.manual_seed(84)
    cfg=config();path=tmp_path/'cache.npz';spec=cfg['env']['pushTask']
    # Noncontiguous masks distinguish valid-frame tail from last30% of time.
    valid=np.array([[1,1,1,0,0,0,0,1,0,0, 1,0,1,0,1,0,0,0,0,0],
                    [0]*20],dtype=np.bool_)
    meta=dict(motion_hashes=spec['motion_hashes'],box=cfg['env']['pushBox'],sim=cfg['sim'])
    np.savez(path,valid=valid,sizes=np.array([[1.2]*3,[1.5]*3]),clip_offsets=np.array([0,10,20]),metadata=np.array(json.dumps(meta)))
    spec.update(rsi_cache_file=str(path),rsi_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    lib=SimpleNamespace(num_motions=lambda:2,_motion_weights=torch.tensor([.2,.8]),_motion_dt=torch.tensor([.1,.1]),_motion_lengths=torch.tensor([.9,.9]))
    task=SimpleNamespace(cfg=cfg,device='cpu',_motion_lib={'push':lib})
    cache=PushNativeRsiCache(task)
    assert cache.available(torch.tensor([[1.2]*3,[1.5]*3])).tolist()==[True,False]
    spec['late_probability']=1.
    sample=cache.sample(torch.full((20000,3),1.2))
    assert abs((sample['motion_id']==0).float().mean().item()-.2)<.015
    for clip in (0,1):
        native=late_frames(torch.from_numpy(valid[0,clip*10:clip*10+10]),torch.ones(10,dtype=torch.bool))
        frames=(sample['time'][sample['motion_id']==clip]/.1).round().long()
        assert set(frames.tolist())==set(native.tolist())
    spec['late_probability']=0.
    sample=cache.sample(torch.full((20000,3),1.2))
    for clip in (0,1):
        expected=np.flatnonzero(valid[0,clip*10:clip*10+10])
        frames=(sample['time'][sample['motion_id']==clip]/.1).round().long()
        assert set(frames.tolist())==set(expected.tolist())
    spec['rsi_cache_sha256']='0'*64
    with pytest.raises(ValueError):PushNativeRsiCache(task)
