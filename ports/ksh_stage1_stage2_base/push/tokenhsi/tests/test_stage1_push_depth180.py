"""Push along the long Y axis, with rear-face hands and a fixed 1.8m depth."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
import torch
import yaml
from utils.stage1_push import validate_env, sample_box_sizes, box_size_profiles, sample_boxes_and_goals, check_checkpoint
from utils.push_point import local_points, world_points
from utils.push_native_rsi import PushNativeRsiCache


def config():
    return yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_depth180_native_rsi.yaml').read_text())


def test_fixed_depth_pushes_rear_face_along_long_axis_and_preserves_reward():
    env=config()['env'];validate_env(env)
    assert env['pushTask']['loco_distance_range_m']==[1.,5.]
    old=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_native_rsi.yaml').read_text())['env']
    for key in ('relationReward','relationGraph','pushGoalRandomization','pushPointReward','skillDiscProb'):
        assert env[key]==old[key]
    torch.manual_seed(21);sizes=sample_box_sizes(10000,'cpu',env)
    assert (sizes[...,1]==1.8).all() and (sizes[...,2]==1.3).all()
    assert sizes[...,0].min()>=1.2 and sizes[...,0].max()<=1.5
    assert len(torch.unique(sizes.reshape(-1,3),dim=0))==7
    torch.testing.assert_close(torch.unique(sizes.reshape(-1,3),dim=0),torch.tensor(box_size_profiles(env)))
    boxes,goals,forward,distance=sample_boxes_and_goals(10000,'cpu',torch.zeros(10000,3),env,sizes)
    assert distance.min()>=1 and distance.max()<=4
    local=local_points(torch.full((10000,2),2),sizes[:,:2],torch.full((10000,2),.65))
    assert (local[...,0]==0).all()
    torch.testing.assert_close(local[...,1],torch.full_like(local[...,1],-.9))
    hands_target=world_points(boxes[:,:2],local)
    torch.testing.assert_close(hands_target-boxes[:,:2,:3],-.9*forward,atol=1e-6,rtol=0)
    torch.testing.assert_close(goals-boxes[:,:2,:3],distance[...,None]*forward,atol=1e-6,rtol=0)
    assert ((hands_target-boxes[:,:2,:3])*(goals-boxes[:,:2,:3])).sum(-1).max()<0
    with pytest.raises(ValueError):check_checkpoint({'relation_experiment_config':{'env':old}},env)
    wrong=deepcopy(env);wrong['pushBox']['size_y_m']=1.7
    with pytest.raises(ValueError):validate_env(wrong)


def test_native_cache_indexes_y180_without_changing_frame_pools(tmp_path):
    cfg=config();env=cfg['env'];spec=env['pushTask'];path=tmp_path/'depth.npz'
    profiles=np.array(box_size_profiles(env))
    meta=dict(motion_hashes=spec['motion_hashes'],box=env['pushBox'],sim=cfg['sim'])
    np.savez(path,valid=np.ones((7,4),dtype=np.bool_),sizes=profiles,clip_offsets=np.array([0,4]),metadata=np.array(json.dumps(meta)))
    spec.update(rsi_cache_file=str(path),rsi_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    lib=SimpleNamespace(num_motions=lambda:1,_motion_weights=torch.ones(1),_motion_dt=torch.tensor([.1]),_motion_lengths=torch.tensor([.3]))
    cache=PushNativeRsiCache(SimpleNamespace(cfg=cfg,device='cpu',_motion_lib={'push':lib}))
    torch.testing.assert_close(cache.profiles(torch.tensor(profiles)),torch.arange(7))
    assert cache.available(torch.tensor(profiles)).all()
    assert (cache.counts[...,0]==4).all()
    assert cache.sample(torch.tensor(profiles))['time'].isfinite().all()
    with pytest.raises(ValueError):cache.profiles(torch.tensor([[1.3,1.3,1.3]]))
