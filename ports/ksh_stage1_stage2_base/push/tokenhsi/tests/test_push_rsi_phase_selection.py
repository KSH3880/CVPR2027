"""Regression checks for the native, height-limited PUSH posture selection."""
from isaacgym import gymapi
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from utils.push_native_rsi import PushNativeRsiCache

MODULE = Path('tokenhsi/scripts/multi_agent/select_push_rsi_phase.py')
spec = importlib.util.spec_from_file_location('push_phase_selector', MODULE)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
DATA = Path('tokenhsi/data')
CACHE = DATA/'stage1_push_depth180_rsi_le1m_push_phase_cache.npz'
REPORT = DATA/'stage1_push_depth180_rsi_le1m_push_phase_selection.json'


def test_short_transition_rejected_and_both_run_ends_trimmed():
    mask=np.zeros(100,dtype=bool);mask[1:20]=True;mask[30:80]=True
    selected,windows=selector.sustained_windows(mask,30)
    assert windows==[[36,74]]
    assert np.array_equal(np.flatnonzero(selected),np.arange(36,74))


def test_selected_cache_is_safe_subset_without_height_retarget():
    with np.load(CACHE) as data, np.load(DATA/'stage1_push_depth180_safe_cache.npz') as old:
        mask=data['valid'];offset=data['clip_offsets'];metrics=data['posture_metrics']
        assert not (mask & ~old['valid']).any()
        assert mask.any(1).all()
        assert np.array_equal(data['sizes'],old['sizes'])
        assert np.array_equal(offset,old['clip_offsets'])
        selected=mask.any(0)
        assert (metrics[selected,0]>=.25).all() and (metrics[selected,1]>=-.25).all()
        assert not mask[:,offset[15]+2].any()  # User's arms-down example.
        assert mask[:,offset[8]+118].all()  # User's accepted pushing example.
        assert json.loads(str(data['metadata']))['height_selection']['max_m']==1.
        assert (data['placed_hand_midpoint_z_m'][selected]<=1.).all()
        assert (data['reference_hand_midpoint_z_m'][selected]<=1.).all()
        assert {c for c in range(21) if mask[:,offset[c]:offset[c+1]].any()}=={0,3,4,7,8}
        for c in range(21):
            local=selected[offset[c]:offset[c+1]]
            for a,b in np.flatnonzero(np.diff(np.r_[False,local,False])).reshape(-1,2):
                assert b-a>=15


def test_existing_sampler_including_late_pool_never_returns_preparation_pose():
    cfg=yaml.safe_load((DATA/'cfg/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg.yaml').read_text())
    cfg['env']['pushTask'].update(rsi_cache_file=str(CACHE),rsi_cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest())
    with np.load(CACHE) as data:
        offsets=data['clip_offsets'];valid=data['valid'];sizes=data['sizes']
    lib=SimpleNamespace(num_motions=lambda:21,_motion_weights=torch.full((21,),1/21),
        _motion_dt=torch.full((21,),1/30),_motion_lengths=torch.tensor(np.diff(offsets)-1,dtype=torch.float32)/30)
    cache=PushNativeRsiCache(SimpleNamespace(cfg=cfg,device='cpu',_motion_lib={'push':lib}))
    torch.manual_seed(1942)
    size=torch.tensor(sizes,dtype=torch.float32).repeat_interleave(2000,0)
    sample=cache.sample(size)
    clip=sample['motion_id'].numpy();frame=np.rint(sample['time'].numpy()*30).astype(int)
    profile=np.repeat(np.arange(7),2000)
    assert valid[profile,offsets[clip]+frame].all()
    assert set(clip)=={0,3,4,7,8}
    assert .67<float(sample['late_requested'].float().mean())<.73


def test_source_files_and_original_cache_unchanged_since_selection():
    report=json.loads(REPORT.read_text())
    source=DATA/'stage1_push_depth180_rsi_le1m_cache.npz'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==report['source_cache_sha256']
    for item in report['motions']:
        assert hashlib.sha256((DATA/item['file']).read_bytes()).hexdigest()==item['source_sha256']
