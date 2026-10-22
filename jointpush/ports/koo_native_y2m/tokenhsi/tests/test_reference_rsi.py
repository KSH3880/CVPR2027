"""Reference RSI preserves physical sizes, source clips and velocity consistency."""
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
import yaml
from utils.reference_rsi import ReferenceRSIBank, digest, interpolate_object_velocity, validate_reference_rsi, REFERENCE_SKILLS, SceneGeometryScreen


def fixture_bank(tmp_path, mode='time_late', fraction=1.):
    dataset = tmp_path/'data.yaml'; dataset.write_text('dataset')
    asset = tmp_path/'asset.xml'; asset.write_text('<mujoco/>')
    motion = tmp_path/'motion.npy'; motion.write_text('motion')
    obj = tmp_path/'obj.npy'; obj.write_text('object')
    meta = {'schema':1,'dataset_sha256':digest(dataset),'asset_sha256':digest(asset),
            'files':{},'objects':{},'motion_sha256':{},'object_sha256':{},'coverage':{}}
    arrays = {'sizes':np.array([[.4,.4,.3],[.65,.65,.55]],dtype=np.float32)}
    libs = {}
    for skill in REFERENCE_SKILLS:
        # Early semantic progress deliberately differs from elapsed late phase.
        for k,v in {'times':[.2,.8],'motion_ids':[0,0],'phase':[.2,.8],
                    'allowed':[True,True],'safe':[[True,True],[False,False]],
                    'semantic':[[True,False],[False,False]]}.items():
            arrays[skill+'_'+k] = np.array(v)
        meta['files'][skill] = ['motion.npy']; meta['objects'][skill] = ['obj.npy']
        meta['motion_sha256'][skill] = [digest(motion)]
        meta['object_sha256'][skill] = [digest(obj)]
        libs[skill] = SimpleNamespace(_motion_files=[str(motion)],_motion_weights=torch.ones(1))
    arrays['metadata'] = np.array(json.dumps(meta))
    path = tmp_path/'bank.npz'; np.savez_compressed(path,**arrays)
    spec = {'mode':mode,'bank':str(path),'late_fraction':fraction,'phase_range':[.65,.98],
            'biased_skills':['climb'],'empty_skill':'loco','restore_box_velocity':True}
    return ReferenceRSIBank(spec,libs,dataset,asset,'cpu')


def test_time_and_semantic_are_different_original_reference_times(tmp_path):
    sizes = torch.tensor([[.4,.4,.3]]).repeat(100,1)
    for mode,expected in [('time_late',.8),('semantic_late',.2)]:
        bank = fixture_bank(tmp_path,mode)
        ids,times,empty = bank.sample('climb',sizes)
        assert not empty.any() and ids.eq(0).all()
        torch.testing.assert_close(times,torch.full((100,),expected))


def test_missing_size_pool_is_reported_without_resizing(tmp_path):
    bank = fixture_bank(tmp_path)
    sizes = torch.tensor([[.65,.65,.55]])
    before = sizes.clone()
    _,_,empty = bank.sample('carryWith',sizes)
    assert empty.all() and bank.stats['empty_skill']==1
    torch.testing.assert_close(sizes,before)
    with pytest.raises(ValueError,match='not covered'):
        bank.sample('climb',torch.tensor([[.5,.5,.5]]))


def test_preferred_fallback_and_legacy_climb_skip_override(tmp_path):
    bank = fixture_bank(tmp_path)
    bank.data['climb']['allowed'][1] = False
    # Preferred late CLIMB may use the old skipped range, as in slow_near_start.
    _,times,empty = bank.sample('climb',torch.tensor([[.4,.4,.3]]))
    assert not empty.any() and times.item()==pytest.approx(.8)
    other = fixture_bank(tmp_path)
    other.data['climb']['safe'][0,1] = False
    _,times,empty = other.sample('climb',torch.tensor([[.4,.4,.3]]))
    assert not empty.any() and times.item()==pytest.approx(.2)
    assert other.stats['preferred_fallback']==1


def test_late_fraction_statistical_sampling(tmp_path):
    torch.manual_seed(3)
    bank = fixture_bank(tmp_path,fraction=.7)
    _,times,_ = bank.sample('climb',torch.tensor([[.4,.4,.3]]).repeat(10000,1))
    # 70% forced late plus half the 30% general pool = 85% total late.
    assert .83 < float((times>.65).float().mean()) < .87


def test_motion_content_fingerprint(tmp_path):
    bank = fixture_bank(tmp_path)
    dataset = tmp_path/'data.yaml'; dataset.write_text('changed')
    libs = {k:SimpleNamespace(_motion_files=[str(tmp_path/'motion.npy')],_motion_weights=torch.ones(1)) for k in REFERENCE_SKILLS}
    with pytest.raises(ValueError,match='fingerprint'):
        ReferenceRSIBank(bank.spec,libs,dataset,tmp_path/'asset.xml','cpu')


def test_reference_velocity_in_world_frame_including_last_segment():
    class Lib:
        _motion_lengths=torch.tensor([2.])
        _motion_num_frames=torch.tensor([3])
        _motion_dt=torch.tensor([1.])
        def get_obj_motion_state(self,ids,t):
            angle=t*torch.pi/2
            pos=torch.stack([t,torch.zeros_like(t),torch.zeros_like(t)],-1)
            q=torch.stack([torch.zeros_like(t),torch.zeros_like(t),torch.sin(angle/2),torch.cos(angle/2)],-1)
            return pos,q
    v,w=interpolate_object_velocity(Lib(),torch.zeros(3,dtype=torch.long),torch.tensor([0.,.5,2.]))
    torch.testing.assert_close(v,torch.tensor([[1.,0.,0.]]).repeat(3,1))
    torch.testing.assert_close(w,torch.tensor([[0.,0.,torch.pi/2]]).repeat(3,1))


def test_experiment_configs_keep_pid289953_contract():
    root=Path(__file__).resolve().parents[1]/'data/cfg/multi_agent'
    baseline=yaml.safe_load((root/'approach_scenario_stage1_unified_shared_random_binding.yaml').read_text())
    for mode in ('time_late','semantic_late'):
        cfg=yaml.safe_load((root/f'approach_reference_rsi_{mode}.yaml').read_text())
        validate_reference_rsi(cfg['env'].pop('referenceRsi'))
        assert cfg==baseline


def test_scene_screen_catches_teammate_box_and_allows_shallow_contact(tmp_path):
    asset=tmp_path/'body.xml'
    asset.write_text('<mujoco><worldbody><body name="hand"><geom type="sphere" size=".04"/></body></worldbody></mujoco>')
    screen=SceneGeometryScreen(asset,['hand'],'cpu')
    bodies=torch.zeros(1,2,1,13); bodies[...,6]=1
    bodies[0,0,0,0]=2.
    boxes=torch.zeros(1,1,13);boxes[...,6]=1
    sizes=torch.tensor([[[.4,.4,.4]]])
    # Teammate is inside a box even if its owner is far away.
    assert screen.invalid(bodies,boxes,sizes).item()
    bodies[0,1,0,0]=.235  # 5mm overlap of hand sphere with box face.
    assert not screen.invalid(bodies,boxes,sizes).item()
    bodies[0,1,0,0]=.22
    assert screen.invalid(bodies,boxes,sizes).item()
