"""Conversion invariants that catch coordinate, retarget and config regressions."""
from isaacgym import gymapi  # Isaac Gym must precede PyTorch/Poselib.
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import pytest
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from tokenhsi.utils.bones_bvh import read_bvh
from tokenhsi.scripts.multi_agent.convert_bones_door import DOF_BODIES, DOF_OFFSETS, joint_bounds

ROOT = Path(__file__).resolve().parents[2]
NAME = 'inside_door_handle_right_side_open_walk_R_001__A515_M'
OUTPUT = ROOT/'tokenhsi/data/dataset_bones_door/motions'/NAME/'phys_humanoid_v3'


def test_soma_absolute_root_and_units():
    path = ROOT/'bones_seed/soma_uniform/bvh/231115'/f'{NAME}.bvh'
    nodes, pos, rotations, dt = read_bvh(path)
    text = path.read_text().split('MOTION', 1)[1].strip().splitlines()
    first = np.fromstring(text[2], sep=' ')
    hips = next(i for i,n in enumerate(nodes) if n['name']=='Hips')
    np.testing.assert_allclose(pos[0, hips], first[6:9]/100, atol=1e-8)
    assert len(pos)==621 and .008 < dt < .009
    assert .8 < pos[:, hips, 1].min() < 1.2
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)


def test_target_motion_can_represent_pose_without_axis_or_velocity_artifacts():
    motion = SkeletonMotion.from_file(str(OUTPUT/'ref_motion.npy'))
    audit = json.loads((OUTPUT/'conversion_report.json').read_text())
    q = motion.local_rotation.numpy()
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5)
    assert q.shape == (103,15,4) and motion.fps==30
    for data in (motion.global_translation, motion.global_velocity, motion.global_angular_velocity):
        assert np.isfinite(data.numpy()).all()
    rv = Rotation.from_quat(q.reshape(-1,4)).as_rotvec().reshape(-1,15,3)
    values = np.concatenate([rv[:,b] if hi-lo==3 else rv[:,b,1:2]
                             for b,lo,hi in zip(DOF_BODIES,DOF_OFFSETS[:-1],DOF_OFFSETS[1:])],axis=-1)
    lo,hi = joint_bounds(motion.skeleton_tree)
    assert (values>=lo-1e-5).all() and (values<=hi+1e-5).all()
    assert np.max(np.abs(np.diff(values,axis=0))*30)<10
    np.testing.assert_allclose(q[:,[5,8]], np.broadcast_to([0,0,0,1],(103,2,4)),atol=1e-5)
    assert audit['foot_mesh_floor_min_m']>=0
    assert max(audit['key_body_fit_error_max_m'])<.05
    assert audit['root_net_horizontal_m']>1.5
    assert .9<audit['left_hand_open_height_m']['median']<1.1


def test_open_expert_separation_preserves_loco_reset_and_existing_task():
    cfgdir=ROOT/'tokenhsi/data/cfg/multi_agent'
    old=yaml.safe_load((cfgdir/'door_open_unified_approach_amp.yaml').read_text())
    new=yaml.safe_load((cfgdir/'door_open_bones_left_amp.yaml').read_text())
    for key in ('skill','skillInitProb','skillDiscProb','ampExpertMapping'):
        old['env'].pop(key);new['env'].pop(key)
    # The original push experiment subsequently added a resistant hinge;
    # BONES intentionally retains its original physical defaults.
    hinge=old['env']['door'].pop('hinge',None)
    assert hinge is None or hinge in ({'damping':5.0,'friction':.5}, {'damping':2.0,'friction':.1})
    assert 'hinge' not in new['env']['door']
    assert old==new
    cfg=yaml.safe_load((cfgdir/'door_open_bones_left_amp.yaml').read_text())['env']
    assert cfg['skill']==['loco','doorOpen'] and cfg['skillInitProb']==[1,0]
    assert cfg['ampExpertMapping']['OPEN']==[.3,.7]
    assert cfg['doorApproachAMP']['gate_enabled'] is False
    manifest_path=ROOT/'tokenhsi/data/dataset_door_bones_left.yaml'
    manifest=yaml.safe_load(manifest_path.read_text())
    for entries in manifest['motions'].values():
        for entry in entries:
            assert (manifest_path.parent/entry['file']).is_file()


def test_new_experts_allowed_only_for_full_push_amp_with_loco_reset():
    from utils.edge_door_spec import validate_env
    from copy import deepcopy
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/door_open_bones_left_amp.yaml').read_text())['env']
    validate_env(cfg)
    for change in ('gate', 'pull', 'reset', 'mixture'):
        bad=deepcopy(cfg)
        if change=='gate': bad['doorApproachAMP']['gate_enabled']=True
        elif change=='pull': bad['door']['openingDirection']='pull'
        elif change=='reset': bad['skillInitProb']=[.5,.5]
        else: bad['skillDiscProb']=[1.,0.]
        with pytest.raises(ValueError): validate_env(bad)
