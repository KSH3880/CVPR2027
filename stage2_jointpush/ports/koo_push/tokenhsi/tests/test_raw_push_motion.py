from pathlib import Path
import numpy as np
import pytest
import torch
from utils.raw_push_motion import source_arrays, SourceAmpMotion, FORMAT, SOURCE_NAMES


def source_fixture():
    xyz=dict(Hips=[0,1,0],Spine2=[0,1.35,0],Chest=[0,1.45,0],Neck1=[0,1.55,0],Head=[0,1.7,0],
        RightArm=[0,1.5,-.2],RightForeArm=[.2,1.4,-.35],RightHand=[.45,1.38,-.35],
        LeftArm=[0,1.5,.2],LeftForeArm=[.2,1.4,.35],LeftHand=[.45,1.38,.35],
        RightLeg=[0,.95,-.1],RightShin=[.05,.5,-.1],RightFoot=[0,.05,-.1],RightToeBase=[.2,.05,-.1],
        LeftLeg=[0,.95,.1],LeftShin=[.05,.5,.1],LeftFoot=[0,.05,.1],LeftToeBase=[.2,.05,.1])
    names=list(xyz);nodes=[dict(name=name) for name in names]
    positions=np.tile(np.array([xyz[name] for name in names]),(5,1,1))
    positions[...,0]+=np.arange(5)[:,None]/30
    rotations=np.tile(np.eye(3),(5,len(names),1,1))
    return nodes,positions,rotations


def test_source_heights_lengths_and_speed_are_not_resized():
    nodes,pos,rot=source_fixture()
    output=source_arrays(nodes,pos,rot,1/30,0,4/30)
    np.testing.assert_allclose(output['global_translation'][:,0,2],1.)
    ids={n['name']:i for i,n in enumerate(nodes)}
    expected=np.linalg.norm(pos[:,ids['RightHand']]-pos[:,ids['LeftHand']],axis=-1)
    actual=np.linalg.norm(output['global_translation'][:,5]-output['global_translation'][:,8],axis=-1)
    np.testing.assert_allclose(actual,expected,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(output['global_velocity'][:,0],axis=-1),1.,atol=1e-6)
    larger=source_arrays(nodes,pos*1.5,rot,1/30,0,4/30)
    np.testing.assert_allclose(larger['global_translation'],output['global_translation']*1.5,atol=1e-6)
    np.testing.assert_allclose(larger['local_rotation'],output['local_rotation'],atol=1e-6)


def test_source_amp_container_roundtrip_and_contract(tmp_path):
    nodes,pos,rot=source_fixture()
    arrays=source_arrays(nodes,pos,rot,1/30,0,4/30)
    path=tmp_path/'source.npz';np.savez_compressed(path,**arrays)
    motion=SourceAmpMotion.from_file(path)
    assert motion.fps==30 and motion.num_joints==15 and motion.tensor.shape[0]==5
    torch.testing.assert_close(motion.global_translation,torch.tensor(arrays['global_translation']))
    assert motion.source_amp
    arrays['format']=np.array('humanoid_ik_corrected')
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='Not an unscaled'):
        SourceAmpMotion.from_file(path)


def test_default_profile_uses_converted_motion_manifest():
    import yaml
    root=Path(__file__).resolve().parents[1]
    env=yaml.safe_load((root/'data/cfg/multi_agent/partition_push_ca4.yaml').read_text())['env']
    assert env['partitionPush'].get('push_amp_source')!=FORMAT
    manifest=yaml.safe_load((root/'data/partition_push_motions.yaml').read_text())
    assert manifest['motions']['push']
    assert all(e['file'].endswith('ref_motion.npy') for e in manifest['motions']['push'])
    assert all(e['file'].endswith('ref_motion.npy') for e in manifest['motions']['loco'])
