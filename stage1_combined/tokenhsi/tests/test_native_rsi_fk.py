from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rsi_native.repair import NativeFK


def test_native_fk_hands_follow_elbow_and_have_no_extra_joint():
    xml=Path(__file__).resolve().parents[1]/'data/assets/mjcf/phys_humanoid_v3.xml'
    fk=NativeFK(xml)
    root=np.zeros((1,13));root[:,6]=1
    joint=np.zeros((1,32));elbow=fk.names.index('right_lower_arm');hand=fk.names.index('right_hand')
    offset,size=fk.joints[elbow];assert size==1
    joint[:,offset]=-.8
    pos,quat=fk.forward(root,joint)
    np.testing.assert_allclose(pos[:,hand]-pos[:,elbow],Rotation.from_quat(quat[:,elbow]).apply(fk.offsets[hand][None]),atol=1e-12)
    np.testing.assert_allclose(quat[:,hand],quat[:,elbow],atol=1e-12)
    state=fk.state(root,joint,np.zeros_like(joint))
    np.testing.assert_allclose(state[...,7:],0,atol=1e-12)
    assert np.isfinite(state).all()


def test_native_joint_clamp_is_bounded_and_idempotent():
    xml=Path(__file__).resolve().parents[1]/'data/assets/mjcf/phys_humanoid_v3.xml'
    fk=NativeFK(xml);joint=np.linspace(-4,4,32)[None]
    clamped=np.clip(joint,fk.lower,fk.upper)
    assert ((clamped>=fk.lower)&(clamped<=fk.upper)).all()
    np.testing.assert_array_equal(np.clip(clamped,fk.lower,fk.upper),clamped)
