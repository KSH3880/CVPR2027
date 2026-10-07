"""BONES coordinate regressions, independent of generated datasets or Isaac Gym."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tokenhsi.utils.bones_bvh import read_bvh


def fixture(tmp_path, wrapper='0 0 0 0 0 0', hips='10 100 20 30 40 50'):
    path=tmp_path/'example.bvh'
    path.write_text("""HIERARCHY
ROOT Root
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Hips
{
OFFSET 0 999 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Hand
{
OFFSET 20 0 0
CHANNELS 3 Zrotation Yrotation Xrotation
End Site { OFFSET 1 0 0 }
}
}
}
MOTION
Frames: 2
Frame Time: 0.008333
"""+f'{wrapper} {hips} 0 0 0\n'*2)
    return path


def test_absolute_hips_translation_ignores_offset(tmp_path):
    nodes,pos,rot,dt=read_bvh(fixture(tmp_path))
    assert len(nodes)==3 and pos.shape==(2,3,3)
    np.testing.assert_allclose(pos[0,1],[.1,1,.2])
    assert dt==.008333


def test_intrinsic_channel_order_and_fk(tmp_path):
    _,pos,rot,_=read_bvh(fixture(tmp_path))
    expected=Rotation.from_euler('ZYX',[30,40,50],degrees=True).as_matrix()
    np.testing.assert_allclose(rot[0,1],expected)
    np.testing.assert_allclose(pos[0,2],pos[0,1]+expected@np.array([.2,0,0]))
    np.testing.assert_allclose(np.linalg.det(rot),1,atol=1e-8)


@pytest.mark.parametrize('wrapper,hips',[('1 0 0 0 0 0','10 100 20 0 0 0'),('0 0 0 0 0 0','nan 100 20 0 0 0')])
def test_reject_moving_wrapper_and_nonfinite(tmp_path,wrapper,hips):
    with pytest.raises(ValueError):
        read_bvh(fixture(tmp_path,wrapper,hips))
