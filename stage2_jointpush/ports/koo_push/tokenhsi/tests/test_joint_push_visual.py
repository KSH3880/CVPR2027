import numpy as np
from utils.joint_push_visual import overlay_lines


def test_distinct_initial_headings():
    roots=np.zeros((2,13));roots[:,6]=1
    roots[1,5:7]=[np.sin(np.pi/4),np.cos(np.pi/4)]
    lines,colors=overlay_lines([3.,4.,1.3],[2.6]*3,0.,roots,roots)
    assert lines.shape==(12,2,3) and colors.shape==(12,3)
    np.testing.assert_allclose(lines[0,1]-lines[0,0],[.8,0,0],atol=1e-6)
    np.testing.assert_allclose(lines[3,1]-lines[3,0],[0,.8,0],atol=1e-6)
