import numpy as np
import trimesh
from rsi_native.geometry import NativeGeometry
from rsi_native.support import supported,project_ground,foot_bottom


def test_vertical_support_rejects_hover_and_side_wall(tmp_path):
    trimesh.creation.box(extents=[2,2,2]).export(tmp_path/'cube.obj')
    (tmp_path/'asset.urdf').write_text('<robot name="cube"><link name="cube"><collision><geometry><mesh filename="cube.obj"/></geometry></collision></link></robot>')
    shapes=(np.array([0,11]),np.zeros((2,3),dtype=np.float32),np.array([.1,.1],dtype=np.float32));geo=NativeGeometry(tmp_path,shapes)
    body=np.zeros((4,15,13));body[...,6]=1;body[...,:3]=[3,0,3]
    body[0,11,:3]=[3,0,.1]  # Foot on floor.
    body[1,0,:3]=[0,0,2.1]  # Pelvis on horizontal cube top.
    body[2,11,:3]=[0,0,2.3] # Hover20cm above top.
    body[3,0,:3]=[1.1,0,1] # Side contact cannot provide vertical support.
    ok,_=supported(body,geo,1.)
    np.testing.assert_array_equal(ok,[True,True,False,False])


def test_ground_projection_preserves_relative_pose_and_removes_support_vertical_speed():
    shapes=(np.array([11]),np.zeros((1,3)),np.array([.1]));body=np.zeros((1,15,13));body[...,6]=1;body[...,2]=1;body[:,11,2]=.3;body[...,9]=.4
    root=body[:,0].copy();original=body.copy();r,b,delta=project_ground(root,body,shapes)
    height,speed=foot_bottom(b,shapes)
    np.testing.assert_allclose(height,.005,atol=1e-12);np.testing.assert_allclose(speed,0,atol=1e-12)
    np.testing.assert_allclose(b[...,:3]-r[:,None,:3],body[...,:3]-root[:,None,:3],atol=1e-12)
    np.testing.assert_array_equal(body,original)
