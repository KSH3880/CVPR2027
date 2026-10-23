import numpy as np
import trimesh
from rsi_native.geometry import NativeGeometry,rotate


def test_triangle_geometry_detects_sphere_overlap_and_floor(tmp_path):
    trimesh.creation.box(extents=[2,2,2]).export(tmp_path/'cube.obj')
    (tmp_path/'asset.urdf').write_text('<robot name="cube"><link name="cube"><collision><geometry><mesh filename="cube.obj"/></geometry></collision></link></robot>')
    shapes=(np.array([0]),np.zeros((1,3),dtype=np.float32),np.array([.1],dtype=np.float32))
    geometry=NativeGeometry(tmp_path,shapes)
    body=np.array([[[0,0,1.25]],[[0,0,1.05]],[[0,0,.8]]],dtype=np.float32)
    rotation=np.tile([0,0,0,1],(3,1,1)).astype(np.float32)
    clearance,floor=geometry.check(body,rotation,0.)
    np.testing.assert_allclose(clearance,[.15,-.05,-.3],atol=2e-6)
    np.testing.assert_allclose(floor,[1.15,.95,.7],atol=2e-6)
    assert geometry.watertight


def test_yaw_rotation_and_uniform_translation_preserve_relative_pose():
    q=np.array([0.,0.,np.sin(np.pi/4),np.cos(np.pi/4)])
    np.testing.assert_allclose(rotate(q,np.array([1.,0.,0.])),[0,1,0],atol=1e-12)
    root=np.array([.2,.3,.9]);key=np.array([[.5,.3,.1],[.2,.6,1.3]])
    shift=np.array([2.,-1.,.15])
    np.testing.assert_allclose(rotate(q,key)+shift-(rotate(q,root)+shift),rotate(q,key-root),atol=1e-12)
