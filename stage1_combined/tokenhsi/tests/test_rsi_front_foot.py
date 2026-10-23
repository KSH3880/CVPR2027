import numpy as np
import torch
import trimesh
from rsi_native.foot_surface import FootSurface
from rsi_native.geometry import NativeGeometry
from utils.chair_front import front_allowed


def test_open_thin_surface_crossing_is_rejected_but_touching_is_allowed(tmp_path):
    trimesh.Trimesh(vertices=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],faces=[[0,1,2],[0,2,3]],process=False).export(tmp_path/'plane.obj')
    (tmp_path/'asset.urdf').write_text('<robot name="plane"><link name="plane"><collision><geometry><mesh filename="plane.obj"/></geometry></collision></link></robot>')
    trimesh.Trimesh(vertices=[[-.2,0,-.02],[.2,0,.02],[.2,.2,.02]],faces=[[0,1,2]],process=False).export(tmp_path/'foot.obj')
    (tmp_path/'human.xml').write_text('<mujoco><asset><mesh name="foot" file="foot.obj"/></asset><worldbody><body name="right_foot"><geom type="mesh" mesh="foot"/></body></worldbody></mujoco>')
    geo=NativeGeometry(tmp_path,(np.array([0]),np.zeros((1,3)),np.zeros(1)));feet=FootSurface(tmp_path/'human.xml',['right_foot'])
    body=np.zeros((3,1,13));body[...,6]=1;body[:,0,2]=[0,.1,.02]
    assert not geo.watertight
    np.testing.assert_array_equal(feet.crossing(body,geo,0),[True,False,False])


def test_chair_front_rejects_rear_and_side_but_accepts_central_seat():
    root=torch.tensor([[0.,.5,0.],[0.,-.5,0.],[.5,.1,0.],[0.,-.05,0.]])
    seat=torch.zeros_like(root);facing=torch.tensor([0.,1.]).expand(4,-1)
    assert front_allowed(root,seat,facing).tolist()==[True,False,False,True]


def test_full_scene_guard_transforms_foreign_object_and_checks_both_humans(tmp_path):
    from types import SimpleNamespace
    from utils.native_foot_guard import NativeFootGuard
    trimesh.Trimesh(vertices=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],faces=[[0,1,2],[0,2,3]],process=False).export(tmp_path/'plane.obj')
    (tmp_path/'asset.urdf').write_text('<robot name="plane"><link name="plane"><collision><geometry><mesh filename="plane.obj"/></geometry></collision></link></robot>')
    trimesh.Trimesh(vertices=[[-.2,0,-.02],[.2,0,.02],[.2,.2,.02]],faces=[[0,1,2]],process=False).export(tmp_path/'foot.obj')
    (tmp_path/'human.xml').write_text('<mujoco><asset><mesh name="foot" file="foot.obj"/></asset><worldbody><body name="right_foot"><geom type="mesh" mesh="foot"/></body></worldbody></mujoco>')
    geom=NativeGeometry(tmp_path,(np.array([0]),np.zeros((1,3)),np.zeros(1)))
    body=torch.zeros(1,2,1,13);body[...,6]=1;body[0,0,0,:3]=torch.tensor([2.,3.,1.]);body[0,1,0,:3]=torch.tensor([2.,3.,1.2])
    objects=torch.zeros(1,1,13);objects[...,6]=1;objects[0,0,:3]=torch.tensor([2.,3.,1.])
    objects[0,0,3:7]=torch.tensor([0.,0.,2**-.5,2**-.5])
    task=SimpleNamespace(_kinematic_humanoid_rigid_body_states=body,_box_states=objects,_native_asset_keys=[['plane']],num_agents=2,num_bodies=1)
    guard=NativeFootGuard.__new__(NativeFootGuard);guard.t=task;guard.geometry={'plane':geom};guard.feet=FootSurface(tmp_path/'human.xml',['right_foot']);guard.rejections=0
    assert guard.rejected(torch.tensor([0])).tolist()==[True]
    body[...,2]+=.2
    assert guard.rejected(torch.tensor([0])).tolist()==[False]
