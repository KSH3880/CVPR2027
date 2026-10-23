"""Contact IK must change joints, respect limits and retain source arrays."""
from pathlib import Path
import numpy as np
import trimesh
from rsi_native.repair import NativeFK
from rsi_native.geometry import NativeGeometry, collision_samples
from rsi_native.ik import ContactIK


def test_leg_ik_fits_support_without_mutating_source(tmp_path):
    xml=Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    fk=NativeFK(xml);shapes=collision_samples(xml,fk.names)
    trimesh.creation.box(extents=[2,2,.4]).export(tmp_path/'top.obj')
    (tmp_path/'asset.urdf').write_text('<robot name="top"><link name="top"><collision><geometry><mesh filename="top.obj"/></geometry></collision></link></robot>')
    geo=NativeGeometry(tmp_path,shapes)
    entry=dict(bbox=[2,2,.4],seat=[0,0,.2],extent=[2,2])
    solver=ContactIK(fk,shapes,geo,entry)
    joint=np.clip(np.zeros(32),fk.lower,fk.upper);joint[20]=.25
    root=np.zeros(13);root[6]=1;root[2]=1
    body=fk.state(root[None],joint[None],np.zeros((1,32)))[0]
    root[2]+=.405-solver.sole(body)[:,2].min()
    before_joint=joint.copy();before_root=root.copy()
    body=fk.state(root[None],joint[None],np.zeros((1,32)))[0]
    initial=np.abs(solver.sole(body)[:,2]-.405).max()
    r,q,b,stats=solver.solve(root,joint,'climb',max_nfev=40)
    assert stats['foot_target_error'] < initial*.6
    assert np.max(np.abs(q-joint))>.01
    assert np.all(q>=fk.lower-1e-7) and np.all(q<=fk.upper+1e-7)
    assert np.max(np.abs(r[:3]-root[:3]))<=.080001
    np.testing.assert_array_equal(joint,before_joint);np.testing.assert_array_equal(root,before_root)
    pos,rot=fk.forward(r[None],q[None]);np.testing.assert_allclose(b[:,:3],pos[0],atol=1e-9)
