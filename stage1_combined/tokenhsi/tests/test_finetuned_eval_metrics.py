import numpy as np
from evaluation.finetuned.backend import classify


def test_point_region_thresholds_and_rotated_narrow_surface():
    objects=np.zeros((4,13));objects[:,6]=1
    objects[:,5]=np.sin(np.pi/4);objects[:,6]=np.cos(np.pi/4)
    surface=np.tile([0.,0.,.3],(4,1));extent=np.tile([.5,1.],(4,1))
    root=np.tile([0.,0.,.4],(4,1))
    root[0,2]+=.19999;root[1,2]+=.20001
    root[2,:2]=[.35,0];root[3,:2]=[0,.21]
    point,region,target=classify(root,objects,surface,extent,'SIT')
    np.testing.assert_allclose(target,np.tile([0,0,.4],(4,1)))
    np.testing.assert_array_equal(point,[True,False,False,False])
    np.testing.assert_array_equal(region,[True,False,True,False])


def test_exact_center_succeeds_and_height_is_action_specific():
    obj=np.zeros((1,13));obj[:,6]=1;obj[:,2]=.4
    surface=np.array([[.1,.2,.3]]);extent=np.array([[.5,.6]])
    for action,h in [('SIT',.1),('CLIMB',.94)]:
        root=np.array([[.1,.2,.7+h]])
        p,r,t=classify(root,obj,surface,extent,action)
        assert p.all() and r.all()
        np.testing.assert_allclose(t,root)
