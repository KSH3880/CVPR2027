import numpy as np
from ksh_jointpush.ksh_view_camera import behind_camera


def test_camera_is_behind_the_agents_and_translates_with_the_group():
    people=np.array([[-3.,-2.,1.],[-1.,-2.,1.],[1.,-2.,1.],[3.,-2.,1.]])
    box=np.array([0.,0.,.8])
    eye,target=behind_camera(people,box,7.2,[0.,1.])
    assert eye[1]<people[:,1].min() and eye[2]>1.8
    assert people[:,1].mean()<target[1]<box[1]
    offset=np.array([12.,-23.,0.])
    shifted_eye,shifted_target=behind_camera(people+offset,box+offset,7.2,[0.,1.])
    np.testing.assert_allclose(shifted_eye,eye+offset)
    np.testing.assert_allclose(shifted_target,target+offset)


def test_rotating_the_push_rotates_the_camera_and_wider_groups_fit():
    people=np.array([[-3.,-2.,1.],[3.,-2.,1.]])
    box=np.array([0.,0.,.8])
    eye,target=behind_camera(people,box,7.2,[0.,1.])
    rotated=people.copy();rotated[:,0]=-people[:,1];rotated[:,1]=people[:,0]
    other_eye,other_target=behind_camera(rotated,box,7.2,[-1.,0.])
    np.testing.assert_allclose(other_eye,[-eye[1],eye[0],eye[2]])
    np.testing.assert_allclose(other_target,[-target[1],target[0],target[2]])
    far,_=behind_camera(people,box,12.,[0.,1.])
    assert far[1]<eye[1]
