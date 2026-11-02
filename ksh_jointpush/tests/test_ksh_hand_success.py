import numpy as np
import torch
from ksh_jointpush.ksh_hand_success import assess, face_distances


def fixture():
    distance = np.full((121,4,2), .04)
    distance[:20] = 2.
    force = np.ones((121,4), bool)
    force[0] = False
    success = np.zeros((121,4), bool)
    success[90] = True
    boxes = np.zeros((121,3))
    boxes[:,1] = np.arange(121)*.03
    return distance, force, success, boxes, np.array([0.,2.7,0.]), 1/30


def test_all_eight_hands_and_real_push_are_required():
    values = fixture()
    assert assess(*values)['passed']
    values[0][:,3,1] = .2
    assert not assess(*values)['passed']
    values = fixture()
    values[1][:,2] = False
    assert not assess(*values)['passed']


def test_goal_saturation_and_static_contact_do_not_pass():
    values = fixture()
    values[0][:] = 1.
    assert not assess(*values)['passed']
    values = fixture()
    values[3][:] = 0.
    assert not assess(*values)['passed']


def test_arrival_detachment_initial_success_and_long_dropouts_are_rejected():
    values = fixture()
    values[0][90,0,0] = .3
    assert not assess(*values)['passed']
    values = fixture()
    values[2][0] = True
    assert not assess(*values)['passed']
    values = fixture()
    values[0][40:48,0,0] = .3
    assert not assess(*values)['passed']
    values = fixture()
    values[0][40:42,0,0] = .3
    assert assess(*values)['passed']


def test_distance_is_to_finite_rear_face_including_height_and_rotation():
    boxes = torch.tensor([[[0.,0.,.5,0.,0.,0.,1.]]]).expand(1,4,7).clone()
    sizes = torch.tensor([[[2.,1.,1.]]]).expand(1,4,3)
    hands = torch.tensor([[[[0.,-.54,.7],[.2,-.54,.7]]]]).expand(1,4,2,3).clone()
    reference = face_distances(hands,boxes,sizes)
    assert torch.allclose(reference, torch.full((1,4,2),.04),atol=1e-6)
    raised = hands.clone();raised[...,2] = 1.4
    assert (face_distances(raised,boxes,sizes) > .4).all()
    front = hands.clone();front[...,1] = .54
    assert (face_distances(front,boxes,sizes) > .08).all()
    boxes[...,5:7] = 2**-.5
    turned = hands.clone();turned[...,0] = -hands[...,1];turned[...,1] = hands[...,0]
    assert torch.allclose(face_distances(turned,boxes,sizes),reference,atol=1e-6)
