import torch
from ksh_jointpush.ksh_initialization_safety import invalid_initialization


def test_all_pairs_minimum_gap_including_loco_loco_and_different_groups():
    states=torch.zeros(3,4,1,13);states[...,6]=1
    states[:,:,:,0]=torch.tensor([0.,2.,4.,6.])[None,:,None]
    states[1,3,0,0]=.3
    states[2,2,0,0]=2.5
    geo=dict(ids=torch.tensor([0]),offsets=torch.zeros(1,3),radii=torch.tensor([.1]))
    assert torch.equal(invalid_initialization(states,geo),torch.tensor([False,True,True]))
    assert torch.equal(invalid_initialization(states[[2,0]],geo),torch.tensor([True,False]))


def test_full_body_overlap_is_rejected_even_if_roots_are_far():
    states=torch.zeros(2,2,2,13);states[...,6]=1
    states[:,1,:,0]=2.
    states[:,0,1,0]=.95;states[:,1,1,0]=1.05
    states[1,1,1,0]=1.5
    geo=dict(ids=torch.tensor([1]),offsets=torch.zeros(1,3),radii=torch.tensor([.1]))
    assert torch.equal(invalid_initialization(states,geo),torch.tensor([True,False]))


def test_nonfinite_start_is_rejected_without_modifying_references():
    states=torch.zeros(2,2,1,13);states[...,6]=1;states[:,1,0,0]=2
    states[1,0,0,2]=float('nan')
    geo=dict(ids=torch.tensor([0]),offsets=torch.zeros(1,3),radii=torch.tensor([.1]))
    before=states.clone()
    assert torch.equal(invalid_initialization(states,geo),torch.tensor([False,True]))
    torch.testing.assert_close(states,before,equal_nan=True)
