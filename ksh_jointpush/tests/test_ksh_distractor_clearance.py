import torch
from ksh_jointpush.ksh_distractor_clearance import clear_approaches


def test_box_on_agent_approach_is_rejected_even_when_far_from_both_endpoints():
    roots=torch.tensor([[[0.,0.]],[[0.,0.]],[[0.,0.]]])
    targets=torch.tensor([[[10.,0.]],[[10.,0.]],[[10.,0.]]])
    points=torch.tensor([[5.,0.],[5.,1.7],[5.,2.0]])
    half=torch.full((3,2),.25)
    assert torch.equal(clear_approaches(points,half,roots,targets,1.5),torch.tensor([False,False,True]))


def test_every_agent_path_and_stationary_rsi_agent_are_checked():
    roots=torch.tensor([[[0.,0.],[5.,3.]],[[0.,0.],[5.,3.]]])
    targets=torch.tensor([[[10.,0.],[5.,3.]],[[10.,0.],[5.,3.]]])
    points=torch.tensor([[5.,3.],[5.,5.]])
    half=torch.full((2,2),.25)
    assert torch.equal(clear_approaches(points,half,roots,targets,1.5),torch.tensor([False,True]))
