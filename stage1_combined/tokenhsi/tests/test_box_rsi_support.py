import torch
from utils.rsi_support import box_supported


def test_box_support_rejects_floating_reference_and_accepts_floor_or_box():
    body=torch.zeros(3,15,13);body[...,6]=1;body[...,:3]=torch.tensor([3.,0.,3.])
    body[0,11,:3]=torch.tensor([0.,0.,.6])
    body[1,11,:3]=torch.tensor([0.,0.,.8])
    body[2,11,:3]=torch.tensor([3.,0.,.1])
    box=torch.zeros(3,13);box[:,6]=1;box[:,2]=.25
    shapes=(torch.tensor([0,11]),torch.zeros(2,3),torch.tensor([.1,.1]))
    result=box_supported(body,box,torch.full((3,3),.5),shapes)
    assert result.tolist()==[True,False,True]
