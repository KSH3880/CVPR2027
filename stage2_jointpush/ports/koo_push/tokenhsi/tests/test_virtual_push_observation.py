import pytest
import torch
from env.tasks.multi_agent.virtual_push_observation import split_box_observation


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[-1.,1.])
def test_cubes_match_rear_face_bottom_and_displacement_without_mutation(axis,sign):
    box=torch.zeros(1,13);box[:,2]=1.3;box[:,6]=1.;box[:,7]=.2;box[:,12]=.5
    goal=box[:,:3]+torch.tensor([0.,2.5,0.]);original=box.clone()
    states,corners,targets,sizes=split_box_observation(box,[2.6]*3,goal,torch.tensor([axis]),torch.tensor([sign]))
    assert torch.equal(box,original)
    assert torch.allclose(targets-states[:,:,:3],(goal-box[:,:3])[:,None])
    assert torch.equal(sizes,torch.full((1,2,3),1.3))
    assert torch.allclose(states[0,:,axis],torch.tensor([-.65,.65]))
    normal=1-axis
    assert torch.allclose(states[0,:,normal]-sign*.65,torch.full((2,),-sign*1.3))
    assert torch.allclose(states[0,:,2]-.65,torch.zeros(2))
    world_corners=corners+states[:,:,:3,None].transpose(-1,-2)
    assert torch.allclose(world_corners[...,2].min(-1).values,torch.zeros(1,2))
    assert corners.shape==(1,2,8,3)
    offset=states[:,:,:3]-box[:,None,:3]
    expected=box[:,None,7:10]+torch.cross(box[:,None,10:13].expand_as(offset),offset,dim=-1)
    assert torch.allclose(states[...,7:10],expected)


def test_rotated_cubes_follow_real_rear_face():
    box=torch.zeros(1,13);box[:,2]=1.3;box[:,5:7]=2**-.5
    states,_,targets,_=split_box_observation(box,[2.6]*3,torch.tensor([[3.,0.,1.3]]),torch.tensor([0]),torch.tensor([-1.]))
    assert torch.allclose(states[0,:,1],torch.tensor([-.65,.65]),atol=1e-6)
    assert torch.allclose(states[0,:,0],torch.full((2,),-.65),atol=1e-6)
    assert torch.allclose(states[0,:,2],torch.full((2,),.65))
    assert torch.allclose(targets-states[:,:,:3],torch.tensor([[[3.,0.,0.]]]))
