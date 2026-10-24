import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from jointpush.partial.geometry import random_regions,sample_rear_spawn,sample_groups,group_layout


def test_rear_half_plane_and_clearance_for_arbitrary_goal_directions():
    torch.manual_seed(731)
    n=2048
    boxes=torch.zeros(n,4,13);boxes[...,6]=1
    boxes[...,0]=torch.tensor([0.,12.,0.,12.]);boxes[...,1]=torch.tensor([0.,0.,12.,12.])
    angle=torch.rand(n,4)*2*torch.pi
    goals=boxes[...,:3].clone();goals[...,:2]+=3*torch.stack((angle.cos(),angle.sin()),-1)
    sizes=torch.tensor([6.8,1.7,1.7]).expand(n,4,3)
    positions,heading=sample_rear_spawn(boxes,sizes,goals,(1.65,6.))
    delta=positions-boxes[...,:2]
    assert ((delta*(goals[...,:2]-boxes[...,:2])).sum(-1)<=1e-4).all()
    radius=delta.norm(dim=-1)
    assert (radius>=sizes[...,:2].norm(dim=-1)/2+.45-1e-5).all()
    assert (radius<=6.+1e-5).all()
    assert (heading>=0).all() and (heading<2*torch.pi).all()


def test_four_agents_on_same_box_do_not_overlap():
    torch.manual_seed(13);n=2048
    boxes=torch.zeros(n,4,13);boxes[...,6]=1
    sizes=torch.tensor([6.8,1.7,1.7]).expand(n,4,3)
    goals=boxes[...,:3].clone();goals[...,1]=5
    positions,_=sample_rear_spawn(boxes,sizes,goals,(1.65,6.),.9)
    distances=torch.cdist(positions,positions);distances.masked_fill_(torch.eye(4,dtype=torch.bool),torch.inf)
    assert (distances>=.9-1e-5).all()
    assert (positions[...,1]<=1e-5).all()
    assert (positions[...,0]>0).any() and (positions[...,0]<0).any()


def test_random_unique_regions_stay_within_group_and_vary():
    torch.manual_seed(24)
    groups=torch.tensor([[0,0,2,2],[3,3,3,3],[0,1,2,3],[0,0,0,3]]).repeat(512,1)
    same=groups[...,None]==groups[:,None,:]
    regions=random_regions(same)
    assert torch.equal(regions.sort(-1).values,torch.arange(4).expand(len(groups),4))
    assert same.gather(2,regions[...,None]).all()
    assert torch.unique(regions[1::4],dim=0).shape[0]==24
    assert (regions[0::4]!=torch.arange(4)).any()


def test_random_physical_groups_only_share_with_their_members():
    torch.manual_seed(38);groups=sample_groups(4096,.8)
    assignment,counts,_=group_layout(groups)
    assert torch.equal(assignment[...,None]==assignment[:,None,:],groups[...,None]==groups[:,None,:])
    assert ((counts>=1)&(counts<=4)).all()
    assert torch.unique(assignment[:,0]).numel()>4
