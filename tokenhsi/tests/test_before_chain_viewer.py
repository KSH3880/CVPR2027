from types import SimpleNamespace
import itertools
import torch
from utils.before_chain_viewer import alias_layout, human_relations, goal_aligned_scene, qualified_success


def example(order=None):
    scene=torch.zeros(1,3*223+6*30+3+12*7+15)
    corners=scene[:,3*223:3*223+6*30].reshape(1,6,30)[...,6:].reshape(1,6,8,3)
    signs=torch.tensor(list(itertools.product([-1.,1.],repeat=3)))
    dims=torch.tensor([[.6,.6,.25],[.5,.5,.25],[.4,.4,.2],[.6,.6,.25],[.5,.5,.25],[.3,.3,.2]])
    corners[:]=signs[None,None]*dims[None,:,None]/2
    poses=scene[:,-99:-15].reshape(1,12,7)
    poses[...,6]=1
    poses[:,3:9,:3]=torch.tensor([[-2,0,.125],[0,-3,.125],[3,0,.1],[-2,0,.125],[0,-3,.125],[6,-6,.1]])
    poses[:,9,:3]=torch.tensor([0,0,.125])
    packet=torch.tensor([[[1,2,0,3,9],[1,3,1,4,3],[1,3,2,5,4]]])
    if order is not None:packet=packet[:,order]
    scene[:,-15:]=packet.flatten(1)
    return scene,packet


def test_chain_aliases_and_directed_relations():
    for order in (None,[2,0,1]):
        scene,packet=example(order)
        dependent,target,alias,count=alias_layout(packet,3,6)
        assert count.tolist()==[3]
        aliases={int(packet[0,i,2]):int(alias[0,i]) for i in range(3) if dependent[0,i]}
        assert aliases=={1:6,2:7}
        assert human_relations(packet,3).tolist()==[[[1,0,0],[2,1,0],[0,2,1]]]
        before=scene.clone();after=goal_aligned_scene(scene,3,6)
        poses=after[:,-99:-15].reshape(1,12,7)
        assert torch.allclose(poses[0,6:8,:3],torch.tensor([[0,0,.125],[0,0,.375]]))
        restored=after.clone()
        restored[:,-99:-15].reshape(1,12,7)[:,6:8,:3]=before[:,-99:-15].reshape(1,12,7)[:,6:8,:3]
        assert torch.equal(restored,before)
        assert torch.equal(scene,before)
        assert torch.equal(after[:,:-99],before[:,:-99])
        assert torch.equal(after[:,-15:],before[:,-15:])
        assert torch.equal(poses[...,3:],before[:,-99:-15].reshape(1,12,7)[...,3:])


def test_chain_success_requires_upstream_success():
    prereq=torch.zeros(1,6,6,dtype=torch.bool)
    prereq[:,3,1]=True;prereq[:,5,3]=True
    graph=SimpleNamespace(num_agents=3,prereq_mask=prereq)
    raw=torch.ones(1,6,dtype=torch.bool)
    raw[:,1]=False
    assert not qualified_success(raw,graph)[0,[1,3,5]].any()
    raw[:,1]=True;raw[:,3]=False
    assert qualified_success(raw,graph)[0,1]
    assert not qualified_success(raw,graph)[0,[3,5]].any()
    raw[:,3]=True
    assert qualified_success(raw,graph).all()
