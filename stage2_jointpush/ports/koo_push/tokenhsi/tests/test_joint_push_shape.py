import copy
import pytest
import torch
from env.tasks.multi_agent.joint_push_shape import shape_mass_grid
from env.tasks.multi_agent.joint_push_spawn import sample_surround_spawn
from env.tasks.multi_agent.joint_push_reward import joint_push_reward,DEFAULT_CONFIG,push_face_geometry

SPEC=dict(mode='env_fixed_grid',size_min=[2.,1.,1.],size_max=[3.,2.6,1.5],step=[.2,.2,.25],mass_mode='random_mass_grid',mass_min=40.,mass_max=60.,mass_step=5.)

def test_shape_mass_grid_and_cap():
    grid=shape_mass_grid(SPEC)
    assert len(grid)==810 and len(set(grid))==810
    assert {mass for size,mass in grid}=={40,45,50,55,60}
    assert min(s[0] for s,m in grid)==2 and min(s[1] for s,m in grid)==1
    assert min(s[2] for s,m in grid)==1 and max(s[2] for s,m in grid)==1.5
    for name,value in [('mass_max',65),('mass_min',-1),('size_min',[1.9,1.,1.])]:
        spec=copy.deepcopy(SPEC);spec[name]=value
        with pytest.raises(ValueError):shape_mass_grid(spec)

def test_batched_spawn_and_reward_match_individual_shapes():
    torch.manual_seed(31)
    sizes=torch.tensor([s for s,m in shape_mass_grid(SPEC)])
    n=len(sizes)
    depth,lateral,heading=sample_surround_spawn(n,'cpu',sizes)
    forward=-depth-sizes[:,1,None]/2
    radius=(forward.square()+lateral.square()).sqrt()
    bound=sizes[:,:2].norm(dim=-1)[:,None]/2
    assert ((radius-bound>=1-1e-5)&(radius-bound<=2+1e-5)).all()
    gap=((forward[:,0]-forward[:,1]).square()+(lateral[:,0]-lateral[:,1]).square()).sqrt()
    assert gap.min()>=.8
    boxes=torch.zeros(n,13);boxes[:,6]=1;boxes[:,2]=sizes[:,2]/2
    goals=boxes[:,:3].clone();goals[:,1]+=3
    center,normal,tangent,width,_=push_face_geometry(boxes,goals,sizes,DEFAULT_CONFIG)
    anchors=center[:,None].expand(-1,2,-1).clone()
    anchors[...,:2]+=width[:,None,None]*.5*torch.tensor([-1.,1.])[None,:,None]*tangent[:,None]
    hands=anchors[:,:,None].expand(-1,-1,2,-1).clone()
    roots=torch.zeros(n,2,13)
    batch=joint_push_reward(roots,hands,boxes,goals,sizes,dict(DEFAULT_CONFIG,at_gate_mode='soft_min'))
    assert batch['coupled_contact'].all()
    torch.testing.assert_close(batch['minimum_separation'],sizes[:,0]*.25)
    torch.testing.assert_close(center[:,2],torch.full((n,),.8))
    for i in (0,129,405,809):
        single=joint_push_reward(roots[i:i+1],hands[i:i+1],boxes[i:i+1],goals[i:i+1],sizes[i],dict(DEFAULT_CONFIG,at_gate_mode='soft_min'))
        torch.testing.assert_close(single['agent_task_reward'],batch['agent_task_reward'][i:i+1])


def test_three_agent_surround_spawn_pairwise_clearance():
    torch.manual_seed(19)
    sizes=torch.tensor([s for s,m in shape_mass_grid(SPEC)])
    depth,lateral,heading=sample_surround_spawn(len(sizes),'cpu',sizes,num_agents=3)
    assert depth.shape==heading.shape==(len(sizes),3)
    forward=-depth-sizes[:,1,None]/2
    positions=torch.stack((forward,lateral),-1)
    distances=torch.cdist(positions,positions)
    distances.diagonal(dim1=-2,dim2=-1).fill_(float('inf'))
    assert distances.min()>=.8
    clearance=positions.norm(dim=-1)-sizes[:,:2].norm(dim=-1)[:,None]/2
    assert ((clearance>=1-1e-5)&(clearance<=2+1e-5)).all()


def test_four_agent_large_payload_geometry_and_reward():
    sizes=torch.tensor([[4.,3.,1.5]]).expand(64,-1)
    depth,lateral,_=sample_surround_spawn(64,'cpu',sizes,num_agents=4)
    forward=-depth-sizes[:,1,None]/2
    distances=torch.cdist(torch.stack((forward,lateral),-1),torch.stack((forward,lateral),-1))
    distances.diagonal(dim1=-2,dim2=-1).fill_(float('inf'))
    assert distances.min()>=.8
    boxes=torch.zeros(64,13);boxes[:,6]=1;boxes[:,2]=.75
    goals=boxes[:,:3].clone();goals[:,1]+=3
    center,normal,tangent,width,_=push_face_geometry(boxes,goals,sizes,DEFAULT_CONFIG)
    anchors=center[:,None,:].expand(-1,4,-1).clone()
    anchors[...,:2]+=width[:,None,None]*torch.tensor([-.9,-.3,.3,.9])[None,:,None]*tangent[:,None]
    hands=anchors[:,:,None].expand(-1,-1,2,-1).clone()
    result=joint_push_reward(torch.zeros(64,4,13),hands,boxes,goals,sizes,dict(DEFAULT_CONFIG,at_gate_mode='soft_min'))
    assert result['coupled_contact'].all()
    assert result['agent_task_reward'].shape==(64,4)
    assert torch.isfinite(result['agent_task_reward']).all()


def test_four_agent_shared_graph_indices():
    from utils.open_push_spec import compile_graph,graph_for_tasks
    spec=dict(sampler='two_agent_open_push',semantic_only=True,task_probabilities=dict(OPEN=0.,PUSH=1.))
    graph=compile_graph(spec,4,4)
    assert graph.edge_src.tolist()==[0,6,1,6,2,6,3,6]
    assert graph.edge_dst.tolist()==[6,8]*4
    assert graph.edge_owner.tolist()==[0,0,1,1,2,2,3,3]
    batch=graph_for_tasks(spec,torch.ones(2,4,dtype=torch.long))
    assert batch.edge_src.shape==(2,8)
