"""Three-person inference geometry and owner/node identities."""
import torch
from env.tasks.multi_agent.joint_push_reward import DEFAULT_CONFIG, joint_push_reward
from utils.open_push_spec import compile_graph


def test_three_person_contact_is_permutation_equivariant():
    box=torch.zeros(1,13);box[:,6]=1;box[:,2]=1.3
    goal=box[:,:3].clone();goal[:,1]+=2.5
    roots=box[:,None].expand(-1,3,-1).clone()
    hands=torch.zeros(1,3,2,3)
    hands[0,:,:,0]=torch.tensor([-.85,0.,.85])[:,None]
    hands[:,:,:,1]=-1.3;hands[:,:,:,2]=.8
    result=joint_push_reward(roots,hands,box,goal,[2.6]*3)
    assert result['coupled_contact'].item()
    order=torch.tensor([2,0,1])
    shuffled=joint_push_reward(roots[:,order],hands[:,order],box,goal,[2.6]*3)
    torch.testing.assert_close(shuffled['agent_task_reward'],result['agent_task_reward'][:,order])
    hands[:,1]=hands[:,0]
    assert not joint_push_reward(roots,hands,box,goal,[2.6]*3)['coupled_contact'].item()


def test_three_person_graph_shares_real_box_and_goal():
    spec=dict(sampler='two_agent_open_push',semantic_only=True,task_probabilities=dict(OPEN=0.,PUSH=1.))
    graph=compile_graph(spec,3,4)
    assert graph.edge_src.tolist()==[0,5,1,5,2,5]
    assert graph.edge_dst.tolist()==[5,7,5,7,5,7]
    assert graph.edge_owner.tolist()==[0,0,1,1,2,2]
