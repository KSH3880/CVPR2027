import unittest
import torch
from task_allocation.core import AllocationPolicy, assignment_from_action, switch_cost, step_reward, gae, sample_layout, nearest_initial_action
from task_allocation.learning import ppo_update


def observation(n=4):
    return dict(agent=torch.randn(n,2,16),box=torch.randn(n,2,20),goal=torch.randn(n,2,3),
                assignment=torch.tensor([0,1]).expand(n,-1).clone(),
                locked=torch.zeros(n,2,dtype=torch.bool),delivered=torch.zeros(n,2,dtype=torch.bool))


class CoreTest(unittest.TestCase):
    def test_joint_assignment_and_lock(self):
        p=AllocationPolicy(16)
        obs=observation()
        obs['assignment'][1]=torch.tensor([1,0]); obs['locked'][1,1]=True
        obs['delivered'][2,0]=True
        dist,value=p(obs)
        self.assertEqual(dist.probs[1,0].item(),0)
        self.assertEqual(dist.probs[2,1].item(),0)
        action=dist.sample()
        assigned=assignment_from_action(action)
        torch.testing.assert_close(assigned.sort(-1).values,torch.tensor([0,1]).expand(4,-1))
        (-dist.log_prob(action).mean()+value.square().mean()).backward()
        self.assertGreater(p.model.query.weight.grad.abs().sum().item(),0)
        for param in p.parameters():
            if param.grad is not None: self.assertTrue(torch.isfinite(param.grad).all())

    def test_time_and_consistency(self):
        prev=torch.tensor([[0,1],[0,1],[0,1]])
        new=torch.tensor([[1,0],[1,0],[0,1]])
        torch.testing.assert_close(switch_cost(prev,new,torch.tensor([False,True,True]),.1),torch.tensor([0.,-.2,0.]))
        r=step_reward(torch.tensor([0,1,0]),torch.tensor([False,False,True]),.02)
        torch.testing.assert_close(r,torch.tensor([-.02,9.98,-40.02]))

    def test_duration_and_terminal_bootstrap(self):
        r=torch.tensor([[-1.,-2.]])
        v=torch.zeros_like(r); nv=torch.full_like(r,10)
        done=torch.tensor([[True,False]])
        adv,ret=gae(r,v,nv,done,torch.tensor([[2.,3.]]),.5,1.)
        torch.testing.assert_close(ret,torch.tensor([[-1.,-.75]]))
        r=torch.tensor([[1.],[2.]])
        adv,_=gae(r,torch.zeros_like(r),torch.zeros_like(r),torch.zeros_like(r,dtype=torch.bool),torch.tensor([[2.],[1.]]),.5,1)
        torch.testing.assert_close(adv,torch.tensor([[1.5],[2.]]))

    def test_nearest_baseline_keeps_assignment(self):
        obs=observation(2)
        obs['agent'][..., :2]=torch.tensor([[[0.,0.],[10.,0.]],[[0.,0.],[10.,0.]]])
        obs['box'][..., :2]=torch.tensor([[[9.,0.],[1.,0.]],[[9.,0.],[1.,0.]]])
        obs['goal'][..., :2]=obs['box'][..., :2]
        action=nearest_initial_action(obs,torch.tensor([False,True]))
        torch.testing.assert_close(action,torch.tensor([1,0]))

    def test_random_layout(self):
        points=sample_layout(32,'cpu')
        d=torch.cdist(points,points)+torch.eye(6)[None]*100
        self.assertGreaterEqual(float(d.min()),1.2-1e-6)
        self.assertLessEqual(float(points.abs().max()),3)

    def test_ppo_changes_attention(self):
        torch.manual_seed(1)
        p=AllocationPolicy(16); opt=torch.optim.Adam(p.parameters(),lr=.001)
        obs=observation(8)
        with torch.no_grad():
            dist,v=p(obs); action=dist.sample(); logp=dist.log_prob(action)
        before=p.model.query.weight.detach().clone()
        loss=ppo_update(p,opt,[obs],action[None],logp[None],v[None],torch.linspace(-1,1,8)[None],v[None]+1,1,8)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before,p.model.query.weight))


if __name__=='__main__': unittest.main()
