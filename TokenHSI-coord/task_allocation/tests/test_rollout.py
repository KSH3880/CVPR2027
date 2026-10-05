import unittest
import torch
from task_allocation.learning import macro_step


class FakeTask:
    def __init__(self):
        self.num_envs=3; self.device='cpu'; self.dt=.1
        self.obs_buf=torch.zeros(6,1); self.tick=0; self.reset_rows=None
    def _compute_observations(self): pass
    def update_delivery(self):
        new=torch.tensor([0,2 if self.tick==1 else 0,0])
        return new,torch.tensor([False,self.tick>=1,False])
    def agent_rows(self,ids): return (ids[:,None]*2+torch.arange(2)[None]).flatten()
    def reset(self,rows): self.reset_rows=rows.clone()


class FakeEnv:
    def __init__(self): self.task=FakeTask(); self.clip_obs=5
    def step(self,action):
        self.task.tick+=1
        done=torch.tensor([False,False,self.task.tick>=3]).repeat_interleave(2)
        return self.task.obs_buf,None,done,{}


class FakePlayer:
    def __init__(self): self.env=FakeEnv(); self.device='cpu'
    def get_batch_size(self,*args): pass
    def get_action(self,*args,**kwargs): return torch.zeros(6,1)


class RolloutTest(unittest.TestCase):
    def test_actual_duration_reward_and_reset_rows(self):
        player=FakePlayer()
        reward,done,duration,diag=macro_step(player,4,1.,1.,10.,40.)
        torch.testing.assert_close(duration,torch.tensor([4.,1.,3.]))
        torch.testing.assert_close(reward,torch.tensor([-.4,19.9,-40.3]))
        torch.testing.assert_close(done,torch.tensor([False,True,True]))
        torch.testing.assert_close(player.env.task.reset_rows,torch.tensor([2,3,4,5]))
        self.assertEqual(diag,dict(delivered=2,success=1,failure=1,executed_steps=8,held_box_steps=0))

    def test_low_step_discount(self):
        reward,_,_,_=macro_step(FakePlayer(),4,.5,1.,10.,40.)
        torch.testing.assert_close(reward,torch.tensor([-.1875,19.9,-10.175]))


if __name__=='__main__': unittest.main()
