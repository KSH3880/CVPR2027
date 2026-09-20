"""Normal PPO training plus assertions on actual graph/reward storage and GAE inputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
import run as entry
from learning.multi_agent.ma_agent import MAAgent
from env.tasks.multi_agent.edge_ontop_reward import mix_task_reward

_original_step=MAAgent.env_step
_original_play=MAAgent.play_steps
_original_discount=MAAgent.discount_values
_original_combine=MAAgent._combine_rewards


def checked_step(self,actions):
    task=self.vec_env.env.task
    saved=self.obs['obs'].clone()
    out=_original_step(self,actions)
    obs,reward,done,info=out
    local=task.relation_runtime.last_result['local_task_reward']
    terms=info['reward_terms'].reshape(task.num_envs,2,7)
    expected=mix_task_reward(local)+terms[...,3:6].sum(-1)
    torch.testing.assert_close(reward.reshape_as(expected),expected,rtol=2e-6,atol=2e-6)
    self._ontop_record.append((saved,obs['obs'].clone(),self.rewards_shaper(reward).clone()))
    return out


def checked_combine(self,task_rewards,amp_rewards):
    if getattr(self,'_ontop_record',None):
        expected=torch.stack([r[2] for r in self._ontop_record])
        torch.testing.assert_close(task_rewards,expected)
    out=_original_combine(self,task_rewards,amp_rewards)
    self._ontop_gae_rewards=out.clone()
    return out


def checked_discount(self,dones,values,rewards,next_values):
    torch.testing.assert_close(rewards,self._ontop_gae_rewards)
    return _original_discount(self,dones,values,rewards,next_values)


def checked_play(self):
    self._ontop_record=[]
    out=_original_play(self)
    graph=self.vec_env.env.task.relation_runtime.graph
    assert torch.unique(graph.edge_relation,dim=0).shape[0]>5
    assert (graph.edge_valid.sum(-1)==2).any() and (graph.edge_valid.sum(-1)==3).any() and (graph.edge_valid.sum(-1)==4).any()
    buf=self.experience_buffer.tensor_dict
    for name,pos in [('obses',0),('next_obses',1),('rewards',2)]:
        torch.testing.assert_close(buf[name],torch.stack([r[pos] for r in self._ontop_record]))
    assert torch.isfinite(out['returns']).all()
    print('PASS actual PPO: saved obs/next graph packets, mixed task + own penalties in rollout rewards and GAE; 2048 environments',flush=True)
    self._ontop_record=[]
    return out


if __name__=='__main__':
    MAAgent.env_step=checked_step;MAAgent.play_steps=checked_play
    MAAgent._combine_rewards=checked_combine;MAAgent.discount_values=checked_discount
    entry.main()
