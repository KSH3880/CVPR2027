"""Actual-simulator smoke using normal CLI/env/player/checkpoint paths."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # MUST precede torch
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.edge_context_reward import edge_context, own_success, edge_context_reward, owner_sum


@torch.no_grad()
def check_simulator(self):
    task=self.env.task;assert task._edge_context
    self._reward_debug=False
    N,M=task.num_envs,task.num_agents
    runtime=task.relation_runtime;g=runtime.graph;cfg=task._relation_cfg;E=len(g.ids)
    obs=self.env_reset();self.get_batch_size(obs['obs'],1)
    ids=torch.arange(N,device=task.device)
    torch.testing.assert_close(runtime.phi,task._evaluate_relations(ids)[0])
    # Commit the first indexed root setter before a subset reset.
    obs,_,_,_=self.env_step(self.env,self.get_action(obs,True))
    before=runtime.suffix().clone();self.env_reset(ids[:1])
    torch.testing.assert_close(runtime.suffix()[1:],before[1:])
    torch.testing.assert_close(runtime.phi[:1],task._evaluate_relations(ids[:1])[0])
    old_clip=self.env.clip_obs;self.env.clip_obs=.1
    clipped=self.env._policy_observation();offset=M*223+task.num_objects*30
    torch.testing.assert_close(clipped[:,offset:],task.obs_buf[:,offset:])
    self.env.clip_obs=old_clip
    resets=[]
    for step in range(64):
        obs=self.env_reset(resets)
        saved=obs['obs'].clone();stored_phi=runtime.phi.clone()
        torch.testing.assert_close(saved[:,-2*E:],edge_context(stored_phi,g).flatten(1))
        normalized=self._preproc_obs(saved)
        if step==0:
            torch.testing.assert_close(normalized[:,offset:],saved[:,offset:])
            actor_before=self.model.a2c_network.eval_actor(normalized)[0].clone()
        obs,reward,done,info=self.env_step(self.env,self.get_action(obs,True))
        phi,diag=task._evaluate_relations()
        expected=edge_context_reward(phi,diag['progress'],own_success(phi,diag['z_error'],g),g,cfg)
        torch.testing.assert_close(runtime.phi,phi)
        torch.testing.assert_close(info['policy_obs'][:,-2*E:],edge_context(phi,g).flatten(1))
        terms=info['reward_terms'].reshape(N,M,7)
        for k,key in enumerate(['state_component','progress_component','success_component']):
            torch.testing.assert_close(terms[...,k],owner_sum(expected[key],g))
        torch.testing.assert_close(terms[...,:-1].sum(-1).flatten(),reward,atol=2e-6,rtol=2e-6)
        assert torch.isfinite(reward).all() and torch.isfinite(obs['obs'] if isinstance(obs, dict) else obs).all()
        if step==0:
            torch.testing.assert_close(self.model.a2c_network.eval_actor(normalized)[0],actor_before)
            permutation=torch.arange(N-1,-1,-1,device=task.device)
            for enc in [self.model.a2c_network.actor_encoder,self.model.a2c_network.critic_encoder]:
                torch.testing.assert_close(enc(normalized[permutation]),enc(normalized)[permutation],atol=2e-5,rtol=2e-5)
        resets=done.nonzero().flatten()[::M]
    print('PASS edge-context simulator: M={} O={} E={}, 64 transitions, subset reset, current context, '
          'stored PPO observation, RMS/clipping bypass, component/penalty agreement'.format(M,task.num_objects,E),flush=True)


if __name__=='__main__':
    MAPlayerContinuous.run=check_simulator
    entry.main()
