"""Real-simulator graph replay, preset/reset, reward and dynamic box contact checks."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymtorch
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.edge_ontop_spec import *
from env.tasks.multi_agent.edge_context_reward import *
from env.tasks.multi_agent.edge_ontop_reward import *


@torch.no_grad()
def check_simulator(self):
    task=self.env.task;assert task._edge_ontop;self._reward_debug=False
    n,m=task.num_envs,task.num_agents;runtime=task.relation_runtime;g=runtime.graph;e=len(g.ids)
    report={};offset=223*m+30*task.num_objects
    obs=self.env_reset();self.get_batch_size(obs['obs'],1)
    obs,_,_,_=self.env_step(self.env,self.get_action(obs,True))
    before=runtime.suffix().clone();self.env_reset(torch.tensor([0],device=task.device))
    torch.testing.assert_close(runtime.suffix()[1:],before[1:])
    old_clip=self.env.clip_obs;self.env.clip_obs=.1
    torch.testing.assert_close(self.env._policy_observation()[:,offset:],task.obs_buf[:,offset:])
    self.env.clip_obs=old_clip
    if task._enable_markers:
        at=owner_sum(((g.edge_relation==AT)&g.edge_valid).long(),g).bool()
        assert (task._marker_pos[...,2][~at]==20).all()
    for preset in PRESETS:
        task._task_graph_preset=preset;obs=self.env_reset();resets=[]
        if preset=='random':
            assert torch.unique(g.edge_relation,dim=0).shape[0]>1, 'sampler must run in actual environment'
        assert task._sampling_counts[0] >= n
        for step in range(32):
            obs=self.env_reset(resets);saved=obs['obs'].clone()
            torch.testing.assert_close(saved[:,-7*e:],runtime.suffix())
            normalized=self._preproc_obs(saved)
            torch.testing.assert_close(normalized[:,offset:],saved[:,offset:])
            if step==0:
                actions_before=self.model.a2c_network.eval_actor(normalized)[0].clone()
            obs,reward,done,info=self.env_step(self.env,self.get_action(obs,True))
            phi,diag=task._evaluate_relations()
            expected=edge_context_reward(phi,diag['progress'],own_success(phi,diag['z_error'],g),g,task._relation_cfg)
            terms=info['reward_terms'].reshape(n,m,7)
            for i,key in enumerate(('state_component','progress_component','success_component')):
                torch.testing.assert_close(terms[...,i],mix_task_reward(owner_sum(expected[key],g)))
            torch.testing.assert_close(reward.reshape(n,m),mix_task_reward(expected['agent_task_reward'])+terms[...,3:6].sum(-1),atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(info['policy_obs'][:,-7*e:],runtime.suffix())
            assert torch.isfinite(saved).all() and torch.isfinite(reward).all()
            at=owner_sum(((g.edge_relation==AT)&g.edge_valid).long(),g).bool()
            assert (task._tar_platform_pos[...,2][~at]>=19.9).all(),task._tar_platform_pos[...,2][~at]
            if step==0:
                # Reset the simulator graph, then replay unchanged past obs in actor and critic.
                saved_graph=g.edge_dst.clone()
                self.env_reset(torch.tensor([0],device=task.device))
                torch.testing.assert_close(self.model.a2c_network.eval_actor(normalized)[0],actions_before)
                order=torch.arange(n-1,-1,-1,device=task.device)
                for enc in [self.model.a2c_network.actor_encoder,self.model.a2c_network.critic_encoder]:
                    torch.testing.assert_close(enc(normalized[order]),enc(normalized)[order],atol=2e-5,rtol=2e-5)
            resets=done.nonzero(as_tuple=False).flatten()[::m]
        if preset!='random':
            expected_g=sample_graph(n,task._relation_graph_spec,task.device,preset,task._task_role_swap)
            for f in fields(g):
                if isinstance(getattr(g,f.name),torch.Tensor):torch.testing.assert_close(getattr(g,f.name),getattr(expected_g,f.name))
        report[preset]='32 transitions + replay/reset/platform checks passed'
    # Controlled fixture ONLY in this disposable smoke scene: actual dynamic boxes stack on floor.
    ids=torch.tensor([0],device=task.device);task._task_graph_preset='ontop_chain';self.env_reset(ids)
    order=task._logical_box_order[0];size=task._logical_box_values(task._box_size)[0]
    states=task._box_states[0];states[:,7:]=0;states[:,3:6]=0;states[:,6]=1
    center=task._env_origins[0,:2].clone()
    for logical in range(3):states[order[logical],:2]=center
    states[order[2],2]=size[2,2]/2
    states[order[0],2]=size[2,2]+size[0,2]/2
    states[order[1],2]=size[2,2]+size[0,2]+size[1,2]/2
    task._platform_pos[0,:,2]=20;task._tar_platform_pos[0,:,2]=20
    task._reset_env_tensors(ids)
    for _ in range(90):
        task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    boxes=task._logical_box_values(task._box_states)[0]
    delta,d=ontop_geometry(boxes[:1],boxes[2:3],size[:1],size[2:3])
    gap=float(d['signed_gap'][0])
    live_phi,live_diag=task._evaluate_relations()
    torch.testing.assert_close(live_phi[0,1],torch.exp(-10*delta.square().sum(-1))[0])
    torch.testing.assert_close(live_diag['z_error'][0,1],d['signed_gap'][0])
    contact=task._logical_box_values(task._ontop_box_contact)[0].norm(dim=-1)
    assert abs(gap)<.01,(gap,boxes[:,:3])
    assert contact[0]>.01 and contact[2]>.01,contact
    old_x=boxes[2,0].clone()
    task._box_states[0,order[2],7]=2.
    actor_ids=task._box_actor_ids[ids].contiguous().view(-1)
    task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),gymtorch.unwrap_tensor(actor_ids),len(actor_ids))
    for _ in range(10):task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    displacement=float(task._logical_box_values(task._box_states)[0,2,0]-old_x)
    assert abs(displacement)>.005,displacement
    report['contact_fixture']=dict(lower_stack_signed_gap=gap,net_contact_norms=contact.tolist(),dynamic_ox_dx=displacement)
    path=Path(task.cfg['args'].output_path);path.mkdir(parents=True,exist_ok=True)
    (path/'simulator_verification.json').write_text(json.dumps(report,indent=2))
    print('PASS OnTop simulator: presets, 128 transitions, saved graph/context, partial resets, RMS/clipping, mixed reward, live dynamic box contact:',report,flush=True)


if __name__=='__main__':
    MAPlayerContinuous.run=check_simulator
    entry.main()
