"""Non-learning reset/RSI/AMP audit using the private runtime."""
import json


def install(output):
    import torch
    from learning.multi_agent.ma_agent import MAAgent

    def audit(self):
        self.init_tensors()
        self.obs=self.env_reset()
        self._init_train()
        task=self.vec_env.env.task
        samples=[]
        for _ in range(16):
            obs=self.env_reset(torch.arange(task.num_envs,device=self.ppo_device)*task.num_agents)
            assert torch.isfinite(obs['obs']).all()
            for skill, slots in task._reset_ref_slots.items():
                if slots is None:continue
                rows,owners=slots
                for r,o in zip(rows.tolist(),owners.tolist()):
                    samples.append(dict(skill=skill,asset=task._ft_profile_keys[r][o],
                        root=task._humanoid_root_states[r,o,:3].cpu().tolist()))
            assert torch.isfinite(task._amp_obs_buf).all()
        roots=task._root_states.clone();dofs=task._dof_state.clone()
        amp_before=task._amp_obs_buf.clone()
        task._reset_envs(torch.tensor([0],device=task.device))
        # Compare untouched environments via public per-environment tensor layout.
        actor_count=task.get_num_actors_per_env()
        roots_unchanged=torch.equal(task._root_states.view(task.num_envs,actor_count,-1)[1:],
                                    roots.view(task.num_envs,actor_count,-1)[1:])
        dofs_unchanged=torch.equal(task._dof_state.view(task.num_envs,-1)[1:],dofs.view(task.num_envs,-1)[1:])
        amp_unchanged=torch.equal(task._amp_obs_buf.view(task.num_envs,-1)[1:],amp_before.view(task.num_envs,-1)[1:])
        assert roots_unchanged and dofs_unchanged and amp_unchanged
        obs=self.env_reset()
        self.set_eval()
        initial=task._humanoid_root_states.clone()
        max_speed=0.
        max_root_speed=0.
        max_sit_speed=0.
        for _ in range(12):
            action=self.get_action_values(obs,self._rand_action_probs)['actions']
            assert torch.isfinite(action).all()
            obs,_,done,_=self.env_step(action)
            assert torch.isfinite(obs['obs']).all() and torch.isfinite(task._amp_obs_buf).all()
            max_speed=max(max_speed,float(task._box_states[:,:2,7:10].norm(dim=-1).max()))
            max_root_speed=max(max_root_speed,float(task._humanoid_root_states[...,7:10].norm(dim=-1).max()))
            sit_speed=task._box_states[:,:2,7:10].norm(dim=-1)[task._ft_sit]
            if sit_speed.numel():max_sit_speed=max(max_sit_speed,float(sit_speed.max()))
            ids=done.nonzero(as_tuple=False).flatten()
            if len(ids):obs=self.env_reset(ids)
        summary={}
        cache=task._size_rsi_cache
        for key, records in cache.entries.items():
            summary[str(key)]=sum(int(mask.sum()) for mask,_ in records)
        result=dict(num_envs=task.num_envs,observation_size=task.num_obs,
                    max_sit_object_speed=max_sit_speed,
                    partial_reset_other_roots_unchanged=roots_unchanged,
                    partial_reset_other_dofs_unchanged=dofs_unchanged,
                    partial_reset_other_amp_unchanged=amp_unchanged,
                    amp_history_and_policy_finite=True,policy_steps=12,max_box_speed=max_speed,
                    max_humanoid_root_speed=max_root_speed,
                    accepted_frames=summary,reset_samples=samples,
                    rsi_requested=task._size_rsi_requested.cpu().tolist(),
                    rsi_executed=task._size_rsi_executed.cpu().tolist(),
                    rsi_fallbacks=task._size_rsi_fallbacks.cpu().tolist())
        from collections import Counter
        result['category_slots']=dict(Counter(k.split('/')[0] for row in task._ft_profile_keys for k in row))
        result['reset_skills_by_category']=dict(Counter(s['asset'].split('/')[0]+':'+s['skill'] for s in samples))
        if task.cfg['env']['objectFineTune']['task']=='joint':
            assert len(result['category_slots'])==3
            assert not any(s['skill']!='loco' and not s['asset'].startswith('StraightChair_Normal/') for s in samples)
            assert any(s['skill']=='sit' for s in samples)
        (output/'rsi_audit.json').write_text(json.dumps(result,indent=2))
        print('[native RSI audit]',{k:v for k,v in result.items() if k not in ('reset_samples','accepted_frames')},flush=True)
        return 0.,0

    MAAgent.train=audit
