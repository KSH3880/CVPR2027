"""Sampled OnTop scene integration. Existing motion initialization and physics stay live."""
import json
import hashlib
import torch
from utils.edge_context_spec import AT
from utils.edge_ontop_spec import (ON_TOP, compile_ontop_graph, sample_graph, expand_graph,
    select_graph, copy_graph_rows, batched)
from env.tasks.multi_agent.edge_context_reward import owner_sum
from env.tasks.multi_agent.edge_ontop_reward import OnTopContextRuntime, evaluate_ontop_edges


class SampledOnTopTaskMixin:
    def _init_ontop_context_runtime(self):
        prototype=compile_ontop_graph(self._relation_graph_spec,self.num_agents,self.num_objects,self.device)
        g=expand_graph(prototype,self.num_envs)
        self.relation_runtime=OnTopContextRuntime(self.num_envs,g,self._relation_cfg,self.device)
        self._sampling_counts=torch.zeros(18,device=self.device)
        self._sampling_retries=torch.zeros((),device=self.device)
        self._sampling_failures=torch.zeros((),device=self.device)
        self._ontop_maintain_steps=torch.zeros(self.num_envs,len(g.ids),device=self.device)
        self._ontop_last_diag=None
        from isaacgym import gymtorch
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self._ontop_box_contact=forces.view(self.num_envs,-1,3)[:,self.num_agents*self.num_bodies:self.num_agents*self.num_bodies+self.num_objects]

    def _evaluate_ontop_context(self, env_ids=None):
        bodies=self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids,...,:3]
        goals=self._tar_pos if env_ids is None else self._tar_pos[env_ids]
        objects=self._logical_box_values(self._box_states,env_ids)
        sizes=self._logical_box_values(self._box_size,env_ids)
        graph=self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph,env_ids)
        phi,diag=evaluate_ontop_edges(bodies[...,self._key_body_ids[[0,1]],:],bodies[...,0,:],objects,sizes,goals,graph,self._relation_cfg)
        if env_ids is None:
            self._ontop_last_diag=diag
        return phi,diag

    def _sample_episode_graph(self, ids):
        if self._relation_graph_spec.get('mode')=='edge_composition':
            new=sample_graph(len(ids),self._relation_graph_spec,self.device,self._task_graph_preset,self._task_role_swap)
        else:
            new=expand_graph(compile_ontop_graph(self._relation_graph_spec,self.num_agents,self.num_objects,self.device),len(ids))
        copy_graph_rows(self.relation_runtime.graph,ids,new)
        self._edge_goal_owners[ids]=owner_sum((new.required_goal & new.edge_valid).float(),new)>0
        self._ontop_maintain_steps[ids]=0
        self._sampling_counts[0]+=len(ids)
        if self.num_agents==2:
            second=owner_sum(((new.edge_relation==AT).long()+2*(new.edge_relation==ON_TOP).long())*new.edge_valid,new)
            self._sampling_counts[1:4]+=torch.stack([(second==k).sum() for k in range(3)])
            for e in (2,3,4):self._sampling_counts[4+e-2]+=(new.edge_valid.sum(-1)==e).sum()
            # Outcome family, counted once per reset (never weighted by episode length).
            lo=second.amin(-1);hi=second.amax(-1)
            dependent=((new.edge_relation==ON_TOP)&(new.edge_dst<self.num_agents+2)&new.edge_valid).any(-1)
            families=[(lo==0)&(hi==0),(lo==0)&(hi==1),(lo==0)&(hi==2),
                      (lo==1)&(hi==1),(lo==1)&(hi==2)&~dependent,
                      (lo==1)&(hi==2)&dependent,(lo==2)&(hi==2)]
            self._sampling_counts[7:14]+=torch.stack([f.sum() for f in families])
            self._sampling_counts[14]+=(families[5]&(second[:,0]==1)).sum()
            self._sampling_counts[15]+=(families[5]&(second[:,1]==1)).sum()
            for a in (0,1):
                lower=((new.edge_relation==ON_TOP)&(new.edge_dst==4)&(new.edge_owner==a)).any(-1)
                self._sampling_counts[16+a]+=(families[6]&lower).sum()

    def _reset_ontop_context_envs(self, env_ids):
        if not len(env_ids):return
        self._sample_episode_graph(env_ids)
        pending=env_ids
        for attempt in range(16):
            self._reset_default_slots=None;self._reset_ref_slots={}
            self._reset_ref_motion_ids={};self._reset_ref_motion_times={}
            self._reset_actors(pending)
            self._sample_box_assignments(pending)
            self._reset_boxes(pending)
            self._reset_task(pending)
            self._configure_ontop_targets(pending)
            rejected=self._ontop_scene_infeasible(pending)
            accepted=pending[~rejected]
            if len(accepted):
                # AMP slots are those initialized this attempt; commit all pending together.
                self._reset_env_tensors(pending)
                self._refresh_sim_tensors()
                self._reset_relation_history(accepted)
                self._compute_observations(accepted)
                self._init_amp_obs(pending)
            self._sampling_retries+=rejected.sum()
            pending=pending[rejected]
            if not len(pending):break
        if len(pending):
            self._sampling_failures+=len(pending)
            print('[edge composition] physical reset failed; graph retained; envs=',pending[:16].tolist(),flush=True)
            raise RuntimeError('OnTop physical scene infeasible after 16 attempts (graph was not resampled)')
        if self._mode=='test' and (env_ids==0).any():
            bindings=json.dumps(self._ontop_trace_bindings(1,include_state=False)[0])
            signature=hashlib.sha256(bindings.encode()).hexdigest()[:12]
            print('[task graph] preset={} role_swap={} seed={} reset={} signature={} graph={}'.format(
                self._task_graph_preset,self._task_role_swap,torch.initial_seed(),
                int(self._relation_episode_id[0]),signature,bindings),flush=True)

    def _configure_ontop_targets(self, ids):
        g=select_graph(self.relation_runtime.graph,ids)
        at=owner_sum(((g.edge_relation==AT)&g.edge_valid).long(),g).bool()
        sizes=self._assigned_box_values(self._box_size,ids)
        # Bound the TOP of the completed stack, using real asset dimensions.
        max_z=self._reset_max_top_surface_height-sizes[...,2]/2
        for a in range(self.num_agents):
            child=(g.edge_relation==ON_TOP)&(g.edge_dst==self.num_agents+a)&g.edge_valid
            source=(g.edge_src-self.num_agents).clamp(0,self.num_objects-1)
            logical_sizes=self._logical_box_values(self._box_size,ids)
            child_height=logical_sizes[torch.arange(len(ids),device=self.device)[:,None],source,2]
            max_z[:,a]-=(child_height*child).sum(-1)
        tar=self._tar_pos[ids].clone()
        tar[...,2]=torch.minimum(tar[...,2],max_z).clamp_min(sizes[...,2]/2)
        self._tar_pos[ids]=tar
        if self._reset_random_height:
            platforms=self._tar_platform_default_pos[ids].clone()
            active=at & (tar[...,2]-sizes[...,2]/2>=self._reset_min_platform_height)
            platforms[...,:2]=torch.where(active[...,None],tar[...,:2],platforms[...,:2])
            platforms[...,2]=torch.where(active,tar[...,2]-sizes[...,2]/2-self._platform_height/2,platforms[...,2])
            # If clamping made an elevated target too low for a platform, put it on the floor.
            self._tar_pos[ids,:,2]=torch.where(at & ~active,sizes[...,2]/2,tar[...,2])
            self._tar_platform_pos[ids]=platforms

    def _ontop_scene_infeasible(self, ids):
        boxes=self._logical_box_values(self._box_states,ids)
        sizes=self._logical_box_values(self._box_size,ids)
        roots=self._humanoid_root_states[ids,:,:3]
        g=select_graph(self.relation_runtime.graph,ids)
        at=owner_sum(((g.edge_relation==AT)&g.edge_valid).long(),g).bool()
        bad=torch.zeros(len(ids),device=self.device,dtype=torch.bool)
        # Bounding circles are conservative under arbitrary yaw. Assigned boxes may
        # legitimately start held; only reject box-box penetration and free-support obstruction.
        radius=sizes[...,:2].norm(dim=-1)/2
        for a in range(self.num_objects):
            for b in range(a):
                xy=(boxes[:,a,:2]-boxes[:,b,:2]).norm(dim=-1)
                z=(boxes[:,a,2]-boxes[:,b,2]).abs()
                bad|=(xy<radius[:,a]+radius[:,b]+.05)&(z<(sizes[:,a,2]+sizes[:,b,2])/2+.02)
        if self.num_objects>self.num_agents:
            free=boxes[:,self.num_agents:,:2]
            bad|=((free[:,:,None]-roots[:,None,:,:2]).norm(dim=-1)<self._box_min_agent_dist).any(-1).any(-1)
        for a in range(self.num_agents):
            goal=self._tar_pos[ids,a]
            for b in range(self.num_objects):
                if b==a:continue  # initially successful AT is valid
                bad|=at[:,a]&((goal[:,:2]-boxes[:,b,:2]).norm(dim=-1)<radius[:,a]+radius[:,b]+.25)
            for b in range(a):
                bad|=at[:,a]&at[:,b]&((goal[:,:2]-self._tar_pos[ids,b,:2]).norm(dim=-1)<radius[:,a]+radius[:,b]+.35)
        return bad

    def _record_ontop_diagnostics(self,result,diag):
        g=self.relation_runtime.graph;mask=g.edge_valid&(g.edge_relation==ON_TOP)
        contact=self._logical_box_values(self._ontop_box_contact).norm(dim=-1)
        batch=torch.arange(self.num_envs,device=self.device)[:,None]
        diag['source_net_contact']=contact[batch,(g.edge_src-self.num_agents).clamp(0,self.num_objects-1)]*mask
        diag['support_net_contact']=contact[batch,(g.edge_dst-self.num_agents).clamp(0,self.num_objects-1)]*mask
        self._ontop_maintain_steps=torch.where(result['own_success']&mask,self._ontop_maintain_steps+1,0)
        diag['maintain_seconds']=self._ontop_maintain_steps*self.dt
        for key in ('signed_gap','source_bottom_z','support_top_z','source_extent','support_extent',
                    'support_speed','relative_speed','source_tilt','support_tilt','maintain_seconds','source_net_contact','support_net_contact'):
            name='ontop/'+key
            value=(diag[key]*mask).sum()
            self._edge_metric_denominators[name]=self._edge_metric_denominators.get(name,0)+mask.sum()
            self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+value
        local=result['local_task_reward'];mixed=result['agent_task_reward']
        for key,value in [('local_task',local.mean()),('mixed_task',mixed.mean()),
                          ('self_contribution',(.9*local).mean()),('other_contribution',(.1*local.flip(-1)).mean())]:
            name='sharing/'+key;self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+value

    def _consume_sampling_diagnostics(self):
        c=self._sampling_counts;den=c[0].clamp_min(1)
        out={'sampling/resets':c[0].clone(),'sampling/physical_retries':self._sampling_retries.clone(),
             'sampling/physical_failures':self._sampling_failures.clone()}
        names=['second_none','second_at','second_ontop','edges_2','edges_3','edges_4',
               'holding_holding','holding_at','holding_ontop_free','at_at','at_ontop_free',
               'at_ontop_dependent','ontop_chain','dependent_a_at','dependent_b_at','chain_a_lower','chain_b_lower']
        for i,name in enumerate(names,1):out['sampling/'+name]=c[i]/(2*den if i<=3 else den)
        c.zero_();self._sampling_retries.zero_();self._sampling_failures.zero_()
        return out

    def _ontop_trace_bindings(self,n,include_state=True):
        g=select_graph(self.relation_runtime.graph,slice(0,n))
        data={k:getattr(g,k).cpu().tolist() for k in ('edge_src','edge_dst','edge_owner','edge_relation','edge_valid','prereq_mask','term_index','required_goal')}
        assignment=self._logical_box_order[:n].cpu().tolist()
        keys=('signed_gap','source_bottom_z','support_top_z','support_speed','relative_speed',
              'source_tilt','support_tilt','maintain_seconds','source_net_contact','support_net_contact')
        diag=self._ontop_last_diag
        values=targets=local=None
        if include_state and diag is not None:
            values=torch.stack([diag.get(key,torch.zeros_like(diag['signed_gap']))[:n] for key in keys],-1).cpu().tolist()
            targets=diag['target'][:n].cpu().tolist()
            local=self.relation_runtime.last_result['local_task_reward'][:n].cpu().tolist()
        result=[]
        for b in range(n):
            rows=[]
            for e in range(len(g.ids)):
                src=data['edge_src'][b][e];dst=data['edge_dst'][b][e];owner=data['edge_owner'][b][e]
                def physical(node):
                    return assignment[b][node-self.num_agents] if self.num_agents<=node<self.num_agents+self.num_objects else -1
                row=dict(valid=data['edge_valid'][b][e],source=src,target=dst,relation=data['edge_relation'][b][e],
                         binding_owner=owner,source_physical=physical(src),target_physical=physical(dst),
                         prerequisites='|'.join(str(i) for i,x in enumerate(data['prereq_mask'][b][e]) if x),
                         term=data['term_index'][b][e],required=data['required_goal'][b][e])
                if values is not None:
                    row.update(zip(keys,values[b][e]))
                    row.update(zip(('target_x','target_y','target_z'),targets[b][e]))
                    row.update(local_task_total=local[b][owner],self_task_contribution=.9*local[b][owner],
                               other_task_contribution=.1*local[b][1-owner])
                rows.append(row)
            result.append(rows)
        return result

    def _update_camera(self):
        if not getattr(self,'_edge_ontop',False) or getattr(self.cfg['args'],'task_camera','stack')!='stack':
            return super()._update_camera()
        if not hasattr(self,'_box_states'):
            return super()._update_camera()
        from isaacgym import gymapi
        points=torch.cat([self._humanoid_root_states[0,:,:3],self._box_states[0,:,:3]],0)
        low=points.amin(0);high=points.amax(0);center=(low+high)/2
        center[2]=max(float(center[2]),.8)
        radius=max(float((high-low)[:2].norm())*.8,3.)
        if self._camera_follow_flag:
            self.gym.viewer_camera_look_at(self.viewer,None,
                gymapi.Vec3(float(center[0])+radius*.65,float(center[1])-radius,2.8+radius*.2),
                gymapi.Vec3(*center.tolist()))
        self._cam_prev_char_pos[:]=self._humanoid_root_states[0,0,:3].cpu().numpy()

    def _draw_ontop_context(self):
        if not getattr(self,'_edge_ontop',False) or self.viewer is None or self._ontop_last_diag is None:
            return
        import numpy as np
        g=self.relation_runtime.graph
        ids=(g.edge_valid[0]&(g.edge_relation[0]==ON_TOP)).nonzero(as_tuple=False).flatten()
        self.gym.clear_lines(self.viewer)
        for e in ids.tolist():
            target=self._ontop_last_diag['target'][0,e].cpu().numpy()
            vertices=[]
            for axis in range(3):
                delta=np.zeros(3);delta[axis]=.12
                vertices.extend([target-delta,target+delta])
            owner=int(g.edge_owner[0,e]);color=self._agent_color(owner)
            colors=np.tile([color.x,color.y,color.z],(3,1)).astype(np.float32)
            self.gym.add_lines(self.viewer,self.envs[0],3,np.asarray(vertices,dtype=np.float32),colors)
