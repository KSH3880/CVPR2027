"""Sampled OnTop scene integration. Existing motion initialization and physics stay live."""
import json
import hashlib
import torch
from utils.edge_context_spec import AT
from utils.edge_ontop_spec import (ON_TOP, compile_ontop_graph, sample_graph, expand_graph,
    select_graph, copy_graph_rows, batched)
from utils.edge_interaction_spec import (SIT, CLIMB, compile_interaction_graph,
    sample_graph as sample_interaction_graph)
from utils.edge_stage1_spec import (compile_stage1_graph,
    sample_graph as sample_stage1_graph)
from env.tasks.multi_agent.edge_context_reward import owner_sum
from env.tasks.multi_agent.edge_ontop_reward import OnTopContextRuntime, evaluate_ontop_edges
from env.tasks.multi_agent.edge_interaction_reward import (InteractionContextRuntime,
    evaluate_interaction_edges)
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime


class SampledOnTopTaskMixin:
    def _keep_reset_slots_for_envs(self, env_ids):
        """Restrict per-attempt AMP reset metadata to the accepted environments."""
        def keep_mask(slot_env):
            return (slot_env[:, None] == env_ids[None, :]).any(dim=1)

        if self._reset_default_slots is not None:
            slot_env, slot_agent = self._reset_default_slots
            keep = keep_mask(slot_env)
            self._reset_default_slots = (slot_env[keep], slot_agent[keep]) if keep.any() else None

        for skill_name, slots in list(self._reset_ref_slots.items()):
            slot_env, slot_agent = slots
            keep = keep_mask(slot_env)
            if keep.any():
                self._reset_ref_slots[skill_name] = (slot_env[keep], slot_agent[keep])
                self._reset_ref_motion_ids[skill_name] = self._reset_ref_motion_ids[skill_name][keep]
                self._reset_ref_motion_times[skill_name] = self._reset_ref_motion_times[skill_name][keep]
            else:
                self._reset_ref_slots.pop(skill_name, None)
                self._reset_ref_motion_ids.pop(skill_name, None)
                self._reset_ref_motion_times.pop(skill_name, None)

    def _init_ontop_context_runtime(self):
        compiler=compile_stage1_graph if getattr(self,'_edge_stage1',False) else (compile_interaction_graph if getattr(self,'_edge_interaction',False) else compile_ontop_graph)
        prototype=compiler(self._relation_graph_spec,self.num_agents,self.num_objects,self.device)
        g=expand_graph(prototype,self.num_envs)
        runtime=Stage1ContextRuntime if getattr(self,'_edge_stage1',False) else (InteractionContextRuntime if getattr(self,'_edge_interaction',False) else OnTopContextRuntime)
        self.relation_runtime=runtime(self.num_envs,g,self._relation_cfg,self.device)
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
        hands=bodies[...,self._key_body_ids[[0,1]],:]
        if getattr(self,'_edge_interaction',False):
            feet=bodies[...,self._key_body_ids[[2,3]],:]
            phi,diag=evaluate_interaction_edges(hands,feet,bodies[...,0,:],objects,sizes,goals,
                                                graph,self._relation_cfg,self._char_h)
        else:
            phi,diag=evaluate_ontop_edges(hands,bodies[...,0,:],objects,sizes,goals,graph,self._relation_cfg)
        if env_ids is None:
            self._ontop_last_diag=diag
        return phi,diag

    def _sample_episode_graph(self, ids):
        if self._relation_graph_spec.get('mode')=='edge_composition':
            sampler=sample_stage1_graph if getattr(self,'_edge_stage1',False) else (sample_interaction_graph if getattr(self,'_edge_interaction',False) else sample_graph)
            new=sampler(len(ids),self._relation_graph_spec,self.device,self._task_graph_preset,self._task_role_swap)
        else:
            compiler=compile_stage1_graph if getattr(self,'_edge_stage1',False) else (compile_interaction_graph if getattr(self,'_edge_interaction',False) else compile_ontop_graph)
            new=expand_graph(compiler(self._relation_graph_spec,self.num_agents,self.num_objects,self.device),len(ids))
        copy_graph_rows(self.relation_runtime.graph,ids,new)
        self._edge_goal_owners[ids]=owner_sum((new.required_goal & new.edge_valid).float(),new)>0
        self._ontop_maintain_steps[ids]=0
        self._sampling_counts[0]+=len(ids)
        if self.num_agents==2 and getattr(self,'_edge_interaction',False):
            for a in (0,1):
                owner=(new.edge_owner==a)&new.edge_valid
                has_hold=((new.edge_relation==6)&owner).any(-1)
                relation=torch.where(((new.edge_relation==SIT)&owner).any(-1),SIT,
                    torch.where(((new.edge_relation==CLIMB)&owner).any(-1),CLIMB,
                    torch.where(((new.edge_relation==7)&owner).any(-1),7,
                    torch.where(((new.edge_relation==ON_TOP)&owner).any(-1),ON_TOP,6))))
                pattern=torch.where(~has_hold & (relation==SIT),1,
                    torch.where(~has_hold & (relation==CLIMB),2,
                    torch.where(has_hold & (relation==7),3,
                    torch.where(has_hold & (relation==ON_TOP),4,
                    torch.where(has_hold & (relation==(SIT if getattr(self,'_edge_stage1',False) else CLIMB)),5,
                    torch.where(has_hold & (relation==(CLIMB if getattr(self,'_edge_stage1',False) else SIT)),6,0))))))
                self._sampling_counts[1:8]+=torch.stack([(pattern==k).sum() for k in range(7)])
            if getattr(self,'_edge_stage1',False):
                ox=self.num_agents+2
                uses_ox=(new.edge_dst==ox)&new.edge_valid
                self._sampling_counts[8]+=uses_ox.any(-1).sum()
                for a in (0,1):
                    self._sampling_counts[9+a]+=((uses_ox&(new.edge_owner==a)).any(-1)).sum()
                for offset,rel in enumerate((6,7,ON_TOP,SIT,CLIMB),11):
                    self._sampling_counts[offset]+=((new.edge_relation==rel)&new.edge_valid).sum()
        elif self.num_agents==2:
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
        accepted_default=[]
        accepted_ref_slots={}
        accepted_ref_motion_ids={}
        accepted_ref_motion_times={}
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
                # Isaac Gym tensor setters may be called only once between simulation steps.
                # Keep accepted candidates in the shared tensors and aggregate their AMP
                # metadata; commit the complete reset batch once after every scene is valid.
                self._keep_reset_slots_for_envs(accepted)
                if self._reset_default_slots is not None:
                    accepted_default.append(self._reset_default_slots)
                for skill_name, slots in self._reset_ref_slots.items():
                    accepted_ref_slots.setdefault(skill_name, []).append(slots)
                    accepted_ref_motion_ids.setdefault(skill_name, []).append(
                        self._reset_ref_motion_ids[skill_name])
                    accepted_ref_motion_times.setdefault(skill_name, []).append(
                        self._reset_ref_motion_times[skill_name])
            self._sampling_retries+=rejected.sum()
            pending=pending[rejected]
            if not len(pending):break
        if len(pending):
            self._sampling_failures+=len(pending)
            print('[edge composition] physical reset failed; graph retained; envs=',pending[:16].tolist(),flush=True)
            raise RuntimeError('OnTop physical scene infeasible after 16 attempts (graph was not resampled)')

        self._reset_default_slots = None if not accepted_default else tuple(
            torch.cat([slots[i] for slots in accepted_default]) for i in (0, 1))
        self._reset_ref_slots = {
            skill_name: tuple(torch.cat([slots[i] for slots in chunks]) for i in (0, 1))
            for skill_name, chunks in accepted_ref_slots.items()
        }
        self._reset_ref_motion_ids = {
            skill_name: torch.cat(chunks) for skill_name, chunks in accepted_ref_motion_ids.items()
        }
        self._reset_ref_motion_times = {
            skill_name: torch.cat(chunks) for skill_name, chunks in accepted_ref_motion_times.items()
        }
        self._reset_env_tensors(env_ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(env_ids)
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)
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
        self_part=local if getattr(self,'_edge_stage1',False) else .9*local
        other_part=torch.zeros_like(local) if getattr(self,'_edge_stage1',False) else .1*local.flip(-1)
        for key,value in [('local_task',local.mean()),('mixed_task',mixed.mean()),
                          ('self_contribution',self_part.mean()),('other_contribution',other_part.mean())]:
            name='sharing/'+key;self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+value
        if getattr(self,'_edge_interaction',False):
            self._record_interaction_diagnostics(result,diag)

    def _record_interaction_diagnostics(self,result,diag):
        graph=self.relation_runtime.graph
        relation=graph.edge_relation
        values={
            'sit/distance_xyz':diag['distance'],
            'sit/distance_xy':diag['distance_xy'],
            'climb/root_target_distance_xyz':diag['distance'],
            'climb/distance_xy':diag['distance_xy'],
            'climb/z_feet':diag['z_feet'],
            'climb/z_surface':diag['z_surface'],
            'climb/feet_height_error':diag['feet_height_error'],
        }
        for name,value in values.items():
            rel=SIT if name.startswith('sit/') else CLIMB
            mask=graph.edge_valid&(relation==rel)
            self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+(value*mask).sum()
            self._edge_metric_denominators[name]=self._edge_metric_denominators.get(name,0)+mask.sum()

    def _consume_sampling_diagnostics(self):
        c=self._sampling_counts;den=c[0].clamp_min(1)
        out={'sampling/resets':c[0].clone(),'sampling/physical_retries':self._sampling_retries.clone(),
             'sampling/physical_failures':self._sampling_failures.clone()}
        if getattr(self,'_edge_interaction',False):
            names=('holding','sit','climb','holding_at','holding_ontop','holding_sit','holding_climb') if getattr(self,'_edge_stage1',False) else ('holding','sit','climb','holding_at','holding_ontop','holding_climb','holding_sit')
            for i,name in enumerate(names,1):
                out['sampling/pattern_'+name]=c[i]/(2*den)
            if getattr(self,'_edge_stage1',False):
                out.update({'sampling/ox_user_count':c[8]/den,
                            'sampling/ox_owner_a':c[9]/den,
                            'sampling/ox_owner_b':c[10]/den})
                for offset,name in enumerate(('holding','at','ontop','sit','climb'),11):
                    out['sampling/relation_'+name]=c[offset]/(2*den)
            c.zero_();self._sampling_retries.zero_();self._sampling_failures.zero_()
            return out
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
        if getattr(self,'_edge_interaction',False):
            keys+=('z_feet','z_surface','feet_height_error')
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
                    stage1=getattr(self,'_edge_stage1',False)
                    row.update(local_task_total=local[b][owner],
                               self_task_contribution=local[b][owner] if stage1 else .9*local[b][owner],
                               other_task_contribution=0. if stage1 else .1*local[b][1-owner])
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
        visible=(g.edge_relation[0]==ON_TOP)
        if getattr(self,'_edge_interaction',False):
            visible|=(g.edge_relation[0]==SIT)|(g.edge_relation[0]==CLIMB)
        ids=(g.edge_valid[0]&visible).nonzero(as_tuple=False).flatten()
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
