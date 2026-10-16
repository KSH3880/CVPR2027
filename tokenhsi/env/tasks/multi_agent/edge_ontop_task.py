"""Sampled OnTop scene integration. Existing motion initialization and physics stay live."""
import json
import hashlib
import torch
from utils.edge_context_spec import AT
from utils.edge_ontop_spec import ON_TOP, expand_graph, select_graph, copy_graph_rows
from utils.edge_interaction_spec import SIT, CLIMB
from utils.mixed_carry_spec import compile_graph as compile_mixed_graph
from env.tasks.multi_agent.edge_context_reward import owner_sum
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.scene_features import (
    task_target_owners, task_marker_vertices, SHARED_TASK_COLOR)


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
        compiler=(compile_mixed_graph)
        prototype=compiler(self._relation_graph_spec,self.num_agents,self.num_objects,self.device)
        g=expand_graph(prototype,self.num_envs)
        from env.tasks.multi_agent.joint_carry_reward import JointCarryRuntime
        runtime = JointCarryRuntime
        self.relation_runtime=runtime(self.num_envs,g,self._relation_cfg,self.device)
        self._sampling_counts=torch.zeros(18,device=self.device)
        self._sampling_retries=torch.zeros((),device=self.device)
        self._sampling_failures=torch.zeros((),device=self.device)
        self._ontop_putdown_rsi_count=torch.zeros((),device=self.device)
        self._stage2_initial_rejections=torch.zeros((),device=self.device)
        self._stage2_rsi_counts=torch.zeros(len(self._skill),device=self.device)
        self._rsi_attempts=torch.zeros(2,device=self.device)
        self._rsi_rejected=torch.zeros(2,device=self.device)
        self._rsi_start_distance=torch.zeros(2,device=self.device)
        self._ontop_maintain_steps=torch.zeros(self.num_envs,len(g.ids),device=self.device)
        self._ontop_last_diag=None
        from isaacgym import gymtorch
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self._ontop_box_contact=forces.view(self.num_envs,-1,3)[:,self.num_agents*self.num_bodies:self.num_agents*self.num_bodies+self.num_physical_objects]

    def _evaluate_ontop_context(self, env_ids=None):
        bodies=self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids,...,:3]
        goals=self._tar_pos if env_ids is None else self._tar_pos[env_ids]
        objects=self._logical_box_values(self._box_states,env_ids)
        sizes=self._logical_box_values(self._box_size,env_ids)
        graph=self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph,env_ids)
        hands=bodies[...,self._key_body_ids[[0,1]],:]
        feet=bodies[...,self._key_body_ids[[2,3]],:]
        phi,diag=evaluate_interaction_edges(hands,feet,bodies[...,0,:],objects,sizes,goals,
                                            graph,self._relation_cfg,self._char_h)
        from env.tasks.multi_agent.joint_carry_reward import apply_joint_geometry
        phi, gate = apply_joint_geometry(phi, diag, hands, objects, sizes, graph, self._relation_cfg)
        if env_ids is None:
            self.relation_runtime.coupled_holding.copy_(gate)
        else:
            self.relation_runtime.coupled_holding[env_ids] = gate
        if env_ids is None:
            self._ontop_last_diag=diag
        return phi,diag

    def _sample_episode_graph(self, ids):
        from utils.mixed_carry_spec import sample_graph as sample_mixed
        new = sample_mixed(len(ids), self._relation_graph_spec, self.device,
            self._task_graph_preset, joint_mask=self._joint_env_mask[ids])
        copy_graph_rows(self.relation_runtime.graph,ids,new)
        self._edge_goal_owners[ids]=owner_sum((new.required_goal & new.edge_valid).float(),new)>0
        self._ontop_maintain_steps[ids]=0
        self._sampling_counts[0]+=len(ids)
        ontop = ((new.edge_relation == ON_TOP) & new.edge_valid).any(-1)
        from utils.mixed_carry_spec import joint_scenes
        joint = joint_scenes(new)
        self._sampling_counts[1] += (joint & ~ontop).sum()
        self._sampling_counts[2] += (joint & ontop).sum()
        self._sampling_counts[3] += (~joint).sum()

    def _reset_ontop_context_envs(self, env_ids, finalize=True):
        if not len(env_ids):return
        self._sample_episode_graph(env_ids)
        accepted_default=[]
        accepted_ref_slots={}
        accepted_ref_motion_ids={}
        accepted_ref_motion_times={}
        pending=env_ids
        max_attempts = (64)
        for attempt in range(max_attempts):
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
                putdown_supports = self._putdown_ontop_supports()
                if putdown_supports is not None:
                    self._ontop_putdown_rsi_count += (
                        (putdown_supports[0][:, None] == accepted[None, :]).any(-1).sum())
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
            raise RuntimeError('OnTop physical scene infeasible after {} attempts '
                               '(graph was not resampled)'.format(max_attempts))

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
        if getattr(self, '_size_aware_rsi', False) and not self._is_eval:
            from utils.edge_scenario_spec import classify_templates
            from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
            reference_phi, reference_diag = self._evaluate_ontop_context(env_ids)
            reference_graph = select_graph(self.relation_runtime.graph, env_ids)
            reference_success = interaction_own_success(reference_phi, reference_diag['z_error'],
                reference_diag['feet_height_error'], reference_graph, self._relation_cfg,
                reference_diag['region_error'], distance=reference_diag['distance'])
            for index, relation in enumerate((6, AT, ON_TOP, SIT, CLIMB)):
                mask = reference_graph.edge_valid & (reference_graph.edge_relation == relation)
                self._size_rsi_success_counts[0, index, 0] += (reference_success & mask).sum()
                self._size_rsi_success_counts[0, index, 1] += mask.sum()
            self._size_rsi_pending_post[env_ids] = True
            for skill_index, skill_name in enumerate(self._skill):
                slots = self._reset_ref_slots.get(skill_name)
                if slots is not None:
                    template = classify_templates(self.relation_runtime.graph, *slots, True)
                    self._size_rsi_executed.index_put_((template, torch.full_like(template, skill_index)),
                        torch.ones_like(template, dtype=torch.float), accumulate=True)
        for skill_index, skill_name in enumerate(self._skill):
            if skill_name in self._reset_ref_slots:
                self._stage2_rsi_counts[skill_index] += len(self._reset_ref_slots[skill_name][0])
        if not finalize:
            return
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
        self._configure_putdown_ontop_supports()


    def _putdown_ontop_supports(self):
        if (not getattr(self, '_size_aware_rsi', False) and self._relation_cfg.get('stage1_variant') !=
                'scenario_independent_stage1_ontop_putdown'
                or 'putDown' not in self._reset_ref_slots):
            return None
        slot_env, slot_agent = self._reset_ref_slots['putDown']
        graph = self.relation_runtime.graph
        edges = (graph.edge_valid[slot_env] &
                 (graph.edge_relation[slot_env] == ON_TOP) &
                 (graph.edge_owner[slot_env] == slot_agent[:, None]))
        selected = edges.any(-1)
        if not selected.any():
            return None
        slot_env = slot_env[selected]
        slot_agent = slot_agent[selected]
        edge = edges[selected].long().argmax(-1)
        logical = graph.edge_dst[slot_env, edge] - self.num_agents
        physical = self._logical_box_order[slot_env, logical]
        motion_ids = self._reset_ref_motion_ids['putDown'][selected]
        return slot_env, slot_agent, physical, motion_ids

    def _configure_putdown_ontop_supports(self):
        slots = self._putdown_ontop_supports()
        if slots is None:
            return
        slot_env, slot_agent, support, motion_ids = slots
        motion = self._motion_lib['putDown']
        final_pos, _ = motion.get_obj_motion_state(
            motion_ids=motion_ids,
            motion_times=motion.get_motion_length(motion_ids))
        offset = self._agent_spawn_offsets[slot_agent] + self._env_origins[slot_env]
        self._box_states[slot_env, support, :2] = (final_pos + offset)[:, :2]
        self._box_states[slot_env, support, 2] = \
            self._box_size[slot_env, support, 2] / 2
        self._box_states[slot_env, support, 7:13] = 0.
        self._box_states[slot_env, support, 3:7] = torch.tensor(
            [0., 0., 0., 1.], device=self.device)


    def _ontop_scene_infeasible(self, ids):
        boxes=self._logical_box_values(self._box_states,ids)
        sizes=self._logical_box_values(self._box_size,ids)
        self._humanoid_root_states[ids,:,:3]
        g=select_graph(self.relation_runtime.graph,ids)
        owner_sum(((g.edge_relation==AT)&g.edge_valid).long(),g).bool()
        bad=torch.zeros(len(ids),device=self.device,dtype=torch.bool)
        # Bounding circles are conservative under arbitrary yaw. Assigned boxes may
        # legitimately start held; only reject box-box penetration and free-support obstruction.
        radius=sizes[...,:2].norm(dim=-1)/2
        screened_pairs = torch.zeros(len(ids), self.num_objects, self.num_objects,
                                     device=self.device, dtype=torch.bool)
        slots = self._putdown_ontop_supports()
        if slots is not None:
            slot_env, slot_agent, physical, _ = slots
            rows = torch.searchsorted(ids, slot_env)
            screened_pairs[rows, physical, slot_agent] = True
            screened_pairs[rows, slot_agent, physical] = True
        for a in range(self.num_objects):
            for b in range(a):
                xy=(boxes[:,a,:2]-boxes[:,b,:2]).norm(dim=-1)
                z=(boxes[:,a,2]-boxes[:,b,2]).abs()
                bad|=(xy<radius[:,a]+radius[:,b]+.05)&(z<(sizes[:,a,2]+sizes[:,b,2])/2+.02)&~screened_pairs[:,a,b]
        putdown_supports = self._putdown_ontop_supports()
        body = self._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        relative = body[:, :, None] - boxes[:, None, :, None, :3]
        rotation = boxes[:, None, :, None, 3:7].expand(*relative.shape[:-1], 4)
        cross = 2 * torch.cross(rotation[..., :3], relative, dim=-1)
        local = relative - rotation[..., 3:4]*cross + torch.cross(rotation[..., :3], cross, dim=-1)
        inside = (local.abs() < sizes[:, None, :, None]/2 - .06).all(-1).any(-1)
        exempt = torch.zeros(len(ids), self.num_agents, self.num_objects,
                              dtype=torch.bool, device=self.device)
        for owner in range(self.num_agents):
            exempt[:, owner, owner] = True
        if putdown_supports is not None:
            slot_env, slot_agent, physical, _ = putdown_supports
            exempt[torch.searchsorted(ids, slot_env), slot_agent, physical] = True
        bad |= (inside & ~exempt).any(-1).any(-1)
        batch=torch.arange(len(ids),device=self.device)[:,None]
        at_edges=(g.edge_relation==AT)&g.edge_valid
        src=(g.edge_src-self.num_agents).clamp(0,self.num_objects-1)
        goal=(g.edge_dst-self.num_agents-self.num_objects).clamp(0,self.num_agents-1)
        source_radius=radius.gather(1,src)
        goal_pos=self._tar_pos[ids][batch,goal]
        for logical in range(self.num_objects):
            other=(logical!=src)&at_edges
            distance=(goal_pos[...,:2]-boxes[:,logical,:2][:,None,:]).norm(dim=-1)
            bad|=(other&(distance<source_radius+radius[:,logical,None]+.25)).any(-1)
        skills = ((('sit', 'climb')))
        for skill_index, skill_name in enumerate(skills):
            slots = self._reset_ref_slots.get(skill_name)
            if slots is None:
                continue
            slot_env, slot_agent = slots
            row = torch.searchsorted(ids, slot_env)
            physical = ((self._scenario_physical_object(slot_env, slot_agent)))
            box = self._box_states[slot_env, physical]
            size = self._box_size[slot_env, physical]
            root = self._humanoid_root_states[slot_env, slot_agent, :3]
            feet = self._kinematic_humanoid_rigid_body_states[
                slot_env, slot_agent][:, self._key_body_ids[[2, 3]], :3]
            body = self._kinematic_humanoid_rigid_body_states[
                slot_env, slot_agent, :, :3]
            rotation = box[:, None, 3:7].expand(-1, body.shape[1], -1)
            relative = body - box[:, None, :3]
            cross = 2 * torch.cross(rotation[..., :3], relative, dim=-1)
            local_body = (relative - rotation[..., 3:4] * cross +
                          torch.cross(rotation[..., :3], cross, dim=-1))
            inner_xy = (local_body[..., :2].abs() <
                (size[:, None, :2] / 2 - .06)).all(-1)
            inner_z = ((local_body[..., 2] > -size[:, None, 2] / 2 + .06) &
                       (local_body[..., 2] < size[:, None, 2] / 2 - .06))
            penetration = (inner_xy & inner_z).any(-1)
            invalid = penetration | (feet[..., 2].amin(-1) < -.02)
            invalid = ~torch.isfinite(body).all(-1).all(-1)
            bad[row] |= invalid
            self._rsi_attempts[skill_index] += len(slot_env)
            self._rsi_rejected[skill_index] += invalid.sum()
            target_z = box[:, 2] + size[:, 2] / 2 + (
                self._relation_cfg['sit']['pelvis_clearance'] if skill_name == 'sit'
                else self._char_h)
            target = torch.cat((box[:, :2], target_z[:, None]), -1)
            self._rsi_start_distance[skill_index] += (root - target).norm(dim=-1).sum()
        return bad

    def _record_ontop_diagnostics(self,result,diag):
        g=self.relation_runtime.graph;mask=g.edge_valid&(g.edge_relation==ON_TOP)
        if getattr(self, '_size_aware_rsi', False) and not self._is_eval:
            for index, relation in enumerate((6, AT, ON_TOP, SIT, CLIMB)):
                first = g.edge_valid & (g.edge_relation == relation) & self._size_rsi_pending_post[:, None]
                self._size_rsi_success_counts[1, index, 0] += (result['own_success'] & first).sum()
                self._size_rsi_success_counts[1, index, 1] += first.sum()
            self._size_rsi_pending_post.zero_()
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
        name='ontop/region_error'
        self._edge_metric_denominators[name]=self._edge_metric_denominators.get(name,0)+mask.sum()
        self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+(diag['region_error']*mask).sum()
        local=result['local_task_reward'];mixed=result['agent_task_reward']
        sharing=self._relation_cfg.get('task_sharing')
        if sharing is not None:
            self_part=sharing['self']*local
            teammate = result.get('teammate_task_reward')
            if teammate is None:
                teammate = ((local.sum(-1, keepdim=True) - local) / (local.shape[-1] - 1)
                            if getattr(self, '_stage2', False) and local.shape[-1] > 2
                            else local.flip(-1))
            other_part=sharing['teammate']*teammate
        else:
            self_part=(local)
            other_part=(torch.zeros_like(local))
        for key,value in [('local_task',local.mean()),('mixed_task',mixed.mean()),
                          ('self_contribution',self_part.mean()),('other_contribution',other_part.mean())]:
            name='sharing/'+key;self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+value
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
        climb_cfg = self._relation_cfg['climb']
        plane = True
        root_pass = ((diag['region_error'] <= 0) &
            (diag['z_error'].abs() <= climb_cfg['root_height_tolerance'])) if plane else \
            result['phi_raw'] >= climb_cfg['success_phi_threshold']
        feet_pass = diag['feet_height_error'] <= climb_cfg['feet_height_tolerance']
        values.update({
            ('climb/root_region_height_pass' if plane else 'climb/root_phi_pass'): root_pass.float(),
            'climb/feet_pass': feet_pass.float(),
            'climb/joint_pass': (root_pass & feet_pass).float(),
            'climb/strict_phi_0_7_feet_0_05': ((result['phi_raw'] >= .7) &
                (diag['feet_height_error'] <= .05)).float(),
            'climb/feet_individual_max_error': diag['feet_individual_max_error'],
        })
        if plane:
            values['climb/region_error'] = diag['region_error']
        sit_cfg = self._relation_cfg['sit']
        values['sit/region_error'] = diag['region_error']
        values['sit/z_error'] = diag['z_error'].abs()
        values['sit/plane_pass'] = ((diag['region_error'] <= 0) &
            (diag['z_error'].abs() <= sit_cfg['z_tolerance'])).float()
        for name,value in values.items():
            rel=SIT if name.startswith('sit/') else CLIMB
            mask=graph.edge_valid&(relation==rel)
            self._edge_metric_sums[name]=self._edge_metric_sums.get(name,0)+(value*mask).sum()
            self._edge_metric_denominators[name]=self._edge_metric_denominators.get(name,0)+mask.sum()

    def _consume_sampling_diagnostics(self):
        c=self._sampling_counts;den=c[0].clamp_min(1)
        out={'sampling/resets':c[0].clone(),'sampling/physical_retries':self._sampling_retries.clone(),
             'sampling/physical_failures':self._sampling_failures.clone()}
        out['sampling/joint_carry_at'] = c[1] / den
        out['sampling/joint_carry_ontop'] = c[2] / den
        joint = ((self._joint_env_mask))
        out['joint/coupled_holding'] = (
            (self.relation_runtime.coupled_holding & joint).sum() / joint.sum().clamp_min(1))
        out['sampling/independent'] = c[3] / den
        out['sampling/joint_environment_fraction'] = joint.float().mean()
        for index, name in enumerate(self._skill):
            out['sampling/rsi_' + name] = self._stage2_rsi_counts[index] / (den * self.num_agents)
        c.zero_(); self._sampling_retries.zero_(); self._sampling_failures.zero_()
        self._stage2_rsi_counts.zero_()
        return out

    def _ontop_trace_bindings(self,n,include_state=True):
        g=select_graph(self.relation_runtime.graph,slice(0,n))
        data={k:getattr(g,k).cpu().tolist() for k in ('edge_src','edge_dst','edge_owner','edge_relation','edge_valid','prereq_mask','term_index','required_goal')}
        assignment=self._logical_box_order[:n].cpu().tolist()
        keys=('signed_gap','source_bottom_z','support_top_z','support_speed','relative_speed',
              'source_tilt','support_tilt','maintain_seconds','source_net_contact','support_net_contact')
        keys+=('z_feet','z_surface','feet_height_error')
        diag=self._ontop_last_diag
        values=targets=local=teammate=None
        if include_state and diag is not None:
            values=torch.stack([diag.get(key,torch.zeros_like(diag['signed_gap']))[:n] for key in keys],-1).cpu().tolist()
            targets=diag['target'][:n].cpu().tolist()
            last=self.relation_runtime.last_result
            local=last['local_task_reward'][:n].cpu().tolist()
            if 'teammate_task_reward' in last:
                teammate=last['teammate_task_reward'][:n].cpu().tolist()
        stage1=getattr(self,'_edge_stage1',False)
        sharing=self._relation_cfg.get('task_sharing',
            {'self': 1. if stage1 else .9, 'teammate': 0. if stage1 else .1})
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
                    row.update(local_task_total=local[b][owner],
                               self_task_contribution=sharing['self']*local[b][owner],
                               other_task_contribution=sharing['teammate']*(
                                   teammate[b][owner] if teammate is not None else
                                   (sum(local[b])-local[b][owner])/(self.num_agents-1)))
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
        visible|=(g.edge_relation[0]==SIT)|(g.edge_relation[0]==CLIMB)
        ids=(g.edge_valid[0]&visible).nonzero(as_tuple=False).flatten()
        owners_by_task = task_target_owners(g, 0)
        self.gym.clear_lines(self.viewer)
        for e in ids.tolist():
            target=self._ontop_last_diag['target'][0,e].cpu().numpy()
            relation=int(g.edge_relation[0,e])
            destination=int(g.edge_dst[0,e])
            vertices=task_marker_vertices(relation, target)
            num_lines=len(vertices)//2
            owner=int(g.edge_owner[0,e])
            if len(owners_by_task.get((relation, destination), ())) > 1:
                color_rgb = SHARED_TASK_COLOR
            else:
                color=self._agent_color(owner)
                color_rgb = (color.x, color.y, color.z)
            colors=np.tile(color_rgb,(num_lines,1)).astype(np.float32)
            self.gym.add_lines(self.viewer,self.envs[0],num_lines,vertices,colors)
