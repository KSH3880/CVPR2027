"""Simulator integration for the explicit Holding/At edge-context experiment."""
import csv
import os
import torch
from utils.edge_context_spec import HOLDING, AT, compile_edge_context_graph
from env.tasks.multi_agent.edge_context_reward import (
    EdgeContextRuntime, evaluate_edge_geometry, edge_context, goal_success, scene_success, owner_sum)
from env.tasks.multi_agent.relation_reward import box_speed_penalty
from utils.edge_ontop_spec import batched, select_graph, ON_TOP
from utils.edge_interaction_spec import SIT, CLIMB
from env.tasks.multi_agent.edge_ontop_reward import mix_task_reward
from env.tasks.multi_agent.edge_stage1_reward import stage1_context


class EdgeContextTaskMixin:
    def _init_relation_runtime(self):
        if not self._edge_context:
            return super()._init_relation_runtime()
        if getattr(self, '_edge_ontop', False):
            self._init_ontop_context_runtime()
            graph = self.relation_runtime.graph
        else:
            graph = compile_edge_context_graph(self._relation_graph_spec, self.num_agents, self.num_objects, self.device)
            self.relation_runtime = EdgeContextRuntime(self.num_envs, graph, self._relation_cfg, self.device)
        self._relation_episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._edge_goal_owners = owner_sum(batched(graph.required_goal & graph.edge_valid, self.num_envs).float(), graph) > 0
        self._edge_ever_goal = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool, device=self.device)
        self._edge_ever_scene = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._edge_metric_sums = {}
        self._edge_metric_denominators = {}
        self._edge_metric_steps = 0
        self._edge_episode_sums = torch.zeros(6, device=self.device)
        self._edge_steps = 0
        self._edge_csv_path = None

    def _evaluate_relations(self, env_ids=None):
        if not self._edge_context:
            return super()._evaluate_relations(env_ids)
        if getattr(self, '_edge_ontop', False):
            return self._evaluate_ontop_context(env_ids)
        bodies = self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids, ..., :3]
        goals = self._tar_pos if env_ids is None else self._tar_pos[env_ids]
        objects = self._logical_box_values(self._box_states, env_ids)[..., :3]
        return evaluate_edge_geometry(bodies[..., self._key_body_ids[[0, 1]], :], bodies[..., 0, :],
                                     objects, goals, self.relation_runtime.graph, self._relation_cfg)

    def _reset_relation_history(self, env_ids):
        if not self._edge_context:
            return super()._reset_relation_history(env_ids)
        phi, diag = self._evaluate_relations(env_ids)
        if getattr(self, '_edge_interaction', False):
            self.relation_runtime.reset(env_ids, phi, diag['z_error'], diag['feet_height_error'])
        else:
            self.relation_runtime.reset(env_ids, phi, diag['z_error'])
        self._prev_root_pos[env_ids] = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0, :3]
        self._prev_box_pos[env_ids] = self._assigned_box_values(self._box_states, env_ids)[..., :3]
        self._relation_episode_id[env_ids] += 1
        self._edge_ever_goal[env_ids] = self.relation_runtime.done[env_ids]
        self._edge_ever_scene[env_ids] = scene_success(self.relation_runtime.own_success[env_ids], select_graph(self.relation_runtime.graph, env_ids))

    @torch.no_grad()
    def _compute_relation_reward(self, collision_fn):
        if not self._edge_context:
            return super()._compute_relation_reward(collision_fn)
        runtime = self.relation_runtime; graph = runtime.graph
        phi, diag = self._evaluate_relations()
        if getattr(self, '_edge_interaction', False):
            result = runtime.step(phi, diag['progress'], diag['z_error'], diag['feet_height_error'])
        else:
            result = runtime.step(phi, diag['progress'], diag['z_error'])
        roots = self._humanoid_root_states[..., :3]
        objects = self._assigned_box_values(self._box_states)[..., :3]
        power = torch.zeros_like(result['agent_task_reward']); collision = torch.zeros_like(power); speed = torch.zeros_like(power)
        if self._power_reward:
            power = -self._power_coefficient * (self.dof_force_tensor * self._dof_vel).abs().sum(-1)
        if self._agent_collision_penalty and self.num_agents > 1:
            collision = -self._agent_collision_coeff * collision_fn(roots, self._agent_collision_dist)
        if self._box_vel_penalty:
            speed = box_speed_penalty(self._prev_box_pos, objects, self.dt, self._box_vel_pen_coeff, self._box_vel_pen_thre)
        reward = result['agent_task_reward'] + power + collision + speed
        components = [owner_sum(result[k], graph) for k in ('state_component', 'progress_component', 'success_component')]
        if getattr(self, '_edge_ontop', False) and not getattr(self, '_edge_stage1', False):
            components = [mix_task_reward(c) for c in components]
        terms = torch.stack(components + [power, collision, speed, reward], -1).flatten(0, 1)
        self.rew_buf.copy_(reward.flatten())
        self.extras['reward_terms'] = terms
        self._reward_term_sums += terms.sum(0); self._reward_term_count += self.num_envs * self.num_agents
        current = runtime.done.clone()
        scene = scene_success(runtime.own_success, graph)
        self._edge_ever_goal |= current; self._edge_ever_scene |= scene
        self.extras.update(subgoal_done=current, all_subgoals_done=scene, current_target_valid=current,
                           current_success_state=current, edge_own_success=runtime.own_success.clone(),
                           edge_reward_saturated=result['reward_saturated'].clone())
        dcfg = self._relation_cfg['diagnostics']; self._edge_steps += 1
        if not dcfg.get('enabled', True):
            return
        is_stage1 = getattr(self, '_edge_stage1', False)
        semantic_only = getattr(self, '_semantic_only_stage1', False)
        context_fields = ([] if semantic_only else
            ['q_start', 'q_keep'] if is_stage1 else ['q_pre', 'q_term'])
        fields = ['phi_raw', 'progress_raw'] + context_fields + ['own_success', 'term_success',
                  'reward_saturated', 'state_component', 'progress_component', 'success_component', 'total',
                  'distance', 'distance_xy', 'z_error']
        values = dict(result, distance=diag['distance'], distance_xy=diag['distance_xy'],
                      z_error=diag['z_error'])
        if not semantic_only:
            context = stage1_context(phi, graph) if is_stage1 else edge_context(phi, graph)
            values[context_fields[0]] = context[..., 0]
            values[context_fields[1]] = context[..., 1]
        packed = torch.stack([values[k].float() for k in fields], -1)
        relations = [(HOLDING, 'holding'), (AT, 'at')]
        if getattr(self, '_edge_ontop', False):
            relations.append((ON_TOP, 'ontop'))
        if getattr(self, '_edge_interaction', False):
            relations.extend([(SIT, 'sit'), (CLIMB, 'climb')])
        for rel, name in relations:
            mask = batched(graph.edge_valid & (graph.edge_relation == rel), self.num_envs)
            numerator = (packed * mask[..., None]).sum((0, 1))
            key = 'edge/' + name
            if key not in self._edge_metric_sums:
                self._edge_metric_sums[key] = torch.zeros_like(numerator)
            self._edge_metric_sums[key] += numerator
            self._edge_metric_denominators[key] = self._edge_metric_denominators.get(key, 0) + mask.sum()
        self._edge_metric_fields = fields
        extra = torch.stack([result['agent_task_reward'].mean(), (current.float() * self._edge_goal_owners).sum() / self._edge_goal_owners.sum().clamp_min(1), scene.float().mean(),
                             power.mean(), collision.mean(), speed.mean()])
        if 'agent' not in self._edge_metric_sums:
            self._edge_metric_sums['agent'] = torch.zeros_like(extra)
        self._edge_metric_sums['agent'] += extra
        if getattr(self, '_edge_ontop', False):
            self._record_ontop_diagnostics(result, diag)
        self._edge_metric_steps += 1
        if self._edge_steps % dcfg.get('log_interval', 30) == 0 and dcfg.get('sample_envs', 2):
            n = min(self.num_envs, dcfg.get('sample_envs', 2))
            if self._edge_csv_path is None:
                directory = os.path.join(getattr(self, '_relation_output_directory', self.cfg['args'].output_path), 'diagnostics')
                os.makedirs(directory, exist_ok=True)
                self._edge_csv_path = os.path.join(directory, 'edge_context_steps.csv')
            exists = os.path.exists(self._edge_csv_path)
            rows = packed[:n].cpu().tolist(); penalties = torch.stack([power[:n], collision[:n], speed[:n], result['agent_task_reward'][:n], reward[:n]], -1).cpu().tolist()
            owners = batched(graph.edge_owner, self.num_envs)[:n].tolist(); episodes = self._relation_episode_id[:n].tolist()
            bindings = self._ontop_trace_bindings(n) if getattr(self, '_edge_ontop', False) else None
            with open(self._edge_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(['step', 'env', 'episode', 'edge', 'owner'] + fields + ['power', 'collision', 'box_speed', 'agent_task_total', 'agent_total'] + (list(bindings[0][0]) if bindings else []))
                for env, edges in enumerate(rows):
                    for i, row in enumerate(edges):
                        writer.writerow([self._edge_steps, env, episodes[env], graph.ids[i], owners[env][i]] + row + penalties[env][owners[env][i]] + (list(bindings[env][i].values()) if bindings else []))

    def _finish_relation_diagnostics(self):
        if not self._edge_context:
            return super()._finish_relation_diagnostics()
        ending = self.reset_buf.bool()
        current = self.relation_runtime.done
        scene = scene_success(self.relation_runtime.own_success, self.relation_runtime.graph)
        self._edge_episode_sums += torch.stack([ending.sum(), (current & ending[:, None]).sum(),
            (self._edge_ever_goal & ending[:, None]).sum(), (scene & ending).sum(), (self._edge_ever_scene & ending).sum(), (self._edge_goal_owners & ending[:, None]).sum()])

    def consume_relation_diagnostics(self):
        if not self._edge_context:
            return super().consume_relation_diagnostics()
        result = {}
        for key, sums in self._edge_metric_sums.items():
            if key.startswith(('ontop/', 'sharing/', 'sit/', 'climb/')):
                denominator = self._edge_metric_denominators.get(key, max(self._edge_metric_steps, 1))
                if isinstance(denominator, torch.Tensor):
                    if denominator == 0:
                        continue
                    denominator = denominator.clamp_min(1)
                result[key] = sums / denominator
                continue
            names = (['agent/task_total', 'goal/current_agent_success', 'goal/current_scene_success',
                      'penalty/power', 'penalty/collision', 'penalty/box_speed'] if key == 'agent'
                     else [key + '/' + field for field in self._edge_metric_fields])
            denominator = self._edge_metric_denominators.get(key, max(self._edge_metric_steps, 1))
            if isinstance(denominator, torch.Tensor):
                if denominator == 0:
                    continue
                denominator = denominator.clamp_min(1)
            result.update(zip(names, sums / denominator))
        sums = self._edge_episode_sums.clone(); count = sums[0].clamp_min(1)
        result.update({'goal/completed_scenes': sums[0], 'goal/final_agent_success': sums[1] / sums[5].clamp_min(1),
            'goal/ever_agent_success': sums[2] / sums[5].clamp_min(1),
            'goal/final_scene_success': sums[3] / count, 'goal/ever_scene_success': sums[4] / count})
        if getattr(self, '_edge_ontop', False):
            result.update(self._consume_sampling_diagnostics())
        self._edge_metric_sums = {}; self._edge_metric_denominators = {}; self._edge_metric_steps = 0; self._edge_episode_sums.zero_()
        return result
