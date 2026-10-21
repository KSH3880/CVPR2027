"""Carry integration of the simulator-independent relation state machine."""
import csv
import os
import torch

from utils.relation_task_spec import compile_carry_subgoal
from env.tasks.multi_agent.relation_reward import (
    RelationRuntime, evaluate_holding, evaluate_at, relation_progress, box_speed_penalty)
from env.tasks.multi_agent.relation_diagnostics import (
    PlacementEpisodeMetrics, RelationTimeline, placement_valid, split_ontop_reward_terms)


class CarryRelationMixin:
    def _init_relation_runtime(self):
        self.relation_runtime = RelationRuntime(self.num_envs,
            compile_carry_subgoal(self.num_agents, self.num_objects, self.device),
            self._relation_cfg, self.device)
        if getattr(self, '_ontop_mixed', False):
            self._init_ontop_runtime()
        self._relation_diagnostic_sums = {}
        self._relation_diagnostic_count = 0
        self._relation_steps = 0
        self._relation_episode_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._relation_first_holding = torch.full((self.num_envs, self.num_agents), -1., device=self.device)
        self._relation_first_valid = torch.full_like(self._relation_first_holding, -1.)
        self._relation_csv_path = None
        diagnostics = self._relation_cfg.get('diagnostics', {})
        self._placement_metrics = (PlacementEpisodeMetrics(
            self.num_envs, self.num_agents, self.dt, self.device)
            if diagnostics.get('enabled', True) else None)
        self._relation_timeline = RelationTimeline(
            os.path.join(self.cfg['args'].output_path, 'diagnostics'), self.num_envs, diagnostics)

    def _evaluate_relations(self, env_ids=None):
        if env_ids is None:
            bodies = self._rigid_body_pos
            goals = self._tar_pos
        else:
            bodies = self._kinematic_humanoid_rigid_body_states[env_ids, ..., :3]
            goals = self._tar_pos[env_ids]
        objects = self._assigned_box_values(self._box_states, env_ids)[..., :3]
        hands = bodies[..., self._key_body_ids[[0, 1]], :]
        h_cfg, a_cfg = self._relation_cfg.get('holding', {}), self._relation_cfg.get('at', {})
        h, hand_error = evaluate_holding(hands, objects, h_cfg.get('hand_distance_scale', 5.))
        a, near, put, xy, z = evaluate_at(objects, goals,
            near_scale=a_cfg.get('near_distance_scale', 10.),
            state_definition=a_cfg.get('state_definition', 'box_near'))
        if getattr(self, '_ontop_mixed', False):
            source, target, _, is_top = self._ontop_geometry(env_ids)
            top, top_near, top_put, top_xy, top_z = evaluate_at(source, target,
                near_scale=self._relation_cfg['ontop']['distance_scale'])
            a, near, put, xy, z = [torch.where(is_top, new, old) for new, old in
                                   zip((top, top_near, top_put, top_xy, top_z), (a, near, put, xy, z))]
        return torch.stack([h, a], -1).flatten(1), dict(
            hand_midpoint_distance=hand_error,
            right_hand_center_distance=(hands[..., 0, :] - objects).norm(dim=-1),
            left_hand_center_distance=(hands[..., 1, :] - objects).norm(dim=-1),
            root_box_distance_xy=(bodies[..., 0, :2] - objects[..., :2]).norm(dim=-1),
            goal_xy_error=xy, goal_z_error=z, near=near, put=put.float())

    def _reset_relation_history(self, env_ids):
        phi, diag = self._evaluate_relations(env_ids)
        self.relation_runtime.reset(env_ids, phi, at_z_error=diag.get('goal_z_error'))
        self._prev_root_pos[env_ids] = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0, :3]
        self._prev_box_pos[env_ids] = self._assigned_box_values(self._box_states, env_ids)[..., :3]
        self._relation_episode_id[env_ids] += 1
        holding = self.relation_runtime.achieved[env_ids, 0::2]
        self._relation_first_holding[env_ids] = torch.where(holding, 0., -1.)
        self._relation_first_valid[env_ids] = torch.where(self.relation_runtime.done[env_ids], 0., -1.)
        if self._placement_metrics is not None:
            self._placement_metrics.reset(env_ids, placement_valid(diag['goal_xy_error'], diag['goal_z_error']))
        self._relation_timeline.reset(env_ids)

    def _finish_relation_diagnostics(self):
        # Called AFTER _compute_reset, while terminal box states are still live.
        if self._placement_metrics is not None:
            self._placement_metrics.finish(self.reset_buf)
        self._relation_timeline.finish(self.reset_buf)
        if getattr(self, '_ontop_mixed', False):
            self._finish_ontop_metrics()

    @torch.no_grad()
    def _compute_relation_reward(self, collision_fn):
        runtime = self.relation_runtime
        phi, diag = self._evaluate_relations()
        objects = self._assigned_box_values(self._box_states)[..., :3]
        roots = self._humanoid_root_states[..., :3]
        cfg = self._relation_cfg.get('progress', {})
        ph = relation_progress(self._prev_root_pos, roots, objects, self.dt, cfg)
        progress_targets = self._ontop_geometry()[2] if getattr(self, '_ontop_mixed', False) else self._tar_pos
        pa = relation_progress(self._prev_box_pos, objects, progress_targets, self.dt, cfg)
        previous_satisfied = runtime.phi >= self._relation_cfg.get('satisfaction_threshold', .9)
        live = ~runtime.done.clone()
        edge_distance_xy = None
        if 'approach_radius' in cfg:
            # Same post-step XY endpoints as each edge's direction progress:
            # Human root -> assigned Object, assigned Object -> Target.
            holding_distance_xy = (objects[..., :2] - roots[..., :2]).norm(dim=-1)
            edge_distance_xy = torch.stack(
                [holding_distance_xy, diag['goal_xy_error']], -1).flatten(1)
        result = runtime.step(phi, torch.stack([ph, pa], -1).flatten(1),
                              edge_distance_xy=edge_distance_xy,
                              at_z_error=diag.get('goal_z_error'))
        if getattr(self, '_ontop_mixed', False):
            self._ontop_step_metrics(result)
        power = torch.zeros_like(result['agent_task_reward'])
        collision = torch.zeros_like(power)
        box_penalty = torch.zeros_like(power)
        if self._power_reward:
            power = -self._power_coefficient * (self.dof_force_tensor * self._dof_vel).abs().sum(-1)
        if self._agent_collision_penalty and self.num_agents > 1:
            collision = -self._agent_collision_coeff * collision_fn(roots, self._agent_collision_dist)
        if self._box_vel_penalty:
            box_penalty = box_speed_penalty(self._prev_box_pos, objects, self.dt,
                                            self._box_vel_pen_coeff, self._box_vel_pen_thre)
        reward = result['agent_task_reward'] + power + collision + box_penalty
        terms = torch.stack([result['state_component'][:, 0::2], result['state_component'][:, 1::2],
            result['progress_component'][:, 0::2], result['progress_component'][:, 1::2],
            result['success_bonus'], power, collision, box_penalty, reward], -1).flatten(0, 1)
        if getattr(self, '_ontop_mixed', False):
            from utils.ontop_task_spec import REL_ONTOP
            terms = split_ontop_reward_terms(terms, (runtime.graph.edge_relation[:, 1::2] == REL_ONTOP).flatten())
        self.rew_buf.copy_(reward.flatten())
        self.extras['reward_terms'] = terms
        self._reward_term_sums += terms.sum(0)
        self._reward_term_count += self.num_envs * self.num_agents
        self.extras['subgoal_done'] = runtime.done.clone()
        self.extras['all_subgoals_done'] = runtime.done.all(-1)
        self.extras['current_target_valid'] = result['valid_next'][:, 1::2]
        self.extras['current_success_state'] = result['current_success_state'].clone()
        self.extras['saturation_active'] = result['saturation_active'].clone()
        self._record_relation_diagnostics(diag, result, previous_satisfied, live, objects, ph, pa)

    def _record_relation_diagnostics(self, diag, result, previous_satisfied, live, objects, ph, pa):
        runtime = self.relation_runtime
        self._relation_steps += 1
        t = self.progress_buf[:, None].float() * self.dt
        h = result['satisfied_next'][:, 0::2]
        self._relation_first_holding.copy_(torch.where(
            (self._relation_first_holding < 0) & h, t, self._relation_first_holding))
        self._relation_first_valid.copy_(torch.where(
            (self._relation_first_valid < 0) & result['first_success'], t, self._relation_first_valid))
        dcfg = self._relation_cfg.get('diagnostics', {})
        if not dcfg.get('enabled', True):
            return
        placed = placement_valid(diag['goal_xy_error'], diag['goal_z_error'])
        self._placement_metrics.step(placed)
        speed = ((objects - self._prev_box_pos) / self.dt).norm(dim=-1)
        diag.update(box_speed=speed, progress_holding=ph, progress_at=pa,
                    task_relation_total=result['agent_task_reward'],
                    success_bonus=result['success_bonus'],
                    first_success_bonus=result['first_success_bonus'],
                    current_success_reward=result['current_success_reward'],
                    box_bottom_height_proxy=objects[..., 2] - self._assigned_box_values(self._box_size)[..., 2] / 2,
                    target_xy_crossing=(((self._tar_pos - self._prev_box_pos)[..., :2] *
                                        (self._tar_pos - objects)[..., :2]).sum(-1) < 0).float(),
                    done=runtime.done.float(), active=live.float(),
                    first_success=result['first_success'].float(),
                    current_target_valid=result['valid_next'][:, 1::2].float(),
                    current_success_state=result['current_success_state'].float(),
                    saturation_active=result['saturation_active'].float(),
                    first_holding_seconds=self._relation_first_holding,
                    first_valid_seconds=self._relation_first_valid)
        for j, name in enumerate(('holding', 'at')):
            for field, value in (('phi', runtime.phi), ('gate', result['gate_next']),
                                 ('satisfied', result['satisfied_next']), ('achieved', runtime.achieved),
                                 ('state_reward', result['state_component']),
                                 ('raw_state_reward', result['raw_state_component']),
                                 ('raw_progress_reward', result['raw_progress_component']),
                                 ('progress_bar', result['pinned_progress']),
                                 ('progress_reward', result['progress_component'])):
                diag[name + '/' + field] = value[:, j::2].float()
            diag[name + '/threshold_up'] = ((~previous_satisfied[:, j::2]) &
                                           result['satisfied_next'][:, j::2]).float()
            diag[name + '/threshold_down'] = (previous_satisfied[:, j::2] &
                                             (~result['satisfied_next'][:, j::2])).float()
        progress_cfg = self._relation_cfg.get('progress', {})
        if 'approach_radius' in progress_cfg:
            diag['holding/approach'] = result['progress_blend'][:, 0::2]
            diag['at/approach'] = result['progress_blend'][:, 1::2]
        if self._relation_timeline.selected:
            timeline = dict(diag)
            timeline['placement_valid'] = placed.float()
            timeline['initially_placed'] = self._placement_metrics.initial.float()
            timeline['saturation_active'] = result['saturation_active'].float()
            for j, name in enumerate(('holding', 'at')):
                timeline[name + '/prerequisite_used'] = result['activation'][:, j::2]
                timeline[name + '/progress_blend_used'] = result['progress_blend'][:, j::2]
            if self._relation_timeline.path is None:
                self._relation_timeline.directory = os.path.join(getattr(
                    self, '_relation_output_directory', self.cfg['args'].output_path), 'diagnostics')
            self._relation_timeline.record(self._relation_steps, self._relation_episode_id,
                                           self.progress_buf, self.dt, timeline)
        stats = {k: v.mean() for k, v in diag.items() if not k.startswith('first_') or k == 'first_success'}
        self.extras['relation_near_unplaced_slow'] = (
            (diag['goal_xy_error'] < .5) & (speed < .05) & ~diag['put'].bool() & live).flatten()
        stats['scene_all_done'] = runtime.done.all(-1).float().mean()
        stats['scene_current_all_valid'] = result['valid_next'][:, 1::2].all(-1).float().mean()
        for name in ('first_holding_seconds', 'first_valid_seconds'):
            valid = diag[name] >= 0
            stats[name + '_observed_fraction'] = valid.float().mean()
            stats[name + '_observed_mean'] = (diag[name] * valid).sum() / valid.sum().clamp_min(1)
        # Distance bands are DIAGNOSTICS ONLY, never reward cutoffs or gates.
        for distance in (.1, .25, .5):
            mask = (diag['goal_xy_error'] < distance) & live
            den = mask.sum().clamp_min(1)
            prefix = 'near_{}m/'.format(distance)
            stats[prefix + 'fraction'] = mask.float().mean()
            stats[prefix + 'speed'] = (speed * mask).sum() / den
            stats[prefix + 'progress_reward'] = (diag['at/progress_reward'] * mask).sum() / den
            stats[prefix + 'stationary_unplaced_fraction'] = (
                mask & (speed < .05) & ~diag['put'].bool()).float().mean()
        for key, value in stats.items():
            if key not in self._relation_diagnostic_sums:
                self._relation_diagnostic_sums[key] = torch.zeros_like(value)
            self._relation_diagnostic_sums[key] += value.detach()
        self._relation_diagnostic_count += 1
        interval = int(dcfg.get('log_interval', 30))
        count = min(self.num_envs, int(dcfg.get('sample_envs', 2)))
        if interval > 0 and count > 0 and self._relation_steps % interval == 0:
            # Only a bounded subset crosses to CPU, in a single transfer.
            names = list(diag)
            rows = torch.stack([self._relation_episode_id[:count, None].expand(-1, self.num_agents),
                                self.progress_buf[:count, None].expand(-1, self.num_agents)] +
                               [diag[k][:count] for k in names], -1).detach().cpu().tolist()
            if self._relation_csv_path is None:
                directory = os.path.join(getattr(self, '_relation_output_directory',
                                                  self.cfg['args'].output_path), 'diagnostics')
                os.makedirs(directory, exist_ok=True)
                self._relation_csv_path = os.path.join(directory, 'relation_samples.csv')
            new = not os.path.exists(self._relation_csv_path)
            with open(self._relation_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(['global_step', 'env', 'agent', 'episode', 'episode_step'] + names)
                for env, agents in enumerate(rows):
                    for agent, row in enumerate(agents):
                        writer.writerow([self._relation_steps, env, agent] + row)

    def consume_relation_diagnostics(self):
        if not self._state_relation:
            return {}
        result = {k: v / self._relation_diagnostic_count for k, v in self._relation_diagnostic_sums.items()}
        if self._placement_metrics is not None:
            result.update(self._placement_metrics.consume())
        self._relation_timeline.flush()
        if getattr(self, '_ontop_mixed', False):
            result.update(self._consume_ontop_metrics())
        self._relation_diagnostic_sums = {}
        self._relation_diagnostic_count = 0
        return result
