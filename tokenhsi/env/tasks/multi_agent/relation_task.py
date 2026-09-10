"""Carry integration of the simulator-independent relation state machine."""
import csv
import os
import torch

from utils.relation_task_spec import compile_carry_subgoal
from env.tasks.multi_agent.relation_reward import (
    RelationRuntime, evaluate_holding, evaluate_at, velocity_progress, box_speed_penalty)


class CarryRelationMixin:
    def _init_relation_runtime(self):
        self.relation_runtime = RelationRuntime(self.num_envs,
            compile_carry_subgoal(self.num_agents, self.num_objects, self.device),
            self._relation_cfg, self.device)
        self._relation_diagnostic_sums = {}
        self._relation_diagnostic_count = 0
        self._relation_steps = 0
        self._relation_episode_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._relation_first_holding = torch.full((self.num_envs, self.num_agents), -1., device=self.device)
        self._relation_first_valid = torch.full_like(self._relation_first_holding, -1.)
        self._relation_csv_path = None

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
            a_cfg.get('near_distance_scale', 10.), a_cfg.get('near_fraction', .5),
            a_cfg.get('putdown_xy_tolerance', .1), a_cfg.get('putdown_z_tolerance', .001))
        return torch.stack([h, a], -1).flatten(1), dict(
            hand_midpoint_distance=hand_error,
            right_hand_center_distance=(hands[..., 0, :] - objects).norm(dim=-1),
            left_hand_center_distance=(hands[..., 1, :] - objects).norm(dim=-1),
            root_box_distance_xy=(bodies[..., 0, :2] - objects[..., :2]).norm(dim=-1),
            goal_xy_error=xy, goal_z_error=z, near=near, put=put.float())

    def _reset_relation_history(self, env_ids):
        phi, _ = self._evaluate_relations(env_ids)
        self.relation_runtime.reset(env_ids, phi)
        self._prev_root_pos[env_ids] = self._kinematic_humanoid_rigid_body_states[env_ids, :, 0, :3]
        self._prev_box_pos[env_ids] = self._assigned_box_values(self._box_states, env_ids)[..., :3]
        self._relation_episode_id[env_ids] += 1
        holding = self.relation_runtime.achieved[env_ids, 0::2]
        self._relation_first_holding[env_ids] = torch.where(holding, 0., -1.)
        self._relation_first_valid[env_ids] = torch.where(self.relation_runtime.done[env_ids], 0., -1.)

    @torch.no_grad()
    def _compute_relation_reward(self, collision_fn):
        runtime = self.relation_runtime
        phi, diag = self._evaluate_relations()
        objects = self._assigned_box_values(self._box_states)[..., :3]
        roots = self._humanoid_root_states[..., :3]
        cfg = self._relation_cfg.get('progress', {})
        args = (self.dt, cfg.get('target_speed', 1.5), cfg.get('velocity_scale', 5.),
                cfg.get('normalization_epsilon', 1e-6), cfg.get('mode', 'gaussian'))
        ph = velocity_progress(self._prev_root_pos, roots, objects, *args)
        pa = velocity_progress(self._prev_box_pos, objects, self._tar_pos, *args)
        previous_satisfied = runtime.phi >= self._relation_cfg.get('satisfaction_threshold', .9)
        live = ~runtime.done.clone()
        result = runtime.step(phi, torch.stack([ph, pa], -1).flatten(1))
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
            result['velocity_component'][:, 0::2], result['velocity_component'][:, 1::2],
            result['success_bonus'], power, collision, box_penalty, reward], -1).flatten(0, 1)
        self.rew_buf.copy_(reward.flatten())
        self.extras['reward_terms'] = terms
        self._reward_term_sums += terms.sum(0)
        self._reward_term_count += self.num_envs * self.num_agents
        self.extras['subgoal_done'] = runtime.done.clone()
        self.extras['all_subgoals_done'] = runtime.done.all(-1)
        self.extras['current_target_valid'] = result['valid_next'][:, 1::2]
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
        speed = ((objects - self._prev_box_pos) / self.dt).norm(dim=-1)
        diag.update(box_speed=speed, progress_holding=ph, progress_at=pa,
                    task_relation_total=result['agent_task_reward'],
                    success_bonus=result['success_bonus'],
                    box_bottom_height_proxy=objects[..., 2] - self._assigned_box_values(self._box_size)[..., 2] / 2,
                    target_xy_crossing=(((self._tar_pos - self._prev_box_pos)[..., :2] *
                                        (self._tar_pos - objects)[..., :2]).sum(-1) < 0).float(),
                    done=runtime.done.float(), active=live.float(),
                    first_success=result['first_success'].float(),
                    current_target_valid=result['valid_next'][:, 1::2].float(),
                    first_holding_seconds=self._relation_first_holding,
                    first_valid_seconds=self._relation_first_valid)
        for j, name in enumerate(('holding', 'at')):
            for field, value in (('phi', runtime.phi), ('gate', result['gate_next']),
                                 ('satisfied', result['satisfied_next']), ('achieved', runtime.achieved),
                                 ('state_delta', result['state_component']),
                                 ('velocity_reward', result['velocity_component'])):
                diag[name + '/' + field] = value[:, j::2].float()
            diag[name + '/threshold_up'] = ((~previous_satisfied[:, j::2]) &
                                           result['satisfied_next'][:, j::2]).float()
            diag[name + '/threshold_down'] = (previous_satisfied[:, j::2] &
                                             (~result['satisfied_next'][:, j::2])).float()
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
            stats[prefix + 'velocity_reward'] = (diag['at/velocity_reward'] * mask).sum() / den
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
        if not self._state_relation or not self._relation_diagnostic_count:
            return {}
        result = {k: v / self._relation_diagnostic_count for k, v in self._relation_diagnostic_sums.items()}
        self._relation_diagnostic_sums = {}
        self._relation_diagnostic_count = 0
        return result
