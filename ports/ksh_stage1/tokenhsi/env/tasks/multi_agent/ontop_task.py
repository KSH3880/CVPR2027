"""Simulator integration helpers for the three fixed mixed-task environment groups."""
import torch

from utils.ontop_task_spec import SCENARIOS, mixed_graph, terminal_geometry


class OnTopTaskMixin:
    def _init_ontop_runtime(self):
        self._ontop_scenario = torch.tensor(self._ontop_scenario_list, device=self.device)
        self._ontop_base_agent = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.relation_runtime.graph = mixed_graph(self._ontop_scenario, self._ontop_base_agent)
        self._ontop_ever_joint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ontop_joint_steps = torch.zeros(self.num_envs, device=self.device)
        self._ontop_after_joint_steps = torch.zeros_like(self._ontop_joint_steps)
        self._ontop_first_distance = torch.full_like(self._ontop_joint_steps, -1.)
        self._ontop_current = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._ontop_metric_sums = {}
        print('[OnTop] scenarios:', {name: self._ontop_scenario_list.count(i)
                                    for i, name in enumerate(SCENARIOS)}, flush=True)

    def _ontop_geometry(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        states = self._gather_entity_slots(self._box_states[ids], self._logical_box_order[ids])
        sizes = self._gather_entity_slots(self._box_size[ids], self._logical_box_order[ids])
        return terminal_geometry(states, sizes, self._tar_pos[ids],
                                 self._ontop_scenario[ids], self._ontop_base_agent[ids])

    def _reset_ontop_roles(self, env_ids):
        self._ontop_base_agent[env_ids] = torch.randint(2, (len(env_ids),), device=self.device)
        self.relation_runtime.graph = mixed_graph(self._ontop_scenario, self._ontop_base_agent)

    def _reset_ontop_targets(self, env_ids):
        active = env_ids[self._ontop_scenario[env_ids] != 0]
        b = 1 - self._ontop_base_agent[active]
        # The inactive goal is masked by both policy encoders. Remove its physical
        # platform as well, so it cannot become an unintended shelf for stacking.
        if self._reset_random_height:
            self._tar_platform_pos[active, b, 2] = self._platform_inactive_height
        self._tar_pos[active, b] = self._env_origins[active]
        self._tar_pos[active, b, 2] = -10.
        self._ontop_ever_joint[env_ids] = False
        self._ontop_joint_steps[env_ids] = 0
        self._ontop_after_joint_steps[env_ids] = 0
        self._ontop_first_distance[env_ids] = -1
        self._ontop_current[env_ids] = False

    def _ontop_reset_rejected(self, env_ids):
        source, target, _, is_top = self._ontop_geometry(env_ids)
        # Exclude already-stacked starts independently of A's goal/history.
        delta = source - target
        stacked = (torch.exp(-10 * delta.square().sum(-1)) >= .9) & (delta[..., 2].abs() <= .001)
        return (stacked & is_top).any(-1)

    def _ontop_step_metrics(self, result):
        current = result['current_success_state']
        self._ontop_current.copy_(current)
        joint = current.all(-1)
        self._ontop_ever_joint |= joint
        self._ontop_joint_steps += joint.float()
        self._ontop_after_joint_steps += self._ontop_ever_joint.float()
        row = torch.arange(self.num_envs, device=self.device)
        first_b = (self._ontop_scenario == 2) & current[row, 1 - self._ontop_base_agent] & (self._ontop_first_distance < 0)
        objects = self._assigned_box_values(self._box_states)[..., :3]
        distance = (objects - self._tar_pos).norm(dim=-1)[row, self._ontop_base_agent]
        self._ontop_first_distance[first_b] = distance[first_b]
        for i, name in enumerate(SCENARIOS):
            mask = self._ontop_scenario == i
            count = mask.sum().clamp_min(1)
            for key, value in (('current_joint_success', joint.float()),
                               ('task_reward', result['agent_task_reward'].mean(-1)),
                               ('base_agent0_fraction', (self._ontop_base_agent == 0).float()),
                               ('a_current_success', current[row, self._ontop_base_agent].float()),
                               ('b_current_success', current[row, 1 - self._ontop_base_agent].float()),
                               ('b_terminal_gate', result['activation'][row, 2 * (1 - self._ontop_base_agent) + 1]),
                               ('b_terminal_phi', self.relation_runtime.phi[row, 2 * (1 - self._ontop_base_agent) + 1])):
                self._ontop_accumulate(name + '/' + key, (value * mask).sum(), count)

    def _ontop_accumulate(self, key, total, count):
        if key not in self._ontop_metric_sums:
            self._ontop_metric_sums[key] = [torch.zeros((), device=self.device), torch.zeros((), device=self.device)]
        self._ontop_metric_sums[key][0] += total.detach()
        self._ontop_metric_sums[key][1] += count.detach()

    def _finish_ontop_metrics(self):
        for i, name in enumerate(SCENARIOS):
            ending = self.reset_buf.bool() & (self._ontop_scenario == i)
            for key, value in (('final_joint_success', self._ontop_current.all(-1).float()),
                               ('ever_joint_success', self._ontop_ever_joint.float())):
                self._ontop_accumulate(name + '/' + key, (value * ending).sum(), ending.sum())
            self._ontop_accumulate(name + '/joint_retention_after_first',
                                   (self._ontop_joint_steps * ending).sum(),
                                   (self._ontop_after_joint_steps * ending).sum())
            valid = ending & (self._ontop_first_distance >= 0)
            if i == 2:
                self._ontop_accumulate(name + '/first_b_success_oa_ga_distance',
                                       (self._ontop_first_distance.clamp_min(0) * valid).sum(), valid.sum())
                self._ontop_accumulate(name + '/b_success_episode_fraction', valid.sum(), ending.sum())

    def _consume_ontop_metrics(self):
        result = {}
        for key, (total, count) in self._ontop_metric_sums.items():
            result['scenario/' + key] = total / count.clamp_min(1)
            if 'first_b_success' in key or 'final_joint' in key:
                result['scenario/' + key + '_samples'] = count.clone()
        self._ontop_metric_sums = {}
        return result
