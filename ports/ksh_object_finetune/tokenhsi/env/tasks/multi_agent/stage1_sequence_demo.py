import os

import torch
from isaacgym import gymtorch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
from utils.edge_ontop_spec import copy_graph_rows
from utils.stage1_sequence import (SequenceProgress, sample_sequence, sequence_graph,
                                  parse_task_sequences, sample_full_sequence, sequence_box_sizes)
from utils.task_role_spec import TASKS, TASK_DISTILL_VARIANT


class Stage1SequenceDemo(HumanoidMACarry):
    def __init__(self, cfg, *args, **kwargs):
        env = cfg['env']
        if not cfg['args'].test or not cfg['args'].eval:
            raise ValueError('Stage 1 sequence demo is evaluation only')
        if (env['numAgents'], env['numObjects'], env['numEnvs']) != (2, 4, 1):
            raise ValueError('Sequence viewer requires 2 agents, 4 objects, 1 environment')
        if env['relationReward']['stage1_variant'] != TASK_DISTILL_VARIANT:
            raise ValueError('Sequence demo requires the four-task distill config')
        self.sequence_sampling = os.environ.get('GRAPH_SAMPLING', 'full_random')
        self.sequence_fixed = parse_task_sequences(os.environ.get('TASK_SEQUENCE_A', ''),
                                                   os.environ.get('TASK_SEQUENCE_B', ''))
        if self.sequence_sampling not in ('full_random', 'size_conditioned'):
            raise ValueError('GRAPH_SAMPLING must be full_random or size_conditioned')
        if self.sequence_sampling == 'size_conditioned' and self.sequence_fixed.any():
            raise ValueError('Fixed sequences require GRAPH_SAMPLING=full_random')
        super().__init__(cfg, *args, **kwargs)
        self.sequence_fixed = self.sequence_fixed.to(self.device)
        self.sequence_mode = os.environ.get('SEQUENCE_MODE', 'independent')
        self.sequence_seconds = float(os.environ.get('SUCCESS_HOLD_SECONDS', '1.0'))
        self.sequence_seed = cfg['args'].seed
        self.sequence_repeat = 0
        self.sequence = SequenceProgress(self.sequence_mode, self.sequence_seconds, self.dt, self.device)
        self.sequence_events = []

    def _sample_size_rsi_boxes(self, preset):
        if self.sequence_sampling == 'size_conditioned':
            return super()._sample_size_rsi_boxes(preset)
        return sequence_box_sizes(self.sequence_fixed, self.device)

    def _reset_ontop_context_envs(self, ids):
        if not len(ids):
            return
        self.sequence = SequenceProgress(self.sequence_mode, self.sequence_seconds, self.dt, self.device)
        self.sequence_events = []
        self.sequence_generator = torch.Generator(device=self.device)
        self.sequence_generator.manual_seed(self.sequence_seed + self.sequence_repeat)
        if self.sequence_sampling == 'size_conditioned':
            self.sequence_templates, self.sequence_targets = sample_sequence(
                self._box_size[0], self.sequence_generator)
        else:
            self.sequence_templates, self.sequence_targets = sample_full_sequence(
                self.sequence_fixed, self.sequence_generator)
        super()._reset_ontop_context_envs(ids)
        self.sequence_goals = self._next_goals()
        self.sequence_repeat += 1
        plan = [[dict(task=TASKS[int(self.sequence_templates[stage, agent]) - 1],
                      object=int(self.sequence_targets[stage, agent]))
                 for stage in range(2)] for agent in range(2)]
        self.sequence_plan = plan
        print('[sequence] sampling={} mode={} hold={:.2f}s horizon={} sizes={} plan={}'.format(
            self.sequence_sampling, self.sequence_mode, self.sequence_seconds, self.max_episode_length,
            self._box_size[0].tolist(), plan), flush=True)

    def _sample_episode_graph(self, ids):
        agents = torch.arange(2, device=self.device)
        stages = self.sequence.stage
        graph = sequence_graph(self.sequence_templates[stages, agents],
                               self.sequence_targets[stages, agents])
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._edge_goal_owners[ids] = True

    def _next_goals(self):
        goals = self._tar_pos[0].clone()
        boxes = self._box_states[0]
        for agent in range(2):
            if self.sequence_templates[1, agent] != 3:
                continue
            center = self._agent_spawn_offsets[agent, :2] + self._env_origins[0, :2]
            candidates = (torch.rand(256, 2, device=self.device, generator=self.sequence_generator) * 2 - 1) * \
                self._tar_reach + center
            clearance = self._box_size[0, :, :2].norm(dim=-1) / 2 + \
                self._box_size[0, agent, :2].norm() / 2 + .15
            valid = (torch.cdist(candidates, boxes[:, :2]) > clearance).all(-1)
            valid &= (torch.cdist(candidates, goals[:, :2]) > .8).all(-1)
            if not valid.any():
                raise RuntimeError('Could not find a clear second AT goal')
            goals[agent, :2] = candidates[valid][0]
            goals[agent, 2] = vertical_extent(boxes[agent, 3:7], self._box_size[0, agent] / 2) + \
                self._env_origins[0, 2]
        return goals

    def _advance_sequence(self, advance):
        runtime = self.relation_runtime
        graph = runtime.graph
        old_keys = torch.stack((graph.edge_owner, graph.edge_src, graph.edge_dst,
                                graph.edge_relation, graph.edge_valid), -1).clone()
        old_achieved = runtime.achieved.clone()
        old_ever = self._edge_ever_goal.clone()
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._sample_episode_graph(ids)
        agents = advance.nonzero().flatten()
        at_agents = agents[self.sequence_templates[1, agents] == 3]
        self._tar_pos[0, at_agents] = self.sequence_goals[at_agents]
        phi, diag = self._evaluate_ontop_context()
        runtime.reset(ids, phi, diag['z_error'], diag['feet_height_error'], diag['region_error'])
        keys = torch.stack((graph.edge_owner, graph.edge_src, graph.edge_dst,
                            graph.edge_relation, graph.edge_valid), -1)
        matching = (keys[:, :, None] == old_keys[:, None]).all(-1)
        unchanged = ~advance[graph.edge_owner] & graph.edge_valid
        preserved = (matching & old_achieved[:, None]).any(-1)
        runtime.achieved[unchanged] = preserved[unchanged]
        self._edge_ever_goal[:] = old_ever
        self._edge_ever_goal[0, advance] = runtime.done[0, advance]
        self._prev_box_pos[0, advance] = self._reward_box_values(self._box_states)[0, advance, :3]
        self._compute_observations()
        self._update_box_assignment_colors(ids)
        if self._enable_markers:
            self._marker_pos[0] = self._tar_pos[0]
            actor_ids = self._marker_actor_ids.flatten().contiguous()
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        self.extras['policy_obs'] = self.obs_buf.clone()

    def post_physics_step(self):
        super().post_physics_step()
        if self.reset_buf.any() or self.sequence.finished():
            return
        agents = torch.arange(2, device=self.device)
        stages = self.sequence.stage.clone()
        templates = self.sequence_templates[stages, agents]
        boxes = self._box_states[0, :2]
        settled = (boxes[:, 7:10].norm(dim=-1) < .15) & (boxes[:, 10:13].norm(dim=-1) < .5)
        success = self.relation_runtime.done[0] & ((templates < 3) | settled)
        newly, advance = self.sequence.step(success)
        for agent in newly.nonzero().flatten().tolist():
            event = dict(step=int(self.progress_buf[0]), seconds=float(self.progress_buf[0] * self.dt),
                         agent=agent, stage=int(stages[agent]) + 1,
                         task=TASKS[int(templates[agent]) - 1])
            self.sequence_events.append(event)
            print('[sequence success]', event, flush=True)
        if advance.any():
            self._advance_sequence(advance)
            print('[sequence advance] agents={} stages={} (scene preserved)'.format(
                advance.nonzero().flatten().tolist(), (self.sequence.stage + 1).tolist()), flush=True)
