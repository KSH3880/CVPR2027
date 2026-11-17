import os

import torch

from env.tasks.multi_agent.cliff_bridge_controller import CliffBridgeController
from env.tasks.multi_agent.cliff_bridge_metrics import CliffBridgeMetrics
from env.tasks.multi_agent.cliff_bridge_placement import CliffBridgePlacement
from env.tasks.multi_agent.humanoid_ma_sensor_stage2 import HumanoidMASensorStage2
from utils.bridge_curriculum import span_probabilities
from utils.cliff_bridge_spec import BRANCHES, CONTROLLED, COOPERATIVE, FREE, sample_branches, sample_roles_for_scene
from utils.sensor_stage2_spec import BRIDGE, sample_bridge_starts, sample_surfaces


class HumanoidMASensorCliffBridge(CliffBridgeController, CliffBridgePlacement,
                                 CliffBridgeMetrics, HumanoidMASensorStage2):
    def __init__(self, cfg, *args, **kwargs):
        self._branch_mode = os.environ.get('BRIDGE_MODE', 'training')
        if self._branch_mode not in ('training', *BRANCHES):
            raise ValueError('BRIDGE_MODE must be training, controlled, free or cooperative')
        if os.environ.get('BRIDGE_MODE') and not cfg['args'].eval:
            raise ValueError('BRIDGE_MODE is evaluation-only')
        self._start_override = os.environ.get('BRIDGE_START', 'training')
        if self._start_override not in ('training', 'present', 'near', 'insert'):
            raise ValueError('BRIDGE_START must be training, present, near or insert')
        if os.environ.get('BRIDGE_START') and not cfg['args'].eval:
            raise ValueError('BRIDGE_START is evaluation-only')
        self._pusher_override = os.environ.get('BRIDGE_PUSHERS', '')
        if self._pusher_override not in ('', '1', '2', '3'):
            raise ValueError('BRIDGE_PUSHERS must be 1, 2 or 3')
        if self._pusher_override and not cfg['args'].eval:
            raise ValueError('BRIDGE_PUSHERS is evaluation-only')
        if os.environ.get('SCENARIO') not in (None, 'cliff', 'bridge'):
            raise ValueError('This variant supports only SCENARIO=cliff or bridge')
        if os.environ.get('BRIDGE_CURRICULUM_EVAL'):
            raise ValueError('Use BRIDGE_MODE for this variant')
        super().__init__(cfg, *args, **kwargs)

    def _init_bridge_curriculum(self):
        super()._init_bridge_curriculum()
        self._bridge_branch = torch.full((self.num_envs,), COOPERATIVE, device=self.device, dtype=torch.long)
        self._cross_stats = torch.zeros(3, 3, 3, 8, device=self.device)
        self._cross_scene_stats = torch.zeros(3, 2, device=self.device)
        self._cross_role_stats = torch.zeros(3, 8, device=self.device)
        self._cliff_exited = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        self._cliff_dropped = torch.zeros_like(self._cliff_exited)
        self._cliff_stats = torch.zeros(9, device=self.device)
        self._branch_samples = torch.zeros(3, 5, device=self.device)
        self._init_controller()

    def _sample_bridge_curriculum(self, ids, counts):
        if self._pusher_override:
            counts = torch.full_like(counts, int(self._pusher_override))
        branches = sample_branches(len(ids), self._scene_cfg['bridge']['branches'], self.device,
            self._branch_mode if self.cfg['args'].eval else 'training')
        bridge = self._scene_ids[ids] == BRIDGE
        self._bridge_branch[ids] = branches
        practice = bridge & (branches != COOPERATIVE)
        spans = counts.clone()
        update = 3000 if self.cfg['args'].eval else self._bridge_curriculum_update
        probabilities = torch.tensor(span_probabilities(self._bridge_curriculum, update), device=self.device)
        if practice.any():
            spans[practice] = torch.multinomial(probabilities, int(practice.sum()), True)+1
        self._bridge_span[ids], self._bridge_walk[ids] = spans, practice
        self._cross_entered[ids], self._cross_completed[ids] = False, False
        self._cliff_exited[ids], self._cliff_dropped[ids] = False, False
        return torch.where(practice, 1, counts), spans

    def _draw_episode_graph(self, ids):
        env = self.cfg['env']
        sizes = self._box_size[ids, self._leader_bank:self._leader_bank+4]
        domain, groups, pre, tasks, roles, push = sample_roles_for_scene(sizes,
            env['box']['build']['taskSizeRanges'], self._relation_graph_spec['template_probabilities'],
            self._scene_ids[ids], self._bridge_branch[ids], self._bridge_count[ids])
        self._roles[ids], self._bridge_pushers[ids] = roles, push
        controlled = (self._scene_ids[ids] == BRIDGE) & (self._bridge_branch[ids] == CONTROLLED)
        es = ids[controlled]
        self._bridge_slot[es] = self._controlled_bank+self._bridge_span[es]-1
        self._controlled_forces[self._controlled_bodies[ids].flatten()] = 0.
        self._surface[ids], self._start_surface[ids], self._unreachable[ids] = sample_surfaces(
            tasks, pre, roles, self._scene_ids[ids], push)
        self._bridge_start_mode[ids] = sample_bridge_starts(self._bridge_training_init, len(ids), self.device)
        if self._start_override != 'training':
            self._bridge_start_mode[ids] = ('present', 'near', 'insert').index(self._start_override)
        spec = self._scene_cfg['bridge']['controller']
        self._controlled_present[ids] = torch.rand(len(ids), device=self.device) < spec['present_probability']
        lo, hi = spec['arrival_time_range']
        self._controlled_arrival[ids] = lo*(hi/lo)**torch.rand(len(ids), device=self.device)
        free = ids[(self._scene_ids[ids] == BRIDGE) & (self._bridge_branch[ids] == FREE)]
        self._bridge_start_mode[free] = 0
        self._bridge_start_mode[es] = (~self._controlled_present[es]).long()
        self._bridge_near[ids] = self._bridge_start_mode[ids] == 1
        self._bridge_aligned[ids] = (self._bridge_start_mode[ids] == 0) | (self._bridge_near[ids] &
            (torch.rand(len(ids), device=self.device) < .5))
        self._bridge_side[ids] = True
        carry = tasks.gather(1, roles[:, 1:2]).flatten()
        cooperative = (self._scene_ids[ids] == BRIDGE) & (self._bridge_branch[ids] == COOPERATIVE)
        self._bridge_carry_from_start[ids] = cooperative & ((carry == 3) | (carry == 4)) & (
            torch.rand(len(ids), device=self.device) < .5)
        self._curriculum_surfaces(ids)
        return domain, groups, pre, tasks

    def _curriculum_surfaces(self, ids):
        self._crosser_mask[ids] = False
        self._cross_phase[ids] = -1
        self._cross_progress[ids] = 0.
        bridge = ids[self._scene_ids[ids] == BRIDGE]
        cross = self._roles[bridge, 1]
        self._crosser_mask[bridge, cross] = True
        self._cross_phase[bridge, cross] = 0
        practice = bridge[self._bridge_branch[bridge] != COOPERATIVE]
        if not len(practice):
            return
        active = ~self._bridge_pushers[practice]
        self._crosser_mask[practice] = active
        self._cross_phase[practice] = torch.where(active, 0, -1)
        start = torch.randint(1, 3, (len(practice), 1), device=self.device)
        self._start_surface[practice] = torch.where(active, start, 0)
        self._surface[practice] = torch.where(active, 3-start, 0)
        free = practice[self._bridge_branch[practice] == FREE]
        person = self._roles[free, 3]
        phase = torch.randint(1, 3, (len(free),), device=self.device)
        self._cross_phase[free, person] = phase
        self._cross_progress[free, person] = torch.where(phase == 1, .25, .65)+.2*torch.rand(len(free), device=self.device)

    def _sample_episode_graph(self, ids):
        super()._sample_episode_graph(ids)
        controlled = ids[(self._scene_ids[ids] == BRIDGE) & (self._bridge_branch[ids] == CONTROLLED)]
        free = ~self._stage2_task_slots[controlled, 4:]
        if len(controlled):
            if not free.any(-1).all():
                raise ValueError('Controlled bridge requires an unused object token')
            slot = 4+free.long().argmax(-1)
            self._logical_box_order[controlled, slot] = self._bridge_slot[controlled]
        for branch in range(3):
            selected = ids[(self._scene_ids[ids] == BRIDGE) & (self._bridge_branch[ids] == branch)]
            before = self.relation_runtime.before[selected]
            coupled = (self._templates[selected] == 5) & (self._members[selected] > 1)
            self._branch_samples[branch] += torch.stack((torch.tensor(len(selected), device=self.device),
                (self._before_predecessor[selected] >= 0).sum(), before.sum(),
                self._crosser_mask[selected].sum(), coupled.sum())).float()
