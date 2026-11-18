"""Sensor training with a fixed-radius waypoint, inactive on apron BEFORE pairs."""
import os
import torch
from env.tasks.multi_agent.humanoid_ma_sensor_stage2 import HumanoidMASensorStage2
from utils.warehouse_waypoint import (
    BOX_ONLY_VERSION, OBS_WIDTH, ROLE_OBS_WIDTH, advance_box_only_phase, advance_phase, door_box,
    role_eligibility, role_holding,
)
from utils.edge_context_spec import HOLDING
from utils.sensor_stage2_spec import WAREHOUSE


class WarehouseWaypointTrainingTask(HumanoidMASensorStage2):
    def __init__(self, cfg, *args, **kwargs):
        spec = cfg['env']['sensorScene']['warehouse']['waypoint']
        self._waypoint_box_only = spec['version'] == BOX_ONLY_VERSION
        self._waypoint_obs_width = OBS_WIDTH if spec['version'] == 1 else ROLE_OBS_WIDTH
        super().__init__(cfg, *args, **kwargs)

    def get_relation_suffix_size(self):
        return super().get_relation_suffix_size()+self._waypoint_obs_width

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        shape = (self.num_envs, 2) if self._waypoint_obs_width == ROLE_OBS_WIDTH else (self.num_envs,)
        self._waypoint_phase = torch.zeros(shape, dtype=torch.long, device=self.device)
        if self._waypoint_box_only:
            self._waypoint_door_pickup = torch.zeros_like(self._warehouse_door_passed)
        self._waypoint_baseline = os.environ.get('WAYPOINT_RESET_BASELINE') == '1'
        if not self.cfg['args'].eval and not self._waypoint_baseline:
            from env.tasks.multi_agent.warehouse_waypoint_cache import WaypointLayoutCache
            self._warehouse_before_cache = WaypointLayoutCache(self, self._warehouse['waypoint']['cache'])

    def _reset_envs(self, env_ids):
        self._waypoint_phase[env_ids] = 0
        if self._waypoint_box_only:
            self._waypoint_door_pickup[env_ids] = False
        return super()._reset_envs(env_ids)

    def _compute_clean_scene_obs(self, env_ids=None):
        base = super()._compute_clean_scene_obs(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if self._waypoint_obs_width == ROLE_OBS_WIDTH:
            return torch.cat((base, self._role_waypoint_metadata(ids)), -1)
        eligible = (self._scene_ids[ids] == WAREHOUSE) & ~self._warehouse_before_pairs[ids]
        pre = self._before_predecessor[ids]
        follower = (pre >= 0).long().argmax(-1)
        leader = pre.gather(1, follower[:, None]).squeeze(1).clamp_min(0)
        payload = self._agent_box_assignment[ids, leader]
        sizes = self._box_size[ids, payload]
        door = door_box(self._env_origins[ids], self._structure_shift[ids, WAREHOUSE],
                        self._structure_yaw[ids, WAREHOUSE], self._surface_half[ids, WAREHOUSE, 0],
                        self._warehouse['wall_thickness'], self._warehouse['floor_height'], sizes)
        graph = self.relation_runtime.graph
        holding = ((graph.edge_valid[ids] & (graph.edge_relation[ids] == HOLDING)
                    & (graph.edge_owner[ids] == leader[:, None])) & self.relation_runtime.own_success[ids]).any(-1)
        phase = advance_phase(self._waypoint_phase[ids], self._humanoid_root_states[ids, leader, :2],
                              self._box_states[ids, payload, :2], door[:, :2], holding,
                              self._warehouse['waypoint']['switch_radius'])
        phase = torch.where(eligible, phase, -1)
        self._waypoint_phase[ids] = phase
        metadata = torch.cat((torch.where(eligible, leader, -1)[:, None].float(),
                              phase[:, None].float(), door-self._env_origins[ids]), -1)
        metadata[~eligible, 2:] = 0
        return torch.cat((base, metadata), -1)

    def _role_waypoint_metadata(self, ids):
        people, eligible = role_eligibility(self._scene_ids[ids], self._warehouse_before_pairs[ids],
                                            self._before_predecessor[ids], self._templates[ids])
        payloads = self._agent_box_assignment[ids[:, None], people]
        sizes = self._box_size[ids, payloads[:, 0]]
        door = door_box(self._env_origins[ids], self._structure_shift[ids, WAREHOUSE],
                        self._structure_yaw[ids, WAREHOUSE], self._surface_half[ids, WAREHOUSE, 0],
                        self._warehouse['wall_thickness'], self._warehouse['floor_height'], sizes)
        holding = role_holding(self.relation_runtime, ids, people)
        roots = self._humanoid_root_states[ids[:, None], people, :2]
        radius = self._warehouse['waypoint']['switch_radius']
        if self._waypoint_box_only:
            phase = advance_box_only_phase(self._waypoint_phase[ids], roots, door[:, None, :2], holding, radius)
        else:
            phase = advance_phase(self._waypoint_phase[ids], roots, self._box_states[ids[:, None], payloads, :2],
                                  door[:, None, :2], holding, radius)
        phase = phase.masked_fill(~eligible, -1)
        self._waypoint_phase[ids] = phase
        roles = torch.stack((people.masked_fill(~eligible, -1), phase), -1).to(door.dtype)
        door_local = (door-self._env_origins[ids]).masked_fill(~eligible[:, :1], 0.)
        return torch.cat((roles[:, 0], door_local, roles[:, 1]), -1)

    def _place_scene(self, ids, room_retry=False):
        if not self._waypoint_baseline:
            return super()._place_scene(ids, room_retry)
        # Diagnostic reference: the original viewer's two-stage placement.
        spec = self._warehouse.pop('waypoint')
        try:
            super()._place_scene(ids, room_retry)
        finally:
            self._warehouse['waypoint'] = spec
        chosen = ids[~self._warehouse_before_pairs[ids]]
        if len(chosen):
            from env.tasks.multi_agent.warehouse_waypoint_placement import finish_legacy_placement
            finish_legacy_placement(self, chosen)

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        stages = ((0, 'inbound'), (1, 'pickup'), (2, 'outbound'), (3, 'final'))
        if self._waypoint_box_only:
            stages = ((0, 'inbound'), (1, 'pickup'), (3, 'held_once'))
        for phase, name in stages:
            current = (self._waypoint_phase == phase).float().mean(0)
            if self._waypoint_obs_width == ROLE_OBS_WIDTH:
                result['warehouse/waypoint/leader/'+name] = current[0]
                result['warehouse/waypoint/follower/'+name] = current[1]
            else:
                result['warehouse/waypoint/'+name] = current
        return result

    def _draw_ontop_context(self):
        super()._draw_ontop_context()
        if self.viewer is None or bool(self._warehouse_before_pairs[0]):
            return
        from qualitative.warehouse_waypoint_viewer import draw, camera
        metadata = self.obs_buf[:1, -self._waypoint_obs_width:]
        leader = metadata[:, 0].long()
        if bool((leader < 0).any()):
            return
        virtual = metadata[:, 2:5]+self._env_origins[:1]
        sizes = self._box_size[:1, self._agent_box_assignment[:1, leader[0]]].reshape(1, 3)
        phase = metadata[:, 1]
        radius = virtual.new_full((1,), self._warehouse['waypoint']['switch_radius'])
        ontop = None
        if self._waypoint_obs_width == ROLE_OBS_WIDTH and metadata[0, 5] >= 0:
            follower, follower_phase = metadata[:, 5].long(), metadata[:, 6]
            ontop = (follower, (follower_phase == 0) | (follower_phase == 2), follower_phase == 2)
        draw(self, virtual, sizes, radius, phase != 0, phase == 2, leader, ontop)
        if int(self.progress_buf[0]) <= 1:
            camera(self, virtual)
