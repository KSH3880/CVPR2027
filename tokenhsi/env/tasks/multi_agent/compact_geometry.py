from isaacgym import gymapi, gymtorch
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from utils.compact_bridge import visitors


class CompactGeometry:
    def _surface_halves(self, cfg, shape):
        if 'stage_size_by_people' in cfg:
            bounds = cfg['stage_size_by_people']
            cfg = dict(size_range=[min(row[0] for row in bounds),max(row[1] for row in bounds)],
                size_step=cfg['size_step'])
        return super()._surface_halves(cfg,shape)

    def _build_scene_geometry(self):
        super()._build_scene_geometry()
        if not hasattr(self, '_compact_halves'):
            cfg = self._scene_cfg['bridge']
            self._compact_halves = torch.stack([
                torch.stack([self._surface_halves(dict(size_range=bounds, size_step=.5),
                    (self.num_envs, 2)) for bounds in cfg['stage_size_by_people']], 1)
                for _ in range(2)], 1)
            self._compact_actor_records = []

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        HumanoidMAStage2._build_env(self, env_id, env_ptr, humanoid_asset)
        origin = self.gym.get_env_origin(env_ptr)
        height = float(self._surface_height[env_id, 1])
        shapes = [self._static_half[env_id, 0, 0, :2]]
        shapes += list(self._compact_halves[env_id].reshape(8, 2))
        for slot, half in enumerate(shapes):
            size = (*map(float, 2*half), height)
            if size not in self._static_assets:
                self._static_assets[size] = self.gym.create_box(self.sim, *size, self._static_options)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(origin.x, origin.y, origin.z-1000.)
            actor = self.gym.create_actor(env_ptr, self._static_assets[size], pose,
                'compact_stage_'+str(slot), env_id, 0)
            self.gym.set_rigid_body_color(env_ptr, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.32,.36,.42))
            index = self.gym.get_actor_index(env_ptr, actor, gymapi.DOMAIN_SIM)
            self._compact_actor_records.append(index)

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        self._compact_actor_ids = torch.tensor(self._compact_actor_records,
            device=self.device, dtype=torch.long).view(self.num_envs, 9)
        self._compact_active = torch.zeros(self.num_envs, 9, device=self.device, dtype=torch.bool)
        self._compact_visitors = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.long)
        self._compact_sizes = torch.zeros(self.num_envs, 2, 2, device=self.device)

    def _activate_compact_geometry(self, ids):
        n = len(ids)
        count = visitors(self._start_surface[ids], self._surface[ids], self._bridge_pushers[ids])
        self._compact_visitors[ids] = count
        index = (count-1).clamp(0,3)
        halves = self._compact_halves[ids[:, None], torch.arange(2, device=self.device), index]
        self._compact_sizes[ids] = 2*halves
        bridge_size = self._box_size[ids, self._bridge_slot[ids]]
        far, gap, bounds = scene.bridge_stages(halves[:,0], halves[:,1], bridge_size,
            self._scene_cfg['bridge'], align_entry=True)
        self._bridge_half_a[ids], self._bridge_half_b[ids] = halves[:,0], halves[:,1]
        self._bridge_far[ids], self._bridge_gap[ids] = far, gap
        self._surface_center[ids,1,0] = 0.
        self._surface_center[ids,1,1] = far
        self._surface_half[ids,1] = halves
        self._static_half[ids,1,:2,:2] = halves
        self._structure_bounds[ids,1] = bounds
        placed = scene.place_structure(bounds, self._scene_cfg['floor_clearance'],
            self._scene_cfg['translation_range'])
        for field in ('yaw','shift','axis','direction','near'):
            getattr(self, '_structure_'+field)[ids,1] = placed[field]
        yaw, shift = placed['yaw'], placed['shift']
        local = torch.stack((torch.zeros_like(far), far), 1)
        self._static_local[ids,1,:2,:2] = scene.rotate_xy(local, yaw[:,None])+shift[:,None]
        self._static_quat[ids,1] = scene.yaw_quat(yaw)[:,None]
        self._scene_origins[ids,1] = self._base_scene_origins[ids,1]
        self._static_centers[ids,1] = self._static_local[ids,1]+self._scene_origins[ids,1,None]
        self._env_origins[ids] = self._scene_origins[ids,self._scene_ids[ids]]
        self._compact_active[ids] = False
        cliff = self._scene_ids[ids] == 0
        self._compact_active[ids[cliff],0] = True
        rows = ids[~cliff]
        self._compact_active[rows,1+index[~cliff,0]] = True
        self._compact_active[rows,5+index[~cliff,1]] = True

    def _reset_env_tensors(self, ids):
        super()._reset_env_tensors(ids)
        actors = self._compact_actor_ids[ids]
        states = self._root_states[actors].clone()
        states[:] = 0.
        states[...,2] = self._env_origins[ids,2,None]-1000.
        states[...,6] = 1.
        centers = self._static_centers[ids]
        quats = self._static_quat[ids]
        for slot in range(9):
            selected = self._compact_active[ids,slot]
            surface, item = (0,0) if slot == 0 else (1,int(slot >= 5))
            states[selected,slot,:3] = centers[selected,surface,item]
            states[selected,slot,3:7] = quats[selected,surface,item]
        self._root_states[actors] = states
        # Commit every reset actor together: the GPU pipeline stages the last indexed root update.
        count = self.get_num_actors_per_env()
        flat = (ids[:,None]*count+torch.arange(count,device=self.device)).flatten().to(torch.int32).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(flat), len(flat))
