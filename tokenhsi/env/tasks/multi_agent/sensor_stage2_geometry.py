from isaacgym import gymapi
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.sensor_stage2_spec import BRIDGE, CLIFF, ROOM, WAREHOUSE, SCENES

ROOM_ACTORS = 6
FLOOR_REACH = 20.  # Stage2 floor tasks stay within about 13m of the centre, plus the shift off the structure.


class SensorStage2Geometry:

    def _load_box_asset(self):
        super()._load_box_asset()
        n, m = self.num_envs, self.num_agents
        base = self._box_size.shape[1]
        self._leader_bank = base
        self._bridge_bank = base+m
        old_assets = [self._box_assets[i*base:(i+1)*base] for i in range(n)]
        before = torch.stack([torch.randint(round(a*20),round(b*20)+1,(n,m),device=self.device)/20
            for a,b in self._scene_cfg['before_leader_size']], -1)
        bridge = self._box_size[:,4*m:4*m+1].repeat(1,3,1)
        bridge[:,:,0] *= torch.arange(1,4,device=self.device)
        extra = torch.cat((before,bridge),1)
        rho = extra.new_full((n,m+3),100.)
        rho[:,m:] = self._joint['box']['density']
        if self._warehouse_coop:
            self._warehouse_bank = self._bridge_bank+3
            boxes = extra.new_tensor(self._warehouse_coop['box_size']).expand(n,3,3).clone()
            boxes[:,:,0] *= torch.arange(1,4,device=self.device)
            extra = torch.cat((extra,boxes),1)
            rho = torch.cat((rho,rho.new_full((n,3),self._joint['box']['density'])),1)
        density = self._box_density.reshape(n,base)
        self._box_size = torch.cat((self._box_size,extra),1)
        self._box_density = torch.cat((density,rho),1).flatten()
        self._box_scale = self._box_size/self._box_size.new_tensor(self._build_base_size)
        self._box_assets = []
        cache = {}
        for env in range(n):
            self._box_assets.extend(old_assets[env])
            for size, mass_density in zip(extra[env].cpu().tolist(),rho[env].cpu().tolist()):
                key = (*size,mass_density)
                if key not in cache:
                    options = gymapi.AssetOptions()
                    options.density = mass_density
                    options.angular_damping = options.linear_damping = .01
                    options.max_angular_velocity = 100.
                    cache[key] = self.gym.create_box(self.sim,*size,options)
                self._box_assets.append(cache[key])
        fields = ('_static_half','_static_quat','_static_local','_surface_center','_surface_half',
            '_surface_height','_structure_bounds','_structure_yaw','_structure_shift','_structure_axis',
            '_structure_direction','_structure_near','_bridge_gap','_bridge_far','_bridge_half_a','_bridge_half_b')
        variants = []
        for variant in range(3):
            self._geometry_bridge_slot = self._bridge_bank+variant
            self._build_scene_geometry()
            variants.append({key:getattr(self,key).clone() for key in fields})
        self._bridge_geometry = {key:torch.stack([v[key] for v in variants],1) for key in fields}
        for key in fields:
            setattr(self,key,variants[0][key])
        # Keep the room wall coordinates paired with variant zero's geometry.
        self._geometry_bridge_slot = self._bridge_bank
        self._build_scene_geometry()
        for key in fields:
            setattr(self,key,getattr(self,key).clone())
            self._bridge_geometry[key][:,0] = getattr(self,key)
        if self._warehouse_coop:
            from env.tasks.multi_agent.warehouse_coop import build_geometry
            build_geometry(self, fields)

    def _surface_halves(self, cfg, shape):
        steps = torch.arange(cfg['size_range'][0], cfg['size_range'][1]+cfg['size_step']/2,
                             cfg['size_step'], device=self.device)
        return steps[torch.randint(len(steps), shape, device=self.device)]/2

    def _build_scene_geometry(self):
        n, s = self.num_envs, self._scene_cfg
        device = self.device
        bridge_cfg, room_cfg = s['bridge'], s['room']
        bridge = self._box_size[:, self._geometry_bridge_slot]
        height = bridge[:, 2]+bridge_cfg['top_drop']
        cliff_half = self._surface_halves(s['cliff'], (n, 2))
        half_a = self._surface_halves(bridge_cfg, (n, 2))
        half_b = self._surface_halves(bridge_cfg, (n, 2))
        room_half = self._surface_halves(room_cfg, (n, 2))
        far, self._bridge_gap, bridge_bounds = scene.bridge_stages(half_a, half_b, bridge, bridge_cfg,
            align_entry=bool(self._bridge_training_init))
        self._bridge_far, self._bridge_half_a, self._bridge_half_b = far, half_a, half_b
        thickness = room_cfg['wall_thickness']
        walls, wall_half, wall_starts, wall_ends = scene.room_walls(
            room_half, thickness, room_cfg['wall_height'], room_cfg['passage_width'])
        walls[..., 2] += room_cfg['floor_height']
        self._room_half = room_half

        bounds = torch.stack((torch.stack((-cliff_half[:, 0], cliff_half[:, 0], -cliff_half[:, 1], cliff_half[:, 1]), -1),
                              bridge_bounds,
                              torch.stack((-room_half[:, 0]-thickness, room_half[:, 0]+thickness,
                                           -room_half[:, 1]-thickness, room_half[:, 1]+thickness), -1)), 1)
        self._structure_bounds = bounds
        placed = [scene.place_structure(bounds[:, index], s['floor_clearance'], s['translation_range'])
                  for index in range(len(SCENES))]
        self._structure_yaw, self._structure_shift = [torch.stack([row[key] for row in placed], 1) for key in ('yaw', 'shift')]
        self._structure_axis, self._structure_direction, self._structure_near = [
            torch.stack([row[key] for row in placed], 1) for key in ('axis', 'direction', 'near')]
        reach = max(float(row['reach'].max()) for row in placed)
        if self._scene_spacing < reach+FLOOR_REACH:
            raise ValueError('sceneSpacing is too small for the sampled structures: reach {:.1f}m'.format(reach))

        centers = torch.zeros(n, 3, ROOM_ACTORS, 3, device=device)
        half = torch.zeros(n, 3, ROOM_ACTORS, 3, device=device)
        centers[..., 2] = scene.HIDDEN
        centers[:, CLIFF, 0, 2], half[:, CLIFF, 0, :2] = height/2, cliff_half
        half[:, CLIFF, 0, 2] = height/2
        centers[:, BRIDGE, 0, 2], half[:, BRIDGE, 0, :2] = height/2, half_a
        half[:, BRIDGE, 0, 2] = height/2
        centers[:, BRIDGE, 1, :2], centers[:, BRIDGE, 1, 2] = far, height/2
        half[:, BRIDGE, 1, :2], half[:, BRIDGE, 1, 2] = half_b, height/2
        centers[:, ROOM, 0, 2] = room_cfg['floor_height']/2
        half[:, ROOM, 0, :2], half[:, ROOM, 0, 2] = room_half, room_cfg['floor_height']/2
        centers[:, ROOM, 1:] = walls
        half[:, ROOM, 1:] = wall_half
        self._static_half = half
        self._static_quat = scene.yaw_quat(self._structure_yaw)[:, :, None].expand(-1, -1, ROOM_ACTORS, -1)
        yaw_grid = self._structure_yaw[:, :, None]
        xy = scene.rotate_xy(centers[..., :2], yaw_grid)+self._structure_shift[:, :, None]
        self._static_local = torch.cat((xy, centers[..., 2:]), -1)
        self._wall_local = (scene.rotate_xy(wall_starts, self._structure_yaw[:, ROOM, None])+self._structure_shift[:, ROOM, None],
                            scene.rotate_xy(wall_ends, self._structure_yaw[:, ROOM, None])+self._structure_shift[:, ROOM, None])

        self._surface_center = torch.zeros(n, 3, 2, 2, device=device)
        self._surface_half = torch.zeros(n, 3, 2, 2, device=device)
        self._surface_half[:, CLIFF, 0], self._surface_half[:, ROOM, 0] = cliff_half, room_half
        self._surface_half[:, BRIDGE, 0], self._surface_half[:, BRIDGE, 1] = half_a, half_b
        self._surface_center[:, BRIDGE, 1] = far
        self._surface_height = torch.stack((height, height, torch.full_like(height, room_cfg['floor_height'])), 1)

        spacing = self._scene_spacing
        self._scene_offsets = torch.tensor([[-spacing/2, -spacing/2, 0.], [spacing/2, -spacing/2, 0.],
                                            [0., spacing/2, 0.]], device=device)
        if self._warehouse:
            from env.tasks.multi_agent.warehouse_sensor import add_geometry
            add_geometry(self)
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        self._static_options = options
        self._static_assets = {}
        self._terrain_actor_records = []

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        if object_id >= self._bridge_bank:
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, self._box_handles[-1])
            for prop in props:
                prop.friction = self._joint['box']['friction']
                prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, self._box_handles[-1], props)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        origin = self.gym.get_env_origin(env_ptr)
        for index in range(len(self._scene_names)):
            warehouse = index == WAREHOUSE and self._warehouse_coop
            geometry = self._warehouse_geometry if warehouse else self._bridge_geometry
            for variant in range(6 if warehouse else 3 if index == BRIDGE else 1):
                for slot in range(ROOM_ACTORS):
                    half = geometry['_static_half'][env_id,variant,index,slot]
                    size = tuple((2*half).cpu().tolist())
                    if size[0] <= 0:
                        continue
                    if size not in self._static_assets:
                        self._static_assets[size] = self.gym.create_box(self.sim,*size,self._static_options)
                    center = geometry['_static_local'][env_id,variant,index,slot]
                    offset = self._scene_offsets[index].clone()
                    if index == BRIDGE or warehouse:
                        offset[0] += variant*2*self._scene_spacing
                    pose = gymapi.Transform()
                    xyz = (center+offset).cpu().tolist()
                    pose.p = gymapi.Vec3(origin.x+xyz[0],origin.y+xyz[1],origin.z+xyz[2])
                    pose.r = gymapi.Quat(*geometry['_static_quat'][env_id,variant,index,slot].cpu().tolist())
                    handle = self.gym.create_actor(env_ptr,self._static_assets[size],pose,
                        f'{self._scene_names[index]}_{variant}_{slot}',env_id,0)
                    self.gym.set_rigid_body_color(env_ptr,handle,0,gymapi.MESH_VISUAL,gymapi.Vec3(.32,.36,.42))
                    actor_index = self.gym.get_actor_index(env_ptr,handle,gymapi.DOMAIN_SIM)
                    self._terrain_actor_records.append((env_id,index,variant,slot,actor_index))
