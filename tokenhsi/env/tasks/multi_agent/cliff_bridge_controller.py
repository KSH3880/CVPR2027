from isaacgym import gymapi, gymtorch
import torch

from utils.cliff_bridge_spec import CONTROLLED, bridge_control, bridge_schedule, bridge_trajectory


class CliffBridgeController:
    def _load_box_asset(self):
        super()._load_box_asset()
        n, base = self._box_size.shape[:2]
        self._controlled_bank = base
        sizes = self._box_size[:, self._bridge_bank:self._bridge_bank+3].clone()
        density = self._scene_cfg['bridge']['controller']['density']
        assets, old = [], self._box_assets
        cache = {}
        for env in range(n):
            assets.extend(old[env*base:(env+1)*base])
            for size in sizes[env].cpu().tolist():
                key = tuple(size)
                if key not in cache:
                    options = gymapi.AssetOptions()
                    options.disable_gravity = True
                    options.max_angular_velocity = 0.
                    options.density = density
                    surface = self._scene_cfg['bridge'].get('top_surface')
                    if surface:
                        from env.tasks.multi_agent.bridge_surface import load_asset
                        cache[key] = load_asset(self.gym, self.sim, self.cfg['args'].output_path+'/bridge_surface_assets',
                            size, options, surface['thickness'], controlled=True)
                    else:
                        cache[key] = self.gym.create_box(self.sim, *size, options)
                assets.append(cache[key])
        self._box_assets = assets
        self._box_size = torch.cat((self._box_size, sizes), 1)
        self._box_scale = self._box_size/self._box_size.new_tensor(self._build_base_size)
        self._box_density = torch.cat((self._box_density.reshape(n, base),
            sizes.new_full((n, 3), density)), 1).flatten()
        self.num_physical_objects += 3
        self._controlled_records = []

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        if object_id >= self._controlled_bank:
            handle = self._box_handles[-1]
            body = self.gym.get_actor_rigid_body_index(env_ptr, handle, 0, gymapi.DOMAIN_SIM)
            mass = self.gym.get_actor_rigid_body_properties(env_ptr, handle)[0].mass
            self._controlled_records.append((env_id, object_id-self._controlled_bank, body, mass))

    def _init_controller(self):
        n, device = self.num_envs, self.device
        records = torch.tensor(self._controlled_records, device=device)
        self._controlled_bodies = records[:, 2].long().view(n, 3)
        self._controlled_mass = records[:, 3].view(n, 3)
        self._controlled_forces = torch.zeros(self.gym.get_sim_rigid_body_count(self.sim), 3, device=device)
        self._controlled_start = torch.zeros(n, 3, device=device)
        self._controlled_goal = torch.zeros_like(self._controlled_start)
        self._controlled_wait = torch.zeros(n, device=device)
        self._controlled_duration = torch.ones(n, device=device)
        self._controlled_arrival = torch.ones(n, device=device)
        self._controlled_present = torch.zeros(n, dtype=torch.bool, device=device)
        self._controller_error = torch.zeros(3, device=device)

    def _reset_controller(self, ids, start, goal):
        spec = self._scene_cfg['bridge']['controller']
        start = torch.where(self._controlled_present[ids, None], goal, start)
        duration, wait = bridge_schedule((goal-start).norm(dim=-1), self._controlled_arrival[ids],
            spec['speed_range'], self.dt)
        wait = torch.where(self._controlled_present[ids], 0., wait)
        self._controlled_start[ids], self._controlled_goal[ids] = start, goal
        self._controlled_duration[ids], self._controlled_wait[ids] = duration, wait
        position, velocity = bridge_trajectory(start, goal, -wait, duration)
        self._box_states[ids, self._bridge_slot[ids], :3] = position
        self._box_states[ids, self._bridge_slot[ids], 7:10] = velocity

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._controlled_forces.zero_()
        ids = ((self._bridge_branch == CONTROLLED) & (self._scene_ids == 1)).nonzero().flatten()
        if not len(ids):
            return
        elapsed = (self.progress_buf[ids]+1)*self.dt-self._controlled_wait[ids]
        target, velocity = bridge_trajectory(self._controlled_start[ids], self._controlled_goal[ids],
            elapsed, self._controlled_duration[ids])
        body = self._controlled_bodies[ids, self._bridge_span[ids]-1]
        mass = self._controlled_mass[ids, self._bridge_span[ids]-1]
        states = self._box_states[ids, self._bridge_slot[ids]]
        self._controlled_forces[body] = bridge_control(target, velocity, states, mass,
            self._scene_cfg['bridge']['controller'])
        error = (states[:, :3]-target).norm(dim=-1)
        self._controller_error[0] += len(ids)
        self._controller_error[1] += error.sum()
        self._controller_error[2] = torch.maximum(self._controller_error[2], error.max())

    def _physics_step(self):
        ids = torch.arange(self.num_envs, device=self.device) if self._v4_rewards else None
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self._controlled_forces),
                None, gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
            if self._v4_rewards:
                self.gym.fetch_results(self.sim, True)
                self._refresh_sim_tensors()
                self._update_push_feedback(ids)
