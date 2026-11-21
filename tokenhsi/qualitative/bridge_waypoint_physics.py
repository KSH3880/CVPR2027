from isaacgym import gymapi

from env.tasks.multi_agent import bridge_surface
from qualitative.bridge_waypoint_task import BridgeWaypointTask
from qualitative.bridge_waypoint_cap import CAP_FILTER, FollowCapMixin, load_body


MODES = ('baseline', 'fixed', 'carry', 'fixed_carry', 'platform', 'baseline_nocap', 'carry_nocap', 'follow_cap')


class BridgeWaypointPhysicsTask(FollowCapMixin, BridgeWaypointTask):
    def __init__(self, *args, bridge_physics, cap_lift=0., **kwargs):
        self.bridge_physics = bridge_physics
        self.cap_lift = cap_lift
        self._waypoint_fixed = bridge_physics in ('fixed', 'fixed_carry', 'platform')
        super().__init__(*args, **kwargs)

    def _load_box_asset(self):
        super()._load_box_asset()
        mode = self.bridge_physics
        cache = {}
        for env in range(self.num_envs):
            for slot in range(self._bridge_bank, self._bridge_bank+3):
                size = self._box_size[env, slot].tolist()
                index = env*self.num_physical_objects+slot
                density = float(self._box_density[index])
                if mode in ('carry', 'fixed_carry', 'carry_nocap'):
                    density = float(self._box_density[env*self.num_physical_objects+self._leader_bank])
                elif mode == 'platform':
                    density = self._static_options.density
                key = (*size, density)
                if key not in cache:
                    if mode == 'platform':
                        asset = self.gym.create_box(self.sim, *size, self._static_options)
                    else:
                        options = gymapi.AssetOptions()
                        options.density = density
                        options.angular_damping = options.linear_damping = .01
                        options.max_angular_velocity = 100.
                        options.fix_base_link = self._waypoint_fixed
                        if mode.endswith('_nocap'):
                            asset = self.gym.create_box(self.sim, *size, options)
                        elif mode == 'follow_cap':
                            asset = load_body(self.gym, self.sim,
                                self.cfg['args'].output_path+'/bridge_surface_assets', size, options,
                                self._scene_cfg['bridge']['top_surface']['thickness'])
                        else:
                            asset = bridge_surface.load_asset(self.gym, self.sim,
                                self.cfg['args'].output_path+'/bridge_surface_assets', size, options,
                                self._scene_cfg['bridge']['top_surface']['thickness'])
                    cache[key] = asset
                self._box_assets[index] = cache[key]
                self._box_density[index] = density

    def _build_box(self, env_id, object_id, env_ptr):
        bridge = self._bridge_bank <= object_id < self._bridge_bank+3
        if not bridge:
            return super()._build_box(env_id, object_id, env_ptr)
        index = env_id*self.num_physical_objects+object_id
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(1000.+12*object_id, 1000., float(self._box_size[env_id, object_id, 2])/2)
        handle = self.gym.create_actor(env_ptr, self._box_assets[index], pose,
            f'box_{object_id}', env_id, 0, 0)
        self._box_handles.append(handle)
        mode = self.bridge_physics
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        if mode in ('carry', 'fixed_carry', 'carry_nocap'):
            source = self._box_handles[env_id*self.num_physical_objects+self._leader_bank]
            carry_props = self.gym.get_actor_rigid_shape_properties(env_ptr, source)
            props = [carry_props[0] for _ in props]
        elif mode in ('fixed', 'baseline_nocap', 'follow_cap'):
            for prop in props:
                prop.friction = self._joint['box']['friction']
                prop.restitution = 0.
            if mode == 'fixed':
                props[1].friction = self._scene_cfg['bridge']['top_surface']['friction']
            if mode == 'follow_cap':
                props[0].filter |= CAP_FILTER
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)


def physics_info(task):
    result = []
    for env in range(task.num_envs):
        slot = int(task._bridge_slot[env])
        handle = task._box_handles[env*task.num_physical_objects+slot]
        body = task.gym.get_actor_rigid_body_properties(task.envs[env], handle)[0]
        shapes = task.gym.get_actor_rigid_shape_properties(task.envs[env], handle)
        size = task._box_size[env, slot]
        leader = int(task._roles[env, 1])
        payload = int(task._agent_box_assignment[env, leader])
        carry = task._box_handles[env*task.num_physical_objects+payload]
        carry_body = task.gym.get_actor_rigid_body_properties(task.envs[env], carry)[0]
        carry_shape = task.gym.get_actor_rigid_shape_properties(task.envs[env], carry)[0]
        result.append(dict(env=env, mode=getattr(task, 'bridge_physics', 'baseline'),
            fixed_base=getattr(task, '_waypoint_fixed', False), size=size.tolist(), mass=body.mass,
            density=body.mass/float(size.prod()),
            shapes=[dict(friction=p.friction, restitution=p.restitution,
                contact_offset=p.contact_offset, rest_offset=p.rest_offset) for p in shapes],
            carry_density=carry_body.mass/float(task._box_size[env, payload].prod()),
            carry_friction=carry_shape.friction))
        if getattr(task, 'bridge_physics', '') == 'follow_cap':
            cap = task._cap_handles[env*3+slot-task._bridge_bank]
            prop = task.gym.get_actor_rigid_shape_properties(task.envs[env], cap)[0]
            result[-1]['cap'] = dict(fixed_base=True, follows='position_and_rotation',
                thickness=task._scene_cfg['bridge']['top_surface']['thickness'],
                lift=task.cap_lift,
                friction=prop.friction, restitution=prop.restitution, filter=prop.filter)
    return result
