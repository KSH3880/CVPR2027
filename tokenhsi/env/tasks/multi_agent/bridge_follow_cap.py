import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate
import torch

from env.tasks.multi_agent import bridge_surface


CAP_FILTER = 1 << 29


def load_body(gym, sim, directory, size, options, thickness):
    source = bridge_surface.write_asset(directory, size, options.density, thickness)
    tree = ET.parse(source)
    link = tree.getroot().find('link')
    link.remove(link.find("collision[@name='top_cap']"))
    path = source.with_name(source.stem+'_body.urdf')
    tree.write(path)
    return gym.load_asset(sim, str(path.parent), path.name, options)


class FollowCapMixin:
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self.bridge_physics != 'follow_cap':
            return
        if env_id == 0:
            self._cap_assets, self._cap_records, self._cap_handles = {}, [], []
        thickness = self._scene_cfg['bridge']['top_surface']['thickness']
        for slot in range(self._bridge_bank, self._bridge_bank+3):
            size = (*self._box_size[env_id, slot, :2].tolist(), thickness)
            if size not in self._cap_assets:
                self._cap_assets[size] = self.gym.create_box(self.sim, *size, self._static_options)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(1000.+12*slot, 1000., -1000.)
            handle = self.gym.create_actor(env_ptr, self._cap_assets[size], pose,
                f'follow_cap_{slot}', env_id, CAP_FILTER)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            props[0].friction = self._scene_cfg['bridge']['top_surface']['friction']
            props[0].restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.2,.8,.9))
            self._cap_handles.append(handle)
            self._cap_records.append(self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM))

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        if self.bridge_physics == 'follow_cap':
            self._cap_actor_ids = torch.tensor(self._cap_records, device=self.device, dtype=torch.long).view(self.num_envs, 3)

    def _sync_follow_cap(self, ids=None, reset=False):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        bodies = self._box_states[ids, self._bridge_bank:self._bridge_bank+3]
        offset = torch.zeros_like(bodies[..., :3])
        thickness = self._scene_cfg['bridge']['top_surface']['thickness']
        offset[..., 2] = (self._box_size[ids, self._bridge_bank:self._bridge_bank+3, 2]-thickness)/2+self.cap_lift
        offset = quat_rotate(bodies[..., 3:7].reshape(-1, 4), offset.reshape(-1, 3)).view_as(offset)
        actors = self._cap_actor_ids[ids]
        self._root_states[actors, :3] = bodies[..., :3]+offset
        self._root_states[actors, 3:7] = bodies[..., 3:7]
        self._root_states[actors, 7:13] = 0.
        if reset:
            # The last indexed update must also retain pending platform and humanoid resets.
            count = self.get_num_actors_per_env()
            indices = (ids[:, None]*count+torch.arange(count, device=self.device)).flatten().to(torch.int32)
        else:
            indices = actors.flatten().to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(indices), len(indices))

    def _reset_env_tensors(self, ids):
        super()._reset_env_tensors(ids)
        if self.bridge_physics == 'follow_cap':
            self._sync_follow_cap(ids, reset=True)

    def _physics_step(self):
        if self.bridge_physics != 'follow_cap':
            return super()._physics_step()
        ids = torch.arange(self.num_envs, device=self.device)
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.apply_rigid_body_force_tensors(self.sim,
                gymtorch.unwrap_tensor(self._controlled_forces), None, gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            if self._v4_rewards:
                self._update_push_feedback(ids)
            self._sync_follow_cap()
