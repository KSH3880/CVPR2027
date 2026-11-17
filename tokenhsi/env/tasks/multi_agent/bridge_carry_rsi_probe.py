import itertools
import math

from isaacgym import gymapi, gymtorch
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.bridge_carry_rsi_cache import frame_valid, object_state
from utils.size_rsi import screen_states
from utils.stage1_geometry import pose_geometry_valid


def profiles(env):
    ranges = env['box']['build']['taskSizeRanges']
    values = set()
    for task in ('HOLDING_AT', 'HOLDING_ON_TOP'):
        values.update(itertools.product(*(range(round(lo*20), round(hi*20)+1) for lo,hi in ranges[task])))
    return sorted(values)


class CarryProbe(HumanoidMACarry):
    def _load_box_asset(self):
        sizes = torch.tensor(self.cfg['carry_profiles'], device=self.device)/20.
        self._box_size = sizes[:, None].expand(-1, self.num_objects, -1).clone()
        self._box_density = torch.full((self.num_envs*self.num_objects,), 100., device=self.device)
        self._box_scale = self._box_size/torch.tensor(self._build_base_size, device=self.device)
        self._box_assets = []
        for size in self._box_size.flatten(0,1):
            options = gymapi.AssetOptions()
            options.angular_damping = options.linear_damping = .01
            options.max_angular_velocity = 100.
            options.density = 100.
            options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, *size.tolist(), options))


@torch.no_grad()
def screen(task):
    lib = task._motion_lib['carryWith']
    n = task.num_envs
    ids = torch.arange(n, device=task.device)
    owner = torch.zeros_like(ids)
    offset = task._env_origins+task._agent_spawn_offsets[0]
    size = task._box_size[:, 0]
    records = {tuple(k): [] for k in task.cfg['carry_profiles']}
    original = task._root_states.clone()
    dofs = task._dof_state.clone()
    for clip in range(lib.num_motions()):
        for frame in range(int(lib._motion_num_frames[clip])):
            mids = torch.full((n,), clip, dtype=torch.long, device=task.device)
            times = torch.full((n,), frame*float(lib._motion_dt[clip]), device=task.device)
            root,q,dof,vel,ang,dof_vel,_ = lib.get_motion_state(mids,times)
            bp,bq,bv,ba = lib.get_motion_state_max(mids,times)
            body = torch.cat((bp,bq,bv,ba),-1)
            human = torch.cat((root,q,vel,ang),-1)
            box = object_state(lib,mids,times)
            good = frame_valid(task,human,body,box,size,times)
            if not good.any():
                continue
            task._root_states.copy_(original)
            task._dof_state.copy_(dofs)
            task._humanoid_root_states[..., :2] = task._env_origins[:, None, :2]+15.
            task._humanoid_root_states[..., 7:] = 0.
            task._box_states[..., :2] = task._env_origins[:, None, :2]-15.
            task._box_states[..., 7:] = 0.
            task._root_states[task._platform_actor_ids, 2] = -1000.
            task._root_states[task._tar_platform_actor_ids, 2] = -1000.
            task._set_env_state(ids,owner,root+offset,q,dof,vel,ang,dof_vel)
            box[:, :3] += offset
            task._box_states[:,0] = box
            initial = task._humanoid_root_states[:,0].clone()
            task.gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
            task.gym.set_dof_state_tensor(task.sim,gymtorch.unwrap_tensor(task._dof_state))
            target = task._dof_pos.contiguous().clone()
            task.gym.set_dof_position_target_tensor(task.sim,gymtorch.unwrap_tensor(target))
            for _ in range(math.ceil(.1/task.sim_params.dt)):
                task.gym.simulate(task.sim); task.gym.fetch_results(task.sim,True)
                task._refresh_sim_tensors()
                current = task._box_states[:,:1]
                good &= screen_states(task._humanoid_root_states[:,0],current,initial,box[:,None])
                bodies = task._rigid_body_pos[:,0]-offset[:,None]
                local_box = current.clone(); local_box[...,:3] -= offset[:,None]
                good &= pose_geometry_valid(bodies,bodies[:,task._key_body_ids[-2:]],local_box,size[:,None])
                low = bodies[...,2] < task._termination_heights
                low[:,task._contact_body_ids] = False
                contact = task._contact_forces[:,0].clone(); contact[:,task._contact_body_ids] = 0
                good &= ~(low.any(-1) & (contact.abs().sum(-1) > .1).any(-1))
            for row in good.nonzero().flatten().tolist():
                records[tuple(task.cfg['carry_profiles'][row])].append((clip,frame))
        print('[carry RSI] screened clip',clip,'frames',sum(map(len,records.values())),flush=True)
    return records
