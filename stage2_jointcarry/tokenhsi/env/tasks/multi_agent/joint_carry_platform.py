import torch

from utils.joint_carry_platform import platform_heights


def build_platform(task, env_id, env_ptr):
    from isaacgym import gymapi

    pose = gymapi.Transform()
    pose.p.z = task._platform_inactive_height
    handle = task.gym.create_actor(env_ptr, task._platform_asset, pose,
        'joint_loco_platform', env_id, task._PLATFORM_COLLISION_FILTER, 0)
    task.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                  gymapi.Vec3(.5, .235, .6))


class JointCarryPlatform:
    def __init__(self, task):
        self.task = task
        self.spec = task._joint_loco_platform_cfg
        count = task.get_num_actors_per_env()
        self.states = task._root_states.view(task.num_envs, count, 13)[:, -1]
        self.actor_ids = torch.arange(task.num_envs, device=task.device, dtype=torch.int32)*count+count-1
        self.height = torch.zeros(task.num_envs, device=task.device)
        self.active = torch.zeros(task.num_envs, device=task.device, dtype=torch.bool)

    def reset(self, ids):
        t = self.task
        self.states[ids] = 0.
        self.states[ids, :2] = t._env_origins[ids, :2]
        self.states[ids, 2] = t._platform_inactive_height
        self.states[ids, 6] = 1.
        self.height[ids] = 0.
        self.active[ids] = False
        if 'loco' not in t._skill:
            return
        loco = ids[t._joint_reset.skill_index[ids] == t._skill.index('loco')]
        chosen = loco[torch.rand(len(loco), device=t.device) < self.spec['probability']]
        if not len(chosen):
            return
        payload = t._logical_box_order[chosen, 0]
        boxes = t._box_states[chosen, payload].clone()
        size = t._box_size[chosen, payload]
        height = platform_heights(size[:, 2], torch.rand(len(chosen), device=t.device), self.spec)
        self.states[chosen, :2] = boxes[:, :2]
        self.states[chosen, 2] = boxes[:, 2]-size[:, 2]/2+height-self.spec['size'][2]/2
        self.states[chosen, 3:7] = boxes[:, 3:7]
        boxes[:, 2] += height+self.spec['initialGap']
        t._box_states[chosen, payload] = boxes
        self.height[chosen] = height
        self.active[chosen] = True
