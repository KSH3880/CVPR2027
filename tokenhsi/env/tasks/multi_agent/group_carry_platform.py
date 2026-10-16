import torch

from utils.joint_carry_platform import platform_heights


class GroupCarryPlatform:
    def __init__(self, task):
        self.task = task
        self.spec = task._joint_loco_platform_cfg
        count = task.get_num_actors_per_env()
        self.states = task._root_states.view(task.num_envs, count, 13)[:, -2:]
        self.actor_ids = torch.arange(task.num_envs, device=task.device, dtype=torch.int32)[:, None]*count + \
            torch.arange(count-2, count, device=task.device, dtype=torch.int32)[None]
        self.height = torch.zeros(task.num_envs, 2, device=task.device)
        self.active = torch.zeros_like(self.height, dtype=torch.bool)

    def reset(self, ids):
        t = self.task
        self.states[ids] = 0.
        self.states[ids, :, :2] = t._env_origins[ids, None, :2]
        self.states[ids, :, 2] = t._platform_inactive_height
        self.states[ids, :, 6] = 1.
        self.height[ids] = 0.
        self.active[ids] = False
        if 'loco' not in t._skill:
            return
        for human in range(4):
            chosen = ids[t._group_joint[ids, human] & (t._group_leader[ids, human] == human) &
                (t._group_skill[ids, human] == t._skill.index('loco')) &
                (torch.rand(len(ids), device=t.device) < self.spec['probability'])]
            payload = t._logical_box_order[chosen, human]
            slot = payload-8
            size = t._box_size[chosen, payload]
            height = platform_heights(size[:, 2], torch.rand(len(chosen), device=t.device), self.spec)
            box = t._box_states[chosen, payload]
            self.states[chosen, slot, :2] = box[:, :2]
            self.states[chosen, slot, 2] = box[:, 2]-size[:, 2]/2+height-self.spec['size'][2]/2
            self.states[chosen, slot, 3:7] = box[:, 3:7]
            t._box_states[chosen, payload, 2] += height+self.spec['initialGap']
            self.height[chosen, slot] = height
            self.active[chosen, slot] = True
