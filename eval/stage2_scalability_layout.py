from copy import deepcopy

import torch

from env.tasks.multi_agent import stage2_group_layout
from utils import coupled_push_scene as scene, stage2_spec
from utils.before_object_alias import alias_layout
from utils.coupled_push_spec import physical_box_count
from utils.task_role_spec import task_graph_packet


original_validate = stage2_spec.validate_env


def validate_env(env):
    if env['numAgents'] != 2:
        return original_validate(env)
    if env['numObjects'] != 4:
        raise ValueError('Two-person protocol requires four tokens')
    equivalent = deepcopy(env)
    equivalent.update(numAgents=4, numObjects=8)
    equivalent['stage2JointPush']['physical_boxes'] = physical_box_count(4, 8)
    original_validate(equivalent)


def token_groups(task, ids):
    n, m, o = len(ids), task.num_agents, task.num_objects
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs)[ids].reshape(n, m, 5).long()
    valid, kind, actor, payload, target = packet.unbind(-1)
    groups = task._groups[ids]
    owner = torch.full((n, o), -1, device=task.device, dtype=torch.long)
    for values, used in ((payload, valid.bool() & (kind >= 2)), (target, valid.bool())):
        used &= (values >= m) & (values < m+o)
        es, record = used.nonzero(as_tuple=True)
        owner[es, values[es, record]-m] = groups[es, actor[es, record]]
    dependent, _, aliases, _ = alias_layout(packet, m, o)
    es, record = dependent.nonzero(as_tuple=True)
    owner[es, aliases[es, record]-m] = groups[es, actor[es, record]]
    if not torch.equal(owner >= 0, task._stage2_task_slots[ids]):
        raise ValueError('Task token group assignment differs')
    required = torch.zeros(n, m, device=task.device, dtype=torch.long)
    required.scatter_add_(1, owner.clamp_min(0), (owner >= 0).long())
    sizes = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    leader = groups == torch.arange(m, device=task.device)[None]
    budget = torch.where(leader, 2*sizes-required, 0)
    if (budget < 0).any():
        raise ValueError('Group exceeds its 2k token budget')
    for row in range(n):
        free = (owner[row] < 0).nonzero().flatten()
        allocation = torch.repeat_interleave(torch.arange(m, device=task.device), budget[row])
        if len(free) != len(allocation):
            raise ValueError('Distractor token allocation mismatch')
        owner[row, free] = allocation
    return owner


original_place = scene.place_distractors


def place_distractors(task, ids):
    if task.num_agents <= 4:
        return original_place(task, ids)
    owners = token_groups(task, ids)
    task._protocol_token_groups[ids] = owners
    valid = task._stage2_task_slots[ids]
    mapping = task._logical_box_order[ids]
    row, slot = (~valid).nonzero(as_tuple=True)
    physical = mapping[row, slot]
    task._box_states[ids[row], physical, :2] = task._env_origins[ids[row], :2]+1000.+12*physical[:, None]
    for slot in range(task.num_objects):
        row = (~valid[:, slot]).nonzero().flatten()
        if not len(row):
            continue
        pending = ids[row]
        centers = task._group_spawn_targets[pending, owners[row, slot]]
        starts, ends, yaw, half = scene.paths(task, pending)
        roots = task._humanoid_root_states[pending, ..., :2]
        for _ in range(256):
            if not len(pending):
                break
            physical = task._logical_box_order[pending, slot]
            point = task._sample_arena_xy(centers)
            task._box_states[pending, physical, :2] = point
            hit = scene.swept_overlap(point[:, None], point[:, None], yaw[:, slot, None], half[:, slot, None],
                starts, ends, yaw, half, .5)
            hit[:, slot] = False
            near = ((point[:, None]-roots).norm(dim=-1) <
                    (half[:, slot].norm(dim=-1)+.35).clamp_min(1.)[:, None]).any(-1)
            rejected = hit.any(-1) | near
            pending, centers = pending[rejected], centers[rejected]
            starts, ends, yaw, half, roots = [v[rejected] for v in (starts, ends, yaw, half, roots)]


def install():
    from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
    original_build = HumanoidMAStage2._build_assignment_tensors
    original_boxes = HumanoidMAStage2._reset_boxes

    def build(self):
        original_build(self)
        self._protocol_token_groups = torch.full((self.num_envs, self.num_objects), -1,
                                                device=self.device, dtype=torch.long)

    def reset_boxes(self, ids):
        slots = self._reset_ref_slots.get('loco')
        if self.num_agents <= 4 or slots is None:
            return original_boxes(self, ids)
        env, owner = slots
        keep = (self._members[env, owner] == 1) | (self._groups[env, owner] == owner)
        self._reset_ref_slots['loco'] = (env[keep], owner[keep])
        try:
            return original_boxes(self, ids)
        finally:
            self._reset_ref_slots['loco'] = slots

    stage2_spec.validate_env = validate_env
    scene.place_distractors = place_distractors
    stage2_group_layout.clean_observation = lambda task, obs, ids: obs
    HumanoidMAStage2._build_assignment_tensors = build
    HumanoidMAStage2._reset_boxes = reset_boxes
