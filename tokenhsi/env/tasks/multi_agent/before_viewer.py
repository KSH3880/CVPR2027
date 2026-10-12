import torch

from env.tasks.multi_agent.before_reset import align_transform, transform_states, quat_multiply
from env.tasks.multi_agent.edge_interaction_reward import quat_rotate
from utils.edge_context_spec import AT


def apply_layout(t, ids):
    ids = ids[t._before_families[ids] > 0]
    if not len(ids):
        return
    origin = t._env_origins[ids]
    physical = t._logical_box_order[ids]
    boxes = t._box_states[ids[:, None], physical].clone()
    sizes = t._box_size[ids[:, None], physical]
    boxes[..., :2] = origin[:, None, :2] + boxes.new_tensor(
        [[0., 0.], [-3., -2.], [-3., 2.], [3., 2.]])
    boxes[..., 2] = origin[:, None, 2] + sizes[..., 2]/2
    boxes[..., 3:7] = boxes.new_tensor([0., 0., 0., 1.])
    boxes[..., 7:] = 0.
    stack = t._before_families[ids] == 3
    boxes[stack, 1, :2] = origin[stack, :2] + boxes.new_tensor([0., 1.1])
    t._box_states[ids[:, None], physical] = boxes
    t._tar_pos[ids, 0] = boxes[:, 0, :3] + boxes.new_tensor([0., -1.5, 0.])
    for owner in (0, 1):
        old = t._humanoid_root_states[ids, owner].clone()
        target = old.clone()
        target[:, :2] = origin[:, :2] + old.new_tensor([-2., 0.] if owner == 0 else [.8, 0.])
        if owner == 1:
            target[stack, 1] += 1.1
        target[:, 3:7] = old.new_tensor([0., 0., 0., 1.] if owner == 0 else [0., 0., 1., 0.])
        rotation, translation = align_transform(old, target)
        root = transform_states(old, rotation, translation)
        bodies = transform_states(t._kinematic_humanoid_rigid_body_states[ids, owner],
                                  rotation, translation)
        root[:, 7:] = 0.
        bodies[..., 7:] = 0.
        t._humanoid_root_states[ids, owner] = root
        t._kinematic_humanoid_rigid_body_states[ids, owner] = bodies
        t._dof_vel[ids, owner] = 0.
        t._before_amp_rotation[ids, owner] = quat_multiply(
            rotation, t._before_amp_rotation[ids, owner])
        t._before_amp_translation[ids, owner] = quat_rotate(
            rotation, t._before_amp_translation[ids, owner]) + translation


class BeforeOrder:
    def __init__(self, graph, families):
        self.families = families.clone()
        self.dependent = graph.prereq_mask.any(-1).clone()
        self.at = (graph.edge_valid & (graph.edge_relation == AT) & (graph.edge_owner == 0)).clone()
        self.times = {name: torch.full_like(families, -1., dtype=torch.float)
                      for name in ('at', 'raw_b', 'gated_b')}
        self.closed_completion = torch.zeros_like(families, dtype=torch.bool)

    def update(self, result, seconds, alive):
        active = alive & (self.families > 0)
        raw_b = (result['raw_own_success'] & self.dependent).any(-1)
        gate = (result['prerequisite_gate'] | ~self.dependent).all(-1)
        self.closed_completion |= active & raw_b & ~gate
        values = {'at': (result['raw_own_success'] & self.at).any(-1),
                  'raw_b': raw_b, 'gated_b': (result['own_success'] & self.dependent).any(-1)}
        for name, value in values.items():
            first = active & value & (self.times[name] < 0)
            self.times[name][first] = seconds
            if first[0]:
                print('[BEFORE order] env=0 {}={:.3f}s'.format(name, seconds), flush=True)

    def report(self):
        rows = []
        names = ('independent', 'before_climb', 'before_sit', 'before_stack')
        for env in (self.families > 0).nonzero().flatten().tolist():
            times = {name: float(value[env]) for name, value in self.times.items()}
            a, b = times['at'], times['raw_b']
            order = ('incomplete' if b < 0 else 'early' if a < 0 or b < a else
                     'same_step' if a == b else 'after_at')
            rows.append(dict(env=env, family=names[int(self.families[env])], order=order,
                **{name + '_seconds': value if value >= 0 else None for name, value in times.items()},
                raw_b_while_gate_closed=bool(self.closed_completion[env])))
        return rows
