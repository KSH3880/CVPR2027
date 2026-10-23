"""Evaluation-only PUSH box teleportation with a fixed goal and live people."""
import os
import torch
from utils.door_reference import rotate
from utils.push_goal_switch import PushGoalSwitch


def relocation_ray(boxes, goals, roots, root_rotations, initial_direction, mode):
    if mode == 'across_goal':
        direction = goals[..., :2]-boxes[..., :2]
        direction = torch.where(direction.norm(dim=-1, keepdim=True)>1e-6,
                                direction, initial_direction)
        anchor = goals[..., :2]
    elif mode == 'behind_agent':
        forward = torch.zeros_like(roots)
        forward[..., 0] = 1
        direction = -rotate(root_rotations, forward)[..., :2]
        fallback = torch.zeros_like(direction)
        fallback[..., 0] = -1
        direction = torch.where(direction.norm(dim=-1, keepdim=True)>1e-6, direction, fallback)
        anchor = roots[..., :2]
    else:
        raise ValueError('Unknown box switch mode: '+mode)
    return anchor, direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class PushBoxSwitch(PushGoalSwitch):
    metric_key = 'box_switch'
    result_prefix = 'push_box_switch_'

    def __init__(self, task, modes, step, distance):
        super().__init__(task, [0.], step, distance)
        if not modes or any(m not in ('across_goal', 'behind_agent') for m in modes):
            raise ValueError('PUSH_BOX_SWITCH_MODES must contain across_goal and/or behind_agent')
        self.modes = modes

    @classmethod
    def from_environment(cls, task):
        value = os.environ.get('PUSH_BOX_SWITCH_MODES', '')
        if not value:
            return None
        if os.environ.get('PUSH_GOAL_SWITCH_ANGLES'):
            raise ValueError('Use either box switching or goal switching, not both')
        return cls(task, value.split(','), int(os.environ.get('PUSH_BOX_SWITCH_STEP', '60')),
                   float(os.environ.get('PUSH_BOX_SWITCH_DISTANCE', '2')))

    def begin(self, repeat):
        super().begin(repeat)
        self.mode = self.modes[repeat % len(self.modes)]
        self.event = dict(mode=self.mode, switched_scenes=0)

    def apply(self, alive):
        from isaacgym import gymtorch
        task = self.task
        ids = alive.nonzero(as_tuple=False).flatten()
        self.switched[ids] = True
        old = task._box_states[ids, :task.num_agents].clone()
        goals = task._tar_pos[ids].clone()
        roots = task._rigid_body_pos[ids, :, 0].clone()
        root_rotations = task._rigid_body_rot[ids, :, 0].clone()
        humanoids = task._humanoid_root_states[ids].clone()
        dof_pos, dof_vel = task._dof_pos[ids].clone(), task._dof_vel[ids].clone()
        anchor, direction = relocation_ray(old, goals, roots, root_rotations,
                                          self.initial_direction[ids], self.mode)
        # Plan all moves before touching the simulator. Increase distance along
        # the same requested ray if a candidate intersects a person or a box.
        planned = task._box_states[ids].clone()
        sizes = task._box_size[ids]
        geometry = task._push_reset_geometry()
        actual_distance = torch.full_like(anchor[..., 0], self.distance)
        for owner in range(task.num_agents):
            candidate = old[:, owner].clone()
            candidate[:, 7:13] = 0
            for attempt in range(41):
                candidate[:, :2] = anchor[:, owner]+actual_distance[:, owner, None]*direction[:, owner]
                unsafe = torch.zeros(len(ids), dtype=torch.bool, device=task.device)
                for person in range(task.num_agents):
                    unsafe |= geometry.invalid(task._rigid_body_pos[ids, person],
                        task._rigid_body_rot[ids, person], candidate, sizes[:, owner])
                # Circumspheres conservatively enclose tilted boxes in XY.
                radii = sizes.norm(dim=-1)/2
                separation = (candidate[:, None, :2]-planned[..., :2]).norm(dim=-1)
                overlap = separation < radii[:, owner, None]+radii+.10
                overlap[:, owner] = False
                unsafe |= overlap.any(-1)
                if not unsafe.any():
                    break
                actual_distance[unsafe, owner] += .5
            else:
                raise RuntimeError('Cannot place the box on the requested ray without overlap')
            planned[:, owner] = candidate
        task._box_states[ids, :task.num_agents] = planned[:, :task.num_agents]
        if len(ids):
            actors = task._box_actor_ids[ids, :task.num_agents].contiguous().view(-1)
            task.gym.set_actor_root_state_tensor_indexed(task.sim,
                gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(actors), len(actors))
            task._prev_box_pos[ids] = task._reward_box_values(task._box_states, ids)[..., :3]
            if hasattr(task, '_push_motion_metrics'):
                task._push_motion_metrics.reset(ids)
        self.refresh_relations(ids)
        assert torch.equal(task._tar_pos[ids], goals), 'Box switching changed a goal'
        assert (torch.equal(task._humanoid_root_states[ids], humanoids)
                and torch.equal(task._dof_pos[ids], dof_pos)
                and torch.equal(task._dof_vel[ids], dof_vel)), 'Box switching reset a humanoid'
        assert torch.isfinite(task.obs_buf).all(), 'Nonfinite observation after box switching'
        self.event = dict(mode=self.mode, switch_step=self.step, switch_seconds=self.step*task.dt,
            requested_distance_m=self.distance, actual_distance_m=actual_distance.tolist(),
            switched_scenes=len(ids), env_ids=ids.tolist(), goals=goals.tolist(),
            old_boxes=old.tolist(), new_boxes=planned[:, :task.num_agents].tolist(),
            agent_roots=roots.tolist(), agent_root_rotations=root_rotations.tolist(),
            goals_unchanged=True, humanoids_unchanged=True, placement_overlap=False)
        print('[PUSH box switch]', self.mode, 'at step', self.step, 'scenes', len(ids), flush=True)
