"""Evaluation-only goal changes; never reset a humanoid or a physical box."""
import math
import os

import torch


def rotated_goal(boxes, initial_direction, old_goals, angle_deg, distance):
    angle = math.radians(angle_deg)
    x, y = initial_direction.unbind(-1)
    direction = torch.stack((math.cos(angle)*x-math.sin(angle)*y,
                             math.sin(angle)*x+math.cos(angle)*y), -1)
    goals = old_goals.clone()
    goals[..., :2] = boxes[..., :2] + distance*direction
    return goals


class PushGoalSwitch:
    metric_key = 'goal_switch'
    result_prefix = 'push_goal_switch_'

    def __init__(self, task, angles, step, distance):
        if (task.__class__.__name__ != 'HumanoidMAFiveTask'
                or task._task_graph_preset != 'push_at' or not task._is_eval):
            raise ValueError('Goal switching requires five-task evaluation with TASK_GRAPH=push_at')
        if not angles or not all(math.isfinite(a) for a in angles):
            raise ValueError('PUSH_GOAL_SWITCH_ANGLES must contain finite angles')
        if not 0 < step < task.max_episode_length-1:
            raise ValueError('PUSH_GOAL_SWITCH_STEP must be inside the episode horizon')
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError('PUSH_GOAL_SWITCH_DISTANCE must be positive and finite')
        self.task, self.angles, self.step, self.distance = task, angles, step, distance

    @classmethod
    def from_environment(cls, task):
        value = os.environ.get('PUSH_GOAL_SWITCH_ANGLES', '')
        if not value:
            return None
        return cls(task, [float(v) for v in value.split(',')],
                   int(os.environ.get('PUSH_GOAL_SWITCH_STEP', '150')),
                   float(os.environ.get('PUSH_GOAL_SWITCH_DISTANCE', '2')))

    def begin(self, repeat):
        task = self.task
        self.angle = self.angles[repeat % len(self.angles)]
        direction = task._tar_pos[..., :2]-task._box_states[:, :task.num_agents, :2]
        self.initial_direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.switched = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        self.event = None

    def apply(self, alive):
        from isaacgym import gymtorch
        task = self.task
        ids = alive.nonzero(as_tuple=False).flatten()
        self.switched[ids] = True
        old = task._tar_pos[ids].clone()
        boxes = task._box_states[ids, :task.num_agents, :3].clone()
        new = rotated_goal(boxes, self.initial_direction[ids], old, self.angle, self.distance)
        task._tar_pos[ids] = new
        # Send only marker actors to Gym. Do not call the physical reset setter.
        if task._enable_markers and len(ids):
            task._marker_pos[ids] = new
            actors = task._marker_actor_ids[ids].contiguous().view(-1)
            task.gym.set_actor_root_state_tensor_indexed(task.sim,
                gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(actors), len(actors))
        self.refresh_relations(ids)
        self.event = dict(angle_deg=self.angle, switch_step=self.step,
            switch_seconds=self.step*task.dt, distance_m=self.distance,
            switched_scenes=len(ids), env_ids=ids.tolist(),
            box_positions=boxes.tolist(), old_goals=old.tolist(), new_goals=new.tolist())
        print('[PUSH goal switch]', self.angle, 'degrees at step', self.step,
              'scenes', len(ids), flush=True)

    def refresh_relations(self, ids):
        task = self.task
        if len(ids):
            task._push_hits[ids] = False
            # Subset evaluation/observations use reset-time kinematics. Use the
            # full live-state path here, then select the switched environments.
            phi, diag = task._evaluate_relations()
            task.relation_runtime.reset(ids, phi[ids], diag['z_error'][ids],
                diag['feet_height_error'][ids], region_error=diag['region_error'][ids])
            task._edge_ever_goal[ids] = False
            task._edge_ever_scene[ids] = False
            task._compute_observations()

    def metrics(self, final, ever, final_scene, ever_scene):
        mask = self.switched
        result = dict(self.event or dict(angle_deg=self.angle, switched_scenes=0))
        for key, values in [('final_agent_success', final), ('ever_agent_success', ever),
                            ('final_scene_success', final_scene), ('ever_scene_success', ever_scene)]:
            result[key] = values[mask].float().mean().item() if mask.any() else None
        return result
