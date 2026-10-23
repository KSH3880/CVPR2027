"""Zero-shot goal override for the original contact+AT warmstart evaluator.

The original YAML and checkpoint contract are checked normally. Only evaluation
reset goals change, after locomotion starts have been sampled and before goal
observations, markers and reward history are initialized.
"""
from isaacgym import gymapi  # Isaac Gym must precede torch.
import json
from pathlib import Path
import runpy
import torch
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush


def random_offsets(boxes, roots, generator):
    """XY +/-1.5m, 1..2m radius, excluding both initial people by 1m."""
    offsets = torch.zeros_like(boxes[..., :3])
    pending = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
    for _ in range(128):
        if not pending.any():
            return offsets
        xy = (torch.rand(offsets.shape[:-1] + (2,), device=boxes.device,
                         generator=generator) * 2 - 1) * 1.5
        goals = boxes[..., :2] + xy
        near_people = (goals[:, :, None] - roots[:, None, :, :2]).norm(dim=-1).amin(-1) < 1.
        radius = xy.norm(dim=-1)
        valid = (radius >= 1.) & (radius <= 2.) & ~near_people
        accept = pending & valid
        offsets[..., :2][accept] = xy[accept]
        pending &= ~accept
    raise RuntimeError('Cannot sample random goals outside box/person exclusions')


original_reset_task = HumanoidMAOpenPush._reset_task


def reset_random_goal(self, env_ids):
    args = self.cfg['args']
    if not (args.test and args.eval and args.task_graph == 'push'):
        raise ValueError('Random-goal viewer requires --test --eval --task_graph push')
    if self.cfg['env']['eval']['skill'] != ['loco']:
        raise ValueError('Zero-shot random goals require locomotion-only starts')
    if 'pushGoalRandomization' in self.cfg['env'] or self.cfg['env'].get('pushGoalContact', False):
        raise ValueError('Use the original straight-goal warmstart YAML')
    if not hasattr(self, '_zero_shot_goal_generator'):
        self._zero_shot_goal_generator = torch.Generator(device=self.device)
        self._zero_shot_goal_generator.manual_seed(args.seed + 1000003)
        self._zero_shot_goal_reset_count = 0
    boxes = self._push_reset_states[env_ids]
    roots = self._humanoid_root_states[env_ids]
    offsets = random_offsets(boxes, roots, self._zero_shot_goal_generator)
    self._push_goal_offsets[env_ids] = offsets
    original_reset_task(self, env_ids)
    assert torch.isfinite(self._tar_pos[env_ids]).all()
    assert (self.relation_runtime.tasks[env_ids] == 1).all()
    # Keep a replay/audit record of actual assignments and sampled world goals.
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(reset=self._zero_shot_goal_reset_count, env_ids=env_ids.cpu().tolist(),
                  box_binding=self._push_binding[env_ids].cpu().tolist(),
                  box_initial_xyz=boxes[..., :3].cpu().tolist(),
                  box_forward_xyz=self._push_directions[env_ids].cpu().tolist(),
                  goal_offsets_xyz=offsets.cpu().tolist(),
                  owner_goal_xyz=self._tar_pos[env_ids].cpu().tolist(),
                  humanoid_initial_xyz=roots[..., :3].cpu().tolist())
    with (out / 'random_goal_resets.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if self._zero_shot_goal_reset_count < 3:
        print('ZERO_SHOT_RANDOM_GOAL', json.dumps(record), flush=True)
    self._zero_shot_goal_reset_count += 1


original_update_camera = HumanoidMAOpenPush._update_camera


def update_goal_camera(self):
    if self.cfg['args'].task_camera == 'agent':
        return original_update_camera(self)
    # Fixed overview per episode: include both initial boxes, people and goals.
    points = torch.cat((self._push_reset_states[0, :, :2],
                        self._tar_pos[0, :, :2],
                        self._humanoid_root_states[0, :, :2]), dim=0)
    low, high = points.amin(0), points.amax(0)
    center = ((low + high) / 2).cpu().tolist()
    radius = max(7., float((high - low).amax()) * 1.1)
    self.gym.viewer_camera_look_at(
        self.viewer, None,
        gymapi.Vec3(center[0] + radius * .6, center[1] - radius * .8, radius * 1.2),
        gymapi.Vec3(center[0], center[1], .4))


if __name__ == '__main__':
    HumanoidMAOpenPush._reset_task = reset_random_goal
    HumanoidMAOpenPush._update_camera = update_goal_camera
    runpy.run_path('tokenhsi/run.py', run_name='__main__')
