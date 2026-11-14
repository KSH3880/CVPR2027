import inspect
import json
import os
from pathlib import Path
import textwrap

import torch
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from env.tasks.multi_agent import scene_observation
from learning.multi_agent import formal_eval

REFERENCE = Path(os.environ.get('STAGE2_REFERENCE_ROOT', str(Path(__file__).resolve().parents[1] / 'output_etc/check/stage2_new_protocol_1500_env512_20261108/reference')))


def observed_boxes(task, boxes, ids):
    if not hasattr(task, '_ghost_state'):
        return boxes
    mask = task._ghost_mask if ids is None else task._ghost_mask[ids]
    states = task._ghost_state if ids is None else task._ghost_state[ids]
    return torch.where(mask[..., None], states, boxes)


original_observation = scene_observation.SceneObservationMixin._compute_clean_scene_obs
source = textwrap.dedent(inspect.getsource(original_observation))
for argument, ids in (('self._box_states', 'None'), ('self._box_states, env_ids', 'env_ids')):
    anchor = f'box_states = self._logical_box_values({argument})'
    if source.count(anchor) != 1:
        raise ValueError('Observation builder changed')
    source = source.replace(anchor, f'box_states = _eval_observed_boxes(self, self._logical_box_values({argument}), {ids})')
namespace = dict(original_observation.__globals__, _eval_observed_boxes=observed_boxes)
exec(compile(source, __file__, 'exec'), namespace)
scene_observation.SceneObservationMixin._compute_clean_scene_obs = namespace['_compute_clean_scene_obs']
original_reset = HumanoidMAStage2._reset_env_tensors


def reset_physics_only(self, ids):
    valid = self._stage2_task_slots[ids]
    mapping = self._logical_box_order[ids]
    needed = torch.zeros(len(ids), self.num_physical_objects, device=self.device, dtype=torch.long)
    needed.scatter_add_(1, mapping, valid.long())
    row, logical = (~valid).nonzero(as_tuple=True)
    physical = mapping[row, logical]
    if (needed[row, physical] != 0).any():
        raise ValueError('Distractor shares a required physical object')
    before = self._logical_box_values(self._box_states, ids).clone()
    if not hasattr(self, '_ghost_state'):
        self._ghost_state = torch.zeros(self.num_envs, self.num_objects, 13, device=self.device)
        self._ghost_mask = torch.zeros(self.num_envs, self.num_objects, dtype=torch.bool, device=self.device)
        reference = REFERENCE
        snapshot = reference/f'general_seed{self.cfg["args"].seed}'/'initial.pt'
        self._reference_boxes = torch.load(snapshot, map_location='cpu')['boxes'].to(self.device)
    difference = (before-self._reference_boxes[ids]).abs()
    self._initial_roundtrip_error = float(difference.max())
    if not torch.allclose(before, self._reference_boxes[ids], atol=.001, rtol=0):
        torch.save(dict(boxes=before.cpu(), roots=self._humanoid_root_states[ids].cpu(),
            goals=self._tar_pos[ids].cpu(), groups=self._groups[ids].cpu(),
            templates=self._templates[ids].cpu(), mapping=mapping.cpu()),
            Path(self.cfg['args'].output_path)/'initial_mismatch.pt')
        raise ValueError(f'Sampled boxes differ from the reference initial scene: {self._initial_roundtrip_error}')
    self._ghost_state[ids] = self._reference_boxes[ids]
    self._ghost_mask[ids] = ~valid
    before_obs = self._compute_clean_scene_obs(ids)
    self._box_states[ids[row], physical, :2] = self._env_origins[ids[row], :2]+1000.+12*physical[:,None]
    self._box_states[ids[row], physical, 7:13] = 0.
    after = self._logical_box_values(self._box_states, ids)
    if not torch.equal(before[valid], after[valid]):
        raise ValueError('Required task/alias/region physical state changed')
    expected = torch.where(valid[...,None], before, self._reference_boxes[ids])
    if not torch.equal(expected, observed_boxes(self, after, ids)):
        raise ValueError('Logical observation states changed at reset')
    if not torch.equal(before_obs, self._compute_clean_scene_obs(ids)):
        raise ValueError('Full observation changed when parking distractors')
    return original_reset(self, ids)


HumanoidMAStage2._reset_env_tensors = reset_physics_only


def check_step(task):
    physical = task._logical_box_values(task._box_states)
    observed = observed_boxes(task, physical, None)
    mask = task._ghost_mask
    if not torch.equal(observed[~mask], physical[~mask]):
        raise ValueError('Task observation differs from physical task state')
    if not torch.equal(observed[mask], task._ghost_state[mask]):
        raise ValueError('Distractor observation changed')
    if ((physical[..., :2]-observed[..., :2]).norm(dim=-1)[mask] < 900).any():
        raise ValueError('Distractor is inside the physical arena')
    obs = task._compute_clean_scene_obs()
    m, o = task.num_agents, task.num_objects
    start = m*223+o*30+m
    poses = obs[:,start:start+(2*m+o)*7].reshape(task.num_envs,2*m+o,7)[:,m:m+o]
    expected = observed[...,:7].clone()
    expected[...,:3] -= task._env_origins[:,None]
    if not torch.equal(poses, expected):
        raise ValueError('GTA observed object pose mismatch')
    task._ghost_checks += 1


original_eval = formal_eval.run_formal_eval


def checked_eval(player):
    task = player.env.task
    reset, step = player.env_reset, player.env_step
    destination = Path(task.cfg['args'].output_path)
    task._ghost_checks = 0
    def reset_and_capture(*args, **kwargs):
        obs = reset(*args, **kwargs)
        reference = REFERENCE/f'general_seed{task.cfg["args"].seed}'
        original_obs = torch.load(reference/'observed_scene.pt', map_location='cpu')
        if not torch.equal(obs['obs'].cpu(), original_obs):
            raise ValueError('Initial full observation differs from the original sampled scene')
        physical = task._logical_box_values(task._box_states)
        torch.save(dict(mask=task._ghost_mask.cpu(), sampled=task._ghost_state.cpu(),
            physical=physical.cpu(), observed=observed_boxes(task, physical, None).cpu(),
            observed_scene=obs['obs'].cpu()), destination/'physics_only_reset.pt')
        check_step(task)
        return obs
    def step_and_check(*args, **kwargs):
        result = step(*args, **kwargs)
        check_step(task)
        return result
    player.env_reset, player.env_step = reset_and_capture, step_and_check
    try:
        result = original_eval(player)
    finally:
        player.env_reset, player.env_step = reset, step
    metadata = dict(checked_frames=task._ghost_checks, physical_removal_only=True,
        observed_distractor_pose_velocity_fixed_at_sample=True, all_object_tokens_retained=True,
        full_observation_unchanged_at_parking=True, task_alias_region_states_unchanged_by_hook=True,
        num_distractor_tokens=int(task._ghost_mask.sum()),
        raw_to_canonical_roundtrip_error=task._initial_roundtrip_error,
        note='Physics parked outside arena; sampled distractor pose/velocity retained only in scene observation. Reward and scoring use actual physical states.')
    (destination/'physics_only_verification.json').write_text(json.dumps(metadata, indent=2))
    result.update(distractor_physics='removed_from_arena', distractor_observation='sampled_pose_velocity_fixed')
    if (destination/'new_conditions.json').exists():
        conditions = json.loads((destination/'new_conditions.json').read_text())
        conditions['distractor_physics'] = 'removed_from_arena'
        (destination/'new_conditions.json').write_text(json.dumps(conditions, indent=2))
        result['enable_size_mode'] = 'common_intersection'
        result['comparison_protocol'] = 'cvpr2027_mixed_v1'
    (destination/'result.json').write_text(json.dumps(result, indent=2))
    return result


formal_eval.run_formal_eval = checked_eval
