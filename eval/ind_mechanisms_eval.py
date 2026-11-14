from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
import isaacgym
import numpy as np
import torch
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from env.tasks.multi_agent import scene_observation
from learning.multi_agent import formal_eval
from utils import stage2_eval_sizes, stage2_spec
from utils.coupled_push_spec import physical_box_count
from utils.task_role_spec import task_graph_packet
from ind_mechanisms_common import SCT, digest, write
from stage1_protocol_policy import Stage1Policy
import stage2_xy100
import stage2_full_observation

SCENE_PATH = Path(os.environ['IND_SCENE'])
SCENE = torch.load(SCENE_PATH, map_location='cpu', weights_only=False)
original_validate = stage2_spec.validate_env


def validate(env):
    m = env['numAgents']
    if m not in (1, 2) or env['numObjects'] != 2*m:
        raise ValueError('Mechanism evaluation requires 1H/2H and M=2N')
    equivalent = deepcopy(env)
    equivalent.update(numAgents=4, numObjects=8)
    equivalent['stage2JointPush']['physical_boxes'] = physical_box_count(4, 8)
    original_validate(equivalent)


def bank(sizes, masses, env, preset):
    if sizes.shape[0] != len(SCENE['roots']):
        raise ValueError('Snapshot/environment count mismatch')
    rows = torch.arange(len(sizes), device=sizes.device)[:, None]
    mapping = SCENE['mapping'].to(sizes.device)
    sizes[rows, mapping] = SCENE['sizes'].to(sizes.device)
    masses[rows, mapping] = (SCENE['sizes'].prod(-1)*SCENE['density']).to(sizes.device)
    groups = SCENE['groups'].to(sizes.device)
    return torch.zeros(len(sizes), device=sizes.device, dtype=torch.long), groups, torch.full_like(groups, -1), SCENE['templates'].to(sizes.device)


def reset_scene(self, ids):
    if not len(ids):
        return
    if not hasattr(self, '_ind_scene'):
        self._ind_scene = {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in SCENE.items()}
    s = self._ind_scene
    if not torch.equal(self._env_origins[ids], s['origins'][ids]):
        raise ValueError('Environment origins changed')
    self._sample_episode_graph(ids)
    packet = task_graph_packet(self.relation_runtime.graph, self.num_envs).reshape(self.num_envs, self.num_agents, 5)
    if not torch.equal(packet[ids].long(), s['packet'][ids]):
        raise ValueError('Restored task graph differs')
    for target, key in ((self._humanoid_root_states, 'roots'), (self._dof_pos, 'dof_pos'),
                        (self._dof_vel, 'dof_vel'), (self._kinematic_humanoid_rigid_body_states, 'kinematic'),
                        (self._tar_pos, 'goals'), (self._every_env_init_dof_pos, 'dof_pos')):
        target[ids] = s[key][ids]
    self._center_tar_pos[ids] = s['goals'][ids]
    self._box_states[ids] = 0
    self._box_states[ids, :, 6] = 1
    self._box_states[ids, :, 0] = self._env_origins[ids, 0, None]+1000+12*torch.arange(self.num_physical_objects, device=self.device)
    self._box_states[ids, :, 1] = self._env_origins[ids, 1, None]+1000
    row, logical = s['used'][ids].nonzero(as_tuple=True)
    self._box_states[ids[row], self._logical_box_order[ids[row], logical]] = s['boxes'][ids[row], logical]
    self._ghost_state, self._ghost_mask = s['boxes'], ~s['used']
    self._reset_default_slots = self._expand_slots(ids)
    self._reset_ref_slots, self._reset_ref_motion_ids, self._reset_ref_motion_times = {}, {}, {}
    self._platform_pos[ids] = self._platform_default_pos[ids]
    self._reset_env_tensors(ids)
    self._refresh_sim_tensors()
    self._reset_relation_history(ids)
    self._compute_observations(ids)
    self._init_amp_obs(ids)


def observed(task, boxes, ids):
    if not hasattr(task, '_ghost_state'):
        return boxes
    mask, state = task._ghost_mask, task._ghost_state
    if ids is not None:
        mask, state = mask[ids], state[ids]
    return torch.where(mask[..., None], state, boxes)


original_obs = scene_observation.SceneObservationMixin._compute_clean_scene_obs
code = textwrap.dedent(inspect.getsource(original_obs))
for argument, ids in (('self._box_states', 'None'), ('self._box_states, env_ids', 'env_ids')):
    anchor = 'box_states = self._logical_box_values('+argument+')'
    if code.count(anchor) != 1:
        raise ValueError('Observation builder changed')
    code = code.replace(anchor, 'box_states = _ind_observed(self, self._logical_box_values('+argument+'), '+ids+')')
namespace = dict(original_obs.__globals__, _ind_observed=observed)
exec(compile(code, __file__, 'exec'), namespace)
scene_observation.SceneObservationMixin._compute_clean_scene_obs = namespace['_compute_clean_scene_obs']
stage2_spec.validate_env = validate
stage2_eval_sizes.prepare_intersection_bank = bank
HumanoidMAStage2._reset_ontop_context_envs = reset_scene
stage2_xy100.install()


def check_weights(player, checkpoint):
    for current, expected in ((player.model.state_dict(), checkpoint['model']),
                              (player.running_mean_std.state_dict(), checkpoint['running_mean_std'])):
        if current.keys() != expected.keys() or any(not torch.equal(current[k].cpu(), v) for k,v in expected.items()):
            raise ValueError('Checkpoint weights/normalization changed')


def proximity(task):
    bodies = task._rigid_body_pos
    near = torch.cdist(bodies[:, 0], bodies[:, 1]).flatten(1).min(-1).values
    boxes = task._assigned_box_values(task._box_states)[:, [1, 0]]
    sizes = task._assigned_box_values(task._box_size)[:, [1, 0]]
    from isaacgym.torch_utils import quat_rotate_inverse
    offset = bodies-boxes[:, :, None, :3]
    quat = boxes[:, :, None, 3:7].expand(-1, -1, bodies.shape[2], -1)
    local = quat_rotate_inverse(quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(offset)
    distance = (local.abs()-sizes[:, :, None]/2).clamp_min(0).norm(dim=-1).min(-1).values
    return near, distance


original_eval = formal_eval.run_formal_eval


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    task.cfg['env']['episodeLength'] = int(os.environ.get('IND_STEPS', '900'))
    old_reset, old_action, old_step = player.env_reset, player.get_action, player.env_step
    checkpoint = torch.load(task.cfg['args'].checkpoint, map_location='cpu', weights_only=False)
    check_weights(player, checkpoint)
    if checkpoint['epoch'] != 2000:
        raise ValueError('Expected V4 epoch 2000')
    native = None
    first = None
    raw = {'human_body_point_distance':[], 'other_box_surface_distance':[], 'root_velocity':[]}

    def reset(*args, **kwargs):
        nonlocal native, first
        obs = old_reset(*args, **kwargs)
        first = obs['obs'].cpu().clone()
        if 'expected_observation' in SCENE and not torch.equal(first, SCENE['expected_observation']):
            delta = (first-SCENE['expected_observation']).abs()
            torch.save(dict(actual=first, expected=SCENE['expected_observation'], delta=delta), path/'input_mismatch.pt')
            raise ValueError('Extracted first observation differs: '+str(float(delta.max())))
        torch.save(first, path/'first_policy_input.pt')
        if os.environ.get('IND_MODEL') == 'sct':
            native = Stage1Policy(player, SCT, obs['obs'])
        actual_size = task._logical_box_values(task._box_size).cpu()
        if not torch.equal(actual_size, SCENE['sizes']):
            raise ValueError('Physical asset sizes changed')
        masses = []
        for row in range(task.num_envs):
            values = []
            for logical in range(task.num_objects):
                physical = int(task._logical_box_order[row, logical])
                handle = task._box_handles[row*task.num_physical_objects+physical]
                values.append(task.gym.get_actor_rigid_body_properties(task.envs[row], handle)[0].mass)
            masses.append(values)
        expected_mass = SCENE['sizes'].prod(-1)*SCENE['density']
        if not torch.allclose(torch.tensor(masses), expected_mass, rtol=2e-5, atol=1e-5):
            raise ValueError('Physical mass differs from source')
        write(path/'initial_verification.json', dict(scene_sha256=digest(SCENE_PATH),
            source_observation_exact='expected_observation' in SCENE, asset_sizes_exact=True,
            asset_mass_matches=True, first_input_sha256=__import__('hashlib').sha256(first.numpy().tobytes()).hexdigest()))
        return obs

    action_count = 0

    def action(obs, *args, **kwargs):
        nonlocal action_count
        if action_count == 0 and not torch.equal(obs['obs'].cpu(), first):
            raise ValueError('First action received a different observation')
        action_count += 1
        return old_action(obs, *args, **kwargs)

    def step(*args, **kwargs):
        result = old_step(*args, **kwargs)
        raw['root_velocity'].append(task._humanoid_root_states[..., 7:10].cpu().numpy().copy())
        if task.num_agents == 2:
            human, box = proximity(task)
            raw['human_body_point_distance'].append(human.cpu().numpy())
            raw['other_box_surface_distance'].append(box.cpu().numpy())
        return result

    player.env_reset, player.get_action, player.env_step = reset, action, step
    result = original_eval(player)
    if native:
        write(path/'stage1_policy.json', native.finish())
    else:
        check_weights(player, checkpoint)
    if action_count != result['simulated_steps']:
        raise ValueError('Unexpected policy call count')
    np.savez_compressed(path/'mechanisms.npz', **{k:np.stack(v) for k,v in raw.items() if v})
    result.update(xy_extent_percent=100, mechanism_scene=str(SCENE_PATH), model=os.environ.get('IND_MODEL', 'ours'),
        scene_sha256=digest(SCENE_PATH), checkpoint_sha256=digest(SCT if native else task.cfg['args'].checkpoint),
        policy_preserved=True, actual_first_input_verified=True,
        collision_metric='body-point/surface proximity, not pairwise contact',
        peak_torch_mib=torch.cuda.max_memory_allocated()/1024**2)
    write(path/'result.json', result)
    return result


formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')

