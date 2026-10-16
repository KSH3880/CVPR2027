import hashlib
import json
from pathlib import Path

import torch
import yaml


def dataset_digest(records):
    entries = [(Path(c['output']).name, c['output_sha256']) for c in records]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def validate_ablation(env, train=None):
    grasp = env.get('jointCarryGraspView')
    if grasp is not None:
        from learning.multi_agent.grasp_view import GRASP_VIEW
        if grasp != GRASP_VIEW or env.get('jointCarryAblation', {}).get('policy') != 'human_sa4_actor_critic' or (env['numAgents'], env['numObjects']) != (2, 4):
            raise ValueError('Grasp-view contract requires two-human SA4')
    spec = env.get('jointCarryAblation')
    if spec is None:
        if train is not None and train['params']['network'].get('coordination', {}).get('mode') == 'stage1_head_only':
            raise ValueError('Head-only policy requires the joint-carry ablation contract')
        return
    from utils.mixed_carry_spec import MIXED_SAMPLER
    if env['relationGraph']['sampler'] != MIXED_SAMPLER or spec['policy'] not in ('task_ca', 'task_ca4', 'task_ca4_actor_critic', 'task_ca4_actor_critic_geometry', 'task_ca4_actor_critic_none_mask', 'human_sa4_actor_critic', 'stage1_head_only'):
        raise ValueError('Locomotion AMP ablation requires Mixed80 and an explicit policy')
    amp = spec['amp']
    if (amp['probabilities'] != {'original': .8, 'backward': .1, 'sideways': .1}
            or amp['sampling'] != 'uniform_clip_then_uniform_time'
            or not env.get('ampTaskConditioning')):
        raise ValueError('Locomotion AMP requires conditioned original/backward/sideways 80/10/10')
    if train is not None:
        expected = {'enabled': False, 'mode': 'stage1_head_only'} if spec['policy'] == 'stage1_head_only' else {
            'enabled': True, 'mode': 'task_coupled', 'grounding_hidden': 128,
            'num_heads': 2, 'shuffle_task_order': True}
        if spec['policy'] in ('task_ca4', 'task_ca4_actor_critic', 'task_ca4_actor_critic_geometry', 'task_ca4_actor_critic_none_mask'):
            expected.update(num_layers=4, feedforward_dim=128)
        if spec['policy'] in ('task_ca4_actor_critic', 'task_ca4_actor_critic_geometry', 'task_ca4_actor_critic_none_mask'):
            expected.update(head_input='ca_only', critic_ca=True)
        if spec['policy'] == 'task_ca4_actor_critic_geometry':
            expected.update(task_input='geometry')
        if spec['policy'] == 'task_ca4_actor_critic_none_mask':
            expected.update(mask_none=True)
        if spec['policy'] == 'human_sa4_actor_critic':
            expected = dict(enabled=True, mode='human_coupled', num_heads=2,
                shuffle_human_order=True, num_layers=4, feedforward_dim=128,
                head_input='ca_only', critic_ca=True)
            if grasp is not None:
                expected['grasp_view'] = True
        if train['params']['network'].get('coordination') != expected:
            raise ValueError('Joint-carry policy and network coordination config differ')


def check_ablation_checkpoint(weights, env):
    if weights.get('joint_carry_grasp_view') != env.get('jointCarryGraspView'):
        raise ValueError('Checkpoint grasp-view observation contract differs')
    if weights.get('joint_carry_loco_platform') != env.get('jointCarryRsi', {}).get('locoPlatform'):
        raise ValueError('Checkpoint joint loco platform differs')
    if weights.get('joint_carry_pickup_sampling') != env.get('jointCarryRsi', {}).get('pickupFrameSampling'):
        raise ValueError('Checkpoint joint pickup frame sampling differs')
    if weights.get('joint_carry_rsi_dataset') != env.get('jointCarryRsi', {}).get('dataset_sha256'):
        raise ValueError('Checkpoint joint-carry RSI dataset differs')
    if weights.get('joint_carry_goal_sampling') != env.get('jointCarryRsi', {}).get('goalSampling'):
        raise ValueError('Checkpoint joint-carry goal sampling differs')
    if weights.get('joint_carry_ablation') != env.get('jointCarryAblation'):
        raise ValueError('Checkpoint joint-carry AMP/policy ablation differs')


def load_amp_libraries(task):
    spec = task.cfg['env'].get('jointCarryAblation')
    if spec is None:
        return {}
    from utils.motion_lib import MotionLib
    path = Path(spec['amp']['motion_file'])
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if (manifest['validation_status'] != 'interface_and_short_physics_pass'
            or dataset_digest(manifest['clips']) != spec['amp']['dataset_sha256']):
        raise ValueError('Locomotion AMP dataset is unvalidated or differs from its config')
    for clip in manifest['clips']:
        if hashlib.sha256(Path(clip['output']).read_bytes()).hexdigest() != clip['output_sha256']:
            raise ValueError('Locomotion AMP clip changed: '+clip['output'])
    config = yaml.safe_load(path.read_text())['motions']
    expected = {group: {Path(c['output']).name for c in manifest['clips']
                       if c['source'].startswith('loco_reverse/' if group == 'backward' else 'sidewalk/')}
                for group in ('backward', 'sideways')}
    if set(config) != set(expected) or any(
            {c['file'] for c in config[group]} != expected[group]
            or len(config[group]) != len(expected[group])
            or any(c['weight'] != 1. for c in config[group]) for group in expected):
        raise ValueError('Locomotion AMP manifest must contain exactly the validated, equally weighted clips')
    return {group: MotionLib(str(path), group, task._dof_body_ids, task._dof_offsets,
                            task._key_body_ids.cpu().numpy(), task.device) for group in expected}


def sample_amp_sources(family):
    sources = torch.multinomial(torch.tensor([.8, .1, .1], device=family.device),
                                len(family), replacement=True)
    return torch.where(family == 0, sources, 0)
