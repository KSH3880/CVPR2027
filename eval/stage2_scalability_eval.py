import json
import os
from pathlib import Path
import runpy

from stage2_protocol_eval import ROOT, capture_initial
import torch
from learning.multi_agent import formal_eval
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
from env.tasks.multi_agent.stage2_group_layout import layout_metadata


stage2_scalability_layout.install()
stage2_xy100.install()
if os.environ.get('STAGE2_SCENE_MODE', 'mixed') not in ('independent', 'mixed'):
    raise ValueError('STAGE2_SCENE_MODE must be independent or mixed')
if os.environ.get('REFERENCE_ONLY') == '1':
    formal_eval.run_formal_eval = capture_initial
else:
    import stage2_distractor_physics
import stage2_protocol_sampling

original_eval = formal_eval.run_formal_eval


def checked_eval(player):
    task = player.env.task
    original_reset = player.env_reset
    encoder_calls = dict(actor=0, critic=0)

    def encoder_hook(name):
        def check(module, args, kwargs):
            if kwargs.get('object_valid') is not None:
                raise ValueError('Observed distractors were masked in attention')
            encoder_calls[name] += 1
        return check

    network = player.model.a2c_network
    hooks = [encoder.register_forward_pre_hook(encoder_hook(name), with_kwargs=True)
             for name, encoder in (('actor', network.actor_encoder), ('critic', network.critic_encoder))]

    def reset(*args, **kwargs):
        obs = original_reset(*args, **kwargs)
        groups = task._groups
        same = groups[:, :, None] == groups[:, None, :]
        centers = torch.bmm(same.float(), task._humanoid_root_states[..., :2])/same.sum(-1, keepdim=True)
        minimum_gap = None
        if task.num_agents > 4:
            minimum_gap = layout_metadata(task)['minimum_initial_group_gap']
            mapping = task._protocol_token_groups
            logical = (task._ghost_state if hasattr(task, '_ghost_state')
                       else task._logical_box_values(task._box_states))
            target = task._group_spawn_targets.gather(1, mapping[..., None].expand(-1, -1, 2))
            distance = (logical[..., :2]-target).norm(dim=-1)
            if (distance[~task._stage2_task_slots] > 4.501).any():
                raise ValueError('Distractor outside its assigned group disc')
        if os.environ.get('STAGE2_SCENE_MODE', 'mixed') == 'independent':
            if (task._before_predecessor >= 0).any() or (same.sum(-1) != 1).any():
                raise ValueError('Independent scene contains relations')
        torch.save(dict(groups=groups.cpu(), token_groups=task._protocol_token_groups.cpu(),
                        group_centers=centers.cpu(), group_targets=task._group_spawn_targets.cpu()),
                   Path(task.cfg['args'].output_path)/'scalability_layout.pt')
        task._protocol_minimum_gap = minimum_gap
        return obs

    player.env_reset = reset
    result = original_eval(player)
    for hook in hooks:
        hook.remove()
    player.env_reset = original_reset
    path = Path(task.cfg['args'].output_path)
    metadata = dict(scene_mode=os.environ.get('STAGE2_SCENE_MODE', 'mixed'), xy_extent_percent=100,
        group_spawn_gap=getattr(task, '_group_spawn_gap', None) if task.num_agents > 4 else None,
        group_spawn_radius=getattr(task, '_group_spawn_radius', None) if task.num_agents > 4 else None,
        minimum_initial_group_gap=task._protocol_minimum_gap,
        distractor_group_budget='2k minus required tokens' if task.num_agents > 4 else None)
    metadata['shared_box_initialization'] = 'group_leader_only' if task.num_agents > 4 else 'native'
    metadata['group_sampler'] = ('all_independent_uniform_skills' if metadata['scene_mode'] == 'independent'
                                 else 'sequential_uniform_feasible_I_E_C')
    metadata['distractor_attention'] = 'all_observed_tokens'
    metadata['encoder_attention_checks'] = encoder_calls
    (path/'scalability_protocol.json').write_text(json.dumps(metadata, indent=2))
    if os.environ.get('REFERENCE_ONLY') != '1':
        result.update(metadata)
        result['comparison_protocol'] = 'cvpr2027_'+metadata['scene_mode']+'_v1'
        if task.num_agents > 4:
            result['group_spawn_over4']['unused_distractors'] = True
            result['group_spawn_over4']['sampler'] = metadata['group_sampler']
        conditions = json.loads((path/'new_conditions.json').read_text())
        conditions.update(scene_mode=metadata['scene_mode'], group_sampler=metadata['group_sampler'])
        (path/'new_conditions.json').write_text(json.dumps(conditions, indent=2))
        (path/'result.json').write_text(json.dumps(result, indent=2))
    return result


formal_eval.run_formal_eval = checked_eval
if os.environ.get('STAGE2_SAMPLING_AUDIT') == '1':
    import stage2_sampling_audit
runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
