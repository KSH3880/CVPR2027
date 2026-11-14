import json
import os
from pathlib import Path
import runpy
import time

from stage2_protocol_eval import ROOT, capture_initial
import numpy as np
import torch
from learning.multi_agent import formal_eval
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
import table2_sampling
from table2_metrics import Table2Metrics
from stage1_protocol_policy import Stage1Policy


setting = os.environ['TABLE2_SETTING']
stage2_scalability_layout.install()
stage2_xy100.install()
reference_only = os.environ.get('REFERENCE_ONLY') == '1'
if reference_only:
    formal_eval.run_formal_eval = capture_initial
else:
    import stage2_distractor_physics
table2_sampling.install(setting)
original_eval = formal_eval.run_formal_eval


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    reset, step, action = player.env_reset, player.env_step, player.get_action
    alive, tracker, policy = None, None, None
    hands, physical = [], []
    physical_sizes = None
    calls = dict(actor=0, critic=0)

    def encoder_hook(name):
        def check(module, args, kwargs):
            if kwargs.get('object_valid') is not None:
                raise ValueError('Table2 masked observed distractors')
            calls[name] += 1
        return check

    network = player.model.a2c_network
    hooks = [encoder.register_forward_pre_hook(encoder_hook(name), with_kwargs=True)
             for name, encoder in (('actor', network.actor_encoder), ('critic', network.critic_encoder))]

    def reset_and_record(*args, **kwargs):
        nonlocal alive
        obs = reset(*args, **kwargs)
        alive = torch.ones(task.num_envs, task.num_agents, dtype=torch.bool, device=task.device)
        return obs

    def native_action(obs, is_determenistic=False):
        nonlocal policy
        if os.environ.get('STAGE1_POLICY_CHECKPOINT') and policy is None:
            policy = Stage1Policy(player, os.environ['STAGE1_POLICY_CHECKPOINT'], obs['obs'])
        return action(obs, is_determenistic)

    def step_and_record(*args, **kwargs):
        nonlocal tracker, alive, physical_sizes
        value = step(*args, **kwargs)
        if tracker is None:
            tracker = Table2Metrics(torch.load(path/'initial.pt', map_location='cpu', weights_only=False),
                                    setting, task.device)
        boxes = task._evaluation_box_values(task._box_states)[..., :7]
        options = {}
        if setting.startswith('coupled_'):
            box = task._assigned_box_values(task._box_states)[:, 0, :7]
            hand = task._rigid_body_pos[:, :, task._key_body_ids[:2]]
            if physical_sizes is None:
                physical_sizes = task._assigned_box_values(task._box_size)[:, 0].clone()
            physical.append(box.cpu().numpy().copy())
            hands.append(hand.cpu().numpy().copy())
            options = dict(physical=box, physical_sizes=physical_sizes, hands=hand)
        tracker.update(task._humanoid_root_states[..., :3], boxes, alive, **options)
        alive &= ~task._terminate_buf.reshape(task.num_envs, task.num_agents).bool()
        return value

    player.env_reset = reset_and_record
    if not reference_only:
        player.env_step, player.get_action = step_and_record, native_action
    try:
        result = original_eval(player)
    finally:
        player.env_reset, player.env_step, player.get_action = reset, step, action
        for hook in hooks:
            hook.remove()
    if reference_only:
        return result
    metadata = dict(setting=setting, scene_mode='single_interaction_group', xy_extent_percent=100,
        distractor_attention='all_observed_tokens', encoder_attention_checks=calls,
        shared_box_initialization='group_leader_only', group_sampler='table2_fixed_'+setting)
    if policy is not None:
        proof = dict(policy.finish(), action_calls=tracker.step)
        (path/'stage1_policy.json').write_text(json.dumps(proof, indent=2))
        result.update(policy='stage1_26k', stage1_policy=proof)
    table = tracker.summary(task.dt)
    table.update(seed=task.cfg['args'].seed, num_agents=task.num_agents, num_envs=task.num_envs,
                 simulated_steps=tracker.step, policy='stage1_26k' if policy is not None else 'stage2_v2_2k')
    metric = ('Enabling Interaction Group Success Rate' if setting.startswith('enable_')
              else 'Coupled Interaction Group Success Rate')
    formal = result['interaction_group_scores']['region'][metric]
    if (formal['success'], formal['groups']) != (table['successes'], table['trials']):
        raise ValueError('Online Table2 success differs from main evaluation')
    np.savez_compressed(path/'table2_trials.npz', **tracker.arrays())
    if hands:
        archive = dict(hands=np.stack(hands), physical_boxes=np.stack(physical),
                       physical_sizes=physical_sizes.cpu().numpy())
        started = time.perf_counter()
        np.savez_compressed(path/'cooperation.pending.npz', **archive)
        (path/'cooperation.pending.npz').replace(path/'cooperation.npz')
        table.update(cooperation_cache_write_s=time.perf_counter()-started,
                     cooperation_cache_bytes=(path/'cooperation.npz').stat().st_size,
                     cooperation_cache_uncompressed_bytes=sum(value.nbytes for value in archive.values()))
    (path/'table2.json').write_text(json.dumps(table, indent=2))
    conditions = json.loads((path/'new_conditions.json').read_text())
    conditions.update(metadata)
    (path/'new_conditions.json').write_text(json.dumps(conditions, indent=2))
    result.update(metadata, table2=table, comparison_protocol='cvpr2027_table2_v1')
    (path/'result.json').write_text(json.dumps(result, indent=2))
    return result


formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
