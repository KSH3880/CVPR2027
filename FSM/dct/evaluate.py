import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT/'FSM'), str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import isaacgym
import numpy as np
import torch

from common import write, MODELS
from learning.multi_agent import formal_eval

if os.environ.get('REFERENCE_ONLY') == '1':
    runpy.run_path(str(ROOT/'eval/table2_eval.py'), run_name='__main__')
    sys.exit(0)

base_eval = formal_eval.run_formal_eval
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
import table2_sampling
stage2_scalability_layout.install()
stage2_xy100.install()
table2_sampling.install(os.environ['TABLE2_SETTING'])
import stage2_distractor_physics as physics
from table2_metrics import Table2Metrics
from freeze import DelayFreeze, normalize_initial
from policy import load, check_weights


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    model, delay = os.environ['DCT_MODEL'], int(os.environ['DCT_DELAY'])
    setting = os.environ['TABLE2_SETTING']
    task.cfg['env']['episodeLength'] = int(os.environ.get('DCT_STEPS', '1050'))
    task.max_episode_length = task.cfg['env']['episodeLength']
    reset, step, action = player.env_reset, player.env_step, player.get_action
    controller = tracker = native = saved = None
    hands, physical, bodies, forces = [], [], [], []
    physical_sizes = None
    task._ghost_checks = 0
    initial_roots = None
    first = True

    def fresh():
        task._compute_observations()
        return player.obs_to_torch(player.env.reset(torch.empty(0, device=task.device, dtype=torch.long)))

    def reset_scene(*args, **kwargs):
        nonlocal controller, initial_roots
        obs = reset(*args, **kwargs)
        reference = physics.REFERENCE/f'general_seed{task.cfg["args"].seed}'
        expected = torch.load(reference/'observed_scene.pt', map_location='cpu', weights_only=False)
        if not torch.equal(obs['obs'].cpu(), expected):
            raise ValueError('Raw initial scene differs from shared reference')
        torch.save(obs['obs'].cpu(), path/'initial_before_normalization.pt')
        normalize_initial(task)
        controller = DelayFreeze(task, delay, model == 'fsm')
        initial_roots = task._humanoid_root_states[..., :3].clone()
        obs = fresh()
        torch.save(obs['obs'].cpu(), path/'common_first_observation.pt')
        physics.check_step(task)
        return obs

    def native_action(obs, is_determenistic=False):
        nonlocal native, saved, first
        if first:
            native, saved = load(player, model, obs['obs'])
        if controller.before_action():
            obs = fresh()
        if first:
            torch.save(obs['obs'].cpu(), path/'first_policy_input.pt')
            if not torch.equal(obs['obs'].cpu(), torch.load(path/'common_first_observation.pt', weights_only=False)):
                raise ValueError('First policy input changed after controller setup')
            first = False
        value = action(obs, is_determenistic).reshape(task.num_envs, task.num_agents, -1)
        return value.masked_fill(controller.frozen[..., None], 0).flatten(0, 1)

    def controlled_step(*args, **kwargs):
        nonlocal tracker, physical_sizes
        value = step(*args, **kwargs)
        controller.frame += 1
        if controller.frame == 1 and (task._humanoid_root_states[..., :3]-initial_roots).norm(dim=-1).max() > .5:
            raise ValueError('Initial physics state was lost')
        if tracker is None:
            tracker = Table2Metrics(torch.load(path/'initial.pt', map_location='cpu', weights_only=False), setting, task.device)
        controller.score_alive = controller.alive.clone()
        options = {}
        if setting.startswith('coupled'):
            box = task._assigned_box_values(task._box_states)[:, 0, :7]
            hand = task._rigid_body_pos[:, :, task._key_body_ids[:2]]
            if physical_sizes is None:
                physical_sizes = task._assigned_box_values(task._box_size)[:, 0].clone()
            physical.append(box.cpu().numpy().copy())
            hands.append(hand.cpu().numpy().copy())
            options = dict(physical=box, physical_sizes=physical_sizes, hands=hand)
        tracker.update(task._humanoid_root_states[..., :3], task._evaluation_box_values(task._box_states)[..., :7],
            controller.score_alive, **options)
        bodies.append(task._rigid_body_pos.cpu().numpy().copy())
        forces.append(task._contact_forces.cpu().numpy().copy())
        controller.alive &= ~task._terminate_buf.reshape(task.num_envs, task.num_agents).bool()
        physics.check_step(task)
        return value

    player.env_reset, player.env_step, player.get_action = reset_scene, controlled_step, native_action
    try:
        result = base_eval(player)
    finally:
        player.env_reset, player.env_step, player.get_action = reset, step, action
        freeze = controller.finish() if controller is not None else None
    if freeze['max_frozen_root_error'] or freeze['max_frozen_dof_error']:
        raise ValueError('Freeze projection failed')
    if native is not None:
        preservation = native.finish()
    else:
        check_weights(player, saved)
        preservation = dict(checkpoint=str(MODELS[model]), epoch=saved['epoch'],
            weights_unchanged=True, normalization_unchanged=True, native_checkpoint_architecture=True)
    table = tracker.summary(task.dt)
    metric = 'Enabling Interaction Group Success Rate' if setting.startswith('enable') else 'Coupled Interaction Group Success Rate'
    score = result['interaction_group_scores']['region'][metric]
    if (score['success'], score['groups']) != (table['successes'], table['trials']):
        raise ValueError('Formal/table score mismatch')
    np.savez_compressed(path/'table2_trials.npz', **tracker.arrays())
    np.savez_compressed(path/'freeze_trace.npz', **{k: np.asarray(v) for k,v in controller.trace.items()})
    np.savez_compressed(path/'body_trace.npz', positions=np.stack(bodies), contact_forces=np.stack(forces))
    if hands:
        np.savez_compressed(path/'cooperation.npz', hands=np.stack(hands), physical_boxes=np.stack(physical),
            physical_sizes=physical_sizes.cpu().numpy())
    table.update(model=model, delay_seconds=delay, simulated_steps=controller.frame, num_envs=task.num_envs,
        num_agents=task.num_agents, control_dt=task.dt, tcoop_status='use CPU protocol reaggregation')
    write(path/'table2.json', table)
    write(path/'freeze.json', freeze)
    write(path/'policy_verification.json', preservation)
    result.update(model=model, policy_checkpoint=str(MODELS[model]), delay_seconds=delay, table2=table, delay_freeze=freeze,
        raw_reference_exact=True, first_policy_input_matches_common_initial=True,
        physics_checked_frames=task._ghost_checks, initial_velocities='zero for all humans in every condition',
        distractor_attention='all_observed_tokens', distractor_physics='removed_from_arena')
    write(path/'result.json', result)
    return result


formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
