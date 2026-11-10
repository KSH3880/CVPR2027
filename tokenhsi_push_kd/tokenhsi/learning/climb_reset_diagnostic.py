"""Opt-in repeated-reset CLIMB diagnostic; never replaces official first-episode scores."""
import os
import random
import json
from pathlib import Path
import numpy as np
import torch


def capture_rng(device):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None)


def restore_rng(state, device):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3], device)


def snapshot(player, dest):
    n = player.env.native
    def cpu(x): return x.detach().cpu().numpy().copy()
    data = dict(obs=cpu(player.env._obs()), root=cpu(n._root_states),
                dof_pos=cpu(n._dof_pos), dof_vel=cpu(n._dof_vel),
                body_pos=cpu(n._rigid_body_pos), body_rot=cpu(n._rigid_body_rot),
                body_vel=cpu(n._rigid_body_vel), contact=cpu(n._contact_forces),
                kinematic=cpu(n._kinematic_humanoid_rigid_body_states),
                goals=cpu(n._climb_tar_pos), task=cpu(n._task_indicator),
                progress=cpu(n.progress_buf), terminate=cpu(n._terminate_buf))
    np.savez_compressed(dest, **data)
    return data


@torch.no_grad()
def run_diagnostic(player):
    from learning.cvpr2027_eval import run_single_eval
    if player.env.reference.cvpr_task != 'climb':
        raise ValueError('CLIMB only')
    if getattr(player, 'is_rnn', False):
        raise ValueError('RNN reset must be implemented before diagnostic use')
    base = Path(os.environ['CVPR_EVAL_OUT'])
    base.mkdir(parents=True, exist_ok=True)
    n = player.env.native
    props = [n.gym.get_actor_dof_properties(e,h).copy() for e,h in zip(n.envs,n.humanoid_handles)]
    prestep = int(os.environ.get('CVPR_CLIMB_PRESTEP', '0'))
    if prestep not in (0, 1): raise ValueError('prestep must be0 or1')
    if prestep:
        snapshot(player, base/'cold_state.npz')
        player.env.step(torch.zeros((n.num_envs,32),device=player.device))
        if str(n.device).startswith('cuda'): torch.cuda.synchronize(n.device)
        restore_rng(player.env.diagnostic_reset_rng,n.device)
        n.reset(torch.arange(n.num_envs,device=n.device))
        player.env.reference.cvpr_observed_initial_boxes.copy_(player.env.reference._box_states)
    baseline = None; results=[]
    for episode in range(1 if prestep else 3):
        dest = base if episode == 0 else base/f'reset_episode_{episode+1}'
        dest.mkdir(parents=True, exist_ok=True)
        if episode:
            if str(n.device).startswith('cuda'): torch.cuda.synchronize(n.device)
            for e,h,p in zip(n.envs,n.humanoid_handles,props):
                n.gym.set_actor_dof_properties(e,h,p)
            restore_rng(player.env.diagnostic_reset_rng,n.device)
            n.reset(torch.arange(n.num_envs,device=n.device))
            player.env.reference.cvpr_observed_initial_boxes.copy_(player.env.reference._box_states)
        state = snapshot(player,dest/'reset_state.npz')
        if baseline is None: baseline=state
        differences={}
        for key,value in state.items():
            delta=np.abs(value.astype(float)-baseline[key].astype(float))
            differences[key]=dict(equal=bool(np.array_equal(value,baseline[key])),max_abs=float(delta.max()),mean_abs=float(delta.mean()))
        os.environ['CVPR_EVAL_OUT']=str(dest)
        run_single_eval(player)
        result=json.loads((dest/'result.json').read_text())
        result.update(diagnostic='one_zero_action_step_then_same_rng_reset' if prestep else 'same_rng_native_reset',episode_index=episode+1,
                      first_episode_only=episode==0 and not prestep,prior_control_steps=prestep+episode*900)
        (dest/'result.json').write_text(json.dumps(result,indent=2))
        results.append(dict(episode=episode+1,region=result['region'],falls=result['falls'],
                            reset_state_difference=differences))
        (base/'reset_comparison.json').write_text(json.dumps(results,indent=2))
        print('CLIMB_RESET_DIAGNOSTIC '+json.dumps(results[-1]),flush=True)
    os.environ['CVPR_EVAL_OUT']=str(base)
