import argparse
import os
import runpy
import sys
from pathlib import Path

import toy_joint_carry_amp as toy
import numpy as np
import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import batched
from env.tasks.multi_agent.edge_context_reward import owner_sum


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    torch.manual_seed(4242)
    np.random.seed(4242)
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    n, m = task.num_envs, task.num_agents
    initial_box = task._box_states[:, 0, :3].clone()
    initial_roots = task._humanoid_root_states.clone()
    physical_hash = toy.digest([task._humanoid_root_states, task._box_states,
                                task._dof_pos, task._dof_vel])
    original_target = task._tar_pos.clone()
    reference = os.environ.get('TOY_DIRECTION_FILE')
    if reference:
        previous = np.load(reference)
        delta = previous['box'][59, :, :2] - previous['box'][0, :, :2]
        length = np.linalg.norm(delta, axis=-1)
        if (length < .05).any():
            raise RuntimeError('Cannot determine motion direction for every scene')
        direction = torch.tensor(delta / length[:, None], device=task.device)
        task._tar_pos[..., :2] = initial_box[:, None, :2] + 1.5 * direction[:, None]
    ids = torch.arange(n, device=task.device)
    task._reset_relation_history(ids)
    task._compute_observations(ids)
    obs = player.obs_to_torch(player.env._policy_observation())
    target = task._tar_pos[:, 0].clone()
    graph = task.relation_runtime.graph
    relation = batched(graph.edge_relation, n)
    masks = {'holding_reward': relation == HOLDING, 'at_reward': relation == AT}
    alive = torch.ones(n, device=task.device, dtype=torch.bool)
    record = {}
    replay_path = os.environ.get('TOY_REPLAY_FILE')
    replay = np.load(replay_path)['action'] if replay_path else None

    def save(name, value):
        record.setdefault(name, []).append(value.detach().cpu().numpy().copy())

    for step in range(180):
        if replay is None:
            action = player.get_action(obs, True)
        else:
            action = torch.tensor(replay[step], device=task.device).reshape(n*m, -1)
        obs, reward, done, info = player.env_step(player.env, action)
        obs = player.obs_to_torch(obs)
        runtime = task.relation_runtime
        result = runtime.last_result
        terms = info['reward_terms'].reshape(n, m, -1)
        amp = player._calc_amp_rewards(info['amp_obs'])['disc_rewards'].reshape(n, m)
        task_reward = terms[..., -1]
        save('distance', (task._box_states[:, 0, :2] - target[:, :2]).norm(dim=-1))
        save('holding', runtime.coupled_holding)
        save('alive', alive)
        save('body', task._rigid_body_pos - task._env_origins[:, None, None])
        save('box', task._box_states[:, 0, :3] - task._env_origins)
        save('root_facing', toy.heading(task._humanoid_root_states[..., 3:7]))
        save('reward_terms', terms)
        save('amp', amp)
        save('combined', player._task_reward_w * task_reward + player._disc_reward_w * amp)
        save('action', action.reshape(n, m, -1))
        for name, mask in masks.items():
            save(name, owner_sum(result['total'] * mask, graph))
        for name in ['state_component', 'progress_component', 'success_component']:
            save('at_' + name, owner_sum(result[name] * masks['at_reward'], graph))
        save('alignment', info['joint_alignment']['reward'])
        save('alignment_leader', info['joint_alignment']['leader'])
        save('alignment_saturated', info['joint_alignment']['saturated'])
        if not torch.allclose(terms[..., :-1].sum(-1), task_reward, atol=1e-5):
            raise RuntimeError('Reward components do not sum to total')
        if not torch.allclose(reward.reshape(n, m), task_reward, atol=1e-5):
            raise RuntimeError('Logged reward differs from environment reward')
        at_reward = owner_sum(result['total'] * masks['at_reward'], graph)
        if not torch.allclose(at_reward[:, 0], at_reward[:, 1], atol=1e-6):
            raise RuntimeError('Shared AT reward differs across humans')
        alive &= ~done.reshape(n, m).bool().any(-1)

    output = Path(task.cfg['args'].output_path)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.stack(value) for key, value in record.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Nonfinite rollout')
    np.savez_compressed(output / 'reward_traces.npz', **arrays,
        target=(target-task._env_origins).cpu().numpy(),
        initial_box=(initial_box-task._env_origins).cpu().numpy(),
        initial_facing=toy.heading(initial_roots[..., 3:7]).cpu().numpy(),
        original_target=(original_target-task._env_origins[:, None]).cpu().numpy(),
        reward_names=np.array(task.REWARD_TERM_NAMES), physical_hash=np.array(physical_hash))
    print('GOAL_REWARD_CHECK', physical_hash, 'finite, component sum and shared AT reward passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--toy-amp-weight', type=float, default=.5)
    parser.add_argument('--toy-updates', type=int, default=100)
    options, remaining = parser.parse_known_args()
    toy.install(options)
    toy.MAPlayerContinuous.run_eval = evaluate
    sys.argv = [str(toy.ROOT / 'tokenhsi/run.py')] + remaining
    runpy.run_path(str(toy.ROOT / 'tokenhsi/run.py'), run_name='__main__')
