"""Save complete, actually successful policy episodes for pose playback."""
import json
import os
from pathlib import Path

import numpy as np
import torch


def first_hits(success):
    """Frame0 is initialization; select genuine later goal arrivals only."""
    success = np.asarray(success, dtype=bool)
    if success.ndim != 2 or len(success) < 2:
        raise ValueError('Expected frames x agents, including the initial frame')
    hits = np.full(success.shape[1], -1, dtype=np.int64)
    for agent in range(success.shape[1]):
        frames = np.flatnonzero(success[1:, agent]) + 1
        if not success[0, agent] and len(frames):
            hits[agent] = frames[0]
    return hits


def install():
    from learning.multi_agent.ma_players import MAPlayerContinuous

    @torch.no_grad()
    def run(player):
        task = player.env.task
        if task.viewer is not None:
            raise ValueError('Collect success cases headlessly; replay uses a separate viewer')
        output = Path(task.cfg['args'].output_path).resolve()
        output.mkdir(parents=True, exist_ok=True)
        requested = int(os.environ.get('KSH_SUCCESS_COUNT', '6'))
        rounds = int(os.environ.get('KSH_SUCCESS_ROUNDS', '4'))
        deterministic = os.environ.get('KSH_SUCCESS_STOCHASTIC', '0') != '1'
        n, m = task.num_envs, task.num_agents
        actors = task._root_states.shape[0] // n
        horizon = task.max_episode_length
        cases = []
        trials = 0
        for repeat in range(rounds):
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], n)
            origins = task._env_origins.cpu().numpy().copy()
            sizes = task._box_size.cpu().numpy().copy()
            targets = task._tar_pos.cpu().numpy().copy()
            groups = task._groups.cpu().numpy().copy()
            assignments = task._agent_box_assignment.cpu().numpy().copy()
            interaction = task._interaction.cpu().numpy().copy()
            roots = np.empty((horizon + 1, n, actors, 13), np.float32)
            dofs = np.empty((horizon + 1, n, m, task.num_dof, 2), np.float32)
            bodies = np.empty((horizon + 1, n, m, task.num_bodies, 3), np.float32)
            success = np.zeros((horizon + 1, n, m), bool)
            alive = np.ones(n, bool)
            resets = []

            def record(frame, initial=False):
                roots[frame] = task._root_states.view(n, actors, 13).cpu().numpy()
                dofs[frame] = task._dof_state.view(n, m, task.num_dof, 2).cpu().numpy()
                body = (task._kinematic_humanoid_rigid_body_states[..., :3]
                        if initial else task._rigid_body_pos)
                bodies[frame] = body.cpu().numpy()
                success[frame] = task.relation_runtime.done.cpu().numpy()

            record(0, initial=True)
            for frame in range(1, horizon + 1):
                obs = player.env_reset(resets)
                action = player.get_action(obs, deterministic)
                obs, _, done, info = player.env_step(player.env, action)
                record(frame)
                ending = done.reshape(n, m).bool().any(-1).cpu().numpy()
                finished = np.flatnonzero(ending & alive)
                for env_id in finished:
                    trials += 1
                    flags = success[:frame + 1, env_id]
                    hits = first_hits(flags)
                    if (hits >= 0).any() and len(cases) < requested:
                        name = 'case_' + str(len(cases) + 1).zfill(3)
                        path = output / (name + '.npz')
                        np.savez_compressed(path,
                            roots=roots[:frame + 1, env_id],
                            dofs=dofs[:frame + 1, env_id],
                            bodies=bodies[:frame + 1, env_id],
                            success=flags, box_sizes=sizes[env_id],
                            targets=targets[env_id], groups=groups[env_id],
                            assignments=assignments[env_id],
                            interaction=interaction[env_id], origin=origins[env_id])
                        asset = task.cfg['env']['asset']
                        metadata = dict(file=str(path), repeat=repeat,
                            env_id=int(env_id), frames=frame + 1, dt=float(task.dt),
                            first_hit_frames=hits.tolist(),
                            ever_agents=np.flatnonzero(hits >= 0).tolist(),
                            final_agents=np.flatnonzero(flags[-1]).tolist(),
                            initial_rsi=interaction[env_id].tolist(),
                            deterministic=deterministic,
                            checkpoint=str(task.cfg['args'].checkpoint),
                            humanoid_asset=str((Path(asset['assetRoot']) /
                                asset['assetFileName']).resolve()),
                            success_definition='Official per-agent goal success at least once after initialization; not all-scene success')
                        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
                        cases.append(metadata)
                        print('KSH_SUCCESS_CASE', json.dumps(metadata), flush=True)
                alive[finished] = False
                resets = np.flatnonzero(ending).tolist()
                resets = torch.tensor(resets, device=player.device, dtype=torch.long) * m
                if frame % 100 == 0:
                    print('KSH_SUCCESS_PROGRESS', repeat, frame, int(alive.sum()), len(cases), flush=True)
                if not alive.any() or len(cases) >= requested:
                    break
            if alive.any() and len(cases) < requested:
                raise RuntimeError('Collection exceeded the original episode horizon')
            if len(cases) >= requested:
                break
        report = dict(cases=cases, completed_scene_trials=trials,
            requested_cases=requested, init_mode='balanced' if task._eval_push_rsi else 'loco',
            deterministic=deterministic, training_started=False)
        (output / 'cases.json').write_text(json.dumps(report, indent=2) + '\n')
        if not cases:
            raise RuntimeError('No successful episodes found; cases.json records the search conditions')
        print('KSH_SUCCESS_COLLECTION_DONE', len(cases), str(output / 'cases.json'), flush=True)
        return report

    MAPlayerContinuous.run = run
    MAPlayerContinuous.run_eval = run
