import json
import os
import time

import torch

from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.config import set_seed


class Stage1SequencePlayer(MAPlayerContinuous):
    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        output = task.cfg['args'].output_path
        os.makedirs(output, exist_ok=True)
        results = []
        for repeat in range(task.cfg['args'].eval_repeats):
            set_seed(task.sequence_seed + repeat)
            obs = self.env_reset()
            self.get_batch_size(obs['obs'], 1)
            initial = dict(boxes=task._box_states[0].tolist(),
                           box_sizes=task._box_size[0].tolist(),
                           humans=task._humanoid_root_states[0].tolist(),
                           dof_pos=task._dof_pos[0].tolist(), dof_vel=task._dof_vel[0].tolist(),
                           goals=task._tar_pos[0].tolist())
            reason = 'timeout'
            started = time.monotonic()
            for step in range(task.max_episode_length):
                obs = self.env_reset([])
                action = self.get_action(obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Non-finite sequence policy action')
                obs, _, done, info = self.env_step(self.env, action)
                if task.viewer:
                    task.render()
                    time.sleep(max(0., (step + 1) * task.dt - (time.monotonic() - started)))
                if task.sequence.finished():
                    reason = 'completed'
                    break
                if done.any():
                    reason = 'fall' if info['terminate'].any() else 'timeout'
                    break
            result = dict(repeat=repeat, mode=task.sequence_mode, graph_sampling=task.sequence_sampling,
                          seed=task.sequence_seed + repeat,
                          reason=reason, steps=step + 1, dt=task.dt,
                          hold_steps=task.sequence.hold_steps, plan=task.sequence_plan,
                          stage=(task.sequence.stage + 1).tolist(),
                          completed=task.sequence.completed.tolist(), events=task.sequence_events,
                          initial=initial, second_goals=task.sequence_goals.tolist(),
                          final_boxes=task._box_states[0].tolist(),
                          final_humans=task._humanoid_root_states[0].tolist())
            results.append(result)
            with open(os.path.join(output, 'sequence_results.json'), 'w') as handle:
                json.dump(results, handle, indent=2)
            print('[sequence result] repeat={} reason={} steps={} completed={}'.format(
                repeat, reason, step + 1, result['completed']), flush=True)
        if task.viewer:
            task.gym.write_viewer_image_to_file(task.viewer, os.path.join(output, 'final.png'))
        if task.viewer and os.environ.get('KEEP_OPEN', '1') != '0':
            print('[sequence] final scene held; close viewer or Ctrl+C to exit', flush=True)
            while True:
                task.render()
                time.sleep(.03)
        return results
