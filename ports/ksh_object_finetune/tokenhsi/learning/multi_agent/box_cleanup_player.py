import copy
import json
import os
import time

import torch
from learning.amp_players import AMPPlayerContinuous
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.box_cleanup_spec import cleanup_reward_config
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.torch_utils import load_checkpoint


class BoxCleanupPlayer(MAPlayerContinuous):
    def _post_step(self, info):
        pass

    def restore(self, path):
        weights = load_checkpoint(path, self.device)
        metadata = copy.deepcopy(weights['relation_metadata'])
        metadata['relation_reward_config'] = cleanup_reward_config(metadata['relation_reward_config'])
        expected = checkpoint_metadata(self.env.task._relation_cfg)
        check_checkpoint_metadata({'relation_metadata': metadata}, expected)
        AMPPlayerContinuous.restore(self, path)
        print('[cleanup] rescue checkpoint loaded; 4 humans / 16 boxes / 8 edges', flush=True)

    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        output = task.cfg['args'].output_path
        os.makedirs(output, exist_ok=True)
        results = []
        for repeat in range(task.cfg['args'].eval_repeats):
            obs = self.env_reset()
            self.get_batch_size(obs['obs'], 1)
            reason = 'timeout'
            started = time.monotonic()
            for step in range(task.max_episode_length):
                obs = self.env_reset([])
                action = self.get_action(obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Non-finite cleanup policy action')
                obs, _, done, info = self.env_step(self.env, action)
                if task.viewer:
                    task.render()
                    time.sleep(max(0., (step + 1) * task.dt - (time.monotonic() - started)))
                    if step in (0, 120, 300, 600):
                        task.gym.write_viewer_image_to_file(task.viewer, os.path.join(output, 'frame_{}.png'.format(step)))
                if task.cleanup_finished:
                    reason = 'completed'
                    break
                if done.any():
                    reason = 'fall' if info['terminate'].any() else 'timeout'
                    break
            boxes = task._box_states[0, task.cleanup_round * 4:task.cleanup_round * 4 + 4]
            result = dict(repeat=repeat, reason=reason, steps=step + 1,
                          completed_boxes=int(task.cleanup_completed.sum()),
                          rounds_completed=2 if task.cleanup_finished else task.cleanup_round,
                          events=task.cleanup_events,
                          target_xy_errors=(boxes[:, :2] - task._tar_pos[0, :, :2]).norm(dim=-1).tolist(),
                          box_sizes=task._box_size[0].tolist(),
                          box_positions=task._box_states[0, :, :3].tolist(),
                          human_positions=task._humanoid_root_states[0, :, :3].tolist())
            results.append(result)
            with open(os.path.join(output, 'cleanup_results.json'), 'w') as handle:
                json.dump(results, handle, indent=2)
            print('[cleanup result]', json.dumps(result), flush=True)
        if task.viewer:
            task.gym.write_viewer_image_to_file(task.viewer, os.path.join(output, 'final.png'))
            if os.environ.get('KEEP_OPEN', '1') != '0':
                print('[cleanup] final scene held; close viewer to exit', flush=True)
                while True:
                    task.render()
                    time.sleep(.03)
        return results
