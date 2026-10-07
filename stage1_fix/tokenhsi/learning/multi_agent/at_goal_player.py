import copy
import json
import os
import random
import time

import numpy as np
import torch
from learning.amp_players import AMPPlayerContinuous
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.box_cleanup_spec import cleanup_reward_config
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.torch_utils import load_checkpoint


class AtGoalPlayer(MAPlayerContinuous):
    def _post_step(self, info):
        pass

    def restore(self, path):
        weights = load_checkpoint(path, self.device)
        metadata = copy.deepcopy(weights['relation_metadata'])
        metadata['relation_reward_config'] = cleanup_reward_config(metadata['relation_reward_config'])
        check_checkpoint_metadata({'relation_metadata': metadata},
                                  checkpoint_metadata(self.env.task._relation_cfg))
        AMPPlayerContinuous.restore(self, path)
        print('[at goal] rescue checkpoint loaded', flush=True)

    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        args = task.cfg['args']
        heights = [float(value) for value in os.environ.get(
            'GOAL_HEIGHTS', '0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0').split(',')]
        modes = os.environ.get('GOAL_XY_MODES', 'above_box,midpoint,side').split(',')
        if not all(np.isfinite(height) and 0 <= height <= 2 for height in heights):
            raise ValueError('GOAL_HEIGHTS must contain heights between 0 and 2 metres')
        if not all(mode in ('above_box', 'midpoint', 'side') for mode in modes):
            raise ValueError('GOAL_XY_MODES must contain above_box,midpoint,side')
        os.makedirs(args.output_path, exist_ok=True)
        results = []
        for repeat in range(args.eval_repeats):
            for height in heights:
                for mode in modes:
                    seed = args.seed + repeat
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    task.goal_height, task.goal_xy_mode = height, mode
                    obs = self.env_reset()
                    self.get_batch_size(obs['obs'], 1)
                    initial_boxes = task._box_states[0, task.demo_boxes, :3].clone()
                    initial_humans = task._humanoid_root_states[0, :, :3].clone()
                    targets = task._tar_pos[0, task.demo_goals].clone()
                    print('[at goal] case={}/{} repeat={} xy={} center_z={:.1f}m'.format(
                        len(results) + 1, args.eval_repeats * len(heights) * len(modes),
                        repeat + 1, mode, height), flush=True)
                    reason = 'timeout'
                    started = time.monotonic()
                    for step in range(task.max_episode_length):
                        obs = self.env_reset([])
                        action = self.get_action(obs, True)
                        if not torch.isfinite(action).all():
                            raise RuntimeError('Non-finite AT demo policy action')
                        obs, _, done, info = self.env_step(self.env, action)
                        if task.viewer:
                            task.render()
                            time.sleep(max(0., (step + 1) * task.dt - (time.monotonic() - started)))
                        if done.any():
                            reason = 'fall' if info['terminate'].any() else 'timeout'
                            break
                    final_boxes = task._box_states[0, task.demo_boxes, :3]
                    if not torch.isfinite(task._box_states).all() or not torch.isfinite(task._humanoid_root_states).all():
                        raise RuntimeError('Non-finite AT demo physical state')
                    results.append(dict(repeat=repeat + 1, seed=seed, xy_mode=mode,
                        goal_center_z=height, reason=reason, steps=step + 1,
                        target_positions=targets.tolist(),
                        initial_box_positions=initial_boxes.tolist(),
                        initial_human_positions=initial_humans.tolist(),
                        final_box_positions=final_boxes.tolist(),
                        final_xyz_errors=(final_boxes - targets).norm(dim=-1).tolist()))
                    with open(os.path.join(args.output_path, 'at_goal_results.json'), 'w') as handle:
                        json.dump(results, handle, indent=2)
                    if task.viewer:
                        filename = 'case_{:03d}_{}_z{:.1f}.png'.format(len(results), mode, height)
                        task.gym.write_viewer_image_to_file(task.viewer, os.path.join(args.output_path, filename))
        if task.viewer and os.environ.get('KEEP_OPEN', '1') != '0':
            print('[at goal] final scene held; close viewer to exit', flush=True)
            while True:
                task.render()
                time.sleep(.03)
        return results
