import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml

from learning.multi_agent.ma_players import MAPlayerContinuous
from learning.amp_players import AMPPlayerContinuous
from utils.stage1_grasp_shift import shared_payload_views, initial_grasp_sides
from utils.edge_stage1_spec import STAGE1_CONTEXT_MODE
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.torch_utils import load_checkpoint


class Stage1GraspShiftPlayer(MAPlayerContinuous):
    def _build_net_config(self):
        config = super()._build_net_config()
        config['relation_reward_mode'] = STAGE1_CONTEXT_MODE
        return config

    def restore(self, path):
        with open('tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml') as handle:
            source = yaml.safe_load(handle)['env']
        check_checkpoint_metadata(load_checkpoint(path, self.device), checkpoint_metadata(source['relationReward']))
        AMPPlayerContinuous.restore(self, path)
        if self.model.a2c_network.stage2:
            raise ValueError('This toy requires the Stage1 checkpoint without Stage2 coordination')
        self.shift = float(os.environ.get('SHIFT', '1'))
        if not -1 <= self.shift <= 1:
            raise ValueError('SHIFT must be between -1 and 1')
        print('[grasp-shift] Shared payload/goal; two Stage1 views; anchor offset strength = {}'.format(self.shift))

    def get_action(self, obs, is_determenistic=False):
        task = self.env.task
        raw = obs['obs'] if isinstance(obs, dict) else obs
        shifted = shared_payload_views(raw, self.grasp_sides, self.shift,
            task._relation_cfg['joint_carry']['anchor_inset'])
        policy_obs = dict(obs, obs=shifted) if isinstance(obs, dict) else shifted
        actions = super().get_action(policy_obs, is_determenistic).reshape(len(raw), 2, 2, -1)
        return torch.stack((actions[:, 0, 0], actions[:, 1, 1]), 1).flatten(0, 1)

    def _post_step(self, info):
        pass

    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        output = task.cfg['args'].output_path
        os.makedirs(output, exist_ok=True)
        results = []
        capture = os.environ.get('SAVE_TRACE', '0') == '1'
        for repeat in range(task.cfg['args'].eval_repeats):
            obs = self.env_reset()
            self.get_batch_size(obs['obs'], 1)
            initial = obs['obs'].detach().cpu().numpy()
            self.grasp_sides = initial_grasp_sides(obs['obs'])
            views = shared_payload_views(obs['obs'], self.grasp_sides, self.shift)
            routing_error = None
            if capture:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    reference = MAPlayerContinuous.get_action(self, dict(obs), True)
                    strength = self.shift
                    self.shift = 0.
                    try:
                        routed = self.get_action(obs, True)
                    finally:
                        self.shift = strength
                    routing_error = (reference - routed).abs().max().item()
                    if routing_error > 1e-5:
                        raise RuntimeError('Two-view action routing differs from original Stage1')
            traces = []
            started = time.monotonic()
            for step in range(task.max_episode_length):
                obs = self.env_reset([])
                action = self.get_action(obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Non-finite action')
                if capture:
                    traces.append(dict(obs=obs['obs'].cpu().numpy().copy(),
                        action=action.cpu().numpy().copy(),
                        hands=task._rigid_body_pos[:, :, task._key_body_ids[:2]].cpu().numpy().copy()))
                obs, _, done, info = self.env_step(self.env, action)
                raw = obs['obs'] if isinstance(obs, dict) else obs
                if not torch.isfinite(raw).all():
                    raise RuntimeError('Non-finite observation')
                if task.viewer:
                    task.render()
                    time.sleep(max(0., (step + 1) * task.dt - (time.monotonic() - started)))
                    if os.environ.get('SAVE_FRAMES', '0') == '1' and step in (0, 60, 150):
                        task.gym.write_viewer_image_to_file(task.viewer,
                            os.path.join(output, 'repeat_{}_frame_{}.png'.format(repeat, step)))
                if done.any():
                    break
            results.append(dict(repeat=repeat, shift_strength=self.shift, steps=step + 1,
                task_packet=initial[:, -10:].reshape(-1, 2, 5).tolist(),
                grasp_sides=self.grasp_sides.cpu().tolist(),
                zero_shift_routing_error=routing_error,
                initial_object_poses=initial[:, 568:624].reshape(-1, 8, 7)[:, 2:6].tolist(),
                policy_view_object_poses=views[:, 568:624].reshape(-1, 8, 7)[:, 2:6].cpu().tolist(),
                initial_obs_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                terminated=bool(info['terminate'].any()), finite=True,
                final_objects=task._box_states.cpu().tolist()))
            if capture:
                np.savez_compressed(os.path.join(output, 'trace_{}.npz'.format(repeat)),
                    **{key: np.stack([item[key] for item in traces]) for key in traces[0]})
            with open(os.path.join(output, 'results.json'), 'w') as handle:
                json.dump(results, handle, indent=2)
            print('[grasp-shift]', {k: v for k, v in results[-1].items()
                if k not in ('final_objects', 'initial_object_poses', 'policy_view_object_poses')}, flush=True)
        if task.viewer and os.environ.get('KEEP_OPEN', '1') == '1':
            while not task.gym.query_viewer_has_closed(task.viewer):
                task.render()
                time.sleep(.03)
