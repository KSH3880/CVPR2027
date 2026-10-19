import json
import os
import time

import torch

from learning.amp_players import AMPPlayerContinuous
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.joint_carry_amp import check_ablation_checkpoint
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.torch_utils import load_checkpoint


class JointCarryDemoPlayer(MAPlayerContinuous):
    def _post_step(self, info):
        pass

    def restore(self, path):
        weights = load_checkpoint(path, self.device)
        source = self.env.task.demo_source_env
        check_ablation_checkpoint(weights, source)
        check_checkpoint_metadata(weights, checkpoint_metadata(source['relationReward']))
        AMPPlayerContinuous.restore(self, path)
        from learning.multi_agent.ca_none_mask_inference import configure_ca_none_mask
        self._ca_none_mask = configure_ca_none_mask(self)
        self._ca_entmax = os.environ.get('DEMO_CA_ENTMAX', '0') == '1'
        if self._ca_entmax:
            from learning.multi_agent.ca_entmax_inference import enable_ca_entmax
            enable_ca_entmax(self.model.a2c_network.coordination)
            print('[mixed-scene] Actor CA entmax alpha=1.5 enabled', flush=True)
        print('[mixed-scene] checkpoint loaded; {} humans / 8 objects / 1 env'.format(
            self.env.task.num_agents), flush=True)

    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        output = task.cfg['args'].output_path
        os.makedirs(output, exist_ok=True)
        results = []
        for repeat in range(task.cfg['args'].eval_repeats):
            obs = self.env_reset()
            self.get_batch_size(obs['obs'], 1)
            initial = dict(humans=task._humanoid_root_states[0, :, :3].tolist(),
                           objects=task._box_states[0, :, :3].tolist(),
                           goals=task._tar_pos[0].tolist(),
                           source_frames=task.demo_source_frames)
            started = time.monotonic()
            reason = 'timeout'
            for step in range(task.max_episode_length):
                obs = self.env_reset([])
                action = self.get_action(obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Non-finite mixed-scene action')
                obs, _, done, info = self.env_step(self.env, action)
                if not torch.isfinite(obs).all():
                    raise RuntimeError('Non-finite mixed-scene observation')
                if task.viewer:
                    task.render()
                    time.sleep(max(0., (step + 1) * task.dt - (time.monotonic() - started)))
                    if os.environ.get('SAVE_FRAMES', '0') == '1' and step in (0, 30, 150, 300):
                        path = os.path.join(output, 'repeat_{}_frame_{}.png'.format(repeat, step))
                        task.gym.write_viewer_image_to_file(task.viewer, path)
                if done.any():
                    reason = 'fall' if info['terminate'].any() else 'timeout'
                    break
            results.append(dict(repeat=repeat, steps=step + 1, reason=reason, initial=initial,
                actor_ca_attention='entmax15' if self._ca_entmax else 'softmax',
                actor_ca_none_mask=self._ca_none_mask,
                final_humans=task._humanoid_root_states[0, :, :3].tolist(),
                final_objects=task._box_states[0, :, :3].tolist()))
            with open(os.path.join(output, 'demo_results.json'), 'w') as handle:
                json.dump(results, handle, indent=2)
            print('[mixed-scene] repeat={} steps={} reason={}'.format(repeat, step + 1, reason), flush=True)
        if task.viewer and os.environ.get('KEEP_OPEN', '1') != '0':
            print('[mixed-scene] final scene held; close viewer to exit', flush=True)
            while not task.gym.query_viewer_has_closed(task.viewer):
                task.render()
                time.sleep(.03)
        return results
