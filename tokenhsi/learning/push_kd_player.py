"""Student-only evaluation, reporting final and ever success separately."""
import json
from pathlib import Path
import torch
from learning.transformer.trans_players import TransPlayerContinuous


class PushKDPlayer(TransPlayerContinuous):
    def run_eval(self):
        if self._eval_task != 'push':
            return super().run_eval()
        count = self.env.num_envs
        collected = torch.zeros(count, dtype=torch.bool, device=self.device)
        success = torch.zeros_like(collected)
        ever, fell = success.clone(), success.clone()
        final_distance = torch.zeros(count, device=self.device)
        done_ids = []
        obs = self.env_reset()
        for _ in range(1201):
            obs = self.env_reset(done_ids)
            action = self.get_action(obs, True)
            obs, _, done, info = self.env_step(self.env, action)
            fresh = done.bool() & ~collected
            success[fresh] = info['success'][fresh].bool()
            ever[fresh] = info['push_success_ever'][fresh]
            fell[fresh] = info['terminate'][fresh].bool()
            final_distance[fresh] = info['precision'][fresh]
            collected |= fresh
            done_ids = done.nonzero().flatten()
            if collected.all():
                break
        if not collected.all():
            raise RuntimeError('Incomplete first-episode PUSH evaluation')
        report = dict(task='push', episodes=count, deterministic=True, horizon=1200,
                      success_distance=.2, height_tolerance=.001,
                      final_success=success.float().mean().item(), ever_success=ever.float().mean().item(),
                      falls=fell.float().mean().item(), final_distance_median=final_distance.median().item(),
                      scene='reference_2humans_4boxes', checkpoint=self._checkpoint_fn)
        output = Path(self.config.get('eval_output', 'output_etc/check/push_eval.json')) if hasattr(self, 'config') else Path('output_etc/check/push_eval.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print('PUSH_EVAL '+json.dumps(report), flush=True)
