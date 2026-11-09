"""Student-only evaluation, reporting final and ever success separately."""
import json
import os
from pathlib import Path
import torch
from learning.transformer.trans_players import TransPlayerContinuous
from learning.native_eval_warmup import run_warmup, warmup_steps


class PushKDPlayer(TransPlayerContinuous):
    def env_reset(self, env_ids=None):
        # Native eval: original TokenHSI resets every env when env_ids is None (humanoid.py reset),
        # but MixedReferenceEnv.reset(None) resets nothing, so each repeat would continue the previous one.
        if env_ids is None and self._eval and self._eval_task != 'push' and not getattr(self.env, 'cvpr_single', False):
            env_ids = torch.arange(self.env.num_envs, device=self.device)
        return super().env_reset(env_ids)

    def run_eval(self):
        if getattr(self.env, 'cvpr_single', False):
            from learning.cvpr2027_eval import run_single_eval
            return run_single_eval(self)
        if self._eval_task != 'push':
            steps = warmup_steps()
            if steps > 0:
                run_warmup(self, steps)
            return super().run_eval()
        count = self.env.num_envs
        ref = self.env.reference
        collected = torch.zeros(count, dtype=torch.bool, device=self.device)
        success = torch.zeros_like(collected)
        ever, fell = success.clone(), success.clone()
        # Requested PUSH criterion (EVALUATION.md): own box centre within 0.2 m of its goal in 3D,
        # no height tolerance or contact condition. Same box/goal pairing as the policy observation
        # (reference_projection.project: row = scene*num_agents + owner -> _box_states/_tar_pos[scene, owner]).
        region_final, region_ever = success.clone(), success.clone()
        push_row = success.clone()
        final_distance = torch.zeros(count, device=self.device)
        final_distance_3d = torch.zeros(count, device=self.device)
        final_step = torch.zeros(count, dtype=torch.long, device=self.device)
        # Behaviour diagnostics (do not change any success metric): where each own box started, how close it
        # got, how far and in which direction it moved, and how often the hands met the reward's CONTACT test.
        start_distance_3d = torch.zeros(count, device=self.device)
        min_distance_3d = torch.full((count,), float('inf'), device=self.device)
        start_box_xy = torch.zeros(count, 2, device=self.device)
        start_goal_xy = torch.zeros(count, 2, device=self.device)
        final_box_xy = torch.zeros(count, 2, device=self.device)
        contact_steps = torch.zeros(count, device=self.device)
        episode_steps = torch.zeros(count, device=self.device)
        done_ids = []
        obs = self.env_reset()
        # as in TransPlayerContinuous.run_eval; without it get_action adds an extra batch dim (crash)
        self.get_batch_size(obs['obs'], 1)
        for step in range(1, 1202):
            obs = self.env_reset(done_ids)
            action = self.get_action(obs, True)
            obs, _, done, info = self.env_step(self.env, action)
            distance_3d = (ref._box_states[:, :ref.num_agents, :3]-ref._tar_pos[..., :3]).norm(dim=-1).flatten().to(self.device)
            box_xy = ref._box_states[:, :ref.num_agents, :2].reshape(-1, 2).to(self.device)
            hits = getattr(ref, '_push_hits', torch.zeros(ref.num_envs, ref.num_agents, dtype=torch.bool)).flatten().to(self.device)
            in_region = distance_3d <= REGION_DISTANCE
            open_rows = ~collected
            if step == 1:
                start_distance_3d[:] = distance_3d
                start_box_xy[:] = box_xy
                start_goal_xy[:] = ref._tar_pos[..., :2].reshape(-1, 2).to(self.device)
            min_distance_3d[open_rows] = torch.minimum(min_distance_3d[open_rows], distance_3d[open_rows])
            contact_steps[open_rows] += hits[open_rows].float()
            episode_steps[open_rows] += 1
            region_ever[open_rows] |= in_region[open_rows]
            push_row[open_rows] |= info['push'][open_rows]
            fresh = done.bool() & ~collected
            success[fresh] = info['success'][fresh].bool()
            ever[fresh] = info['push_success_ever'][fresh]
            fell[fresh] = info['terminate'][fresh].bool()
            final_distance[fresh] = info['precision'][fresh]
            final_distance_3d[fresh] = distance_3d[fresh]
            region_final[fresh] = in_region[fresh]
            final_box_xy[fresh] = box_xy[fresh]
            final_step[fresh] = step
            collected |= fresh
            done_ids = done.nonzero().flatten()
            if collected.all():
                break
        if not collected.all():
            raise RuntimeError('Incomplete first-episode PUSH evaluation')
        rows = push_row
        rate = lambda x: x[rows].float().mean().item()
        median = lambda x: x[rows].median().item()
        displacement = final_box_xy - start_box_xy
        to_goal = start_goal_xy - start_box_xy
        direction = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        progress = (displacement * direction).sum(-1)                     # + toward the goal, - away
        lateral = (displacement - progress[:, None] * direction).norm(dim=-1)
        moved = displacement.norm(dim=-1)
        contact_rate = contact_steps / episode_steps.clamp_min(1)
        diagnostics = dict(start_distance_3d_median=median(start_distance_3d), min_distance_3d_median=median(min_distance_3d),
                           box_moved_median=median(moved), progress_toward_goal_median=median(progress),
                           progress_positive=rate(progress > 0), progress_over_1m=rate(progress > 1.0),
                           moved_under_0p5m=rate(moved < 0.5), lateral_median=median(lateral),
                           progress_ratio_median=median(progress / to_goal.norm(dim=-1).clamp_min(1e-6)),
                           contact_ever=rate(contact_steps > 0), contact_step_rate_mean=contact_rate[rows].mean().item(),
                           episode_steps_median=median(episode_steps))
        report = dict(task='push', episodes=int(rows.sum()), rows_total=count, non_push_rows=int((~rows).sum()),
                      deterministic=True, horizon=1200, seed=self.config.get('seed') if hasattr(self, 'config') else None,
                      region_distance_3d=REGION_DISTANCE,
                      final_success=rate(region_final), ever_success=rate(region_ever),
                      native_final_success=rate(success), native_ever_success=rate(ever),
                      native_success_rule='reference AT success: distance 0.2 m and height error <= 0.001 m',
                      falls=rate(fell),
                      final_distance_3d_median=final_distance_3d[rows].median().item(),
                      final_distance_3d_mean=final_distance_3d[rows].mean().item(),
                      final_distance_xy_median=final_distance[rows].median().item(),
                      scene=f'reference_{ref.num_agents}humans_{ref.num_objects}boxes', checkpoint=self._checkpoint_fn, diagnostics=diagnostics)
        output = Path(os.environ.get('PUSH_EVAL_OUT') or (self.config.get('eval_output', 'output_etc/check/push_eval.json') if hasattr(self, 'config') else 'output_etc/check/push_eval.json'))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        per_row = dict(push_row=push_row, region_final=region_final, region_ever=region_ever,
                       native_final=success, native_ever=ever, fell=fell, final_step=final_step,
                       final_distance_3d=final_distance_3d, final_distance_xy=final_distance,
                       start_distance_3d=start_distance_3d, min_distance_3d=min_distance_3d, box_moved=moved,
                       progress_toward_goal=progress, lateral=lateral, contact_steps=contact_steps, episode_steps=episode_steps)
        output.with_suffix('.rows.json').write_text(json.dumps({k: v.tolist() for k, v in per_row.items()}))
        print('PUSH_EVAL '+json.dumps(report), flush=True)


REGION_DISTANCE = 0.2
