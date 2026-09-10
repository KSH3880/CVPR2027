# Test-time player for the multi-agent relation-transformer policy.

import os
import json
import datetime
import torch

from rl_games.algos_torch import torch_ext

import learning.amp_players as amp_players
from learning.multi_agent.ma_agent import EntityRunningMeanStd
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, LEGACY_MODE
from utils.torch_utils import load_checkpoint


class MAPlayerContinuous(amp_players.AMPPlayerContinuous):

    def __init__(self, config):
        super().__init__(config)
        debug_cfg = self.env.task.cfg["env"].get("debug", {})
        self._reward_debug = debug_cfg.get("reward", False)
        self._reward_print_interval = max(1, int(debug_cfg.get("rewardInterval", 30)))
        self._reward_debug_step = 0
        self._task_reward_w = config["task_reward_w"]
        self._disc_reward_w = config["disc_reward_w"]
        return

    def run(self):
        # record one clip of the checkpoint being tested, so a headless run still
        # produces something watchable
        task = self.env.task
        if getattr(task, "_video_enabled", False):
            out = task.cfg["args"].output_path
            task.request_video(os.path.join(out, "videos", "test.mp4"))
        return super().run()

    def restore(self, fn):
        if fn != 'Base':
            checkpoint = load_checkpoint(fn, self.device)
            check_checkpoint_metadata(checkpoint, checkpoint_metadata(self.env.task._relation_cfg))
        return super().restore(fn)

    def _build_net(self, config):
        super()._build_net(config)

        if self.normalize_input:
            task = self.env.task
            if task.is_scene_policy():
                self.running_mean_std = SceneRunningMeanStd(
                    entity_sizes=task.get_scene_entity_sizes(),
                    entity_counts=[task.num_agents, task.num_objects, task.num_agents],
                    normalized_sizes=task.get_scene_normalized_entity_sizes(),
                    kinematic_size=task.get_scene_kinematic_size(),
                    extra_passthrough_size=task.get_relation_suffix_size(),
                ).to(self.device)
            else:
                self.running_mean_std = EntityRunningMeanStd(
                    entity_sizes=[task.get_humanoid_obs_size(),
                                  task.get_object_obs_size(),
                                  task.get_goal_obs_size()],
                    num_agents=task.num_agents,
                    num_objects=task.num_objects,
                ).to(self.device)
            self.running_mean_std.eval()
        return

    def _build_net_config(self):
        config = super()._build_net_config()

        if hasattr(self, 'env'):
            task = self.env.task
            config["num_agents"] = task.num_agents
            config["num_objects"] = task.num_objects
            config["humanoid_obs_size"] = task.get_humanoid_obs_size()
            config["object_obs_size"] = task.get_object_obs_size()
            config["goal_obs_size"] = task.get_goal_obs_size()
            config["observation_mode"] = task.get_policy_obs_mode()
            config['relation_reward_mode'] = task._relation_cfg.get('mode', LEGACY_MODE)
            if task.is_scene_policy():
                config["scene_entity_sizes"] = task.get_scene_entity_sizes()
                config["scene_kinematic_size"] = task.get_scene_kinematic_size()
                config["scene_arena_scale"] = task.get_scene_arena_scale()
            config["device"] = self.device
        return config

    def get_batch_size(self, obses, batch_size):
        scene_batch = super().get_batch_size(obses, batch_size)
        if hasattr(self, 'env') and self.env.task.is_scene_policy():
            # Observation batch is N scenes, while reward/action/done accounting stays
            # at N*M agent slots in the stock player loop.
            self.batch_size = scene_batch * self.env.task.num_agents
            return self.batch_size
        return scene_batch

    def env_step(self, env, actions):
        result = super().env_step(env, actions)
        if self._eval:
            self._post_step(result[3])
        return result

    def _post_step(self, info):
        super()._post_step(info)
        if not self._reward_debug or "reward_terms" not in info:
            return

        self._reward_debug_step += 1
        if self._reward_debug_step % self._reward_print_interval != 0 and not info["terminate"][0]:
            return

        terms = info["reward_terms"][0].detach().cpu().tolist()
        amp_reward = self._calc_amp_rewards(info["amp_obs"][0:1])["disc_rewards"][0, 0].item()
        combined = self._task_reward_w * terms[-1] + self._disc_reward_w * amp_reward
        values = dict(zip(self.env.task.REWARD_TERM_NAMES, terms))
        if self.env.task._state_relation:
            print('reward step={} {} amp={:.4f} combined={:.4f}'.format(
                self._reward_debug_step, values, amp_reward, combined), flush=True)
            return
        print(
            "reward step={}: walk={:.4f} carry={:.4f} handheld={:.4f} "
            "putdown={:.4f} power={:.4f} collision={:.4f} total={:.4f} "
            "amp={:.4f} combined={:.4f} terminate={}".format(
                self._reward_debug_step,
                values["walk"], values["carry"], values["handheld"],
                values["putdown"], values["power"], values["collision"],
                values["total"], amp_reward, combined,
                int(info["terminate"][0].item())), flush=True)
        return

    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        if not task._state_relation:
            return super().run_eval()
        N, M = task.num_envs, task.num_agents
        results = {}
        for repeat in range(task.cfg['args'].eval_repeats):
            obs = self.env_reset()
            self.get_batch_size(obs['obs'], 1)
            alive = torch.ones(N, device=self.device, dtype=torch.bool)
            done_final = torch.zeros(N, M, device=self.device, dtype=torch.bool)
            current_final = torch.zeros_like(done_final)
            raw_ever = torch.zeros_like(done_final)
            raw_current = torch.zeros_like(done_final)
            first_h = torch.full((N, M), -1., device=self.device)
            first_valid = torch.full_like(first_h, -1.)
            terminated = torch.zeros(N, device=self.device, dtype=torch.bool)
            resets = []
            for step in range(task.max_episode_length + 1):
                obs = self.env_reset(resets)
                action = self.get_action(obs, self.is_determenistic)
                obs, _, done, info = self.env_step(self.env, action)
                _, diagnostic = task._evaluate_relations()
                raw_ever |= diagnostic['put'].bool() & alive[:, None]
                ending = done.reshape(N, M).bool().any(-1)
                collect = ending & alive
                done_final[collect] = info['subgoal_done'][collect]
                current_final[collect] = info['current_target_valid'][collect]
                raw_current[collect] = diagnostic['put'][collect].bool()
                first_h[collect] = task._relation_first_holding[collect]
                first_valid[collect] = task._relation_first_valid[collect]
                terminated[collect] = info['terminate'].reshape(N, M)[collect].bool().any(-1)
                alive &= ~ending
                resets = ending.nonzero(as_tuple=False).flatten() * M
                if not alive.any():
                    break
            if alive.any():
                raise RuntimeError('Evaluation exceeded task horizon without collecting every scene')
            def time_mean(value):
                valid = value[value >= 0]
                return valid.mean().item() if valid.numel() else None
            metrics = dict(num_scene_trials=N, num_agent_trials=N * M,
                agent_success_rate=done_final.float().mean().item(),
                scene_all_subgoals_done_rate=done_final.all(-1).float().mean().item(),
                scene_current_all_valid_rate=current_final.all(-1).float().mean().item(),
                geometric_put_ever_rate=raw_ever.float().mean().item(),
                geometric_put_final_rate=raw_current.float().mean().item(),
                first_holding_seconds_mean=time_mean(first_h),
                first_valid_success_seconds_mean=time_mean(first_valid),
                scene_termination_rate=terminated.float().mean().item())
            results['repeat_{}'.format(repeat)] = metrics
            print('[relation evaluation]', metrics, flush=True)
        directory = os.path.join(task.cfg['args'].output_path, 'metrics')
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'relation_{}.json'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print('Saved relation evaluation:', path)
