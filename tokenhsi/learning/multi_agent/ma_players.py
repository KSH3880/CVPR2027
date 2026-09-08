# Test-time player for the multi-agent relation-transformer policy.

import os

from rl_games.algos_torch import torch_ext

import learning.amp_players as amp_players
from learning.multi_agent.ma_agent import EntityRunningMeanStd
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


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
