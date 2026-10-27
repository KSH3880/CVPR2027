import os
import torch
import learning.amp_players as amp_players
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from env.tasks.multi_agent.collision_reward import check_agent_collision_checkpoint
from utils.torch_utils import load_checkpoint


class MAPlayerContinuous(amp_players.AMPPlayerContinuous):

    def _preproc_amp_obs(self, amp_obs):
        normalized = super()._preproc_amp_obs(amp_obs)
        from utils.unified_training import preserve_amp_labels

        normalized = preserve_amp_labels(
            amp_obs, normalized, self.env.task._num_amp_obs_steps, getattr(self.env.task, "_amp_num_families", 3)
        )
        return normalized

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
        task = self.env.task
        if getattr(task, "_video_enabled", False):
            out = task.cfg["args"].output_path
            task.request_video(os.path.join(out, "videos", "test.mp4"))
        return super().run()

    def restore(self, fn):
        if fn != "Base":
            checkpoint = load_checkpoint(fn, self.device)
            check_checkpoint_metadata(
                checkpoint, checkpoint_metadata(self.env.task._relation_cfg)
            )
            check_agent_collision_checkpoint(checkpoint, self.env.task.cfg["env"])
            if getattr(self.env.task, "_stage2", False):
                from utils.stage2_spec import check_checkpoint
                check_checkpoint(checkpoint, self.env.task.cfg["env"])
        return super().restore(fn)

    def _build_net(self, config):
        super()._build_net(config)
        if self.normalize_input:
            task = self.env.task
            self.running_mean_std = SceneRunningMeanStd(
                entity_sizes=task.get_scene_entity_sizes(),
                entity_counts=[task.num_agents, task.num_objects, task.num_agents],
                normalized_sizes=task.get_scene_normalized_entity_sizes(),
                kinematic_size=task.get_scene_kinematic_size(),
                extra_passthrough_size=task.get_relation_suffix_size(),
                num_views=task.num_agents if getattr(task, "_stage2", False) and not getattr(task, "_shared_support", False) and not getattr(task, "_before", False) else 1,
            ).to(self.device)
            self.running_mean_std.eval()
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if hasattr(self, "env"):
            task = self.env.task
            config["num_agents"] = task.num_agents
            config["num_objects"] = task.num_objects
            config["humanoid_obs_size"] = task.get_humanoid_obs_size()
            config["object_obs_size"] = task.get_object_obs_size()
            config["goal_obs_size"] = task.get_goal_obs_size()
            config["observation_mode"] = task.get_policy_obs_mode()
            config["relation_reward_mode"] = task._relation_cfg["mode"]
            config["relation_graph_spec"] = task._relation_graph_spec
            config["scene_entity_sizes"] = task.get_scene_entity_sizes()
            config["scene_kinematic_size"] = task.get_scene_kinematic_size()
            config["scene_arena_scale"] = task.get_scene_arena_scale()
            config["device"] = self.device
        return config

    def get_batch_size(self, obses, batch_size):
        scene_batch = super().get_batch_size(obses, batch_size)
        if hasattr(self, "env"):
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
        if self._reward_debug_step % self._reward_print_interval != 0 and (
            not info["terminate"][0]
        ):
            return
        task = self.env.task
        terms = info["reward_terms"][0].detach().cpu().tolist()
        amp_reward = self._calc_amp_rewards(info["amp_obs"][0:1])["disc_rewards"][
            0, 0
        ].item()
        combined = self._task_reward_w * terms[-1] + self._disc_reward_w * amp_reward
        values = dict(zip(self.env.task.REWARD_TERM_NAMES, terms))
        print(
            "reward step={} {} amp={:.4f} combined={:.4f}".format(
                self._reward_debug_step, values, amp_reward, combined
            ),
            flush=True,
        )
        return

    @torch.no_grad()
    def run_eval(self):
        task = self.env.task
        from learning.multi_agent.edge_context_eval import run_edge_context_eval

        return run_edge_context_eval(self)
