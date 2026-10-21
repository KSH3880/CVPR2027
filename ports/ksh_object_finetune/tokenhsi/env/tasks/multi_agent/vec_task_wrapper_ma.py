# VecTask wrapper for multi-agent tasks.
#
# legacy_multirow exposes M observations per env. clean_scene exposes one policy
# observation per env while actions/rewards/dones remain env-major / agent-minor.
# MAAgent owns the one-scene-to-M-agent rollout mapping in the latter mode.
#
# Observations and rewards are already per-row inside the task; only the done flag is
# per env (the episode is shared) and gets broadcast here. The per-agent `terminate`
# flag stays per row so that GAE bootstraps correctly for agents that were reset by a
# teammate falling rather than by falling themselves.

from gym import spaces
import numpy as np
import torch

from env.tasks.vec_task_wrappers import VecTaskPythonWrapper


class VecTaskPythonWrapperMA(VecTaskPythonWrapper):

    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        self.num_agents = task.num_agents
        return

    def get_number_of_agents(self):
        return self.num_agents

    def _policy_observation(self):
        obs = self.task.obs_buf
        if getattr(self.task, '_state_relation', False):
            # Semantic truth/history, the constant target and algebraic GTA poses
            # must not be corrupted by an optional flat observation clip.
            if not np.isinf(self.clip_obs):
                sizes = self.task.get_scene_entity_sizes()
                width = self.num_agents * sizes[0] + self.task.num_objects * sizes[1]
                obs = torch.cat([torch.clamp(obs[:, :width], -self.clip_obs, self.clip_obs),
                                 obs[:, width:]], -1)
            return obs.to(self.rl_device)
        return torch.clamp(obs, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)

        obs = self._policy_observation()
        rew = self.task.rew_buf.to(self.rl_device)
        done = self.task.reset_buf.repeat_interleave(self.num_agents).to(self.rl_device)

        return obs, rew, done, self.task.extras

    def reset(self, env_ids=None):
        # rl_games hands us `all_done_indices[::num_agents]`, i.e. ROW indices (env * M + 0),
        # one per finished env. The task resets whole envs, so map them back to env indices.
        if env_ids is not None and len(env_ids) > 0 and self.num_agents > 1:
            env_ids = torch.div(env_ids, self.num_agents, rounding_mode='floor').unique()

        self.task.reset(env_ids)
        return self._policy_observation()
