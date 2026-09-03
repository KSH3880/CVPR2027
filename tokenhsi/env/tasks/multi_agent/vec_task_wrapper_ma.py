# VecTask wrapper for multi-agent tasks.
#
# Exposes M rows per env to rl_games, ordered env-major / agent-minor
# (row = env * M + agent). That ordering is what rl_games assumes when it does
# `all_done_indices[::num_agents]`, i.e. the agents of one env must be contiguous.
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

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)

        obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        rew = self.task.rew_buf.to(self.rl_device)
        done = self.task.reset_buf.repeat_interleave(self.num_agents).to(self.rl_device)

        return obs, rew, done, self.task.extras

    def reset(self, env_ids=None):
        # rl_games hands us `all_done_indices[::num_agents]`, i.e. ROW indices (env * M + 0),
        # one per finished env. The task resets whole envs, so map them back to env indices.
        if env_ids is not None and len(env_ids) > 0 and self.num_agents > 1:
            env_ids = torch.div(env_ids, self.num_agents, rounding_mode='floor').unique()

        self.task.reset(env_ids)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
