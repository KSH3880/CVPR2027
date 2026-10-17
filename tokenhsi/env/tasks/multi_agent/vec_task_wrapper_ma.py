# Stage1 wrapper: one scene observation and M action/reward/done rows per env.
#
# Observations and rewards are already per-row inside the task; only the done flag is
# per env (the episode is shared) and gets broadcast here. The per-agent `terminate`
# flag stays per row so that GAE bootstraps correctly for agents that were reset by a
# teammate falling rather than by falling themselves.

import torch

from env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from env.tasks.multi_agent.scene_features import clip_scene_observation


class VecTaskPythonWrapperMA(VecTaskPythonWrapper):

    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        self.num_agents = task.num_agents
        return

    def get_number_of_agents(self):
        return self.num_agents

    def _policy_observation(self):
        if getattr(self.task, "_stage2", False):
            obs = self.task.obs_buf
            views = obs.reshape(-1, obs.shape[-1]//self.num_agents)
            return clip_scene_observation(views, self.num_agents, self.task.num_objects,
                self.task.get_scene_entity_sizes(), self.clip_obs).reshape_as(obs).to(self.rl_device)
        return clip_scene_observation(self.task.obs_buf, self.num_agents,
            self.task.num_objects, self.task.get_scene_entity_sizes(), self.clip_obs).to(self.rl_device)

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
