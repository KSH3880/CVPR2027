# PPO/AMP agent for the multi-agent relation-transformer policy.
#
# Almost everything multi-agent is handled by rl_games itself: the env reports
# num_agents = M and emits one row per (env, agent) in env-major order, so values,
# advantages, GAE, log-probs, per-agent task rewards and per-agent AMP style rewards all
# come out per agent with no change to the PPO loop.
#
# Only two things need fixing up here:
#   1. observation normalisation must be shared across entity slots, otherwise the shared
#      tokenizers see the same physical quantity on different scales depending on which
#      slot it landed in;
#   2. the eps-greedy mask in AMPAgent is sized by num_envs, which is no longer the batch.

import numpy as np
import torch
import torch.nn as nn

from isaacgym.torch_utils import to_torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.amp_agent as amp_agent


class EntityRunningMeanStd(nn.Module):
    """Running normalisation that is shared across all slots of the same entity type.

    The observation row is entity-blocked -- [M humanoids | O objects | M goals] --
    so every block is folded into the batch dimension before being normalised. This keeps
    the statistics permutation-invariant (ego and non-ego humanoids share one estimate)
    and independent of M/O, which lets a policy run with different entity counts.
    """

    def __init__(self, entity_sizes, num_agents, num_objects=None):
        super().__init__()
        self.entity_sizes = list(entity_sizes)
        self.num_agents = num_agents
        self.num_objects = num_agents if num_objects is None else num_objects
        self.entity_counts = [self.num_agents, self.num_objects, self.num_agents]
        self.running_mean_std = nn.ModuleList([RunningMeanStd((sz,)) for sz in self.entity_sizes])
        return

    def forward(self, input, denorm=False, mask=None):
        B = input.shape[0]
        out = []
        offset = 0
        for rms, size, count in zip(self.running_mean_std, self.entity_sizes, self.entity_counts):
            width = count * size
            block = input[:, offset:offset + width].reshape(B * count, size)
            block = rms(block, denorm)
            out.append(block.reshape(B, width))
            offset += width

        assert offset == input.shape[1], \
            "obs row is {} wide, entity blocks cover {}".format(input.shape[1], offset)

        return torch.cat(out, dim=-1)


class MAAgent(amp_agent.AMPAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        task = self.vec_env.env.task
        if self.normalize_input:
            self.running_mean_std = EntityRunningMeanStd(
                entity_sizes=[task.get_humanoid_obs_size(),
                              task.get_object_obs_size(),
                              task.get_goal_obs_size()],
                num_agents=task.num_agents,
                num_objects=task.num_objects,
            ).to(self.ppo_device)
        return

    def _build_net_config(self):
        config = super()._build_net_config()

        task = self.vec_env.env.task
        config["num_agents"] = task.num_agents
        config["num_objects"] = task.num_objects
        config["humanoid_obs_size"] = task.get_humanoid_obs_size()
        config["object_obs_size"] = task.get_object_obs_size()
        config["goal_obs_size"] = task.get_goal_obs_size()
        config["device"] = self.ppo_device
        return config

    def _build_rand_action_probs(self):
        # one row per (env, agent), not per env
        num_rows = self.vec_env.env.task.num_envs * self.vec_env.env.task.num_agents
        row_ids = to_torch(np.arange(num_rows), dtype=torch.float32, device=self.ppo_device)

        self._rand_action_probs = 1.0 - torch.exp(10 * (row_ids / (num_rows - 1.0) - 1.0))
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0

        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        task = self.vec_env.env.task
        means = task.consume_reward_term_means()
        if means is not None:
            train_info["reward_term_means"] = means
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        disc_reward_mean = train_info["disc_rewards"].mean()
        self.writer.add_scalar("reward_terms/amp", disc_reward_mean.item(), frame)

        means = train_info.get("reward_term_means")
        if means is not None:
            for name, value in zip(self.vec_env.env.task.REWARD_TERM_NAMES, means):
                self.writer.add_scalar("reward_terms/{}".format(name), value.item(), frame)
            combined = self._task_reward_w * means[-1] + self._disc_reward_w * disc_reward_mean
            self.writer.add_scalar("reward_terms/combined", combined.item(), frame)
        return
