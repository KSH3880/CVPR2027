# PPO/AMP agent for the multi-agent relation-transformer policy.
#
# legacy_multirow delegates the standard N*M row layout to rl_games. clean_scene keeps
# observations at (T,N,scene) and groups all agent-specific PPO quantities as
# (T,N,M,...); only the loss-facing quantities are flattened to N*M after one scene
# forward. AMP observations remain per humanoid in both modes.

import time
import numpy as np
import torch
import torch.nn as nn

from isaacgym.torch_utils import to_torch
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.amp_agent as amp_agent
import learning.amp_datasets as amp_datasets
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


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
        self._scene_policy = task.is_scene_policy()
        if self.normalize_input:
            if self._scene_policy:
                self.running_mean_std = SceneRunningMeanStd(
                    entity_sizes=task.get_scene_entity_sizes(),
                    entity_counts=[task.num_agents, task.num_objects, task.num_agents],
                    normalized_sizes=task.get_scene_normalized_entity_sizes(),
                    kinematic_size=task.get_scene_kinematic_size(),
                ).to(self.ppo_device)
            else:
                self.running_mean_std = EntityRunningMeanStd(
                    entity_sizes=[task.get_humanoid_obs_size(),
                                  task.get_object_obs_size(),
                                  task.get_goal_obs_size()],
                    num_agents=task.num_agents,
                    num_objects=task.num_objects,
                ).to(self.ppo_device)

        if self._scene_policy:
            assert not self.is_rnn, "clean_scene currently supports the feed-forward PPO path only"
            assert self.minibatch_size % task.num_agents == 0, \
                "minibatch_size must be divisible by numAgents in clean_scene mode"
            scene_batch_size = self.batch_size // task.num_agents
            scene_minibatch_size = self.minibatch_size // task.num_agents
            self.dataset = amp_datasets.AMPDataset(
                scene_batch_size, scene_minibatch_size, self.is_discrete,
                self.is_rnn, self.ppo_device, self.seq_len)
        return

    def init_tensors(self):
        super().init_tensors()
        if self._scene_policy:
            # rl_games allocates observations at (T,N*M). Replace only the policy
            # observation tensors with true one-copy (T,N); all PPO quantities remain
            # per-agent in the standard buffers.
            shape = (self.horizon_length, self.num_actors) + tuple(self.obs_shape)
            self.experience_buffer.tensor_dict['obses'] = torch.zeros(
                shape, dtype=torch.float32, device=self.ppo_device)
            self.experience_buffer.tensor_dict['next_obses'] = torch.zeros(
                shape, dtype=torch.float32, device=self.ppo_device)
        return

    def _build_net_config(self):
        config = super()._build_net_config()

        task = self.vec_env.env.task
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
        config["device"] = self.ppo_device
        return config

    def _build_rand_action_probs(self):
        # one row per (env, agent), not per env
        num_rows = self.vec_env.env.task.num_envs * self.vec_env.env.task.num_agents
        row_ids = to_torch(np.arange(num_rows), dtype=torch.float32, device=self.ppo_device)

        if num_rows == 1:
            self._rand_action_probs = torch.ones_like(row_ids)
        else:
            self._rand_action_probs = 1.0 - torch.exp(
                10 * (row_ids / (num_rows - 1.0) - 1.0))
            self._rand_action_probs[0] = 1.0
            self._rand_action_probs[-1] = 0.0

        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0
        return

    def _group_agent_tensor(self, tensor):
        """(T,N*M,...) -> (N*T,M,...), keeping every scene together."""
        T = tensor.shape[0]
        tail = tensor.shape[2:]
        tensor = tensor.view(T, self.num_actors, self.num_agents, *tail)
        return tensor.transpose(0, 1).reshape(
            self.num_actors * T, self.num_agents, *tail)

    def _flatten_scene_tensor(self, tensor):
        """(T,N,...) -> (N*T,...), matching _group_agent_tensor scene order."""
        return tensor.transpose(0, 1).reshape(
            self.num_actors * tensor.shape[0], *tensor.shape[2:])

    def play_steps(self):
        if not self._scene_policy:
            return super().play_steps()

        self.set_eval()
        done_indices = []
        update_list = self.update_list

        for step_idx in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', step_idx, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            for key in update_list:
                self.experience_buffer.update_data(key, step_idx, res_dict[key])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', step_idx, shaped_rewards)
            self.experience_buffer.update_data('next_obses', step_idx, self.obs['obs'])
            self.experience_buffer.update_data('dones', step_idx, self.dones)
            self.experience_buffer.update_data('amp_obs', step_idx, infos['amp_obs'])
            self.experience_buffer.update_data('rand_action_mask', step_idx,
                                               res_dict['rand_action_mask'])

            terminated = infos['terminate'].float().unsqueeze(-1)
            next_vals = self._eval_critic(self.obs) * (1.0 - terminated)
            self.experience_buffer.update_data('next_values', step_idx, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            if hasattr(self.vec_env.env.task, '_multiple_task_names'):
                for task_name in self.vec_env.env.task._multiple_task_names:
                    if infos.get(task_name, None) is not None:
                        task_done = torch.logical_and(
                            self.dones.bool(), infos[task_name]).nonzero(as_tuple=False)
                        self.multi_task_rwd_recoders[task_name].update(
                            self.current_rewards[task_done])

            not_dones = 1.0 - self.dones.float()
            self.current_rewards *= not_dones.unsqueeze(1)
            self.current_lengths *= not_dones

            if self.vec_env.env.task.viewer:
                self._amp_debug(infos)
            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']

        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = {}
        for key in self.tensor_list:
            value = self.experience_buffer.tensor_dict.get(key)
            if value is None:
                continue
            if key in ('obses', 'next_obses'):
                batch_dict[key] = self._flatten_scene_tensor(value)
            elif key == 'states':
                batch_dict[key] = a2c_common.swap_and_flatten01(value)
            else:
                batch_dict[key] = self._group_agent_tensor(value)

        batch_dict['returns'] = self._group_agent_tensor(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        for key, value in amp_rewards.items():
            batch_dict[key] = self._group_agent_tensor(value)
        return batch_dict

    def prepare_dataset(self, batch_dict):
        if not self._scene_policy:
            return super().prepare_dataset(batch_dict)

        values = batch_dict['values']
        returns = batch_dict['returns']
        advantages = torch.sum(returns - values, dim=-1)  # (scene, M)
        rand_action_mask = batch_dict['rand_action_mask']
        if self.normalize_advantage:
            advantages = torch_ext.normalization_with_masks(
                advantages, rand_action_mask)

        if self.normalize_value:
            value_shape = values.shape
            values = self.value_mean_std(values.reshape(-1, value_shape[-1])).view(value_shape)
            returns = self.value_mean_std(returns.reshape(-1, value_shape[-1])).view(value_shape)

        dataset_dict = {
            'old_values': values,
            'old_logp_actions': batch_dict['neglogpacs'],
            'advantages': advantages,
            'returns': returns,
            'actions': batch_dict['actions'],
            'obs': batch_dict['obses'],
            'rnn_states': None,
            'rnn_masks': None,
            'mu': batch_dict['mus'],
            'sigma': batch_dict['sigmas'],
            'amp_obs': batch_dict['amp_obs'],
            'amp_obs_demo': batch_dict['amp_obs_demo'],
            'amp_obs_replay': batch_dict['amp_obs_replay'],
            'rand_action_mask': rand_action_mask,
        }
        self.dataset.update_values_dict(dataset_dict)
        return

    def train_epoch(self):
        if not self._scene_policy:
            return super().train_epoch()

        play_time_start = time.time()
        with torch.no_grad():
            batch_dict = self.play_steps()
        play_time_end = time.time()
        update_time_start = time.time()

        self._update_amp_demos()
        scene_count, M, amp_dim = batch_dict['amp_obs'].shape
        num_agent_samples = scene_count * M
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_agent_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo.view(scene_count, M, amp_dim)

        if self._amp_replay_buffer.get_total_count() == 0:
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        else:
            replay = self._amp_replay_buffer.sample(num_agent_samples)['amp_obs']
            batch_dict['amp_obs_replay'] = replay.view(scene_count, M, amp_dim)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        train_info = None
        for _ in range(self.mini_epochs_num):
            for minibatch_idx in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[minibatch_idx])
                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(
                            curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0,
                        curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = {key: [value] for key, value in curr_train_info.items()}
                else:
                    for key, value in curr_train_info.items():
                        train_info[key].append(value)

            av_kls = torch_ext.mean_list(train_info['kl'])
            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            av_kls = torch_ext.mean_list(train_info['kl'])
            if self.multi_gpu:
                av_kls = self.hvd.average_value(av_kls, 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        flat_amp_obs = batch_dict['amp_obs'].reshape(num_agent_samples, amp_dim)
        self._store_replay_amp_obs(flat_amp_obs)

        train_info['play_time'] = play_time_end - play_time_start
        train_info['update_time'] = update_time_end - update_time_start
        train_info['total_time'] = update_time_end - play_time_start
        self._record_train_batch_info(batch_dict, train_info)
        return train_info

    def calc_gradients(self, input_dict):
        if not self._scene_policy:
            return super().calc_gradients(input_dict)

        self.set_train()
        obs_batch = self._preproc_obs(input_dict['obs'])

        def flatten_agents(value):
            return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

        value_preds_batch = flatten_agents(input_dict['old_values'])
        old_action_log_probs_batch = flatten_agents(
            input_dict['old_logp_actions'].unsqueeze(-1)).squeeze(-1)
        advantage = flatten_agents(input_dict['advantages'].unsqueeze(-1)).squeeze(-1)
        old_mu_batch = flatten_agents(input_dict['mu'])
        old_sigma_batch = flatten_agents(input_dict['sigma'])
        return_batch = flatten_agents(input_dict['returns'])
        actions_batch = flatten_agents(input_dict['actions'])
        rand_action_mask = flatten_agents(
            input_dict['rand_action_mask'].unsqueeze(-1)).squeeze(-1)

        amp_obs = flatten_agents(input_dict['amp_obs'])[:self._amp_minibatch_size]
        amp_obs_replay = flatten_agents(
            input_dict['amp_obs_replay'])[:self._amp_minibatch_size]
        amp_obs_demo = flatten_agents(
            input_dict['amp_obs_demo'])[:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        rand_action_sum = torch.sum(rand_action_mask)
        curr_e_clip = self.e_clip
        model_input = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
            'amp_obs': amp_obs,
            'amp_obs_replay': amp_obs_replay,
            'amp_obs_demo': amp_obs_demo,
        }

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(model_input)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(
                old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            c_info = self._critic_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            b_loss = self.bound_loss(mu)

            a_loss = torch.sum(rand_action_mask * a_info['actor_loss']) / rand_action_sum
            c_loss = torch.mean(c_info['critic_loss'])
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(
                rand_action_mask * a_info['actor_clipped'].float()) / rand_action_sum

            disc_agent_cat_logit = torch.cat([
                res_dict['disc_agent_logit'], res_dict['disc_agent_replay_logit']], dim=0)
            disc_info = self._disc_loss(
                disc_agent_cat_logit, res_dict['disc_demo_logit'], amp_obs_demo)
            disc_loss = disc_info['disc_loss']
            loss = (a_loss + self.critic_coef * c_loss
                    - self.entropy_coef * entropy
                    + self.bounds_loss_coef * b_loss
                    + self._disc_coef * disc_loss)

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, True)

        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': 1.0,
            'b_loss': b_loss,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
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
