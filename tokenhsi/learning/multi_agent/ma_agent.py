# PPO/AMP agent for the multi-agent relation-transformer policy.
#
# legacy_multirow delegates the standard N*M row layout to rl_games. clean_scene keeps
# observations at (T,N,scene) and groups all agent-specific PPO quantities as
# (T,N,M,...); only the loss-facing quantities are flattened to N*M after one scene
# forward. AMP observations remain per humanoid in both modes.

import time
import os
import copy
import json
import math
import yaml
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
from learning.multi_agent.distillation import gaussian_forward_kl, gradient_report, teacher_digest
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, LEGACY_MODE
from utils.edge_stage1_spec import CURRICULUM_VARIANTS
from utils.rsi_curriculum import SkillInitCurriculum
from env.tasks.multi_agent.relation_diagnostics import relation_tensorboard_tag


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
        curriculum_cfg = task.cfg['env'].get('skillInitCurriculum')
        args = task.cfg['args']
        self._skill_init_curriculum = None
        if (curriculum_cfg is not None and task._mode == 'train'
                and not any(getattr(args, flag, False) for flag in ('test', 'play', 'eval'))):
            self._skill_init_curriculum = SkillInitCurriculum(
                task._skill, task.cfg['env']['skillInitProb'], curriculum_cfg)
        if task._state_relation:
            task._relation_output_directory = self.experiment_dir
            with open(task.cfg['args'].cfg_train) as f:
                resolved = yaml.safe_load(f)
            for key in resolved['params']['config']:
                if key in config:
                    try:
                        yaml.safe_dump(config[key])
                    except yaml.YAMLError:
                        # rl_games replaces reward_shaper with a callable; keep its YAML definition.
                        continue
                    resolved['params']['config'][key] = copy.deepcopy(config[key])
            resolved['params']['seed'] = config['seed']
            resolved['params']['load_checkpoint'] = task.cfg['args'].resume > 0
            if task.cfg['args'].checkpoint != 'Base':
                resolved['params']['load_path'] = task.cfg['args'].checkpoint
            self._relation_experiment_config = {'env': copy.deepcopy(task.cfg['env']),
                'train': resolved, 'num_envs': task.num_envs, 'num_agents': task.num_agents,
                'num_objects': task.num_objects, 'control_dt': task.dt,
                'observation_size': task.get_obs_size(),
                'experiment': copy.deepcopy(task.cfg.get('experiment', {}))}
            with open(os.path.join(self.experiment_dir, 'relation_config.yaml'), 'w') as f:
                yaml.safe_dump(self._relation_experiment_config, f, sort_keys=False)
        self._scene_policy = task.is_scene_policy()
        if self.normalize_input:
            if self._scene_policy:
                self.running_mean_std = SceneRunningMeanStd(
                    entity_sizes=task.get_scene_entity_sizes(),
                    entity_counts=[task.num_agents, task.num_objects, task.num_agents],
                    normalized_sizes=task.get_scene_normalized_entity_sizes(),
                    kinematic_size=task.get_scene_kinematic_size(),
                    extra_passthrough_size=task.get_relation_suffix_size(),
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
        if getattr(task, '_ontop_mixed', False) and args.resume <= 0:
            from learning.multi_agent.transfer import transfer_carry_weights
            path = task.cfg['experiment']['transfer']['checkpoint']
            checkpoint = torch.load(path, map_location=self.ppo_device, weights_only=False)
            report = transfer_carry_weights(self.model, checkpoint, task._relation_cfg)
            self.set_stats_weights(checkpoint)
            report['checkpoint'] = path
            report['normalizers'] = [key for key in ('running_mean_std', 'reward_mean_std', 'amp_input_mean_std')
                                     if key in checkpoint]
            with open(os.path.join(self.experiment_dir, 'transfer_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
            print('[OnTop transfer] epoch {}, {} tensors copied, {} embeddings extended; no extra freeze'.format(
                report['source_epoch'], len(report['copied_tensors']), len(report['expanded_embeddings'])), flush=True)
        self._stage2_policy = getattr(task, '_stage2', False)
        if self._stage2_policy:
            from learning.multi_agent.stage2_transfer import (
                freeze_stage2_encoder, transfer_stage1_weights)
            self._stage2_checkpoint_info = None
            if args.resume <= 0 and not (args.test or args.eval):
                path = task.cfg['experiment']['transfer']['checkpoint']
                checkpoint = torch.load(path, map_location=self.ppo_device, weights_only=False)
                report = transfer_stage1_weights(self.model, checkpoint)
                if not self.normalize_input or 'running_mean_std' not in checkpoint:
                    raise ValueError('Stage-2 transfer requires Stage-1 actor observation RMS')
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
                if self._normalize_amp_input:
                    self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
                report['checkpoint'] = path
                report['normalizers'] = ['running_mean_std'] + (
                    ['amp_input_mean_std'] if self._normalize_amp_input else [])
                report['reward_normalizer'] = 'reset_for_stage2'
                report['source_config_available'] = 'relation_experiment_config' in checkpoint
                report['source_commit'] = checkpoint.get('git_commit')
                self._stage2_checkpoint_info = {
                    'stage': 2, 'coordination': 'grounded_edge_cross_attention',
                    'edge_width': 64, 'context_width': 64,
                    'source_checkpoint': path,
                    'source_config': checkpoint.get('relation_experiment_config'),
                    'source_reward_config': checkpoint['relation_metadata']['relation_reward_config'],
                    'source_experiment': checkpoint.get('relation_experiment_config', {}).get('experiment'),
                    'source_commit': checkpoint.get('git_commit'),
                    'freeze': 'actor_encoder_and_actor_observation_rms',
                    'rsi': 'template_with_shared_interaction_loco',
                }
                with open(os.path.join(self.experiment_dir, 'stage2_transfer_report.json'), 'w') as f:
                    json.dump(report, f, indent=2)
            freeze_stage2_encoder(self.model, self.optimizer)
            self.model.a2c_network.actor_encoder.eval()
            self.running_mean_std.eval()
        self._teacher = None
        self._distill_config = config.get('teacher_distillation', {})
        if self._distill_config.get('checkpoint'):
            from learning.multi_agent.stage1_unified_teacher import Stage1UnifiedTeacher
            if (not self._scene_policy or task.__class__.__name__ != 'HumanoidMACarry'
                    or task._relation_cfg.get('stage1_variant') != 'scenario_independent_stage1_self_sum'):
                raise ValueError('Teacher distillation requires Stage-1 experiment 34')
            self._teacher_kl_coef = float(self._distill_config.get('kl_coef', 1e-3))
            if not math.isfinite(self._teacher_kl_coef) or self._teacher_kl_coef <= 0:
                raise ValueError('teacher kl_coef must be finite and positive')
            with torch.random.fork_rng(devices=[torch.device(self.ppo_device)]):
                self._teacher = Stage1UnifiedTeacher(
                    self._distill_config['checkpoint'], self.ppo_device)
            if (task.get_action_size() != 32
                    or task.cfg['env']['asset']['assetFileName'] != self._teacher.env_cfg['asset']['assetFileName']
                    or not math.isclose(task.dt, self._teacher.env_cfg['controlFrequencyInv'] / 60,
                                        rel_tol=1e-6)):
                raise ValueError('Teacher requires matching humanoid, action layout and control timestep')
            self._teacher_reset_ids = None
            self._teacher_labeled_resets = 0
            self._teacher_updates = 0
            self._teacher_grad_checks = int(self._distill_config.get('grad_checks', 0))
            if self._teacher_grad_checks < 0:
                raise ValueError('teacher grad_checks must be nonnegative')
            if self._teacher_grad_checks:
                self._teacher_initial_digest = teacher_digest(self._teacher)
            print('teacher distillation:', json.dumps(self._distill_config), flush=True)
        return

    def set_train(self):
        super().set_train()
        if getattr(self, '_stage2_policy', False):
            self.model.a2c_network.actor_encoder.eval()
            self.running_mean_std.eval()

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
        if self._teacher is not None:
            for name, source in [('teacher_mu', 'mus'), ('teacher_sigma', 'sigmas')]:
                self.experience_buffer.tensor_dict[name] = torch.zeros_like(
                    self.experience_buffer.tensor_dict[source])
                self.tensor_list.append(name)
        return

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        if self._teacher is not None:
            task = self.vec_env.env.task
            if env_ids is None:
                self._teacher_reset_ids = torch.arange(task.num_envs, device=task.device)
            elif len(env_ids):
                self._teacher_reset_ids = torch.div(
                    env_ids, task.num_agents, rounding_mode='floor').flatten().unique().to(task.device)
        return obs

    def _update_training_curriculum(self):
        if self._skill_init_curriculum is None:
            return
        task = self.vec_env.env.task
        probabilities, blend = self._skill_init_curriculum.at_epoch(self.epoch_num)
        # Only future reset sampling changes; AMP demonstrations and live episodes
        # retain their state. Keep skillInitProb in the config as the final target.
        task._skill_init_prob.copy_(task._skill_init_prob.new_tensor(probabilities))
        if self.rank == 0:
            for skill, probability in zip(self._skill_init_curriculum.skills, probabilities):
                self.writer.add_scalar('rsi/init_prob/' + skill, probability, self.epoch_num)
            self.writer.add_scalar('rsi/blend', blend, self.epoch_num)

    def get_stats_weights(self):
        weights = super().get_stats_weights()
        task = self.vec_env.env.task
        weights['relation_metadata'] = checkpoint_metadata(task._relation_cfg)
        if getattr(task, '_edge_context', False):
            from utils.edge_context_spec import task_instance
            weights['relation_task_instance'] = task_instance(task._relation_graph_spec, task.num_agents, task.num_objects)
        if task._state_relation:
            weights['relation_experiment_config'] = self._relation_experiment_config
        if getattr(task, '_stage2', False):
            weights['stage2_checkpoint_info'] = self._stage2_checkpoint_info
        if self._teacher is not None:
            weights['teacher_distillation'] = dict(self._distill_config)
        return weights

    def set_weights(self, weights):
        task = self.vec_env.env.task
        stage2_evaluation = getattr(task, '_stage2', False) and (
            task.cfg['args'].test or task.cfg['args'].eval)
        if getattr(task, '_edge_context', False) and not stage2_evaluation:
            from utils.edge_context_spec import check_task_resume
            check_task_resume(weights, task._relation_graph_spec, task.num_agents, task.num_objects)
        check_checkpoint_metadata(weights, checkpoint_metadata(self.vec_env.env.task._relation_cfg))
        if getattr(task, '_stage2', False) and not stage2_evaluation:
            self._stage2_checkpoint_info = weights.get('stage2_checkpoint_info')
        return super().set_weights(weights)

    def _build_net_config(self):
        config = super()._build_net_config()

        task = self.vec_env.env.task
        config["num_agents"] = task.num_agents
        config["num_objects"] = task.num_objects
        config["humanoid_obs_size"] = task.get_humanoid_obs_size()
        config["object_obs_size"] = task.get_object_obs_size()
        config["goal_obs_size"] = task.get_goal_obs_size()
        config["observation_mode"] = task.get_policy_obs_mode()
        config['relation_reward_mode'] = task._relation_cfg.get('mode', LEGACY_MODE)
        config['relation_graph_spec'] = task._relation_graph_spec
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
        relation_near_masks = []

        for step_idx in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', step_idx, self.obs['obs'])

            if self._teacher is not None:
                teacher_obs = self._teacher.observation(self.vec_env.env.task, self._teacher_reset_ids)
                if self._teacher_reset_ids is not None:
                    self._teacher_labeled_resets += len(self._teacher_reset_ids)
                teacher_mu, teacher_sigma = self._teacher.distribution(teacher_obs)
                self.experience_buffer.update_data('teacher_mu', step_idx, teacher_mu)
                self.experience_buffer.update_data('teacher_sigma', step_idx, teacher_sigma)
                self._teacher_reset_ids = None

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            for key in update_list:
                self.experience_buffer.update_data(key, step_idx, res_dict[key])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            if 'relation_near_unplaced_slow' in infos:
                relation_near_masks.append(infos['relation_near_unplaced_slow'].clone())
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
        if relation_near_masks:
            mask = torch.stack(relation_near_masks).unsqueeze(-1)
            count = mask.sum().clamp_min(1)
            batch_dict['relation_conditional_diagnostics'] = {
                'near_unplaced_slow/amp': (amp_rewards['disc_rewards'] * mask).sum() / count,
                'near_unplaced_slow/combined': (mb_rewards * mask).sum() / count,
                'near_unplaced_slow/sample_count': mask.sum().float()}
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
        if self._teacher is not None:
            for name in ('teacher_mu', 'teacher_sigma'):
                dataset_dict[name] = batch_dict[name]
        self.dataset.update_values_dict(dataset_dict)
        return

    def train_epoch(self):
        task = self.vec_env.env.task
        if (task._relation_cfg.get('stage1_variant') in CURRICULUM_VARIANTS or
                'independent_training' in task._relation_cfg):
            task._hard_skill_training_step = self.epoch_num * self.horizon_length
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
        if self._teacher is not None and self._teacher_grad_checks:
            frozen = (teacher_digest(self._teacher) == self._teacher_initial_digest
                      and all(p.grad is None for p in self._teacher.model.parameters()))
            if not frozen:
                raise RuntimeError('Teacher weights or gradients changed during training')
            print('distill teacher frozen: true', flush=True)
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
        report = None

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

            if self._teacher is not None:
                teacher_mu = flatten_agents(input_dict['teacher_mu'])
                teacher_sigma = flatten_agents(input_dict['teacher_sigma'])
                teacher_kl = gaussian_forward_kl(mu, sigma, teacher_mu, teacher_sigma)
                weighted_teacher_kl = self._teacher_kl_coef * teacher_kl
                loss = loss + weighted_teacher_kl
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite distillation/PPO loss')
                if self._teacher_updates < self._teacher_grad_checks:
                    report = gradient_report(self.model, weighted_teacher_kl, a_loss)
                    report.update(update=self._teacher_updates + 1,
                                  kl_joint=teacher_kl.item(), kl_weighted=weighted_teacher_kl.item())
                    before_step = {name: p.detach().clone() for name, p in self.model.named_parameters()
                                   if p.requires_grad and ('actor_encoder' in name or 'action_head' in name)}

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if report is not None:
            report['student_backward_finite'] = all(torch.isfinite(p.grad).all().item()
                                                    for p in self.model.parameters() if p.grad is not None)
            if not report['student_backward_finite'] or not report['kl_critic_grad_absent']:
                raise RuntimeError('Distillation gradient check failed')
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

        if report is not None:
            with torch.no_grad():
                delta = sum((p - before_step[name]).square().sum()
                            for name, p in self.model.named_parameters() if name in before_step)
                report['actor_parameter_update_norm'] = delta.sqrt().item()
                if not math.isfinite(report['actor_parameter_update_norm']) or delta == 0:
                    raise RuntimeError('Student actor optimizer update failed')
            print('distill gradients:', json.dumps(report), flush=True)

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
        if self._teacher is not None:
            self._teacher_updates += 1
            self.train_result.update(
                teacher_kl=teacher_kl.detach(),
                teacher_kl_weighted=weighted_teacher_kl.detach(),
                teacher_mu_rmse=(mu.detach() - teacher_mu).square().mean().sqrt(),
                teacher_sigma=teacher_sigma.mean(), student_sigma=sigma.detach().mean(),
                student_policy_kl_exact=gaussian_forward_kl(
                    old_mu_batch, old_sigma_batch, mu.detach(), sigma.detach()))
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        task = self.vec_env.env.task
        means = task.consume_reward_term_means()
        if means is not None:
            train_info["reward_term_means"] = means
        train_info['relation_diagnostics'] = task.consume_relation_diagnostics()
        train_info['relation_diagnostics'].update(batch_dict.get('relation_conditional_diagnostics', {}))
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        disc_reward_mean = train_info["disc_rewards"].mean()
        self.writer.add_scalar("reward_terms/amp", disc_reward_mean.item(), frame)
        for key, value in train_info.get('relation_diagnostics', {}).items():
            self.writer.add_scalar(relation_tensorboard_tag(key), value.item(), frame)
        if self._teacher is not None:
            metrics = {key: torch_ext.mean_list(train_info[key]).item() for key in
                       ('teacher_kl', 'teacher_kl_weighted', 'teacher_mu_rmse',
                        'teacher_sigma', 'student_sigma', 'actor_loss', 'critic_loss',
                        'disc_loss', 'kl', 'student_policy_kl_exact', 'actor_clip_frac', 'b_loss')}
            metrics['coefficient'] = self._teacher_kl_coef
            metrics['labeled_scene_resets'] = self._teacher_labeled_resets
            for name, count in zip(('holding', 'sit', 'climb', 'holding_at', 'holding_ontop'),
                                   self._teacher.counts.cpu().tolist()):
                metrics['template_' + name] = count
            for key, value in metrics.items():
                self.writer.add_scalar('distill/' + key, value, frame)
            print('distill epoch:', json.dumps(dict(epoch=self.epoch_num, **metrics)), flush=True)
        network = self.model.a2c_network
        for name in ('actor', 'critic'):
            encoder = getattr(network, name + '_encoder')
            for key, value in encoder.last_diagnostics.items():
                self.writer.add_scalar('relation_attention/{}/{}'.format(name, key), value.item(), frame)

        means = train_info.get("reward_term_means")
        if means is not None:
            for name, value in zip(self.vec_env.env.task.REWARD_TERM_NAMES, means):
                self.writer.add_scalar("reward_terms/{}".format(name), value.item(), frame)
            combined = self._task_reward_w * means[-1] + self._disc_reward_w * disc_reward_mean
            self.writer.add_scalar("reward_terms/combined", combined.item(), frame)
        return
