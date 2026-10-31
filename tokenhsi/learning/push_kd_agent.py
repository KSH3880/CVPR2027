"""PUSH-only PPO/critic/AMP with old-task on-policy KL supervision."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.trans_agent import TransAgent
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder
from learning.amp_models import ModelAMPContinuous
from learning.push_kd_losses import gaussian_forward_kl, masked_mean, push_advantages, teacher_observation, normalize_policy
import yaml


class PushKDAgent(TransAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        if self.is_rnn or self.multi_gpu or self.has_central_value or self._enable_eps_greedy:
            raise ValueError('PUSH KD supports feed-forward, single GPU, stochastic PPO only')
        self._kd_coef = float(config.get('kd_coef', .001))
        if not 0 <= self._kd_coef < float('inf'):
            raise ValueError('kd_coef must be finite and nonnegative')
        path = Path(config['teacher_checkpoint'])
        self._teacher_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        teacher = torch.load(path, map_location=self.ppo_device, weights_only=True)
        params = yaml.safe_load(Path('tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml').read_text())['params']
        info = dict(onehot_size=4, tota_subtask_obs_size=127, each_subtask_obs_size=[20,38,42,27], each_subtask_obs_indx=[0,20,58,100,127], each_subtask_name=['traj','sit','carry','climb'], enable_task_mask_obs=True)
        builder = AMPTransformerMultiTaskBuilder()
        builder.load(params['network'])
        self._teacher = ModelAMPContinuous.Network(builder.build('amp', actions_num=32, input_shape=(354,), num_seqs=1, amp_input_shape=(1330,), self_obs_size=223, task_obs_size=131, multi_task_info=info, device=self.ppo_device)).to(self.ppo_device)
        self._teacher.load_state_dict(teacher['model'], strict=True)
        self._teacher.eval().requires_grad_(False)
        self._teacher_rms = RunningMeanStd((354,)).to(self.ppo_device)
        self._teacher_rms.load_state_dict(teacher['running_mean_std'])
        self._teacher_rms.eval().requires_grad_(False)
        initial = torch.load(config['initial_checkpoint'], map_location=self.ppo_device, weights_only=True)
        self.model.load_state_dict(initial['model'], strict=True)
        self.running_mean_std.load_state_dict(initial['running_mean_std'])
        self._amp_input_mean_std.load_state_dict(initial['amp_input_mean_std'])
        self.value_mean_std.load_state_dict(initial['reward_mean_std'])
        task = self.vec_env.env.task
        contract_env = copy.deepcopy(task.cfg['env'])
        contract_env.pop('checkpoint', None)
        self._contract = dict(version=2, teacher_sha256=self._teacher_sha, kd_coef=self._kd_coef, reference_26k=task.cfg['env']['push']['reference_checkpoint_sha256'], env=contract_env, scene='reference_2humans_4boxes_plus_native_kd', ppo='push_only', amp='push_only')
        self._contract['reference_environment'] = yaml.safe_load(Path('tokenhsi/data/cfg/multi_task/push_reference.yaml').read_text())
        self._contract['reference_source_sha256'] = hashlib.sha256(Path('push_reference/PROVENANCE.json').read_bytes()).hexdigest()
        Path(self.experiment_dir, 'method_contract.json').write_text(json.dumps(self._contract, indent=2, default=str))

    def init_tensors(self):
        super().init_tensors()
        self.update_list = list(self.update_list)
        shape = self.experience_buffer.obs_base_shape
        for key in ('teacher_mu', 'teacher_sigma'):
            self.experience_buffer.tensor_dict[key] = torch.zeros(shape+(self.actions_num,), device=self.ppo_device)
            self.tensor_list.append(key)
            self.update_list.append(key)
        self.experience_buffer.tensor_dict['kd_mask'] = torch.zeros(shape, dtype=torch.bool, device=self.ppo_device)
        self.tensor_list.append('kd_mask')
        self.update_list.append('kd_mask')

    def _preproc_obs(self, obs):
        return normalize_policy(obs, self.running_mean_std, 5) if self.normalize_input else obs

    def _preproc_amp_obs(self, obs):
        labels = obs.view(-1, 134)[:, -5:].clone()
        result = super()._preproc_amp_obs(obs)
        result.view(-1, 134)[:, -5:] = labels
        return result

    def get_action_values(self, obs_dict, rand_action_probs):
        result = super().get_action_values(obs_dict, rand_action_probs)
        if (obs_dict['obs'][:, 392] > .5).any():
            raise RuntimeError('TRAJ samples must not enter PUSH KD training')
        old = self.vec_env.env.task.kd_mask
        result['kd_mask'] = old
        result['teacher_mu'] = torch.zeros_like(result['mus'])
        result['teacher_sigma'] = torch.ones_like(result['sigmas'])
        with torch.no_grad():
            if old.any():
                raw = teacher_observation(obs_dict['obs'][old])
                normalized = normalize_policy(raw, self._teacher_rms, 4)
                mu, logstd = self._teacher.a2c_network.eval_actor(normalized, raw)
                result['teacher_mu'][old] = mu
                result['teacher_sigma'][old] = logstd.exp().expand_as(mu)
        # Used by inherited GAE advantage normalization and actor masking.
        result['rand_action_mask'] *= (obs_dict['obs'][:, -1] > .5).float()
        return result

    def play_steps(self):
        batch = super().play_steps()
        push = batch['obses'][:, -1] > .5
        self._push_agent_samples = batch['amp_obs'][push].detach()
        if not len(self._push_agent_samples):
            raise RuntimeError('No PUSH experience in rollout')
        self._rollout_push_mask = push
        return batch

    def _sample_push_amp(self, count):
        x = self._push_agent_samples
        return x[torch.randint(len(x), (count,), device=x.device)]

    def _calc_amp_rewards(self, obs):
        shape = obs.shape[:-1]
        flat = obs.reshape(-1, obs.shape[-1])
        push = flat[:, 133] > .5
        result = flat.new_zeros(len(flat), 1)
        if push.any():
            result[push] = self._calc_disc_rewards(flat[push])
        return dict(disc_rewards=result.reshape(*shape, 1))

    def _combine_rewards(self, task_rewards, amp_rewards):
        # The rollout mask is read from the same pre-step policy task labels.
        push = self.experience_buffer.tensor_dict['obses'][..., -1:] > .5
        return super()._combine_rewards(task_rewards, amp_rewards)*push

    def prepare_dataset(self, batch):
        push = batch['obses'][:, -1] > .5
        values, returns = batch['values'], batch['returns']
        advantages = push_advantages(returns, values, push, self.normalize_advantage)
        if self.normalize_value:
            self.value_mean_std.train()
            self.value_mean_std(values[push])
            self.value_mean_std(returns[push])
            self.value_mean_std.eval()
            values, returns = self.value_mean_std(values), self.value_mean_std(returns)
        data = dict(old_values=values, returns=returns, advantages=advantages, old_logp_actions=batch['neglogpacs'], actions=batch['actions'], obs=batch['obses'], mu=batch['mus'], sigma=batch['sigmas'], rnn_states=None, rnn_masks=None)
        for k in ('amp_obs','amp_obs_demo','amp_obs_replay','rand_action_mask','teacher_mu','teacher_sigma','kd_mask'):
            data[k] = batch[k]
        data['push_mask'] = push
        # First epoch has no replay yet: substitute ONLY PUSH samples.
        if self._amp_replay_buffer.get_total_count() == 0:
            data['amp_obs_replay'] = self._sample_push_amp(len(push))
        self.dataset.update_values_dict(data)

    def _store_replay_amp_obs(self, obs):
        super()._store_replay_amp_obs(obs[obs[:, 133] > .5])

    def get_full_state_weights(self):
        weights = super().get_full_state_weights()
        weights['push_kd_contract'] = self._contract
        return weights

    def set_full_state_weights(self, weights):
        if weights.get('push_kd_contract') != self._contract:
            raise ValueError('PUSH KD resume contract mismatch')
        super().set_full_state_weights(weights)

    def _log_train_info(self, info, frame):
        super()._log_train_info(info, frame)
        self.writer.add_scalar('distill/teacher_kl', torch_ext.mean_list(info['teacher_kl']).item(), frame)
        self.writer.add_scalar('distill/coefficient', self._kd_coef, frame)
        self.writer.add_scalar('samples/push_rollout', self._rollout_push_mask.sum().item(), frame)
        self.writer.add_scalar('samples/kd_rollout', self.vec_env.env.task.kd_mask.sum().item()*self.horizon_length, frame)

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = self._sample_push_amp(self._amp_minibatch_size)
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)
        
        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask).clamp_min(1)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'not_normalized_obs': input_dict['obs'],
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            c_loss = masked_mean(c_loss, input_dict['push_mask'])
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']


            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss
            
            if self._reg_loss_enabled:
                reg_loss_info = self._reg_loss()
                reg_loss_v = reg_loss_info['reg_loss']
                loss += reg_loss_v
            
            old = input_dict['kd_mask']
            teacher_kl = gaussian_forward_kl(mu[old], sigma[old], input_dict['teacher_mu'][old], input_dict['teacher_sigma'][old])
            loss = loss + self._kd_coef * teacher_kl
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
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
            reduce_kl = not self.is_rnn
            active = input_dict['push_mask']
            kl_dist = torch_ext.policy_kl(mu[active].detach(), sigma[active].detach(), old_mu_batch[active], old_sigma_batch[active], reduce_kl) if active.any() else mu.sum().detach()*0
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss,
            'teacher_kl': teacher_kl.detach(),
            'push_samples': input_dict['push_mask'].sum().detach()
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        if self._reg_loss_enabled:
            self.train_result.update(reg_loss_info)

        return
