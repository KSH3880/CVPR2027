# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import os
import json
import time
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn

import learning.replay_buffer as replay_buffer
import learning.amp_agent as amp_agent

from tensorboardX import SummaryWriter


def mstop_phase_reward_weights(command):
    task_weight = torch.full_like(command[..., :1], 0.5)
    amp_weight = torch.full_like(task_weight, 0.5)
    stop = (command[..., 2:3] > 0.5) | (command[..., 3:4] > 0.5)
    task_weight = torch.where(stop, torch.ones_like(task_weight), task_weight)
    amp_weight = torch.where(stop, torch.zeros_like(amp_weight), amp_weight)
    resume = command[..., 4:5] > 0.5
    resume_amp = 0.5 * command[..., 1:2].clamp(0.0, 1.0)
    amp_weight = torch.where(resume, resume_amp, amp_weight)
    task_weight = torch.where(resume, 1.0 - resume_amp, task_weight)
    return task_weight, amp_weight


class TransAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self._mstop_diag = (os.environ.get("MSTOP_DIAG", "0") == "1"
                            and (os.environ.get("MSTOP_RESIDUAL", "0") == "1"
                                 or os.environ.get("MSTOP_STOP_HEAD", "0") == "1"))
        self._mstop_diag_collect = False
        self._mstop_diag_steps = []
        self._last_mstop_diag = None
        self._mstop_phase_reward = os.environ.get("MSTOP_PHASE_REWARD", "0") == "1"
        return

    def set_full_state_weights(self, weights):
        if os.environ.get("MSTOP_STOP_HEAD_WARMSTART", "0") != "1":
            return super().set_full_state_weights(weights)

        incompatible = self.model.load_state_dict(weights['model'], strict=False)
        allowed_missing = ("a2c_network.stop_composer.",
                           "a2c_network.stop_internal_adapt_mlp.")
        bad_missing = [k for k in incompatible.missing_keys
                       if not k.startswith(allowed_missing)]
        allowed_unexpected = ("a2c_network.stop_residual.",)
        bad_unexpected = [k for k in incompatible.unexpected_keys
                          if not k.startswith(allowed_unexpected)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"STOP-head warm-start mismatch: missing={bad_missing} "
                               f"unexpected={bad_unexpected}")

        net = self.model.a2c_network
        net.stop_composer.load_state_dict(net.composer.state_dict())
        if net.use_internal_adaptation:
            net.stop_internal_adapt_mlp.load_state_dict(net.internal_adapt_mlp.state_dict())
        self.set_stats_weights(weights)
        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)
        if hasattr(self, 'vec_env'):
            self.vec_env.set_env_state(weights.get('env_state', None))
        print("MSTOP_STOP_HEAD_WARMSTART optimizer=fresh head=cloned_A", flush=True)

    def set_train(self):
        super().set_train()
        if os.environ.get("MSTOP_FROZEN_A", "0") == "1" and self.normalize_input and hasattr(self, "running_mean_std"):
            self.running_mean_std.eval()

        return

    def _preproc_obs(self, obs_batch):
        if isinstance(obs_batch, dict) or os.environ.get("MSTOP_FROZEN_A", "0") != "1":
            return super()._preproc_obs(obs_batch)
        start = self.model.a2c_network.self_obs_size
        command = obs_batch[..., start:start + 6].clone()
        obs_batch = super()._preproc_obs(obs_batch)
        obs_batch[..., start:start + 6] = command
        return obs_batch


    def _build_net_config(self):
        config = super()._build_net_config()

        config["hrl_checkpoint"] = self.config.get("hrl_checkpoint", "")

        config["device"] = self.ppo_device

        return config

    def get_action_values(self, obs_dict, rand_action_probs):
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'not_normalized_obs': obs_dict['obs'],
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self._mstop_diag_collect:
                diag = getattr(self.model.a2c_network, "_mstop_diag_last", None)
                if diag is not None:
                    self._mstop_diag_steps.append(diag)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)

        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict

    def play_steps(self):
        if not self._mstop_diag:
            return super().play_steps()
        self._mstop_diag_steps = []
        self._mstop_diag_collect = True
        try:
            batch_dict = super().play_steps()
        finally:
            self._mstop_diag_collect = False
        self._record_mstop_diag(batch_dict)
        return batch_dict

    def _combine_rewards(self, task_rewards, amp_rewards):
        if not self._mstop_phase_reward:
            return super()._combine_rewards(task_rewards, amp_rewards)
        net = self.model.a2c_network
        start = net.self_obs_size + net.task_obs_each_indx[net.extra_ids[0]]
        command = self.experience_buffer.tensor_dict['obses'][..., start:start + 6]
        task_weight, amp_weight = mstop_phase_reward_weights(command)
        return task_weight * task_rewards + amp_weight * amp_rewards['disc_rewards']

    def _record_mstop_diag(self, batch_dict):
        if not self._mstop_diag_steps:
            return
        net = self.model.a2c_network
        action_diag = {}
        for key in self._mstop_diag_steps[0]:
            stacked = torch.stack([x[key] for x in self._mstop_diag_steps])
            action_diag[key] = a2c_common.swap_and_flatten01(stacked)

        command_start = net.self_obs_size + net.task_obs_each_indx[net.extra_ids[0]]
        command = batch_dict['obses'][:, command_start:command_start + 6]
        phase_masks = {
            "go": ~(command[:, 2:5] > .5).any(-1),
            "brake": command[:, 2] > .5,
            "hold": command[:, 3] > .5,
            "resume": command[:, 4] > .5,
        }
        task_reward = a2c_common.swap_and_flatten01(
            self.experience_buffer.tensor_dict['rewards']).reshape(-1)
        disc_reward = batch_dict['disc_rewards'].reshape(-1)
        scaled_disc_reward = (disc_reward / self._disc_reward_scale).clamp_min(1e-8)
        disc_logit = torch.log(torch.expm1(scaled_disc_reward))
        if self._mstop_phase_reward:
            task_weight, amp_weight = mstop_phase_reward_weights(command)
            task_weight = task_weight.reshape(-1)
            amp_weight = amp_weight.reshape(-1)
        else:
            task_weight = torch.full_like(task_reward, self._task_reward_w)
            amp_weight = torch.full_like(disc_reward, self._disc_reward_w)
        combined = task_weight * task_reward + amp_weight * disc_reward
        applied = action_diag['gate'] * action_diag['delta']
        physics = self.vec_env.env.task.consume_mstop_diag()
        scale = net._mstop_residual_scale
        result = {"epoch": int(self.epoch_num), "iteration": int(self.epoch_num), "phases": {}}
        physics_names = ("count", "root_speed", "box_speed", "target_speed",
                         "root_error", "box_error", "held_rate", "dropped_rate",
                         "base_carry_reward", "stop_bonus", "env_task_reward")
        for phase_i, (name, mask) in enumerate(phase_masks.items()):
            count = int(mask.sum().item())
            row = physics[phase_i]
            physical_count = max(float(row[0].item()), 1.0)
            values = {
                "count": count,
                "task_reward": float(task_reward[mask].mean().item()) if count else 0.0,
                "disc_reward": float(disc_reward[mask].mean().item()) if count else 0.0,
                "combined_reward": float(combined[mask].mean().item()) if count else 0.0,
                "disc_logit": float(disc_logit[mask].mean().item()) if count else 0.0,
                "task_weight": float(task_weight[mask].mean().item()) if count else 0.0,
                "amp_weight": float(amp_weight[mask].mean().item()) if count else 0.0,
            }
            for i, metric in enumerate(physics_names[1:], 1):
                values[metric] = float((row[i] / physical_count).item())
            if count:
                values.update({
                    "base_action_abs": float(action_diag['base'][mask].abs().mean().item()),
                    "delta_abs": float(action_diag['delta'][mask].abs().mean().item()),
                    "applied_abs": float(applied[mask].abs().mean().item()),
                    "residual_saturation": float(
                        (action_diag['delta'][mask].abs() > .95 * scale).float().mean().item()),
                    "action_clamp": float(
                        (action_diag['pre_clamp'][mask].abs() > 1).float().mean().item()),
                    "final_base_abs": float(
                        (action_diag['final'][mask] - action_diag['base'][mask]).abs().mean().item()),
                })
            values.update({
                "phase": name,
                "target_v": values["target_speed"],
                "root_v": values["root_speed"],
                "box_v": values["box_speed"],
                "held": values["held_rate"],
                "dropped": values["dropped_rate"],
                "task_reward_total": values["task_reward"],
                "amp_reward": values["disc_reward"],
                "amp_logit": values["disc_logit"],
            })
            result["phases"][name] = values

        self._last_mstop_diag = result
        if self.rank != 0:
            return
        path = os.environ.get("MSTOP_DIAG_PATH")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
        every = max(1, int(os.environ.get("MSTOP_DIAG_EVERY", "10")))
        if self.epoch_num % every == 0:
            h = result['phases']['hold']
            print("[mstop-diag] "
                  f"epoch={self.epoch_num} HOLD n={h['count']} "
                  f"root/box={h['root_speed']:.3f}/{h['box_speed']:.3f} "
                  f"task/disc={h['task_reward']:.3f}/{h['disc_reward']:.3f} "
                  f"delta={h.get('applied_abs', 0):.3f} "
                  f"sat/clamp={h.get('residual_saturation', 0):.3f}/"
                  f"{h.get('action_clamp', 0):.3f} held={h['held_rate']:.3f}",
                  flush=True)

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

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

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

            c_loss = torch.mean(c_loss)
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

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()

        # MS_GRADCHK=1 : 첫 backward 뒤 토크나이저별 grad 를 한 번만 찍는다.
        # extra 토큰이 학습되는지 코드를 읽는 것보다 이게 확실하다.
        import os as _og
        if _og.environ.get("MS_GRADCHK") and not getattr(self, "_gradchk_done", False):
            self._gradchk_done = True
            net = self.model.a2c_network
            print("\n[gradchk] ===== 토크나이저별 requires_grad / grad =====", flush=True)
            names = getattr(net, "each_subtask_name", None) or []
            for i, enc in enumerate(net.task_encoder):
                ps = list(enc.parameters())
                rq = sum(int(p.requires_grad) for p in ps)
                gn = [p.grad for p in ps if p.grad is not None]
                tot = sum(float(g.abs().sum()) for g in gn)
                nm = names[i] if i < len(names) else "?"
                print(f"[gradchk]  enc{i} ({nm:<12}) params={len(ps)} "
                      f"requires_grad={rq}/{len(ps)} grad텐서={len(gn)}/{len(ps)} |grad|합={tot:.6e}",
                      flush=True)
            for nm2, mod in (("self_encoder", getattr(net, "self_encoder", None)),
                             ("transformer", getattr(net, "transformer_encoder", None)),
                             ("composer", getattr(net, "composer", None)),
                             ("adapt_mlp", getattr(net, "internal_adapt_mlp", None)),
                             ("stop_residual", getattr(net, "stop_residual", None)),
                             ("stop_composer", getattr(net, "stop_composer", None)),
                             ("stop_adapt", getattr(net, "stop_internal_adapt_mlp", None))):
                if mod is None: continue
                ps = list(mod.parameters())
                gn = [p.grad for p in ps if p.grad is not None]
                print(f"[gradchk]  {nm2:<12} requires_grad={sum(int(p.requires_grad) for p in ps)}/{len(ps)} "
                      f"grad텐서={len(gn)}/{len(ps)} |grad|합={sum(float(g.abs().sum()) for g in gn):.6e}", flush=True)
            for nm2 in ("weight_token", "pos_embed"):
                p = getattr(net, nm2, None)
                if p is None: continue
                print(f"[gradchk]  {nm2:<12} requires_grad={p.requires_grad} "
                      f"grad={'없음' if p.grad is None else f'{p.grad.abs().sum():.6e}'}", flush=True)
            print("[gradchk] --- 토크나이저 구조 ---", flush=True)
            for i in (0, 2):
                if i < len(net.task_encoder):
                    print(f"[gradchk]  enc{i}: {net.task_encoder[i]}".replace("\n", " "), flush=True)
            print("[gradchk] ==========================================\n", flush=True)

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
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': lr_mul,
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        if self._reg_loss_enabled:
            self.train_result.update(reg_loss_info)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._reg_loss_coef = config.get('reg_loss_coef', 0.0)
        self._reg_loss_enabled = config.get('reg_loss_enabled', False)
        return

    def _reg_loss(self):

        nn_parameters = self.model.a2c_network.get_compensate_composer_parameters()
        if len(nn_parameters) > 0:
            nn_parameters = torch.cat(nn_parameters, dim=-1)
            reg_loss = torch.sum(torch.square(nn_parameters))

            loss = reg_loss * self._reg_loss_coef
        else:
            loss = torch.tensor(0.0, device=self.device)

        disc_info = {
            'reg_loss': loss,
        }
        return disc_info

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        if self._reg_loss_enabled:
            self.writer.add_scalar('losses/compensate_composer_reg_loss', torch_ext.mean_list(train_info['reg_loss']).item(), frame)
        if self._last_mstop_diag is not None:
            for phase, values in self._last_mstop_diag['phases'].items():
                for name, value in values.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'mstop_diag/{phase}/{name}', value, frame)
        return
