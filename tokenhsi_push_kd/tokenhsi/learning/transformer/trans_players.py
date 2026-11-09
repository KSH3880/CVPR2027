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

import torch 
import datetime
import os
import json

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.players import rescale_actions, unsqueeze_obs

import learning.common_player as common_player
from utils.torch_utils import load_checkpoint

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

class TransPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):

        self._eval = config["eval"]
        self._eval_task = config["eval_task"]

        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']

        super().__init__(config)

        return

    def restore(self, fn):
        if (fn != 'Base'):
            # super().restore(fn)
            self._checkpoint_fn = fn
            checkpoint = load_checkpoint(fn, self.device)
            self.model.load_state_dict(checkpoint['model'])
            if self.normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            if self._normalize_amp_input:
                checkpoint = load_checkpoint(fn, self.device)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return

    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            # refer to CALM https://github.com/NVlabs/CALM
            self._amp_input_mean_std = RunningMeanStd((self._amp_observation_space[0] // self.env.task._num_amp_obs_steps,)).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            config['amp_obs_steps'] = self.env.task._num_amp_obs_steps
            if self.env.task._enable_task_obs:
                config['self_obs_size'] = self.env.task.get_obs_size() - self.env.task.get_task_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()
                if hasattr(self.env.task, '_enable_task_mask_obs'):
                    config['multi_task_info'] = self.env.task.get_multi_task_info()

                config["hrl_checkpoint"] = self.config.get("hrl_checkpoint", "")

                config["device"] = self.device
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            config['amp_obs_steps'] = self.env_info['num_amp_obs_steps']

        self._amp_observation_space = config['amp_input_shape']
        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            env = 0
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:4]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[:, 0]
            disc_reward = disc_reward.cpu().numpy()[:, 0]
            print("env: {} disc_pred: {} disc_reward: {}".format(env, disc_pred, disc_reward))

        return
    
    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            # refer to CALM https://github.com/NVlabs/CALM
            shape = amp_obs.shape
            amp_obs = amp_obs.view(-1, self.env.amp_observation_space.shape[0] // self.env.task._num_amp_obs_steps)
            amp_obs = self._amp_input_mean_std(amp_obs)
            amp_obs = amp_obs.view(shape)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def get_action(self, obs_dict, is_determenistic = False):
        
        obs = obs_dict['obs']

        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'not_normalized_obs': obs_dict['obs'],
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            if hasattr(self.env.task, '_enable_task_mask_obs'):
                if self.env.task._enable_task_mask_obs:
                    onehot_size = self.env.task.get_multi_task_info()["onehot_size"]
                    obs_batch = self.running_mean_std(obs_batch)

                    # recover the onehoe code to un-normalized state
                    onehot_code = obs_batch[:, -onehot_size:] # we assume the last part is the onehot code
                    mean = self.running_mean_std.running_mean[-onehot_size:]
                    var = self.running_mean_std.running_var[-onehot_size:]
                    unnormalized_onehot_code = (onehot_code * var.sqrt() + mean)

                    obs_batch[:, -onehot_size:] = unnormalized_onehot_code
                else:
                    obs_batch = self.running_mean_std(obs_batch)
            else:
                obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def run(self):

        if self._eval:
            self.run_eval()
        else:
            super().run()

        return
    
    def run_eval(self):
        is_determenistic = self.is_determenistic
        num_envs = self.env.num_envs
        num_trials = num_envs
        assert num_envs == num_trials
        num_repeat = 3

        print("evaluating policy: {} trials".format(num_envs))

        eval_res = {}

        for i in range(num_repeat):
            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
            
            done_indices = []
            normal_done_indices = []

            games_played = 0
            games_success = 0
            sum_success_precision = 0

            games_fail_because_terminate = 0
            games_fail = 0

            has_collected = torch.zeros(num_envs, device=self.device)

            while games_played < num_trials:
                obs_dict = self.env_reset(normal_done_indices)
                action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)

                normal_all_done_indices = done.nonzero(as_tuple=False)
                normal_done_indices = normal_all_done_indices[::self.num_agents, 0]

                done = torch.where(has_collected == 0, done, torch.zeros_like(done))
                has_collected[done == 1] += 1

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents, 0]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    success_done = info['success'][done_indices]
                    percision_done = info['precision'][done_indices]
                    terminate_done = info['terminate'][done_indices]

                    # compute number of success
                    success_indices = success_done.nonzero(as_tuple=False)[:, 0]
                    success_count = len(success_indices)
                    games_success += success_count

                    terminate_count = torch.logical_and(terminate_done, success_done == 0).sum().cpu().item()
                    fail_count = done_count - success_count - terminate_count

                    games_fail_because_terminate += terminate_count
                    games_fail += fail_count

                    if success_count > 0:
                        sum_success_precision += percision_done[success_indices].sum().cpu().numpy()
  
                # self._post_step(info)
            
            success_rate = games_success / games_played
            if games_success > 0:
                mean_success_precision = sum_success_precision / games_success
            else:
                mean_success_precision = 0
            
            curr_exp_name = "ObjectSet_{}_{}".format("test", i)
            eval_res[curr_exp_name] = {
                "num_trials": games_played,
                "success_trials": games_success,
                "success_rate": success_rate,
                "success_precision": mean_success_precision,
                "fail_trials": games_fail,
                "fail_trials_because_terminate": games_fail_because_terminate,
                "fail_trials_total": games_fail + games_fail_because_terminate,
            }

            print(curr_exp_name)
            print(eval_res[curr_exp_name])

        # save metrics
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = os.path.dirname(self._checkpoint_fn)
        ckp_name = os.path.basename(self._checkpoint_fn)
        save_dir = os.path.join(folder_name, "../metrics", ckp_name[:-4])
        if self._eval_task != "":
            save_dir = os.path.join(save_dir, self._eval_task)
        os.makedirs(save_dir, exist_ok=True)
        json.dump(eval_res, open(os.path.join(save_dir, "metrics_{}.json".format(time)), 'w'))
        print("save at {}".format(os.path.join(save_dir, "metrics_{}.json".format(time))))

        return
