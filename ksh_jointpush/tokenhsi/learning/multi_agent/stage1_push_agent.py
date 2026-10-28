from copy import deepcopy
import json
from pathlib import Path

import torch

from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.ma_players import MAPlayerContinuous
from learning.multi_agent.stage1_push_network import transfer
from utils.stage1_push import contract, check_contract, check_navigation_transfer


class Stage1PushAgent(MAAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        task = self.vec_env.env.task
        self._push_spec = task.cfg['env']['stage1Push']
        source = None
        if task.cfg['args'].resume <= 0:
            warm_start = self._push_spec.get('warm_start_checkpoint')
            if warm_start:
                weights = torch.load(warm_start, map_location=self.ppo_device, weights_only=False)
                check_navigation_transfer(weights, task.cfg['env'])
                super().set_weights(weights)
                report = dict(source_checkpoint=warm_start, source_epoch=weights['epoch'], optimizer='fresh')
            else:
                source = torch.load(self._push_spec['source_checkpoint'], map_location=self.ppo_device, weights_only=False)
                amp_stats, report = transfer(self.model, source)
                self.running_mean_std.load_state_dict(source['running_mean_std'], strict=True)
                self._amp_input_mean_std.load_state_dict(amp_stats, strict=True)
                if self.normalize_value:
                    self.value_mean_std.load_state_dict(source['reward_mean_std'], strict=True)
            Path(self.experiment_dir, 'push_transfer_report.json').write_text(json.dumps(report, indent=2))
        self.teacher = None
        if self._push_spec['teacher_kl_coef'] > 0:
            if source is None:
                source = torch.load(self._push_spec['source_checkpoint'], map_location=self.ppo_device, weights_only=False)
            self.teacher = deepcopy(self.model.a2c_network)
            teacher_state = self.teacher.state_dict()
            for name, value in teacher_state.items():
                key = 'a2c_network.'+name
                if key in source['model']:
                    saved = source['model'][key]
                    if saved.shape == value.shape:
                        value.copy_(saved)
                    elif name.endswith('category_embed.weight'):
                        value[:4].copy_(saved)
            self.teacher.load_state_dict(teacher_state)
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        self.running_mean_std.eval()
        self._amp_input_mean_std.eval()
        self._teacher_kl_sum = 0.
        self._teacher_kl_count = 0

    def set_train(self):
        super().set_train()
        if hasattr(self, '_push_spec'):
            self.running_mean_std.eval()
            self._amp_input_mean_std.eval()

    def _actor_preservation_loss(self, obs, mu, sigma):
        coefficient = self._push_spec['teacher_kl_coef']
        if coefficient == 0:
            return mu.new_zeros(())
        m = self.num_agents
        packet = obs[:, -5*m:].reshape(-1, m, 5)
        old = ((packet[..., 1] != 4) | (packet[..., 0] == 0)).all(-1)
        if not old.any():
            return mu.sum()*0
        with torch.no_grad():
            teacher_mu, teacher_log_sigma = self.teacher.eval_actor(obs[old])
            teacher_sigma = teacher_log_sigma.exp()
        student_mu = mu.reshape(len(obs), m, -1)[old].reshape_as(teacher_mu)
        student_sigma = sigma.reshape(len(obs), m, -1)[old].reshape_as(teacher_sigma)
        kl = (torch.log(student_sigma/teacher_sigma) +
              (teacher_sigma.square()+(teacher_mu-student_mu).square())/(2*student_sigma.square())-.5).sum(-1).mean()
        self._teacher_kl_sum += float(kl.detach())
        self._teacher_kl_count += 1
        return coefficient*kl

    def get_stats_weights(self):
        weights = super().get_stats_weights()
        weights['stage1_push_contract'] = contract(self.vec_env.env.task.cfg['env'])
        return weights

    def set_weights(self, weights):
        task = self.vec_env.env.task
        check_contract(weights, task.cfg['env'])
        result = super().set_weights(weights)
        task._push_curriculum_epoch = weights.get('epoch', 0)
        return result

    def train_epoch(self):
        self.vec_env.env.task._push_curriculum_epoch = self.epoch_num
        return super().train_epoch()

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        self.writer.add_scalar('losses/teacher_old_task_kl', self._teacher_kl_sum/max(self._teacher_kl_count, 1), frame)
        self._teacher_kl_sum = 0.
        self._teacher_kl_count = 0


class Stage1PushPlayer(MAPlayerContinuous):
    def restore(self, path):
        weights = torch.load(path, map_location=self.device, weights_only=False)
        check_contract(weights, self.env.task.cfg['env'])
        return super().restore(path)
