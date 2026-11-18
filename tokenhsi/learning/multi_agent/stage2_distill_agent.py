import copy
import json
import os

import torch
from torch.nn import functional as F

from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.distill_replay import DistillReplay
from learning.multi_agent.stage2_teachers import Stage2Teachers
from utils.stage2_distill_spec import teacher_probability


class Stage2DistillAgent(MAAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        task = self.vec_env.env.task
        self.distill = config['distillation']
        self.model.requires_grad_(False)
        actor = self.model.a2c_network
        actor.coordination.requires_grad_(True)
        actor.action_head.requires_grad_(True)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                          lr=config['learning_rate'])
        self.teachers = Stage2Teachers(task, self.distill['teacher_before'],
            self.distill['teacher_coupled'], output_path=self.experiment_dir)
        self.distill_contract = dict(settings=copy.deepcopy(self.distill),
            before_sha256=self.teachers.before.metadata['sha256'],
            coupled_sha256=self.teachers.coupled.metadata['sha256'])
        self.replay = DistillReplay(self.distill['replay_capacity'], task.get_obs_size(),
                                    task.num_agents*task.get_action_size(), self.ppo_device)

    def get_stats_weights(self):
        weights = super().get_stats_weights()
        weights['distillation_contract'] = self.distill_contract
        return weights

    def get_full_state_weights(self):
        weights = super().get_full_state_weights()
        weights['distillation_state'] = dict(replay=self.replay.state_dict())
        return weights

    def set_full_state_weights(self, weights):
        if weights.get('distillation_contract') != self.distill_contract:
            raise ValueError('Distillation resume requires the same teachers and training settings')
        super().set_full_state_weights(weights)
        self.replay.load_state_dict(weights['distillation_state']['replay'])

    def _student_means(self, obs):
        return self.model.a2c_network.eval_actor(self.running_mean_std(obs))[0]

    def _collect(self, obs, resets, beta):
        task = self.vec_env.env.task
        teacher_scenes = before_people = coupled_people = 0
        reward_sum = 0.
        with torch.no_grad():
            for _ in range(self.distill['rollout_steps']):
                obs = self.env_reset(resets)
                raw = obs['obs']
                target = self.teachers.means(raw, task._members > 1)
                student = self._student_means(raw)
                self.replay.add(raw, target.reshape(task.num_envs, -1))
                use_teacher = torch.rand(task.num_envs, device=self.ppo_device) < beta
                mask = use_teacher[:, None].expand(-1, task.num_agents).reshape(-1, 1)
                action = torch.where(mask, target, student).clamp(-1., 1.)
                obs, reward, done, _ = self.env_step(action)
                resets = done.nonzero(as_tuple=False)[::task.num_agents, 0]
                teacher_scenes += int(use_teacher.sum())
                coupled_people += int((task._members > 1).sum())
                before_people += int((task._members == 1).sum())
                reward_sum += float(reward.mean())
        scenes = task.num_envs*self.distill['rollout_steps']
        return obs, resets, dict(teacher_scenes=teacher_scenes, student_scenes=scenes-teacher_scenes,
            before_labels=before_people, coupled_labels=coupled_people,
            reward_mean=reward_sum/self.distill['rollout_steps'])

    def _imitate(self):
        obs, target = self.replay.sample(self.distill['minibatch_scenes'])
        target = target.reshape(-1, self.vec_env.env.task.get_action_size())
        with torch.no_grad():
            before = float(F.mse_loss(self._student_means(obs), target))
        losses = []
        for _ in range(self.distill['updates_per_iteration']):
            batch, labels = self.replay.sample(self.distill['minibatch_scenes'])
            loss = F.mse_loss(self._student_means(batch), labels.reshape(-1, target.shape[-1]))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.grad_norm)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError('Non-finite distillation loss or gradient')
            self.optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            after = float(F.mse_loss(self._student_means(obs), target))
        return dict(mse_before=before, mse_after=after, train_mse=sum(losses)/len(losses))

    def train(self):
        self.model.eval()
        self.running_mean_std.eval()
        obs, resets = self.env_reset(), []
        output = os.path.join(self.nn_dir, self.config['name'])
        task = self.vec_env.env.task
        while self.epoch_num < self.max_epochs:
            self.epoch_num += 1
            beta = teacher_probability(self.epoch_num, self.distill)
            obs, resets, rollout = self._collect(obs, resets, beta)
            metrics = dict(iteration=self.epoch_num, teacher_probability=beta,
                           replay_scenes=self.replay.size, **rollout, **self._imitate())
            self.frame += task.num_envs*task.num_agents*self.distill['rollout_steps']
            metrics['frame'] = self.frame
            for key, value in metrics.items():
                self.writer.add_scalar('distill/'+key, value, self.epoch_num)
            with open(os.path.join(self.experiment_dir, 'distill_metrics.jsonl'), 'a') as stream:
                stream.write(json.dumps(metrics)+'\n')
            print('[distill] '+json.dumps(metrics), flush=True)
            if self.save_freq > 0 and self.epoch_num%self.save_freq == 0:
                self.save(output+'_'+str(self.epoch_num).zfill(8))
        self.save(output)
        self.writer.flush()
        return 0., self.epoch_num
