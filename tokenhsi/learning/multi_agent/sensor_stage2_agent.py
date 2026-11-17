import hashlib
from pathlib import Path

import torch
from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.sensor_stage2_network import freeze_backbone, transfer


TRAINING_CONTRACT = 'square_sensor_frozen_actor_coordination_v1'


class SensorStage2Agent(MAAgent):
    def _init_before(self, task):
        args = task.cfg['args']
        self.sensor_base = None
        if args.resume <= 0:
            path = Path(task.cfg['experiment']['transfer']['checkpoint'])
            checkpoint = torch.load(path, map_location=self.ppo_device, weights_only=False)
            transfer(self.model, checkpoint)
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
            self.sensor_base = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    epoch=checkpoint['epoch'])
        freeze_backbone(self.model)
        for group in self.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
        self.running_mean_std.eval()

    def set_train(self):
        super().set_train()
        self.model.a2c_network.coordination.eval()
        self.running_mean_std.eval()

    def play_steps(self):
        task = self.vec_env.env.task
        if getattr(task, '_bridge_curriculum', None):
            task._bridge_curriculum_update = max(0, self.epoch_num-1)
        return super().play_steps()

    def get_stats_weights(self):
        result = super().get_stats_weights()
        result['sensor_base'] = self.sensor_base
        result['sensor_training_contract'] = TRAINING_CONTRACT
        return result

    def set_weights(self, weights):
        if weights.get('sensor_training_contract') != TRAINING_CONTRACT:
            raise ValueError('Sensor resume requires the square-sensor training contract; start fresh from Distill1.5k')
        super().set_weights(weights)
        self.sensor_base = weights['sensor_base']
        self.running_mean_std.eval()
