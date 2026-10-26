"""Full fine-tuning with an unchanged 30D object tokenizer architecture."""
import json
from pathlib import Path
import torch
from learning.multi_agent.ma_agent import MAAgent
from utils.hold_at_ft import transfer_78k

class HoldAtFTAgent(MAAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self._inclusive_max_epochs = True
        task = self.vec_env.env.task
        source = task.cfg['env']['holdAtFT']['source_checkpoint']
        checkpoint = torch.load(source,map_location=self.ppo_device,weights_only=False)
        transfer_78k(self.model,checkpoint)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'],strict=True)
        if self.normalize_value:
            self.value_mean_std.load_state_dict(checkpoint['reward_mean_std'],strict=True)
        self.model.requires_grad_(True)
        Path(self.experiment_dir,'hold_at_transfer.json').write_text(json.dumps(dict(source=source,source_epoch=checkpoint['epoch'],object_input_size=30,strict_model_load=True,normalizers_loaded=True,trainable='all_layers'),indent=2))

    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)
        if not hasattr(self, '_hold_at_gradient_report'):
            report = {}
            for branch in ('actor_encoder', 'critic_encoder'):
                params = dict(self.model.named_parameters())
                report[branch] = sum(float(p.grad.detach().abs().sum()) for name,p in params.items() if f'{branch}.tokenizers.1.' in name and p.grad is not None)
            if not all(value > 0 for value in report.values()):
                raise RuntimeError(f'Object tokenizer has no gradient: {report}')
            self._hold_at_gradient_report = report
            Path(self.experiment_dir,'hold_at_gradients.json').write_text(json.dumps(report,indent=2))
            print('[hold_at_ft] object tokenizer gradient L1:', report,flush=True)
