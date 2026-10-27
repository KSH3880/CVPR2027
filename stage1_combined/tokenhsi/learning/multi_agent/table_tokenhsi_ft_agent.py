"""Reference adaptation mapped to our actor/critic architecture."""
from pathlib import Path
import json
import torch
from learning.multi_agent.ma_agent import MAAgent
from utils.hold_at_ft import transfer_78k


class TableTokenHSIFTAgent(MAAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._inclusive_max_epochs=True
        task=self.vec_env.env.task
        fresh={k:v.detach().clone() for k,v in self.model.state_dict().items() if '_disc' in k}
        if not fresh:raise RuntimeError('Fresh discriminator not captured')
        source=task.cfg['env']['holdAtFT']['source_checkpoint']
        checkpoint=torch.load(source,map_location=self.ppo_device,weights_only=False)
        transfer_78k(self.model,checkpoint)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        if self.normalize_value:self.value_mean_std.load_state_dict(checkpoint['reward_mean_std'],strict=True)
        # Reference adaptation trains a fresh task discriminator; keep its fresh RMS too.
        self.model.load_state_dict(fresh,strict=False)
        for name,p in self.model.named_parameters():
            train=('actor_encoder.tokenizers.1.' in name or 'critic_' in name or 'value_head' in name or '_disc' in name)
            p.requires_grad_(train)
        for group in self.optimizer.param_groups:group['params']=[p for p in group['params'] if p.requires_grad]
        report={name:bool(p.requires_grad) for name,p in self.model.named_parameters()}
        Path(self.experiment_dir,'native_adaptation_trainable.json').write_text(json.dumps(report,indent=2))
        Path(self.experiment_dir,'native_adaptation.json').write_text(json.dumps(dict(source=source,source_epoch=78000,actor_backbone_frozen=True,actor_object_tokenizer_trainable=True,critic_trainable=True,discriminator_fresh=True,amp_normalizer_fresh=True,human_normalizer_frozen=True,sigma_frozen=True),indent=2))

    def set_train(self):
        super().set_train()
        self.model.a2c_network.actor_encoder.eval()
        self.model.a2c_network.actor_encoder.tokenizers[1].train()
        self.running_mean_std.running_mean_std[0].eval()

    def calc_gradients(self,input_dict):
        super().calc_gradients(input_dict)
        if not hasattr(self,'_native_grad_report'):
            report={}
            for branch in ('actor_encoder','critic_encoder'):
                report[branch]=sum(float(p.grad.detach().abs().sum()) for n,p in self.model.named_parameters() if f'{branch}.tokenizers.1.' in n and p.grad is not None)
            assert all(v>0 for v in report.values()),report
            self._native_grad_report=report
            Path(self.experiment_dir,'native_gradients.json').write_text(json.dumps(report,indent=2))
            print('[Table TokenHSI] gradients',report,flush=True)
