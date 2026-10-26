"""Restore the full CLIMB FT optimizer state, then apply the requested lower LR."""
import json
from pathlib import Path
import torch
from learning.multi_agent.chair_ft_agent import ChairFTAgent

class ClimbContinueAgent(ChairFTAgent):
    def restore(self, fn):
        super().restore(fn)
        source=torch.load(fn,map_location='cpu',weights_only=False)
        for name,value in self.model.state_dict().items():
            assert torch.equal(value.detach().cpu(),source['model'][name]),name
        for key,module in [('running_mean_std',self.running_mean_std),('amp_input_mean_std',self._amp_input_mean_std),('reward_mean_std',self.value_mean_std)]:
            for name,value in module.state_dict().items():
                assert torch.equal(value.detach().cpu(),source[key][name]),(key,name)
        optimizer=self.optimizer.state_dict()
        assert optimizer['state'].keys()==source['optimizer']['state'].keys()
        for key,state in optimizer['state'].items():
            for name,value in state.items():
                expected=source['optimizer']['state'][key][name]
                assert torch.equal(value.detach().cpu(),expected.detach().cpu()) if torch.is_tensor(value) else value==expected
        lr=float(self.config['learning_rate'])
        assert lr==5e-6
        self.last_lr=lr;self.update_lr(lr)
        assert all(group['lr']==lr for group in self.optimizer.param_groups)
        assert self.epoch_num==source['epoch'] and self.frame==source['frame']
        report=dict(source=str(Path(fn).resolve()),restored_epoch=self.epoch_num,restored_frame=self.frame,max_epoch=self.max_epochs,additional_iterations=self.max_epochs-self.epoch_num,learning_rate=self.last_lr,optimizer_group_lrs=[group['lr'] for group in self.optimizer.param_groups],model_exact=True,normalizers_exact=True,adam_moments_exact=True,all_layers_trainable=all(p.requires_grad for p in self.model.parameters()))
        Path(self.experiment_dir,'continuation_validation.json').write_text(json.dumps(report,indent=2))
        print('CLIMB_CONTINUATION',json.dumps(report),flush=True)
