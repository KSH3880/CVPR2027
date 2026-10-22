import json
from pathlib import Path
import torch
from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.ma_players import MAPlayerContinuous
from learning.multi_agent.chair_ft_network import transfer,freeze_except_chair

class ChairFTAgent(MAAgent):
    def __init__(self,base_name,config):
        super().__init__(base_name,config)
        self._inclusive_max_epochs=True
        task=self.vec_env.env.task
        checkpoint=torch.load(task.cfg['env']['chairFT']['source_checkpoint'],map_location=self.ppo_device,weights_only=False)
        loaded=transfer(self.model,checkpoint)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'],strict=True)
        if self.normalize_value:self.value_mean_std.load_state_dict(checkpoint['reward_mean_std'],strict=True)
        names=freeze_except_chair(self.model)
        for group in self.optimizer.param_groups:group['params']=[p for p in group['params'] if p.requires_grad]
        self._frozen={name:p.detach().cpu().clone() for name,p in self.model.named_parameters() if not p.requires_grad}
        self._initial_chair={name:p.detach().cpu().clone() for name,p in self.model.named_parameters() if p.requires_grad}
        self._normalizers={'scene':self.running_mean_std,'amp':self._amp_input_mean_std}
        if self.normalize_value:self._normalizers['value']=self.value_mean_std
        self._normalizer_initial={kind:{name:t.detach().cpu().clone() for name,t in module.state_dict().items()} for kind,module in self._normalizers.items()}
        self._gradients={};self._checked=False
        Path(self.experiment_dir,'chair_transfer.json').write_text(json.dumps({'source_epoch':checkpoint['epoch'],'loaded_tensors':len(loaded),'trainable_names':names,'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),'frozen_parameter_tensors':len(self._frozen)},indent=2))
        self.set_train()
    def set_train(self):
        super().set_train()
        self.model.eval()
        for enc in (self.model.a2c_network.actor_encoder,self.model.a2c_network.critic_encoder):
            enc.tokenizers[1].chair.train()
        if hasattr(self,'running_mean_std'):self.running_mean_std.eval()
        if hasattr(self,'_amp_input_mean_std'):self._amp_input_mean_std.eval()
        if self.normalize_value and hasattr(self,'value_mean_std'):self.value_mean_std.eval()
    def calc_gradients(self,input_dict):
        super().calc_gradients(input_dict)
        for name,p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:self._gradients[name]=max(self._gradients.get(name,0.),float(p.grad.detach().abs().sum()))
    def verify_frozen(self):
        changed=[name for name,p in self.model.named_parameters() if name in self._frozen and not torch.equal(p.detach().cpu(),self._frozen[name])]
        if changed:raise RuntimeError('Frozen weights changed: '+str(changed))
        deltas={name:float((p.detach().cpu()-self._initial_chair[name]).abs().max()) for name,p in self.model.named_parameters() if p.requires_grad}
        branches={}
        for enc in ['actor_encoder','critic_encoder']:
            for kind in ['chair']:
                prefix=enc+'.tokenizers.1.'+kind+'.'
                branches[enc+'.'+kind]=sum(v for k,v in self._gradients.items() if prefix in k)
        if not all(v>0 for v in branches.values()):raise RuntimeError('Dead tokenizer branch: '+str(branches))
        for kind,module in self._normalizers.items():
            for name,value in module.state_dict().items():
                if not torch.equal(value.detach().cpu(),self._normalizer_initial[kind][name]):raise RuntimeError('Normalizer changed: '+kind+'.'+name)
        Path(self.experiment_dir,'chair_freeze_validation.json').write_text(json.dumps({'normalizers_unchanged':list(self._normalizers),'frozen_tensors_unchanged':len(self._frozen),'branch_gradient_l1':branches,'trainable_max_deltas':deltas,'epoch':self.epoch_num},indent=2))
    def train_epoch(self):
        result=super().train_epoch()
        if hasattr(self,'_frozen') and (not self._checked or self.epoch_num%100==0):self.verify_frozen();self._checked=True
        return result
    def get_stats_weights(self):
        result=super().get_stats_weights();result['chair_ft_contract']=self.vec_env.env.task.cfg['env']['chairFT'];return result

class ChairFTPlayer(MAPlayerContinuous):
    def restore(self,path):
        weights=torch.load(path,map_location=self.device,weights_only=False)
        if weights.get('chair_ft_contract',{}).get('shape_encoder')!='bbox35_mlp':raise ValueError('Expected compact 35D chair checkpoint')
        return super().restore(path)
