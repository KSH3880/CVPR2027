"""Strict original78k transfer; all original layers train, no shape/key mapping."""
from pathlib import Path
import json,torch
from learning.multi_agent.ma_agent import MAAgent
class Climb30Agent(MAAgent):
    def __init__(self,base_name,config):
        super().__init__(base_name,config);self._inclusive_max_epochs=True
        task=self.vec_env.env.task;source=task.cfg['env']['chairFT']['source_checkpoint']
        checkpoint=torch.load(source,map_location=self.ppo_device,weights_only=False)
        assert task.get_scene_entity_sizes()[1]==30
        assert set(self.model.state_dict())==set(checkpoint['model'])
        self.model.load_state_dict(checkpoint['model'],strict=True)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'],strict=True)
        if self.normalize_value:self.value_mean_std.load_state_dict(checkpoint['reward_mean_std'],strict=True)
        assert all(torch.equal(v,checkpoint['model'][k].to(v.device)) for k,v in self.model.state_dict().items())
        self.model.requires_grad_(True);self.gradients={};self.checked=False
        for name in ['actor_encoder','critic_encoder']:
            layer=getattr(self.model.a2c_network,name).tokenizers[1][0]
            assert layer.in_features==30
        record=dict(source=source,source_epoch=checkpoint['epoch'],strict_keys_and_shapes=True,all_model_tensors_exact=True,original_model_tensor_count=len(checkpoint['model']),object_input_size=30,no_added_parameters=True,all_layers_trainable=True,normalizers_strictly_restored=True)
        Path(self.experiment_dir,'original30_transfer.json').write_text(json.dumps(record,indent=2));print('ORIGINAL30_TRANSFER',json.dumps(record),flush=True);self.set_train()
    def restore(self,fn):
        super().restore(fn)
        source=torch.load(fn,map_location='cpu',weights_only=False)
        assert self.epoch_num==source['epoch'] and self.frame==source['frame']
        for name,value in self.model.state_dict().items():assert torch.equal(value.detach().cpu(),source['model'][name]),name
        for key,module in [('running_mean_std',self.running_mean_std),('amp_input_mean_std',self._amp_input_mean_std),('reward_mean_std',self.value_mean_std)]:
            for name,value in module.state_dict().items():assert torch.equal(value.detach().cpu(),source[key][name]),(key,name)
        actual=self.optimizer.state_dict()['state'];expected=source['optimizer']['state'];assert actual.keys()==expected.keys()
        for key,state in actual.items():
            for name,value in state.items():
                other=expected[key][name]
                assert torch.equal(value.detach().cpu(),other.detach().cpu()) if torch.is_tensor(value) else value==other
        report=dict(checkpoint=str(Path(fn).resolve()),restored_epoch=self.epoch_num,restored_frame=self.frame,max_epochs=self.max_epochs,save_frequency=self.save_freq,model_exact=True,normalizers_exact=True,optimizer_exact=True)
        Path(self.experiment_dir,'original30_resume.json').write_text(json.dumps(report,indent=2));print('ORIGINAL30_RESUME',json.dumps(report),flush=True)
    def calc_gradients(self,input_dict):
        super().calc_gradients(input_dict)
        for name in ['actor_encoder','critic_encoder']:
            p=getattr(self.model.a2c_network,name).tokenizers[1][0].weight
            if p.grad is not None:self.gradients[name]=max(self.gradients.get(name,0.),float(p.grad.abs().sum()))
    def train_epoch(self):
        result=super().train_epoch()
        if not self.checked:
            assert all(self.gradients.get(n,0)>0 for n in ['actor_encoder','critic_encoder']),self.gradients
            Path(self.experiment_dir,'original30_gradient.json').write_text(json.dumps(self.gradients,indent=2));print('ORIGINAL30_GRADIENT',self.gradients,flush=True);self.checked=True
        return result
