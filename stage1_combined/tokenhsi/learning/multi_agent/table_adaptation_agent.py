"""Isolated policy adaptation; all pretrained encoders and scene RMS are frozen."""
from pathlib import Path
import json
import torch
from learning.multi_agent.ma_agent import MAAgent


ADAPTER_PREFIXES=('a2c_network.action_residual.','a2c_network.adapt_object_tokenizer.')


def transfer_policy(model,checkpoint):
    if int(checkpoint['epoch'])!=78000:raise ValueError('Fresh adaptation requires original 78k')
    expected=set(model.state_dict());source=set(checkpoint['model'])
    extra=expected-source
    if source-expected or not extra or any(not k.startswith(ADAPTER_PREFIXES) for k in extra):
        raise ValueError('Unexpected checkpoint keys')
    missing=model.load_state_dict(checkpoint['model'],strict=False)
    assert set(missing.missing_keys)==extra and not missing.unexpected_keys
    model.a2c_network.initialize_adapter()


class TableAdaptationAgent(MAAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._inclusive_max_epochs=True
        task=self.vec_env.env.task;source=task.cfg['env']['holdAtFT']['source_checkpoint']
        fresh={k:v.detach().clone() for k,v in self.model.state_dict().items() if '_disc' in k}
        checkpoint=torch.load(source,map_location=self.ppo_device,weights_only=False)
        transfer_policy(self.model,checkpoint)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        if self.normalize_value:self.value_mean_std.load_state_dict(checkpoint['reward_mean_std'],strict=True)
        self.model.load_state_dict(fresh,strict=False)
        for name,p in self.model.named_parameters():
            p.requires_grad_(name.startswith(ADAPTER_PREFIXES) or 'value_head.' in name or '_disc' in name)
        for group in self.optimizer.param_groups:group['params']=[p for p in group['params'] if p.requires_grad]
        self._adapter_grad_calls=0
        self.running_mean_std.eval()
        flags={n:bool(p.requires_grad) for n,p in self.model.named_parameters()}
        Path(self.experiment_dir,'adaptation_trainable.json').write_text(json.dumps(flags,indent=2))
        Path(self.experiment_dir,'adaptation.json').write_text(json.dumps(dict(source=source,source_epoch=78000,mode=self.model.a2c_network.adaptation_mode,all_pretrained_encoders_frozen=True,base_action_head_frozen=True,scene_normalizer_frozen=True,sigma_frozen=True,critic_value_head_trainable=True,fresh_discriminator=True,tokenizer_initialization='copy original78k'),indent=2))
        rows=min(8,task.num_envs)
        torch.save(dict(root=task._humanoid_root_states[:rows].detach().cpu(),model_ids=task._object_model_ids[:rows].detach().cpu()),Path(self.experiment_dir,'construction_sample.pth'))

    def set_train(self):
        super().set_train();net=self.model.a2c_network
        net.actor_encoder.eval();net.critic_encoder.eval();net.action_head.eval()
        self.running_mean_std.eval()
        if net.adaptation_mode=='residual':net.action_residual.train()
        else:net.adapt_object_tokenizer.train()

    def calc_gradients(self,input_dict):
        super().calc_gradients(input_dict)
        if self._adapter_grad_calls<2:
            report={n:float(p.grad.detach().abs().sum()) for n,p in self.model.named_parameters() if n.startswith(ADAPTER_PREFIXES) and p.grad is not None}
            assert report and sum(report.values())>0,report
            assert all(p.grad is None for n,p in self.model.named_parameters() if 'actor_encoder.' in n or 'critic_encoder.' in n or 'action_head.' in n), 'Frozen encoder received gradients'
            self._adapter_grad_calls+=1
            Path(self.experiment_dir,f'adaptation_gradient_{self._adapter_grad_calls}.json').write_text(json.dumps(report,indent=2))
            print('[Table adaptation] gradient',self._adapter_grad_calls,sum(report.values()),flush=True)

    def _log_train_info(self,info,frame):
        super()._log_train_info(info,frame)
        net=self.model.a2c_network
        if net.adaptation_mode=='residual':self.writer.add_scalar('adaptation/residual_rms',getattr(net,'last_residual_rms',0.),frame)

    def restore(self,fn):
        super().restore(fn)
        saved=torch.load(fn,map_location='cpu',weights_only=False)
        assert self.epoch_num==saved['epoch'] and self.frame==saved.get('frame',0)
        assert all(torch.equal(v.detach().cpu(),saved['model'][k]) for k,v in self.model.state_dict().items())
        for module,key in ((self.running_mean_std,'running_mean_std'),(self._amp_input_mean_std,'amp_input_mean_std'),(self.value_mean_std,'reward_mean_std')):
            assert all(torch.equal(v.detach().cpu(),saved[key][k]) for k,v in module.state_dict().items())
        actual=self.optimizer.state_dict();expected=saved['optimizer']
        assert actual['param_groups']==expected['param_groups']
        for index,values in expected['state'].items():
            for key,value in values.items():
                got=actual['state'][index][key]
                assert torch.equal(got.detach().cpu(),value) if torch.is_tensor(value) else got==value
        Path(self.experiment_dir,'resume_restore.json').write_text(json.dumps(dict(checkpoint=str(fn),epoch=self.epoch_num,frame=self.frame,model_exact=True,all_normalizers_exact=True,optimizer_exact=True,optimizer_entries=len(expected['state'])),indent=2))
        print('[Table adaptation] full-state resume verified',self.epoch_num,flush=True)
