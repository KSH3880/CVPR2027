"""Verify the first real PPO updates without changing optimization."""
import hashlib,json
from pathlib import Path
import torch


def digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def install():
    from learning.multi_agent.ma_agent import MAAgent
    original_gradient=MAAgent.calc_gradients
    original_epoch=MAAgent.train_epoch
    def gradients(self,*args,**kwargs):
        result=original_gradient(self,*args,**kwargs)
        if not getattr(self,'_ksh_gradient_verified',False):
            values=[p.grad.detach().abs().sum() for p in self.model.a2c_network.coordination.parameters() if p.grad is not None]
            value=float(torch.stack(values).sum()) if values else 0.
            assert value>0 and torch.isfinite(torch.tensor(value))
            self._ksh_gradient_verified=True
            print('KSH_STAGE2_COORDINATION_GRAD',value,flush=True)
        return result
    def epoch(self):
        if not hasattr(self,'_ksh_audit_start'):
            self._ksh_audit_start=self.epoch_num
            self._ksh_actor=digest(self.model.a2c_network.actor_encoder)
            self._ksh_rms=digest(self.running_mean_std)
            self._ksh_coord=digest(self.model.a2c_network.coordination)
        result=original_epoch(self)
        if self.epoch_num==self._ksh_audit_start+1:
            report=dict(epoch=self.epoch_num,actor_encoder_unchanged=digest(self.model.a2c_network.actor_encoder)==self._ksh_actor,scene_normalizer_unchanged=digest(self.running_mean_std)==self._ksh_rms,coordination_updated=digest(self.model.a2c_network.coordination)!=self._ksh_coord,gradient_verified=self._ksh_gradient_verified,optimizer_states=len(self.optimizer.state))
            assert all(report[k] for k in ('actor_encoder_unchanged','scene_normalizer_unchanged','coordination_updated','gradient_verified'))
            Path(self.experiment_dir,'startup_verification.json').write_text(json.dumps(report,indent=2))
            self.save(str(Path(self.experiment_dir,'nn/Stage2JointPushRsi_startup')))
            print('KSH_STAGE2_UPDATES_VERIFIED',json.dumps(report),flush=True)
        return result
    MAAgent.calc_gradients=gradients
    MAAgent.train_epoch=epoch
