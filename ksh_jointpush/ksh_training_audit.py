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
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi import HumanoidMAStage2JointPushRsi
    original_sample = HumanoidMAStage2JointPushRsi._sample_episode_graph
    def sample(self, ids):
        result = original_sample(self, ids)
        if len(ids) and not getattr(self, '_ksh_rsi_audited', False):
            groups = self._groups[ids]
            interaction = self._interaction[ids]
            same = groups[:, :, None] == groups[:, None, :]
            mixed = same & (interaction[:, :, None] != interaction[:, None, :])
            mode = self.cfg['env']['stage2JointPush']['coupled_rsi_selection']
            if mode == 'group':
                assert not mixed.any()
            elif mode in ('independent_agents_v1','balanced_group_ternary_v1'):
                assert mixed.any()
            self._ksh_rsi_audited = True
            self._ksh_rsi_report = dict(mode=mode, environments=len(ids),
                push_people=int(interaction.sum()), loco_people=int((~interaction).sum()),
                scenes_with_mixed_group=int(mixed.flatten(1).any(-1).sum()))
            if mode == 'balanced_group_ternary_v1':
                members = same.sum(-1)
                pushes = (same & interaction[:,None,:]).sum(-1)
                leaders = (groups == torch.arange(groups.shape[1],device=groups.device)[None]) & (members > 1)
                self._ksh_rsi_report['shared_group_categories'] = dict(
                    all_push=int((leaders & (pushes == members)).sum()),
                    mixed=int((leaders & (pushes > 0) & (pushes < members)).sum()),
                    all_loco=int((leaders & (pushes == 0)).sum()))
            print('KSH_TRAINING_RSI_VERIFIED', json.dumps(self._ksh_rsi_report), flush=True)
        return result
    HumanoidMAStage2JointPushRsi._sample_episode_graph = sample
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
            report['rsi_initialization'] = self.vec_env.env.task._ksh_rsi_report
            Path(self.experiment_dir,'startup_verification.json').write_text(json.dumps(report,indent=2))
            self.save(str(Path(self.experiment_dir,'nn/Stage2JointPushRsi_startup')))
            print('KSH_STAGE2_UPDATES_VERIFIED',json.dumps(report),flush=True)
        return result
    MAAgent.calc_gradients=gradients
    MAAgent.train_epoch=epoch
