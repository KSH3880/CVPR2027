"""Real teacher/student checkpoint inference for CPU pool integration checks."""
from pathlib import Path
import torch
import yaml
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder
from learning.amp_models import ModelAMPContinuous
from learning.push_kd_losses import teacher_observation, normalize_policy, gaussian_forward_kl


class PolicyProbe:
    def __init__(self):
        torch.set_num_threads(2)
        params = yaml.safe_load(Path('tokenhsi/data/cfg/train/rlg/push_kd.yaml').read_text())['params']['network']
        self.models, self.stats = [], []
        for sizes, dim, path in [([20,38,42,27],354,'teacher_stage1.pth'), ([20,38,42,27,42],397,'push_init.pth')]:
            edges=[0]
            for size in sizes: edges.append(edges[-1]+size)
            info=dict(onehot_size=len(sizes),tota_subtask_obs_size=sum(sizes),each_subtask_obs_size=sizes,
                each_subtask_obs_indx=edges,each_subtask_name=['traj','sit','carry','climb','push'][:len(sizes)],enable_task_mask_obs=True)
            builder=AMPTransformerMultiTaskBuilder();builder.load(params)
            model=ModelAMPContinuous.Network(builder.build('amp',actions_num=32,input_shape=(dim,),num_seqs=1,
                amp_input_shape=(10*(129+len(sizes)),),self_obs_size=223,task_obs_size=dim-223,multi_task_info=info,device='cpu'))
            checkpoint=torch.load(Path('weights')/path,map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['model'],strict=True)
            model.eval().requires_grad_(False)
            rms=RunningMeanStd((dim,)).eval()
            rms.load_state_dict(checkpoint['running_mean_std'])
            self.models.append(model);self.stats.append(rms)

    @torch.no_grad()
    def action(self, obs, kd):
        mu, sigma = self.models[1].a2c_network.eval_actor(normalize_policy(obs,self.stats[1],5),obs)
        assert torch.isfinite(mu).all()
        assert not (obs[:,392] > .5).any(), 'TRAJ selected'
        kl=0.
        if kd.any():
            raw=teacher_observation(obs[kd])
            tm,ts=self.models[0].a2c_network.eval_actor(normalize_policy(raw,self.stats[0],4),raw)
            kl=gaussian_forward_kl(mu[kd],sigma.exp().expand_as(mu)[kd],tm,ts.exp().expand_as(tm)).item()
            assert abs(kl) < 1e-5, kl
        return mu,kl
