"""Evaluation path with both TeamHOI discriminator/window normalizers."""
import copy
import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.torch_utils import load_checkpoint
from utils.dual_amp import mask_history,blend_rewards,style_reward,check_dual_checkpoint


class DualAMPPlayer(MAPlayerContinuous):
    def _build_net(self,config):
        super()._build_net(config)
        shape=self._amp_observation_space
        self._amp_input_mean_std=RunningMeanStd(shape).to(self.device).eval()
        self._amp_masked_input_mean_std=RunningMeanStd(shape).to(self.device).eval()
        self._dual_cfg=copy.deepcopy(self.env.task._dual_amp_cfg)
        self._dual_metadata=dict(config=self._dual_cfg,amp_shape=list(shape),native_features=129,version=1)

    def restore(self,path):
        if path!='Base':
            checkpoint=load_checkpoint(path,self.device)
            check_dual_checkpoint(checkpoint,self._dual_metadata)
            self._amp_masked_input_mean_std.load_state_dict(checkpoint['amp_masked_input_mean_std'])
        return super().restore(path)

    def _preproc_amp_obs(self,obs):return self._amp_input_mean_std(obs)

    def _calc_amp_rewards(self,obs):
        with torch.no_grad():
            task=self.env.task
            condition=task._amp_cond_buf.reshape(-1,task._num_amp_obs_steps)[:obs.numel()//obs.shape[-1]].reshape(*obs.shape[:-1],-1)
            full=style_reward(self.model.a2c_network.eval_disc(self._preproc_amp_obs(obs)),self._disc_reward_scale)
            masked=style_reward(self.model.a2c_network.eval_disc_masked(self._amp_masked_input_mean_std(mask_history(obs))),self._disc_reward_scale)
            return {'disc_rewards':blend_rewards(full,masked,condition,self._dual_cfg['distance_threshold'],self._dual_cfg['sharpness'])}
