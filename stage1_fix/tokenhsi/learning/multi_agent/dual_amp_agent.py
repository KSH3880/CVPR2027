"""TeamHOI full+masked AMP on scene-batched MA PPO.

Adaptation of sail-sg/TeamHOI@6fdc9885f8a9c82854be99adca6f2570e67ac63a.
Copyright2026 Sea AI Lab, MIT; see third_party/teamhoi/LICENSE.
"""
import copy
import math
import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.replay_buffer import ReplayBuffer
from learning.multi_agent.ma_agent import MAAgent
from utils.dual_amp import (mask_history,blend_weight,blend_rewards,outside_history,
                           style_reward,discriminator_loss,check_dual_checkpoint)


class DualAMPAgent(MAAgent):
    def __init__(self,base_name,config):
        super().__init__(base_name,config)
        task=self.vec_env.env.task
        if not getattr(task,'_dual_amp',False) or not self._scene_policy:
            raise ValueError('Dual AMP requires a dedicated clean-scene lift configuration')
        if not self._normalize_amp_input:raise ValueError('TeamHOI dual AMP requires separate normalizers')
        self._dual_cfg=copy.deepcopy(task._dual_amp_cfg)
        self._dual_metadata=dict(config=self._dual_cfg,amp_shape=list(self._amp_observation_space.shape),
                                 native_features=129,version=1)
        # TeamHOI uses one statistic per window feature, separately per discriminator.
        self._amp_input_mean_std=RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        self._amp_masked_input_mean_std=RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

    def set_eval(self):
        super().set_eval()
        if hasattr(self,'_amp_masked_input_mean_std'):self._amp_masked_input_mean_std.eval()

    def set_train(self):
        super().set_train()
        if hasattr(self,'_amp_masked_input_mean_std'):self._amp_masked_input_mean_std.train()

    def get_stats_weights(self):
        weights=super().get_stats_weights()
        weights['amp_masked_input_mean_std']=self._amp_masked_input_mean_std.state_dict()
        weights['dual_amp_metadata']=self._dual_metadata
        return weights

    def set_stats_weights(self,weights):
        check_dual_checkpoint(weights,self._dual_metadata)
        super().set_stats_weights(weights)
        self._amp_masked_input_mean_std.load_state_dict(weights['amp_masked_input_mean_std'])

    def set_weights(self,weights):
        check_dual_checkpoint(weights,self._dual_metadata)
        return super().set_weights(weights)

    def _preproc_amp_obs(self,obs):return self._amp_input_mean_std(obs)

    def _preproc_amp_obs_masked(self,obs):return self._amp_masked_input_mean_std(mask_history(obs))

    def _build_amp_buffers(self):
        super()._build_amp_buffers()
        history=self.vec_env.env.task._num_amp_obs_steps
        self.experience_buffer.tensor_dict['amp_cond']=torch.zeros(
            self.experience_buffer.obs_base_shape+(history,),device=self.ppo_device)
        self.tensor_list.append('amp_cond')
        self._masked_demo_buffer=ReplayBuffer(int(self.config['amp_obs_demo_buffer_size']),self.ppo_device)

    def _fetch_amp_obs_demo(self,count):
        return self.vec_env.env.task.fetch_amp_obs_demo_full(count)

    def _init_amp_demo_buf(self):
        super()._init_amp_demo_buf()
        for _ in range(math.ceil(self._masked_demo_buffer.get_buffer_size()/self._amp_batch_size)):
            self._masked_demo_buffer.store({'amp_obs':self.vec_env.env.task.fetch_amp_obs_demo(self._amp_batch_size)})

    def _update_amp_demos(self):
        super()._update_amp_demos()
        self._masked_demo_buffer.store({'amp_obs':self.vec_env.env.task.fetch_amp_obs_demo(self._amp_batch_size)})

    def _record_extra_amp_step(self,step,infos):
        self.experience_buffer.update_data('amp_cond',step,infos['amp_cond'])

    def _rollout_amp_rewards(self,obs):
        return self._calc_amp_rewards(obs,self.experience_buffer.tensor_dict['amp_cond'])

    def _calc_amp_rewards(self,obs,condition=None):
        with torch.no_grad():
            if condition is None:
                condition=self.vec_env.env.task._amp_cond_buf.reshape(-1,self.vec_env.env.task._num_amp_obs_steps)
                condition=condition[:obs.numel()//obs.shape[-1]].reshape(*obs.shape[:-1],-1)
            full=style_reward(self.model.a2c_network.eval_disc(self._preproc_amp_obs(obs)),self._disc_reward_scale)
            masked=style_reward(self.model.a2c_network.eval_disc_masked(self._preproc_amp_obs_masked(obs)),self._disc_reward_scale)
            weight=blend_weight(condition,self._dual_cfg['distance_threshold'],self._dual_cfg['sharpness'])
            return dict(disc_rewards=full*(1-weight)+masked*weight,
                        full_disc_rewards=full,masked_disc_rewards=masked,amp_blend_weight=weight)

    def _extra_amp_dataset(self,batch):
        return {key:batch[key] for key in ('amp_cond','amp_cond_replay','amp_obs_demo_masked')}

    def _sample_amp_replay(self,batch,count):
        if self._amp_replay_buffer.get_total_count()==0:
            return {key:batch[key].reshape(count,-1) for key in ('amp_obs','amp_cond')}
        return self._amp_replay_buffer.sample(count)

    def _prepare_extra_amp_batch(self,batch,replay):
        scenes,agents,width=batch['amp_obs'].shape
        batch['amp_cond_replay']=replay['amp_cond'].reshape(scenes,agents,-1)
        batch['amp_obs_demo_masked']=self._masked_demo_buffer.sample(scenes*agents)['amp_obs'].reshape(scenes,agents,width)

    def _store_amp_batch_replay(self,batch):
        data={key:batch[key].reshape(-1,batch[key].shape[-1]) for key in ('amp_obs','amp_cond')}
        n=len(data['amp_obs']);capacity=self._amp_replay_buffer.get_buffer_size()
        keep=torch.arange(n,device=self.ppo_device)
        if self._amp_replay_buffer.get_total_count()>capacity:
            keep=keep[torch.rand(n,device=self.ppo_device)<self._amp_replay_keep_prob]
        if len(keep)>capacity:keep=keep[torch.randperm(len(keep),device=self.ppo_device)[:capacity]]
        if len(keep):self._amp_replay_buffer.store({key:value[keep] for key,value in data.items()})

    def _prepare_amp_training_inputs(self,obs,replay,demo,batch):
        count=len(obs)
        condition=batch['amp_cond'].reshape(-1,batch['amp_cond'].shape[-1])[:count]
        condition_replay=batch['amp_cond_replay'].reshape(-1,batch['amp_cond_replay'].shape[-1])[:len(replay)]
        outside=outside_history(condition,self._dual_cfg['distance_threshold'])
        outside_replay=outside_history(condition_replay,self._dual_cfg['distance_threshold'])
        masked_demo=batch['amp_obs_demo_masked'].reshape(-1,obs.shape[-1])[:len(demo)]
        result=dict(amp_obs_masked=self._preproc_amp_obs_masked(obs),
            amp_obs_replay_masked=self._preproc_amp_obs_masked(replay),
            amp_obs_demo_masked=self._preproc_amp_obs_masked(masked_demo).requires_grad_(True))
        minimum=self._dual_cfg['full_min_samples']
        if int(outside.sum())>=minimum and int(outside_replay.sum())>=minimum:
            # Same outside-only selection and randomized demo count as TeamHOI.
            result.update(amp_obs=self._preproc_amp_obs(obs[outside]),
                amp_obs_replay=self._preproc_amp_obs(replay[outside_replay]))
            chosen=torch.randperm(len(demo),device=demo.device)[:int(outside.sum())]
            result['amp_obs_demo']=self._preproc_amp_obs(demo[chosen]).requires_grad_(True)
        else:result.update(amp_obs=None,amp_obs_replay=None,amp_obs_demo=None)
        return result

    def _compute_discriminator_loss(self,result,inputs):
        network=self.model.a2c_network
        masked=discriminator_loss(torch.cat((result['disc_agent_logit_masked'],result['disc_agent_replay_logit_masked']),0),
            result['disc_demo_logit_masked'],inputs['amp_obs_demo_masked'],network,True,self.config)
        info=dict(masked)
        info.update({'masked_'+key:value for key,value in masked.items()})
        if result['disc_agent_logit'] is not None:
            full=discriminator_loss(torch.cat((result['disc_agent_logit'],result['disc_agent_replay_logit']),0),
                result['disc_demo_logit'],inputs['amp_obs_demo'],network,False,self.config)
            info['disc_loss']=full['disc_loss']+masked['disc_loss']
            info['full_disc_loss']=full['disc_loss'].detach()
            info['full_updated']=inputs['amp_obs_demo'].new_tensor(1.)
        else:
            info['full_disc_loss']=masked['disc_loss'].detach()*0
            info['full_updated']=masked['disc_loss'].new_tensor(0.)
        return info

    def _record_train_batch_info(self,batch,info):
        super()._record_train_batch_info(batch,info)
        for key in ('amp_blend_weight','full_disc_rewards','masked_disc_rewards'):
            info[key]=batch[key]

    def _log_train_info(self,info,frame):
        super()._log_train_info(info,frame)
        from rl_games.algos_torch import torch_ext
        for key in ('full_disc_loss','masked_disc_loss','full_updated','amp_blend_weight','full_disc_rewards','masked_disc_rewards'):
            value=info[key]
            value=torch_ext.mean_list(value) if isinstance(value,list) else value.float().mean()
            self.writer.add_scalar('dual_amp/'+key,float(value),frame)
