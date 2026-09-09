from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch.network_builder import NetworkBuilder

from learning.amp_network_builder import AMPBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder

from tokenhsi.learning.transformer.composer import Composer

from utils.torch_utils import load_checkpoint

import torch
import torch.nn as nn

class AMPTransformerMultiTaskCompBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        if self.params["network_structure_id"] == 2:
            net = AMPTransformerMultiTaskCompBuilder.Network(self.params, **kwargs)
        else:
            raise NotImplementedError

        return net

    class Network(AMPTransformerMultiTaskBuilder.UnifiedNetworkClass):

        def __init__(self, params, **kwargs):

            kwargs["multi_task_info"]["enable_task_mask_obs"] = True # fool the basic class
            assert params["network_structure_id"] == 2, "only support transformer now"

            # whether we mask out tokens relevant to the comp task
            self.use_prior_knowledge = params.get("use_prior_knowledge", False)

            # whether we add a new trainable MLP after the extra feats to generate compensate actions
            self.use_compensate_actions = params.get("use_compensate_actions", False)

            # whether we add internal adaptation layers after the extra feats
            self.use_internal_adaptation = params.get("use_internal_adaptation", False)
            self.apply_adapter_on_actions = params.get("apply_adapter_on_actions", False)

            if self.apply_adapter_on_actions:
                assert self.use_internal_adaptation

            assert int(self.use_compensate_actions) + int(self.use_internal_adaptation) <= 1

            super().__init__(params, **kwargs)

            ckp = load_checkpoint(kwargs["hrl_checkpoint"], device=self.device)
            state_dict_loaded = ckp["model"]

            ##### self token
            key = "self_encoder"
            any_loaded = False
            state_dict = self.self_encoder.state_dict()
            for k in state_dict_loaded.keys():
                if key in k:
                    any_loaded = True
                    pnn_dict_key = k.split(key + ".")[1]
                    state_dict[pnn_dict_key].copy_(state_dict_loaded[k])
            assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"

            self.self_encoder.requires_grad_(False)
            self.self_encoder.eval()

            ##### multi task tokens
            state_dict_loaded_task_encoders = {}
            for k in state_dict_loaded.keys():
                if "task_encoder" in k:
                    idx = int(k.split(".")[2])
                    if idx not in state_dict_loaded_task_encoders.keys():
                        state_dict_loaded_task_encoders[idx] = []
                    state_dict_loaded_task_encoders[idx].append(state_dict_loaded[k])

            self.basic_num_tasks = len(state_dict_loaded_task_encoders.keys())
            self.basic_task_obs_each_size = [v[0].shape[-1] for k, v in state_dict_loaded_task_encoders.items()]
            self.basic_task_obs_tota_size = sum(self.basic_task_obs_each_size)
            self.basic_task_obs_each_indx = torch.cumsum(torch.tensor([0] + self.basic_task_obs_each_size), dim=0).to(self.device)

            self.basic_task_obs_running_mean_stds_needed = nn.ModuleList()
            self.basic_self_running_mean_std = RunningMeanStd(self.self_obs_size)
            self.basic_self_running_mean_std.running_mean = ckp["running_mean_std"]["running_mean"][:self.self_obs_size].clone()
            self.basic_self_running_mean_std.running_var = ckp["running_mean_std"]["running_var"][:self.self_obs_size].clone()
            self.basic_self_running_mean_std.count = ckp["running_mean_std"]["count"].clone()
            
            for i, s in enumerate(self.task_obs_each_size[1:], 1): # exclude the new task obs
                if s in self.basic_task_obs_each_size:
                    id = self.basic_task_obs_each_size.index(s)

                    rms = RunningMeanStd(s)
                    rms.running_mean = ckp["running_mean_std"]["running_mean"][self.self_obs_size:][self.basic_task_obs_each_indx[id]:self.basic_task_obs_each_indx[id + 1]].clone()
                    rms.running_var = ckp["running_mean_std"]["running_var"][self.self_obs_size:][self.basic_task_obs_each_indx[id]:self.basic_task_obs_each_indx[id + 1]].clone()
                    rms.count = ckp["running_mean_std"]["count"].clone()

                    self.basic_task_obs_running_mean_stds_needed.append(rms)
                
                    key = "task_encoder.{}".format(id)
                    any_loaded = False
                    state_dict = self.task_encoder[i].state_dict()
                    for k in state_dict_loaded.keys():
                        if key in k:
                            any_loaded = True
                            pnn_dict_key = k.split(key + ".")[1]
                            state_dict[pnn_dict_key].copy_(state_dict_loaded[k])

                    assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"
                    
                    self.task_encoder[i].requires_grad_(False)
                    self.task_encoder[i].eval()

                else:
                    raise NotImplementedError
            
            ##### extra output token
            self.weight_token.requires_grad_(False)
            self.weight_token.copy_(state_dict_loaded["a2c_network.weight_token"])

            ##### backbone
            key = "transformer_encoder"
            any_loaded = False
            state_dict = self.transformer_encoder.state_dict()
            for k in state_dict_loaded.keys():
                if key in k:
                    any_loaded = True
                    pnn_dict_key = k.split(key + ".")[1]
                    state_dict[pnn_dict_key].copy_(state_dict_loaded[k])
            assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"

            self.transformer_encoder.requires_grad_(False)
            self.transformer_encoder.eval()
            
            ##### extra mlp
            key = "composer"
            any_loaded = False
            state_dict = self.composer.state_dict()
            for k in state_dict_loaded.keys():
                if key in k:
                    pnn_dict_key = k.split(key + ".")[1]
                    state_dict[pnn_dict_key].copy_(state_dict_loaded[k])

            self.composer.requires_grad_(False)
            self.composer.eval()

            #################### trainable networks

            self.new_net_input_size = self.self_obs_size + self.task_obs_each_size[0]

            num_features = self.weight_token.shape[-1]

            if self.use_compensate_actions:
                mlp_args = {
                    'input_size': num_features,
                    'units': params["plugin_units"],
                    'activation': self.activation,
                    'dense_func': torch.nn.Linear,
                }
                output_size = self.dof_action_size * 2

                last_layer_zero_init = False
                self.w_act = torch.nn.Sigmoid()
                self.extra_act_mlp = Composer(mlp_args, output_size=output_size, activation="identity", last_layer_all_zero_init=last_layer_zero_init)

            if self.use_internal_adaptation:
                plugin_units = params["plugin_units"]
                if self.apply_adapter_on_actions:
                    plugin_units.append(self.dof_action_size)
                mlp_args = {
                    'input_size' : num_features,
                    'units' : plugin_units,
                    'activation' : "None", # [don't use any activation func] 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.internal_adapt_mlp = self._build_mlp(**mlp_args)

                for m in self.internal_adapt_mlp.modules(): # zero init
                    if isinstance(m, nn.Linear):
                        torch.nn.init.zeros_(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            del self.critic_mlp # recreate critic_mlp
            self.critic_mlp = nn.Sequential()
            mlp_args = {
                'input_size' : self.new_net_input_size, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

                for m in self.critic_mlp.modules():         
                    if isinstance(m, nn.Linear):
                        # mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)    

            return
        
        def eval_critic(self, obs):
            return super().eval_critic(obs[..., :self.new_net_input_size])
        
        def _eval_Transformer(self, obs, not_normalized_obs):

            B = obs.shape[0]

            # obs >> new_task_token
            new_task_token = self.task_encoder[0](obs[..., self.self_obs_size:self.new_net_input_size]).unsqueeze(1) # the only place where the obs is used

            # not_normalized_obs >> self_token
            self.basic_self_running_mean_std.training = False
            self_obs = self.basic_self_running_mean_std(not_normalized_obs[..., :self.self_obs_size])
            self_token = self.self_encoder(self_obs).unsqueeze(1) # (B, 1, num_feats)

            # not_normalized_obs >> relevant task tokens
            task_token = []
            for i in range(self.task_obs_onehot_size - 1):
                curr_task_obs = not_normalized_obs[..., self.self_obs_size:][..., self.task_obs_each_indx[i + 1]:self.task_obs_each_indx[i + 2]]
                self.basic_task_obs_running_mean_stds_needed[i].training = False
                curr_task_obs = self.basic_task_obs_running_mean_stds_needed[i](curr_task_obs)
                task_token.append(self.task_encoder[i + 1](curr_task_obs))
            task_token = torch.stack(task_token, dim=1)

            # [1, 1, num_feats] -> [B, 1, num_feats]
            weight_token = self.weight_token.expand(B, -1, -1)

            x = torch.cat((weight_token, self_token, new_task_token, task_token), dim=1)

            # compute key padding mask
            src_key_padding_mask = torch.ones((B, x.shape[1]), dtype=torch.bool, device=x.device) # init to all True
            src_key_padding_mask[:, [0, 1]] = False

            src_key_padding_mask[:, 2] = False

            if self.use_prior_knowledge:
                if "traj" in self.each_subtask_names:
                    src_key_padding_mask[:, self.each_subtask_names.index("traj") + 2] = False
                else:
                    src_key_padding_mask[:, 3:] = False

            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

            embeddings = x[:, 0]

            if self.use_internal_adaptation:

                for j, op in enumerate(self.composer.actors[:-1]):
                    embed = self.internal_adapt_mlp[j]
                    if isinstance(embed, torch.nn.Identity):
                        embeddings = op(embeddings)
                    else:
                        embeddings = op(embeddings) + embed(embeddings)
            
                if self.apply_adapter_on_actions:
                    action = self.composer.act(self.composer.actors[-1](embeddings) + self.internal_adapt_mlp[-2](embeddings)) # self.internal_adapt_mlp[-1] is Identity
                else:
                    action = self.composer.act(self.composer.actors[-1](embeddings))

            else:

                meta_action = self.composer(embeddings)

                if self.use_compensate_actions:
                    output = self.extra_act_mlp(embeddings).view(-1, 2, self.dof_action_size)
                    new_action = output[:, 0]
                    w = self.w_act(output[:, 1])
                
                    action = new_action + w * meta_action
                else:
                    action = meta_action
            
            return action
