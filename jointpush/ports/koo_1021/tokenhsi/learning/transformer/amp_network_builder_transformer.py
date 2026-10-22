from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch.network_builder import NetworkBuilder

from learning.amp_network_builder import AMPBuilder
from tokenhsi.learning.transformer.composer import Composer

import torch
import torch.nn as nn

class AMPTransformerMultiTaskBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPTransformerMultiTaskBuilder.UnifiedNetworkClass(self.params, **kwargs)
        return net
    
    class UnifiedNetworkClass(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs["self_obs_size"]
            self.task_obs_size = kwargs["task_obs_size"]

            # support multi task masking machanism
            self.multi_task_info = kwargs.get('multi_task_info', None)
            assert self.multi_task_info is not None
            assert self.multi_task_info["enable_task_mask_obs"] is True

            # create pre-defined task obs mask matrix
            self.task_obs_onehot_size = self.multi_task_info["onehot_size"] # equal to num_tasks
            self.task_obs_tota_size = self.multi_task_info["tota_subtask_obs_size"]
            self.task_obs_each_size = self.multi_task_info["each_subtask_obs_size"]
            self.task_obs_each_indx = self.multi_task_info["each_subtask_obs_indx"]

            self.each_subtask_names = self.multi_task_info["each_subtask_name"]

            super().__init__(params, **kwargs)

            del self.actor_mlp, self.mu, self.mu_act # delete useless networks
            
            self.device = kwargs["device"]
            self.dof_action_size = kwargs['actions_num']

            self.type_id = params["network_structure_id"]
            if self.type_id == 2:
                self._build_Transformer(params, **kwargs)
            else:
                raise NotImplementedError

            return

        def _build_Transformer(self, params, **kwargs):

            num_features = params["transformer"]["num_features"]
            num_tokens = 1 + len(self.task_obs_each_size) + 1 # self + multiple tasks + mu
            drop_ratio = 0.0 # using dropout will make training failed [BUG]
            tokenizer_units = params["transformer"]["tokenizer_units"]

            print("build tokenizer for self obs")
            mlp_args = {
                'input_size' : self.self_obs_size, 
                'units' : tokenizer_units + [num_features], 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.self_encoder = self._build_mlp(**mlp_args)

            self.task_encoder = nn.ModuleList()
            for idx, i in enumerate(self.task_obs_each_size):
                print("build tokenizer for subtask obs with size {}".format(i))
                mlp_args = {
                    'input_size' : i, 
                    'units' : tokenizer_units + [num_features], 
                    'activation' : self.activation, 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.task_encoder.append(self._build_mlp(**mlp_args))

            mlp_init = self.init_factory.create(**{"name": "default"})
            for nets in [self.self_encoder, self.task_encoder]:
                for m in nets.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)
            
            self.weight_token = nn.Parameter(torch.zeros(1, 1, num_features)) # extra token

            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, num_features))
            self.pos_drop = nn.Identity() # nn.Dropout(p=drop_ratio)
            self.use_pos_embed = params["transformer"].get("use_pos_embed", True)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=num_features,
                nhead=params["transformer"]["layer_num_heads"],
                dim_feedforward=params["transformer"]["layer_dim_feedforward"],
                dropout=drop_ratio,
                activation='relu',
                batch_first=True,
            )

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params["transformer"]["num_layers"])

            # weight init
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.weight_token, std=0.02)

            # extra mlp
            mlp_args = {
                'input_size': num_features,
                'units': params["transformer"]["extra_mlp_units"],
                'activation': self.activation,
                'dense_func': torch.nn.Linear,
            }

            output_size = kwargs['actions_num']
            comp_act = "identity"
            
            self.composer = Composer(mlp_args, output_size=output_size, activation=comp_act)

            return
        
        def _eval_Transformer(self, obs, not_normalized_obs):
            B = obs.shape[0]

            self_obs = obs[..., :self.self_obs_size]
            self_token = self.self_encoder(self_obs).unsqueeze(1) # (B, 1, num_feats)

            task_obs = obs[..., self.self_obs_size:]
            task_obs_real = task_obs[..., :self.task_obs_tota_size] # exclude onehot code
            
            task_token = torch.stack([self.task_encoder[i](task_obs_real[:, self.task_obs_each_indx[i]:self.task_obs_each_indx[i + 1]]) for i in range(self.task_obs_onehot_size)], dim=1)

            # [1, 1, num_feats] -> [B, 1, num_feats]
            weight_token = self.weight_token.expand(B, -1, -1)

            x = torch.cat((weight_token, self_token, task_token), dim=1)  # [B, num_tokens, num_feats]

            if self.use_pos_embed:
                x = self.pos_drop(x + self.pos_embed)

            # compute key padding mask

            src_key_padding_mask = torch.ones((B, x.shape[1]), dtype=torch.bool, device=x.device) # init to all True
            src_key_padding_mask[:, [0, 1]] = False

            task_obs_onehot_idx = task_obs[..., self.task_obs_tota_size:].max(dim=-1)[1] + 2
            task_obs_onehot_idx_mask = nn.functional.one_hot(task_obs_onehot_idx, num_classes=self.task_obs_onehot_size + 2).bool()
            
            src_key_padding_mask[task_obs_onehot_idx_mask] = False

            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

            weights = self.composer(x[:, 0])

            return weights

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            not_normalized_obs = obs_dict['not_normalized_obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs, not_normalized_obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output
        
        def eval_actor(self, obs, not_normalized_obs):

            if self.is_continuous and self.space_config['fixed_sigma']:
                
                mu = self.eval_composer(obs, not_normalized_obs)

                sigma = self.sigma_act(self.sigma)

                return mu, sigma

            else:
                raise NotImplementedError
    
        def eval_composer(self, obs, not_normalized_obs):
            if self.type_id == 2:
                weights = self._eval_Transformer(obs, not_normalized_obs)
            else:
                raise NotImplementedError

            return weights
