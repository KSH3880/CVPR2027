from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning.amp_network_builder import AMPBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder

from tokenhsi.learning.transformer.composer import Composer

from utils.torch_utils import load_checkpoint

import torch
import torch.nn as nn
import copy

class AMPTransformerMultiTaskAdaptBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPTransformerMultiTaskAdaptBuilder.Network(self.params, **kwargs)
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

            # MA_NOFREEZE=1 : 사전학습 가중치는 그대로 불러오되 **동결만 푼다**.
            # 회피는 보행 자체를 바꾸는 일인데 transformer/composer 가 얼려 있으면
            # 정책이 쓸 수 있는 여지가 토크나이저와 adapt mlp 로 제한된다.
            # 한 변수만 바꾸기 위해 로드 경로는 건드리지 않는다.
            import os as _osf
            _frz = _osf.environ.get("MA_NOFREEZE", "0") != "1"
            self._ma_freeze = _frz
            self._ma_old_carry_only = _osf.environ.get("MA_OLD_CARRY_ONLY", "0") == "1"
            self._ma_teammate_mask = _osf.environ.get("MA_TEAMMATE_MASK", "0") == "1"
            self._mstop_residual_arch = _osf.environ.get("MSTOP_RESIDUAL_ARCH", "0") == "1"
            self._mstop_residual = _osf.environ.get("MSTOP_RESIDUAL", "0") == "1"
            self._mstop_stop_head_arch = _osf.environ.get("MSTOP_STOP_HEAD_ARCH", "0") == "1"
            self._mstop_stop_head = _osf.environ.get("MSTOP_STOP_HEAD", "0") == "1"
            self._mstop_frozen_a = _osf.environ.get("MSTOP_FROZEN_A", "0") == "1"
            self._mstop_diag = _osf.environ.get("MSTOP_DIAG", "0") == "1"
            self._mstop_parity = bool(_osf.environ.get("MSTOP_PARITY_PATH"))
            self._mstop_residual_scale = float(
                _osf.environ.get("MSTOP_RESIDUAL_SCALE", "2"))
            self._mstop_arm_scale = float(_osf.environ.get("MSTOP_ARM_SCALE", "1"))
            self._mstop_resume_residual_scale = float(
                _osf.environ.get("MSTOP_RESUME_RESIDUAL_SCALE", "1"))
            if self._mstop_residual_scale <= 0:
                raise ValueError("MSTOP_RESIDUAL_SCALE must be positive")
            if self._mstop_residual and self._mstop_stop_head:
                raise ValueError("MSTOP_RESIDUAL and MSTOP_STOP_HEAD are mutually exclusive")
            if not 0 <= self._mstop_arm_scale <= 1:
                raise ValueError("MSTOP_ARM_SCALE must be in [0, 1]")
            if not 0 <= self._mstop_resume_residual_scale <= 1:
                raise ValueError("MSTOP_RESUME_RESIDUAL_SCALE must be in [0, 1]")
            if _frz:
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

            self.how_many_new_tasks = 0
            for task_name in self.each_subtask_names:
                if "new" in task_name:
                    self.how_many_new_tasks += 1
            
            for i, s in enumerate(self.task_obs_each_size[self.how_many_new_tasks:], self.how_many_new_tasks): # exclude the new task obs
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
                    
                    if self._ma_freeze:
                        self.task_encoder[i].requires_grad_(False)
                        self.task_encoder[i].eval()

                else:
                    raise NotImplementedError
            
            ##### extra output token
            if self._ma_freeze:
                self.weight_token.requires_grad_(False)
                self.pos_embed.requires_grad_(False)
            # 동결을 풀면 leaf 가 grad 를 요구해서 in-place copy_ 가 막힌다.
            # 값을 싣는 것과 학습 가능 여부는 별개다.
            with torch.no_grad():
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

            if self._ma_freeze:
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

            if self._ma_freeze:
                self.composer.requires_grad_(False)
                self.composer.eval()

            #################### trainable networks

            self.major_task_name = self.multi_task_info["major_task_name"]
            self.each_subtask_name = self.multi_task_info["each_subtask_name"]
            self.has_extra = self.multi_task_info["has_extra"]

            self.major_task_obs_size = self.task_obs_each_size[self.each_subtask_name.index("old_{}".format(self.major_task_name))]
            self.new_major_id = self.each_subtask_name.index("new_{}".format(self.major_task_name))
            self.old_major_id = self.each_subtask_name.index("old_{}".format(self.major_task_name))
            
            if self.has_extra:
                # extra 토큰이 여럿일 수 있다 (ma+steer 는 teammate + steering 둘).
                # 이름이 new_extra 로 시작하는 것을 전부 모은다. 하나뿐이면
                # 리스트 길이가 1 이라 기존 동작과 같다.
                self.extra_ids = [i for i, nm in enumerate(self.each_subtask_name)
                                  if nm.startswith("new_extra")]
                assert self.extra_ids, self.each_subtask_name
                self.extra_id = self.extra_ids[0]
                self.extra_task_obs_size = self.task_obs_each_size[self.extra_id]

            ######## the first trainable task tokenizer
            s = self.task_obs_each_size[self.new_major_id]
            if s in self.basic_task_obs_each_size:
                id = self.basic_task_obs_each_size.index(s)

                self.new_task_trainable_rms = RunningMeanStd(s)

                self.new_task_trainable_rms.running_mean = ckp["running_mean_std"]["running_mean"][self.self_obs_size:][self.basic_task_obs_each_indx[id]:self.basic_task_obs_each_indx[id + 1]]
                self.new_task_trainable_rms.running_var = ckp["running_mean_std"]["running_var"][self.self_obs_size:][self.basic_task_obs_each_indx[id]:self.basic_task_obs_each_indx[id + 1]]
                self.new_task_trainable_rms.count = ckp["running_mean_std"]["count"].clone()
        
                key = "task_encoder.{}".format(id)
                any_loaded = False
                state_dict = self.task_encoder[self.new_major_id].state_dict()
                for k in state_dict_loaded.keys():
                    if key in k:
                        any_loaded = True
                        pnn_dict_key = k.split(key + ".")[1]
                        state_dict[pnn_dict_key].copy_(state_dict_loaded[k])
                assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"

            else:
                raise NotImplementedError
            
            num_features = self.weight_token.shape[-1]
            
            if self.has_extra:
                ######## the second trainable task tokenizer
                ### load extra module for extra obs
                mlp_args = {
                    'input_size' : self.extra_task_obs_size, 
                    'units' : params["new_input_tokenizer_units"] + [num_features], 
                    'activation' : self.activation, 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                # extra 마다 자기 토크나이저를 만든다. 입력 크기가 서로 다르므로
                # mlp_args 를 매번 새로 만들어야 한다.
                for _eid in self.extra_ids:
                    _a = dict(mlp_args); _a['input_size'] = self.task_obs_each_size[_eid]
                    self.task_encoder[_eid] = self._build_mlp(**_a)

                # 원본은 bias 만 0 이고 weight 는 랜덤이다. 그러면 extra 토큰이 학습된
                # 트랜스포머에 **랜덤 사영**을 주입해서 시작부터 정책을 흔든다
                # (실측: 같은 체크포인트에서 zero 0.862 vs live 0.757, 10 포인트).
                # 그러면 live 는 자기가 만든 잡음을 복구하는 데 예산을 쓰고, zero 와의
                # 비교가 "토큰이 도움이 되나" 가 아니라 "잡음을 복구하고도 남나" 가 된다.
                #
                # MA_TOKENIZER_ZERO=1 이면 **마지막 층만** 0 으로 둔다. 그러면 출력이
                # 정확히 0 이라 시작 시점 정책이 zero 모드와 **동일**하고, 마지막 층의
                # gradient 는 이전 층의 활성값이라 0 이 아니어서 학습은 정상으로 된다.
                # (전 레이어를 0 으로 두면 gradient 도 0 이라 영원히 못 배운다 --
                #  steer 의 STEER_TOKENIZER_ZERO 는 그 용도, 즉 '토큰 완전 제거' 대조군이다.)
                import os as _os
                _tz = _os.environ.get("MA_TOKENIZER_ZERO", "0") == "1"
                for _eid in self.extra_ids:
                    enc = self.task_encoder[_eid]
                    # **마지막 활성함수를 반드시 걷어낸다.** _build_mlp 는 units 의 모든
                    # 항목 뒤에 활성을 붙이는데, 여기서는 units 마지막이 곧 토큰 차원이라
                    # 출력 뒤에 ReLU 가 남는다. 그 상태로 마지막 Linear 를 0 으로 두면
                    # 활성 전 값이 정확히 0 이고 PyTorch 의 ReLU'(0)=0 이라
                    # **backward 가 0 을 곱해 토크나이저 전체가 영원히 안 배운다.**
                    # 실측: ma·masteer 전 런에서 extra 토크나이저가 초기값 그대로였고
                    # (max = 1/sqrt(fan_in) 정확히 일치) grad 합이 정확히 0 이었다.
                    # 그 탓에 "teammate 토큰 효과 없음(Δ=+0.004)" 은 토큰이 늘 0 이었던 것이다.
                    # 토큰 임베딩에 ReLU 를 두면 음수 성분을 못 갖는 제약도 생기는데,
                    # weight_token·self_token 은 그런 제약이 없다.
                    if isinstance(enc, nn.Sequential) and len(enc) > 0 and \
                            not isinstance(enc[-1], nn.Linear):
                        enc[len(enc) - 1] = nn.Identity()
                    _lin = [m for m in enc.modules() if isinstance(m, nn.Linear)]
                    for m in _lin:
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)
                    if _tz and _lin:
                        torch.nn.init.zeros_(_lin[-1].weight)

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

            if self._ma_teammate_mask:
                self.task_encoder[self.extra_ids[0]].requires_grad_(False)
            if self._ma_old_carry_only:
                self.task_encoder[self.new_major_id].requires_grad_(False)

            if self._mstop_residual_arch:
                self.stop_residual = nn.Sequential(
                    nn.Linear(num_features + 6, 256),
                    nn.ELU(),
                    nn.Linear(256, self.dof_action_size),
                )
                torch.nn.init.zeros_(self.stop_residual[-1].weight)
                torch.nn.init.zeros_(self.stop_residual[-1].bias)
                if not self._mstop_residual:
                    self.stop_residual.requires_grad_(False)

            if self._mstop_stop_head_arch:
                if self.use_compensate_actions:
                    raise ValueError("MSTOP_STOP_HEAD does not support compensate actions")
                self.stop_composer = copy.deepcopy(self.composer)
                self.stop_composer.requires_grad_(self._mstop_stop_head)
                if self.use_internal_adaptation:
                    self.stop_internal_adapt_mlp = copy.deepcopy(self.internal_adapt_mlp)
                    self.stop_internal_adapt_mlp.requires_grad_(self._mstop_stop_head)

            if self._mstop_frozen_a:
                self.self_encoder.requires_grad_(False)
                self.task_encoder.requires_grad_(False)
                self.transformer_encoder.requires_grad_(False)
                self.composer.requires_grad_(False)
                self.weight_token.requires_grad_(False)
                self.pos_embed.requires_grad_(False)
                self.pos_embed.requires_grad_(False)
                if self.use_internal_adaptation:
                    self.internal_adapt_mlp.requires_grad_(False)
                if self._mstop_stop_head:
                    self.task_encoder[self.extra_ids[0]].requires_grad_(True)

            return
        
        def _compose_action(self, embeddings, composer, adapt_mlp=None):
            if self.use_internal_adaptation:
                for j, op in enumerate(composer.actors[:-1]):
                    adapt = adapt_mlp[j]
                    embeddings = op(embeddings) if isinstance(adapt, torch.nn.Identity) \
                        else op(embeddings) + adapt(embeddings)
                if self.apply_adapter_on_actions:
                    return composer.act(composer.actors[-1](embeddings) + adapt_mlp[-2](embeddings))
                return composer.act(composer.actors[-1](embeddings))
            return composer(embeddings)

        def _eval_Transformer(self, obs, not_normalized_obs):

            B = obs.shape[0]

            # new major task token
            new_task_obs = self.new_task_trainable_rms(not_normalized_obs[..., self.self_obs_size:][..., self.task_obs_each_indx[self.new_major_id]:self.task_obs_each_indx[self.new_major_id + 1]])
            new_task_token = self.task_encoder[self.new_major_id](new_task_obs).unsqueeze(1)

            if self.has_extra:
                # extra 가 여럿이면 각각 자기 토큰이 된다 (teammate, steering, ...).
                # 하나뿐이면 리스트 길이가 1 이라 기존과 동일한 모양이다.
                _et = []
                for _eid in self.extra_ids:
                    _o = obs[..., self.self_obs_size:][..., self.task_obs_each_indx[_eid]:self.task_obs_each_indx[_eid + 1]]
                    _et.append(self.task_encoder[_eid](_o).unsqueeze(1))
                extra_task_token = torch.cat(_et, dim=1)

            # not_normalized_obs >> self_token
            self.basic_self_running_mean_std.training = False
            self_obs = self.basic_self_running_mean_std(not_normalized_obs[..., :self.self_obs_size])
            self_token = self.self_encoder(self_obs).unsqueeze(1) # (B, 1, num_feats)

            # not_normalized_obs >> relevant task tokens
            task_token = []
            for i, s in enumerate(self.task_obs_each_size[self.how_many_new_tasks:], self.how_many_new_tasks): # exclude the new task obs
                if s in self.basic_task_obs_each_size:
                    id = self.basic_task_obs_each_size.index(s)

                    curr_task_obs = not_normalized_obs[..., self.self_obs_size:][..., self.task_obs_each_indx[i - 1]:self.task_obs_each_indx[i]] # env only returns one task obs, because all subtasks are the same,
                    
                    self.basic_task_obs_running_mean_stds_needed[0].training = False # HACK
                    curr_task_obs = self.basic_task_obs_running_mean_stds_needed[0](curr_task_obs) # HACK
                    task_token.append(self.task_encoder[i](curr_task_obs))
                
                else:
                    raise NotImplementedError
            task_token = torch.stack(task_token, dim=1)

            # [1, 1, num_feats] -> [B, 1, num_feats]
            weight_token = self.weight_token.expand(B, -1, -1)

            if self.has_extra:
                x = torch.cat((weight_token, self_token, extra_task_token, new_task_token, task_token), dim=1)
            else:
                x = torch.cat((weight_token, self_token, new_task_token, task_token), dim=1)

            # compute key padding mask
            src_key_padding_mask = torch.ones((B, x.shape[1]), dtype=torch.bool, device=x.device) # init to all True
            src_key_padding_mask[:, [0, 1]] = False

            src_key_padding_mask[:, :self.how_many_new_tasks + 2] = False

            if self.use_prior_knowledge:
                src_key_padding_mask[:, self.how_many_new_tasks + 2:] = False

            if self._ma_old_carry_only:
                new_major_token = 2 + (len(self.extra_ids) if self.has_extra else 0)
                src_key_padding_mask[:, new_major_token] = True
                src_key_padding_mask[:, new_major_token + 1:] = False

            if self._ma_teammate_mask:
                src_key_padding_mask[:, 2] = True
            
            # src_key_padding_mask[:, 2:] = True # 只imitate style

            if self._mstop_stop_head:
                base_mask = src_key_padding_mask.clone()
                base_mask[:, 2] = True
                stop_mask = src_key_padding_mask.clone()
                stop_mask[:, 2] = False
                base_embeddings = self.transformer_encoder(
                    x, src_key_padding_mask=base_mask)[:, 0]
                stop_embeddings = self.transformer_encoder(
                    x, src_key_padding_mask=stop_mask)[:, 0]
            else:
                base_embeddings = self.transformer_encoder(
                    x, src_key_padding_mask=src_key_padding_mask)[:, 0]
                stop_embeddings = base_embeddings

            action = self._compose_action(
                base_embeddings, self.composer,
                self.internal_adapt_mlp if self.use_internal_adaptation else None)
            stop_latent = base_embeddings

            if self.use_compensate_actions:
                output = self.extra_act_mlp(base_embeddings).view(-1, 2, self.dof_action_size)
                new_action = output[:, 0]
                w = self.w_act(output[:, 1])
                action = new_action + w * action

            if self._mstop_stop_head:
                command_start = self.self_obs_size + self.task_obs_each_indx[self.extra_ids[0]]
                command = not_normalized_obs[..., command_start:command_start + 6]
                stop_action = self._compose_action(
                    stop_embeddings, self.stop_composer,
                    self.stop_internal_adapt_mlp if self.use_internal_adaptation else None)
                gate = command[:, :1].clamp(0.0, 1.0)
                delta = stop_action - action
                base_action = action
                pre_clamp = base_action + gate * delta
                action = torch.where(gate > 0, torch.clamp(pre_clamp, -1.0, 1.0),
                                     base_action)
                if self._mstop_diag or self._mstop_parity:
                    self._mstop_diag_last = {
                        "base": base_action.detach(),
                        "delta": delta.detach(),
                        "pre_clamp": pre_clamp.detach(),
                        "final": action.detach(),
                        "gate": gate.detach(),
                    }

            if self._mstop_residual:
                command_start = self.self_obs_size + self.task_obs_each_indx[self.extra_ids[0]]
                command = not_normalized_obs[..., command_start:command_start + 6]
                delta = self._mstop_residual_scale * torch.tanh(
                    self.stop_residual(torch.cat((stop_latent, command), dim=-1)))
                if self._mstop_arm_scale != 1:
                    delta = delta.clone()
                    delta[:, 6:14] *= self._mstop_arm_scale
                resume_scale = torch.where(
                    command[:, 4:5] > .5,
                    torch.full_like(command[:, 4:5], self._mstop_resume_residual_scale),
                    torch.ones_like(command[:, 4:5]))
                delta = delta * resume_scale
                base_action = action
                pre_clamp = base_action + command[:, :1] * delta
                action = torch.clamp(pre_clamp, -1.0, 1.0)
                if self._mstop_diag or self._mstop_parity:
                    self._mstop_diag_last = {
                        "base": base_action.detach(),
                        "delta": delta.detach(),
                        "pre_clamp": pre_clamp.detach(),
                        "final": action.detach(),
                        "gate": command[:, :1].detach(),
                    }

            return action
