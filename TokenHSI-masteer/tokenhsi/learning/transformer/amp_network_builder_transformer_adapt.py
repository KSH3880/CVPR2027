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
import math
import os

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
                self.mask_teammate_token = os.environ.get("MA_TOKEN", "live") == "mask"
                self.teammate_token_pos = 2
                if self.mask_teammate_token:
                    print("[ma] teammate token: exact attention mask (MA_TOKEN=mask)",
                          flush=True)

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

            # Optional hierarchical controller. PPO still emits the original
            # joint-dimensional action, but the only trainable actor module is
            # this two-output command head. It edits the local steering window
            # (speed = radial scale, direction = planar rotation) before a
            # previously trained masteer actor consumes it.
            self.use_hier_steer = os.environ.get("MS_HIER", "0") == "1"
            if self.use_hier_steer:
                assert self.has_extra, "MS_HIER requires teammate + steering extra observations"
                self.hier_target = os.environ.get("MS_HIER_TARGET", "steer")
                assert self.hier_target in ("steer", "traj"), self.hier_target
                if self.hier_target == "steer":
                    target_ids = [i for i in self.extra_ids if self.task_obs_each_size[i] == 12]
                    assert len(target_ids) == 1, (
                        "MS_HIER_TARGET=steer expects exactly one 12-D extra token; got "
                        f"{[(i, self.task_obs_each_size[i]) for i in self.extra_ids]}")
                    self.hier_target_id = target_ids[0]
                    context_ids = [i for i in self.extra_ids if i != self.hier_target_id]

                else:
                    assert self.major_task_name == "traj", self.major_task_name
                    assert not self.use_pos_embed, (
                        "MS_HIER_TARGET=traj reorders the active frozen token; "
                        "the adapt config must set transformer.use_pos_embed=False")
                    self.hier_target_id = self.new_major_id
                    context_ids = list(self.extra_ids)

                assert context_ids, "MS_HIER requires at least one teammate token"

                self.hier_context_ids = context_ids
                target_dim = self.task_obs_each_size[self.hier_target_id]

                assert target_dim % 2 == 0, target_dim

                hier_in = sum(self.task_obs_each_size[i] for i in context_ids) + target_dim
                hidden = int(os.environ.get("MS_HIER_HIDDEN", "128"))
                self.hier_speed_span = float(os.environ.get("MS_HIER_SPEED_SPAN", "0.5"))
                self.hier_turn_max = math.radians(float(os.environ.get("MS_HIER_TURN_DEG", "25")))
                self.hier_head = nn.Sequential(
                    nn.Linear(hier_in, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, 2))
                
                # Identity command at initialization: scale=1, rotation=0.
                nn.init.zeros_(self.hier_head[-1].weight)
                nn.init.zeros_(self.hier_head[-1].bias)

                base_path = os.environ.get("MS_HIER_BASE", "")
                if not base_path:
                    raise ValueError("MS_HIER=1 requires MS_HIER_BASE=<trained masteer .pth>")
                base_ckp = load_checkpoint(base_path, device=self.device)
                base = base_ckp["model"]
                own = self.state_dict()
                copied = 0
                with torch.no_grad():
                    for key, value in base.items():
                        key = key[len("a2c_network."):] if key.startswith("a2c_network.") else key
                        if key.startswith("hier_head."):
                            continue
                        if key in own and own[key].shape == value.shape:
                            own[key].copy_(value)
                            copied += 1
                if copied == 0:
                    raise ValueError(f"MS_HIER_BASE has no compatible actor weights: {base_path}")
                # The frozen extra tokenizers were trained behind this exact
                # observation normalizer. Reusing only their weights while the
                # outer agent starts a fresh RMS silently changes their input.
                if self.hier_target == "steer":
                    self.hier_base_rms = RunningMeanStd(kwargs["input_shape"])
                    self.hier_base_rms.load_state_dict(base_ckp["running_mean_std"])
                    self.hier_base_rms.requires_grad_(False)
                    self.hier_base_rms.eval()

                # Freeze every module that turns commands into joint actions.
                # Critic/discriminator parameters remain trainable for PPO/AMP.
                frozen = [self.self_encoder, self.task_encoder,
                          self.transformer_encoder, self.composer]
                if hasattr(self, "internal_adapt_mlp"):
                    frozen.append(self.internal_adapt_mlp)
                if hasattr(self, "extra_act_mlp"):
                    frozen.append(self.extra_act_mlp)
                for module in frozen:
                    module.requires_grad_(False)
                    module.eval()
                self.weight_token.requires_grad_(False)
                self.pos_embed.requires_grad_(False)
                self.hier_head.requires_grad_(True)
                print(f"[hier] loaded frozen low-level actor from {base_path}; "
                      f"copied={copied}, target={self.hier_target}, head_in={hier_in}, "
                      f"speed_span={self.hier_speed_span}, "
                      f"turn={math.degrees(self.hier_turn_max):.1f}deg", flush=True)

            # Sequential-stack fine-tuning keeps the complete trained
            # MA-steer actor fixed except for the carry tokenizer that was
            # introduced during adaptation and its residual adapter.  The
            # full MA-steer checkpoint is restored by rl_games *after* this
            # network is constructed, so requires_grad is selected here while
            # the actual starting weights still come from --checkpoint.
            self.finetune_newcarry_residual = (
                os.environ.get("MA_FINETUNE_NEWCARRY_RESIDUAL", "0") == "1")
            self.finetune_steer_token = (
                os.environ.get("MA_FINETUNE_STEER_TOKEN", "0") == "1")
            if (self.finetune_steer_token
                    and not self.finetune_newcarry_residual):
                raise ValueError(
                    "MA_FINETUNE_STEER_TOKEN requires "
                    "MA_FINETUNE_NEWCARRY_RESIDUAL=1")
            if self.finetune_newcarry_residual:
                if self.use_hier_steer:
                    raise ValueError(
                        "MA_FINETUNE_NEWCARRY_RESIDUAL is incompatible with MS_HIER")
                if not self.use_internal_adaptation:
                    raise ValueError(
                        "new-carry residual fine-tuning requires "
                        "use_internal_adaptation=True")

                # Freeze every actor module first, then open new-carry, the
                # residual, and optionally the steering tokenizer.  Critic
                # and AMP discriminator are separate and remain trainable.
                actor_modules = [self.self_encoder, self.task_encoder,
                                 self.transformer_encoder, self.composer,
                                 self.internal_adapt_mlp]
                if hasattr(self, "extra_act_mlp"):
                    actor_modules.append(self.extra_act_mlp)
                for module in actor_modules:
                    module.requires_grad_(False)
                    module.eval()
                self.weight_token.requires_grad_(False)
                self.pos_embed.requires_grad_(False)

                new_carry = self.task_encoder[self.new_major_id]
                new_carry.requires_grad_(True)
                new_carry.train()
                self.finetune_steer_id = None
                if self.finetune_steer_token:
                    steer_ids = [i for i in self.extra_ids
                                 if self.task_obs_each_size[i] == 12]
                    if len(steer_ids) != 1:
                        raise ValueError(
                            "MA_FINETUNE_STEER_TOKEN expects exactly one "
                            "12-D steering tokenizer; got {}".format(
                                [(i, self.task_obs_each_size[i])
                                 for i in self.extra_ids]))
                    self.finetune_steer_id = steer_ids[0]
                    steer_tokenizer = self.task_encoder[
                        self.finetune_steer_id]
                    steer_tokenizer.requires_grad_(True)
                    steer_tokenizer.train()
                self.internal_adapt_mlp.requires_grad_(True)
                self.internal_adapt_mlp.train()

                # This RMS has buffers rather than optimizer parameters.  It
                # must nevertheless stay fixed, otherwise the tokenizer input
                # changes even while all other actor weights are frozen.
                self.new_task_trainable_rms.requires_grad_(False)
                self.new_task_trainable_rms.eval()
                steer_state = ("trainable" if self.finetune_steer_token
                               else "frozen")
                print("[sequential-ft] actor trainable: new_carry tokenizer + "
                      "internal action residual; steer tokenizer={}; "
                      "teammate masked/frozen; "
                      "base actor RMS frozen".format(steer_state), flush=True)

            return

        def train(self, mode=True):
            """Keep the frozen MA-steer actor deterministic during PPO updates."""
            super().train(mode)
            if not getattr(self, "finetune_newcarry_residual", False):
                return self

            self.self_encoder.eval()
            self.task_encoder.eval()
            self.transformer_encoder.eval()
            self.composer.eval()
            self.new_task_trainable_rms.eval()
            if hasattr(self, "extra_act_mlp"):
                self.extra_act_mlp.eval()
            if mode:
                self.task_encoder[self.new_major_id].train()
                if getattr(self, "finetune_steer_id", None) is not None:
                    self.task_encoder[self.finetune_steer_id].train()
                self.internal_adapt_mlp.train()
            else:
                self.internal_adapt_mlp.eval()
            return self

        def _hier_transform(self, raw_obs, head_obs):
            """Apply the 2-D head to one Kx2 task-observation block."""
            raw_task = raw_obs[..., self.self_obs_size:]
            head_task = head_obs[..., self.self_obs_size:]
            parts = [head_task[..., self.task_obs_each_indx[i]:self.task_obs_each_indx[i + 1]]
                     for i in self.hier_context_ids]
            lo = self.task_obs_each_indx[self.hier_target_id]
            hi = self.task_obs_each_indx[self.hier_target_id + 1]
            raw_window = raw_task[..., lo:hi]
            head_window = head_task[..., lo:hi]
            command = self.hier_head(torch.cat(parts + [head_window], dim=-1))
            speed = 1.0 + self.hier_speed_span * torch.tanh(command[..., 0])
            angle = self.hier_turn_max * torch.tanh(command[..., 1])
            pts = raw_window.view(*raw_window.shape[:-1], -1, 2)
            c, s = torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)
            x, y = pts[..., 0], pts[..., 1]
            rot = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
            edited = (rot * speed[..., None, None]).reshape_as(raw_window)
            edited_raw = raw_obs.clone()
            edited_raw[..., self.self_obs_size + lo:self.self_obs_size + hi] = edited
            return edited_raw

        def _hier_command(self, raw_obs, normalized_obs):
            if self.hier_target == "steer":
                self.hier_base_rms.training = False
                base_obs = self.hier_base_rms(raw_obs)
                edited_raw = self._hier_transform(raw_obs, base_obs)

                return self.hier_base_rms(edited_raw), raw_obs

            # teammate values are already clipped in the environment; scaling
            # them keeps the head input bounded without inventing RMS entries
            # that do not exist in the single-agent stage1 checkpoint.  The
            # edited raw trajectory is subsequently normalized by the frozen
            # stage1 traj RMS in the old-task tokenizer path.
            head_obs = normalized_obs.clone()
            edited_raw = self._hier_transform(raw_obs, head_obs)
            return normalized_obs, edited_raw
        
        def _eval_Transformer(self, obs, not_normalized_obs):

            if getattr(self, "use_hier_steer", False):
                obs, not_normalized_obs = self._hier_command(not_normalized_obs, obs)

            B = obs.shape[0]

            # new major task token
            _new_raw = not_normalized_obs[..., self.self_obs_size:][..., self.task_obs_each_indx[self.new_major_id]:self.task_obs_each_indx[self.new_major_id + 1]]
            if getattr(self, "use_hier_steer", False) and self.hier_target == "traj":
                # This token is masked below; only the frozen old-traj token is
                # the low-level command path.  Keeping this RMS in train mode
                # would store head-dependent tensors in its running statistics,
                # then reuse that graph in the next PPO minibatch ("backward
                # through the graph a second time").
                self.new_task_trainable_rms.training = False
                _new_raw = _new_raw.detach()
            new_task_obs = self.new_task_trainable_rms(_new_raw)
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

            if self.has_extra and self.mask_teammate_token:
                src_key_padding_mask[:, self.teammate_token_pos] = True

            if getattr(self, "use_hier_steer", False) and self.hier_target == "traj":
                # Exact stage1 low-level path: mask newly added teammate/new
                # tokens and expose only weight, self and the frozen old-traj
                # tokenizer. The head still controls that old token because the
                # legacy adapt slice intentionally reads the duplicated new
                # traj block from not_normalized_obs.
                src_key_padding_mask[:] = True
                src_key_padding_mask[:, [0, 1]] = False
                old_pos = (2 + len(self.extra_ids) + 1
                           + self.old_major_id - self.how_many_new_tasks)
                src_key_padding_mask[:, old_pos] = False
            
            # src_key_padding_mask[:, 2:] = True # 只imitate style

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
