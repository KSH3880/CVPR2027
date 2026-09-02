"""Frozen f22 with a coordination token inside the shared transformer.

The original f22 self/steer/carry tokenizers, transformer, composer and steer
adapter remain frozen. A new coordination tokenizer feeds the same frozen
transformer, and a zero-initialized internal adapter learns from its output.
In ``putdown_retreat`` both additions are gated to A1_RETREAT.
"""

import copy
import os

import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning.transformer.amp_network_builder_transformer_adapt_f22 import (
    AMPTransformerMultiTaskAdaptBuilder as F22Builder,
)
from utils.torch_utils import load_checkpoint


SELF_DIM = 223
COORD_DIM = 30
STEER_DIM = 24
CARRY_DIM = 42
F22_OBS_DIM = SELF_DIM + STEER_DIM + CARRY_DIM


class AMPTransformerF22StackBuilder(F22Builder):

    def build(self, name, **kwargs):
        return AMPTransformerF22StackBuilder.Network(self.params, **kwargs)

    class Network(F22Builder.Network):

        def __init__(self, params, **kwargs):
            full_info = kwargs["multi_task_info"]
            assert full_info["each_subtask_obs_size"] == [
                COORD_DIM, STEER_DIM, CARRY_DIM, CARRY_DIM]
            assert kwargs["self_obs_size"] == SELF_DIM

            # Construct the exact five-token f22 topology first. The sixth
            # coordination token is appended manually without shifting any
            # existing token or checkpoint tensor.
            device = kwargs["device"]
            each = [STEER_DIM, CARRY_DIM, CARRY_DIM]
            index = torch.cumsum(torch.tensor([0] + each), dim=0).to(device)
            mask = torch.zeros(3, sum(each), dtype=torch.bool, device=device)
            for i in range(len(each)):
                mask[i, index[i]:index[i + 1]] = True
            base_kwargs = dict(kwargs)
            base_kwargs["task_obs_size"] = STEER_DIM + CARRY_DIM
            base_kwargs["multi_task_info"] = {
                "onehot_size": 3,
                "tota_subtask_obs_size": sum(each),
                "each_subtask_obs_size": each,
                "each_subtask_obs_mask": mask,
                "each_subtask_obs_indx": index,
                "enable_task_mask_obs": False,
                "each_subtask_name": ["new_extra", "new_carry", "old_carry"],
                "major_task_name": "carry",
                "has_extra": True,
            }
            super().__init__(params, **base_kwargs)
            if self.use_pos_embed:
                raise ValueError("f22 stack requires transformer.use_pos_embed=False")
            if not self.use_internal_adaptation:
                raise ValueError("f22 stack requires use_internal_adaptation=True")

            checkpoint = load_checkpoint(kwargs["hrl_checkpoint"], device=self.device)
            loaded = checkpoint["model"]
            own = self.state_dict()
            copied = 0
            with torch.no_grad():
                for key, value in loaded.items():
                    key = key[len("a2c_network."):] \
                        if key.startswith("a2c_network.") else key
                    if key in own and own[key].shape == value.shape:
                        own[key].copy_(value)
                        copied += 1
            if copied == 0:
                raise ValueError("f22 checkpoint contained no compatible actor weights")

            self.f22_input_rms = RunningMeanStd((F22_OBS_DIM,))
            self.f22_input_rms.load_state_dict(checkpoint["running_mean_std"])
            self.f22_input_rms.requires_grad_(False)
            self.f22_input_rms.eval()

            frozen = [self.self_encoder, self.task_encoder,
                      self.transformer_encoder, self.composer,
                      self.new_task_trainable_rms,
                      self.basic_self_running_mean_std,
                      self.basic_task_obs_running_mean_stds_needed,
                      self.internal_adapt_mlp]
            for module in frozen:
                module.requires_grad_(False)
                module.eval()
            self.weight_token.requires_grad_(False)
            self.pos_embed.requires_grad_(False)

            num_features = self.weight_token.shape[-1]
            token_hidden = int(params.get("coord_token_hidden", 128))
            self.coord_encoder = nn.Sequential(
                nn.Linear(COORD_DIM, token_hidden), nn.ReLU(),
                nn.Linear(token_hidden, num_features))
            nn.init.zeros_(self.coord_encoder[-1].weight)
            nn.init.zeros_(self.coord_encoder[-1].bias)

            # New trainable adapter, parallel to the frozen steer-era adapter.
            self.coord_adapter = copy.deepcopy(self.internal_adapt_mlp)
            self.coord_adapter.requires_grad_(True)
            self.coord_adapter.train()
            for module in self.coord_adapter.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            self.stack_task_mode = os.environ.get("STACK_TASK_MODE", "stack")
            self.retreat_carry_weight = float(os.environ.get(
                "STACK_RETREAT_CARRY_WEIGHT", "1.0"))
            if not 0.0 <= self.retreat_carry_weight <= 1.0:
                raise ValueError(
                    "STACK_RETREAT_CARRY_WEIGHT must be in [0, 1], got {}".format(
                        self.retreat_carry_weight))
            self._coord_forward_count = 0
            print(f"[f22-stack] frozen base copied={copied}; coord token "
                  f"{COORD_DIM}->{num_features} enters shared transformer; "
                  "new internal adapter zero-init; retreat carry weight="
                  f"{self.retreat_carry_weight}", flush=True)

        @staticmethod
        def _split_full_obs(obs):
            coord = obs[..., SELF_DIM:SELF_DIM + COORD_DIM]
            base = torch.cat((obs[..., :SELF_DIM],
                              obs[..., SELF_DIM + COORD_DIM:]), dim=-1)
            assert base.shape[-1] == F22_OBS_DIM, base.shape
            return base, coord

        def _coord_gate(self, coord_raw):
            if self.stack_task_mode != "putdown_retreat":
                return torch.ones_like(coord_raw[..., :1])
            # role[0] is A1; phase one-hot starts at coord[2], phase 2 is [4].
            return (coord_raw[..., 0:1].clamp(0.0, 1.0)
                    * coord_raw[..., 4:5].clamp(0.0, 1.0))

        def _eval_shared_transformer(self, base_obs, base_raw, coord, coord_raw):
            """Run the original f22 tokens and coord token together."""
            B = base_obs.shape[0]
            task_raw = base_raw[..., SELF_DIM:]
            task_obs = base_obs[..., SELF_DIM:]

            new_raw = task_raw[..., self.task_obs_each_indx[self.new_major_id]:
                               self.task_obs_each_indx[self.new_major_id + 1]]
            self.new_task_trainable_rms.training = False
            new_token = self.task_encoder[self.new_major_id](
                self.new_task_trainable_rms(new_raw)).unsqueeze(1)

            extra_obs = task_obs[..., self.task_obs_each_indx[self.extra_id]:
                                  self.task_obs_each_indx[self.extra_id + 1]]
            extra_token = self.task_encoder[self.extra_id](extra_obs).unsqueeze(1)

            self.basic_self_running_mean_std.training = False
            self_token = self.self_encoder(self.basic_self_running_mean_std(
                base_raw[..., :SELF_DIM])).unsqueeze(1)

            old_i = self.old_major_id
            old_raw = task_raw[..., self.task_obs_each_indx[old_i - 1]:
                               self.task_obs_each_indx[old_i]]
            old_rms = self.basic_task_obs_running_mean_stds_needed[0]
            old_rms.training = False
            old_token = self.task_encoder[old_i](old_rms(old_raw)).unsqueeze(1)

            weight_token = self.weight_token.expand(B, -1, -1)
            coord_token = self.coord_encoder(coord).unsqueeze(1)

            # In the isolated retreat task, the placement dependency is
            # already complete at A1 phase 2. Allow experiments to remove the
            # now-conflicting carry command while retaining frozen steer. A
            # zero weight also masks the token key entirely so a zero vector
            # cannot change the attention softmax denominator.
            gate = self._coord_gate(coord_raw)
            retreat_row = gate[:, 0] > 0.0
            carry_scale = torch.ones((B, 1, 1), device=new_token.device,
                                     dtype=new_token.dtype)
            carry_scale[retreat_row] = self.retreat_carry_weight
            new_token = new_token * carry_scale
            x = torch.cat((weight_token, self_token, extra_token,
                           new_token, old_token, coord_token), dim=1)

            # Original active set is weight, self, steer, new carry. Old carry
            # follows prior-knowledge mode; coord is opened per row and phase.
            padding = torch.ones((B, 6), dtype=torch.bool, device=x.device)
            padding[:, 0:4] = False
            if self.use_prior_knowledge:
                padding[:, 4] = False
            if self.retreat_carry_weight == 0.0:
                padding[retreat_row, 3] = True
            padding[:, 5] = gate[:, 0] <= 0.0
            x = self.transformer_encoder(x, src_key_padding_mask=padding)
            return x[:, 0], gate

        def _decode_with_coord_adapter(self, embeddings, gate):
            for j, op in enumerate(self.composer.actors[:-1]):
                base_adapter = self.internal_adapt_mlp[j]
                new_adapter = self.coord_adapter[j]
                base_delta = (0.0 if isinstance(base_adapter, nn.Identity)
                              else base_adapter(embeddings))
                new_delta = (0.0 if isinstance(new_adapter, nn.Identity)
                             else new_adapter(embeddings))
                embeddings = op(embeddings) + base_delta + gate * new_delta

            logits = self.composer.actors[-1](embeddings)
            if self.apply_adapter_on_actions:
                logits = logits + self.internal_adapt_mlp[-2](embeddings)
                logits = logits + gate * self.coord_adapter[-2](embeddings)
            return self.composer.act(logits)

        def eval_actor(self, obs, not_normalized_obs):
            base_raw, coord_raw = self._split_full_obs(not_normalized_obs)
            _, coord = self._split_full_obs(obs)
            self.f22_input_rms.training = False
            base_obs = self.f22_input_rms(base_raw)
            embeddings, gate = self._eval_shared_transformer(
                base_obs, base_raw, coord, coord_raw)
            mu = self._decode_with_coord_adapter(embeddings, gate)

            self._coord_forward_count += 1
            if (os.environ.get("STACK_RESIDUAL_DEBUG", "0") == "1" and
                    self._coord_forward_count % 1000 == 0):
                with torch.no_grad():
                    token_abs = float(self.coord_encoder(coord).abs().mean())
                    print("[f22-stack] coord_token mean_abs={:.5f} active={:.3f}".format(
                        token_abs, float(gate.mean())), flush=True)

            if (os.environ.get("STACK_PARITY_CHECK", "0") == "1" and
                    not getattr(self, "_coord_parity_checked", False)):
                self._coord_parity_checked = True
                with torch.no_grad():
                    base_mu = super().eval_composer(base_obs, base_raw)
                    inactive = gate[:, 0] == 0
                    err = (float((mu[inactive] - base_mu[inactive]).abs().max())
                           if bool(inactive.any()) else 0.0)
                if err > 1e-6:
                    raise RuntimeError(
                        f"masked coord token broke f22 parity: max|delta|={err}")
                print("[f22-stack] parity OK: masked coord rows match frozen f22",
                      flush=True)

            sigma = self.sigma_act(self.sigma)
            return mu, sigma
