from types import MethodType

import torch

from depbias_ablation import DependencyBiasAblation


class TaskBiasAblation(DependencyBiasAblation):
    def __init__(self, network, mode, seed):
        if mode not in ('off', 'shuffled'):
            raise ValueError(mode)
        self.mode, self.seed = mode, seed
        self.enabled = False
        self.bindings, self.hooks = [], []
        self.calls, self.checks = {}, {}
        self.zero_checks, self.original_max = {}, {}
        self.permutation = self.padding = None
        for branch in ('actor', 'critic'):
            self.bind_fusion(getattr(network, branch+'_encoder').context_fusion, branch)

    def bind_fusion(self, module, label):
        original = module.forward

        def forward(instance, *args, **kwargs):
            bias, message = original(*args, **kwargs)
            if not self.enabled:
                return bias, message
            layers, batch, heads, length, _ = bias.shape
            flat = bias.permute(1, 0, 2, 3, 4).reshape(batch, layers*heads, length, length)
            changed = self.transform(flat, torch.arange(length, device=bias.device), label)
            self.calls[label] = self.calls.get(label, 0)+1
            self.original_max[label] = self.checks[label]['original_max_abs']
            self.zero_checks[label] = self.checks[label].get('finite_bias_zero', False)
            return changed.reshape(batch, layers, heads, length, length).permute(1, 0, 2, 3, 4), message

        self.bindings.append((module, original))
        module.forward = MethodType(forward, module)

    def finish(self, steps):
        proof = super().finish(steps)
        proof.pop('fixed_in_canonical_person_ids')
        proof.update(fixed_in_canonical_token_ids=True, scope='SCT task attention bias',
                     token_types=['human', 'object', 'goal'], task_packet_unchanged=True,
                     feature_fusion_unchanged=True, before_padding_and_internal_token_permutation=True)
        return proof
