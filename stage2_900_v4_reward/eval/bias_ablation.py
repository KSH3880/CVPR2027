import hashlib
from types import MethodType

import torch


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class BiasAblation:
    def __init__(self, network, mode):
        if mode not in ('none', 'task', 'dependency'):
            raise ValueError('Unknown bias ablation: '+mode)
        self.mode = mode
        self.enabled = False
        self.bindings = []
        self.calls = {}
        self.original_max = {}
        self.zero_checks = {}
        for branch in ('actor', 'critic'):
            if mode == 'task':
                module = getattr(network, branch+'_encoder').edge_encoder
                self.bind(module, 'bias_table', branch)
            elif mode == 'dependency':
                coordination = getattr(network, 'coordination' if branch == 'actor' else 'critic_coordination')
                for index, block in enumerate(coordination.blocks):
                    self.bind(block, 'effective_relation_bias', branch+'/'+str(index))

    def bind(self, module, name, label):
        original = getattr(module, name)

        def output(instance, *args):
            value = original(*args)
            if not self.enabled:
                return value
            self.calls[label] = self.calls.get(label, 0)+1
            result = torch.zeros_like(value)
            if label not in self.zero_checks:
                self.original_max[label] = float(value.detach().abs().max())
                self.zero_checks[label] = not bool(result.any())
            return result

        self.bindings.append((module, name, original))
        setattr(module, name, MethodType(output, module))

    def reset_counts(self):
        self.calls = {}

    def finish(self, steps):
        if self.mode != 'none':
            if len(self.calls) != len(self.bindings) or any(count != steps for count in self.calls.values()):
                raise ValueError('Bias bypass was not applied on every policy frame')
            if not all(self.zero_checks.values()) or not any(self.original_max.values()):
                raise ValueError('Bias bypass did not remove a nonzero bias')
        for module, name, original in self.bindings:
            setattr(module, name, original)
        return dict(mode=self.mode, inference_only=True, weights_modified=False,
                    removed_bias_calls=self.calls, zero_output_verified=self.zero_checks,
                    original_bias_max_abs=self.original_max)
