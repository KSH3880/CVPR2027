from types import MethodType

import torch
from torch.nn import functional as F


class DependencyBiasAblation:
    def __init__(self, network, mode, seed):
        if mode not in ('off', 'shuffled'):
            raise ValueError(mode)
        self.mode = mode
        self.seed = seed
        self.enabled = False
        self.bindings = []
        self.hooks = []
        self.orders = {}
        self.calls = {}
        self.checks = {}
        self.permutation = None
        self.padding = None
        for branch, coordination in (('actor', network.coordination),
                                     ('critic', network.critic_coordination)):
            self.hooks.append(coordination.register_forward_pre_hook(self.order_hook(branch)))
            for index, block in enumerate(coordination.blocks):
                self.bind(block, branch, f'{branch}/{index}')

    def order_hook(self, branch):
        def capture(module, args):
            if not self.enabled:
                return
            humans = args[0]
            order = torch.arange(humans.shape[1], device=humans.device)
            if module.shuffle_human_order:
                devices = [humans.device.index] if humans.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    order = torch.randperm(humans.shape[1], device=humans.device)
            self.orders[branch] = order
        return capture

    def transform(self, bias, order, label):
        padding = torch.isneginf(bias)
        if self.mode == 'off':
            result = torch.zeros_like(bias).masked_fill(padding, float('-inf'))
        else:
            inverse = order.argsort()
            canonical_padding = padding[:, 0][:, inverse][:, :, inverse]
            if self.permutation is None:
                valid = ~canonical_padding.cpu()
                batch, people, _ = valid.shape
                permutation = torch.arange(people).expand(batch, people, people).clone()
                generator = torch.Generator().manual_seed(self.seed)
                for scene in range(batch):
                    for person in range(people):
                        keys = valid[scene, person].nonzero().flatten()
                        keys = keys[keys != person]
                        permutation[scene, person, keys] = keys[torch.randperm(len(keys), generator=generator)]
                self.permutation = permutation.to(bias.device)
                self.padding = canonical_padding.clone()
            if not torch.equal(canonical_padding, self.padding):
                raise ValueError('Padding changed within the fixed episode')
            indices = inverse[self.permutation[:, order][:, :, order]]
            result = bias.gather(-1, indices[:, None].expand_as(bias))
        if label not in self.checks:
            finite = torch.isfinite(bias)
            check = dict(padding_exact=torch.equal(padding, torch.isneginf(result)),
                         original_max_abs=float(bias[finite].abs().max()),
                         changed_entries=int(((result != bias) & finite).sum()))
            if self.mode == 'shuffled':
                check.update(row_multiset_exact=torch.equal(bias.sort(-1).values, result.sort(-1).values),
                             diagonal_exact=torch.equal(bias.diagonal(dim1=-2, dim2=-1),
                                                        result.diagonal(dim1=-2, dim2=-1)))
            else:
                check['finite_bias_zero'] = not bool(result[finite].any())
            if not all(value for key, value in check.items()
                       if key.endswith('_exact') or key == 'finite_bias_zero'):
                raise ValueError('Invalid bias intervention')
            self.checks[label] = check
        return result

    def bind(self, block, branch, label):
        original = block.forward

        def forward(instance, *args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            attention = F.scaled_dot_product_attention

            def intervene(query, key, value, **options):
                options['attn_mask'] = self.transform(options['attn_mask'], self.orders[branch], label)
                self.calls[label] = self.calls.get(label, 0) + 1
                return attention(query, key, value, **options)

            F.scaled_dot_product_attention = intervene
            try:
                return original(*args, **kwargs)
            finally:
                F.scaled_dot_product_attention = attention

        self.bindings.append((block, original))
        block.forward = MethodType(forward, block)

    def reset_counts(self):
        self.calls = {}

    def finish(self, steps):
        for block, original in self.bindings:
            block.forward = original
        for hook in self.hooks:
            hook.remove()
        if len(self.calls) != len(self.bindings) or any(n != steps for n in self.calls.values()):
            raise ValueError('Bias intervention missing on a policy frame')
        if not any(c['changed_entries'] for c in self.checks.values()):
            raise ValueError('Intervention never changed bias')
        return dict(mode=self.mode, permutation_seed=self.seed, actor_and_critic=True,
                    fixed_in_canonical_person_ids=True, calls=self.calls, checks=self.checks)
