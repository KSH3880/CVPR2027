from types import MethodType

import torch


class DctAblation:
    def __init__(self, network, mode):
        if mode not in ('none', 'self_only', 'bypass'):
            raise ValueError(mode)
        self.mode = mode
        self.enabled = False
        self.calls = {}
        self.bindings = []
        self.hook = None
        if mode == 'self_only':
            for index, block in enumerate(network.coordination.blocks):
                self.bind(block, str(index))
        elif mode == 'bypass':
            self.hook = network.coordination.register_forward_hook(self.bypass)

    def bind(self, block, label):
        original = block.forward

        def forward(instance, humans, tokens, relations, padding, fusion):
            if self.enabled:
                people = humans.shape[1]
                padding = padding | ~torch.eye(people, dtype=torch.bool, device=humans.device)[None, None]
                self.calls[label] = self.calls.get(label, 0) + 1
            return original(humans, tokens, relations, padding, fusion)

        self.bindings.append((block, original))
        block.forward = MethodType(forward, block)

    def bypass(self, module, args, output):
        if self.enabled:
            self.calls['coordination'] = self.calls.get('coordination', 0) + 1
            return args[0]
        return output

    def reset_counts(self):
        self.calls = {}

    def finish(self, steps):
        expected = len(self.bindings) if self.mode == 'self_only' else int(self.mode == 'bypass')
        if len(self.calls) != expected or any(n != steps for n in self.calls.values()):
            raise ValueError('DCT intervention was not applied on every actor frame')
        for block, original in self.bindings:
            block.forward = original
        if self.hook is not None:
            self.hook.remove()
        return dict(mode=self.mode, actor_only=True, calls=self.calls,
                    original_coordination_executed_for_rng_parity=self.mode == 'bypass')
