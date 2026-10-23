"""TokenHSI-style zero-initialized linear residuals at every action-head layer."""
from torch import nn


def layers(head):
    for child in head.children():
        if isinstance(child, nn.Sequential):
            yield from layers(child)
        else:
            yield child


def make_adapters(head):
    adapters = nn.ModuleList()
    for layer in layers(head):
        if isinstance(layer, nn.Linear):
            adapter = nn.Linear(layer.in_features, layer.out_features)
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            adapters.append(adapter)
    if not adapters: raise ValueError("Action head has no Linear layers")
    return adapters


def adapted_action(head, adapters, encoded):
    index = 0
    for layer in layers(head):
        if isinstance(layer, nn.Linear):
            encoded = layer(encoded) + adapters[index](encoded)
            index += 1
        else:
            encoded = layer(encoded)
    return encoded
