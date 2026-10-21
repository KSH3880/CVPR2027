"""Exact parameter boundary for reusing the original 30D object tokenizer."""
import torch

SCHEMA = 'native_sit_carry_v1'


def trainable_name(name, mode="tokenizer_only"):
    if mode not in ("tokenizer_only", "adapter", "head", "copied_adapter"):
        raise ValueError(mode)
    if mode == "copied_adapter":
        if name.startswith("a2c_network.actor_encoder.tokenizers.1."): return False
        if name.startswith(("a2c_network.actor_encoder.adapted_object_tokenizer.", "a2c_network.action_adapter.")): return True
    if mode == "adapter" and name.startswith("a2c_network.action_adapter."): return True
    if mode == "head" and name.startswith("a2c_network.action_head."): return True
    return name.startswith(('a2c_network.actor_encoder.tokenizers.1.',
                            'a2c_network.critic_encoder.', 'a2c_network.value_head.',
                            'a2c_network._disc_mlp.', 'a2c_network._disc_logits.'))


def freeze_actor(model, mode="tokenizer_only"):
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable_name(name, mode))
        if parameter.requires_grad:
            selected.append(name)
    object_prefix = 'actor_encoder.adapted_object_tokenizer.' if mode == 'copied_adapter' else 'actor_encoder.tokenizers.1.'
    for prefix in (object_prefix, 'critic_encoder.', 'value_head.'):
        if not any(prefix in name for name in selected):
            raise ValueError('Missing trainable module: ' + prefix)
    if not any('_disc_' in name for name in selected):
        raise ValueError('Discriminator parameters were not found')
    if mode in ('adapter', 'copied_adapter', 'head'):
        prefix = 'action_adapter.' if mode in ('adapter', 'copied_adapter') else 'action_head.'
        if not any(prefix in name for name in selected):
            raise ValueError('Missing trainable module: ' + prefix)
    return selected


def frozen_snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}


def verify_frozen(model, reference):
    current = dict(model.named_parameters())
    changed = [n for n, value in reference.items() if not torch.equal(current[n].detach().cpu(), value)]
    if changed:
        raise RuntimeError('Frozen actor parameters changed: ' + str(changed[:5]))
    return len(reference)
