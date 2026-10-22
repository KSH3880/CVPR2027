import torch

from utils.edge_stage1_spec import STAGE1_CONTEXT_MODE


FIRST_HEAD_WEIGHT = 'a2c_network.action_head.0.0.weight'
COORDINATION_PREFIX = 'a2c_network.coordination.'


def transfer_stage1_weights(model, checkpoint):
    metadata = checkpoint.get('relation_metadata', {})
    if (metadata.get('reward_mode') != STAGE1_CONTEXT_MODE or
            metadata.get('schema_version') != 9 or
            metadata.get('packet_version') != 3 or
            metadata.get('context_fusion') != 'semantic_only'):
        raise ValueError('Stage-2 import needs a semantic Stage-1 schema-9 checkpoint')
    source = checkpoint.get('model')
    if not isinstance(source, dict) or FIRST_HEAD_WEIGHT not in source:
        raise ValueError('Stage-1 checkpoint has no compatible model state')
    target = model.state_dict()
    copied = []
    new = []
    for key, value in target.items():
        if key.startswith(COORDINATION_PREFIX):
            new.append(key)
            continue
        if key not in source:
            raise ValueError('Stage-1 checkpoint is missing ' + key)
        old = source[key]
        if key == FIRST_HEAD_WEIGHT:
            if old.ndim != 2 or value.shape != (old.shape[0], 2 * old.shape[1]):
                raise ValueError('Stage-2 action head is not a 64+64 expansion')
            value[:, :old.shape[1]] = old
            value[:, old.shape[1]:] = 0.
        elif value.shape != old.shape:
            raise ValueError('Stage-1 tensor shape mismatch: ' + key)
        else:
            value.copy_(old)
        copied.append(key)
    unused = sorted(set(source) - set(target))
    if unused:
        raise ValueError('Unexpected Stage-1 tensors: ' + ', '.join(unused[:8]))
    model.load_state_dict(target, strict=True)
    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith('a2c_network.actor_encoder.'):
            parameter.requires_grad_(False)
            frozen.append(name)
    if not frozen:
        raise ValueError('Stage-2 actor encoder freeze did not match model parameters')
    return {'source_epoch': checkpoint.get('epoch'),
            'source_variant': metadata['relation_reward_config'].get('stage1_variant'),
            'copied_tensors': copied, 'new_tensors': new,
            'frozen_parameters': frozen,
            'head_expansion': {'source_columns': 64, 'context_columns': 64}}


def freeze_stage2_encoder(model, optimizer):
    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith('a2c_network.actor_encoder.'):
            parameter.requires_grad_(False)
            frozen.append(name)
    if not frozen:
        raise ValueError('Stage-2 actor encoder was not found')
    for group in optimizer.param_groups:
        group['params'] = [parameter for parameter in group['params']
                           if parameter.requires_grad]
    return frozen
