import torch

from utils.edge_stage1_spec import STAGE1_CONTEXT_MODE


FIRST_HEAD_WEIGHT = 'a2c_network.action_head.0.0.weight'
COORDINATION_PREFIX = 'a2c_network.coordination.'


def transfer_stage1_weights(model, checkpoint):
    metadata = checkpoint.get('relation_metadata', {})
    task_ca = getattr(model.a2c_network.actor_encoder, 'task_role_input', False)
    from utils.task_role_spec import TASK_EMBEDDING_VARIANT, TASK_DISTILL_VARIANT, CARRY_DISTILL_VARIANT
    expected_packet = 5 if task_ca else 3
    if getattr(getattr(model.a2c_network, 'coordination', None), 'before', False) and \
            metadata.get('relation_reward_config', {}).get('stage1_variant') == CARRY_DISTILL_VARIANT:
        raise ValueError('BEFORE requires a four-task Stage-1 source, not carry-only distillation')
    expected_fusion = ('task_category_embedding64_type_pair_projection'
                       if task_ca else 'semantic_only')
    if (metadata.get('reward_mode') != STAGE1_CONTEXT_MODE or
            metadata.get('schema_version') != 9 or
            metadata.get('packet_version') != expected_packet or
            metadata.get('context_fusion') != expected_fusion or
            task_ca and metadata.get('relation_reward_config', {}).get('stage1_variant')
                not in (TASK_EMBEDDING_VARIANT, TASK_DISTILL_VARIANT, CARRY_DISTILL_VARIANT)):
        raise ValueError('Stage-2 import needs a semantic Stage-1 schema-9 checkpoint')
    source = checkpoint.get('model')
    if not isinstance(source, dict) or FIRST_HEAD_WEIGHT not in source:
        raise ValueError('Stage-1 checkpoint has no compatible model state')
    target = model.state_dict()
    context_width = 64 if hasattr(model.a2c_network, 'coordination') else 0
    copied = []
    new = []
    for key, value in target.items():
        if key.startswith(COORDINATION_PREFIX):
            new.append(key)
            continue
        if key not in source:
            raise ValueError('Stage-1 checkpoint is missing ' + key)
        old = source[key]
        if key == FIRST_HEAD_WEIGHT and context_width:
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
            'temporary_distill_source': task_ca and metadata['relation_reward_config'].get(
                'stage1_variant') == TASK_DISTILL_VARIANT,
            'source_variant': metadata['relation_reward_config'].get('stage1_variant'),
            'copied_tensors': copied, 'new_tensors': new,
            'frozen_parameters': frozen,
            'head_expansion': {'source_columns': 64, 'context_columns': context_width}}


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
