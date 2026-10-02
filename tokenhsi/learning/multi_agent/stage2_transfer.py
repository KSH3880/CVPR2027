import torch

from utils.edge_stage1_spec import (STAGE1_CONTEXT_MODE,
    UNIFIED_VARIANTS)
from utils.edge_stage2_spec import STAGE2_UNIFIED_SAMPLER, STAGE2_RESCUE_SAMPLER, STAGE2_SHARED_RESCUE_SAMPLER


FIRST_HEAD_WEIGHT = 'a2c_network.action_head.0.0.weight'
COORDINATION_PREFIX = 'a2c_network.coordination.'


def transfer_stage1_weights(model, checkpoint):
    metadata = checkpoint.get('relation_metadata', {})
    owner_target = model.a2c_network.actor_encoder.owner_holding_state
    unified_target = (model.a2c_network.relation_graph_spec or {}).get('sampler') == STAGE2_UNIFIED_SAMPLER
    rescue_target = (model.a2c_network.relation_graph_spec or {}).get('sampler') in (STAGE2_RESCUE_SAMPLER, STAGE2_SHARED_RESCUE_SAMPLER)
    variant = metadata.get('relation_reward_config', {}).get('stage1_variant')
    if (metadata.get('reward_mode') != STAGE1_CONTEXT_MODE or
            metadata.get('schema_version') != 9 or
            (owner_target and (variant != UNIFIED_VARIANTS[1] or
                metadata.get('packet_version') != 4 or
                metadata.get('context_fusion') !=
                'semantic64_owner_holding_residual65x64x64')) or
            (not owner_target and (metadata.get('packet_version') != 3 or
                metadata.get('context_fusion') != 'semantic_only')) or
            (unified_target and variant != UNIFIED_VARIANTS[0]) or
            (rescue_target and variant != 'scenario_independent_stage1_self_sum_rescue')):
        raise ValueError('Stage-2 import needs the matching Stage-1 schema-9 checkpoint')
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


def load_stage1_for_evaluation(model, checkpoint, actor_rms, amp_rms=None):
    """Import an untrained Stage-2 policy with zero action contribution from context."""
    if not getattr(model.a2c_network, 'stage2', False):
        raise ValueError('Stage-1 baseline evaluation requires a Stage-2 model')
    if actor_rms is None or 'running_mean_std' not in checkpoint:
        raise ValueError('Stage-1 baseline evaluation requires actor observation RMS')
    if amp_rms is not None and 'amp_input_mean_std' not in checkpoint:
        raise ValueError('Stage-1 baseline evaluation requires AMP observation RMS')
    report = transfer_stage1_weights(model, checkpoint)
    actor_rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    actor_rms.eval()
    if amp_rms is not None:
        amp_rms.load_state_dict(checkpoint['amp_input_mean_std'], strict=True)
        amp_rms.eval()
    model.eval()
    model.requires_grad_(False)
    report['evaluation_mode'] = 'stage1_only'
    report['stage2_training_steps'] = 0
    report['normalizers'] = ['running_mean_std'] + (
        ['amp_input_mean_std'] if amp_rms is not None else [])
    report['context_action_weight_max_abs'] = model.state_dict()[
        FIRST_HEAD_WEIGHT][:, 64:].abs().max().item()
    return report


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
