import torch
from torch import nn


def sensor_encoder(points, width):
    return nn.Sequential(nn.Linear(points, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, width))


def check_sensor_checkpoint(checkpoint, task):
    expected = getattr(task, 'sensor_metadata', None)
    if checkpoint.get('sensor_metadata') != expected:
        raise ValueError('Sensor checkpoint configuration mismatch; use the matching cliff config')


def transfer_sensor_weights(model, checkpoint):
    metadata = checkpoint.get('relation_metadata', {})
    if (metadata.get('schema_version') != 9 or metadata.get('packet_version') != 3
            or metadata.get('context_fusion') != 'semantic_only'
            or checkpoint.get('sensor_metadata') is not None):
        raise ValueError('Cliff transfer requires an original semantic Stage-1 schema-9 checkpoint')
    source = checkpoint['model']
    target = model.state_dict()
    heads = ('a2c_network.action_head.0.0.weight', 'a2c_network.value_head.0.0.weight')
    prefixes = ('a2c_network.actor_sensor.', 'a2c_network.critic_sensor.')
    copied, new = [], []
    for key, value in target.items():
        if key.startswith(prefixes):
            new.append(key)
            continue
        old = source[key]
        if key in heads:
            if value.shape != (old.shape[0], old.shape[1] * 2):
                raise ValueError('Cliff head must expand 64 to 128 inputs')
            value[:, :old.shape[1]].copy_(old)
            value[:, old.shape[1]:].zero_()
        else:
            if value.shape != old.shape:
                raise ValueError('Cliff transfer shape mismatch: ' + key)
            value.copy_(old)
        copied.append(key)
    unused = set(source) - set(target)
    if unused:
        raise ValueError('Unexpected source parameters: ' + ', '.join(sorted(unused)))
    model.load_state_dict(target, strict=True)
    return {'source_epoch': checkpoint.get('epoch'), 'copied_tensors': copied,
            'new_tensors': new, 'head_expansion': 'actor_and_critic_[W,0]',
            'freeze': 'actor_encoder_and_actor_observation_rms'}
