import json
import os
import types

import torch
import yaml

from learning.multi_agent.task_coordination import CoordinationBlock


def ca_none_mask_forward(block, humans, tokens, relations, padding):
    blocked = padding | (relations == 0)[:, None]
    empty = blocked.all(-1).squeeze(1)
    safe = blocked.clone()
    safe[..., 0] &= ~empty[:, None]
    output = CoordinationBlock.forward(block, humans, tokens, relations, safe)
    return torch.where(empty[..., None], humans, output)


def enable_ca_none_mask(coordination):
    for block in coordination.blocks:
        block.forward = types.MethodType(ca_none_mask_forward, block)


def configure_ca_none_mask(player):
    path = os.environ.get('TOKENHSI_INFERENCE_CONFIG')
    if not path:
        return False
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if config != {'actor_ca_none_mask': True}:
        raise ValueError('Expected inference config: actor_ca_none_mask: true')
    if os.environ.get('DEMO_CA_ENTMAX', '0') == '1':
        raise ValueError('NONE mask toy uses softmax; disable DEMO_CA_ENTMAX')
    enable_ca_none_mask(player.model.a2c_network.coordination)
    output = player.env.task.cfg['args'].output_path
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'inference_override.json'), 'w') as handle:
        json.dump(dict(config=config, source=os.path.abspath(path),
                       checkpoint=player.env.task.cfg['args'].checkpoint), handle, indent=2)
    print('[inference] Actor CA: NONE masked to -inf; SELF/COUPLED unchanged', flush=True)
    return True
