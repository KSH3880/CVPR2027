from copy import deepcopy
import torch
from torch import nn

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder


class Stage1PushBuilder(AMPMultiAgentBuilder):
    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)

    class Network(AMPMultiAgentBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            for encoder in (self.actor_encoder, self.critic_encoder):
                encoder.edge_encoder.category_embed = nn.Embedding(5, 64)
                nn.init.trunc_normal_(encoder.edge_encoder.category_embed.weight, std=.02)


def transfer(model, checkpoint):
    from utils.relation_task_spec import check_checkpoint_metadata, checkpoint_metadata
    import yaml
    from pathlib import Path
    env = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1.yaml').read_text())['env']
    check_checkpoint_metadata(checkpoint, checkpoint_metadata(env['relationReward']))
    saved = checkpoint['relation_experiment_config']['env']
    if saved['box']['reset'].get('platformPhysics') != 'fixed_warmup_v1' or 'stage1_push_contract' in checkpoint or 'stage1_jointpush_contract' in checkpoint:
        raise ValueError('PUSH warm-start requires an original Stage1V2Fixed checkpoint')
    state = model.state_dict()
    expanded = []
    for key, target in state.items():
        source = checkpoint['model'][key]
        if source.shape == target.shape:
            target.copy_(source)
        elif key.endswith('category_embed.weight'):
            target[:4].copy_(source)
            target[4].copy_(source[2])
            expanded.append(key)
        elif key.endswith('_disc_mlp.0.weight'):
            target.zero_()
            target.reshape(target.shape[0], 10, 133)[..., :132].copy_(source.reshape(source.shape[0], 10, 132))
            expanded.append(key)
        else:
            raise ValueError('Unexpected PUSH transfer shape: '+key)
    model.load_state_dict(state, strict=True)
    stats = deepcopy(checkpoint['amp_input_mean_std'])
    for key in ('running_mean', 'running_var'):
        fill = stats[key].new_zeros(1) if key == 'running_mean' else stats[key].new_ones(1)
        stats[key] = torch.cat((stats[key], fill))
    return stats, dict(source_epoch=checkpoint['epoch'], expanded=expanded)
