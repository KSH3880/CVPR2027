from pathlib import Path
import pytest
import torch
from torch import nn
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.stage1_push_network import Stage1PushBuilder, transfer
from stage1_test_utils import observations, all_pairs

ROOT = Path(__file__).resolve().parents[2]


def test_finetune_mixture_and_preserved_environment():
    from copy import deepcopy
    from utils.stage1_jointpush import sample_assets, validate_env
    base = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_jointpush_density3_cone90.yaml').read_text())
    cfg = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_jointpush_density3_cone90_finetune.yaml').read_text())
    validate_env(cfg['env'])
    assert cfg['experiment']['transfer']['checkpoint'] == 'pth/Stage1V2Fixed_00078000.pth'
    restored = deepcopy(cfg)
    restored.pop('experiment')
    restored['env']['relationGraph']['template_probabilities'] = base['env']['relationGraph']['template_probabilities']
    assert restored == base
    torch.manual_seed(42)
    _, weights = sample_assets(50000, cfg['env'], 'cpu')
    kinds = torch.multinomial(weights.flatten(0, 1), 1).flatten()
    observed = torch.bincount(kinds, minlength=6).float()/len(kinds)
    torch.testing.assert_close(observed, torch.tensor([0., .05, .125, .1625, .1625, .5]), atol=.006, rtol=0)


def test_fixed78000_transfer_preserves_old_tasks():
    path = ROOT/'pth/Stage1V2Fixed_00078000.pth'
    if not path.is_file():
        pytest.skip('Local fixed78000 checkpoint required')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 78000
    env = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_jointpush_density3_cone90.yaml').read_text())['env']
    train = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage1_jointpush_density3_cone90.yaml').read_text())
    models = []
    for builder, amp_size in ((AMPMultiAgentBuilder(), 1320), (Stage1PushBuilder(), 1330)):
        builder.load(train['params']['network'])
        model = nn.Module()
        model.a2c_network = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(amp_size,),
            value_size=1, num_agents=2, num_objects=4, observation_mode='clean_scene', scene_entity_sizes=[223,30,1],
            scene_kinematic_size=7, scene_arena_scale=5., relation_reward_mode=env['relationReward']['mode'],
            relation_graph_spec=env['relationGraph'], device='cpu')
        for encoder in (model.a2c_network.actor_encoder, model.a2c_network.critic_encoder):
            encoder.gta_diagnostics_first_forward = False
        models.append(model)
    old, new = models
    old.load_state_dict(checkpoint['model'], strict=True)
    stats, report = transfer(new, checkpoint)
    assert report['source_epoch'] == 78000 and len(report['expanded']) == 3
    for key, tensor in new.state_dict().items():
        if key not in report['expanded']:
            torch.testing.assert_close(tensor, checkpoint['model'][key], rtol=0, atol=0)
    for encoder in (new.a2c_network.actor_encoder, new.a2c_network.critic_encoder):
        torch.testing.assert_close(encoder.edge_encoder.category_embed.weight[4], encoder.edge_encoder.category_embed.weight[2])
    disc = new.a2c_network._disc_mlp[0].weight.reshape(1024, 10, 133)
    assert torch.count_nonzero(disc[..., -1]) == 0
    for key in ('running_mean', 'running_var'):
        torch.testing.assert_close(stats[key][:-1], checkpoint['amp_input_mean_std'][key], rtol=0, atol=0)
    assert stats['running_mean'][-1] == 0 and stats['running_var'][-1] == 1
    raw = observations(all_pairs())
    with torch.no_grad():
        for method in ('eval_actor', 'eval_critic'):
            first, second = (getattr(model.a2c_network, method)(raw) for model in models)
            if isinstance(first, tuple):
                first, second = first[0], second[0]
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
        x = torch.randn(8, 10, 132)
        expanded = torch.cat((x, torch.zeros(8, 10, 1)), -1).flatten(1)
        torch.testing.assert_close(old.a2c_network.eval_disc(x.flatten(1)), new.a2c_network.eval_disc(expanded),
                                   rtol=1e-5, atol=1e-5)
