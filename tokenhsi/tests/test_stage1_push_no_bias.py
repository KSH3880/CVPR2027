"""Task-bias ablation preserves GTA, Action MLP, masks and experiment identity."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from learning.multi_agent.stage1_push_network import Stage1PushBuilder
from stage1_test_utils import observations
from utils.experiments import native_config, validate_experiment, check_push_bias_checkpoint
from utils.stage1_jointpush import compose_graph
from utils.unified_training import validate_stage1_network
from utils.output_paths import resolve_output_path

NAME = 'stage1_push_no_bias'


def build(name=NAME):
    torch.set_num_threads(1)
    torch.manual_seed(25)
    env = native_config(name)['env']
    builder = Stage1PushBuilder()
    builder.load(native_config(name, training=True)['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1330,),
        value_size=1, num_agents=2, num_objects=4, observation_mode='clean_scene',
        scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=env['relationReward']['mode'], relation_graph_spec=env['relationGraph'], device='cpu')
    for enc in (net.actor_encoder, net.critic_encoder):
        enc.gta_diagnostics_first_forward = False
        enc.relation_graph_spec['shuffle_token_order'] = False
    return net


def test_only_attention_bias_and_run_name_differ_from_baseline():
    assert resolve_output_path(NAME, 'train') == 'output/' + NAME
    assert resolve_output_path(NAME, 'train', check=True).startswith('output_etc/check/')
    assert resolve_output_path(NAME, 'eval').startswith('output_etc/check/')
    assert native_config(NAME) == native_config('stage1_push')
    train = native_config(NAME, training=True)
    validate_stage1_network(native_config(NAME)['env'], train)
    validate_experiment(NAME, native_config(NAME)['env'], train)
    normalized = deepcopy(train)
    normalized['params']['network']['transformer']['relation_bias'] = True
    normalized['params']['config']['name'] = 'Stage1Push'
    assert normalized == native_config('stage1_push', training=True)
    for name, enabled in ((NAME, True), ('stage1_push', False)):
        wrong = native_config(name, training=True)
        wrong['params']['network']['transformer']['relation_bias'] = enabled
        with pytest.raises(ValueError, match='bias'):
            validate_experiment(name, native_config(name)['env'], wrong)


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_bias_computation_is_skipped_and_zero_bias_matches_baseline(branch):
    baseline, ablated = build('stage1_push'), build()
    ablated.load_state_dict(baseline.state_dict(), strict=True)
    reference = getattr(baseline, branch + '_encoder')
    enc = getattr(ablated, branch + '_encoder')
    obs = observations(compose_graph(torch.tensor([[1, 2], [3, 4], [5, 5]]), num_objects=4))
    with torch.no_grad(), patch.object(enc.context_fusion, 'forward', side_effect=AssertionError('bias fusion called')), \
         patch.object(enc.edge_encoder, 'forward', side_effect=AssertionError('bias background called')), \
         patch.object(enc.edge_encoder, 'bias_table', side_effect=AssertionError('bias projection called')):
        torch.testing.assert_close(enc(obs), reference(obs), rtol=0, atol=0)
        with torch.no_grad():
            enc.edge_encoder.bias_projection.fill_(100)
        torch.testing.assert_close(enc(obs), reference(obs), rtol=0, atol=0)
    assert not enc.use_relation_bias
    assert not enc.edge_encoder.bias_projection.requires_grad
    assert enc.edge_encoder.category_embed.weight.requires_grad
    assert enc.last_shape_flow['gta'] and enc.last_shape_flow['relation_bias'] is False


@pytest.mark.parametrize('branch', ['actor', 'critic'])
def test_task_condition_gradients_and_permuted_padding_are_preserved(branch):
    net = build()
    enc = getattr(net, branch + '_encoder')
    head = net.action_head if branch == 'actor' else net.value_head
    with torch.no_grad():
        enc.task_action_fusion.mlp[-1].weight.normal_(std=.03)
    obs = observations(compose_graph(torch.tensor([[5, 3], [1, 2], [4, 5]]), num_objects=4))
    mask = torch.tensor([[True, True, True, False]]).expand(len(obs), -1)
    original = enc(obs, object_valid=mask)
    permuted = enc(obs, token_order=torch.tensor([7, 5, 0, 2, 1, 6, 4, 3]), object_valid=mask)
    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-5)
    altered = obs.clone()
    altered[:, 446+3*30:446+4*30] += 100  # Masked object token.
    torch.testing.assert_close(original, enc(altered, object_valid=mask), rtol=0, atol=0)
    changed_task = obs.clone()
    changed_task[:, -9] = (changed_task[:, -9]+1) % 5
    assert not torch.equal(enc(obs), enc(changed_task))
    loss = head(original).square().mean()
    loss.backward()
    assert enc.edge_encoder.bias_projection.grad is None
    embedding_grad = enc.edge_encoder.category_embed.weight.grad
    assert torch.isfinite(embedding_grad).all() and embedding_grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in enc.task_action_fusion.parameters())


@pytest.mark.parametrize('saved_bias', [False, True])
def test_checkpoint_bias_cannot_silently_change(saved_bias):
    name = 'stage1_push' if saved_bias else NAME
    checkpoint = {'relation_experiment_config': {'train': native_config(name, training=True)}}
    check_push_bias_checkpoint(checkpoint, saved_bias)
    with pytest.raises(ValueError, match='bias'):
        check_push_bias_checkpoint(checkpoint, not saved_bias)
    check_push_bias_checkpoint({}, True)  # Historical checkpoint fallback.
    with pytest.raises(ValueError, match='bias'):
        check_push_bias_checkpoint({}, False)


def test_formal_single_evaluator_routes_all_five_tasks(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    checkpoint = tmp_path / 'model.pth'
    checkpoint.touch()
    output = tmp_path / 'evaluation'
    commands = []
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', NAME, str(checkpoint), '--profile',
        'stage1_single', '--output', str(output), '--envs', '9', '--seeds', '0'])
    monkeypatch.setattr('subprocess.run', lambda command, **kwargs:
        (commands.append(command) or SimpleNamespace(returncode=0)))
    runpy.run_path(str(Path('tokenhsi/evaluate.py').resolve()), run_name='__main__')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['tasks'] == ['AT', 'ON_TOP', 'SIT', 'CLIMB', 'PUSH']
    assert manifest['formal_report'] is False and manifest['horizon'] == 900
    assert len(commands) == 5
    for command, task in zip(commands, manifest['tasks']):
        assert command[command.index('--task')+1] == 'HumanoidMAPush'
        assert command[command.index('--cfg_train')+1].endswith('/'+NAME+'.yaml')
        assert command[command.index('--motion_file')+1].endswith('dataset_stage1_jointpush.yaml')
        assert command[command.index('--eval_single_task')+1] == task
