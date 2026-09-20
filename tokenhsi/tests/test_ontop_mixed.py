import copy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from env.tasks.multi_agent.relation_reward import RelationRuntime, relation_gate
from env.tasks.multi_agent.ontop_task import OnTopTaskMixin
from learning.multi_agent.amp_network_builder_ma import RelationEncoder
from learning.multi_agent.transfer import transfer_carry_weights
from utils.ontop_task_spec import (ONTOP_MODE, SCENARIOS, mixed_graph, mixed_policy_graph,
                                    scenario_ids, terminal_geometry)
from utils.relation_task_spec import (STATE_MODE, validate_relation_config,
                                     compile_carry_subgoal, checkpoint_metadata)


def configs():
    directory = Path(__file__).parents[1] / 'data/cfg/multi_agent'
    return [yaml.safe_load((directory / name).read_text()) for name in
            ('approach_distance_success.yaml', 'approach_distance_success_ontop_mixed.yaml')]


def make_encoder(mode):
    return RelationEncoder([223, 30, 1], 2, 3, 16, 2, 2, 32,
        lambda size: nn.Sequential(nn.Linear(size, 16), nn.ReLU()),
        observation_mode='clean_scene', kinematic_size=7, relation_bias_mode='edge_mlp',
        gta_cfg={'enable': True}, relation_reward_mode=mode)


def scenes():
    x = torch.randn(6, 607)
    x[:, 536:538] = 1
    x[:, 538:587].reshape(6, 7, 7)[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    x[:, 587:605] = torch.rand(6, 18)
    x[:, -2] = torch.arange(6) // 2
    x[:, -1] = torch.arange(6) % 2
    return x


def test_config_split_validation_and_graph_roles():
    _, cfg = configs()
    validate_relation_config(cfg['env']['relationReward'])
    mix = cfg['env']['scenarioMixture']
    ids = scenario_ids(mix, 2048)
    assert ids.bincount().tolist() == [512, 768, 768]
    assert scenario_ids(mix, 16, True).bincount().tolist() == [4, 6, 6]
    assert scenario_ids(mix, 1, True, SCENARIOS[2]).tolist() == [2]
    with pytest.raises(ValueError, match='numEnvs'):
        scenario_ids(mix, 16)
    bad = copy.deepcopy(mix)
    bad['scenarios'][2]['edges'][-1]['prerequisites'] = ['holding_b']
    with pytest.raises(ValueError, match='graph'):
        scenario_ids(bad, 2048)
    scenario, a = torch.arange(6) // 2, torch.arange(6) % 2
    graph, matrices, valid = mixed_policy_graph(scenario, a)
    for row in range(6):
        b = 1 - a[row].item()
        terminal = 2 * b + 1
        assert graph.prereq_mask[row, terminal, 2 * b]
        assert graph.prereq_mask[row, terminal, 2 * a[row] + 1] == (scenario[row] == 2)
        assert matrices[row, 2 + b, graph.edge_dst[row, terminal]] == (7 if row < 2 else 8)
        assert valid[row].sum() == (7 if row < 2 else 6)


def test_surface_geometry_tracks_rotated_support_and_progress_center():
    states = torch.zeros(3, 3, 13)
    states[..., 6] = 1
    sizes = torch.full((3, 3, 3), .4)
    states[:, 0, 2] = .2
    states[:, 1, 2] = .6
    states[:, 2, 0] = 2
    states[:, 2, 2] = .2
    goals = torch.zeros(3, 2, 3)
    source, target, progress, mask = terminal_geometry(states, sizes, goals, torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long))
    assert mask.tolist() == [[False, False], [False, True], [False, True]]
    torch.testing.assert_close(source[2, 1], target[2, 1])
    torch.testing.assert_close(progress[2, 1], states[2, 0, :3])
    assert target[1, 1, 0] == 2
    states[2, 0, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    _, target, _, _ = terminal_geometry(states, sizes, goals, torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(target[2, 1], torch.tensor([0., -.2, .2]))


def test_soft_dependency_previous_state_and_local_success_override():
    _, cfg = configs()
    graph = mixed_graph(torch.tensor([1, 2]), torch.zeros(2, dtype=torch.long))
    runtime = RelationRuntime(2, graph, cfg['env']['relationReward'], 'cpu')
    phi = torch.tensor([[1., 0., 1., .5], [1., 0., 1., .5]])
    runtime.reset(torch.arange(2), phi, torch.ones(2, 2))
    result = runtime.step(phi, torch.ones_like(phi), at_z_error=torch.ones(2, 2))
    assert result['activation'][0, 3] > .99
    assert result['activation'][1, 3] < 1e-9
    next_phi = phi.clone()
    next_phi[:, 1] = 1
    result = runtime.step(next_phi, torch.ones_like(phi), at_z_error=torch.ones(2, 2))
    assert result['activation'][1, 3] < 1e-9  # previous step A state
    result = runtime.step(next_phi, torch.ones_like(phi), at_z_error=torch.ones(2, 2))
    assert result['activation'][1, 3] > .99
    runtime.phi.copy_(phi)
    phi[:, 3] = 1
    z = torch.ones(2, 2)
    z[:, 1] = 0
    result = runtime.step(phi, torch.ones_like(phi), at_z_error=z)
    assert not result['current_success_state'][:, 0].any()
    torch.testing.assert_close(result['agent_task_reward'][:, 1], torch.ones(2))
    assert result['activation'][1, 3] < 1e-9  # saturation intentionally bypasses gating


def test_carry_group_exact_reward_parity_and_partial_reset_isolation():
    base, cfg = configs()
    graph = mixed_graph(torch.tensor([0, 0]), torch.tensor([0, 1]))
    mixed = RelationRuntime(2, graph, cfg['env']['relationReward'], 'cpu')
    original = RelationRuntime(2, compile_carry_subgoal(2, 3), base['env']['relationReward'], 'cpu')
    for _ in range(10):
        phi, progress, z = torch.rand(2, 4), torch.rand(2, 4), torch.rand(2, 2)
        a = mixed.step(phi, progress, at_z_error=z)
        b = original.step(phi, progress, at_z_error=z)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    before = mixed.suffix().clone()
    mixed.reset(torch.tensor([1]), torch.ones(1, 4), torch.zeros(1, 2))
    torch.testing.assert_close(mixed.suffix()[0], before[0])


def test_encoder_permutation_batch_reordering_mask_and_gradient():
    torch.manual_seed(7)
    encoder, x = make_encoder(ONTOP_MODE), scenes()
    with torch.no_grad():
        encoder.edge_encoder.bias_projection.normal_(std=.1)
        encoder.dynamic_bias_projection.normal_(std=.1)
    output = encoder(x)
    indices = torch.tensor([4, 0, 5, 2, 1, 3])
    torch.testing.assert_close(encoder(x[indices]), output[indices])
    y = torch.cat([x[:, :446].reshape(6, 2, 223)[:, [1, 0]].flatten(1),
                   x[:, 446:536].reshape(6, 3, 30)[:, [1, 0, 2]].flatten(1),
                   x[:, 536:538][:, [1, 0]],
                   x[:, 538:587].reshape(6, 7, 7)[:, [1, 0, 3, 2, 4, 6, 5]].flatten(1),
                   x[:, 587:603].reshape(6, 2, 2, 4)[:, [1, 0]].flatten(1),
                   x[:, 603:605][:, [1, 0]], x[:, -2:-1], 1 - x[:, -1:]], -1)
    torch.testing.assert_close(encoder(y), output[:, [1, 0]], atol=3e-6, rtol=3e-6)
    changed = x.clone()
    # Row 4: dependent, base=0, inactive goal token 6. Its content/pose must not affect humans.
    changed[4, 537] = 100
    changed[4, 538 + 6 * 7:538 + 6 * 7 + 3] = 123
    torch.testing.assert_close(encoder(changed), output)
    output.square().mean().backward()
    assert encoder.edge_encoder.relation_embed.weight.grad[8].abs().sum() > 0
    assert encoder.tokenizers[0][0].weight.grad.abs().sum() > 0


def test_transfer_copies_every_old_tensor_and_preserves_new_rows():
    base, cfg = configs()
    def model(mode):
        result = nn.Module()
        result.actor_encoder = make_encoder(mode)
        result.critic_encoder = make_encoder(mode)
        return result
    source, target = model(STATE_MODE), model(ONTOP_MODE)
    checkpoint = dict(model=source.state_dict(), epoch=18000,
                      relation_metadata=checkpoint_metadata(base['env']['relationReward']))
    new_row = target.actor_encoder.edge_encoder.relation_embed.weight[8].detach().clone()
    report = transfer_carry_weights(target, checkpoint, cfg['env']['relationReward'])
    assert len(report['expanded_embeddings']) == 2
    for key, old in source.state_dict().items():
        actual = target.state_dict()[key]
        if key.endswith('relation_embed.weight'):
            actual = actual[:8]
        torch.testing.assert_close(old, actual, atol=0, rtol=0)
    torch.testing.assert_close(target.actor_encoder.edge_encoder.relation_embed.weight[8], new_row)
    x = scenes()[:2]
    torch.testing.assert_close(source.actor_encoder(x[:, :605]), target.actor_encoder(x))
    bad = copy.deepcopy(checkpoint)
    bad['relation_metadata']['relation_reward_config']['holding']['hand_distance_scale'] = 10
    with pytest.raises(ValueError, match='reward mismatch'):
        transfer_carry_weights(target, bad, cfg['env']['relationReward'])


def test_dependent_metrics_record_first_b_success_without_requiring_a_success():
    task = OnTopTaskMixin()
    task.num_envs, task.device = 3, 'cpu'
    task._ontop_scenario_list = [0, 1, 2]
    _, cfg = configs()
    task.relation_runtime = RelationRuntime(3, compile_carry_subgoal(2, 3),
                                           cfg['env']['relationReward'], 'cpu')
    task._init_ontop_runtime()
    task._box_states = torch.zeros(3, 3, 13)
    task._tar_pos = torch.zeros(3, 2, 3)
    task._tar_pos[2, 0, 0] = 2
    task._assigned_box_values = lambda states: states[:, :2]
    result = dict(current_success_state=torch.tensor([[True, True], [False, False], [False, True]]),
                  agent_task_reward=torch.ones(3, 2), activation=torch.ones(3, 4))
    task._ontop_step_metrics(result)
    assert task._ontop_first_distance.tolist() == [-1., -1., 2.]
    task._tar_pos[2, 0, 0] = 0
    result['current_success_state'][2, 0] = True
    task._ontop_step_metrics(result)
    assert task._ontop_first_distance[2] == 2  # preserve first occurrence
    task.reset_buf = torch.ones(3, dtype=torch.long)
    task._finish_ontop_metrics()
    metrics = task._consume_ontop_metrics()
    prefix = 'scenario/carry_ontop_dependent/'
    assert metrics[prefix + 'first_b_success_oa_ga_distance'] == 2
    assert metrics[prefix + 'first_b_success_oa_ga_distance_samples'] == 1
    assert metrics[prefix + 'final_joint_success'] == 1
    assert metrics[prefix + 'joint_retention_after_first'] == 1
