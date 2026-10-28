from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.stage2_jointpush_reward import Stage2JointPushRuntime
from utils.stage2_jointpush import (
    checkpoint_contract, compose_graph, group_contact_gate, validate_env,
)


@pytest.mark.parametrize('base', ['stage2_jointpush', 'stage2_jointpush_rsi'])
def test_goal_saturation_config_preserves_base(base):
    folder = Path('tokenhsi/data/cfg/multi_agent')
    original = yaml.safe_load((folder / (base + '.yaml')).read_text())
    variant = yaml.safe_load((folder / (base + '_goal_saturation.yaml')).read_text())
    validate_env(variant['env'])
    assert variant['env']['stage2JointPush']['goal_saturation'] is True
    assert checkpoint_contract(original['env']) != checkpoint_contract(variant['env'])
    restored = deepcopy(variant)
    restored['env']['stage2JointPush'].pop('goal_saturation')
    restored['experiment']['name'] = original['experiment']['name']
    assert restored == original
    folder = Path('tokenhsi/data/cfg/train/rlg')
    original = yaml.safe_load((folder / (base + '.yaml')).read_text())
    variant = yaml.safe_load((folder / (base + '_goal_saturation.yaml')).read_text())
    assert variant['params']['config']['name'] == original['params']['config']['name'] + 'GoalSaturation'
    variant['params']['config']['name'] = original['params']['config']['name']
    assert variant == original


@pytest.mark.parametrize('base', ['stage2_jointpush', 'stage2_jointpush_rsi'])
def test_goal_saturation_arrive_release_leave_and_height(base):
    env = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/' + base + '_goal_saturation.yaml').read_text())['env']
    groups = torch.tensor([[0, 0, 2, 3]])
    graph = compose_graph(torch.full((1, 4), 5), groups, False)
    runtime = Stage2JointPushRuntime(1, graph, env['relationReward'], 'cpu', contact_weights=[.2] * 3)
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 6] = 1.
    sizes = torch.tensor([4., 2., 1.5]).expand(1, 4, 3)
    hands = torch.full((1, 4, 2, 3), 5.)
    goals = torch.zeros(1, 4, 3)
    shared_at = (graph.edge_relation == 7) & (graph.edge_owner < 2)
    zero = torch.zeros(1, 8)

    def step(y, z=0.):
        goals[..., 1], goals[..., 2] = y, z
        runtime.gate[:] = group_contact_gate(
            hands, boxes, sizes, goals, groups,
            goal_saturation=env['stage2JointPush']['goal_saturation'])
        distance = goals.norm(dim=-1)
        phi = torch.where(graph.edge_relation == 7,
                          torch.exp(-10 * distance.square()).gather(1, graph.edge_owner), .3)
        progress = torch.where(graph.edge_relation == 7,
                               (1 / (1 + (goals[..., :2].norm(dim=-1) - .1).clamp_min(0))).gather(1, graph.edge_owner), .4)
        z_error = goals[..., 2].abs().gather(1, graph.edge_owner)
        return runtime.step(phi, progress, z_error, zero, zero)

    result = step(1.)
    assert not runtime.gate[0, :2].any()
    assert (result['total'][shared_at] == 0).all()
    assert runtime.gate[0, 2:].all()

    result = step(.1)
    assert runtime.gate.all()
    assert not runtime.contact.any()
    torch.testing.assert_close(result['agent_task_reward'], torch.full((1, 4), 1.2))
    assert result['own_success'][shared_at].all()
    for key in ('state_component', 'progress_component', 'success_component'):
        torch.testing.assert_close(result[key], torch.full((1, 8), .2))
    assert not group_contact_gate(hands, boxes, sizes, goals, groups)[0, :2].any()

    for y, z in ((.21, 0.), (.1, .002)):
        result = step(y, z)
        assert not runtime.gate[0, :2].any()
        assert not result['own_success'][shared_at].any()
        assert (result['total'][shared_at] == 0).all()
        assert runtime.achieved[shared_at].all()

    hands[:] = 0.
    hands[..., 1] = -1.
    result = step(1.)
    assert runtime.gate.all()
    assert (result['state_component'][shared_at] > 0).all()
    assert (result['progress_component'][shared_at] > 0).all()
    assert (result['success_component'][shared_at] == 0).all()
