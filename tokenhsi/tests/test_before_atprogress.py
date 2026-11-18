from copy import deepcopy
from pathlib import Path
import math

import pytest
import torch
import yaml

from env.tasks.multi_agent.before_reward import BeforeRuntime, predecessor_at_progress
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from utils.before_spec import sample_graph, validate_env, checkpoint_contract
from utils.edge_ontop_spec import permute_graph, select_graph
from utils.before_spec import check_checkpoint

ROOT = Path(__file__).resolve().parents[2]
NAME = 'stage2_before'
CFG = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{NAME}.yaml').read_text())
ENV = CFG['env']




def fixture(family):
    torch.manual_seed(25)
    g, pre, _ = sample_graph(torch.full((4, 4, 3), .4), ENV['box']['build']['taskSizeRanges'], 'one_pair_'+family)
    roots = torch.zeros(4, 4, 3)
    boxes = torch.zeros(4, 8, 13); boxes[..., 6] = 1.
    sizes = torch.tensor([.4, .6, .4]).expand(4, 8, 3).clone()
    goals = torch.zeros(4, 4, 3)
    rows, followers = (pre >= 0).nonzero(as_tuple=True)
    edges = (g.prereq_mask.any(-1)).long().argmax(-1)
    leader_edges = g.prereq_mask[rows, edges].long().argmax(-1)
    goal_ids = g.edge_dst[rows, leader_edges] - 12
    support = g.edge_dst[rows, edges] - 4
    boxes[rows, support, :2] = torch.tensor([-3., 1.])
    roots[rows, followers, :2] = torch.tensor([3., 4.])
    if family == 'stack':
        boxes[rows, g.edge_src[rows, edges]-4, :2] = torch.tensor([6., 8.])
    return g, roots, boxes, sizes, goals, rows, edges, goal_ids, support


@pytest.mark.parametrize('family,buffer,distance', [('sit', .3, 5.), ('climb', .3, 5.), ('stack', .1, 10.)])
def test_progress_uses_leader_goal_with_original_radius_and_preserves_other_edges(family, buffer, distance):
    g, roots, boxes, sizes, goals, rows, edges, goal_ids, support = fixture(family)
    # The geometry evaluator is unchanged; only its progress output is retargeted.
    hands = roots[:, :, None].expand(-1, -1, 2, -1)
    phi, diag = evaluate_interaction_edges(hands, hands, roots, boxes, sizes, goals, g, ENV['relationReward'], .9)
    old = diag['progress'].clone()
    args = (roots, boxes, sizes, goals, g, ENV['relationReward'])
    actual = predecessor_at_progress(old, *args)
    radius = math.hypot(.4, .6)/2 + buffer
    expected = 1/(1+distance-radius)
    torch.testing.assert_close(actual[rows, edges], torch.full((4,), expected))
    unchanged = ~g.prereq_mask.any(-1)
    assert torch.equal(actual[unchanged], old[unchanged])  # Includes HOLDING and solo edges.
    assert torch.equal(diag['progress'], old)

    moved = boxes.clone(); moved[rows, support, :2] += torch.tensor([17., -9.])
    torch.testing.assert_close(predecessor_at_progress(old, roots, moved, sizes, goals, g, ENV['relationReward']), actual)
    moved_goals = goals.clone()
    moved_goals[rows, goal_ids, :2] = torch.tensor([3., 4.] if family != 'stack' else [6., 8.])
    close = predecessor_at_progress(old, roots, boxes, sizes, moved_goals, g, ENV['relationReward'])
    assert (close[rows, edges] == 1).all()
    moved_goals[..., 2] = 100  # Proximity remains XY-only.
    torch.testing.assert_close(predecessor_at_progress(old, roots, boxes, sizes, moved_goals, g, ENV['relationReward']), close)
    if family == 'stack':
        moved_roots = roots + 20
        torch.testing.assert_close(predecessor_at_progress(old, moved_roots, boxes, sizes, goals, g, ENV['relationReward']), actual)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_gate_does_not_switch_progress_target_or_change_state_success_holding(family):
    g, roots, boxes, sizes, goals, rows, edges, _, _ = fixture(family)
    old = torch.full(g.edge_relation.shape, .4)
    progress = predecessor_at_progress(old, roots, boxes, sizes, goals, g, ENV['relationReward'])
    phi = torch.full_like(old, .1); zero = torch.zeros_like(old)
    runtime = BeforeRuntime(4, g, ENV['relationReward'], 'cpu')
    for at_distance in (.3, .1, .3):
        phi[g.edge_relation == 7] = math.exp(-10*at_distance**2)
        # Prevent follower success so saturation does not hide the proximity term.
        failed = torch.ones_like(zero)
        failed[g.edge_relation == 7] = 0
        a = runtime.components(phi, progress, failed, failed, failed, g)
        b = runtime.components(phi, old, failed, failed, failed, g)
        assert (a['prerequisite_gate'][rows, edges] == (at_distance < .2)).all()
        torch.testing.assert_close(a['progress_component'][rows, edges], progress[rows, edges]*.2)
        for key in ('state_component', 'success_component', 'own_success', 'reward_saturated'):
            assert torch.equal(a[key], b[key])
        assert torch.equal(a['total'][~g.prereq_mask.any(-1)], b['total'][~g.prereq_mask.any(-1)])
    # Real success still saturates progress, including paired HOLDING on ON_TOP.
    success = runtime.components(torch.ones_like(phi), progress, zero, zero, zero, g)
    assert (success['progress_component'][rows, edges] == .2).all()


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_graph_permutation_and_partial_batch_preserve_goal_binding(family):
    g, roots, boxes, sizes, goals, *_ = fixture(family)
    old = torch.rand(4, 8)
    expected = predecessor_at_progress(old, roots, boxes, sizes, goals, g, ENV['relationReward'])
    permutation = torch.rand(4, 8).argsort(-1)
    actual = predecessor_at_progress(old.gather(1, permutation), roots, boxes, sizes, goals,
                                    permute_graph(g, permutation), ENV['relationReward'])
    torch.testing.assert_close(actual, expected.gather(1, permutation))
    ids = torch.tensor([3, 1])
    actual = predecessor_at_progress(old[ids], roots[ids], boxes[ids], sizes[ids], goals[ids],
                                    select_graph(g, ids), ENV['relationReward'])
    torch.testing.assert_close(actual, expected[ids])


def test_all_solo_is_exactly_unchanged():
    g, *_ = sample_graph(torch.full((3, 4, 3), .4), ENV['box']['build']['taskSizeRanges'], 'all_solo')
    old = torch.rand(3, 8)
    actual = predecessor_at_progress(old, torch.randn(3, 4, 3), torch.randn(3, 8, 13),
        torch.ones(3, 8, 3), torch.randn(3, 4, 3), g, ENV['relationReward'])
    assert torch.equal(actual, old)
