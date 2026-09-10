import torch
import pytest
from utils.relation_task_spec import compile_carry_subgoal, build_state_relation_matrix
from env.tasks.multi_agent.relation_reward import RelationRuntime


def runtime(n=2, m=2, o=3):
    return RelationRuntime(n, compile_carry_subgoal(m, o), {}, 'cpu')


def test_graph_layout_and_directions():
    g = compile_carry_subgoal(2, 3)
    assert g.edge_src.tolist() == [0, 2, 1, 3]
    assert g.edge_dst.tolist() == [2, 5, 3, 6]
    assert g.edge_relation.tolist() == [6, 7, 6, 7]
    assert g.edge_owner.tolist() == [0, 0, 1, 1]
    assert g.prereq_mask.nonzero().tolist() == [[1, 0], [3, 2]]
    matrix = build_state_relation_matrix(2, 3)
    assert matrix[2, 0] == 0 and matrix[0, 2] == 6 and matrix[2, 5] == 7
    assert (matrix.diag() == 1).all()
    with pytest.raises(ValueError):
        compile_carry_subgoal(3, 2)


def test_one_time_bonus_previous_achievement_and_independent_agents():
    r = runtime(1)
    r.reset(torch.tensor([0]), torch.zeros(1, 4))
    phi = torch.tensor([[1., 1., 0., 0.]])
    first = r.step(phi, torch.zeros_like(phi))
    assert not first['first_success'].any()  # same-step touch isn't previous achievement
    second = r.step(phi, torch.zeros_like(phi))
    assert second['success_bonus'].tolist() == [[5., 0.]]
    third = r.step(torch.tensor([[0., 0., .5, 0.]]), torch.ones_like(phi))
    assert third['agent_task_reward'][0, 0] == 0
    assert third['agent_task_reward'][0, 1] > 0
    assert r.done.tolist() == [[True, False]]
    fourth = r.step(phi, torch.ones_like(phi))
    assert fourth['success_bonus'].sum() == 0


def test_reset_seeding_subset_and_saved_suffix():
    r = runtime()
    r.reset(torch.arange(2), torch.tensor([[1., 1., .1, .2], [.3, .4, .5, .6]]))
    assert r.done.tolist() == [[True, False], [False, False]]
    stored = r.suffix().clone()
    assert stored.shape == (2, 18)
    r.reset(torch.tensor([0]), torch.tensor([[.2, .3, .4, .5]]))
    torch.testing.assert_close(r.suffix()[1], stored[1])
    assert not r.achieved[0].any() and not r.done[0].any()
    assert stored[0, -2] == 1  # reset cannot mutate a stored rollout observation
    same = runtime()
    same.phi.copy_(r.phi)
    same.achieved[0, 0] = True
    assert not torch.equal(same.suffix()[0], r.suffix()[0])


def test_documented_pick_once_then_kick_and_goal_first_touch():
    r = runtime(1, 1, 1)
    r.reset(torch.tensor([0]), torch.tensor([[1., 0.]]))
    out = r.step(torch.tensor([[0., 1.]]), torch.zeros(1, 2))
    assert out['success_bonus'].item() == 5  # limitation, not prevention assertion
    r.reset(torch.tensor([0]), torch.tensor([[0., 1.]]))
    assert not r.done.any()
    out = r.step(torch.tensor([[1., 1.]]), torch.zeros(1, 2))
    assert out['success_bonus'].item() == 0
    out = r.step(torch.tensor([[0., 1.]]), torch.zeros(1, 2))
    assert out['success_bonus'].item() == 5


def test_signed_runtime_keeps_state2_weight_and_semantic_suffix():
    cfg = {'state_delta_weight': 2., 'progress': {'mode': 'signed_linear'},
           'diagnostics': {'validate_tensors': True}}
    r = RelationRuntime(1, compile_carry_subgoal(1, 1), cfg, 'cpu')
    r.reset(torch.tensor([0]), torch.tensor([[.8, .2]]))
    out = r.step(torch.tensor([[.9, .4]]), torch.full((1, 2), -.5))
    torch.testing.assert_close(out['state_component'], torch.tensor([[.2, .2]]))
    torch.testing.assert_close(out['velocity_component'], torch.tensor([[-.05, -.05]]))
    torch.testing.assert_close(out['agent_task_reward'], torch.tensor([[.3]]))
    assert r.suffix().shape == (1, 9)
    assert ((r.suffix() >= 0) & (r.suffix() <= 1)).all()
