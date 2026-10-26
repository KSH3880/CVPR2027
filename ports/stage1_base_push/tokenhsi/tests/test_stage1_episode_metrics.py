import torch
from utils.stage1_episode_metrics import Stage1EpisodeMetrics


def test_completed_episodes_task_denominators_and_ever_success():
    m = Stage1EpisodeMetrics(2, 2, 'cpu')
    ids = torch.arange(2)
    start = torch.tensor([[6., 1.], [8., 1.]])
    no = torch.zeros(2, 2, dtype=torch.bool)
    m.reset(ids, torch.tensor([[5, 1], [5, 4]]),
            torch.tensor([[True, False], [False, False]]), start, no)
    m.observe(torch.tensor([[3., 1.], [2., 1.]]), torch.tensor([[True, False], [False, True]]))
    # First scene ends. Ever-success differs from success at the final step.
    m.finish(torch.tensor([True, False]), torch.tensor([[4., 1.], [2., 1.]]), no)
    a = m.consume()
    assert a['episode/push_at/completed_agents'] == 1
    assert a['episode/push_at/final_success'] == 0
    assert a['episode/push_at/ever_success'] == 1
    assert a['episode/push_at/start_distance_m'] == 6  # post-RSI, not original 10 m
    torch.testing.assert_close(a['episode/push_at/progress_fraction'], torch.tensor(1/3))
    assert a['episode/push_at/best_progress_fraction'] == .5
    assert a['episode/push_at_rsi/completed_agents'] == 1
    assert a['episode/push_at_loco/completed_agents'] == 0
    assert 'episode/push_at_loco/final_success' not in a
    assert a['episode/sit/completed_agents'] == 1
    assert a['episode/holding_ontop/completed_agents'] == 0
    # Consume doesn't erase ongoing episode history; completed agents aren't counted twice.
    m.finish(torch.tensor([True, True]), torch.tensor([[4., 1.], [0., 1.]]), no)
    b = m.consume()
    assert b['episode/push_at/completed_agents'] == 1
    assert b['episode/push_at/ever_success'] == 0
    assert b['episode/push_at_loco/progress_fraction'] == 1
    assert b['episode/holding_ontop/ever_success'] == 1


def test_partial_reset_and_moving_away_and_zero_distance():
    m = Stage1EpisodeMetrics(2, 2, 'cpu')
    no = torch.zeros(2, 2, dtype=torch.bool)
    m.reset(torch.arange(2), torch.full((2, 2), 5), no, torch.ones(2, 2), no)
    m.observe(torch.zeros(2, 2), ~no)
    m.reset(torch.tensor([0]), torch.full((1, 2), 5), no[:1], torch.zeros(1, 2), no[:1])
    m.observe(torch.tensor([[2., 2.], [0., 0.]]), no)
    m.finish(torch.ones(2, dtype=torch.bool), torch.tensor([[2., 2.], [0., 0.]]), no)
    a = m.consume()
    assert a['episode/push_at/ever_success'] == .5
    assert a['episode/push_at/progress_fraction'] == .5
    assert a['episode/push_at/best_progress_fraction'] == .5
    assert a['episode/push_at/distance_reduction_m'] == -.5
    assert all(torch.isfinite(v) for v in a.values())
    assert m.consume()['episode/push_at/completed_agents'] == 0
