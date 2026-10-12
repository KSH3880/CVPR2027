"""Speed penalties must follow task ownership, including shuffled physical slots."""
from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
from env.tasks.multi_agent.relation_reward import box_speed_penalty


class Task(EdgeContextTaskMixin):
    num_agents = 2
    _scenario_no_climb = True

    def __init__(self, relation=6):
        self._logical_box_order = torch.tensor([[2, 0, 3, 1], [3, 1, 0, 2]])
        self._agent_box_assignment = self._logical_box_order[:, :2]
        self.relation_runtime = SimpleNamespace(graph=SimpleNamespace(
            num_agents=2,
            edge_valid=torch.ones(2, 2, dtype=torch.bool),
            edge_owner=torch.tensor([[1, 0], [0, 1]]),
            edge_relation=torch.full((2, 2), relation),
            # H0 uses logical O2, H1 uses O3; edge row order differs by scene.
            edge_dst=torch.tensor([[5, 4], [4, 5]])))

    def _assigned_box_values(self, values, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(len(values))
        return values[env_ids[:, None], self._agent_box_assignment[env_ids]]


@pytest.mark.parametrize('relation', [6, 9, 10])  # HOLDING, SIT, CLIMB primary objects
def test_only_owner_of_moving_graph_object_gets_penalty(relation):
    task = Task(relation)
    before = torch.zeros(2, 4, 3)
    after = before.clone()
    after[0, 3, 0] = .1  # H0's object in scene 0
    after[1, 2, 0] = .1  # H1's object in scene 1
    penalty = box_speed_penalty(task._reward_box_values(before),
                               task._reward_box_values(after), 1/30)
    expected = torch.tensor([[-.39346934, 0.], [0., -.39346934]])
    torch.testing.assert_close(penalty, expected)
    legacy = box_speed_penalty(task._assigned_box_values(before),
                              task._assigned_box_values(after), 1/30)
    assert torch.count_nonzero(legacy) == 0  # Original bug silently missed both.


def test_partial_reset_history_uses_new_binding_without_spurious_velocity():
    task = Task()
    values = torch.arange(24.).reshape(2, 4, 3)
    history = task._reward_box_values(values).clone()
    saved_other = history[0].clone()
    # Resample scene 1 only: swap which primary object each owner receives.
    task.relation_runtime.graph.edge_dst[1] = torch.tensor([5, 4])
    ids = torch.tensor([1])
    history[ids] = task._reward_box_values(values, ids)
    torch.testing.assert_close(history[0], saved_other)
    torch.testing.assert_close(history[1], values[1, [2, 0]])
    assert torch.count_nonzero(box_speed_penalty(
        history, task._reward_box_values(values), 1/30)) == 0
    assert task._reward_box_values(values, torch.tensor([], dtype=torch.long)).shape == (0, 2, 3)


def test_legacy_assignment_is_preserved_for_non_scenario_tasks():
    task = Task()
    task._scenario_no_climb = False
    values = torch.arange(24.).reshape(2, 4, 3)
    torch.testing.assert_close(task._reward_box_values(values),
                              values[torch.arange(2)[:, None], task._agent_box_assignment])
    torch.testing.assert_close(task._reward_box_values(values, torch.tensor([1])),
                              values[1:2, [3, 1]])
