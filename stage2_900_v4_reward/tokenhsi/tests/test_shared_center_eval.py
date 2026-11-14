import json

import pytest
import torch

from utils.coupled_push_spec import region_states, compose_graph
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import (spatial_success, shared_center_success,
    shared_center_scores, shared_centers_from_tokens)


@pytest.mark.parametrize('count', [2, 3, 4])
def test_rotated_box_center_can_succeed_while_personal_targets_fail(count):
    boxes = torch.zeros(1, count, 13)
    boxes[..., 6] = 1.
    sizes = torch.tensor([4., 2., 1.]).expand(1, count, 3).clone()
    groups = torch.zeros(1, count, dtype=torch.long)
    members = torch.full_like(groups, count)
    regions = torch.arange(count)[None]
    tokens, cells = region_states(boxes, sizes, members, regions)
    goals = tokens[..., :3]+torch.tensor([0., 4., 0.])
    center_goals = boxes[..., :3]+torch.tensor([0., 4., 0.])
    boxes[..., 1] = 4.
    boxes[..., 5], boxes[..., 6] = 2**-.5, 2**-.5
    rotated, _ = region_states(boxes, sizes, members, regions)
    graph = compose_graph(torch.full_like(groups, 5), groups, False, individual_goals=True)
    packet = task_graph_packet(graph, 1).reshape(1, count, 5)
    state = torch.cat((rotated, rotated), 1)
    personal = spatial_success(packet, torch.zeros(1, count, 3), state,
        torch.cat((cells, cells), 1), goals)
    assert not personal.all()
    valid = torch.ones_like(groups, dtype=torch.bool)
    assert shared_center_success(boxes[..., :3], center_goals, groups, valid).all()
    recovered, recovered_goals = shared_centers_from_tokens(packet, state, goals, groups)
    torch.testing.assert_close(recovered, boxes[..., :3].double(), atol=1e-6, rtol=0)
    torch.testing.assert_close(recovered_goals, center_goals.double(), atol=1e-6, rtol=0)


def test_shared_center_uses_3d_distance_and_any_remaining_participant():
    groups = torch.tensor([[0, 0, 2, 2]])
    centers = torch.zeros(1, 4, 3)
    goals = centers.clone()
    valid = torch.tensor([[False, True, False, False]])
    assert shared_center_success(centers, goals, groups, valid).tolist() == [[True, True, False, False]]
    goals[:, :2, :2] = .15
    assert not shared_center_success(centers, goals, groups, valid).any()
    goals.zero_(); goals[..., 2] = .21
    assert not shared_center_success(centers, goals, groups, valid).any()
    groups[:] = torch.arange(4)
    goals.zero_(); valid.fill_(True)
    assert not shared_center_success(centers, goals, groups, valid).any()


def test_groups_are_counted_once_and_ever_can_differ_from_final():
    groups = torch.tensor([[1, 1, 2, 2], [3, 3, 3, 3]])
    ever = torch.tensor([[True, True, False, False], [True, True, True, True]])
    final = torch.tensor([[False, False, False, False], [True, True, True, True]])
    scores = shared_center_scores(groups, ever, final)
    assert scores['2'] == dict(groups=2, ever_success=1, final_success=0, ever_rate=.5, final_rate=0.)
    assert scores['3'] == dict(groups=0, ever_success=0, final_success=0, ever_rate=None, final_rate=None)
    assert scores['4']['ever_success'] == scores['4']['final_success'] == 1


def test_summary_merges_shared_center_counts_without_changing_task_scores(tmp_path):
    from evaluate import summarize
    for seed, (count, ever, final) in enumerate(((3, 2, 1), (1, 1, 0))):
        directory = tmp_path/str(seed)
        directory.mkdir()
        row = dict(groups=count, ever_success=ever, final_success=final,
                   ever_rate=ever/count, final_rate=final/count)
        empty = dict(groups=0, ever_success=0, final_success=0, ever_rate=None, final_rate=None)
        result = dict(seed=seed, protocol='general', scores={'point': {}, 'region': {}},
                      shared_center_scores={'2': row, '3': empty, '4': empty})
        (directory/'result.json').write_text(json.dumps(result))
    result = summarize(tmp_path)
    assert result['shared_center_scores']['2'] == dict(groups=4, ever_success=3,
        final_success=1, ever_rate=.75, final_rate=.25)
    assert result['scores'] == {'point': {}, 'region': {}}
