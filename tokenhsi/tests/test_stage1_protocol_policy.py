from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'eval'))
from stage1_protocol_policy import stage1_scene
from utils.before_object_alias import alias_layout
from utils.stage2_spec import compose_graph
from utils.task_role_spec import task_graph_packet


def observation(people, follower=None, coupled=False):
    objects = 2*people
    tasks = torch.ones(1, people, dtype=torch.long)
    groups = torch.arange(people)[None]
    parent = torch.full_like(groups, -1)
    if follower is not None:
        tasks[0, :2] = torch.tensor([3, follower])
        groups[0, 1] = 0
        parent[0, 1] = 0
    if coupled:
        tasks[0, :2] = 5
        groups[0, :2] = 0
    packet = task_graph_packet(compose_graph(tasks, groups, parent, objects, shuffle=False), 1)
    intrinsic = torch.randn(1, people*223+objects*30+people)
    poses = torch.arange((2*people+objects)*7).reshape(1, -1).float()/10
    return torch.cat((intrinsic, poses, packet, groups.float()), -1), packet


@pytest.mark.parametrize('people', (2, 4, 8, 16))
@pytest.mark.parametrize('coupled', (False, True))
def test_independent_and_push_keep_native_scene(people, coupled):
    raw, _ = observation(people, coupled=coupled)
    before = raw.clone()
    converted = stage1_scene(raw, people, 2*people)
    assert torch.equal(converted, raw[:, :-people])
    assert torch.equal(raw, before)


@pytest.mark.parametrize('people', (2, 4, 8, 16))
@pytest.mark.parametrize('follower', (1, 2, 4))
def test_enabling_only_redirects_follower_to_goal_alias(people, follower):
    raw, packet = observation(people, follower=follower)
    before = raw.clone()
    objects = 2*people
    converted = stage1_scene(raw, people, objects)
    dependent, target, aliases, _ = alias_layout(packet, people, objects)
    records = converted[:, -5*people:].reshape(1, people, 5)
    assert records[0, 1, 4] == aliases[0, 1]
    assert torch.equal(records[0, 0], packet.reshape(1, people, 5)[0, 0])
    start = people*223+objects*30+people
    old_poses = raw[:, start:start+(2*people+objects)*7].reshape(1, -1, 7)
    new_poses = converted[:, start:start+(2*people+objects)*7].reshape(1, -1, 7)
    goal = packet.reshape(1, people, 5)[0, 0, 4].long()
    assert torch.equal(new_poses[0, aliases[0, 1], :3], old_poses[0, goal, :3])
    assert torch.equal(new_poses[0, people, :], old_poses[0, people, :])
    assert torch.equal(raw, before)
    assert torch.equal(converted[:, :start], raw[:, :start])
