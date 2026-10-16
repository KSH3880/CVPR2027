import copy
from pathlib import Path

import pytest
import torch
import yaml

from learning.multi_agent.task_coordination import task_relations
from utils.edge_stage1_spec import compile_stage1_graph
from utils.joint_carry_demo import demo_graph_spec, demo_layout
from utils.joint_carry_spec import validate_env
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.task_role_spec import task_graph_packet


def test_two_pairs_are_coupled_and_solo_tasks_have_no_partner():
    graph = compile_stage1_graph(demo_graph_spec(), 6, 8)
    packet = task_graph_packet(graph, 1)
    relations = task_relations(packet, 6)[0]
    expected = torch.eye(6, dtype=torch.long)
    expected[0, 1] = expected[1, 0] = 2
    expected[2, 3] = expected[3, 2] = 2
    assert torch.equal(relations, expected)
    tasks = packet.reshape(6, 5)
    assert tasks[:, 1].tolist() == [2, 2, 2, 2, 2, 3]
    assert tasks[:, 3].tolist() == [6, 6, 7, 7, 8, 9]
    assert tasks[:, 4].tolist() == [14, 14, 16, 16, 18, 10]


def test_paths_are_separate_and_targets_are_forward():
    centers, goals = demo_layout('cpu')
    assert torch.allclose(goals[:, 0] - centers[:, 0], torch.full((4,), 2.5))
    assert torch.equal(goals[:, 1], centers[:, 1])
    assert (centers[1:, 1] - centers[:-1, 1] > 2.5).all()
    for counts in ((2, 4), (6, 7), (8, 8)):
        with pytest.raises(ValueError):
            compile_stage1_graph(demo_graph_spec(), *counts)


def test_four_people_keep_one_joint_team_and_two_independent_tasks():
    graph = compile_stage1_graph(demo_graph_spec(4), 4, 8)
    packet = task_graph_packet(graph, 1)
    expected = torch.eye(4, dtype=torch.long)
    expected[0, 1] = expected[1, 0] = 2
    assert torch.equal(task_relations(packet, 4)[0], expected)
    tasks = packet.reshape(4, 5)
    assert tasks[:, 1].tolist() == [2, 2, 2, 3]
    assert tasks[:, 3].tolist() == [4, 4, 6, 7]
    assert tasks[:, 4].tolist() == [12, 12, 14, 8]


def test_training_contract_stays_two_people_and_cpa_checkpoint_stays_strict():
    path = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding.yaml'
    env = yaml.safe_load(path.read_text())['env']
    validate_env(env)
    demo = copy.deepcopy(env)
    demo.update(numAgents=6, numObjects=8, relationGraph=demo_graph_spec())
    with pytest.raises(ValueError):
        validate_env(demo)
    expected = checkpoint_metadata(env['relationReward'])
    check_checkpoint_metadata({'relation_metadata': expected}, expected)
    wrong = copy.deepcopy(expected)
    wrong['relation_reward_config']['joint_carry']['collision'].pop('release_buffer')
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': wrong}, expected)
