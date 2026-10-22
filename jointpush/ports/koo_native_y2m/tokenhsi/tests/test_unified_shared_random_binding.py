"""Random bindings, mixed AMP and mapped curriculum must agree."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from utils.edge_scenario_spec import sample_graph, agent_object_indices
from utils.unified_training import validate_unified_env
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'data/cfg/multi_agent'
ENV = yaml.safe_load((CFG/'approach_scenario_stage1_unified_shared_random_binding.yaml').read_text())['env']
BASE = yaml.safe_load((CFG/'approach_scenario_stage1_unified_shared_edge_encoder.yaml').read_text())['env']
OLD34 = yaml.safe_load((CFG/'approach_scenario_stage1_self_sum.yaml').read_text())['env']


def test_exact_combination_and_checkpoint_isolation():
    expected = deepcopy(BASE)
    expected['relationGraph']['random_binding'] = {'objects': 'disjoint', 'goals': 'disjoint'}
    expected['relationReward']['stage1_variant'] = ENV['relationReward']['stage1_variant']
    expected['relationReward']['hard_skill_training']['near_start'] = OLD34['relationReward']['hard_skill_training']['near_start']
    expected['box']['build'] = OLD34['box']['build']
    expected['ampTaskConditioning'] = False
    expected['skillDiscProb'] = OLD34['skillDiscProb']
    assert ENV == expected
    validate_relation_config(ENV['relationReward'])
    validate_unified_env(ENV)
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(BASE['relationReward'])}, checkpoint_metadata(ENV['relationReward']))
    for original in (BASE, ENV):
        invalid = deepcopy(original)
        invalid['ampTaskConditioning'] = not original['ampTaskConditioning']
        with pytest.raises(ValueError, match='must be paired'):
            validate_unified_env(invalid)
        invalid = deepcopy(original)
        invalid['relationGraph']['random_binding'] = ENV['relationGraph']['random_binding'] if original is BASE else BASE['relationGraph']['random_binding']
        with pytest.raises(ValueError, match='binding'):
            validate_unified_env(invalid)


@pytest.mark.parametrize('preset', ['random_scenario', 'holding_at', 'holding_ontop'])
def test_all_targets_are_random_and_roles_remain_disjoint(preset):
    n = 256
    graph = sample_graph(n, ENV['relationGraph'], preset=preset, generator=torch.Generator().manual_seed(401))
    env = torch.arange(n).repeat_interleave(2)
    agent = torch.arange(2).repeat(n)
    primary = agent_object_indices(graph, env, agent).reshape(n, 2)
    assert (primary[:, 0] != primary[:, 1]).all()
    for owner in range(2):
        assert primary[:, owner].unique().tolist() == [0, 1, 2, 3]
        at = graph.edge_valid & (graph.edge_relation == 7) & (graph.edge_owner == owner)
        if at.any():
            assert (graph.edge_dst[at] - 6).unique().tolist() == [0, 1]
    for row in range(n):
        top = graph.edge_valid[row] & (graph.edge_relation[row] == 8)
        support = graph.edge_dst[row, top] - 2
        assert len(support.unique()) == len(support)
        assert not (support[:, None] == primary[row][None]).any()
    assert not graph.prereq_mask.any()
    assert graph.term_index.eq(-1).all()


def test_curriculum_and_penalty_follow_binding_on_partial_reset():
    n = 128
    graph = sample_graph(n, ENV['relationGraph'], generator=torch.Generator().manual_seed(402))
    reward = deepcopy(ENV['relationReward'])
    reward['hard_skill_training']['near_start']['probability_start'] = 1.
    task = SimpleNamespace(_relation_cfg=reward, relation_runtime=SimpleNamespace(graph=graph), device='cpu', num_agents=2, num_objects=4, _reset_ref_slots={}, _scenario_no_climb=True)
    task._logical_box_order = torch.tensor([3, 1, 0, 2]).expand(n, -1)
    task._box_states = torch.randn(n, 4, 13)
    task._tar_pos = torch.full((n, 2, 3), 100.)
    task._logical_box_values = lambda values, ids: values[ids[:, None], task._logical_box_order[ids]]
    before = task._box_states.clone()
    ids = torch.arange(0, n, 2)
    SampledOnTopTaskMixin._configure_hard_skill_near_starts(task, ids)
    torch.testing.assert_close(task._box_states[1::2], before[1::2])
    torch.testing.assert_close(task._tar_pos[1::2], torch.full_like(task._tar_pos[1::2], 100.))
    seen = set()
    for row in ids.tolist():
        for edge in graph.edge_valid[row].nonzero().flatten().tolist():
            relation = int(graph.edge_relation[row, edge])
            if relation not in (7, 8):
                continue
            source = int(graph.edge_src[row, edge]) - 2
            physical_source = int(task._logical_box_order[row, source])
            torch.testing.assert_close(task._box_states[row, physical_source], before[row, physical_source])
            if relation == 7:
                dst = int(graph.edge_dst[row, edge]) - 6
                xy = task._tar_pos[row, dst, :2]
                lo, hi = .75, 1.5
                assert task._tar_pos[row, dst, 2] == 100.
            else:
                dst = int(graph.edge_dst[row, edge]) - 2
                physical = int(task._logical_box_order[row, dst])
                xy = task._box_states[row, physical, :2]
                lo, hi = 1.1, 1.8
                torch.testing.assert_close(task._box_states[row, physical, 2:], before[row, physical, 2:])
            assert lo <= (xy - before[row, physical_source, :2]).norm() <= hi
            seen.add(relation)
    assert seen == {7, 8}
    primary = agent_object_indices(graph, ids.repeat_interleave(2), torch.arange(2).repeat(len(ids))).reshape(-1, 2)
    expected = task._box_states[ids[:, None], task._logical_box_order[ids[:, None], primary]]
    torch.testing.assert_close(EdgeContextTaskMixin._reward_box_values(task, task._box_states, ids), expected)
