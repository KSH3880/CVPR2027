"""Fixed region goals preserve translation distance and distinguish box rotation."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.coupled_push_spec import (region_states, region_goal_offsets, compose_graph,
    checkpoint_contract, check_checkpoint)
from utils.experiments import native_config, validate_experiment
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.human_coordination import stage2_human_relations

BASE = 'stage2_coupled_push'
REGION = 'stage2_coupled_push_region_goals'


@pytest.mark.parametrize('members', [2, 3, 4])
def test_fixed_region_targets_preserve_the_exact_original_translation(members):
    torch.manual_seed(812)
    n = 32
    yaw = torch.rand(n, 1, dtype=torch.float64) * 2 * torch.pi
    boxes = torch.zeros(n, members, 13, dtype=torch.float64)
    boxes[..., :3] = torch.randn(n, 1, 3)
    boxes[..., 5] = (yaw / 2).sin()
    boxes[..., 6] = (yaw / 2).cos()
    sizes = torch.tensor([1.8 * members, 1.7, 1.5], dtype=torch.float64).expand(n, members, -1)
    counts = torch.full((n, members), members)
    regions = torch.rand(n, members).argsort(-1)
    cells, _ = region_states(boxes, sizes, counts, regions)
    offsets = region_goal_offsets(boxes, sizes, counts, regions)
    direction = torch.stack((-yaw.sin(), yaw.cos(), torch.zeros_like(yaw)), -1)
    distance = 2 + 8 * torch.rand(n, 1, 1, dtype=torch.float64)
    translation = direction * distance
    center_goal = boxes[..., :3] + translation
    goals = center_goal + offsets
    torch.testing.assert_close(goals - cells[..., :3], translation.expand_as(goals))
    torch.testing.assert_close(offsets.mean(1), torch.zeros(n, 3, dtype=offsets.dtype))
    torch.testing.assert_close(goals.mean(1), center_goal[:, 0])
    # Pure translation reaches every assigned target, even with shuffled regions.
    moved = boxes.clone()
    moved[..., :3] += translation
    moved_cells, _ = region_states(moved, sizes, counts, regions)
    torch.testing.assert_close(moved_cells[..., :3], goals)
    # A turned box at the same center must not make its goals turn with it.
    moved[..., 5] = ((yaw + .4) / 2).sin()
    moved[..., 6] = ((yaw + .4) / 2).cos()
    turned, _ = region_states(moved, sizes, counts, regions)
    assert (turned[..., :3] - goals).norm(dim=-1).amax() > .3


def test_unique_targets_keep_coupled_attention_and_other_tasks_unchanged():
    groups = torch.tensor([[0, 0, 2, 2], [0, 0, 0, 3]])
    templates = torch.tensor([[5, 5, 5, 5], [5, 5, 5, 3]])
    old = compose_graph(templates, groups, False)
    new = compose_graph(templates, groups, False, region_goals=True)
    a, b = [task_graph_packet(g, 2).reshape(2, 4, 5) for g in (old, new)]
    torch.testing.assert_close(b[..., 4], torch.arange(12, 16).float().expand(2, -1))
    assert torch.equal(a[1, 3], b[1, 3])  # Independent carry still uses its own goal.
    assert torch.equal(stage2_human_relations(a, groups), stage2_human_relations(b, groups))
    for key in ('edge_src', 'edge_owner', 'edge_relation', 'edge_valid', 'prereq_mask'):
        assert torch.equal(getattr(old, key), getattr(new, key))


def test_only_goal_contract_changes_and_old_checkpoints_cannot_resume():
    old = native_config(BASE)
    new = native_config(REGION)
    assert new['experiment']['transfer'] == old['experiment']['transfer']
    candidate = deepcopy(new)
    assert candidate['env']['stage2JointPush'].pop('goal_representation') == 'initial_region_offset_v1'
    candidate['experiment']['name'] = BASE
    assert candidate == old
    a, b = [native_config(name, True) for name in (BASE, REGION)]
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    for name, cfg in ((BASE, old), (REGION, new)):
        validate_experiment(name, cfg['env'], native_config(name, True))
        saved = {'coupled_push_contract': checkpoint_contract(cfg['env'])}
        check_checkpoint(saved, cfg['env'])
        other = new if name == BASE else old
        with pytest.raises(ValueError):
            check_checkpoint(saved, other['env'])
    with pytest.raises(ValueError):
        validate_experiment(REGION, old['env'], native_config(REGION, True))


def test_solo_has_zero_offset():
    boxes = torch.zeros(2, 4, 13)
    boxes[..., 6] = 1
    sizes = torch.ones(2, 4, 3)
    offsets = region_goal_offsets(boxes, sizes, torch.ones(2, 4), torch.zeros(2, 4))
    assert not offsets.any()
