import pytest
import torch

from utils import before_push_layout as layout
from utils.experiments import native_config
from utils.jointpush_scene import goal_angles_valid
from test_before_push_source5 import blocked_four_push_scene


SPEC = native_config('stage2_before_push_near50_pair50_source5')['env']['jointPush']['scene']


def test_first_valid_candidate_and_successful_rows_survive_later_batches(monkeypatch):
    boxes = torch.zeros(3, 2, 13)
    sizes = torch.ones(3, 2, 3)
    roots = torch.zeros(3, 1, 3)
    roots[:, 0, 0] = torch.arange(3)
    push = torch.tensor([[False], [True], [True]])
    primary = torch.zeros(3, 1, dtype=torch.long)
    active = torch.ones(3, 2, dtype=torch.bool)
    calls = []
    def candidates(b, s, r, p, o, a, spec, max_attempts):
        ids = r[:, 0, 0].long()
        assert max_attempts == 1
        calls.append(ids.tolist())
        local = torch.arange(len(ids)) % (16 if len(calls) == 1 else 3)
        good = ((ids == 1) & ((local == 3) | (local == 5))) if len(calls) == 1 else (local == 1)
        result = b.clone()
        result[:, 0, 0] = 100*ids+20*len(calls)+local
        goal = result[:, :1, :3].clone()
        raise layout.PushLayoutError(result, goal, (~good).nonzero().flatten())
    monkeypatch.setattr(layout, 'sample_layout', candidates)
    result, goals = layout.sample_batched_layout(boxes, sizes, roots, push, primary, active, SPEC,
        max_attempts=19)
    assert calls == [[1]*16+[2]*16, [2]*3]
    assert result[:, 0, 0].tolist() == [0., 123., 241.]
    assert torch.equal(goals[..., 0], result[:, :1, 0])
    assert not boxes.any()


def test_exact_candidate_budget_and_unresolved_rows(monkeypatch):
    args = blocked_four_push_scene()
    counts = []
    def reject(boxes, sizes, roots, push, primary, active, spec, max_attempts):
        counts.append(len(boxes))
        raise layout.PushLayoutError(boxes.clone(), boxes[:, :4, :3].clone(), torch.arange(len(boxes)))
    monkeypatch.setattr(layout, 'sample_layout', reject)
    with pytest.raises(layout.PushLayoutError) as caught:
        layout.sample_batched_layout(*args, SPEC, max_attempts=19, candidate_batch=7)
    assert counts == [7, 7, 5]
    assert caught.value.pending.tolist() == [0]
    assert torch.equal(caught.value.boxes, args[0])


def test_no_push_preserves_inputs_goals_and_rng():
    boxes, sizes, roots, push, primary, active = blocked_four_push_scene()
    push.zero_()
    state = torch.get_rng_state()
    result, goals = layout.sample_batched_layout(boxes, sizes, roots, push, primary, active, SPEC)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(result, boxes)
    assert torch.equal(goals, boxes[:, :4, :3])


def test_real_partial_failure_retains_accepted_and_nonpush_rows():
    torch.manual_seed(72)
    boxes = torch.zeros(3, 2, 13)
    boxes[..., 6] = 1.
    boxes[:, 0, :2] = 1000.
    sizes = torch.full((3, 2, 3), .4)
    sizes[1, 1] = 100.
    roots = torch.zeros(3, 1, 3)
    push = torch.tensor([[True], [True], [False]])
    primary = torch.zeros(3, 1, dtype=torch.long)
    active = torch.tensor([[True, False], [True, True], [True, False]])
    with pytest.raises(layout.PushLayoutError) as caught:
        layout.sample_batched_layout(boxes, sizes, roots, push, primary, active, SPEC, max_attempts=19)
    error = caught.value
    assert error.pending.tolist() == [1]
    assert layout.paths_valid(error.boxes[:1], sizes[:1], error.goals[:1], roots[:1],
        push[:1], primary[:1], active[:1], SPEC).all()
    assert torch.equal(error.boxes[2], boxes[2])
    assert torch.equal(error.boxes[:, 1], boxes[:, 1])


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_four_push_fallback_preserves_all_existing_geometry(seed):
    torch.set_num_threads(1)
    args = blocked_four_push_scene()
    boxes, sizes, roots, push, primary, active = args
    saved = [v.clone() for v in args]
    torch.manual_seed(seed)
    try:
        result, goals = layout.sample_batched_layout(*args, SPEC)
    except layout.PushLayoutError as error:
        result, goals = error.boxes, error.goals
        pending = error.pending
        extra, target = layout.sample_sequential_layout(result[pending], sizes[pending], roots[pending],
            push[pending], primary[pending], active[pending], SPEC)
        result[pending], goals[pending] = extra, target
    assert layout.paths_valid(result, sizes, goals, roots, push, primary, active, SPEC).all()
    assert goal_angles_valid(result[:, :4], goals, push, SPEC).all()
    assert torch.equal(result[:, 4:], boxes[:, 4:])
    for current, original in zip(args, saved):
        assert torch.equal(current, original)
    for distance, key in (((result[:, :4, :2]-roots[..., :2]).norm(dim=-1), 'source_distance'),
                          ((goals[..., :2]-result[:, :4, :2]).norm(dim=-1), 'goal_distance')):
        lo, hi = SPEC[key]
        assert ((distance >= lo-2e-5) & (distance <= hi+2e-5)).all()
