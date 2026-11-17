import isaacgym
import torch

from env.tasks.multi_agent.square_sensor_scene import place_rigid_batched, rotate_xy


def test_batched_placement_keeps_valid_entities_inside_and_obeys_collision_rejection():
    torch.manual_seed(7)
    xy = torch.tensor([[[-.5, 0.], [.5, 0.], [100., 100.]]]).repeat(3, 1, 1)
    xy[2, 1, 0] = 100.
    margin = torch.full((3, 3), .2)
    valid = torch.tensor([[True, True, False]]).repeat(3, 1)
    center = torch.tensor([[3., -2.], [-2., 1.], [0., 0.]])
    half = torch.ones(3, 2)
    calls = []

    def accept(candidate, rows, attempt):
        calls.append(rows)
        return (rows == 0) & (candidate[:, 0, 0] > center[rows, 0])

    angle, offset, placed = place_rigid_batched(xy, margin, valid, center, half, accept, 64)
    assert placed.tolist() == [True, False, False]
    assert len(calls) == 1 and not (calls[0] == 2).any()
    final = rotate_xy(xy, angle[:, None])+offset[:, None]
    assert ((final[0, :2]-center[0]).abs()+margin[0, :2, None] <= half[0]+1e-6).all()
    assert final[0, 0, 0] > center[0, 0]
    torch.testing.assert_close((final[0, 0]-final[0, 1]).norm(), torch.tensor(1.))


def test_impossible_group_never_runs_collision_checks():
    def accept(*args):
        raise RuntimeError('Impossible groups must be rejected before collision checks')

    _, _, placed = place_rigid_batched(torch.tensor([[[0., 0.], [10., 0.]]]),
        torch.full((1, 2), .3), torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2), torch.ones(1, 2), accept, 16)
    assert not placed.any()
