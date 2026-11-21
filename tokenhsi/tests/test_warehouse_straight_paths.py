from copy import deepcopy

import pytest
import torch

from utils.experiments import native_config
from utils.warehouse_coop import box_layout, sample_layout, sample_roles_graph
from utils.warehouse_paths import clear_segments, enabled, wall_rectangles


NAME = 'stage2_sensor_900_warehouse_fixedroles_v4_beforepairs'


def test_option_is_evaluation_only_and_does_not_change_config():
    cfg = native_config(NAME)['env']['sensorScene']['warehouse']
    original = deepcopy(cfg)
    assert enabled('1', True, cfg)
    assert not enabled('0', False, cfg)
    assert cfg == original
    for value, evaluation, warehouse in (('1', False, cfg), ('1', True, None),
                                          ('1', True, {}), ('yes', True, cfg)):
        with pytest.raises(ValueError, match='WAREHOUSE_STRAIGHT_PATHS'):
            enabled(value, evaluation, warehouse)


def test_wall_thickness_jambs_endpoints_and_zero_length_segments():
    cfg = native_config(NAME)['env']['sensorScene']['warehouse']
    centers, half = wall_rectangles(cfg, torch.tensor([3.72]))
    start = torch.tensor([[[-2., 0.], [-2., 3.], [-2., 0.], [-.2, 1.4],
                           [-.2, 3.], [-2., 0.], [7., 0.], [2., 3.8]]])
    end = torch.tensor([[[2., 0.], [2., 3.], [-2., 3.], [-.2, 2.],
                         [-.2, 3.], [-2., 0.], [9., 0.], [2., 4.5]]])
    assert clear_segments(start, end, centers, half).tolist() == [
        [True, False, True, False, False, True, False, False]]
    near_jamb = torch.tensor([[[-2., 1.6]]])
    inside = torch.tensor([[[2., 1.6]]])
    assert clear_segments(near_jamb, inside, centers, half).item()
    assert not clear_segments(near_jamb+.26, inside+.26, centers, half).item()
    # The inner face is clear, but the segment clips the outer doorway corner.
    assert not clear_segments(torch.tensor([[[-1., 2.5]]]), torch.tensor([[[1., .5]]]),
                              centers, half).item()


def test_sampling_preserves_constraints_and_both_routes_clear_physical_walls():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    env = native_config(NAME)['env']
    cfg = env['sensorScene']['warehouse']
    n = 96
    sizes = torch.tensor([.4, .4, .35])+torch.rand(n, 4, 3)*torch.tensor([.2, .2, .15])
    supports = torch.tensor([.7, .7, .4]).expand(n, 4, 3)
    counts = torch.full((n,), 2)
    directions, modes = counts.clone(), torch.arange(n)%3
    _, pre, tasks, roles = sample_roles_graph(sizes, env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'], counts, 1.)
    near = (pre >= 0) & (torch.rand(n, 4) < .5)
    inputs = (tasks, pre, roles, sizes, supports, near, counts, directions, modes, cfg, 16)
    state = torch.get_rng_state()
    default = sample_layout(*inputs)
    torch.set_rng_state(state)
    disabled = sample_layout(*inputs, straight_paths=False)
    for key in default:
        torch.testing.assert_close(default[key], disabled[key], rtol=0, atol=0)
    result = sample_layout(*inputs[:-1], 128, straight_paths=True)
    good = result['valid']
    assert good.float().mean() > .8
    r, follower = (pre[good] >= 0).nonzero(as_tuple=True)
    leader = pre[good][r, follower]
    root = result['root'][good][r, leader]
    source = result['source'][good][r, leader]
    goal = result['goal'][good][r, leader]
    approach = (root-source).norm(dim=-1)
    transport = (source-goal).norm(dim=-1)
    assert ((approach >= 1.) & (approach <= 5.)).all()
    assert ((transport >= 1.) & (transport <= 5.)).all()
    assert ((root[:, 0] >= -3.5) & (root[:, 0] <= -.9) & (root[:, 1].abs() <= 3.6)).all()
    geometry = box_layout(counts[good], directions[good], modes[good], cfg)
    # Independent dense sweep through the actual five wall rectangles.
    centers, half = wall_rectangles(cfg, geometry['opening'])
    fraction = torch.linspace(0., 1., 1001)[None, :, None]
    for start, end in ((root, source), (source, goal)):
        points = start[:, None]+fraction*(end-start)[:, None]
        distance = ((points[:, :, None]-centers[:, None]).abs()-half[:, None]).clamp_min(0).norm(dim=-1)
        assert (distance > 0).all()
    push = tasks[good] == 5
    push_source = result['source'][good][push].reshape(-1, 2, 2)
    torch.testing.assert_close(push_source[:, 0], push_source[:, 1])
    fixed = modes[good] != 1
    torch.testing.assert_close(push_source[fixed, 0], geometry['source'][fixed])
    partial = modes[good] == 1
    assert ((push_source[partial, 0, 0] > geometry['door'][partial, 0])
            & (push_source[partial, 0, 0] < geometry['landing'][partial, 0])).all()
