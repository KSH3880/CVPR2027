from dataclasses import fields

import pytest
import torch

from utils.before_box_size import sample_intersection
from utils.before_push_spec import sample_graph
from utils.experiments import native_config


ENV = native_config('stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg')['env']
BOUNDS = {1: ((.25, .60), (.25, .60), (.35, .50)),
          2: ((.40, .60), (.40, .60), (.20, .60)),
          4: ((.40, .60), (.40, .60), (.25, .50))}


@pytest.mark.parametrize('mode', ['family', 'common'])
@pytest.mark.parametrize('people,objects,preset', [
    (4, 8, 'random_scenario'), (8, 12, 'random_scenario'),
    (4, 8, 'before_sit'), (4, 8, 'before_climb'), (4, 8, 'before_stack'),
    (1, 1, 'random_scenario'), (4, 4, 'push')])
def test_intersections_preserve_graph_budget_and_nonshared_boxes(mode, people, objects, preset):
    torch.set_num_threads(1)
    sizes = torch.full((512, people, 3), .4)
    args = (sizes, ENV['box']['build']['taskSizeRanges'],
            ENV['relationGraph']['template_probabilities'], preset, objects, .5)
    torch.manual_seed(91)
    expected, parent, tasks = sample_graph(*args)
    torch.rand(sizes.shape)
    next_random = torch.rand(20)
    torch.manual_seed(91)
    selected, graph, predecessor, templates = sample_intersection(*args, mode)
    assert torch.equal(torch.rand(20), next_random)
    assert torch.equal(predecessor, parent) and torch.equal(templates, tasks)
    for field in fields(graph):
        actual, wanted = getattr(graph, field.name), getattr(expected, field.name)
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, wanted)
    rows, followers = (predecessor >= 0).nonzero(as_tuple=True)
    leaders = predecessor[rows, followers]
    shared = torch.zeros(sizes.shape[:2], dtype=torch.bool)
    shared[rows, leaders] = True
    assert torch.equal(selected[~shared], sizes[~shared])
    assert torch.equal(sizes, torch.full_like(sizes, .4))
    assert (people + (templates == 4).sum(-1) <= objects).all()
    for kind in (1, 2, 4):
        chosen = templates[rows, followers] == kind
        dims = selected[rows[chosen], leaders[chosen]]
        bounds = torch.tensor(((.40, .60), (.40, .60), (.35, .50)) if mode == 'common' else BOUNDS[kind])
        assert ((dims >= bounds[:, 0]-1e-6) & (dims <= bounds[:, 1]+1e-6)).all()
        torch.testing.assert_close(dims * 20, (dims * 20).round(), rtol=0, atol=1e-5)
