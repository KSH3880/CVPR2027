from copy import deepcopy

import pytest
import torch

from utils.experiments import native_config
from utils.coupled_push_spec import asset_bank
from utils.stage2_eval_sizes import prepare_intersection_bank


@pytest.mark.parametrize('preset', ['random_scenario', 'mixed', 'before', 'coupled'])
def test_intersection_sizes_change_only_enable_shared_physical_boxes(preset):
    torch.manual_seed(17)
    env = native_config('stage2_distill_900')['env']
    sizes, masses = asset_bank(512, env, 'cpu')
    original_sizes, original_masses = sizes.clone(), masses.clone()
    _, groups, parent, templates = prepare_intersection_bank(sizes, masses, env, preset)
    rows, follower = (parent >= 0).nonzero(as_tuple=True)
    leader = parent[rows, follower]
    shared = torch.zeros_like(masses, dtype=torch.bool)
    shared[rows, leader] = True
    assert torch.equal(sizes[~shared], original_sizes[~shared])
    assert torch.equal(masses[~shared], original_masses[~shared])
    torch.testing.assert_close(masses[shared], sizes[shared].prod(-1)*100.)
    ranges = {1: [[.25,.60],[.25,.60],[.35,.50]],
              2: [[.40,.60],[.40,.60],[.20,.60]],
              4: [[.40,.60],[.40,.60],[.25,.50]]}
    for kind, bounds in ranges.items():
        selected = templates[rows, follower] == kind
        values = sizes[rows[selected], leader[selected]]
        limits = torch.tensor(bounds)
        assert ((values >= limits[:,0]-1e-6) & (values <= limits[:,1]+1e-6)).all()
        torch.testing.assert_close(values*20, (values*20).round())
    if preset == 'coupled':
        assert not shared.any()
    else:
        assert len(rows) > 0
        assert set(templates[rows, follower].tolist()) == {1,2,4}


def test_intersection_reuses_native_group_and_action_sampling_before_resizing():
    from utils.stage2_spec import sample_episode
    env = deepcopy(native_config('stage2_distill_900')['env'])
    torch.manual_seed(42)
    sizes, masses = asset_bank(512, env, 'cpu')
    state = torch.random.get_rng_state()
    expected = sample_episode(sizes[:,:4], env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'], 'random_scenario',
        env['stage2']['scene_weights'], num_objects=8)
    torch.random.set_rng_state(state)
    actual = prepare_intersection_bank(sizes, masses, env, 'random_scenario')
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
