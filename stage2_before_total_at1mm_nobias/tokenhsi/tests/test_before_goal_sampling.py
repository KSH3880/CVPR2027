from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from utils.before_goal_sampling import DISTANCE_MIXTURE, ontop_distances, route_distances, sample_layout
from utils.before_sampling import sample_distance_modes
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5'
SOURCES = [BASE+'_goal5', BASE+'_virtual_goal5', BASE+'_virtual_asym_goal5']


@pytest.mark.parametrize('source', SOURCES)
def test_new_configs_only_change_layout_and_keep_checkpoint_contract_separate(source):
    old = native_config(source)
    new = native_config(source+'_goalroute')
    validate_unified_env(new['env'])
    validate_experiment(source+'_goalroute', new['env'], native_config(source+'_goalroute', True))
    restored = deepcopy(new['env'])
    assert restored['before']['distance_mixture'] == DISTANCE_MIXTURE
    restored['before']['distance_mixture'] = old['env']['before']['distance_mixture']
    assert restored == old['env']
    assert new['sim'] == old['sim']
    assert new['experiment']['transfer'] == old['experiment']['transfer']
    train = native_config(source+'_goalroute', True)
    assert train['params']['config']['name'] == native_config(source, True)['params']['config']['name']+'GoalRoute'
    train['params']['config']['name'] = native_config(source, True)['params']['config']['name']
    assert train == native_config(source, True)
    for origin, target in [(old, new), (new, old)]:
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(origin['env'])}, target['env'])


@pytest.mark.parametrize('key,value', [('sit_climb_distance', [1.,5.]),
    ('ontop_goal_distance', [1.,10.]), ('ontop_pickup_distance', [0.,5.]),
    ('goal_min_human_distance', 0.), ('mode', 'unknown')])
def test_invalid_layout_contract_is_rejected(key, value):
    env = native_config(SOURCES[0]+'_goalroute')['env']
    env['before']['distance_mixture'][key] = value
    with pytest.raises(ValueError):
        validate_unified_env(env)


def test_goal_route_requires_goal5():
    env = native_config(SOURCES[0]+'_goalroute')['env']
    env['before'].pop('goal_distance')
    with pytest.raises(ValueError):
        validate_unified_env(env)


@pytest.mark.parametrize('limit', [2.002, 3., 6., 8., 10.])
def test_ontop_routes_match_conditioned_uniform_distance_distribution(limit):
    torch.manual_seed(1101)
    first, second = ontop_distances(torch.full((100000,), limit))
    total = first+second
    assert ((first >= 1.) & (first <= 5.) & (second >= 1.) & (second <= 5.)).all()
    assert (total <= limit+2e-6).all()
    cdf = torch.where(total <= 6., (total-2.).square()/32., 1.-(10.-total).square()/32.)
    maximum = (limit-2.)**2/32 if limit <= 6 else 1.-(10.-limit)**2/32
    transformed = cdf/maximum
    assert abs(float(transformed.mean())-.5) < .004
    assert abs(float((transformed <= .25).float().mean())-.25) < .006
    assert abs(float((first-second).mean())) < .025


@pytest.mark.parametrize('template', [1, 2, 4])
@pytest.mark.parametrize('near', [False, True])
def test_goal_centered_geometry_preserves_requested_ranges(template, near):
    torch.manual_seed(45)
    n = 12000
    goal = torch.rand(n, 2)*200.-100.
    leader_route = torch.linspace(2.005, 10., n)
    templates = torch.full((n,), template)
    follower, payload = sample_layout(goal, leader_route, templates, torch.full((n,), near))
    distance = (follower-goal).norm(dim=-1)
    assert (distance >= 1.-2e-5).all()
    assert (distance <= 10.+2e-5).all()
    if template == 4:
        pickup = (payload-follower).norm(dim=-1)
        transport = (payload-goal).norm(dim=-1)
        assert ((pickup >= 1.-2e-5) & (pickup <= 5.+2e-5)).all()
        assert ((transport >= 1.-2e-5) & (transport <= 5.+2e-5)).all()
        distance = pickup+transport
    if near:
        assert (distance < leader_route-.0009).all()
    else:
        assert (distance > leader_route).any()


def test_near50_modes_are_sampled_per_pair_without_forcing_other_half_far():
    torch.manual_seed(501)
    pre = torch.tensor([[-1, 0, -1, 2]]).expand(30000, -1)
    modes = sample_distance_modes(pre, DISTANCE_MIXTURE)
    assert not modes[:, [0, 2]].any()
    assert .49 < float(modes[:, [1, 3]].float().mean()) < .51
    assert torch.equal(sample_distance_modes(pre, DISTANCE_MIXTURE, force_near=True), (pre >= 0).long())


def test_near_metric_includes_leader_transport_and_ontop_pickup():
    leader = torch.tensor([[0., 0.], [0., 0.]])
    support = torch.tensor([[3., 0.], [3., 0.]])
    goal = torch.tensor([[7., 0.], [7., 0.]])
    follower = torch.tensor([[9., 0.], [10., 0.]])
    payload = torch.tensor([[50., 50.], [8., 0.]])
    first, second = route_distances(leader, support, goal, follower, payload, torch.tensor([1, 4]))
    assert torch.equal(first, torch.tensor([7., 7.]))
    assert torch.equal(second, torch.tensor([2., 3.]))


@pytest.mark.parametrize('distance', [None, 2.5, 7., 10.])
@pytest.mark.parametrize('follower_distance', [None, .5, 4., 12.])
@pytest.mark.parametrize('family', [1, 2])
def test_viewer_goal_distance_preserves_leader_physics_and_other_goals(distance, follower_distance, family):
    from env.tasks.multi_agent.before_goal_sampling import apply_layout, invalid
    torch.manual_seed(77)
    roots = torch.rand(2, 4, 13)
    boxes = torch.rand(2, 4, 13)
    goals = torch.rand(2, 4, 3)+torch.tensor([3., 4., 0.])
    mapping = torch.tensor([3, 1, 0, 2])
    goal_mapping = torch.tensor([2, 0, 1, 3])
    task = SimpleNamespace(cfg={'args': SimpleNamespace(before_at_goal_distance=distance,
                                                     before_follower_goal_distance=follower_distance)},
        _before_predecessor=torch.tensor([[2, -1, -1, 1], [2, -1, -1, 1]]),
        _before_near_pairs=torch.zeros(2, 4, dtype=torch.bool),
        _templates=torch.tensor([[family, 3, 3, 4], [family, 3, 3, 4]]),
        _humanoid_root_states=roots, _box_states=boxes, _tar_pos=goals,
        _kinematic_humanoid_rigid_body_states=roots[:, :, None].clone(),
        _scenario_physical_object=lambda env, owner: mapping[owner],
        _scenario_goal=lambda env, owner: goal_mapping[owner])
    old_roots, old_boxes, old_goals = [x.clone() for x in (roots, boxes, goals)]
    baseline = deepcopy(task)
    baseline.cfg['args'].before_follower_goal_distance = None
    rng = torch.get_rng_state()
    apply_layout(baseline, torch.tensor([0]))
    torch.set_rng_state(rng)
    apply_layout(task, torch.tensor([0]))
    source = boxes[0, [0, 1], :2]
    old_delta = old_goals[0, [1, 0], :2]-source
    expected = old_goals[0, [1, 0], :2] if distance is None else source+distance*old_delta/old_delta.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(goals[0, [1, 0], :2], expected)
    assert torch.equal(goals[..., 2], old_goals[..., 2])
    assert torch.equal(goals[0, 2:], old_goals[0, 2:])
    assert torch.equal(goals[1], old_goals[1])
    assert torch.equal(roots[0, [1, 2]], old_roots[0, [1, 2]])
    assert torch.equal(roots[..., 2:], old_roots[..., 2:])
    assert torch.equal(roots[1], old_roots[1])
    assert torch.equal(boxes[0, [0, 1, 3]], old_boxes[0, [0, 1, 3]])
    assert torch.equal(boxes[..., 2:], old_boxes[..., 2:])
    assert torch.equal(boxes[1], old_boxes[1])
    if follower_distance is None:
        assert torch.equal(roots, baseline._humanoid_root_states)
        assert torch.equal(boxes, baseline._box_states)
        assert 1.-1e-5 <= float((roots[0, 0, :2]-goals[0, 1, :2]).norm()) <= 10.+1e-5
        assert 1.-1e-5 <= float((boxes[0, 2, :2]-goals[0, 0, :2]).norm()) <= 5.+1e-5
    else:
        actual = (roots[0, [0, 3], :2]-goals[0, [1, 0], :2]).norm(dim=-1)
        torch.testing.assert_close(actual, torch.full_like(actual, follower_distance), rtol=0, atol=5e-6)
        torch.testing.assert_close(roots[0, 3, :2]-boxes[0, 2, :2],
            baseline._humanoid_root_states[0, 3, :2]-baseline._box_states[0, 2, :2])
        task._before_near_pairs[:] = True
        assert not invalid(task, torch.tensor([0])).any()
    assert 1.-1e-5 <= float((roots[0, 3, :2]-boxes[0, 2, :2]).norm()) <= 5.+1e-5


@pytest.mark.parametrize('follower_distance', [None, 4.])
def test_static_screen_excludes_objects_that_will_be_placed_later(follower_distance):
    from env.tasks.multi_agent.before_goal_sampling import static_infeasible
    boxes = torch.zeros(1, 5, 13)
    boxes[..., 6] = 1.
    boxes[..., 2] = .5
    boxes[0, 0, 0] = 10.
    boxes[0, 3:, 0] = 20.
    bodies = torch.zeros(1, 1, 1, 13)
    bodies[..., 0] = 20.
    bodies[..., 2] = .5
    task = SimpleNamespace(cfg={'args': SimpleNamespace(before_follower_goal_distance=follower_distance)},
        num_agents=1, num_objects=2, num_physical_objects=5,
        _box_states=boxes, _box_size=torch.ones(1, 5, 3), _templates=torch.tensor([[3]]),
        _before_predecessor=torch.tensor([[-1]]), _before_near_pairs=torch.tensor([[False]]),
        _task_token_count=torch.tensor([1]), _logical_box_order=torch.tensor([[0,4]]),
        _kinematic_humanoid_rigid_body_states=bodies, _humanoid_root_states=bodies[:, :, 0],
        _agent_box_assignment=torch.tensor([[0]]), _tar_pos=torch.zeros(1,1,3),
        _scenario_physical_object=lambda env, owner: owner,
        _scenario_goal=lambda env, owner: owner)
    ids = torch.tensor([0])
    assert not static_infeasible(task, ids).any()
    task._box_states[0, 0, 0] = 20.
    assert static_infeasible(task, ids).all()
    task._box_states[0, 0, 0] = 10.
    task._templates.fill_(4)
    task._task_token_count.fill_(2)
    task._logical_box_order[0, 1] = 3
    assert not static_infeasible(task, ids).any()
    task._templates.fill_(5)
    task._logical_box_order[0] = torch.tensor([4, 0])
    task._task_token_count.fill_(1)
    assert not static_infeasible(task, ids).any()
