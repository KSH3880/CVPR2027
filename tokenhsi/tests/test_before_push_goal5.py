from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.scene_validation import scene_infeasible
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.stage1_sampling import sample_goal_xy
from utils.unified_training import validate_unified_env

BASE = 'stage2_before_push_near50_pair50_source5'
SOURCES = [BASE, BASE+'_virtual', BASE+'_virtual_asym']


@pytest.mark.parametrize('source', SOURCES)
def test_goal5_changes_only_requested_distances_and_separates_contracts(source):
    old, new = native_config(source), native_config(source+'_goal5')
    validate_unified_env(new['env'])
    validate_experiment(source+'_goal5', new['env'], native_config(source+'_goal5', True))
    restored = deepcopy(new['env'])
    assert restored['before'].pop('goal_distance') == [1., 5.]
    assert restored['jointPush']['scene']['goal_distance'] == [2., 5.]
    restored['jointPush']['scene']['goal_distance'] = [2., 10.]
    assert restored == old['env']
    assert new['sim'] == old['sim']
    assert new['experiment']['transfer']['checkpoint'] == 'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
    train = native_config(source+'_goal5', True)
    train['params']['config']['name'] = native_config(source, True)['params']['config']['name']
    assert train == native_config(source, True)
    for origin, target in [(old, new), (new, old)]:
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(origin['env'])}, target['env'])
        with pytest.raises(ValueError, match='selected experiment'):
            validate_experiment(origin['experiment']['name'], target['env'], native_config(origin['experiment']['name'], True))


@pytest.mark.parametrize('value', [None, [], [1., 6.], [True, 5.], [float('nan'), 5.]])
def test_invalid_goal_ranges_are_rejected(value):
    env = native_config(BASE+'_goal5')['env']
    env['before']['goal_distance'] = value
    with pytest.raises(ValueError, match='goal distances'):
        validate_unified_env(env)


def test_short_push_range_without_goal5_contract_is_rejected():
    env = native_config(BASE)['env']
    env['jointPush']['scene']['goal_distance'] = [2., 5.]
    with pytest.raises(ValueError, match='physics/contact'):
        validate_unified_env(env)


def test_radial_targets_use_payload_center_and_keep_human_exclusion():
    torch.manual_seed(81)
    payload = torch.tensor([100., -100.]).expand(4096, -1)
    human = payload+torch.tensor([2., 0.])
    goal = sample_goal_xy(human, payload, torch.zeros_like(human), [1., 5.])
    distance = (goal-payload).norm(dim=-1)
    assert ((distance >= 1.-2e-5) & (distance <= 5.+2e-5)).all()
    assert ((goal-human).norm(dim=-1) >= 1.).all()
    assert (goal[:, 0] > 90.).all()
    assert (distance < 2.).any() and (distance > 4.).any()


def test_legacy_square_targets_and_rng_are_unchanged():
    torch.manual_seed(123)
    human = torch.tensor([[0., 0.], [1., 2.], [-3., 1.]])
    payload = human+torch.tensor([[2., 0.], [0., 3.], [-1., 1.]])
    result = sample_goal_xy(human, payload, torch.zeros_like(human))
    expected = torch.tensor([[-1.834992527961731, .14906054735183716],
                             [-2.2349634170532227, 1.697011113166809],
                             [-3.8342478275299072, 3.2986974716186523]])
    assert torch.equal(result, expected)
    assert torch.equal(torch.rand(8), torch.tensor([.13657987117767334, .10247904062271118,
        .18405646085739136, .7264467477798462, .3152539134025574, .6871066689491272,
        .07563531398773193, .19663816690444946]))


def test_candidate_clearance_does_not_relax_final_collision_rule():
    boxes = torch.zeros(1, 2, 13)
    boxes[..., 6] = 1.
    boxes[0, 1, 0] = 2**.5+.0505
    sizes = torch.ones(1, 2, 3)
    bodies = torch.zeros(1, 1, 1, 13)
    bodies[..., 0] = 20.
    task = SimpleNamespace(_box_states=boxes, _box_size=sizes,
        _kinematic_humanoid_rigid_body_states=bodies, _tar_pos=torch.zeros(1, 1, 3),
        _agent_box_assignment=torch.zeros(1, 1, dtype=torch.long),
        _templates=torch.full((1, 1), 5), num_physical_objects=2, num_agents=1,
        _active_physical=lambda ids: torch.ones(len(ids), 2, dtype=torch.bool))
    ids = torch.tensor([0])
    assert not scene_infeasible(task, ids).any()
    assert scene_infeasible(task, ids, margin=.001).all()
    assert boxes[0, 1, 0] == 2**.5+.0505
