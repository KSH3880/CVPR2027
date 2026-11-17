import pytest
import torch

from utils.experiments import native_config, validate_experiment
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint
from utils.stage1_sampling import sample_goal_xy


BASE = 'stage2_coupled_push_individual_goal_solo5'
NAME = BASE+'_carry5'
GROUND = NAME+'_ground'


def test_ground_only_changes_height_and_experiment_identity():
    before = native_config(NAME)
    after = native_config(GROUND)
    after['env']['box']['reset'].update({key: before['env']['box']['reset'][key]
        for key in ('platformMode', 'randomHeight', 'randomHeightProb')})
    after['experiment']['name'] = NAME
    assert after == before
    training = native_config(GROUND, True)
    training['params']['config']['name'] = native_config(NAME, True)['params']['config']['name']
    assert training == native_config(NAME, True)


def test_ground_checkpoint_cannot_be_mixed_with_elevated_carry5():
    env = native_config(GROUND)['env']
    validate_experiment(GROUND, env, native_config(GROUND, True))
    checkpoint = {'coupled_push_contract': checkpoint_contract(env)}
    check_checkpoint(checkpoint, env)
    with pytest.raises(ValueError, match='checkpoint settings differ'):
        check_checkpoint(checkpoint, native_config(NAME)['env'], inference=True)
    with pytest.raises(ValueError, match='selected experiment'):
        validate_experiment(NAME, env, native_config(NAME, True))


@pytest.mark.parametrize('key,value', [('randomHeight', True), ('randomHeightProb', .5)])
def test_ground_rejects_elevated_height_settings(key, value):
    env = native_config(GROUND)['env']
    env['box']['reset'][key] = value
    with pytest.raises(ValueError, match='platform'):
        validate_experiment(GROUND, env, native_config(GROUND, True))


def test_carry5_only_changes_carry_distance_and_experiment_identity():
    before = native_config(BASE)
    after = native_config(NAME)
    assert after['env']['stage2JointPush'].pop('solo_carry_goal_distance') == [1.,5.]
    after['experiment']['name'] = BASE
    assert after == before
    training = native_config(NAME,True)
    training['params']['config']['name'] = native_config(BASE,True)['params']['config']['name']
    assert training == native_config(BASE,True)


def test_carry5_contracts_are_separate():
    env = native_config(NAME)['env']
    validate_experiment(NAME,env,native_config(NAME,True))
    checkpoint = {'coupled_push_contract':checkpoint_contract(env)}
    check_checkpoint(checkpoint,env)
    with pytest.raises(ValueError,match='checkpoint settings differ'):
        check_checkpoint(checkpoint,native_config(BASE)['env'],inference=True)
    with pytest.raises(ValueError,match='selected experiment'):
        validate_experiment(BASE,env,native_config(BASE,True))


def test_radial_goals_use_uniform_distance_and_all_directions():
    torch.manual_seed(42)
    payload = torch.tensor([10.,-8.]).expand(8192,2)
    human = torch.full_like(payload,100.)
    target = sample_goal_xy(human,payload,torch.zeros_like(payload),[1.,5.])
    delta = target-payload
    distance = delta.norm(dim=-1)
    assert ((distance >= 1.-1e-5) & (distance <= 5.+1e-5)).all()
    assert abs(float(distance.mean())-3.) < .04
    for x,y in ((True,True),(True,False),(False,True),(False,False)):
        fraction = ((delta[:,0] > 0) == x) & ((delta[:,1] > 0) == y)
        assert abs(float(fraction.float().mean())-.25) < .02


def test_radial_goals_reject_nearby_humans_without_relaxing_bounds():
    torch.manual_seed(7)
    payload = torch.zeros(8192,2)
    human = payload+torch.tensor([1.5,0.])
    target = sample_goal_xy(human,payload,payload,[1.,5.])
    assert ((target-human).norm(dim=-1) >= 1.-1e-6).all()
    assert ((target-payload).norm(dim=-1) <= 5.+1e-6).all()


@pytest.mark.parametrize('bounds',[[0.,5.],[5.,1.],[1.]])
def test_invalid_carry_bounds_are_rejected(bounds):
    env = native_config(NAME)['env']
    env['stage2JointPush']['solo_carry_goal_distance'] = bounds
    with pytest.raises(ValueError,match='CARRY goal distance'):
        validate_experiment(NAME,env,native_config(NAME,True))


def test_missing_range_keeps_legacy_square_draws():
    torch.manual_seed(42)
    human = torch.full((1024,2),20.)
    payload = torch.full_like(human,30.)
    center = torch.zeros_like(human)
    expected = center+(torch.rand_like(human)*2.-1.)*4.5
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    target = sample_goal_xy(human,payload,center)
    assert torch.equal(target,expected) and torch.equal(torch.get_rng_state(),rng)
