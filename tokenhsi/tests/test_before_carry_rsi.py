from copy import deepcopy
from pathlib import Path
import torch
import yaml
import pytest

from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint
from utils.before_carry_rsi import select_carry
from utils.before_layout import transform

ROOT = Path(__file__).resolve().parents[2]
NAME = 'stage2_before_ground_mix226_atprogress_carry50_1200'
BASE = 'stage2_before_ground_mix226_atprogress_1200'


def test_config_changes_only_carry_initialization_and_contract():
    old = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{BASE}.yaml').read_text())
    new = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{NAME}.yaml').read_text())
    validate_env(new['env'])
    env = deepcopy(new['env'])
    assert env['before'].pop('follower_carry_rsi') == dict(probability=.5, skill='carryWith')
    assert env == old['env'] and new['sim'] == old['sim']
    assert new['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    check_checkpoint({'before_contract': checkpoint_contract(new['env'])}, new['env'])
    for source, target in ((old['env'], new['env']), (new['env'], old['env'])):
        with pytest.raises(ValueError, match='BEFORE checkpoint'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)
    a,b = [yaml.safe_load((ROOT/f'tokenhsi/data/cfg/train/rlg/{n}.yaml').read_text()) for n in (BASE,NAME)]
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b


def test_selection_is_half_of_dependent_ontop_only_and_counts_unavailable():
    torch.manual_seed(26)
    pre = torch.tensor([[-1,0,0,0,-1]]).expand(50000,-1)
    template = torch.tensor([[3,1,2,4,4]]).expand_as(pre)
    available = torch.ones_like(pre,dtype=torch.bool)
    eligible,requested,selected = select_carry(pre,template,.5,available)
    assert eligible.sum() == 50000
    assert not selected[:,[0,1,2,4]].any()
    assert .49 < selected.sum().item()/eligible.sum().item() < .51
    assert torch.equal(requested,selected)
    available[:,3] = False
    _,requested,selected = select_carry(pre,template,1.,available)
    assert requested.sum() == 50000 and not selected.any()
    state = torch.random.get_rng_state().clone()
    _,requested,selected = select_carry(pre,template,0.,~available)
    assert not requested.any() and not selected.any()
    assert torch.equal(state, torch.random.get_rng_state())


def test_rigid_transform_preserves_carried_object_and_body_distances():
    states = torch.randn(20,5,13)
    states[...,3:7] /= states[...,3:7].norm(dim=-1,keepdim=True)
    angle = torch.rand(20)*6.28
    q = torch.zeros(20,4);q[:,2]=torch.sin(angle/2);q[:,3]=torch.cos(angle/2)
    shifted = transform(states,q,torch.randn(20,3))
    torch.testing.assert_close(torch.cdist(states[...,:3],states[...,:3]),
                               torch.cdist(shifted[...,:3],shifted[...,:3]),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(states[...,2]-states[:,0:1,2],shifted[...,2]-shifted[:,0:1,2])
    torch.testing.assert_close(states[...,7:10].norm(dim=-1),shifted[...,7:10].norm(dim=-1))


def test_carry_selection_is_independent_of_twenty_twenty_sixty_distance_modes():
    from utils.before_sampling import sample_distance_modes, MIX226_DISTANCE_MIXTURE
    torch.manual_seed(51)
    pre = torch.tensor([[-1, 0]]).expand(100000, -1)
    templates = torch.tensor([[3, 4]]).expand_as(pre)
    modes = sample_distance_modes(pre, MIX226_DISTANCE_MIXTURE)
    _, _, carry = select_carry(pre, templates, .5, torch.ones_like(pre, dtype=torch.bool))
    for mode, fraction in enumerate((.2, .2, .6)):
        rows = modes[:, 1] == mode
        assert abs(rows.float().mean().item()-fraction) < .01
        assert .49 < carry[rows, 1].float().mean().item() < .51
