"""Fixed 1.30m/20kg PUSH with the original no-tilt contact/AT objective."""
from copy import deepcopy
from pathlib import Path
import json
from isaacgym import gymapi
import pytest
import yaml
from utils.open_push_spec import validate_env
from utils.open_push_sizes import size_values,density
from utils.open_push_rsi import bank_contract
from utils.push_size_subset import project_size_subset
from utils.push_only import training_contract

CFG=Path('tokenhsi/data/cfg/multi_agent')
BASE='open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors'
NAME='push_fixed130_20kg_no_tilt'

def env(name=NAME):
    e=yaml.safe_load((CFG/(name+'.yaml')).read_text())['env']
    e['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
    return e

def test_fixed_domain_and_exactly_20kg():
    e=env();validate_env(e)
    assert 'pushSizeRandomization' not in e
    assert size_values(e)==[1.3]
    assert e['pushBox']['mass']==20.
    assert density(e)*1.3**3==pytest.approx(20.)
    assert e['numEnvs']==2048 and e['pushOnly']

def test_only_size_domain_and_bank_changed():
    a=env(BASE);b=env()
    a.pop('pushSizeRandomization')
    for e in (a,b):e['openPushRSI'].pop('bank_file')
    assert a==b
    assert not {'max_tilt','tilt_fade_start','tilt_fade_end'} & set(b['relationReward']['contact_at'])

def test_bank_is_same_candidates_with_fixed_domain_provenance():
    a=env(BASE);b=env();data=project_size_subset(a,b)
    assert data==json.loads(Path(b['openPushRSI']['bank_file']).read_text())
    source=json.loads(Path(a['openPushRSI']['bank_file']).read_text())
    assert data['candidates']==source['candidates'] and len(data['candidates']['push'])==943
    assert data['contract']==bank_contract(b)
    assert data['contract']['push_size_randomization'] is None
    assert data['size_subset_provenance']['retained_sizes_m']==[1.3]

@pytest.mark.parametrize('part',['mass','friction','size'])
def test_reject_bank_reuse_if_physics_changes(part):
    a=env(BASE);b=env()
    b['pushBox'][part]=[1.2]*3 if part=='size' else .8
    with pytest.raises(ValueError):project_size_subset(a,b)

def test_checkpoint_contract_separates_random_and_fixed_domain():
    path='tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml'
    a=training_contract(env(BASE),path);b=training_contract(env(),path)
    assert a!=b and b['push_size_randomization'] is None
    assert a['reward']==b['reward'] and a['motion_hashes']==b['motion_hashes']
