"""Near50 changes only BEFORE distances in the common five-task Stage2."""
from copy import deepcopy
import torch
import pytest
from utils.experiments import native_config,validate_experiment,check_before_checkpoint
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract
from utils.before_sampling import sample_distance_modes
from utils.before_relative_sampling import DISTANCE_MIXTURE,pair_distances,sample_layout
from utils.before_push_spec import sample_graph

NAME='stage2_before_push_near50'


def test_near50_differs_from_mix226_only_in_distance_and_experiment_names():
    base=native_config('stage2_before_push');near=native_config(NAME)
    reference=native_config('stage2_before_nonemask_rsi433_near50')
    assert near['env']['before']['distance_mixture']==reference['env']['before']['distance_mixture']==DISTANCE_MIXTURE
    projected=deepcopy(near)
    projected['env']['before']['distance_mixture']=base['env']['before']['distance_mixture']
    projected['experiment']['name']=base['experiment']['name']
    assert projected==base
    train=native_config(NAME,True);old=native_config('stage2_before_push',True)
    train['params']['config']['name']=old['params']['config']['name']
    assert train==old
    validate_unified_env(near['env']);validate_experiment(NAME,near['env'],native_config(NAME,True))
    assert 'follower_pickup_carry_rsi' not in near['env']['before']
    assert 'actor_human_bias' not in near['env']['relationGraph']


def test_actual_before_pair_flags_and_loco_layout_follow_near50():
    torch.set_num_threads(1);torch.manual_seed(291)
    env=native_config(NAME)['env'];n=20000
    _,pre,templates=sample_graph(torch.full((n,4,3),.4),env['box']['build']['taskSizeRanges'],env['relationGraph']['template_probabilities'])
    modes=sample_distance_modes(pre,env['before']['distance_mixture'])
    mask=pre>=0
    assert not modes[~mask].any()
    assert .49<modes[mask].float().mean()<.51
    r,f=mask.nonzero(as_tuple=True);task=templates[r,f];count=len(r)
    roots=torch.zeros(count,2);support=torch.ones(count,2)
    box,follower,payload=sample_layout(roots,support,roots,roots,task,modes[r,f].bool(),torch.zeros(count,dtype=torch.bool))
    first,second=pair_distances(roots,box,follower,payload,task)
    assert (second[modes[r,f].bool()]<first[modes[r,f].bool()]).all()
    assert (second[~modes[r,f].bool()]>=first[~modes[r,f].bool()]).any()
    assert not ((templates==5)&(pre>=0)).any()


def test_mix226_and_near50_checkpoints_remain_distinct():
    near=native_config(NAME)['env'];mix=native_config('stage2_before_push')['env']
    for source,target in ((near,mix),(mix,near)):
        with pytest.raises(ValueError):check_before_checkpoint({'before_contract':checkpoint_contract(source)},target)
    with pytest.raises(ValueError):validate_experiment(NAME,mix,native_config(NAME,True))
