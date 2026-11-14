from types import SimpleNamespace
from copy import deepcopy
import math
import json
import pytest
import torch
from utils.evaluation_protocol import configure, spatial_success, single_at_sizes, AT_SIZES
from utils.evaluation_protocol import interaction_group_scores, summarize_interaction_groups, INTERACTION_METRICS


def test_interaction_groups_exclude_enabling_leader_and_count_coupled_groups_once():
    valid = torch.ones(1, 10, dtype=torch.bool)
    parent = torch.tensor([[-1, 0, -1, -1, -1, -1, -1, -1, -1, -1]])
    groups = torch.tensor([[0, 1, 2, 2, 2, 2, 6, 6, 8, 9]])
    ever = torch.zeros(1, 10, 2, dtype=torch.bool)
    ever[0, 0] = True
    ever[0, 1, 0] = True
    ever[0, 8] = True
    ever[0, 9, 1] = True
    centers = torch.tensor([[False, False, True, True, True, True, False, False, False, False]])
    simultaneous = torch.zeros_like(ever)
    simultaneous[0, 1, 0] = True
    scores = interaction_group_scores(valid, parent, groups, ever, centers, simultaneous)
    for label, rates in [('point', [.5, 1., .5, 2/3]), ('region', [1., 0., .5, .5])]:
        assert [scores[label][name]['rate'] for name in INTERACTION_METRICS] == rates
        assert [scores[label][name]['groups'] for name in INTERACTION_METRICS[:3]] == [2, 1, 2]


def test_missing_interaction_type_does_not_become_zero_or_partial_macro():
    scores = interaction_group_scores(torch.tensor([[True, False]]),
        torch.tensor([[-1, -1]]), torch.tensor([[0, 1]]),
        torch.ones(1, 2, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, 2, dtype=torch.bool))
    for label in ('point', 'region'):
        assert scores[label][INTERACTION_METRICS[0]] == dict(groups=1, success=1, rate=1.)
        assert scores[label][INTERACTION_METRICS[1]]['rate'] is None
        assert scores[label][INTERACTION_METRICS[-1]]['rate'] is None


def test_enabling_requires_success_in_the_same_frame():
    parent = torch.tensor([[-1, 0, -1, 2]])
    current = torch.zeros(2, 1, 4, 2, dtype=torch.bool)
    current[0, 0, 0] = True
    current[1, 0, 1:] = True
    ever = current.any(0)
    simultaneous = (current & current[:, torch.arange(1)[:, None], parent.clamp_min(0)]).any(0)
    valid = torch.ones(1, 4, dtype=torch.bool)
    groups = torch.arange(4)[None]
    centers = torch.zeros_like(valid)
    scores = interaction_group_scores(valid, parent, groups, ever, centers, simultaneous)
    legacy = interaction_group_scores(valid, parent, groups, ever, centers, simultaneous,
                                     enabling_success='both_ever')
    for label in ('point', 'region'):
        assert scores[label][INTERACTION_METRICS[1]] == dict(groups=2, success=1, rate=.5)
        assert legacy[label][INTERACTION_METRICS[1]] == dict(groups=2, success=2, rate=1.)


def test_interaction_summary_averages_seed_rates_before_pooling_counts():
    runs = []
    for rates, counts in [([1., 0., 0.], [1, 1, 1]), ([0., 1., 1.], [9, 1, 1])]:
        scores = {name: dict(groups=count, success=int(rate*count), rate=rate)
                  for name, rate, count in zip(INTERACTION_METRICS, rates, counts)}
        scores[INTERACTION_METRICS[-1]] = dict(rate=sum(rates)/3)
        runs.append(dict(point=scores, region=scores))
    result = summarize_interaction_groups(runs)
    assert result['point'][INTERACTION_METRICS[0]]['mean'] == .5
    assert result['point'][INTERACTION_METRICS[0]]['groups'] == 10
    macro = result['point'][INTERACTION_METRICS[-1]]
    assert macro['mean'] == .5
    assert macro['std'] == pytest.approx(math.sqrt(1/18))


def test_summary_rejects_mixed_enabling_success_criteria(tmp_path):
    from evaluate import summarize
    for seed, mode in enumerate((None, 'simultaneous')):
        path = tmp_path / ('general_seed' + str(seed))
        path.mkdir()
        data = dict(interaction_group_scores={})
        if mode:
            data['enabling_success'] = mode
        (path / 'result.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different enabling success criteria'):
        summarize(tmp_path)


def test_task_macros_use_the_requested_seeds(tmp_path):
    from evaluate import summarize
    for seed, rate in ((3, .2), (4, .4), (5, .9)):
        path = tmp_path / ('general_seed' + str(seed))
        path.mkdir()
        tasks = {name: dict(trials=10, success=int(rate * 10), rate=rate, pair_trials=0)
                 for name in ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')}
        data = dict(seed=seed, protocol='general', scores=dict(point=tasks, region=tasks))
        (path / 'result.json').write_text(json.dumps(data))
    summary = summarize(tmp_path)
    assert summary['scores']['region']['macro5']['seeds'] == pytest.approx([.2, .4, .9])
    assert summary['scores']['region']['macro5']['mean'] == pytest.approx(.5)


def inputs(kind):
    # One human, one payload, one support, one target node.
    packet=torch.tensor([[[1,kind,0,1,2 if kind != 2 else 3]]])
    roots=torch.tensor([[[0.,0.,.6]]])
    boxes=torch.tensor([[[0.,0.,.7,0.,0.,0.,1.], [0.,0.,.25,0.,0.,0.,1.]]])
    sizes=torch.tensor([[[.4,.4,.4],[.5,.5,.5]]])
    goals=torch.tensor([[[0.,0.,.7]]])
    return packet,roots,boxes,sizes,goals


def test_task_offsets_and_at_identity():
    for kind, z in [(0,.6),(1,1.44),(2,.7),(3,.7)]:
        p,r,b,s,g=inputs(kind)
        r[...,2]=z
        assert spatial_success(p,r,b,s,g).all()
    p,r,b,s,g=inputs(2)
    for offset in (.0,.19,.21):
        b[:,0,0]=offset
        result=spatial_success(p,r,b,s,g)
        assert result[...,0].equal(result[...,1])
        assert result.all().item() == (offset <= .2)


def test_region_is_axis_aligned_in_support_yaw_and_independent_z():
    p,r,b,s,g=inputs(0)
    s[:,1,:2]=torch.tensor([1.,.25])
    r[...,:3]=torch.tensor([.30,0.,.78])
    result=spatial_success(p,r,b,s,g)
    assert result.tolist()==[[[False,True]]]
    b[:,1,3:7]=torch.tensor([0.,0.,math.sqrt(.5),math.sqrt(.5)])
    assert not spatial_success(p,r,b,s,g).any()


def test_ontop_uses_upright_payload_height():
    p,r,b,s,g=inputs(3)
    s[:,0]=torch.tensor([.8,.4,.2])
    b[:,0,2]=.6
    b[:,0,3:7]=torch.tensor([0.,math.sqrt(.5),0.,math.sqrt(.5)])
    assert spatial_success(p,r,b,s,g).all()


def test_small_support_region_can_be_stricter():
    p,r,b,s,g=inputs(0)
    r[...,0]=.15
    s[:,1,:2]=.25
    assert spatial_success(p,r,b,s,g).tolist()==[[[True,False]]]


def args(**kw):
    values=dict(eval_protocol='general',eval=True,test=True,headless=True,cfg_env='',num_agents=0,
        num_objects=0,episode_length=0,eval_repeats=1,seed=0,before_follower_near=False,
        before_follower_short=False,before_pickup_carry_rsi='loco',task_role_swap=False,
        eval_skills='',eval_skill_probs='',eval_single_task='',task_graph='random_scenario')
    values.update(kw)
    return SimpleNamespace(**values)


def test_general_keeps_saved_counts_and_sizes():
    cfg={'env':dict(numAgents=4,numObjects=8,episodeLength=600,eval={},box={'range':'saved'})}
    configure(args(),cfg,'stage2_before')
    assert cfg['env']['numAgents']==4 and cfg['env']['numObjects']==8
    assert cfg['env']['episodeLength']==1200
    assert cfg['env']['box']=={'range':'saved'}


@pytest.mark.parametrize('seed', [3, 4, 5])
def test_formal_accepts_additional_seeds(seed):
    cfg = {'env': {'eval': {}}}
    configure(args(seed=seed), cfg, 'stage2_900')
    assert cfg['env']['episodeLength'] == 900


def test_formal_rejects_random_seed_sentinel():
    with pytest.raises(ValueError, match='nonnegative seed'):
        configure(args(seed=-1), {'env': {}}, 'stage2_900')


@pytest.mark.parametrize('name', ['stage2_before','stage2_before_rsi433','stage2_before_nonemask_rsi433_near50'])
def test_single_is_stage1_exclusive(name):
    with pytest.raises(ValueError,match='exclusive'):
        configure(args(eval_protocol='stage1_single',eval_single_task='AT'), {'env':{}}, name)


@pytest.mark.parametrize('kw',[{'num_agents':1},{'num_objects':1},{'episode_length':900},
    {'task_graph':'sit'},{'before_pickup_carry_rsi':'mix'},{'eval_skills':'sit'}, {'eval_repeats':3}])
def test_formal_rejects_viewer_overrides(kw):
    with pytest.raises(ValueError):configure(args(**kw),{'env':{}},'stage1')


def test_single_sizes_have_exact_nine_and_all_present():
    torch.manual_seed(0)
    sizes=single_at_sizes(512,'cpu')
    assert sizes.shape==(512,3)
    assert set(map(tuple,sizes.tolist()))==set(map(tuple,torch.tensor(AT_SIZES).tolist()))
    cfg={'env':{'eval':{}}}
    a=args(eval_protocol='stage1_single',eval_single_task='ON_TOP')
    configure(a,cfg,'stage1')
    assert (cfg['env']['numAgents'],cfg['env']['numObjects'],a.task_graph)==(1,2,'holding_ontop')


def test_prerequisite_is_owner_of_at_edge_not_holding_edge():
    from utils.evaluation_protocol import predecessors
    graph=SimpleNamespace(num_agents=2, edge_valid=torch.tensor([[True,True,True]]),
        required_goal=torch.tensor([[False,True,True]]), edge_owner=torch.tensor([[0,0,1]]),
        prereq_mask=torch.tensor([[[False,False,False],[False,False,False],[False,True,False]]]))
    assert predecessors(graph,1,'cpu').tolist()==[[-1,0]]


def test_fallen_actor_loses_all_joint_drives_but_actor_is_kept():
    import numpy as np
    from learning.multi_agent.formal_eval import freeze_fallen
    class Gym:
        def __init__(self):self.writes=[]
        def get_actor_dof_properties(self,env,handle):
            return np.ones(3,dtype=[('stiffness','f4'),('damping','f4'),('driveMode','i4')])
        def set_actor_dof_properties(self,env,handle,props):self.writes.append((env,handle,props))
    gym=Gym()
    task=SimpleNamespace(gym=gym,envs=['a','b'],humanoid_handles_all=[[10,11],[20,21]])
    freeze_fallen(task,torch.tensor([[False,True],[False,False]]))
    assert len(gym.writes)==1
    env,handle,props=gym.writes[0]
    assert (env,handle)==('a',11)
    assert all((props[name]==0).all() for name in props.dtype.names)
