from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_wait_reward import wait_clear_penalties, segment_distance
from utils.experiments import native_config, validate_experiment
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.unified_training import validate_unified_env
from utils.edge_ontop_spec import permute_graph, select_graph
from test_before_atprogress import fixture

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute'
NAME = BASE + '_wait_clear'
ENV = native_config(NAME)['env']


def setup(family='sit'):
    g, pos, boxes, sizes, goals, rows, edges, goal_ids, support = fixture(family)
    roots = torch.zeros(4, 4, 13); roots[..., :3] = pos; roots[..., 6] = 1
    pred = g.prereq_mask[rows, edges].long().argmax(-1)
    leader = g.edge_owner[rows, pred]
    follower = g.edge_owner[rows, edges]
    roots[rows, leader, :2] = torch.tensor([-4., 0.])
    boxes[rows, support, :2] = torch.tensor([-3., 0.])
    roots[rows, follower, :2] = torch.tensor([0., 1.5])
    own = (g.edge_src[rows, edges]-4).clamp(0, 7)
    if family == 'stack': boxes[rows, own, :2] = roots[rows, follower, :2]
    raw = torch.zeros(4, len(g.ids), dtype=torch.bool)
    args = [roots, boxes, sizes, goals, g, raw, ENV['relationReward'], ENV['before']['wait_clear']]
    return args, (rows, edges, pred, leader, follower, own)


def run(args): return wait_clear_penalties(*args)


def test_config_is_opt_in_and_checkpoint_contract_separate():
    base, new = native_config(BASE), native_config(NAME)
    validate_unified_env(new['env'])
    validate_experiment(NAME, new['env'], native_config(NAME, True))
    env = deepcopy(new['env']); env['before'].pop('wait_clear')
    assert env == base['env'] and new['sim'] == base['sim']
    assert new['experiment']['transfer'] == base['experiment']['transfer']
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint({'before_contract': checkpoint_contract(base['env'])},new['env'])


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, None])
def test_invalid_config(value):
    env = deepcopy(ENV);env['before']['wait_clear']['motion_weight'] = value
    with pytest.raises(ValueError, match='wait_clear'): validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit','climb','stack'])
def test_stop_is_better_than_circling_and_current_success_releases(family):
    args, (rows, edges, pred, leader, follower, own) = setup(family)
    radius = run(args)['waiting_radius'][rows, follower]
    args[0][rows, follower, :2] = torch.stack((torch.zeros_like(radius),radius),-1)
    if family == 'stack':args[1][rows, own, :2] = args[0][rows, follower, :2]
    stopped = run(args)
    assert (stopped['blocking'] == 0).all()
    assert (stopped['waiting_weight'][rows, follower] > .99).all()
    assert (stopped['waiting_motion'] == 0).all()
    # Tangent to the goal-centered circle, with unchanged radius.
    args[0][rows, follower, 7] = 1.
    circling = run(args)
    assert (circling['waiting_motion'][rows,follower] < 0).all()
    assert torch.equal(circling['blocking'], stopped['blocking'])
    args[5][rows,pred] = True
    complete = run(args)
    assert (complete['blocking'] == 0).all() and (complete['waiting_motion'] == 0).all()
    args[5][rows,pred] = False
    assert torch.equal(run(args)['waiting_motion'], circling['waiting_motion'])
    assert (circling['waiting_motion'][rows,leader] == 0).all()


def test_blocked_follower_can_yield_and_far_approach_is_not_slowed():
    args,(rows,edges,pred,leader,follower,own)=setup()
    args[0][rows,follower,:2] = torch.tensor([-2.,0.])
    args[0][rows,follower,7:9] = torch.tensor([0.,1.])
    blocked=run(args)
    assert (blocked['blocking'][rows,follower] < 0).all()
    assert (blocked['waiting_motion'] == 0).all()
    args[0][rows,follower,:2] = torch.tensor([-2.,.4])
    assert (run(args)['blocking'][rows,follower] > blocked['blocking'][rows,follower]).all()
    args[0][rows,follower,:2] = torch.tensor([0.,10.])
    far=run(args)
    assert (far['blocking']==0).all() and (far['waiting_motion']==0).all()


def test_payload_blocks_even_when_human_is_clear_and_large_boxes_have_waiting_space():
    args,(rows,edges,pred,leader,follower,own)=setup('stack')
    args[0][rows,follower,:2] = torch.tensor([0.,5.])
    args[1][rows,own,:2] = torch.tensor([-2.,0.])
    assert (run(args)['blocking'][rows,follower]<0).all()
    args[2].fill_(2.)
    radius=run(args)['waiting_radius'][rows,follower]
    xy=torch.stack((torch.zeros_like(radius),radius),-1)
    args[0][rows,follower,:2]=xy;args[1][rows,own,:2]=xy
    safe=run(args)
    assert (safe['blocking']==0).all()
    assert (safe['waiting_weight'][rows,follower]>.99).all()
    args[1][rows,own,7]=1.
    assert (run(args)['waiting_motion'][rows,follower]<0).all()


def test_finite_segment_and_zero_length():
    point=torch.tensor([[2.,0.],[0.,2.],[1.,1.]])
    start=torch.zeros(3,2);end=torch.tensor([[1.,0.],[0.,0.],[0.,0.]])
    torch.testing.assert_close(segment_distance(point,start,end),torch.tensor([1.,2.,2.**.5]))


def test_no_input_mutation_partial_batch_and_edge_permutation():
    args,_=setup('stack');original=[x.clone() for x in args[:4]]
    expected=run(args)
    for x,y in zip(args[:4],original): assert torch.equal(x,y)
    ids=torch.tensor([3,1]);part=[x[ids] for x in args[:4]]+[select_graph(args[4],ids),args[5][ids]]+args[6:]
    for key,value in run(part).items():torch.testing.assert_close(value,expected[key][ids])
    order=torch.rand(4,len(args[4].ids)).argsort(-1)
    shuffled=args[:4]+[permute_graph(args[4],order),args[5].gather(1,order)]+args[6:]
    for key,value in run(shuffled).items():torch.testing.assert_close(value,expected[key])


def test_no_dependencies_is_noop():
    args,_=setup();args[4].prereq_mask.zero_()
    for value in run(args).values():assert (value==0).all()
