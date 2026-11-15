from types import SimpleNamespace
from copy import deepcopy
import math
import pytest
import torch
from utils.evaluation_protocol import configure, spatial_success, single_at_sizes, AT_SIZES


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
