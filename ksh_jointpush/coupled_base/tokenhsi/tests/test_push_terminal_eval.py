from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
from ksh_jointpush.ksh_push_terminal_eval import push_terminal_records, distance_stats


def trace():
    steps,n,m,o=3,2,4,8
    packet=np.zeros((n,m,5),dtype=np.int64)
    packet[:,:,0]=1;packet[:,:,1]=4;packet[:,:,2]=np.arange(m)
    packet[:,:,3]=m+np.arange(m);packet[:,:,4]=m+o+np.arange(m)
    boxes=np.zeros((steps,n,o,7));boxes[...,6]=1
    x=np.arange(m)-1.5
    boxes[:,:,:m,0]=x
    boxes[:,0,:m,1]=np.array([2.,1.,.3])[:,None]
    boxes[:,1,:m,1]=np.array([1.,.4,9.])[:,None]
    goals=np.zeros((n,m,3));goals[:,:,0]=x
    return dict(packet=packet,boxes=boxes,goals=goals,
                groups=np.array([[0,0,0,0],[0,0,2,2]]),
                fall_step=np.array([[1,-1,-1,-1],[1,2,1,2]]),
                ever=np.zeros((n,m,2),dtype=bool))


def test_terminal_distance_uses_horizon_or_all_fallen_without_survivor_filter():
    agents,boxes=push_terminal_records(trace(),0)
    assert len(agents)==8 and len(boxes)==3
    assert [r['terminal_step'] for r in agents]==[3]*4+[2]*4
    np.testing.assert_allclose([r['distance_3d_m'] for r in agents],[.3]*4+[.4]*4)
    assert agents[0]['fallen'] and agents[0]['agent_last_valid_distance_3d_m']==2.
    np.testing.assert_allclose([r['distance_3d_m'] for r in boxes],[.3,.4,.4])
    assert not any(r['geometric_at_goal'] for r in agents)
    assert distance_stats(agents)['trials']==8


def test_payload_and_goal_indices_follow_packet_and_agent_order_can_shuffle():
    data=trace()
    reference=push_terminal_records(data,2)
    order=np.array([2,0,3,1])
    data['packet']=data['packet'][:,order]
    data['ever']=data['ever'][:,order]
    shuffled=push_terminal_records(data,2)
    sort=lambda rows:sorted(rows,key=lambda r:(r['env'],r.get('agent',r['group'])))
    assert sort(reference[0])==sort(shuffled[0])
    assert sort(reference[1])==sort(shuffled[1])
    data['packet'][:,:,3]=4+np.array([3,0,1,2])
    changed=push_terminal_records(data,2)[0]
    assert any(r['distance_3d_m']>1 for r in changed)


def test_rotation_can_miss_region_goals_while_full_box_center_is_at_goal():
    data=trace()
    for key in ('packet','goals','groups','fall_step','ever'):
        data[key]=data[key][:1].copy()
    data['boxes']=data['boxes'][:1,:1].copy()
    data['fall_step'][:]=-1
    x=np.arange(4)-1.5
    data['boxes'][0,0,:4,0]=0
    data['boxes'][0,0,:4,1]=x
    agents,boxes=push_terminal_records(data,0)
    assert all(r['distance_3d_m']>.7 for r in agents)
    assert boxes[0]['distance_3d_m']==0
    assert boxes[0]['geometric_at_goal']


def test_fallen_solo_push_uses_scene_end_while_other_tasks_continue():
    data=trace()
    data['groups']=np.broadcast_to(np.arange(4),(2,4))
    data['packet'][:,1:,1]=2  # Other agents perform independent AT tasks.
    agents,boxes=push_terminal_records(data,0)
    assert len(agents)==len(boxes)==2
    assert agents[0]['terminal_step']==3
    assert agents[0]['agent_last_valid_step']==1
    assert agents[0]['distance_3d_m']==.3
