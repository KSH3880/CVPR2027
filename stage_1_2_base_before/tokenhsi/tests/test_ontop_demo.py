import numpy as np
from learning.multi_agent.ontop_demo import qualifying_wait


def trace():
    return [np.full(80,1.7),np.full(80,.1),np.ones(80,bool),
            np.arange(80)>=40,np.ones(80,bool)]


def test_requires_wait_before_first_success():
    values=trace()
    assert qualifying_wait(*values,1/30)==(0,29,40)
    values[3][0]=True
    assert qualifying_wait(*values,1/30) is None


def test_rejects_far_moving_not_holding_and_interrupted_wait():
    for column,value in [(0,2.01),(1,.21),(2,False),(4,False)]:
        values=trace()
        values[column][10:30]=value
        assert qualifying_wait(*values,1/30) is None


def test_missing_predecessor_and_early_predecessor_fail():
    values=trace(); values[3][:]=False
    assert qualifying_wait(*values,1/30) is None
    values[3][20:]=True
    assert qualifying_wait(*values,1/30) is None


def test_three_metre_fallback_does_not_relax_order_or_duration():
    values=trace(); values[0][:]=2.7
    assert qualifying_wait(*values,1/30) is None
    assert qualifying_wait(*values,1/30,radius=3)==(0,29,40)
    values[0][:]=3.01
    assert qualifying_wait(*values,1/30,radius=3) is None
    values[0][:]=2.7; values[3][20:]=True
    assert qualifying_wait(*values,1/30,radius=3) is None


def test_explicit_short_wait_keeps_order_and_carry_requirement():
    values=trace();values[3][20:]=True
    assert qualifying_wait(*values,1/30) is None
    assert qualifying_wait(*values,1/30,seconds=.5)==(0,14,20)
    values[2][7]=False
    assert qualifying_wait(*values,1/30,seconds=.5) is None


def test_extent_wait_allows_sway_but_rejects_travel():
    values=trace(); values[1][:]=1.0
    xy=np.zeros((80,2)); xy[:,0]=np.sin(np.arange(80))*.1
    assert qualifying_wait(*values,1/30,seconds=.5,positions=xy,max_extent=.3)==(0,14,40)
    xy[:,0]=np.arange(80)*.05
    assert qualifying_wait(*values,1/30,seconds=.5,positions=xy,max_extent=.3) is None


def test_continuous_carry_keeps_later_wait_after_drop():
    from learning.multi_agent.ontop_demo import holding_through_gate
    holding=np.ones(10,bool); holding[3]=False
    gate=np.arange(10)>=8
    assert np.array_equal(holding_through_gate(holding,gate),[False]*4+[True]*5+[False])


def test_center_above_full_footprint_accepts_edge_not_beside_or_below():
    import torch
    from learning.multi_agent.ontop_demo import center_above_support
    support=torch.zeros(4,13);support[:,6]=1
    payload=support.clone();payload[:,2]=1
    payload[:,0]=torch.tensor([.49,.51,0.,0.]);payload[2,2]=.4
    assert center_above_support(payload,support,torch.ones(4,3)).tolist()==[True,False,False,True]
