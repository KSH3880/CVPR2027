import numpy as np
from utils.finetune_evaluation import classify, distance_bins, summarize_rows


def scene():
    root=np.zeros((1,2,3));box=np.zeros((1,2,13));box[...,6]=1
    sizes=np.ones((1,2,3));goals=np.zeros((1,2,3))
    return root,box.copy(),box.copy(),sizes.copy(),sizes.copy(),goals


def test_point_is_3d_and_at_has_no_extra_z_gate():
    data=scene();data[1][...,2]=.19
    p,r=classify('push_at',*data)
    assert p.all() and np.array_equal(p,r)
    data[1][...,0]=.10
    assert not classify('push_at',*data)[0].any()


def test_region_yaw_and_no_set_inclusion_with_point():
    data=scene()
    data[4][...,0]=2.;data[4][...,1]=.25
    data[0][...,0]=.5;data[0][...,2]=.6
    p,r=classify('sit',*data)
    assert not p.any() and r.all()
    data[0][...,0]=0;data[0][...,1]=.15
    p,r=classify('sit',*data)
    assert p.all() and not r.any()
    data[2][...,5]=np.sin(np.pi/4);data[2][...,6]=np.cos(np.pi/4)
    data[0][...,1]=.5
    assert classify('sit',*data)[1].all()


def test_ontop_uses_upright_payload_half_height_and_rotated_support_top():
    data=scene();data[3][...,2]=.4
    data[1][...,2]=.7
    assert classify('holding_ontop',*data)[0].all()
    data[1][...,3]=np.sin(.6);data[1][...,6]=np.cos(.6)
    assert classify('holding_ontop',*data)[0].all()


def row(distance,success):
    return dict(initial_box_goal_distance_m=distance,initial_agent_box_distance_m=2.,
        final_box_goal_distance_m=0.,progress_fraction=1.,best_progress_fraction=1.,
        point_ever=success,region_ever=success,point_final=False,region_final=False,
        strict_ever=False,terminated=False,first_point_success_seconds=3. if success else None)


def test_distance_boundaries_include_outliers_and_empty_bins_are_not_zero_rates():
    rows=[row(.5,False),row(1.,True),row(2.,False),row(19.999,True),row(20.,False),row(30.,True)]
    b=distance_bins(rows)
    assert b['0-1m']['agent_trials']==1
    assert b['1-2m']['point_ever_rate']==1
    assert b['2-3m']['point_ever_rate']==0
    assert b['3-4m']['point_ever_rate'] is None
    assert b['20m+']['agent_trials']==2
    assert sum(x['agent_trials'] for x in b.values())==6
    assert summarize_rows(rows)['point_ever_rate']==.5
