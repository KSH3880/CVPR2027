import numpy as np
from ksh_jointpush.ksh_departure_eval import measure


def arrays():
    return (np.zeros((20,4,2),bool),np.zeros((20,2)),np.zeros((20,2)),np.ones((20,4),bool))


def test_wait_then_depart():
    near,xy,v,alive=arrays();near[3:]=True;v[7:,0]=.1
    r=measure(near,xy,v,alive,1/30)
    assert r['departed'] and not r['premature'] and r['ready_fraction']==1
    assert np.isclose(r['ready_to_departure_seconds'],4/30)


def test_premature_and_distance():
    near,xy,v,alive=arrays();near[:,0]=True;near[12:]=True;v[2:,0]=.1;xy[:,0]=np.arange(20)*.1
    r=measure(near,xy,v,alive,1/30)
    assert r['premature'] and r['unready_at_departure'] and r['ready_fraction']==.25
    assert np.isclose(r['pre_ready_path_m'],1.2)


def test_jitter_and_no_departure():
    near,xy,v,alive=arrays();v[2:7,0]=.1
    r=measure(near,xy,v,alive,1/30)
    assert not r['departed'] and not r['ever_ready'] and not r['premature']


def test_fall_censors_and_no_credit_after_fall():
    near,xy,v,alive=arrays();alive[5:,0]=False;near[6:]=True;v[3:,0]=.2
    r=measure(near,xy,v,alive,1/30)
    assert r['censored_by_fall'] and not r['departed'] and not r['ever_ready']


def test_ready_then_slip_distinct_from_never_waiting():
    near,xy,v,alive=arrays();near[:3]=True;v[5:,0]=.1
    r=measure(near,xy,v,alive,1/30)
    assert not r['premature'] and r['unready_at_departure']


def test_reset_velocity_and_threshold_do_not_trigger():
    near,xy,v,alive=arrays();v[:,0]=.05;v[0,0]=10
    assert not measure(near,xy,v,alive,1/30)['departed']


def test_process_counts_each_physical_box_once(tmp_path):
    from ksh_jointpush.ksh_departure_eval import process
    folder=tmp_path/'general_seed0';folder.mkdir()
    near,xy,v,alive=arrays();near[2:]=True;v[5:,0]=.1
    np.savez(folder/'departure.npz',near=near[:,None],
             xy=np.broadcast_to(xy[:,None,None],(20,1,4,2)),
             velocity=np.broadcast_to(v[:,None,None],(20,1,4,2)),
             groups=np.ones((1,4),int),templates=np.full((1,4),5),dt=1/30)
    np.savez(folder/'trace.npz',valid=alive[1:,None],fall_step=np.full((1,4),-1))
    summary=process(tmp_path)
    assert summary['all']['groups']==1 and summary['4']['departed']==1
    assert summary['all']['premature_all']==0
