from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import pytest
from ksh_jointpush.ksh_push_signed_eval import signed_geometry,signed_terminal_records


def test_sideways_error_is_separate_from_forward_overshoot_and_sign_is_correct():
    ahead=signed_geometry([0,0,0],[3,7,0],[0,5,0])
    behind=signed_geometry([0,0,0],[3,3,0],[0,5,0])
    assert ahead['signed_along_m']==2 and ahead['lateral_error_m']==3
    assert behind['signed_along_m']==-2
    assert ahead['signed_goal_distance_m']==pytest.approx(np.sqrt(13))
    assert behind['signed_goal_distance_m']==pytest.approx(-np.sqrt(13))
    plane=signed_geometry([0,0,0],[3,5,0],[0,5,0])
    assert plane['signed_along_m']==0 and not plane['terminal_past_goal']
    assert plane['signed_goal_distance_m']==-3
    exact=signed_geometry([0,0,0],[0,5,0],[0,5,0])
    assert exact['signed_goal_distance_m']==0


@pytest.mark.parametrize('yaw',[0.,.5,1.57,3.14])
def test_signed_geometry_is_invariant_to_world_rotation_and_translation(yaw):
    rotation=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    transform=lambda p:rotation@np.asarray(p)+np.array([300.,-100.,.6])
    r=signed_geometry(transform([0,0,0]),transform([3,7,0]),transform([0,5,0]))
    assert r['signed_along_m']==pytest.approx(2)
    assert r['signed_goal_distance_m']==pytest.approx(np.sqrt(13))


def test_zero_initial_direction_is_rejected():
    with pytest.raises(ValueError):signed_geometry([0,0,0],[0,1,0],[0,0,0])


@pytest.mark.parametrize('all_fallen',[False,True])
def test_reset_snapshot_sets_direction_and_terminal_frame_is_reused(all_fallen):
    packet=np.array([[[1,4,i,4+i,12+i] for i in range(4)]])
    boxes=np.zeros((3,1,8,7));boxes[...,6]=1
    x=np.arange(4)-1.5
    boxes[:,:,:4,0]=x
    boxes[:,0,:4,1]=np.array([10.,3.,1.])[:,None]
    goals=np.zeros((1,4,3));goals[:,:,0]=x;goals[:,:,1]=5
    groups=np.zeros((1,4),dtype=np.int64)
    data=dict(packet=packet,boxes=boxes,goals=goals,groups=groups,
              fall_step=np.full((1,4),2 if all_fallen else -1),ever=np.zeros((1,4,2),dtype=bool))
    initial_boxes=boxes[0].copy();initial_boxes[:,:4,1]=0
    initial=dict(packet=packet,boxes=initial_boxes,goals=goals,groups=groups)
    agents,physical=signed_terminal_records(data,initial,0)
    expected=-2 if all_fallen else -4
    assert all(r['signed_along_m']==expected for r in agents+physical)
    assert all(r['signed_goal_distance_m']==expected for r in agents+physical)
    assert all(r['terminal_step']==(2 if all_fallen else 3) for r in agents+physical)
