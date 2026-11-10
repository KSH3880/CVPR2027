import math
from pathlib import Path
import sys

import pytest
import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'tokenhsi')]
from table2_metrics import Table2Metrics, surface_distance
from table2_sampling import SETTINGS, prepare_bank
from rescore_table2_speed import speed_window_counts


def initial(people=2, enable=False):
    packet = torch.tensor([[[1,2,0,2,6], [1,0,1,2,2]]]) if enable else torch.zeros(1,people,5,dtype=torch.long)
    sizes = torch.ones(1,2*people,3)*.4
    goals = torch.zeros(1,people,3)
    goals[...,2] = .2
    center = torch.zeros(1,people,3)
    target = center.clone()
    target[...,0] = 1.
    return dict(packet=packet, groups=torch.zeros(1,people,dtype=torch.long), sizes=sizes,
                goals=goals, shared_center_goals=target, shared_centers=center)


def test_surface_distance_uses_rotated_box_and_true_surface_inside():
    hands = torch.tensor([[[[1.06,0.,0.], [0.,0.,0.]]]],dtype=torch.float64)
    boxes = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],dtype=torch.float64)
    sizes = torch.tensor([[2.,4.,6.]],dtype=torch.float64)
    torch.testing.assert_close(surface_distance(hands,boxes,sizes),torch.tensor([[[.06,1.]]],dtype=torch.float64))
    boxes[0,5:] = torch.tensor([math.sin(math.pi/4),math.cos(math.pi/4)])
    hands = torch.tensor([[[[0.,1.06,0.], [0.,0.,0.]]]],dtype=torch.float64)
    torch.testing.assert_close(surface_distance(hands,boxes,sizes),torch.tensor([[[.06,1.]]],dtype=torch.float64),atol=1e-7,rtol=0)


def test_enable_time_requires_first_simultaneous_success():
    metric = Table2Metrics(initial(enable=True),'enable_sit','cpu')
    roots = torch.zeros(1,2,3)
    roots[0,1] = torch.tensor([2.,0.,.5])
    boxes = torch.zeros(1,4,7)
    boxes[...,6] = 1.
    boxes[0,0,2] = .2
    alive = torch.ones(1,2,dtype=torch.bool)
    metric.update(roots,boxes,alive)
    boxes[0,0,0] = 2.
    metric.update(roots,boxes,alive)
    assert metric.first_success.item() == -1
    boxes[0,0,0] = roots[0,1,0] = 0.
    metric.update(roots,boxes,alive)
    assert metric.first_success.item() == 3
    assert metric.summary(1/30)['completion_time_s'] == pytest.approx(.1)


def push(metric, x, gaps, alive=None):
    people = len(gaps)
    box = torch.tensor([[x,0.,0.,0.,0.,0.,1.]],dtype=torch.float64)
    hands = torch.zeros(1,people,2,3,dtype=torch.float64)
    hands[...,0] = x+1+torch.tensor(gaps,dtype=torch.float64)[None,:,None]
    alive = torch.ones(1,people,dtype=torch.bool) if alive is None else alive
    metric.update(torch.zeros(1,people,3),torch.zeros(1,2*people,7),alive,
                  physical=box,physical_sizes=torch.tensor([[2.,2.,2.]]),hands=hands)


def test_coupled_window_stops_at_first_success_and_reaggregates_thresholds():
    metric = Table2Metrics(initial(),'coupled_2','cpu')
    push(metric,.1,[.01,.01])
    push(metric,.3,[.05,.05])
    push(metric,.5,[.07,.07])
    push(metric,.85,[.11,.11])
    push(metric,1.,[.01,.01])
    result = metric.summary(1/30)
    assert metric.start.item() == 2
    assert metric.first_success.item() == 4
    assert metric.frames.item() == 3
    assert result['cooperative_time_ratio_percent'] == pytest.approx({'6':100/3,'8':200/3,'12':100.})


def test_failed_push_keeps_actual_ratio_and_original_fallen_participants():
    metric = Table2Metrics(initial(),'coupled_2','cpu')
    push(metric,.3,[.01,.01])
    push(metric,.4,[.01,.01],torch.tensor([[True,False]]))
    result = metric.summary(1/30)
    assert result['successes'] == 0
    assert result['cooperative_time_ratio_percent']['6'] == 50.


def test_no_push_start_is_zero_even_with_close_hands():
    metric = Table2Metrics(initial(),'coupled_2','cpu')
    push(metric,.1,[.01,.01])
    assert metric.summary(1/30)['cooperative_time_ratio_percent']['6'] == 0.


def test_40cm_start_changes_window_but_preserves_success():
    metrics = [Table2Metrics(initial(),'coupled_2','cpu',start_displacement_m=value) for value in (.2,.4)]
    for metric in metrics:
        push(metric,.2,[.2,.2])
        push(metric,.4,[.01,.01])
        push(metric,.85,[.01,.01])
        push(metric,1.,[.2,.2])
    assert [metric.start.item() for metric in metrics] == [1,2]
    assert [metric.frames.item() for metric in metrics] == [3,2]
    assert [metric.first_success.item() for metric in metrics] == [3,3]
    assert [metric.summary(1/30)['cooperative_time_ratio_percent']['6'] for metric in metrics] == pytest.approx([200/3,100.])


def test_40cm_start_not_reached_is_zero():
    metric = Table2Metrics(initial(),'coupled_2','cpu',start_displacement_m=.4)
    push(metric,.3,[.01,.01])
    assert metric.start.item() == -1
    assert metric.summary(1/30)['cooperative_time_ratio_percent']['6'] == 0.


def test_hand_start_requires_same_frame_valid_hand_and_latches():
    metric = Table2Metrics(initial(),'coupled_2','cpu',start_hand_distance_m=.16)
    push(metric,.3,[.2,.2])
    push(metric,.3,[.15,.3],torch.tensor([[False,True]]))
    push(metric,.1,[.01,.01])
    assert metric.start.item() == -1
    push(metric,.4,[.15,.3])
    push(metric,.5,[.01,.01])
    push(metric,.85,[.3,.3])
    push(metric,1.,[.01,.01])
    assert metric.start.item() == 4
    assert metric.first_success.item() == 6
    assert metric.frames.item() == 3
    assert metric.summary(1/30)['cooperative_time_ratio_percent']['12'] == pytest.approx(100/3)


def test_hand_start_distance_is_strict():
    metric = Table2Metrics(initial(),'coupled_2','cpu',start_hand_distance_m=.125)
    push(metric,.5,[.125,.125])
    assert metric.start.item() == -1
    push(metric,.5,[.12,.3])
    assert metric.start.item() == 2


def test_no_hand_start_preserves_success_with_zero_contact_ratio():
    metric = Table2Metrics(initial(),'coupled_2','cpu',start_hand_distance_m=.16)
    push(metric,.85,[.3,.3])
    assert metric.first_success.item() == 1
    assert metric.start.item() == -1
    assert metric.summary(1/30)['cooperative_time_ratio_percent']['12'] == 0.


def test_speed_start_backdates_and_requires_continuous_nearby_hand():
    speed = np.array([[.2],[.2],[0.],[.2],[.2],[.2],[.2]])
    nearby = np.array([[True],[False],[True],[True],[True],[True],[True]])
    contact = np.array([[False],[False],[False],[True],[False],[True],[True]])
    result = speed_window_counts(speed, nearby, contact, np.array([6]), .1, 3)
    assert result['start'].tolist() == [4]
    assert result['window_frames'].tolist() == [3]
    assert result['contact_frames'].tolist() == [2]


def test_speed_start_confirmation_after_success_does_not_count():
    speed = np.ones((4,1))
    mask = np.ones((4,1),dtype=bool)
    result = speed_window_counts(speed, mask, mask, np.array([2]), .1, 3)
    assert result['start'].tolist() == [-1]
    assert result['window_frames'].tolist() == [0]
    assert result['contact_frames'].tolist() == [0]


@pytest.mark.parametrize('duration_frames',[1,3,9])
@pytest.mark.parametrize('use_hand',[False,True])
def test_speed_windows_match_scalar_reference(duration_frames,use_hand):
    rng = np.random.default_rng(37)
    speed = rng.choice([0.,.1,.2],size=(40,32))
    nearby = rng.random((40,32))>.1 if use_hand else None
    contact = rng.random((40,32))>.5
    success = rng.integers(-1,41,size=32)
    success[success==0] = -1
    result = speed_window_counts(speed,nearby,contact,success,.1,duration_frames)
    for trial in range(32):
        end = success[trial] if success[trial]>=1 else len(speed)
        start = -1
        for frame in range(duration_frames-1,end):
            begin = frame-duration_frames+1
            if all(speed[t,trial]>=.1 and (nearby is None or nearby[t,trial]) for t in range(begin,frame+1)):
                start = begin
                break
        assert result['start'][trial] == (start+1 if start>=0 else -1)
        assert result['window_frames'][trial] == (end-start if start>=0 else 0)
        assert result['contact_frames'][trial] == (contact[start:end,trial].sum() if start>=0 else 0)


def test_speed_only_start_does_not_remove_contact_requirement():
    result = speed_window_counts(np.ones((3,1)),None,np.zeros((3,1),dtype=bool),np.array([-1]),.5,2)
    assert result['start'].tolist() == [1]
    assert result['window_frames'].tolist() == [3]
    assert result['contact_frames'].tolist() == [0]


@pytest.mark.parametrize('setting',SETTINGS)
def test_fixed_bank_has_only_requested_group_and_preserves_density(setting):
    people = SETTINGS[setting]
    sizes = torch.ones(128,2*people,3)*.5
    masses = sizes.prod(-1)*100
    _, groups, parent, tasks = prepare_bank(sizes,masses,dict(numAgents=people),'random_scenario',setting)
    assert (groups == groups[:,:1]).all()
    if setting.startswith('enable_'):
        rows,followers = (parent>=0).nonzero(as_tuple=True)
        assert len(rows) == 128
        assert set(parent[rows,followers].tolist()) == {0,1}
        assert (tasks[rows,followers] == dict(enable_sit=1,enable_climb=2,enable_ontop=4)[setting]).all()
        assert (sizes[rows,parent[rows,followers],2] >= .35).all()
    else:
        assert not (parent>=0).any()
        assert (tasks==5).all()
    torch.testing.assert_close(masses,sizes.prod(-1)*100)
