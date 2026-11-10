from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from state import CoordinationState


def test_enable_waits_for_predecessor_and_releases_once():
    state = CoordinationState(torch.tensor([[-1,0]]),torch.tensor([[0,0]]))
    alive = torch.ones(1,2,dtype=torch.bool)
    no = torch.zeros_like(alive)
    assert state.frozen.tolist() == [[False,True]]
    assert state.update(torch.tensor([[False,True]]),no,alive).tolist() == [[False,True]]
    assert state.update(torch.tensor([[True,False]]),no,alive).tolist() == [[False,False]]
    assert state.update(no,no,alive).tolist() == [[False,False]]


def test_last_arrival_releases_everyone_without_freezing():
    state = CoordinationState(torch.full((1,3),-1),torch.zeros(1,3,dtype=torch.long))
    alive = torch.ones(1,3,dtype=torch.bool)
    no = torch.zeros_like(alive)
    assert state.update(no,torch.tensor([[True,False,False]]),alive).tolist() == [[True,False,False]]
    assert state.update(no,torch.tensor([[False,True,False]]),alive).tolist() == [[True,True,False]]
    assert not state.released.any()
    assert not state.update(no,torch.tensor([[False,False,True]]),alive).any()
    assert state.released.all()
    assert not state.update(no,no,alive).any()


def test_arrivals_in_one_coupled_group_do_not_unlock_another():
    state = CoordinationState(torch.full((1,5),-1),torch.tensor([[0,0,2,2,4]]))
    alive = torch.ones(1,5,dtype=torch.bool)
    no = torch.zeros_like(alive)
    state.update(no,torch.tensor([[True,False,True,False,False]]),alive)
    frozen = state.update(no,torch.tensor([[False,True,False,False,False]]),alive)
    assert frozen.tolist() == [[False,False,True,False,False]]
    assert state.released.tolist() == [[True,True,False,False,False]]


def test_fallen_member_cannot_satisfy_coupled_barrier():
    state = CoordinationState(torch.full((1,2),-1),torch.zeros(1,2,dtype=torch.long))
    contact = torch.ones(1,2,dtype=torch.bool)
    frozen = state.update(torch.zeros_like(contact),contact,torch.tensor([[True,False]]))
    assert frozen.tolist() == [[True,False]]
    assert not state.released.any()


def test_groups_release_independently_and_solo_is_never_frozen():
    parent = torch.tensor([[1,-1,-1,-1,-1]])
    state = CoordinationState(parent,torch.tensor([[1,1,3,3,4]]))
    alive = torch.ones(1,5,dtype=torch.bool)
    success = torch.tensor([[False,True,False,False,False]])
    contact = torch.tensor([[False,False,True,True,False]])
    assert not state.update(success,contact,alive).any()
    assert state.released.tolist() == [[True,False,True,True,False]]


def test_initial_follower_fall_stops_freeze_without_unlocking_dependency():
    state = CoordinationState(torch.tensor([[-1,0]]),torch.tensor([[0,0]]))
    no = torch.zeros(1,2,dtype=torch.bool)
    assert not state.update(no,no,torch.tensor([[True,False]])).any()
    assert not state.released.any()


def test_enable_accepts_predecessor_success_on_last_valid_fall_frame():
    state = CoordinationState(torch.tensor([[-1,0]]),torch.tensor([[0,0]]))
    assert not state.update(torch.tensor([[True,False]]),torch.zeros(1,2,dtype=torch.bool),
                            torch.tensor([[False,True]])).any()
    assert state.released.tolist() == [[False,True]]
