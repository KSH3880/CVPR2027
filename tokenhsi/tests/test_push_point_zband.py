"""Inclusive height band, smooth outside penalty, and contact boundaries."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
import math
import pytest
import torch
import yaml
from utils.push_point import validate_point_reward,point_reward_phi,point_reward_contact
from test_push_point import scene


def band_scene():
    e,b,s,l,h=scene()
    e=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_point_zband_goal3to5.yaml').read_text())['env']
    f=torch.zeros_like(h);f[...,1]=-2
    bf=torch.zeros_like(b[...,:3]);bf[...,1]=4
    return e,b,s,l,h,f,bf


def test_entire_band_including_endpoints_gets_full_reward_and_contact():
    e,b,s,l,h,f,bf=band_scene();validate_point_reward(e)
    for height in (.585,.600,.6175,.635,.650):
        h[...,2]=height
        torch.testing.assert_close(point_reward_phi(h,b,[1.3]*3,s,l,e['pushPointReward']),torch.ones(2,2))
        assert point_reward_contact(h,f,b,bf,[1.3]*3,e['relationReward']['push'],s,l,e['pushPointReward']).all()
    h[...,0]+=.3
    assert not point_reward_contact(h,f,b,bf,[1.3]*3,e['relationReward']['push'],s,l,e['pushPointReward']).any()


def test_outside_band_penalty_and_contact_are_symmetric():
    e,b,s,l,h,f,bf=band_scene()
    for extra in (.0001,.01,.04):
        for height in (.585-extra,.650+extra):
            h[...,2]=height
            phi=point_reward_phi(h,b,[1.3]*3,s,l,e['pushPointReward'])
            torch.testing.assert_close(phi,torch.full((2,2),math.exp(-(extra/.04)**2)),atol=2e-6,rtol=0)
            assert not point_reward_contact(h,f,b,bf,[1.3]*3,e['relationReward']['push'],s,l,e['pushPointReward']).any()


def test_band_is_fixed_known_geometry_not_random_hidden_target():
    e,*_=band_scene();validate_point_reward(e)
    for bounds in ([.65,.585],[.58,.65],[-.1,.65],[.585,float('nan')]):
        bad=deepcopy(e);bad['pushPointReward']['height_band_m']=bounds
        with pytest.raises(ValueError):validate_point_reward(bad)
