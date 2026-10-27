from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest
import torch
import yaml

from utils.push_box_asset import plate_box_urdf
from utils.push_navigation import sample_box_properties
from utils.stage1_push import check_contract, check_navigation_transfer, contract, load_bank
from env.tasks.multi_agent.push_reward import contact_geometry

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid.yaml').read_text())['env']
OLD = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw.yaml').read_text())['env']
BOX = ENV['stage1Push']['box']


def test_cuboids_respect_axis_ranges_order_and_volume_mass():
    torch.manual_seed(42)
    size, mass = sample_box_properties(32768, BOX, 'cpu')
    low, high = torch.tensor(BOX['size_ranges']).unbind(1)
    assert ((size >= low) & (size <= high)).all()
    assert (size[:, 1] >= size[:, 2]).all()
    assert (size[:, 1] > size[:, 2]+.01).float().mean() > .8
    assert (size[:, 0] < 1.21).any() and (size[:, 0] > 1.69).any()
    assert mass.min() >= 15 and mass.max() <= 20
    torch.testing.assert_close(mass, 15+5*(size.prod(-1)-2.523)/(4.913-2.523))
    assert (mass[size.prod(-1).argsort()].diff() >= 0).all()


def test_volume_extremes_map_to_mass_endpoints():
    with patch('utils.push_navigation.torch.rand', return_value=torch.tensor([[0.,0.,0.],[1.,1.,1.],[.5,.25,.75]])):
        size, mass = sample_box_properties(3, BOX, 'cpu')
    torch.testing.assert_close(size[0], torch.tensor([1.2,1.45,1.45]))
    torch.testing.assert_close(size[1], torch.tensor([1.7,1.7,1.7]))
    torch.testing.assert_close(size[2], torch.tensor([1.45,1.6375,1.5125]))
    torch.testing.assert_close(mass[:2], torch.tensor([15.,20.]))


def test_cuboid_contact_uses_x_width_and_y_depth():
    box = torch.zeros(1,1,13)
    box[...,2],box[...,6] = .725,1.
    size = torch.tensor([[[1.2,1.7,1.45]]])
    hands = torch.tensor([[[[0.,-.85,-2.],[.7,-.85,5.]]]])
    forces = torch.zeros_like(hands)
    forces[...,1] = -1.
    phi,hit,_,_ = contact_geometry(hands,box,size,forces,torch.tensor([[[0.,2.,0.]]]),face_xy=True)
    torch.testing.assert_close(phi,torch.full_like(phi,torch.exp(torch.tensor(-.1))))
    assert hit.all()


@pytest.mark.parametrize('size,mass', [([1.2,1.45,1.45],15.),([1.2,1.7,1.45],17.),([1.7,1.7,1.7],20.)])
def test_cuboid_plate_keeps_bounds_and_rectangular_inertia(size, mass):
    link = ET.fromstring(plate_box_urdf(size, mass, BOX['bottom_plate'])).find('link')
    x,y,z = size
    inertia = link.find('inertial/inertia')
    assert float(inertia.get('ixx')) == pytest.approx(mass*(y*y+z*z)/12)
    assert float(inertia.get('iyy')) == pytest.approx(mass*(x*x+z*z)/12)
    assert float(inertia.get('izz')) == pytest.approx(mass*(x*x+y*y)/12)
    body,plate = link.findall('collision')
    assert [float(v) for v in body.find('geometry/box').get('size').split()] == pytest.approx([x,y,z-.01])
    assert [float(v) for v in plate.find('geometry/box').get('size').split()] == pytest.approx([x,y,.01])
    assert float(plate.find('origin').get('xyz').split()[2])-.005 == pytest.approx(-z/2)


def test_cuboid_bank_horizon_and_preserved_settings():
    bank = load_bank(ENV['stage1Push'], 'tokenhsi/data/dataset_stage1_push.yaml')
    old_bank = json.loads((ROOT/OLD['stage1Push']['rsi']['bank']).read_text())
    assert bank['candidates'] == old_bank['candidates']
    assert bank['contract']['motion_hashes'] == old_bank['contract']['motion_hashes']
    expected = deepcopy(OLD)
    expected['stage1Push']['box'] = BOX
    expected['stage1Push']['rsi']['bank'] = ENV['stage1Push']['rsi']['bank']
    expected['stage1Push']['navigation']['episode_length'] = 1200
    assert ENV == expected
    assert BOX['friction'] == .9 and BOX['bottom_plate']['friction'] == .3


def test_cuboid_requires_explicit_transfer_for_old_checkpoints():
    weights = {'stage1_push_contract':contract(OLD)}
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract(weights, ENV)
    check_navigation_transfer(weights, ENV)


def test_rsi_uniform_only_changes_rsi_distance_and_final_stage_and_warm_starts_from_cuboid():
    path = ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml'
    uniform = yaml.safe_load(path.read_text())['env']
    expected = deepcopy(ENV)
    expected['stage1Push']['rsi']['uniform_goal_distance'] = True
    expected['stage1Push']['rsi']['goal_distance_max'] = 12.0
    final = deepcopy(expected['stage1Push']['navigation']['curriculum'][-1])
    final['epoch'] = 0
    expected['stage1Push']['navigation']['curriculum'] = [final]
    assert uniform == expected
    assert not ENV['stage1Push']['rsi'].get('uniform_goal_distance', False)
    saved = {'stage1_push_contract':contract(ENV)}
    check_navigation_transfer(saved, uniform)
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract(saved, uniform)
    check_contract({'stage1_push_contract':contract(uniform)}, uniform)
    previous = deepcopy(uniform)
    previous['stage1Push']['rsi'].pop('goal_distance_max')
    old_uniform = {'stage1_push_contract':contract(previous)}
    check_navigation_transfer(old_uniform, uniform)
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract(old_uniform, uniform)
    changed = deepcopy(uniform)
    changed['stage1Push']['rsi']['late_probability'] = .2
    with pytest.raises(ValueError, match='RSI'):
        check_navigation_transfer(saved, changed)


def test_warm_start_accepts_same_teacher_path_but_rejects_other_teacher():
    absolute = deepcopy(ENV)
    absolute['stage1Push']['source_checkpoint'] = str((ROOT/ENV['stage1Push']['source_checkpoint']).resolve())
    saved = {'stage1_push_contract':contract(absolute)}
    check_navigation_transfer(saved, ENV)
    different = deepcopy(ENV)
    different['stage1Push']['source_checkpoint'] = 'output/stage1/Stage1V2Fixed_00060000.pth'
    with pytest.raises(ValueError, match='source teacher'):
        check_navigation_transfer(saved, different)


def test_rsi_uniform_keeps_optimizer_and_amp_configuration():
    directory = ROOT/'tokenhsi/data/cfg/train/rlg'
    old = yaml.safe_load((directory/'stage1_v2_fixed_push_360_raw_cuboid.yaml').read_text())
    uniform = yaml.safe_load((directory/'stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml').read_text())
    old['params']['config']['name'] += 'RsiUniform'
    assert uniform == old
