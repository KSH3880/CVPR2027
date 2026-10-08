"""Handedness, spring strength and reproducibility of the physical door asset."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from tokenhsi.utils.door_asset import DoorSpec,write_door_asset

ROOT=Path(__file__).resolve().parents[2]


def test_left_hinge_right_handle_and_opening_direction():
    spec=DoorSpec()
    hinge=spec.hinge
    closed=spec.handle_position(0.)
    opened=spec.handle_position(math.pi/2)
    assert hinge[1]>0 and closed[1]<0 and closed[0]<0
    assert opened[0]>closed[0] and opened[1]>closed[1]
    assert closed[2]==opened[2]==spec.handle_height
    # Rotation/translation of the complete scene must preserve this convention.
    base=np.array([2.,-3.,0.]);yaw=math.pi/2
    expected=base+np.array([-closed[1],closed[0],closed[2]])
    np.testing.assert_allclose(spec.handle_position(0.,base,yaw),expected,atol=1e-8)


def test_closing_spring_requires_holding_force_and_damps():
    spec=DoorSpec()
    assert spec.closing_torque(0.)==0.
    assert -10 < spec.closing_torque(math.pi/2) < -8
    assert spec.closing_torque(.5,1.) < spec.closing_torque(.5,0.)
    assert spec.closing_torque(0.,-1.) > 0
    radius=np.linalg.norm(spec.front_handle_local[:2])
    assert 10 < abs(spec.closing_torque(math.pi/2))/radius < 13


def test_asset_has_one_revolute_hinge_and_collidable_handles(tmp_path):
    generated=write_door_asset(tmp_path/'door.urdf')
    assert generated.read_bytes()==(ROOT/'tokenhsi/data/assets/door/left_hinge_right_handle.urdf').read_bytes()
    robot=ET.parse(generated).getroot()
    hinges=[j for j in robot.findall('joint') if j.attrib['type']!='fixed']
    assert len(hinges)==1
    hinge=hinges[0]
    assert hinge.attrib['name']=='door_hinge'
    assert hinge.find('axis').attrib['xyz']=='0 0 1'
    limit=hinge.find('limit').attrib
    assert float(limit['lower'])==0 and math.isclose(float(limit['upper']),DoorSpec().max_angle)
    links={l.attrib['name']:l for l in robot.findall('link')}
    assert set(links)=={'frame','panel','handle','handle_back'}
    assert len(links['frame'].findall('collision'))==3
    assert all(links[name].find('collision') is not None for name in ('panel','handle','handle_back'))
    for link in links.values():
        inertia=link.find('inertial/inertia').attrib
        assert all(float(inertia[key])>0 for key in ('ixx','iyy','izz'))


@pytest.mark.parametrize('kwargs',[{'spring_stiffness':0},{'spring_damping':-1},{'width':float('nan')},
                                 {'handle_inset':.6},{'handle_height':3},{'frame_gap':.01}])
def test_reject_invalid_physical_specs(kwargs):
    with pytest.raises(ValueError):DoorSpec(**kwargs)


def test_lower_handle_asset_matches_geometry_and_preserves_default(tmp_path):
    from tokenhsi.utils.door_asset import cached_door_asset
    spec=DoorSpec(handle_height=.95)
    path=cached_door_asset(tmp_path,spec)
    assert cached_door_asset(tmp_path,spec)==path
    robot=ET.parse(path).getroot()
    for name in ('handle_mount','handle_back_mount'):
        xyz=robot.find("joint[@name='"+name+"']/origin").attrib['xyz']
        assert float(xyz.split()[2])==.95
    for angle in (0.,.7,math.pi/2):assert spec.handle_position(angle)[2]==.95
    assert DoorSpec().handle_height==1.05
    assert cached_door_asset(tmp_path,DoorSpec())!=path
