import xml.etree.ElementTree as ET

import pytest

from utils.push_box_asset import plate_box_urdf


@pytest.mark.parametrize('size,mass', [(1.2, 12.), (1.45, 16.), (1.7, 20.)])
@pytest.mark.parametrize('inset', [0., .015])
def test_plate_box_keeps_outer_bounds_mass_and_inertia(size, mass, inset):
    plate = dict(friction=.3, thickness=.01, inset=inset)
    root = ET.fromstring(plate_box_urdf([size]*3, mass, plate))
    links = root.findall('link')
    assert len(links) == 1 and not root.findall('joint')
    body, bottom = links[0].findall('collision')
    body_size = [float(v) for v in body.find('geometry/box').get('size').split()]
    bottom_size = [float(v) for v in bottom.find('geometry/box').get('size').split()]
    body_z = float(body.find('origin').get('xyz').split()[2])
    bottom_z = float(bottom.find('origin').get('xyz').split()[2])
    assert body_size == pytest.approx([size, size, size-.01])
    assert bottom_size == pytest.approx([size-2*inset, size-2*inset, .01])
    assert body_z+body_size[2]/2 == pytest.approx(size/2)
    assert bottom_z-bottom_size[2]/2 == pytest.approx(-size/2)
    assert body_z-body_size[2]/2 == pytest.approx(bottom_z+bottom_size[2]/2)
    inertial = links[0].find('inertial')
    assert float(inertial.find('mass').get('value')) == mass
    assert inertial.find('origin').get('xyz') == '0 0 0'
    for axis in ('ixx', 'iyy', 'izz'):
        assert float(inertial.find('inertia').get(axis)) == pytest.approx(mass*size*size/6)


@pytest.mark.parametrize('thickness,inset', [(0., .015), (1.3, .015), (.01, .65), (.01, -.01)])
def test_plate_must_fit_inside_the_box(thickness, inset):
    with pytest.raises(ValueError, match='fit inside'):
        plate_box_urdf([1.3]*3, 16., dict(thickness=thickness, inset=inset))
