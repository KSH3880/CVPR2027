import math
import xml.etree.ElementTree as ET

import pytest
import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent.bridge_entry_chamfer import footprint, write_asset, evaluation_enabled, WIDTH, DEPTH


def test_evaluation_guard():
    assert not evaluation_enabled('0', False)
    assert evaluation_enabled('1', True)
    with pytest.raises(ValueError, match='evaluation-only'):
        evaluation_enabled('1', False)
    with pytest.raises(ValueError, match='0 or 1'):
        evaluation_enabled('.2', True)


@pytest.mark.parametrize('side', [1, -1])
def test_convex_platform_and_matching_height(tmp_path, side):
    polygon = footprint((4.,6.,2.), side)
    area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1]))/2
    assert area == pytest.approx(24.-WIDTH*DEPTH)
    path = write_asset(tmp_path, (4.,6.,2.), side)
    assert path.name.count('.') == 1
    assert len(ET.parse(path).findall('.//collision')) == 1
    obj = path.with_suffix('.obj').read_text().splitlines()
    assert sum(row.startswith('v ') for row in obj) == 12
    assert sum(row.startswith('f ') for row in obj) == 20
    centers = torch.tensor([[[0.,0.,1.], [5.,5.,.5]]])
    halves = torch.tensor([[[2.,3.,1.], [.1,.1,.5]]])
    chamfer = torch.tensor([[[side,WIDTH,DEPTH],[0.,0.,0.]]])
    local = torch.tensor([[[side*1.95,2.95], [side*1.95,2.5], [side*1.75,2.95], [0.,0.], [5.,5.]]])
    for yaw in (0., math.pi/3):
        c,s = math.cos(yaw),math.sin(yaw)
        xy = torch.stack((c*local[...,0]-s*local[...,1],s*local[...,0]+c*local[...,1]),-1)
        current = centers.clone()
        current[...,:2] = torch.stack((c*centers[...,0]-s*centers[...,1],s*centers[...,0]+c*centers[...,1]),-1)
        quats = torch.tensor([[[0.,0.,math.sin(yaw/2),math.cos(yaw/2)]]]).expand(1,2,4)
        height = sensor.vertical_height(xy,current,quats,halves,torch.zeros(1),chamfer)
        torch.testing.assert_close(height,torch.tensor([[0.,2.,2.,2.,1.]]))
        full = sensor.vertical_height(xy,current,quats,halves,torch.zeros(1))
        torch.testing.assert_close(full,torch.tensor([[2.,2.,2.,2.,1.]]))
