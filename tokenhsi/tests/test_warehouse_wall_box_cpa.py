from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.warehouse_sensor import collision
from utils.experiments import native_config
from utils.sensor_stage2_spec import WAREHOUSE, validate_sensor
from utils.warehouse_sensor import walls

HX, HY, T, MOUTH = 4., 4., .4, 1.12


def risk(roots, vel, mode, yaw=.7):
    """Wall CPA for people given in the room frame of a rotated, shifted warehouse."""
    n, m = roots.shape[:2]
    yaw = torch.full((n,), yaw)
    place = lambda xy: scene.rotate_xy(xy, yaw[:, None])+torch.tensor([13., 18.])
    centers, halves, starts, ends = walls(torch.tensor([HX, HY]).expand(n, 2), T, 2.5, torch.full((n,), 2*MOUTH))
    static_centers, static_half = torch.zeros(n, WAREHOUSE+1, 6, 3), torch.ones(n, WAREHOUSE+1, 6, 3)
    static_centers[:, WAREHOUSE, 1:] = torch.cat((place(centers[..., :2]), centers[..., 2:]), -1)
    static_half[:, WAREHOUSE, 1:] = halves
    states = torch.zeros(n, m, 13)
    states[..., :2], states[..., 2], states[..., 7:9] = place(roots), .9, scene.rotate_xy(vel, yaw[:, None])
    task = SimpleNamespace(dt=1/30, _scene_ids=torch.full((n,), WAREHOUSE), _humanoid_root_states=states,
        _warehouse={'wall_ttc_discount': .95, 'wall_cpa': mode}, _wall_collision_scale=1., _push_box_collision_scale=0.,
        _warehouse_wall_starts=place(starts), _warehouse_wall_ends=place(ends), _static_centers=static_centers,
        _static_half=static_half, _static_quat=scene.yaw_quat(yaw)[:, None, None].expand(n, WAREHOUSE+1, 6, 4))
    return collision(task, torch.zeros(n, m), states[..., :3], .7)


def test_inside_the_room_box_walls_match_inner_face_segments():
    torch.manual_seed(0)
    roots = (torch.rand(256, 4, 2)*2-1)*torch.tensor([HX-.25, HY-.25])
    vel = torch.randn(256, 4, 2)
    box = risk(roots, vel, 'box')
    assert box.max() > .3
    torch.testing.assert_close(box, risk(roots, vel, 'segment'))


@pytest.mark.parametrize('gap', [.2, .3, .5])
def test_outside_approach_matches_inside_approach(gap):
    # outside the front wall (away from the door) vs inside towards the back wall, both head-on at 1 m/s
    roots = torch.tensor([[[-HX-T-gap, 2.5], [HX-gap, 2.5]]])
    vel = torch.tensor([[[1., 0.], [1., 0.]]])
    box, segment = risk(roots, vel, 'box')[0], risk(roots, vel, 'segment')[0]
    torch.testing.assert_close(box[0], box[1])
    torch.testing.assert_close(segment[1], box[1])
    assert segment[0] < .6*box[0]


def test_door_jamb_is_a_wall_face():
    # inside the doorway moving sideways into the jamb equals a head-on approach to the back wall
    roots = torch.tensor([[[-HX-T/2, MOUTH-.3], [HX-.3, 2.5]]])
    vel = torch.tensor([[[0., 1.], [1., 0.]]])
    box = risk(roots, vel, 'box')[0]
    torch.testing.assert_close(box[0], box[1])
    assert risk(roots, vel, 'segment')[0, 0] < box[0]


def test_wall_cpa_option_is_validated():
    env = native_config('stage2_sensor_900_warehouse_fixedroles_v4')['env']
    assert 'wall_cpa' not in env['sensorScene']['warehouse']
    env['sensorScene']['warehouse']['wall_cpa'] = 'box'
    validate_sensor(env)
    env['sensorScene']['warehouse']['wall_cpa'] = 'boxes'
    with pytest.raises(ValueError, match='wall_cpa'):
        validate_sensor(env)
