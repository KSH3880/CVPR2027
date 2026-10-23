"""Native-object evaluation: source safety and independent seat scoring."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location('object_zero_shot',
    Path(__file__).parents[1] / 'scripts/multi_agent/evaluate_object_zero_shot.py')
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_seat_scoring_does_not_change_policy_geometry():
    objects = torch.zeros(2, 4, 13)
    objects[..., 2] = .45
    objects[..., 6] = 1
    sizes = torch.tensor([.50, .55, .90]).expand(2, 4, 3).clone()
    targets = torch.zeros(2, 4, 3)
    # Native tarSitPos at local Z=.10 means a .45m seat plus .10 clearance.
    targets[:, :2, 2] = .10
    before_obj, before_size = objects.clone(), sizes.clone()
    proxy, proxy_size = EVAL.seat_scoring_geometry(objects, sizes, targets, 2)
    assert torch.equal(objects, before_obj)
    assert torch.equal(sizes, before_size)
    assert torch.equal(proxy[:, 2:], objects[:, 2:])
    assert torch.equal(proxy_size[:, 2:], sizes[:, 2:])
    # Original trained clearance .12 targets pelvis Z=.57, not backrest+.12=1.02.
    assert torch.allclose(proxy[:, :2, 2] + proxy_size[:, :2, 2] / 2 + .12,
                          torch.full((2, 2), .57))
    assert torch.equal(proxy_size[..., :2], sizes[..., :2])


def test_seat_proxy_rotates_in_object_frame():
    objects = torch.zeros(1, 4, 13)
    objects[..., 6] = 1
    # 90 degrees about X rotates the scoring center from -Z to +Y.
    objects[:, :2, 3] = 2 ** -.5
    objects[:, :2, 6] = 2 ** -.5
    sizes = torch.ones(1, 4, 3)
    targets = torch.zeros(1, 4, 3)
    targets[..., 2] = .10
    proxy, _ = EVAL.seat_scoring_geometry(objects, sizes, targets, 2)
    assert torch.allclose(proxy[:, :2, 1], torch.full((1, 2), .25))
    assert torch.allclose(proxy[:, :2, 2], torch.zeros(1, 2), atol=1e-7)


def test_invalid_seat_rejected():
    objects = torch.zeros(1, 4, 13)
    objects[..., 6] = 1
    with pytest.raises(ValueError, match='above'):
        EVAL.seat_scoring_geometry(objects, torch.ones(1, 4, 3),
                                  torch.full((1, 4, 3), -1.), 2)


def test_multibody_asset_rejected(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(dict(bbox=[1, 1, 1], center=[0, 0, 0])))
    (tmp_path / 'asset.urdf').write_text('<robot><link name="a"/><link name="b"/></robot>')
    with pytest.raises(ValueError, match='one rigid body'):
        EVAL.asset_spec(tmp_path)


def test_no_output_can_be_written_to_ksh():
    opts = type('Options', (), {'output': EVAL.UPSTREAM / 'output/forbidden'})()
    with pytest.raises(ValueError, match='inside this checkout'):
        EVAL.prepare(opts)


def test_viewer_step_tensor_matches_common_player_contract():
    observation = torch.zeros(2, 634)
    wrapped = {'obs': observation}
    assert EVAL.policy_observation(wrapped) is wrapped
    # Initial env_reset is a dict; subsequent env_step observations are tensors.
    for value in (wrapped, observation, observation):
        player_input = EVAL.policy_observation(value)
        assert player_input['obs'] is observation
        assert player_input['obs'].shape == (2, 634)


def test_viewer_relaunch_preserves_old_output(tmp_path, monkeypatch):
    monkeypatch.setattr(EVAL, 'ROOT', tmp_path)
    original = tmp_path / 'viewer'
    original.mkdir()
    log = original / 'old.log'
    log.write_text('keep this')
    (tmp_path / 'viewer_run001').mkdir()
    selected = EVAL.output_directory(original, visualize=True)
    assert selected == tmp_path / 'viewer_run002'
    assert log.read_text() == 'keep this'
    assert not selected.exists()
    with pytest.raises(FileExistsError):
        EVAL.output_directory(original, visualize=False)


def test_reset_key_survives_release_and_multiple_render_calls():
    task = SimpleNamespace(_object_viewer_reset=False, _object_viewer_pause=False)
    event = lambda action, value: SimpleNamespace(action=action, value=value)
    assert EVAL.viewer_events(task, [event('object_reset', 1)]) == []
    assert task._object_viewer_reset
    assert EVAL.viewer_events(task, [event('object_reset', 0)]) == []
    assert task._object_viewer_reset
    EVAL.viewer_events(task, [])
    assert task._object_viewer_reset


def test_native_viewer_shortcuts_are_forwarded_once():
    task = SimpleNamespace(_object_viewer_reset=False, _object_viewer_pause=False)
    original = [SimpleNamespace(action=name, value=1)
                for name in ('QUIT', 'follow', 'show_lines', 'toggle_video_record', 'toggle_viewer_sync')]
    pause = SimpleNamespace(action='object_pause', value=1)
    assert EVAL.viewer_events(task, original + [pause]) == original
    assert task._object_viewer_pause
    EVAL.viewer_events(task, [SimpleNamespace(action='object_pause', value=0)])
    assert task._object_viewer_pause


def test_support_proxy_pose_velocity_and_input_immutability():
    states = torch.zeros(2, 4, 13)
    states[..., 6] = 1
    states[:, :2, 12] = 2  # world angular velocity around Z
    sizes = torch.ones(2, 4, 3)
    targets = torch.zeros(2, 4, 3)
    targets[:, :2, 0] = .1
    targets[:, :2, 2] = .1
    original = states.clone()
    proxy, small = EVAL.support_proxy_geometry(states, sizes, targets, 2, .75)
    assert torch.equal(states, original)
    assert torch.equal(sizes, torch.ones_like(sizes))
    assert torch.allclose(proxy[:, :2, 0], torch.full((2, 2), .1))
    assert torch.allclose(proxy[:, :2, 8], torch.full((2, 2), .2))
    assert torch.allclose(proxy[:, :2, 2] + small[:, :2, 2] / 2, torch.zeros(2, 2))
    assert torch.equal(proxy[:, 2:], states[:, 2:])
    obs = torch.randn(2, 634)
    before = obs.clone()
    result = EVAL.proxy_policy_observation(obs, states, sizes, proxy, small, 2)
    assert torch.equal(obs, before)
    assert torch.equal(result[:, :446], obs[:, :446])
    assert torch.equal(result[:, 506:582], obs[:, 506:582])
    assert torch.equal(result[:, 596:], obs[:, 596:])
    for offset in (446, 476):
        corners = result[:, offset + 6:offset + 30].reshape(2, 8, 3)
        assert torch.allclose(corners.mean(1), torch.zeros(2, 3))
        assert torch.allclose(corners.amax(1) - corners.amin(1), small[:, 0])
    assert torch.allclose(result[:, 582:585] - obs[:, 582:585], proxy[:, 0, :3] - states[:, 0, :3])
    # The same pure transform must work for subset resets and noncontiguous rows.
    subset = EVAL.proxy_policy_observation(obs[1:], states[1:], sizes[1:], proxy[1:], small[1:], 2)
    assert torch.equal(subset, result[1:])


def test_support_proxy_rotation_and_invalid_layout():
    states = torch.zeros(1, 4, 13)
    states[..., 3] = 2 ** -.5
    states[..., 6] = 2 ** -.5
    sizes = torch.ones(1, 4, 3)
    targets = torch.zeros(1, 4, 3)
    targets[..., 2] = .1
    proxy, small = EVAL.support_proxy_geometry(states, sizes, targets, 2)
    assert torch.allclose(proxy[:, :2, 1], torch.full((1, 2), .25))
    with pytest.raises(ValueError, match='layout'):
        EVAL.proxy_policy_observation(torch.zeros(1, 633), states, sizes, proxy, small, 2)
    with pytest.raises(ValueError, match='fraction'):
        EVAL.support_proxy_geometry(states, sizes, targets, 2, 0)


def test_ascent_accepts_offcenter_standing_and_rejects_airborne_contactless_pose():
    objects = torch.zeros(1, 1, 13)
    objects[..., 6] = 1
    sizes = torch.ones(1, 1, 3)
    feet = torch.tensor([[[[.4, -.1, .53], [.4, .1, .53]]]])
    roots = torch.tensor([[[.49, .2, 1.2]]])  # deliberately away from central root target
    forces = torch.zeros_like(feet)
    forces[..., 0, 2] = 100
    original = (feet.clone(), roots.clone(), objects.clone(), sizes.clone(), forces.clone())
    geometry, supported = EVAL.climb_ascent_success(feet, roots, objects, sizes, forces)
    assert geometry.item() and supported.item()
    geometry, supported = EVAL.climb_ascent_success(feet, roots, objects, sizes, forces * 0)
    assert geometry.item() and not supported.item()
    for before, after in zip(original, (feet, roots, objects, sizes, forces)):
        assert torch.equal(before, after)


@pytest.mark.parametrize('failure', ['one_foot_outside', 'one_foot_low', 'lying', 'fallen_object'])
def test_ascent_rejects_false_positive_cases(failure):
    objects = torch.zeros(1, 1, 13)
    objects[..., 6] = 1
    sizes = torch.ones(1, 1, 3)
    feet = torch.tensor([[[[.1, -.1, .53], [.1, .1, .53]]]])
    roots = torch.tensor([[[0., 0., 1.2]]])
    forces = torch.zeros_like(feet)
    forces[..., 2] = 100
    if failure == 'one_foot_outside':
        feet[..., 0, 0] = .501
    elif failure == 'one_foot_low':
        feet[..., 0, 2] = .42
    elif failure == 'lying':
        roots[..., 2] = .6
    else:
        # A tipped-over object's original top is not a horizontal support.
        objects[..., 3] = 2 ** -.5
        objects[..., 6] = 2 ** -.5
    geometry, supported = EVAL.climb_ascent_success(feet, roots, objects, sizes, forces)
    assert not geometry.item() and not supported.item()


def test_training_size_range_is_inclusive_and_checks_all_axes():
    bounds = [[.4, .8], [.4, .8], [.25, .5]]
    assert EVAL.size_in_range([.4, .8, .5], bounds)
    assert EVAL.size_in_range([.8, .4, .25], bounds)
    assert not EVAL.size_in_range([.4, .3, .4], bounds)
    assert not EVAL.size_in_range([.5, .5, .501], bounds)
    assert not EVAL.size_in_range([.81, .5, .4], bounds)
