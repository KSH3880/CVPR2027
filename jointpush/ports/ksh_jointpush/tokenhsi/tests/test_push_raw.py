from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from env.tasks.multi_agent.push_reward import contact_geometry
from env.tasks.multi_agent.push_rsi import face_retreat
from utils.push_navigation import sample_box_properties
from utils.stage1_push import check_contract, check_navigation_transfer, contract, load_bank

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw.yaml').read_text())['env']
OLD = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360.yaml').read_text())['env']


def test_random_boxes_are_cubes_and_mass_increases_linearly_with_size():
    torch.manual_seed(42)
    sizes, masses = sample_box_properties(4096, ENV['stage1Push']['box'], 'cpu')
    torch.testing.assert_close(sizes[:, 0], sizes[:, 1])
    torch.testing.assert_close(sizes[:, 0], sizes[:, 2])
    assert sizes.min() >= 1.2 and sizes.max() <= 1.7
    assert masses.min() >= 12 and masses.max() <= 20
    assert sizes[:, 0].min() < 1.201 and sizes[:, 0].max() > 1.699
    torch.testing.assert_close(masses, 12+8*(sizes[:, 0]-1.2)/.5)
    order = sizes[:, 0].argsort()
    assert (masses[order].diff() >= 0).all()
    state = torch.random.get_rng_state()
    fixed, mass = sample_box_properties(3, OLD['stage1Push']['box'], 'cpu')
    torch.testing.assert_close(fixed, torch.full((3, 3), 1.3))
    torch.testing.assert_close(mass, torch.full((3,), 25.))
    assert torch.equal(state, torch.random.get_rng_state())


def test_legacy_raw_boxes_keep_independent_mass_sampling():
    options = json.loads((ROOT/'tokenhsi/data/push_raw_cube120to170_mass25to30_bank.json').read_text())['contract']['push_box']
    torch.manual_seed(42)
    sizes, masses = sample_box_properties(4096, options, 'cpu')
    assert masses.min() >= 25 and masses.max() <= 30
    assert abs(torch.corrcoef(torch.stack((sizes[:, 0], masses)))[0, 1]) < .05


def test_legacy_contact_height_is_104cm_for_all_cube_sizes():
    edges = torch.linspace(1.2, 1.7, 51)
    box = torch.zeros(51, 1, 13)
    box[:, 0, 2], box[..., 6] = edges/2, 1
    size = edges[:, None, None].expand(-1, 1, 3)
    hands = torch.zeros(51, 1, 2, 3)
    hands[..., 0, 0], hands[..., 1, 0] = -.1, .1
    hands[..., 1] = -edges[:, None, None]/2
    hands[..., 2] = 1.04
    force = torch.zeros_like(hands)
    force[..., 1] = -1
    box_force = torch.tensor([0., 2., 0.]).expand(51, 1, -1)
    phi, hit, point, _ = contact_geometry(hands, box, size, force, box_force, midpoint_only=True, contact_height=1.04)
    torch.testing.assert_close(point[..., 2], torch.full((51, 1), 1.04))
    torch.testing.assert_close(phi, torch.ones_like(phi))
    assert hit.all()


@pytest.fixture
def face_scene():
    box = torch.zeros(1, 1, 13)
    box[..., 2], box[..., 6] = .65, 1
    size = torch.full((1, 1, 3), 1.3)
    hands = torch.tensor([[[[-.4, -.65, .1], [.4, -.65, 2.5]]]])
    force = torch.zeros_like(hands)
    force[..., 1] = -1
    box_force = torch.tensor([[[0., 2., 0.]]])
    return hands, box, size, force, box_force


@pytest.mark.parametrize('height', [-4., 0., 1.04, 9.])
def test_face_xy_ignores_hand_midpoint_and_height(face_scene, height):
    hands, box, size, force, box_force = face_scene
    hands[..., 0] += .2
    hands[..., 2] = height
    phi, hit, _, _ = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    torch.testing.assert_close(phi, torch.ones_like(phi))
    assert hit.all()


def test_face_xy_uses_the_farther_hand_and_finite_face_edges(face_scene):
    hands, box, size, force, box_force = face_scene
    hands[..., 1, 0], hands[..., 1, 1] = .95, -1.05
    phi, hit, _, _ = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    torch.testing.assert_close(phi, torch.full_like(phi, torch.exp(torch.tensor(-2.5))))
    assert not hit.any()
    hands[..., 1, 0], hands[..., 1, 1] = .75, -.65
    phi, hit, _, _ = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    torch.testing.assert_close(phi, torch.full_like(phi, torch.exp(torch.tensor(-.1))))
    assert hit.all()


def test_face_xy_keeps_one_common_designated_face(face_scene):
    hands, box, size, force, box_force = face_scene
    hands[..., 0, 0], hands[..., 1, 0] = -.65, .65
    hands[..., 1] = 0
    phi, hit, _, _ = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    torch.testing.assert_close(phi, torch.full_like(phi, torch.exp(torch.tensor(-10*.65**2))))
    assert not hit.any()


@pytest.mark.parametrize('missing', ['left', 'right', 'box', 'reverse_left', 'reverse_box'])
def test_face_xy_still_requires_each_contact_force(face_scene, missing):
    hands, box, size, force, box_force = face_scene
    if missing in ('left', 'right'):
        force[..., 0 if missing == 'left' else 1, :] *= .49
    elif missing == 'box':
        box_force *= .24
    elif missing == 'reverse_left':
        force[..., 0, :] *= -1
    else:
        box_force *= -1
    phi, hit, _, _ = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    torch.testing.assert_close(phi, torch.ones_like(phi))
    assert not hit.any()


def test_face_xy_tracks_box_yaw_and_translation(face_scene):
    hands, box, size, force, box_force = face_scene
    hands[..., 1, 0] = .75
    before = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = torch.tensor([12., -7., 3.])
    hands = hands@rotation.T+shift
    box[..., :3] = box[..., :3]@rotation.T+shift
    box[..., 5:7] = 2**-.5
    after = contact_geometry(hands, box, size, force@rotation.T, box_force@rotation.T, face_xy=True)
    torch.testing.assert_close(after[0], before[0])
    assert torch.equal(after[1], before[1])


@pytest.mark.parametrize('axis', [0, 1])
def test_face_xy_ignores_hand_z_with_a_tilted_box(face_scene, axis):
    from isaacgym.torch_utils import quat_rotate
    hands, box, size, force, box_force = face_scene
    box[..., axis], box[..., 6] = torch.sin(torch.tensor(.35)), torch.cos(torch.tensor(.35))
    local = torch.tensor([[[[.1, -.65, -.1], [.5, -.65, .3]]]])
    hands = quat_rotate(box[..., None, 3:7].expand(1, 1, 2, 4).reshape(-1, 4), local.reshape(-1, 3)).reshape_as(local)+box[..., None, :3]
    normal = quat_rotate(box[..., 3:7].reshape(-1, 4), torch.tensor([[0., 1., 0.]])).reshape(1, 1, 3)
    force = -normal[..., None, :].expand_as(hands)
    box_force = 2*normal
    before = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    hands[..., 0, 2], hands[..., 1, 2] = -100., 100.
    after = contact_geometry(hands, box, size, force, box_force, face_xy=True)
    torch.testing.assert_close(after[0], torch.ones_like(after[0]))
    assert after[1].all()
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])


def test_raw_environment_uses_face_xy_geometry(face_scene):
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    hands, box, size, force, box_force = face_scene
    task = SimpleNamespace(num_agents=1, num_envs=1, device='cpu', _box_states=box, _box_size=size,
        _rigid_body_pos=hands, _contact_forces=force, _key_body_ids=[0, 1], _ontop_box_contact=box_force,
        _push_options=ENV['stage1Push'], _tar_pos=box[..., :3].clone(),
        _humanoid_root_states=torch.zeros_like(box))
    result = HumanoidMAStage1Push._push_state(task)
    torch.testing.assert_close(result['phi_contact'], torch.ones(1, 1))
    assert result['hit'].all()


def test_retreat_moves_whole_pose_only_as_far_as_needed():
    p = torch.tensor([[[0., -.03, 1.04], [0., -.5, 1.3]], [[0., -.2, 1.04], [0., -.5, 1.3]]])
    q = torch.zeros(2, 2, 4)
    q[..., 3] = 1
    shapes = (torch.tensor([0, 1]), torch.zeros(2, 3), torch.tensor([.04, .095]))
    normal = torch.tensor([[0., 1., 0.]]).expand(2, -1)
    face = torch.zeros(2, 3)
    retreat = face_retreat(p, q, face, normal, shapes, .005)
    torch.testing.assert_close(retreat, torch.tensor([.015, 0.]))
    translated = p-retreat[:, None, None]*normal[:, None]
    torch.testing.assert_close(translated[..., 2], p[..., 2])
    torch.testing.assert_close(face_retreat(translated, q, face, normal, shapes, .005), torch.zeros(2), atol=1e-7, rtol=0)


def test_new_settings_require_explicit_transfer_and_preserve_old_tasks():
    for key in OLD:
        if key != 'stage1Push':
            assert ENV[key] == OLD[key]
    assert ENV['stage1Push']['rsi']['mode'] == 'raw'
    assert ENV['stage1Push']['contact_mode'] == 'face_xy'
    assert 'contact_height' not in ENV['stage1Push']
    assert len(load_bank(ENV['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')['candidates']['push']) == 11
    previous = deepcopy(OLD)
    previous['stage1Push']['source_checkpoint'] = ENV['stage1Push']['source_checkpoint']
    source = {'stage1_push_contract': contract(previous)}
    with pytest.raises(ValueError, match='physics'):
        check_navigation_transfer(source, ENV)
    with pytest.raises(ValueError, match='source teacher'):
        check_navigation_transfer({'stage1_push_contract': contract(OLD)}, ENV)
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract(source, ENV)
    legacy = deepcopy(ENV)
    legacy['stage1Push'].pop('contact_mode')
    legacy['stage1Push'].update(contact_midpoint_only=True, contact_height=1.04)
    legacy_checkpoint = {'stage1_push_contract': contract(legacy)}
    check_navigation_transfer(legacy_checkpoint, ENV)
    check_contract(legacy_checkpoint, legacy)
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract(legacy_checkpoint, ENV)
    changed = deepcopy(ENV)
    changed['stage1Push']['box']['friction'] = .3
    with pytest.raises(ValueError, match='physics'):
        check_navigation_transfer(legacy_checkpoint, changed)
    check_contract({'stage1_push_contract':contract(ENV)}, ENV)


def test_raw_training_cannot_silently_use_corrected_pose():
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    task = object.__new__(HumanoidMAStage1Push)
    args = SimpleNamespace(eval=False, push_rsi_mode='corrected')
    with pytest.raises(ValueError, match='requires evaluation'):
        task.__init__({'env':deepcopy(ENV), 'args':args})
