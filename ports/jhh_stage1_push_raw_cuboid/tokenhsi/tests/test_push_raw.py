from copy import deepcopy
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


def test_random_boxes_are_cubes_and_mass_is_sampled_independently():
    torch.manual_seed(42)
    sizes, masses = sample_box_properties(4096, ENV['stage1Push']['box'], 'cpu')
    torch.testing.assert_close(sizes[:, 0], sizes[:, 1])
    torch.testing.assert_close(sizes[:, 0], sizes[:, 2])
    assert sizes.min() >= 1.2 and sizes.max() <= 1.7
    assert masses.min() >= 25 and masses.max() <= 30
    assert sizes[:, 0].min() < 1.201 and sizes[:, 0].max() > 1.699
    assert abs(torch.corrcoef(torch.stack((sizes[:, 0], masses)))[0, 1]) < .05
    state = torch.random.get_rng_state()
    fixed, mass = sample_box_properties(3, OLD['stage1Push']['box'], 'cpu')
    torch.testing.assert_close(fixed, torch.full((3, 3), 1.3))
    torch.testing.assert_close(mass, torch.full((3,), 25.))
    assert torch.equal(state, torch.random.get_rng_state())


def test_contact_height_is_104cm_for_all_cube_sizes():
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
    assert ENV['stage1Push']['contact_height'] == 1.04
    assert len(load_bank(ENV['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')['candidates']['push']) == 11
    with pytest.raises(ValueError, match='source teacher'):
        check_navigation_transfer({'stage1_push_contract': contract(OLD)}, ENV)
    old = deepcopy(OLD)
    old['stage1Push']['source_checkpoint'] = ENV['stage1Push']['source_checkpoint']
    source = {'stage1_push_contract': contract(old)}
    check_navigation_transfer(source, ENV)
    with pytest.raises(ValueError, match='contract mismatch'):
        check_contract(source, ENV)
    changed = deepcopy(ENV)
    changed['stage1Push']['box']['friction'] = .3
    with pytest.raises(ValueError, match='physics'):
        check_navigation_transfer(source, changed)
    check_contract({'stage1_push_contract':contract(ENV)}, ENV)


def test_raw_training_cannot_silently_use_corrected_pose():
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    task = object.__new__(HumanoidMAStage1Push)
    args = SimpleNamespace(eval=False, push_rsi_mode='corrected')
    with pytest.raises(ValueError, match='requires evaluation'):
        task.__init__({'env':deepcopy(ENV), 'args':args})
