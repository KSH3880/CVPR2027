from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
import torch
import yaml

from utils.stage1_jointpush import box_densities, sample_assets, validate_env
from utils.jointpush_scene import goal_angles_valid, paths_clear, sample_layout
from utils.jointpush_rsi import PushBank

ROOT = Path(__file__).resolve().parents[2]
NAME = 'stage1_jointpush_density3_cone90'
ENV = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{NAME}.yaml').read_text())['env']
BASE = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml').read_text())['env']


def test_only_density_cone_and_bank_reference_change():
    validate_env(ENV)
    restored = deepcopy(ENV)
    restored['jointPush']['box'].pop('density')
    restored['jointPush']['box']['mass'] = 20.
    restored['jointPush']['scene']['goal_angle_degrees'] = 360.
    restored['jointPush']['rsi'].pop('reference_mass')
    assert restored == BASE
    train = yaml.safe_load((ROOT/f'tokenhsi/data/cfg/train/rlg/{NAME}.yaml').read_text())
    baseline = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage1_jointpush.yaml').read_text())
    assert train['params']['config']['name'] == 'Stage1JointPushDensity3Cone90'
    train['params']['config']['name'] = baseline['params']['config']['name']
    assert train == baseline


def test_mass_for_all_profiles_and_other_objects():
    profiles = torch.tensor(ENV['jointPush']['box']['profiles'])
    sizes = profiles[:, None].expand(-1, 4, -1).clone()
    sizes[:, 1:] = .4
    push = torch.tensor([True, False]).expand(len(sizes), -1)
    rho = box_densities(sizes, push, ENV['jointPush']['box'])
    mass = rho*sizes.prod(-1)
    torch.testing.assert_close(mass[:, 0], profiles.prod(-1)*3.)
    torch.testing.assert_close(rho[:, 1:], torch.full((167, 3), 100.))
    assert mass[:, 0].min().item() == pytest.approx(11.136)
    assert mass[:, 0].mean().item() == pytest.approx(18.001428)
    assert mass[:, 0].max().item() == pytest.approx(24.684)
    old_mass = box_densities(sizes, push, BASE['jointPush']['box'])*sizes.prod(-1)
    torch.testing.assert_close(old_mass[:, 0], torch.full((167,), 20.))


def test_cone_boundary_rotated_box_and_inactive_slots():
    angles = torch.tensor([-45.1, -45., 0., 45., 45.1, 180.])*math.pi/180
    yaw = .73
    boxes = torch.zeros(len(angles), 4, 13)
    boxes[..., 5], boxes[..., 6] = math.sin(yaw/2), math.cos(yaw/2)
    goals = torch.zeros(len(angles), 2, 3)
    goals[:, 0, 0], goals[:, 0, 1] = -(angles+yaw).sin()*3, (angles+yaw).cos()*3
    goals[:, 1, 0], goals[:, 1, 1] = math.sin(yaw)*3, -math.cos(yaw)*3
    push = torch.tensor([True, False]).expand(len(angles), -1)
    valid = goal_angles_valid(boxes, goals, push, ENV['jointPush']['scene'])
    assert valid.tolist() == [False, True, True, True, False, False]
    assert goal_angles_valid(boxes, goals, push, BASE['jointPush']['scene']).all()


def test_rejected_cone_resamples_box_and_goal(monkeypatch):
    import utils.jointpush_scene as scene
    calls = []

    def radial(center, bounds):
        offsets = [(2., 0.), (0., -2.), (3., 0.), (0., 2.)]
        result = center+center.new_tensor(offsets[len(calls)])
        calls.append(result.clone())
        return result

    monkeypatch.setattr(scene, 'radial_xy', radial)
    monkeypatch.setattr(scene.torch, 'rand', lambda *shape, **kw: torch.zeros(*shape, **kw))
    roots = torch.tensor([[[0., 0., 0.], [50., 0., 0.]]])
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 6] = 1
    boxes[:, 1, 0] = 50.
    sizes = torch.tensor([2., 2., 1.5]).expand(1, 4, 3)
    result, goals = sample_layout(roots, boxes, sizes, torch.tensor([[True, False]]), ENV['jointPush']['scene'])
    assert len(calls) == 4
    torch.testing.assert_close(result[0, 0, :2], torch.tensor([3., 0.]))
    torch.testing.assert_close(goals[0, 0, :2], torch.tensor([3., 2.]))
    torch.testing.assert_close(result[:, 1:], boxes[:, 1:])


@pytest.mark.parametrize('seed', [6, 42])
def test_full_batch_cone_distances_and_paths(seed):
    torch.manual_seed(seed)
    sizes, weights = sample_assets(2048, ENV, 'cpu', 'push')
    roots = torch.zeros(2048, 2, 3)
    roots[:, 0, 0], roots[:, 1, 0] = 2.5, -2.5
    boxes = torch.zeros(2048, 4, 13)
    boxes[..., 6] = 1
    push = weights[..., 5].bool()
    push[:256, 1] = False
    boxes[:256, 1, 0] = -4.
    boxes[:256, 1, 1] = 4.
    original = boxes.clone()
    spec = ENV['jointPush']['scene']
    boxes, goals = sample_layout(roots, boxes, sizes, push, spec)
    q = boxes[:, :2, 3:7]
    forward = torch.stack((-2*q[..., 2]*q[..., 3], 1-2*q[..., 2].square()), -1)
    delta = goals[..., :2]-boxes[:, :2, :2]
    angle = torch.atan2(forward[..., 0]*delta[..., 1]-forward[..., 1]*delta[..., 0],
                        (forward*delta).sum(-1))*180/math.pi
    assert (angle[push].abs() <= 45.001).all()
    assert angle[push].min() < -40 and angle[push].max() > 40
    for delta in (boxes[:, :2, :2]-roots[..., :2], goals[..., :2]-boxes[:, :2, :2]):
        radius = delta.norm(dim=-1)[push]
        assert ((radius >= 2.-1e-5) & (radius <= 10.+1e-5)).all()
    assert paths_clear(boxes, sizes, goals, roots, push, spec).all()
    torch.testing.assert_close(boxes[:, :2][~push], original[:, :2][~push])
    torch.testing.assert_close(boxes[:, 2:], original[:, 2:])


def test_reference_bank_hashes_and_mass_contract(monkeypatch):
    monkeypatch.setattr('utils.jointpush_rsi.geometry', lambda *args: None)
    motions = yaml.safe_load((ROOT/ENV['motion_file']).read_text())['motions']['push']
    lib = SimpleNamespace(gts=torch.zeros(1),
        _motions=[SimpleNamespace(skeleton_tree=SimpleNamespace(node_names=[]))],
        _motion_weights=torch.tensor([r['weight'] for r in motions]),
        _motion_lengths=torch.full((len(motions),), 10000.))
    base = PushBank(BASE, {'push': lib})
    variant = PushBank(ENV, {'push': lib})
    torch.testing.assert_close(variant.ids, base.ids)
    torch.testing.assert_close(variant.times, base.times)
    torch.testing.assert_close(variant.weights, base.weights)
    torch.testing.assert_close(variant.late, base.late)
    bad = deepcopy(ENV)
    bad['jointPush']['rsi']['reference_mass'] = 18.
    with pytest.raises(ValueError, match='physics mismatch'):
        PushBank(bad, {'push': lib})
    bad['jointPush']['rsi'].pop('reference_mass')
    with pytest.raises(ValueError, match='declare reuse'):
        validate_env(bad)
