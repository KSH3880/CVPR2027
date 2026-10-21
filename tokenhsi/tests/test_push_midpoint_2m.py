"""Height-free hand geometry and physics-tick net movement credit regressions."""
from copy import deepcopy
from pathlib import Path

from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml

from env.tasks.multi_agent.open_push_reward import (
    OpenPushRuntime, push_reward, push_hand_contact, push_midpoint_xy_distance,
    push_rear_xy_distance, push_hand_targets,
)
from utils.open_push_spec import validate_env

ROOT = Path(__file__).resolve().parents[2]


def scene(n=2):
    env = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_midpoint_2m.yaml').read_text())['env']
    boxes = torch.zeros(n, 2, 13)
    boxes[..., 2] = .65
    boxes[..., 6] = 1
    roots = boxes[..., :3].clone()
    hands = push_hand_targets(boxes, env['pushBox']['size'], env['relationReward']['push'])
    goals = roots.clone()
    goals[..., 1] += 2
    runtime = OpenPushRuntime(n, env['relationGraph'], env['relationReward'], env['doorHold'],
                              1/30, 'cpu', env['pushBox']['size'], env['manipulationContact'])
    runtime.reset(torch.arange(n), torch.ones(n, 2, dtype=torch.long), boxes)
    runtime.initialize_manipulation(torch.arange(n), boxes, goals, torch.zeros(n, 2))
    return env, runtime, roots, hands, boxes, goals


def tick(rt, roots, hands, boxes, goals, valid=True, body=False):
    rt.push_contact[:] = valid
    rt.clean_push_contact[:] = valid
    rt.body_contact[:] = body
    rt.collect_manipulation(boxes, goals, torch.zeros_like(rt.elapsed),
                            torch.zeros_like(rt.contact), rt.clean_push_contact)
    return rt.step(roots, hands, torch.zeros_like(roots), torch.zeros_like(rt.elapsed), boxes, goals)


def test_config_and_height_spacing_invariance():
    env, rt, roots, hands, boxes, goals = scene()
    validate_env(env)
    eligible = torch.zeros_like(rt.contact)
    def reward():
        return push_reward(roots, hands, boxes, goals, rt.start, eligible,
                           env['relationReward'], env['pushBox']['size'])['state']
    initial = reward()
    hands[..., 0, 2] = .15
    hands[..., 1, 2] = 2.0
    hands[..., 0, 0] = .6
    hands[..., 1, 0] = -.6
    torch.testing.assert_close(reward(), initial)
    hands[..., 0] += .2
    assert (reward() < initial).all()


def test_xy_segment_contact_requires_both_hand_and_box_reactions():
    env, rt, roots, hands, boxes, goals = scene()
    spec = env['relationReward']['push']
    hands[..., 0, 0], hands[..., 1, 0] = .6, -.6
    hands[..., 0, 2], hands[..., 1, 2] = .1, 2.5
    forces = torch.zeros_like(hands)
    forces[..., 1] = -.5
    box_forces = torch.zeros_like(roots)
    box_forces[..., 1] = .5
    def contact():
        return push_hand_contact(hands, forces, boxes, box_forces, env['pushBox']['size'], spec)
    assert contact().all()
    hands[..., 0, 0] = .84  # 19 cm beyond the rear segment endpoint
    assert not contact().any()
    hands[..., 0, 0] = .6
    forces[..., 1, 1] = 0
    assert not contact().any()
    forces[..., 1, 1] = -.5
    box_forces[..., 1] = 0
    assert not contact().any()


def test_yaw_follows_box_and_pitch_does_not_move_xy_line():
    env, rt, roots, hands, boxes, goals = scene()
    yaw, pitch = .8, .2
    c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    local = hands[..., :2] - boxes[..., :2][:, :, None]
    hands[..., 0] = c*local[..., 0] - s*local[..., 1]
    hands[..., 1] = s*local[..., 0] + c*local[..., 1]
    # Quaternion for yaw(Z) * pitch(Y).
    boxes[..., 3] = -torch.sin(torch.tensor(yaw/2))*torch.sin(torch.tensor(pitch/2))
    boxes[..., 4] = torch.cos(torch.tensor(yaw/2))*torch.sin(torch.tensor(pitch/2))
    boxes[..., 5] = torch.sin(torch.tensor(yaw/2))*torch.cos(torch.tensor(pitch/2))
    boxes[..., 6] = torch.cos(torch.tensor(yaw/2))*torch.cos(torch.tensor(pitch/2))
    torch.testing.assert_close(push_midpoint_xy_distance(hands, boxes, [1.3]*3),
                               torch.zeros(2, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(push_rear_xy_distance(hands, boxes, [1.3]*3),
                               torch.zeros(2, 2, 2), atol=1e-6, rtol=0)


def test_same_tick_credit_cannot_reclaim_invalid_movement():
    env, rt, roots, hands, boxes, goals = scene()
    boxes[..., 1] = 1.0
    tick(rt, roots, hands, boxes, goals, False, True)
    tick(rt, roots, hands, boxes, goals, True)
    assert not rt.push_hand_distance.any()
    boxes[..., 1] = 1.2
    tick(rt, roots, hands, boxes, goals, True, True)
    torch.testing.assert_close(rt.push_hand_distance, torch.full((2, 2), .2))
    assert not rt.push_eligible.any()
    boxes[..., 1] = 2.0
    assert not tick(rt, roots, hands, boxes, goals, False)['success'].any()


def test_backward_motion_deducts_without_contact_and_caps_oscillations():
    env, rt, roots, hands, boxes, goals = scene()
    boxes[..., 1] = .3
    tick(rt, roots, hands, boxes, goals)
    boxes[..., 1] = .1
    tick(rt, roots, hands, boxes, goals, False)
    torch.testing.assert_close(rt.push_hand_distance, torch.full((2, 2), .1), atol=1e-6, rtol=0)
    for _ in range(4):
        boxes[..., 1] = .3
        tick(rt, roots, hands, boxes, goals)
        boxes[..., 1] = .1
        tick(rt, roots, hands, boxes, goals, False)
    torch.testing.assert_close(rt.push_hand_distance, torch.full((2, 2), .1), atol=1e-6, rtol=0)


def test_rsi_excluded_full_required_distance_and_release_current_success():
    env, rt, roots, hands, boxes, goals = scene()
    boxes[..., 1] = 1.7
    rt.initialize_manipulation(torch.arange(2), boxes, goals, torch.zeros(2, 2))
    assert not rt.push_hand_distance.any() and not rt.push_eligible.any()
    boxes[..., 1] = 1.8
    assert not tick(rt, roots, hands, boxes, goals)['success'].any()
    boxes[..., 1] = 1.9
    assert tick(rt, roots, hands, boxes, goals, True, True)['success'].all()
    torch.testing.assert_close(rt.push_hand_distance, torch.full((2, 2), .2))
    hands[:] = 20
    released = tick(rt, roots, hands, boxes, goals, False, True)
    assert released['success'].all() and (released['state'] < .4).all()
    assert (released['progress'] == .2).all()
    boxes[..., 7] = .3
    assert not tick(rt, roots, hands, boxes, goals, False)['success'].any()
    boxes[..., 7] = 0
    assert tick(rt, roots, hands, boxes, goals, False)['success'].all()
    boxes[..., 1] = 1.7
    assert not tick(rt, roots, hands, boxes, goals, False)['success'].any()
    assert not rt.push_hand_distance.any()


def test_minimum_credit_and_partial_reset():
    env, rt, roots, hands, boxes, goals = scene()
    boxes[..., 1] = 1.9
    rt.initialize_manipulation(torch.arange(2), boxes, goals, torch.zeros(2, 2))
    assert not tick(rt, roots, hands, boxes, goals)['success'].any()
    boxes[..., 1] = 1.96
    assert tick(rt, roots, hands, boxes, goals)['success'].all()
    before = rt.push_hand_distance[1].clone()
    rt.reset(torch.tensor([0]), torch.ones(1, 2, dtype=torch.long), boxes[:1])
    assert not rt.push_hand_distance[0].any() and not rt.push_previous_distance[0].any()
    torch.testing.assert_close(rt.push_hand_distance[1], before)


def test_unknown_mode_is_rejected():
    env, *_ = scene()
    env = deepcopy(env)
    env['relationReward']['push']['control_mode'] = 'typo'
    with pytest.raises(ValueError, match='mode'):
        validate_env(env)
