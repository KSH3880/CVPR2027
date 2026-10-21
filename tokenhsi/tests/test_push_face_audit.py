"""Geometry classification and completed-episode masking without CUDA imports."""
import math
from utils.push_face_audit import target_region, hand_layout, PushFaceAudit, physical_layout, side_face_3d


BOX = [0., 0., .65, 0., 0., 0., 1., 0., .1, 0., 0., 0., 0.]
SIZE = [1.3, 1.3, 1.3]


def test_face_corner_and_rotation():
    assert target_region(BOX, [0, 2, .65], SIZE) == 'face_center'
    assert target_region(BOX, [2, 2, .65], SIZE) == 'corner'
    rotated = BOX.copy()
    rotated[5:7] = [math.sin(math.pi/8), math.cos(math.pi/8)]
    assert target_region(rotated, [0, 2, .65], SIZE) == 'corner'


def test_same_different_and_ambiguous_hands():
    assert hand_layout([[-.2, -.65], [.2, -.65]], BOX, SIZE) == 'same_face'
    assert hand_layout([[-.4, -.65], [-.65, -.4]], BOX, SIZE) == 'different_faces'
    assert hand_layout([[-.65, -.65], [-.65, -.4]], BOX, SIZE) == 'ambiguous_corner'
    assert hand_layout([[3., 3.], [-.65, -.4]], BOX, SIZE) == 'not_both_near'


def test_mask_and_success_timing():
    audit = PushFaceAudit([[BOX]], [[[0, 2, .65]]], SIZE, 1/30)
    args = ([[BOX]], [[[0, 2, .65]]], [[[[-.2, -.65], [.2, -.65]]]])
    audit.step(*args, [[False]], [True])
    audit.step(*args, [[True]], [True])
    audit.step(*args, [[False]], [False])
    row = audit.result()['agents'][0]
    assert row['steps'] == 2 and row['final_success'] and row['ever_success']
    assert row['first_success_seconds'] == 2/30
    assert row['layout_steps'] == {'same_face': 2}
    assert row['moving_unsuccessful_layout_steps'] == {'same_face': 1}


def test_tilted_obb_and_face_normal_force():
    box = BOX.copy()
    angle = math.pi/3
    box[3:7] = [math.sin(angle/2), 0., 0., math.cos(angle/2)]
    def world(p):
        x, y, z = p
        return [x, math.cos(angle)*y-math.sin(angle)*z,
                .65+math.sin(angle)*y+math.cos(angle)*z]
    hands = [world([-.2, -.65, .2]), world([.2, -.65, .2])]
    force = [0., -math.cos(angle), -math.sin(angle)]
    assert physical_layout(hands, box, SIZE, [force, force], [0., 1., 0.]) == ('same_face', True)
    assert physical_layout(hands, box, SIZE, [force, [0., 0., 0.]], [0., 1., 0.]) == ('same_face', False)
    assert side_face_3d(world([0., 0., .65]), box, SIZE) == 'top_bottom'
    assert side_face_3d(world([-.65, -.65, .2]), box, SIZE) == 'ambiguous'
    assert side_face_3d(world([0., -.65, 2.]), box, SIZE) is None


def test_force_proxy_and_pre_success_masking():
    audit = PushFaceAudit([[BOX]], [[[0, 2, .65]]], SIZE, 1/30)
    hands = [[[[ -.4, -.65, .8], [-.65, -.4, .8]]]]
    forces = [[[[0., -1., 0.], [-1., 0., 0.]]]]
    args = ([[BOX]], [[[0, 2, .65]]], hands)
    for current, alive in [(False, True), (True, True), (False, True), (False, False)]:
        audit.step(*args, [[current]], [alive], forces, [[[0., 1., 0.]]])
    row = audit.result()['agents'][0]
    assert row['force_proxy_layout_stats']['different_faces']['steps'] == 3
    assert row['before_first_success_moving_force_stats']['different_faces']['steps'] == 1
