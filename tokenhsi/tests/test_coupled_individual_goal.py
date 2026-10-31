from copy import deepcopy

import pytest
import torch

from utils.coupled_push_contact import group_face_geometry, individual_goal_gate
from utils.coupled_push_spec import compose_graph, region_states, checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.task_role_spec import task_graph_packet
from utils.evaluation_protocol import spatial_success, coupled_current_success
from env.tasks.multi_agent.coupled_push_reward import CoupledPushRuntime


NAME = 'stage2_coupled_push_individual_goal'


def test_all_solo_source_caps_preserve_sit_and_default_sampler():
    from utils.stage1_sampling import sample_source_xy
    templates = torch.arange(1, 5).repeat_interleave(1024)
    roots = torch.zeros(len(templates), 2)
    torch.manual_seed(29)
    original = sample_source_xy(roots, templates)
    torch.manual_seed(29)
    limited = sample_source_xy(roots, templates, max_distance=5.)
    distances = limited.norm(dim=-1)
    assert ((distances >= 1.-1e-6) & (distances <= 5.+1e-6)).all()
    torch.testing.assert_close(limited[templates == 1], original[templates == 1], rtol=0, atol=0)
    assert (original[templates != 1].norm(dim=-1) > 5.).any()


def scene(count):
    boxes = torch.zeros(1, count, 13)
    boxes[..., 2], boxes[..., 6] = .5, 1.
    sizes = torch.tensor([4., 2., 1.]).expand(1, count, 3).clone()
    groups = torch.zeros(1, count, dtype=torch.long)
    members = torch.full_like(groups, count)
    regions = torch.arange(count)[None]
    return boxes, sizes, groups, members, regions


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('direction,face', [([1., 0.], 0), ([-1., 0.], 1), ([0., 1.], 2), ([0., -1.], 3)])
def test_all_four_real_faces_and_group_partition(count, direction, face):
    boxes, sizes, groups, members, regions = scene(count)
    normal = torch.tensor(direction)
    tangent = torch.tensor([0., 1.] if face < 2 else [1., 0.])
    half, width = (2., 1.) if face < 2 else (1., 2.)
    expected = -normal*half+tangent*((2*regions[0]+1)*width/count-width)[:, None]
    hands = torch.zeros(1, count, 2, 3)
    hands[..., :2] = expected[None, :, None]-normal*.01
    hands[..., 2] = .5
    goals = boxes[..., :3].clone()
    goals[..., :2] += normal*4
    result = group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)
    assert (result['face'] == face).all()
    torch.testing.assert_close(result['slot_center'][0], expected)
    torch.testing.assert_close(result['distance'], torch.full((1, count), .01), atol=1e-6, rtol=0)
    assert result['on_face'].all()
    torch.testing.assert_close(result['slot_half'], torch.full((1, count), width/count))


def test_two_candidate_faces_choose_group_worst_hand_and_preserve_agent_order():
    boxes, sizes, groups, members, regions = scene(4)
    goals = boxes[..., :3].clone(); goals[..., :2] = 4.
    hands = torch.zeros(1, 4, 2, 3)
    hands[..., 0] = ((regions+.5)-2)[..., None]
    hands[..., 1], hands[..., 2] = -1., .5
    result = group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)
    assert (result['face'] == 2).all()
    perm = torch.tensor([2, 0, 3, 1])
    permuted = group_face_geometry(hands[:, perm], boxes[:, perm], sizes[:, perm], goals[:, perm],
                                  groups[:, perm], members[:, perm], regions[:, perm])
    torch.testing.assert_close(result['slot_center'][:, perm], permuted['slot_center'])
    hands[..., 0] = -2.
    hands[..., 1] = ((regions+.5)*.5-1)[..., None]
    assert (group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)['face'] == 0).all()


def test_overshoot_switches_contact_and_gate_together():
    boxes, sizes, groups, members, regions = scene(2)
    hands = torch.zeros(1, 2, 2, 3)
    hands[..., 0] = torch.tensor([-1., 1.])[None, :, None]
    hands[..., 1], hands[..., 2] = -1., .5
    goals = boxes[..., :3].clone(); goals[..., 1] = -1.
    result = group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)
    assert (result['face'] == 3).all()
    assert (result['distance'] == 2.).all() and not result['on_face'].any()
    hands[..., 1] = 1.
    result = group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)
    assert result['on_face'].all() and (result['distance'] == 0).all()
    tokens, _ = region_states(boxes, sizes, members, regions)
    assert individual_goal_gate(result['on_face'], tokens, tokens[..., :3]+10, groups).all()


def test_groups_choose_independently_and_geometry_rotates_with_box():
    boxes, sizes, groups, members, regions = scene(4)
    groups[:] = torch.tensor([0, 0, 2, 2])
    members[:] = 2
    regions[:] = torch.tensor([0, 1, 0, 1])
    goals = boxes[..., :3].clone(); goals[:, :2, 1] = 4.; goals[:, 2:, 1] = -4.
    hands = torch.zeros(1, 4, 2, 3)
    hands[..., 0] = torch.tensor([-1., 1., -1., 1.])[None, :, None]
    hands[:, :2, :, 1], hands[:, 2:, :, 1], hands[..., 2] = -1., 1., .5
    expected = group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)
    assert expected['face'].tolist() == [[2, 2, 3, 3]]
    for values in (hands, goals):
        xy = values[..., :2].clone()
        values[..., 0], values[..., 1] = -xy[..., 1], xy[..., 0]
    boxes[..., 5], boxes[..., 6] = 2**-.5, 2**-.5
    rotated = group_face_geometry(hands, boxes, sizes, goals, groups, members, regions)
    assert torch.equal(expected['face'], rotated['face'])
    torch.testing.assert_close(rotated['distance'], expected['distance'], atol=1e-6, rtol=0)
    assert rotated['on_face'].all()


def test_individual_arrival_saturates_both_edges_then_releases_without_latching():
    boxes, sizes, groups, members, regions = scene(2)
    tokens, _ = region_states(boxes, sizes, members, regions)
    goals = tokens[..., :3].clone(); goals[:, 1, 1] += 2.
    on_face = torch.zeros_like(groups, dtype=torch.bool)
    gate = individual_goal_gate(on_face, tokens, goals, groups)
    assert gate.tolist() == [[True, False]]
    graph = compose_graph(torch.full_like(groups, 5), groups, False, individual_goals=True)
    cfg = native_config(NAME)['env']
    runtime = CoupledPushRuntime(1, graph, cfg['relationReward'], 'cpu', contact_weights=[.2]*3)
    runtime.gate.copy_(gate)
    phi = torch.where(graph.edge_relation == 7, 1., .25)
    progress = torch.full_like(phi, .5)
    zero = torch.zeros_like(phi)
    result = runtime.step(phi, progress, zero, zero, zero)
    torch.testing.assert_close(result['local_task_reward'], torch.tensor([[1.2, .15]]))
    assert result['reward_saturated'][graph.edge_owner == 0].all()
    goals[:, 0, 1] += 1.
    runtime.gate.copy_(individual_goal_gate(on_face, tokens, goals, groups))
    result = runtime.step(phi, progress, zero, zero, zero)
    assert not result['reward_saturated'].any()
    torch.testing.assert_close(result['local_task_reward'], torch.tensor([[.15, .15]]))
    on_face[:] = True
    assert individual_goal_gate(on_face, tokens, goals, groups).all()


def test_goal_offsets_translation_rotation_and_scoring():
    boxes, sizes, groups, members, regions = scene(4)
    tokens, cells = region_states(boxes, sizes, members, regions)
    goals = tokens[..., :3]+torch.tensor([0., 4., 0.])
    graph = compose_graph(torch.full_like(groups, 5), groups, False, individual_goals=True)
    packet = task_graph_packet(graph, 1).reshape(1, 4, 5)
    assert packet[..., 4].tolist() == [[12, 13, 14, 15]]
    moved = boxes.clone(); moved[..., 1] += 4.
    target_tokens, _ = region_states(moved, sizes, members, regions)
    torch.testing.assert_close(target_tokens[..., :3], goals)
    score = spatial_success(packet, torch.zeros(1, 4, 3),
        torch.cat((target_tokens, target_tokens), 1), torch.cat((cells, cells), 1), goals)
    assert score.all()
    moved[..., 5], moved[..., 6] = 2**-.5, 2**-.5
    turned, _ = region_states(moved, sizes, members, regions)
    assert ((turned[..., :3]-goals).norm(dim=-1) > .2).all()
    a = torch.zeros(1, 4, 2, dtype=torch.bool); a[:, :2] = True
    b = ~a
    assert (a | b).all()
    assert not (coupled_current_success(a, groups) | coupled_current_success(b, groups)).any()


def test_new_config_isolated_and_rejects_old_checkpoint():
    cfg = native_config(NAME)
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    old = native_config('stage2_coupled_push_goal5')
    env = deepcopy(cfg['env'])
    assert 'solo_source_max_distance' not in env['stage2JointPush']
    assert 'solo_goal_max_distance' not in env['stage2JointPush']
    for key in ('goal_mode', 'contact_face', 'gate_alignment'):
        env['stage2JointPush'].pop(key)
    assert env['stage2JointPush']['coupled_probability'] == .5
    assert old['env']['stage2JointPush']['coupled_probability'] == .8
    env['stage2JointPush']['coupled_probability'] = .8
    assert env == old['env'] and cfg['sim'] == old['sim']
    for source, target in ((cfg, old), (old, cfg)):
        with pytest.raises(ValueError):
            check_checkpoint({'coupled_push_contract': checkpoint_contract(source['env'])}, target['env'])


def test_solo5_variant_only_caps_solo_distances_and_has_separate_contract():
    original = native_config(NAME)
    name = NAME+'_solo5'
    short = native_config(name)
    validate_experiment(name, short['env'], native_config(name, True))
    env = deepcopy(short['env'])
    for key in ('solo_source_max_distance', 'solo_goal_max_distance'):
        assert env['stage2JointPush'].pop(key) == 5.
    assert env == original['env'] and short['sim'] == original['sim']
    assert short['experiment']['transfer'] == original['experiment']['transfer']
    for source, target in ((short, original), (original, short)):
        with pytest.raises(ValueError):
            check_checkpoint({'coupled_push_contract': checkpoint_contract(source['env'])}, target['env'])
