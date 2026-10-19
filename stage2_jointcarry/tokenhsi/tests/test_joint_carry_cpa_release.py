from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from env.tasks.multi_agent.joint_carry_reward import joint_cpa_release, cpa_collision_penalty
from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import permute_graph
from utils.mixed_carry_spec import sample_graph
from utils.joint_carry_spec import validate_env
from utils.joint_carry_amp import validate_ablation
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from test_joint_carry_amp_ablation import configs


SUFFIX = 'locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding'


def test_config_only_changes_collision_and_checkpoint_contract():
    env, train = configs(SUFFIX)
    base, base_train = configs('locoamp_ca4_goal360_fullpickup_task_embedding')
    validate_env(env); validate_ablation(env, train); validate_relation_config(env['relationReward'])
    restored = deepcopy(env)
    assert restored['relationReward']['joint_carry']['collision'].pop('release_buffer') == 1.
    assert restored == base
    restored = deepcopy(train)
    restored['params']['config']['name'] = base_train['params']['config']['name']
    assert restored == base_train
    saved = {'relation_metadata':checkpoint_metadata(env['relationReward'])}
    check_checkpoint_metadata(saved, checkpoint_metadata(env['relationReward']))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata(saved, checkpoint_metadata(base['relationReward']))
    with pytest.raises(ValueError, match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(base['relationReward'])},
                                  checkpoint_metadata(env['relationReward']))
    for value in (True, -1., .6, float('nan')):
        bad = deepcopy(env['relationReward']);bad['joint_carry']['collision']['release_buffer'] = value
        with pytest.raises(ValueError, match='CPA release'):
            validate_relation_config(bad)


@pytest.mark.parametrize('preset', ['joint_carry_at', 'joint_carry_ontop'])
def test_both_roots_boundary_independent_and_no_latch(preset):
    env, _ = configs(SUFFIX)
    graph = sample_graph(5, env['relationGraph'], preset=preset,
                         joint_mask=torch.tensor([True, True, True, True, False]))
    sizes = torch.tensor([.52, .8, .4]).expand(5, 4, 3)
    objects = torch.zeros(5, 4, 13)
    roots = torch.zeros(5, 2, 3)
    radius = sizes[0, 0, :2].norm()/2+1.
    roots[:, 0, 0] = -radius;roots[:, 1, 0] = radius
    roots[1, 0, 0] = 0.;roots[1, 1, 0] = radius+.01
    roots[2, :, 0] *= 1.01
    roots[3, :, 0] *= .5
    expected = torch.tensor([True, False, False, True, False])
    release = joint_cpa_release(roots, objects, sizes, graph, 1.)
    assert torch.equal(release, expected)
    velocities = torch.zeros_like(roots);velocities[:, 0, 0] = .5;velocities[:, 1, 0] = -.5
    raw = -.5*cpa_collision_penalty(roots, velocities, .7, .99, 1/30)
    actual = raw.masked_fill(release[:, None], 0.)
    assert (raw < 0).all() and not actual[expected].any()
    torch.testing.assert_close(actual[~expected], raw[~expected], rtol=0, atol=0)
    roots[3, 1, 0] = radius+.1
    assert not joint_cpa_release(roots, objects, sizes, graph, 1.)[3]
    roots[3, 1, 0] = radius*.5
    assert joint_cpa_release(roots, objects, sizes, graph, 1.)[3]
    order = torch.rand(5, 4).argsort(-1)
    assert torch.equal(joint_cpa_release(roots, objects, sizes, permute_graph(graph, order), 1.), expected)


@pytest.mark.parametrize('preset', ['joint_carry_at', 'joint_carry_ontop'])
def test_binding_goal_task_human_and_coordinate_invariance(preset):
    env, _ = configs(SUFFIX)
    graph = sample_graph(2, env['relationGraph'], preset=preset)
    objects = torch.zeros(2, 4, 13);objects[:, 0, :2] = 100.;objects[:, 1, :2] = 5.
    sizes = torch.tensor([.52, .8, .4]).expand(2, 4, 3).clone()
    sizes[:, 1, :2] = .2
    roots = torch.zeros(2, 2, 3);roots[..., :2] = 5.
    roots[:, 0, 0] -= 1.2;roots[:, 1, 0] += 1.2
    graph.edge_src[graph.edge_src==2] = 3;graph.edge_dst[graph.edge_dst==2] = 3
    assert not joint_cpa_release(roots, objects, sizes, graph, 1.).any()
    sizes[:, 1, :2] = .8
    assert joint_cpa_release(roots, objects, sizes, graph, 1.).all()
    swapped = deepcopy(graph)
    primary = swapped.edge_relation == HOLDING
    swapped.edge_src[primary] = 1-swapped.edge_src[primary]
    swapped = replace(swapped, edge_owner=1-swapped.edge_owner)
    assert joint_cpa_release(roots.flip(1), objects, sizes, swapped, 1.).all()
    shifted_roots=roots.clone();shifted_objects=objects.clone()
    shifted_roots[..., :2]=roots[..., [1,0]]*torch.tensor([-1.,1.])+7.
    shifted_objects[..., :2]=objects[..., [1,0]]*torch.tensor([-1.,1.])+7.
    assert joint_cpa_release(shifted_roots, shifted_objects, sizes, graph, 1.).all()
    placement = (graph.edge_relation != HOLDING) & (graph.edge_owner == 1)
    changed = deepcopy(graph);changed.edge_dst[placement] += 1
    assert not joint_cpa_release(roots, objects, sizes, changed, 1.).any()
    changed = deepcopy(graph);changed.edge_relation[placement] = 8 if preset=='joint_carry_at' else AT
    assert not joint_cpa_release(roots, objects, sizes, changed, 1.).any()
