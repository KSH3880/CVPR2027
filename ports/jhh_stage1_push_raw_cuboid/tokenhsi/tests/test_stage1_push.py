from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from learning.multi_agent.stage1_push_network import Stage1PushBuilder, transfer
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from utils.stage1_push import push_rows, load_bank, PUSH
from utils.task_role_spec import task_graph_packet
from utils.edge_scenario_spec import compose_canonical_graph
from utils.unified_training import append_family, sample_family_matched, preserve_amp_labels
from stage1_test_utils import observations, all_pairs

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push.yaml').read_text())['env']
TRAIN = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push.yaml').read_text())


def model(push):
    builder = Stage1PushBuilder() if push else AMPMultiAgentBuilder()
    builder.load(TRAIN['params']['network'])
    result = nn.Module()
    result.a2c_network = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1330 if push else 1320,),
        value_size=1, num_agents=2, num_objects=4, observation_mode='clean_scene', scene_entity_sizes=[223,30,1],
        scene_kinematic_size=7, scene_arena_scale=5., relation_reward_mode=ENV['relationReward']['mode'],
        relation_graph_spec=ENV['relationGraph'], device='cpu')
    for encoder in (result.a2c_network.actor_encoder, result.a2c_network.critic_encoder):
        encoder.gta_diagnostics_first_forward = False
    return result


@pytest.mark.parametrize('fraction', [.3, .5, .7])
def test_transition_pool_ratio(fraction):
    options = dict(enabled=True, old_task_fraction=fraction)
    selected = push_rows(100, options, 'cpu')
    assert int((~selected).sum()) == round(100*fraction)
    assert push_rows(4, dict(enabled=True, old_task_fraction=.5), 'cpu').tolist() == [True, False, True, False]


def test_bank_and_source_contracts():
    bank = load_bank(ENV['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')
    assert len(bank['candidates']['push']) == 11
    base = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    for key in base:
        assert ENV[key] == base[key]


def test_push_packet_names_a_distinct_task():
    graph = compose_canonical_graph(torch.full((2,2), 3))
    graph.edge_relation[graph.edge_relation == 6] = PUSH
    packet = task_graph_packet(graph, 2).reshape(2,2,5)
    assert (packet[..., 1] == 4).all()
    assert (packet[..., 3] == torch.tensor([2,3])).all()
    assert (packet[..., 4] == torch.tensor([6,7])).all()


def test_amp_four_families_keep_labels_and_match_replay():
    labels = torch.arange(4)
    raw = append_family(torch.randn(4,10,129), labels[:,None].expand(-1,10), 4).flatten(1)
    shuffled = raw[torch.tensor([3,1,0,2])]
    matched = sample_family_matched(shuffled, raw, 10, num_families=4)
    torch.testing.assert_close(matched, raw)
    normalized = preserve_amp_labels(raw, torch.zeros_like(raw), 10, 4).reshape(4,10,133)
    torch.testing.assert_close(normalized[..., -4:], raw.reshape(4,10,133)[..., -4:])


def test_transfer_preserves_all_old_policies_and_discriminator():
    torch.set_num_threads(1)
    cuboid = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid.yaml').read_text())
    checkpoint = torch.load(ROOT/cuboid['env']['stage1Push']['source_checkpoint'], map_location='cpu', weights_only=False)
    old, new = model(False), model(True)
    old.load_state_dict(checkpoint['model'])
    stats, report = transfer(new, checkpoint)
    assert len(report['expanded']) == 3
    assert stats['running_mean'].shape == (133,)
    raw = observations(all_pairs())
    with torch.no_grad():
        for method in ('eval_actor', 'eval_critic'):
            first = getattr(old.a2c_network, method)(raw)
            second = getattr(new.a2c_network, method)(raw)
            if isinstance(first, tuple):
                first, second = first[0], second[0]
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
        x = torch.randn(3,10,132)
        expanded = torch.cat((x, torch.zeros(3,10,1)), -1).flatten(1)
        before = old.a2c_network._disc_mlp(x.flatten(1))
        after = new.a2c_network._disc_mlp(expanded)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
    push = raw.clone()
    push[:, -10:].reshape(-1,2,5)[...,1] = 4
    new.a2c_network.eval_actor(push)[0].square().mean().backward()
    gradient = new.a2c_network.actor_encoder.edge_encoder.category_embed.weight.grad
    assert gradient[4].abs().sum() > 0


def test_contact_geometry_requires_force_and_tracks_fixed_face():
    from env.tasks.multi_agent.push_reward import contact_geometry
    box = torch.zeros(1,2,13)
    box[..., 2], box[..., 6] = .65, 1.
    size = torch.full((1,2,3), 1.3)
    hands = torch.tensor([[[[-.1,-.65,.6175],[.1,-.65,.6175]], [[-.1,-.65,.6175],[.1,-.65,.6175]]]])
    hand_force = torch.zeros_like(hands)
    hand_force[..., 1] = -1
    box_force = torch.tensor([[[0.,2.,0.],[0.,2.,0.]]])
    phi, hit, point, normal = contact_geometry(hands, box, size, hand_force, box_force)
    assert hit.all()
    torch.testing.assert_close(phi, torch.ones_like(phi))
    assert not contact_geometry(hands, box, size, hand_force*0, box_force)[1].any()
    hands[..., 2] = .7
    assert not contact_geometry(hands, box, size, hand_force, box_force)[1].any()
    goal = box[..., :3]+normal*3
    torch.testing.assert_close(torch.cross(goal-box[..., :3], normal, dim=-1), torch.zeros_like(normal))


def test_midpoint_reward_ignores_hand_spread_but_keeps_physical_contact_checks():
    from env.tasks.multi_agent.push_reward import contact_geometry
    box = torch.zeros(1,1,13)
    box[...,2], box[...,6] = .65, 1.
    size = torch.full((1,1,3),1.3)
    hands = torch.tensor([[[[-.1,-.65,.6175],[.1,-.65,.6175]]]])
    force = torch.zeros_like(hands)
    force[...,1] = -1
    box_force = torch.tensor([[[0.,2.,0.]]])
    original = contact_geometry(hands, box, size, force, box_force, midpoint_only=True)
    spread = hands.clone()
    spread[...,0,1] -= .3
    spread[...,1,1] += .3
    midpoint = contact_geometry(spread, box, size, force, box_force, midpoint_only=True)
    legacy = contact_geometry(spread, box, size, force, box_force)
    torch.testing.assert_close(midpoint[0], original[0])
    assert (legacy[0] < midpoint[0]).all()
    assert original[1].all() and not midpoint[1].any()
    shifted = spread.clone()
    shifted[...,0] += .1
    assert (contact_geometry(shifted, box, size, force, box_force, midpoint_only=True)[0] < midpoint[0]).all()


def test_teacher_kl_is_old_only_and_teacher_frozen():
    from learning.multi_agent.stage1_push_agent import Stage1PushAgent
    student = torch.ones(4,32, requires_grad=True)
    teacher_parameter = nn.Parameter(torch.zeros(1), requires_grad=False)
    class Teacher:
        def eval_actor(self, obs):
            return teacher_parameter.expand(len(obs)*2,32), torch.zeros(len(obs)*2,32)
    agent = SimpleNamespace(num_agents=2, teacher=Teacher(), _push_spec={'teacher_kl_coef':.001}, _teacher_kl_sum=0., _teacher_kl_count=0)
    obs = observations(compose_canonical_graph(torch.full((2,2),3)))
    obs[1, -10:].reshape(2,5)[:,1] = 4
    loss = Stage1PushAgent._actor_preservation_loss(agent, obs, student, torch.ones_like(student))
    torch.testing.assert_close(loss, torch.tensor(.016))
    loss.backward()
    assert student.grad[:2].abs().sum() > 0
    assert student.grad[2:].abs().sum() == 0
    assert teacher_parameter.grad is None


def test_resume_allows_sampling_controls_but_rejects_physics_changes():
    from utils.stage1_push import contract, check_contract
    saved = {'stage1_push_contract': contract(ENV)}
    adjusted = deepcopy(ENV)
    adjusted['stage1Push']['old_task_fraction'] = .7
    adjusted['stage1Push']['teacher_kl_coef'] = .003
    check_contract(saved, adjusted)
    adjusted['stage1Push']['box']['mass'] = 30.
    with pytest.raises(ValueError):
        check_contract(saved, adjusted)


def test_push_runtime_saturation_and_original_rows():
    from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
    from env.tasks.multi_agent.push_reward import PushRuntime
    graph = compose_canonical_graph(torch.full((2,2),3))
    graph.edge_relation[0, graph.edge_relation[0] == 6] = PUSH
    zeros = torch.zeros(2,2)
    task = SimpleNamespace(_push_env=torch.tensor([True,False]), _push_hits=torch.zeros(2,2,dtype=torch.bool),
        _push_state=lambda ids: dict(phi_at=torch.ones(len(ids),2), z_error=torch.zeros(len(ids),2)))
    runtime = PushRuntime(task, Stage1ContextRuntime(2,graph,ENV['relationReward'],'cpu'))
    result = runtime.step(torch.ones(2,4),torch.ones(2,4),torch.zeros(2,4),torch.zeros(2,4),torch.zeros(2,4))
    torch.testing.assert_close(result['agent_task_reward'], torch.full((2,2),1.2))
    assert runtime.done.all()
    assert result['reward_saturated'][0].all()
    assert (result['own_success'][0] == (graph.edge_relation[0] == 7)).all()


def test_evaluation_uses_each_task_and_start_denominator():
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    tasks = torch.tensor([[0,1],[4,4]])
    starts = torch.tensor([[False,False],[False,True]])
    final = torch.tensor([[True,False],[True,False]])
    ever = torch.ones_like(final)
    result = HumanoidMAStage1Push.evaluation_metrics(None, final, ever, (tasks,starts))['task_goal_success']
    assert result['sit']['agent_trials'] == 1
    assert result['sit']['final_success'] == 1
    assert result['carry_at']['final_success'] is None
    assert result['push']['agent_trials'] == 2
    assert result['push']['final_success'] == .5
    assert result['push_loco']['final_success'] == 1
    assert result['push_rsi']['final_success'] == 0
