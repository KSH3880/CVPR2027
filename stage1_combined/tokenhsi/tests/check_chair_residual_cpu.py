"""Validate the native Chair residual profile; optionally export an untrained smoke policy."""
import isaacgym  # Must precede torch.
from pathlib import Path
import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tokenhsi'), str(ROOT)]
import torch
import yaml
from isaacgym.torch_utils import quat_rotate
from env.tasks.multi_agent.humanoid_chair_tokenhsi_ft import HumanoidChairTokenHSIFT
from env.tasks.multi_agent.humanoid_table_tokenhsi_ft import HumanoidTableTokenHSIFT
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.table_adaptation_network import TableAdaptationBuilder
from learning.multi_agent.table_adaptation_agent import transfer_policy, ADAPTER_PREFIXES
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet
from utils.hold_at_ft import object_catalog
from utils.unified_training import validate_unified_env, validate_stage1_network

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--smoke-checkpoint', type=Path)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(42)
load = lambda path: yaml.safe_load((ROOT / 'tokenhsi/data/cfg' / path).read_text())
cfg = load('multi_agent/chair_adapt_residual.yaml')
train = load('train/rlg/chair_adapt_residual.yaml')
validate_unified_env(cfg['env'])
validate_stage1_network(cfg['env'], train)
expected = load('multi_agent/table_adapt_residual.yaml')
expected['env']['holdAtFT'].update(category='ArmChair_Normal', face_human=True, reward='tokenhsi_native_chair')
assert cfg == expected
expected_train = load('train/rlg/table_adapt_residual.yaml')
expected_train['params']['config']['name'] = 'ChairAdaptResidual'
assert train == expected_train
assert HumanoidChairTokenHSIFT._compute_reward is HumanoidTableTokenHSIFT._compute_reward
assert HumanoidChairTokenHSIFT._compute_reset is HumanoidTableTokenHSIFT._compute_reset
counts = {}
for split, count in [('train', 63), ('test', 27)]:
    catalog = object_catalog(ROOT / 'tokenhsi/data/dataset_dynamic_objects', split, 'ArmChair_Normal')
    assert len(catalog) == count
    for obj in catalog:
        for mesh in ET.parse(Path(obj['path']) / 'asset.urdf').iter('mesh'):
            assert (Path(obj['path']) / mesh.attrib['filename']).is_file()
    counts[split] = len(catalog)
# Exercise Chair's overridden reset on a noncontiguous subset.
t = HumanoidChairTokenHSIFT.__new__(HumanoidChairTokenHSIFT)
t.device = 'cpu'; t.num_envs = 32
n = t.num_envs
t._humanoid_root_states = torch.zeros(n, 1, 13)
t._box_states = torch.full((n, 1, 13), -99.)
t._object_ground_z = torch.full((n,), .6)
t._object_front = torch.tensor([[0., -1., 0.]]).repeat(n, 1)
t._platform_pos = torch.zeros(n, 1, 3)
t._platform_default_pos = t._platform_pos.clone()
ids = torch.arange(0, n, 2)
t._reset_boxes(ids)
assert (t._box_states[1::2] == -99).all() and (t._native_source[1::2] == 0).all()
torch.testing.assert_close(t._native_source[ids], t._box_states[ids, 0, :3])
d = t._box_states[ids, 0, :2].norm(dim=-1)
assert ((d >= 1) & (d <= 5)).all()
front = quat_rotate(t._box_states[ids, 0, 3:7], t._object_front[ids])
expected_front = -t._box_states[ids, 0, :2] / d[:, None]
torch.testing.assert_close(front[:, :2], expected_front, rtol=1e-6, atol=1e-6)
source = torch.load(ROOT / 'checkpoints/Stage1V2Fixed_00078000.pth', map_location='cpu', weights_only=False)
kwargs = dict(actions_num=32, input_shape=(280,), amp_input_shape=(1320,), value_size=1,
    num_agents=1, num_objects=1, humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
    observation_mode='clean_scene', scene_entity_sizes=[223,30,1], scene_kinematic_size=7,
    scene_arena_scale=5., relation_reward_mode=cfg['env']['relationReward']['mode'],
    relation_graph_spec=cfg['env']['relationGraph'], device='cpu')
b = TableAdaptationBuilder(); b.load(train['params']['network'])
model = torch.nn.Module(); model.a2c_network = b.build('amp', **kwargs)
transfer_policy(model, source); model.eval()
b = AMPMultiAgentBuilder(); b.load(train['params']['network'])
base = b.build('amp', **kwargs)
base.load_state_dict({k[len('a2c_network.'):]: v for k,v in source['model'].items()}, strict=True)
base.eval()
kin = torch.randn(4,3,7); kin[...,3:] = torch.nn.functional.normalize(kin[...,3:], dim=-1)
graph = compose_canonical_graph(torch.full((4,1), 3), num_objects=1)
obs = torch.cat((torch.randn(4,254), kin.flatten(1), task_graph_packet(graph,4)), -1)
with torch.no_grad():
    initial = model.a2c_network.eval_actor(obs)[0]
    torch.testing.assert_close(initial, base.eval_actor(obs)[0], rtol=1e-6, atol=1e-6)
# Save BEFORE the synthetic gradient updates: this is not a trained checkpoint.
if a.smoke_checkpoint:
    assert 'output_etc/check' in str(a.smoke_checkpoint)
    smoke = copy.deepcopy(source)
    smoke.update(model=copy.deepcopy(model.state_dict()), epoch=0,
        relation_experiment_config=dict(cfg, train=train),
        preparation_only=True, source_epoch=78000)
    smoke.pop('optimizer', None)
    a.smoke_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(smoke, a.smoke_checkpoint)
before = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith(ADAPTER_PREFIXES)}
for name, param in model.named_parameters():
    param.requires_grad_(name.startswith('a2c_network.action_residual.'))
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
gradients = []
for step in range(2):
    optimizer.zero_grad()
    model.a2c_network.eval_actor(obs)[0].square().mean().backward()
    gradients.append(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None))
    assert gradients[-1] > 0
    if step == 1:
        assert model.a2c_network.action_residual[0].weight.grad.abs().sum() > 0
    optimizer.step()
assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
print('CHAIR_RESIDUAL_CPU', json.dumps(dict(asset_counts=counts, original_settings_equal=True,
    partial_reset=True, native_source_correct=True, chair_front_alignment=True,
    initial_actor_matches_78k=True, residual_gradients=gradients, frozen_base_exact=True,
    shared_native_reward_and_termination=True, smoke_checkpoint=str(a.smoke_checkpoint))))
