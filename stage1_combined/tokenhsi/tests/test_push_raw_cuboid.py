from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from utils.push_navigation import max_box_edge, sample_box_properties
from utils.stage1_push import check_contract, check_navigation_transfer, contract, load_bank

ROOT = Path(__file__).resolve().parents[2]
ENV = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid.yaml').read_text())['env']
RAW = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw.yaml').read_text())['env']
BOX = ENV['stage1Push']['box']


def test_cuboid_sizes_masses_and_depth_order():
    torch.manual_seed(42)
    sizes, masses = sample_box_properties(4096, BOX, 'cpu')
    x, y, z = sizes.T
    assert x.min() >= 1.2 and x.max() <= 1.7 and x.min() < 1.21 and x.max() > 1.69
    assert y.min() >= 1.45 and y.max() <= 1.7 and z.min() >= 1.45 and z.max() <= 1.7
    assert (y >= z).all() and (y > z).float().mean() > .99
    assert masses.min() >= 15 and masses.max() <= 20
    assert abs(torch.corrcoef(torch.stack((x, y)))[0, 1]) < .05
    assert abs(torch.corrcoef(torch.stack((y, masses)))[0, 1]) < .05
    assert max_box_edge(BOX) == 1.7
    assert max_box_edge(RAW['stage1Push']['box']) == 1.7
    assert max_box_edge({'size': [1.3, 1.4, 1.3]}) == 1.4


def test_depth_slides_before_tipping_at_raw_hand_height():
    friction = (BOX['friction']+ENV['plane']['staticFriction'])/2
    assert BOX['size_range'][1][0] > 2*friction*ENV['stage1Push']['contact_height']


def test_only_push_box_physics_bank_and_horizon_change():
    changed = deepcopy(ENV)
    original = deepcopy(RAW)
    for spec in (changed['stage1Push'], original['stage1Push']):
        spec.pop('box')
        spec['rsi'].pop('bank')
        spec['navigation'].pop('episode_length')
    assert changed == original
    assert ENV['stage1Push']['navigation']['episode_length'] == 1200
    bank = load_bank(ENV['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')
    raw_bank = load_bank(RAW['stage1Push'], ROOT/'tokenhsi/data/dataset_stage1_push.yaml')
    assert bank['candidates'] == raw_bank['candidates']
    check_contract({'stage1_push_contract': contract(ENV)}, ENV)
    with pytest.raises(ValueError, match='physics'):
        check_navigation_transfer({'stage1_push_contract': contract(RAW)}, ENV)


def test_friction_applies_only_to_push_boxes(monkeypatch):
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    props = {}

    def build(self, env_id, object_id, env_ptr):
        self._box_handles.append((env_id, object_id))
        props[env_id, object_id] = [SimpleNamespace(friction=1., restitution=.5)]
    monkeypatch.setattr(HumanoidMACarry, '_build_box', build)
    task = object.__new__(HumanoidMAStage1Push)
    task.num_agents, task._box_handles = 2, []
    task._push_env = torch.tensor([True, False])
    task._push_options = ENV['stage1Push']
    task.gym = SimpleNamespace(get_actor_rigid_shape_properties=lambda env, handle: props[handle],
                               set_actor_rigid_shape_properties=lambda env, handle, value: None)
    for env_id in range(2):
        for object_id in range(4):
            task._build_box(env_id, object_id, None)
    friction = {key: value[0].friction for key, value in props.items()}
    assert friction == {(e, o): .3 if e == 0 and o < 2 else 1. for e in range(2) for o in range(4)}
