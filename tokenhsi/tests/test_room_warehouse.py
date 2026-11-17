from copy import deepcopy

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.sensor_stage2_observation import SensorStage2Observation
from env.tasks.multi_agent.square_sensor_scene import compute_wall_cpa_collision_penalty
from utils.coupled_push_spec import check_checkpoint, checkpoint_contract
from utils.experiments import native_config, validate_experiment
from utils.sensor_stage2_spec import CLIFF, ROOM, WAREHOUSE, validate_sensor


NAME = 'stage2_sensor_900_room_warehouse'
BASE = 'stage2_sensor_900_warehouse_coop_cliffroom'


def test_room_warehouse_changes_only_mixture_and_wall_coefficient():
    current, previous = native_config(NAME), native_config(BASE)
    train, old_train = native_config(NAME, True), native_config(BASE, True)
    validate_experiment(NAME, current['env'], train)
    assert current['env']['sensorScene']['mixture'] == dict(cliff=0., bridge=0., room=.5, warehouse=.5)
    assert current['env']['sensorScene']['wall_collision_coefficient'] == .7
    assert current['env']['agentCollisionCoeff'] == .5
    check_checkpoint(dict(coupled_push_contract=checkpoint_contract(current['env'])), current['env'])
    with pytest.raises(ValueError):
        check_checkpoint(dict(coupled_push_contract=checkpoint_contract(previous['env'])), current['env'])
    restored = deepcopy(current)
    restored['experiment']['name'] = BASE
    assert current['experiment']['transfer']['checkpoint'] == 'output/stage2/Stage2Distill900V4_00002000.pth'
    restored['experiment']['transfer'] = previous['experiment']['transfer']
    restored['env']['sensorScene']['mixture'] = previous['env']['sensorScene']['mixture']
    del restored['env']['sensorScene']['wall_collision_coefficient']
    assert restored == previous
    train['params']['config']['name'] = old_train['params']['config']['name']
    assert train == old_train
    probabilities = torch.tensor(list(current['env']['sensorScene']['mixture'].values()))
    assert set(torch.multinomial(probabilities, 10000, True).tolist()) == {ROOM, WAREHOUSE}


@pytest.mark.parametrize('coefficient', [-.1, float('nan'), float('inf')])
def test_invalid_wall_coefficient_is_rejected(coefficient):
    env = native_config(NAME)['env']
    env['sensorScene']['wall_collision_coefficient'] = coefficient
    with pytest.raises(ValueError):
        validate_sensor(env)


@pytest.mark.parametrize('scale', [1., 1.4])
def test_wall_weight_changes_both_rooms_without_scaling_human_risk(scale):
    class Base:
        def _compute_agent_collision_violation(self, root_pos, min_dist):
            return root_pos.new_tensor([[0.], [.8], [0.], [.8], [.3]])

    class Task(SensorStage2Observation, Base):
        pass

    task = Task()
    task._wall_collision_scale = scale
    task._push_box_collision_scale = 0.
    task._scene_ids = torch.tensor([ROOM, ROOM, WAREHOUSE, WAREHOUSE, CLIFF])
    task._scene_cfg = {'room': {'wall_ttc_discount': .95}}
    task._warehouse = {'wall_ttc_discount': .95}
    task.dt = 1/30
    task._humanoid_root_states = torch.zeros(5, 1, 13)
    task._humanoid_root_states[[0, 2], :, 7] = 1.
    task._wall_starts = torch.tensor([[[.1, -5.]]]).expand(5, -1, -1)
    task._wall_ends = torch.tensor([[[.1, 5.]]]).expand(5, -1, -1)
    task._warehouse_wall_starts = task._wall_starts
    task._warehouse_wall_ends = task._wall_ends
    roots = torch.zeros(5, 1, 3)
    walls = compute_wall_cpa_collision_penalty(roots, task._humanoid_root_states[..., 7:10],
        task._wall_starts, task._wall_ends, .7, .95, task.dt)
    actual_reward = -.5*task._compute_agent_collision_violation(roots, .7)
    expected = torch.tensor([[-.5*scale*walls[0, 0]], [-.4],
                             [-.5*scale*walls[2, 0]], [-.4], [-.15]])
    torch.testing.assert_close(actual_reward, expected)
