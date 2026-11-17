from copy import deepcopy

import torch

from utils.bridge_curriculum import span_probabilities, walk_layout
from utils.experiments import native_config, validate_experiment
from env.tasks.multi_agent.sensor_stage2_curriculum import SensorStage2Curriculum


NAME = 'stage2_sensor_900_bridge_curriculum'


def test_config_changes_only_curriculum_and_run_name():
    cfg = native_config(NAME)
    current = deepcopy(cfg)
    current['env']['sensorScene']['bridge'].pop('curriculum')
    current['experiment']['name'] = 'stage2_sensor_900_bridge_init'
    assert current == native_config('stage2_sensor_900_bridge_init')
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))


def test_span_schedule():
    spec = native_config(NAME)['env']['sensorScene']['bridge']['curriculum']
    assert span_probabilities(spec, 0) == [.8, .15, .05]
    assert span_probabilities(spec, 1000) == [.8, .15, .05]
    torch.testing.assert_close(torch.tensor(span_probabilities(spec, 2000)), torch.tensor([.8/2+1/6, .15/2+1/6, .05/2+1/6]))
    assert span_probabilities(spec, 3000) == [1/3]*3
    assert span_probabilities(spec, 5000) == [1/3]*3


def test_layout_all_lengths_and_directions():
    torch.manual_seed(42)
    n = 1200
    spec = native_config(NAME)['env']['sensorScene']['bridge']['curriculum']
    span = torch.arange(n)%3+1
    bridge = torch.tensor([1.8, 1.8, 1.5]).repeat(n, 1)
    bridge[:, 0] *= span
    gap = bridge[:, 0]+.2
    start = torch.zeros(n, 2)
    dest = torch.stack((8+gap, torch.zeros(n)), 1)
    landing = torch.stack((4+gap/2, torch.zeros(n)), 1)
    reverse = torch.arange(n)%2 == 0
    start[reverse], dest[reverse] = dest[reverse].clone(), start[reverse].clone()
    tasks = torch.randint(1, 5, (n, 3))
    radii = torch.full((n, 3, 2), .4)
    phases = torch.zeros(n, 3, dtype=torch.long)
    phases[:, 2] = torch.arange(n)%2+1
    progress = torch.zeros(n, 3)
    progress[:, 2] = torch.where(phases[:, 2] == 1, .4, .75)
    source, second, human, failed = walk_layout(start, torch.full((n, 2), 4.), dest, torch.full((n, 2), 4.),
        landing, bridge, gap, tasks, radii, phases, progress, spec)
    assert not failed.any()
    direction = torch.sign(start[:, 0]-landing[:, 0])
    assert (((human[:, :2, 0]-landing[:, 0, None])*direction[:, None]-gap[:, None]/2) > 0).all()
    assert (((source[..., 0]-landing[:, 0, None])*(-direction[:, None])-gap[:, None]/2) >= 1.9-1e-5).all()
    assert ((human[:, 2, 0]-landing[:, 0]).abs() < bridge[:, 0]/2).all()
    distance = (human[:, :, None]-human[:, None]).norm(dim=-1)
    pairs = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    assert (distance[:, pairs] >= 1.2).all()


def test_role_sampling_and_default_evaluation():
    task = SensorStage2Curriculum()
    task.num_envs, task.num_agents, task.device = 12000, 4, 'cpu'
    task._bridge_curriculum = native_config(NAME)['env']['sensorScene']['bridge']['curriculum']
    task._scene_ids = torch.ones(task.num_envs, dtype=torch.long)
    task._is_eval = False
    task._init_bridge_curriculum()
    ids = torch.arange(task.num_envs)
    original = torch.randint(1, 4, (task.num_envs,))
    counts, spans = task._sample_bridge_curriculum(ids, original)
    assert abs(task._bridge_walk.float().mean().item()-.7) < .02
    assert counts[task._bridge_walk].eq(1).all()
    assert torch.equal(counts[~task._bridge_walk], original[~task._bridge_walk])
    fraction = torch.bincount(spans[task._bridge_walk]-1, minlength=3).float()/task._bridge_walk.sum()
    torch.testing.assert_close(fraction, torch.tensor([.8, .15, .05]), rtol=0, atol=.02)
    task._is_eval, task._curriculum_eval_mode = True, 'cooperative'
    counts, spans = task._sample_bridge_curriculum(ids, original)
    assert not task._bridge_walk.any()
    assert torch.equal(counts, original) and torch.equal(spans, original)


def test_agent_denominators_and_interruption_metrics():
    task = SensorStage2Curriculum()
    task.num_envs, task.num_agents, task.device = 2, 4, 'cpu'
    task._bridge_curriculum = native_config(NAME)['env']['sensorScene']['bridge']['curriculum']
    task._init_bridge_curriculum()
    task._scene_ids = torch.tensor([1, 2])
    task._bridge_walk[0], task._bridge_span[0] = True, 2
    task._crosser_mask[0, 1:] = True
    task._cross_phase[0, 1:] = torch.tensor([0, 0, 2])
    task._bridge_slot = torch.zeros(2, dtype=torch.long)
    task._box_states = torch.zeros(2, 1, 13)
    task._box_states[:, 0, 2], task._box_states[:, 0, 6] = .75, 1.
    task._box_size = torch.tensor([[[3.6, 1.8, 1.5]]]*2)
    task._surface = torch.full((2, 4), 2, dtype=torch.long)
    task._start_surface = torch.ones(2, 4, dtype=torch.long)
    task._unreachable = torch.zeros(2, 4, dtype=torch.bool)
    task._surface_center = torch.zeros(2, 3, 2, 2)
    task._surface_center[:, 1, 1, 0] = 4.
    task._surface_half = torch.ones(2, 3, 2, 2)
    task._env_origins, task._structure_shift = torch.zeros(2, 3), torch.zeros(2, 3, 2)
    task._structure_yaw, task._surface_height = torch.zeros(2, 3), torch.full((2, 3), 1.51)
    task._bridge_arrived = torch.tensor([True, False])
    task.progress_buf, task.max_episode_length = torch.tensor([100, 100]), 900
    roots = torch.zeros(2, 4, 3)
    roots[..., 2] = 2.4
    roots[0, 1, 0], roots[0, 3, 0] = 4., -3.
    fallen = torch.zeros(2, 4, dtype=torch.bool)
    fallen[0, 0] = True
    task._record_crossing_curriculum(fallen, torch.tensor([True, False]), roots)
    stats = task._consume_crossing_curriculum()
    assert stats['crossing/exposure/crosser_agent_steps'] == 3
    assert stats['crossing/exposure/all_agent_steps'] == 8
    assert stats['crossing/walk/all/episodes'] == 3
    torch.testing.assert_close(stats['crossing/walk/all/crossed'], torch.tensor(1/3))
    torch.testing.assert_close(stats['crossing/walk/all/interrupted_before_crossing'], torch.tensor(2/3))
    assert stats['crossing/walk/entry/episodes'] == 2
