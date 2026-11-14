import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.before_stop_evaluation import stop_pairs, write_report


def saved_run(folder, frames=60, legacy=False):
    folder.mkdir()
    packet = np.array([[[1, 2, 0, 2, 6], [1, kind, 1, 3, 2]] for kind in (0, 1, 3)])
    roots = np.zeros((frames, 3, 2, 3), np.float32)
    roots[:, :, 1, 0] = [1, 2, 3]
    boxes = np.zeros((frames, 3, 4, 7), np.float32)
    boxes[:, :, 1, 0] = .5
    waiting = np.zeros((frames, 3, 3), np.float32)
    waiting[:, :, 1] = 1
    values = dict(packet=packet, predecessor=np.array([[-1, 0]]*3), roots=roots, boxes=boxes,
        sizes=np.ones((3, 4, 3)), goals=np.zeros((3, 2, 3)), valid=np.ones((frames, 3, 2), bool),
        ever=np.ones((3, 2, 2), bool), simultaneous=np.ones((3, 2, 2), bool),
        waiting=waiting, waiting_rows=np.arange(3), waiting_owners=np.ones(3, int))
    (folder/'result.json').write_text(json.dumps(dict(seed=0, control_dt=1/30)))
    if legacy:
        torch.save(dict(values=torch.from_numpy(waiting), columns=['speed_xyz', 'armed', 'released'],
            rows=torch.arange(3), owners=torch.ones(3, dtype=torch.long)), folder/'waiting_trace.pt')
        values = {k: v for k, v in values.items() if not k.startswith('waiting')}
    return values


@pytest.mark.parametrize('legacy', [False, True])
def test_first_stop_distances_and_success_link(tmp_path, legacy):
    folder = tmp_path/'seed0'
    values = saved_run(folder, legacy=legacy)
    np.savez(folder/'trace.npz', **values)
    rows = stop_pairs(folder)
    assert [r['stop_step'] for r in rows] == [30]*3
    assert [r['root_distance'] for r in rows] == [1, 2, 3]
    assert [r['payload_distance'] for r in rows] == [None, None, .5]
    report = write_report([folder], tmp_path/'summary')
    assert report['groups'][-1]['stopped_success'] == dict(n=3, follower_simultaneous=3, same_pair_leader_ever=3)


def test_movement_release_and_fall_break_streak(tmp_path):
    folder = tmp_path/'seed0'
    values = saved_run(folder)
    values['waiting'][29, 0, 0] = .31
    values['waiting'][29:, 1, 2] = 1
    values['valid'][29:, 2, 1] = False
    np.savez(folder/'trace.npz', **values)
    rows = stop_pairs(folder)
    assert [r['stop_step'] for r in rows] == [60, None, None]
    assert [r['eligible'] for r in rows] == [1, 0, 0]


def test_short_rollout_and_missing_telemetry(tmp_path):
    folder = tmp_path/'seed0'
    values = saved_run(folder, frames=20)
    np.savez(folder/'trace.npz', **values)
    assert not any(r['stopped'] for r in stop_pairs(folder))
    np.savez(folder/'trace.npz', **{k: v for k, v in values.items() if not k.startswith('waiting')})
    assert write_report([folder], tmp_path/'summary') is None


def test_duplicate_seed_rejected(tmp_path):
    folder = tmp_path/'seed0'
    np.savez(folder/'trace.npz', **saved_run(folder))
    with pytest.raises(ValueError, match='Duplicate seeds'):
        write_report([folder, folder], tmp_path/'summary')


def test_evaluate_summary_includes_same_rollout_stops(tmp_path):
    from evaluate import summarize as summarize_evaluation
    folder = tmp_path/'seed0'
    np.savez(folder/'trace.npz', **saved_run(folder))
    result = json.loads((folder/'result.json').read_text())
    result.update(protocol='general', scores={'point': {}, 'region': {}})
    (folder/'result.json').write_text(json.dumps(result))
    summary = summarize_evaluation(tmp_path)
    assert summary['before_stops']['groups'][-1]['stopped'] == 3
    assert (tmp_path/'stops.json').exists()


def test_formal_records_xyz_without_extra_steps_or_score_changes(tmp_path, monkeypatch):
    from learning.multi_agent import formal_eval
    packet = torch.tensor([[[1, 2, 0, 2, 4], [1, 1, 1, 1, 3]]])
    monkeypatch.setattr(formal_eval, 'task_graph_packet', lambda *args: packet)
    monkeypatch.setattr(formal_eval, 'predecessors', lambda *args: torch.tensor([[-1, 0]]))
    monkeypatch.setattr(formal_eval, 'spatial_success', lambda *args: torch.ones(1, 2, 2, dtype=torch.bool))
    outputs = []
    for enabled in (False, True):
        folder = tmp_path/str(enabled)
        args = SimpleNamespace(output_path=str(folder), eval_protocol='general', eval_single_task='',
            cfg_train='test.yaml', seed=0, resolved_env_config='test.yaml')
        wait = SimpleNamespace(rows=torch.tensor([0]), owners=torch.tensor([1]), edges=torch.tensor([0]),
            armed=torch.ones(1, 1, dtype=torch.bool), released=torch.zeros(1, 1, dtype=torch.bool))
        task = SimpleNamespace(cfg={'args': args, 'env': {'episodeLength': 34}}, num_envs=1, num_agents=2,
            num_objects=2, device='cpu', dt=1/30, relation_runtime=SimpleNamespace(graph=None),
            _logical_box_values=lambda x: x, _box_size=torch.ones(1, 2, 3), _tar_pos=torch.zeros(1, 2, 3),
            _humanoid_root_states=torch.zeros(1, 2, 13), _box_states=torch.zeros(1, 2, 13),
            _dof_pos=torch.zeros(1), _dof_vel=torch.zeros(1), _logical_box_order=torch.tensor([[0, 1]]),
            _terminate_buf=torch.zeros(1, 2, dtype=torch.bool), _before_wait=wait if enabled else None)
        steps = []
        def step(*args):
            steps.append(1)
            task._humanoid_root_states[0, 1, 9] = .4 if len(steps) <= 4 else 0.
            return torch.zeros(1, 1), None, None, None
        player = SimpleNamespace(env=SimpleNamespace(task=task), env_reset=lambda: {'obs': torch.zeros(1, 1)},
            get_batch_size=lambda *args: None, get_action=lambda *args: torch.zeros(1, 2, 1), env_step=step)
        outputs.append(formal_eval.run_formal_eval(player))
        assert len(steps) == 34
    assert outputs[0]['scores'] == outputs[1]['scores']
    assert stop_pairs(tmp_path/'False') is None
    rows = stop_pairs(tmp_path/'True')
    assert rows[0]['stop_step'] == 34
