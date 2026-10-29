import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('chair_cli_helpers', Path(__file__).with_name('cli_helpers.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def write_checkpoint(path, epoch, category='ArmChair_Normal'):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(epoch=epoch, relation_experiment_config=dict(
        env=dict(holdAtFT=dict(category=category)),
        train=dict(params=dict(network=dict(table_adaptation=dict(mode='residual')))))), path)


def metadata(tmp_path):
    active = tmp_path/'active.json'
    active.write_text(json.dumps(dict(run=str(tmp_path))))
    return active


def test_selects_largest_number_and_ignores_best_alias(tmp_path):
    folder = tmp_path/'train/ChairAdaptResidual_test/nn'
    write_checkpoint(folder/'ChairAdaptResidual_9999.pth', 9999)
    latest = folder/'ChairAdaptResidual_10000.pth'
    write_checkpoint(latest, 10000)
    write_checkpoint(folder/'ChairAdaptResidual.pth', 12345)
    assert helpers.select_latest(metadata(tmp_path)) == latest.resolve()


def test_waits_for_incomplete_latest_save_instead_of_selecting_old_file(tmp_path, monkeypatch):
    folder = tmp_path/'train/ChairAdaptResidual_test/nn'
    write_checkpoint(folder/'ChairAdaptResidual_10000.pth', 10000)
    latest = folder/'ChairAdaptResidual_10100.pth'
    latest.write_bytes(b'incomplete torch checkpoint')
    monkeypatch.setattr(helpers.time, 'sleep', lambda delay: write_checkpoint(latest, 10100))
    assert helpers.select_latest(metadata(tmp_path), attempts=2, delay=0) == latest.resolve()


def test_rejects_wrong_object_category(tmp_path):
    write_checkpoint(tmp_path/'train/ChairAdaptResidual_test/nn/ChairAdaptResidual_10000.pth',
                     10000, category='Table_Circle')
    with pytest.raises(ValueError, match='Chair residual'):
        helpers.select_latest(metadata(tmp_path))


def test_completed_result_formats_mean_std_without_episode_arrays():
    text = helpers.format_summary(dict(smoke=False, records=[{}, {}, {}], statistics=dict(
        mean_percent=89.583333333333, sample_std_percentage_points=.6278418464)), '/results')
    assert 'Mean ± Std: 89.58 ± 0.63%' in text
    assert '3 independent evaluation seeds' in text and 'ddof=1' in text
    assert 'episode_steps' not in text and '/results' in text


def test_smoke_does_not_report_a_performance_mean_or_std():
    text = helpers.format_summary(dict(smoke=True, records=[{}]), '/smoke')
    assert 'SMOKE ONLY' in text and 'Mean ± Std' not in text


def test_dual_result_labels_secondary_as_same_episode_not_native_reproduction():
    text = helpers.format_summary(dict(smoke=False, records=[{}, {}, {}], statistics=dict(
        mean_percent=89.583333333333, sample_std_percentage_points=.6278418464),
        tokenhsi_criteria_statistics=dict(mean_percent=84.44, sample_std_percentage_points=1.08)), '/results')
    assert 'Ours (30s AT) Mean ± Std: 89.58 ± 0.63%' in text
    assert 'TokenHSI criteria (20s AT + upright, same episode) Mean ± Std: 84.44 ± 1.08%' in text
