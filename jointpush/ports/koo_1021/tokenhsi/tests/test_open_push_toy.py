"""Single-task reset and AMP sampling must agree with the saved contract."""
import json
from pathlib import Path
from types import SimpleNamespace

from isaacgym import gymapi  # Isaac Gym must precede torch.
import pytest
import torch
import yaml

from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('task,config,family', [('open', 'open_toy', 0), ('push', 'push_toy', 1), ('push', 'push_toy_midpoint_2m', 1), ('open', 'open_toy_strong_close', 0), ('open', 'open_toy_reference_close', 0)])
def test_single_task_reset_and_expert_labels(task, config, family):
    from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

    env = yaml.safe_load((ROOT / f'tokenhsi/data/cfg/multi_agent/{config}.yaml').read_text())['env']
    validate_env(env)
    sampled_skills = []

    def library(skill):
        def sample_motions(count):
            sampled_skills.append(skill)
            return torch.zeros(count, dtype=torch.long)
        return SimpleNamespace(sample_motions=sample_motions,
                               sample_time=lambda ids, **kwargs: torch.ones(len(ids)))

    fake = SimpleNamespace(
        _task_graph_preset='random_open_push',
        _relation_graph_spec=env['relationGraph'], device='cpu', cfg={'env': env},
        get_num_amp_obs=lambda: 1310, dt=1 / 30, _num_amp_obs_steps=10,
        _motion_lib={skill: library(skill) for skill in env['skill']},
        build_amp_obs_demo=lambda ids, times, lib: torch.zeros(len(ids), 10, 129),
    )
    tasks = HumanoidMAOpenPush._sample_tasks(fake, torch.arange(1024))
    assert tasks.shape == (1024, 2) and (tasks == family).all()
    fake._task_graph_preset = task
    assert (HumanoidMAOpenPush._sample_tasks(fake, torch.arange(16)) == family).all()
    demo = HumanoidMAOpenPush._fetch_conditioned_amp_demo(fake, 1024).reshape(1024, 10, 131)
    assert torch.isfinite(demo).all()
    assert (demo[..., 129 + family] == 1).all()
    assert (demo[..., 129 + (1 - family)] == 0).all()
    assert set(sampled_skills) == {'loco', 'doorOpen' if family == 0 else 'push'}


@pytest.mark.parametrize('config', ['open_toy', 'push_toy', 'push_toy_midpoint_2m', 'open_toy_strong_close', 'open_toy_reference_close'])
def test_copied_rsi_bank_matches_local_motion_files(config, monkeypatch):
    monkeypatch.chdir(ROOT)
    env = yaml.safe_load(Path(f'tokenhsi/data/cfg/multi_agent/{config}.yaml').read_text())['env']
    env['motion_file'] = 'tokenhsi/data/dataset_open_push.yaml'
    bank = json.loads(Path(env['openPushRSI']['bank_file']).read_text())
    assert bank['screen'] == 'PD_12_steps'
    assert bank['contract'] == bank_contract(env)
