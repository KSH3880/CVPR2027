from pathlib import Path

import torch
import yaml

from scripts.expand_checkpoint import push_tokenizer_state
from push_reference.utils.stage1_jointpush import sample_assets, validate_env

ROOT = Path(__file__).resolve().parents[1]


def test_random_tokenizer_only_changes_push_weights():
    legacy = torch.load(ROOT/'weights/push_init.pth', map_location='cpu', weights_only=True)
    fresh = torch.load(ROOT/'weights/push_init_random_seed42.pth', map_location='cpu', weights_only=True)
    for name, value in fresh['model'].items():
        if name.startswith('a2c_network.task_encoder.4.'):
            if name.endswith('weight'):
                assert not torch.equal(value, legacy['model'][name])
            else:
                assert torch.count_nonzero(value) == 0
        else:
            torch.testing.assert_close(value, legacy['model'][name], rtol=0, atol=0)
    for key in ('running_mean_std', 'amp_input_mean_std', 'reward_mean_std'):
        for name, value in fresh[key].items():
            torch.testing.assert_close(value, legacy[key][name], rtol=0, atol=0)
    same = push_tokenizer_state(legacy['model'], seed=42)
    other = push_tokenizer_state(legacy['model'], seed=43)
    for name, value in same.items():
        torch.testing.assert_close(value, fresh['model'][name], rtol=0, atol=0)
        if name.endswith('weight'):
            assert not torch.equal(value, other[name])


def test_reference_sampler_uses_only_push():
    env = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_task/push_reference.yaml').read_text())['env']
    validate_env(env)
    torch.manual_seed(42)
    _, probabilities = sample_assets(10000, env, 'cpu')
    observed = (probabilities[..., 5] > 0).float().mean().item()
    assert observed == 1.0, observed
    assert torch.count_nonzero(probabilities[..., :5]) == 0
