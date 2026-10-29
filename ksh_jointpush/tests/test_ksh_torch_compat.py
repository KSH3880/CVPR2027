import torch
import yaml
from pathlib import Path

from ksh_jointpush.ksh_torch_compat import install_attention_compat


def test_eval_per_head_float_bias_matches_training_math_and_keeps_gradients():
    install_attention_compat()
    torch.manual_seed(5)
    attention = torch.nn.MultiheadAttention(16, 2, dropout=0., batch_first=True)
    x = torch.randn(3, 4, 16)
    bias = torch.randn(6, 4, 4)
    attention.train()
    expected = attention(x, x, x, attn_mask=bias, need_weights=False)[0]
    attention.eval()
    with torch.no_grad():
        actual = attention(x, x, x, attn_mask=bias, need_weights=False)[0]
    torch.testing.assert_close(actual, expected)
    attention(x, x, x, attn_mask=bias, need_weights=False)[0].square().sum().backward()
    assert attention.in_proj_weight.grad.abs().sum() > 0
    assert not attention.training and attention.dropout == 0.


def test_cuda_volume_fallback_preserves_sampling_values_and_imported_aliases(monkeypatch):
    from ksh_jointpush.ksh_torch_compat import install_cuda_volume_compat
    from utils import stage2_jointpush, size_rsi, task_role_spec
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
    env = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'tokenhsi/data/cfg/multi_agent/stage2_jointpush_rsi.yaml').read_text())['env']
    torch.manual_seed(12)
    expected_sizes, expected_masses = stage2_jointpush.asset_bank(7, env, 'cpu')
    sizes = torch.full((7, 4, 3), .4)
    expected_probs = task_role_spec.task_size_probabilities(sizes)
    functions = [stage2_jointpush.asset_bank, size_rsi.task_probabilities,
                 HumanoidMAStage2JointPush._load_box_asset]
    for fn in functions:
        monkeypatch.setattr(fn, '__code__', fn.__code__)
        monkeypatch.setattr(fn, '_ksh_volume_compat', False, raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (12, 0))
    monkeypatch.setattr(torch.version, 'cuda', '11.8')
    assert len(install_cuda_volume_compat()) == 3
    torch.manual_seed(12)
    actual_sizes, actual_masses = stage2_jointpush.asset_bank(7, env, 'cpu')
    torch.testing.assert_close(actual_sizes, expected_sizes, rtol=0, atol=0)
    torch.testing.assert_close(actual_masses, expected_masses)
    torch.testing.assert_close(task_role_spec.task_size_probabilities(sizes), expected_probs)
