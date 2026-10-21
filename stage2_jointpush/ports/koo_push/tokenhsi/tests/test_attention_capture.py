import importlib.util
import math
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / 'scripts/multi_agent/native4_zeroshot/attention_capture.py'
spec = importlib.util.spec_from_file_location('attention_capture', path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_capture_matches_layer_softmax_and_identity_gta():
    torch.manual_seed(23)
    qkv = torch.randn(2, 8, 192)
    bias = torch.randn(2, 2, 8, 8)
    q, k, _ = qkv.reshape(2, 8, 3, 2, 32).permute(2, 0, 3, 1, 4)
    scores = (q @ k.transpose(-1, -2)) * (1 / math.sqrt(32))
    expected = torch.softmax(scores + bias, -1)
    actual_scores, actual_bias, attention = capture.attention_parts(qkv, 2, bias)
    assert torch.equal(actual_scores, scores)
    assert torch.equal(actual_bias, bias)
    assert torch.equal(attention, expected)
    identity = torch.eye(4).expand(2, 8, 4, 4)
    _, _, transformed = capture.attention_parts(qkv, 2, bias, identity, identity)
    assert torch.allclose(transformed, expected, atol=1e-7)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 2, 8))


def test_gta_transform_preserves_token_and_head_axes():
    matrices = torch.eye(4).repeat(2, 3, 1, 1)
    matrices[..., 0, 3] = torch.arange(6).reshape(2, 3)
    features = torch.randn(2, 2, 3, 8)
    actual = capture.transform(matrices, features)
    expected = features.clone()
    for batch in range(2):
        for head in range(2):
            for token in range(3):
                for block in range(2):
                    sl = slice(4 * block, 4 * block + 4)
                    expected[batch, head, token, sl] = matrices[batch, token] @ features[batch, head, token, sl]
    assert torch.allclose(actual, expected)
