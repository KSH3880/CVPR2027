"""Policy clipping must preserve literal targets, GTA poses and task packets."""
import pytest
import torch

from env.tasks.multi_agent.scene_features import clip_scene_observation


@pytest.mark.parametrize('clip', [1., float('inf')])
def test_clipping_preserves_algebraic_and_task_fields(clip):
    values = torch.full((3, 634), 7.)
    actual = clip_scene_observation(values, 2, 4, [223, 30, 1], clip)
    assert actual.shape == values.shape
    torch.testing.assert_close(actual[:, :566], torch.full((3, 566), min(7., clip)))
    torch.testing.assert_close(actual[:, 566:], values[:, 566:], atol=0, rtol=0)
    torch.testing.assert_close(values, torch.full_like(values, 7.), atol=0, rtol=0)
