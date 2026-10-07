"""Normalization for clean-scene observations, independent of the simulator."""

import torch
import torch.nn as nn

from rl_games.algos_torch.running_mean_std import RunningMeanStd


class SceneRunningMeanStd(nn.Module):
    """Normalize selected intrinsic nodes while leaving GTA pose records untouched.

    The final L*7 pose block is consumed algebraically to construct GTA transforms.
    Normalizing it as an ordinary flat feature vector would corrupt both the shared
    coordinate system and unit quaternions. The Target constant also bypasses RMS.
    """

    def __init__(self, entity_sizes, entity_counts, normalized_sizes=None,
                 kinematic_size=13, extra_passthrough_size=0):
        super().__init__()
        self.entity_sizes = list(entity_sizes)
        self.entity_counts = list(entity_counts)
        self.normalized_sizes = (list(entity_sizes) if normalized_sizes is None
                                 else list(normalized_sizes))
        self.kinematic_size = kinematic_size
        self.extra_passthrough_size = extra_passthrough_size
        assert len(self.normalized_sizes) == len(self.entity_sizes)
        assert all(0 <= norm_size <= size
                   for norm_size, size in zip(self.normalized_sizes, self.entity_sizes))
        self.running_mean_std = nn.ModuleList(
            [RunningMeanStd((sz,)) for sz in self.normalized_sizes if sz > 0])

    def forward(self, input, denorm=False, mask=None):
        B = input.shape[0]
        out = []
        offset = 0
        rms_id = 0
        for size, count, norm_size in zip(self.entity_sizes, self.entity_counts,
                                          self.normalized_sizes):
            width = count * size
            block = input[:, offset:offset + width].reshape(B * count, size)
            if norm_size > 0:
                normalized = self.running_mean_std[rms_id](block[:, :norm_size], denorm)
                rms_id += 1
                block = torch.cat([normalized, block[:, norm_size:]], dim=-1)
            out.append(block.reshape(B, width))
            offset += width

        num_tokens = sum(self.entity_counts)
        expected = offset + num_tokens * self.kinematic_size + self.extra_passthrough_size
        assert input.shape[1] == expected, \
            "scene obs is {} wide, expected {}".format(input.shape[1], expected)
        out.append(input[:, offset:])
        return torch.cat(out, dim=-1)
