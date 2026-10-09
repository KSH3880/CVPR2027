from pathlib import Path

import numpy as np
import torch


def load_paired_snapshots(directory, device):
    pools = {}
    for skill in ('pickUp', 'carryWith', 'putDown'):
        path = Path(directory) / skill
        with np.load(path / 'screened_rsi_snapshots.npz', allow_pickle=False) as snapshot, \
                np.load(path / 'cooperative_rsi_candidate.npz', allow_pickle=False) as reference:
            frame = snapshot['source_frame']
            np.testing.assert_allclose(snapshot['box_size'], [.52, .8, .4])
            np.testing.assert_allclose(snapshot['source_time'], reference['time'][frame])
            np.testing.assert_allclose(snapshot['root_state'][..., :3], reference['root_position'][frame])
            np.testing.assert_allclose(snapshot['dof_position'], reference['dof_position'][frame])
            if not snapshot['short_rsi_screen_pass'].all():
                raise ValueError('Joint RSI contains an unscreened snapshot')
            body = np.concatenate([reference[key][frame] for key in (
                'body_position', 'body_rotation_xyzw', 'body_linear_velocity',
                'body_angular_velocity')], axis=-1)
            values = {key: snapshot[key] for key in
                      ('root_state', 'dof_position', 'dof_velocity', 'box_state')}
            values['body_state'] = body
            if not all(np.isfinite(value).all() for value in values.values()):
                raise ValueError('Joint RSI contains non-finite state')
            pools[skill] = {key: torch.tensor(value, device=device, dtype=torch.float)
                           for key, value in values.items()}
            pools[skill]['source_frame'] = torch.tensor(frame, device=device)
    return pools


def rotate_vectors(v, yaw):
    while yaw.ndim < v.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c*v[..., 0]-s*v[..., 1], s*v[..., 0]+c*v[..., 1], v[..., 2]), -1)


def transform_states(state, yaw, translation):
    result = state.clone()
    shift = translation
    while shift.ndim < state.ndim:
        shift = shift.unsqueeze(-2)
    result[..., :3] = rotate_vectors(state[..., :3], yaw) + shift
    result[..., 7:10] = rotate_vectors(state[..., 7:10], yaw)
    result[..., 10:13] = rotate_vectors(state[..., 10:13], yaw)
    while yaw.ndim < state.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    c, s = (yaw/2).cos(), (yaw/2).sin()
    x, y, z, w = state[..., 3:7].unbind(-1)
    result[..., 3:7] = torch.stack((c*x-s*y, c*y+s*x, c*z+s*w, c*w-s*z), -1)
    return result
