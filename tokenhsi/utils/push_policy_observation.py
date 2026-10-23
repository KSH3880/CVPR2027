"""Door-free PUSH policy observations, independent of physical actor slots."""
import torch

PUSH_OBJECT_COLUMNS = tuple(range(30))


def remove_door_tokens(obs, num_agents, entity_sizes, pose_size=7):
    """Delete physical door slots 0/1 and remap semantic edge endpoints."""
    h, o, g = entity_sizes
    m = num_agents
    node_end = m*h + 4*o + m*g
    pose_end = node_end + (2*m+4)*pose_size
    poses = obs[:, node_end:pose_end].reshape(obs.shape[0], 2*m+4, pose_size)
    keep = torch.cat((torch.arange(m, device=obs.device),
                      torch.arange(m+2, 2*m+4, device=obs.device)))
    packet = obs[:, pose_end:].reshape(obs.shape[0], -1, 5).clone()
    valid = packet[..., 0].bool()
    for field in (1, 2):
        endpoint = packet[..., field]
        if (valid & (endpoint >= m) & (endpoint < m+2)).any():
            raise ValueError('Door-free PUSH policy cannot reference a door')
        packet[..., field] = torch.where(endpoint >= m+2, endpoint-2, endpoint)
    return torch.cat((obs[:, :m*h], obs[:, m*h+2*o:node_end],
                      poses[:, keep].flatten(1), packet.flatten(1)), dim=-1)


def project_source_model(state):
    """Keep only the original 30 intrinsic box features in the tokenizer."""
    result = dict(state)
    for branch in ('actor_encoder', 'critic_encoder'):
        key = 'a2c_network.' + branch + '.tokenizers.1.0.weight'
        if key not in result or result[key].ndim != 2 or result[key].shape[1] != 40:
            raise ValueError('Expected the original 40-column object tokenizer')
        if not torch.isfinite(result[key]).all():
            raise ValueError('Nonfinite source object tokenizer')
        result[key] = result[key][:, PUSH_OBJECT_COLUMNS]
    return result


def project_source_normalizer(state):
    """Slice object RMS statistics; shared entity statistics retain their count."""
    result = dict(state)
    for field in ('running_mean', 'running_var'):
        key = 'running_mean_std.1.' + field
        if key not in result or result[key].shape != (40,):
            raise ValueError('Expected original 40-column object RMS')
        if not torch.isfinite(result[key]).all():
            raise ValueError('Nonfinite source object RMS')
        result[key] = result[key][list(PUSH_OBJECT_COLUMNS)]
    return result
