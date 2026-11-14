import torch


def quat_rotate_dimflex(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (B,4)
    # v: (K,3) or (B,3) or (B,K,3)

    B = q.shape[0]
    q_vec = q[:, :3]           # (B,3)
    q_w  = q[:,  3]            # (B,)

    # Prepare shapes for broadcasting
    squeeze_K = False
    if v.dim() == 2:
        # (N,3): either (B,3) or (K,3)
        if v.shape[0] == B:
            # (B,3) -> treat as (B,1,3), squeeze later
            v = v.unsqueeze(1)         # (B,1,3)
            q_vec = q_vec.unsqueeze(1) # (B,1,3)
            q_w  = q_w.unsqueeze(1)    # (B,1)
            squeeze_K = True
        else:
            # (K,3) -> (1,K,3), broadcast over B
            v = v.unsqueeze(0)         # (1,K,3)
            q_vec = q_vec.unsqueeze(1) # (B,1,3)
            q_w  = q_w.unsqueeze(1)    # (B,1)
    elif v.dim() == 3:
        # (B,K,3)
        q_vec = q_vec.unsqueeze(1)     # (B,1,3)
        q_w  = q_w.unsqueeze(1)        # (B,1)
    else:
        raise RuntimeError("v must be (K,3), (B,3), or (B,K,3)")

    # Now v, q_vec broadcast to (B,*,3); q_w to (B,*,)
    s = (2.0 * q_w * q_w - 1.0).unsqueeze(-1)          # (B,*,1)
    a = v * s                                          # (B,*,3)
    b = 2.0 * q_w.unsqueeze(-1) * torch.cross(q_vec, v, dim=-1)
    c = 2.0 * q_vec * (q_vec * v).sum(-1, keepdim=True)
    out = a + b + c                                    # (B,*,3)

    if squeeze_K:
        out = out.squeeze(1)  # back to (B,3)
    return out


def compute_rim_velocity_variance_norm(log_rim, start_t, end_t, env_ids, delta_t=0.0333, eps=1e-12):
    """
    Jerk-focused metric: masked VARIANCE of JERK over time,
    averaged over rim points (K) and axes (x,y,z).

    Args:
        log_rim:  (N_envs, Tmax, K, 3)  positions buffer
        start_t:  (N_envs,)             inclusive start indices (position timeline)
        end_t:    (N_envs,)             exclusive end indices   (position timeline)
        env_ids:  (M,)                  subset of env indices
        delta_t:  float                 dt between frames
        eps:      float

    Returns:
        var_all_env_norm: (M,)  per-env jerkiness metric (jerk variance).
                            (RMS-jerk normalization available but commented)
    """
    # Select env subset
    pos = log_rim[env_ids]                      # (M, Tmax, K, 3)
    start_idx = start_t[env_ids]                # (M,)
    end_idx   = end_t[env_ids]                  # (M,)

    # --- Velocities (T-1) ---
    vel = (pos[:, 1:] - pos[:, :-1]) / float(delta_t)      # (M, T-1, K, 3)
    Tm1 = vel.size(1)

    # Validity on velocity timeline for [start_idx, end_idx)
    t_v = torch.arange(Tm1, device=vel.device)[None, :]    # (1, T-1)
    valid_v = (t_v >= start_idx[:, None]) & (t_v < end_idx[:, None])   # (M, T-1)

    # --- Accelerations (T-2) ---
    acc = (vel[:, 1:] - vel[:, :-1]) / float(delta_t)      # (M, T-2, K, 3)
    valid_a = valid_v[:, 1:] & valid_v[:, :-1]             # (M, T-2)

    # --- Jerks (T-3) ---
    jerk = (acc[:, 1:] - acc[:, :-1]) / float(delta_t)     # (M, T-3, K, 3)

    
    valid_j = valid_a[:, 1:] & valid_a[:, :-1]             # (M, T-3)
    m_j = valid_j[:, :, None, None].to(jerk.dtype).expand_as(jerk)     # (M, T-3, K, 3)

    # --- Masked mean & variance of jerk over time ---
    count = m_j.sum(dim=1)                                  # (M, K, 3)
    sum_jerk = (jerk * m_j).sum(dim=1)                      # (M, K, 3)
    mean_jerk = sum_jerk / count.clamp(min=1.0)             # (M, K, 3)

    jerk_abs = jerk.abs()                               # (M, T-3, K, 3)

    # masked mean of absolute jerk
    mean_abs_jerk = (jerk_abs * m_j).sum(dim=(1, 2, 3)) / m_j.sum(dim=(1, 2, 3)).clamp(min=1.0)  # (M,)

    return mean_abs_jerk
