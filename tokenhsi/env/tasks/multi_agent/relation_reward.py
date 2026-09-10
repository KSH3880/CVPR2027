"""Pure tensor state-relation rewards. No simulator imports or mutation of input history."""
import torch


def evaluate_holding(hands_pos, object_pos, hand_scale=5.):
    error = torch.linalg.vector_norm(hands_pos.mean(-2) - object_pos, dim=-1)
    return torch.exp(-hand_scale * error.square()), error


def evaluate_at(object_pos, goal_pos, near_scale=10., alpha=.5,
                xy_tolerance=.1, z_tolerance=.001):
    delta = goal_pos - object_pos
    xy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    z = delta[..., 2].abs()
    near = torch.exp(-near_scale * delta.square().sum(-1))
    put = (xy <= xy_tolerance) & (z <= z_tolerance)
    return near * (alpha + (1 - alpha) * put.float()), near, put, xy, z


def relation_gate(phi, beta=30., center=.8):
    return torch.sigmoid(beta * (phi - center))


def velocity_progress(source_prev, source_next, target_next, dt,
                      target_speed=1.5, velocity_scale=5., eps=1e-6, mode='gaussian'):
    """Source XY velocity along the post-step target direction.

    gaussian preserves v0/state2 exactly. signed_linear uses target_speed as
    a normalization/saturation speed, not a preferred-speed Gaussian peak.
    Neither mode measures relative velocity to a moving target.
    """
    if dt <= 0:
        raise ValueError('control dt must be positive')
    if target_speed <= 0:
        raise ValueError('target_speed must be positive')
    delta = target_next[..., :2] - source_next[..., :2]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    direction = delta / distance.clamp_min(eps).unsqueeze(-1)
    along = (((source_next - source_prev) / dt)[..., :2] * direction).sum(-1)
    if mode == 'gaussian':
        p = torch.exp(-velocity_scale * (target_speed - along).square())
        return torch.where((distance > eps) & (along > 0), p, torch.zeros_like(p))
    if mode == 'signed_linear':
        p = (along / target_speed).clamp(-1., 1.)
        return torch.where(distance > eps, p, torch.zeros_like(p))
    raise ValueError('Unsupported progress mode: ' + str(mode))


def box_speed_penalty(previous, current, dt, coefficient=1., threshold=2.5):
    speed = torch.linalg.vector_norm((current - previous) / dt, dim=-1)
    return -coefficient * (1 - torch.exp(-2 * (speed.clamp_min(threshold) - threshold).square()))


def prerequisite_product(values, mask):
    return torch.where(mask.unsqueeze(0), values.unsqueeze(1), torch.ones_like(values).unsqueeze(1)).prod(-1)


def prerequisite_all(values, mask):
    return torch.where(mask.unsqueeze(0), values.unsqueeze(1), torch.ones_like(values).unsqueeze(1)).all(-1)


def advance_relation_history(satisfied_next, achieved_prev, done_prev, graph):
    valid = satisfied_next & prerequisite_all(achieved_prev, graph.prereq_mask)
    live = ~done_prev[:, graph.edge_owner]
    achieved = achieved_prev | (valid & live)
    target_valid = valid[:, graph.subgoal_target]
    first = target_valid & ~done_prev
    return achieved, done_prev | target_valid, valid, first


def relation_step(phi_prev, phi_next, progress, achieved_prev, done_prev, graph,
                  state_weight=1., velocity_weight=.2, success_bonus=5.,
                  beta=30., gate_center=.8, satisfaction_threshold=.9, validate=True,
                  progress_mode='gaussian'):
    if progress_mode not in ('gaussian', 'signed_linear'):
        raise ValueError('Unsupported progress mode: ' + str(progress_mode))
    if phi_prev.shape != phi_next.shape or progress.shape != phi_prev.shape:
        raise ValueError('phi/progress shapes must agree')
    if phi_prev.ndim != 2 or phi_prev.shape[1] != graph.edge_src.numel():
        raise ValueError('phi must have shape [N,E] matching the compiled graph')
    if achieved_prev.shape != phi_prev.shape or done_prev.shape != (phi_prev.shape[0], graph.subgoal_target.numel()):
        raise ValueError('history must have shapes achieved[N,E], done[N,M]')
    if achieved_prev.dtype != torch.bool or done_prev.dtype != torch.bool:
        raise ValueError('history flags must be boolean')
    if not phi_prev.is_floating_point() or phi_next.dtype != phi_prev.dtype or progress.dtype != phi_prev.dtype:
        raise ValueError('phi and progress must share a floating point dtype')
    if any(t.device != phi_prev.device for t in (phi_next, progress, achieved_prev, done_prev, graph.edge_src)):
        raise ValueError('relation tensors must share one device')
    if validate:
        # CPU fixtures/debug only. Production runtime avoids per-step GPU synchronization.
        values = torch.stack([phi_prev, phi_next])
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('phi must be finite values in [0,1]')
        minimum = -1. if progress_mode == 'signed_linear' else 0.
        if not torch.isfinite(progress).all() or (progress < minimum).any() or (progress > 1).any():
            raise ValueError('progress must be finite values in [{},1]'.format(minimum))
    gate = relation_gate(phi_prev, beta, gate_center)
    activation = prerequisite_product(gate, graph.prereq_mask)
    live = (~done_prev[:, graph.edge_owner]).to(phi_prev.dtype) * graph.edge_mask
    state = state_weight * activation * (phi_next - phi_prev) * live
    velocity = velocity_weight * activation * (1 - gate) * progress * live
    satisfied = phi_next >= satisfaction_threshold
    achieved, done, valid, first = advance_relation_history(satisfied, achieved_prev, done_prev, graph)
    agent = torch.zeros_like(done_prev, dtype=phi_prev.dtype)
    agent.scatter_add_(1, graph.edge_owner.unsqueeze(0).expand(phi_prev.shape[0], -1), state + velocity)
    bonus = success_bonus * first.to(phi_prev.dtype)
    return dict(agent_task_reward=agent + bonus, edge_reward=state + velocity,
                activation=activation, delta_phi=phi_next - phi_prev, state_component=state,
                velocity_component=velocity, success_bonus=bonus, first_success=first,
                achieved_next=achieved, done_next=done, valid_next=valid,
                gate_next=relation_gate(phi_next, beta, gate_center), satisfied_next=satisfied)


class RelationRuntime:
    """Owns history, reset seeding and the rollout-observation suffix."""

    def __init__(self, num_envs, graph, config, device):
        self.graph, self.config = graph, config
        self.phi = torch.zeros(num_envs, graph.edge_src.numel(), device=device)
        self.achieved = torch.zeros_like(self.phi, dtype=torch.bool)
        self.done = torch.zeros(num_envs, graph.subgoal_target.numel(), dtype=torch.bool, device=device)

    def reset(self, env_ids, phi):
        self.phi[env_ids] = phi
        satisfied = phi >= self.config.get('satisfaction_threshold', .9)
        # Root relations are seeded from initial state; targets use these initial parents.
        seeded = satisfied & ~self.graph.prereq_mask.any(-1).unsqueeze(0)
        achieved, done, _, _ = advance_relation_history(
            satisfied, seeded, torch.zeros_like(self.done[env_ids]), self.graph)
        self.achieved[env_ids], self.done[env_ids] = achieved, done

    def step(self, phi, progress):
        c = self.config
        result = relation_step(self.phi, phi, progress, self.achieved, self.done, self.graph,
            c.get('state_delta_weight', 1.), c.get('velocity_progress_weight', .2),
            c.get('subgoal_success_bonus', 5.), c.get('soft_gate', {}).get('beta', 30.),
            c.get('soft_gate', {}).get('center', .8), c.get('satisfaction_threshold', .9),
            validate=c.get('diagnostics', {}).get('validate_tensors', False),
            progress_mode=c.get('progress', {}).get('mode', 'gaussian'))
        self.phi.copy_(phi)
        self.achieved.copy_(result['achieved_next'])
        self.done.copy_(result['done_next'])
        return result

    def suffix(self, env_ids=None):
        phi = self.phi if env_ids is None else self.phi[env_ids]
        achieved = self.achieved if env_ids is None else self.achieved[env_ids]
        done = self.done if env_ids is None else self.done[env_ids]
        gate = relation_gate(phi, self.config.get('soft_gate', {}).get('beta', 30.),
                             self.config.get('soft_gate', {}).get('center', .8))
        satisfied = phi >= self.config.get('satisfaction_threshold', .9)
        return torch.cat([torch.stack([phi, gate, satisfied.float(), achieved.float()], -1).flatten(1),
                          done.float()], -1)
