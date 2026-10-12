"""Pure tensor state-relation rewards. No simulator imports or mutation of input history."""
import math
import torch


def evaluate_holding(hands_pos, object_pos, hand_scale=5.):
    error = torch.linalg.vector_norm(hands_pos.mean(-2) - object_pos, dim=-1)
    return torch.exp(-hand_scale * error.square()), error


def evaluate_at(object_pos, goal_pos, near_scale=10.,
                xy_tolerance=.1, z_tolerance=.001, state_definition='box_near'):
    delta = goal_pos - object_pos
    xy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    z = delta[..., 2].abs()
    near = torch.exp(-near_scale * delta.square().sum(-1))
    put = (xy <= xy_tolerance) & (z <= z_tolerance)
    if state_definition != 'box_near':
        raise ValueError('Unsupported At state definition: ' + str(state_definition))
    return near, near, put, xy, z


def relation_gate(phi, beta=30., center=.8):
    return torch.sigmoid(beta * (phi - center))


def box_speed_penalty(previous, current, dt, coefficient=1., threshold=2.5):
    speed = torch.linalg.vector_norm((current - previous) / dt, dim=-1)
    return -coefficient * (1 - torch.exp(-2 * (speed.clamp_min(threshold) - threshold).square()))


def direction_progress(source_prev, source_next, target_next, dt, eps=1e-6):
    """Positive XY motion cosine; no distance pinning or object-height mask."""
    if dt <= 0 or eps <= 0:
        raise ValueError('control dt and normalization epsilon must be positive')
    delta = target_next[..., :2] - source_next[..., :2]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    direction = delta / distance.clamp_min(eps).unsqueeze(-1)
    velocity = ((source_next - source_prev) / dt)[..., :2]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    cosine = (velocity * direction).sum(-1) / speed.clamp_min(eps)
    return torch.where(distance > eps, cosine.clamp(0., 1.), torch.zeros_like(cosine))


def distance_progress(source, target, delta=.5, sigma=1.):
    """Relation-agnostic current XY approach; no motion, Z, or type inputs."""
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError('distance delta must be finite and nonnegative')
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('distance sigma must be finite and positive')
    distance = torch.linalg.vector_norm(source[..., :2] - target[..., :2], dim=-1)
    return 1. / (1. + (distance - delta).clamp_min(0.) / sigma)


def relation_progress(source_prev, source_next, target_next, dt, config=None):
    """Select current-distance or all-edge direction progress."""
    cfg = config or {}
    kind = cfg.get('kind', 'distance')
    if kind == 'distance':
        return distance_progress(source_next, target_next, cfg.get('delta', .5), cfg.get('sigma', 1.))
    eps = cfg.get('normalization_epsilon', 1e-6)
    if kind == 'direction':
        return direction_progress(source_prev, source_next, target_next, dt, eps)
    raise ValueError('Unsupported progress kind: ' + str(kind))


def prerequisite_minimum(values, mask):
    """Return the bottleneck prerequisite value, or one for a root edge."""
    return torch.where(
        (mask.unsqueeze(0) if mask.ndim == 2 else mask), values.unsqueeze(1), torch.ones_like(values).unsqueeze(1)
    ).amin(-1)


def pin_progress(progress, gate):
    """Blend raw progress toward its maximum as the relation is satisfied."""
    return (1 - gate) * progress + gate


def approach_satisfaction(distance_xy, radius=.5):
    """XY approach credit: half credit at radius, full credit at zero distance."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('approach radius must be finite and positive')
    return 1. / (1. + (distance_xy / radius).square())


def prerequisite_all(values, mask):
    return torch.where((mask.unsqueeze(0) if mask.ndim == 2 else mask), values.unsqueeze(1), torch.ones_like(values).unsqueeze(1)).all(-1)


def current_target_success(satisfied, graph, at_z_error=None, z_tolerance=None):
    """Current-state success, independent of prerequisite achievement history."""
    valid = satisfied & prerequisite_all(satisfied, graph.prereq_mask)
    target = valid[:, graph.subgoal_target]
    if z_tolerance is not None:
        if at_z_error is None or at_z_error.shape != target.shape:
            raise ValueError('Success Z errors must have shape [N,M]')
        if at_z_error.device != satisfied.device:
            raise ValueError('Success Z errors must share the relation device')
        target = target & (at_z_error.abs() <= z_tolerance)
    return target


def current_at_success(satisfied, graph, at_z_error, z_tolerance):
    """Current At/Z success only; deliberately independent of Holding/history."""
    target = satisfied[:, graph.subgoal_target]
    if at_z_error is None or at_z_error.shape != target.shape:
        raise ValueError('Current saturation Z errors must have shape [N,M]')
    if at_z_error.device != satisfied.device:
        raise ValueError('Current saturation Z errors must share the relation device')
    return target & (at_z_error.abs() <= z_tolerance)


def advance_relation_history(satisfied_next, achieved_prev, done_prev, graph, target_success=None):
    valid = satisfied_next & prerequisite_all(achieved_prev, graph.prereq_mask)
    if target_success is not None:
        valid[:, graph.subgoal_target] = target_success
    history_live = ~done_prev[:, graph.edge_owner]
    achieved = achieved_prev | (valid & history_live)
    target_valid = valid[:, graph.subgoal_target]
    first = target_valid & ~done_prev
    return achieved, done_prev | target_valid, valid, first


def relation_step(phi_prev, phi_next, progress, achieved_prev, done_prev, graph,
                  state_weight=.2, progress_weight=.2, success_bonus=0.,
                  beta=30., gate_center=.8, satisfaction_threshold=.9, validate=True,
                  edge_distance_xy=None, approach_radius=None,
                  require_current_target_prerequisites=False, at_z_error=None,
                  success_z_tolerance=None, saturate_edge_rewards=False, progress_kind='distance',
                  saturate_edge_rewards_while_current=False,
                  current_saturation_z_tolerance=None, current_success_reward=0.):
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
        if not torch.isfinite(progress).all() or (progress < 0).any() or (progress > 1).any():
            raise ValueError('progress must be finite values in [0,1]')
    gate = relation_gate(phi_prev, beta, gate_center)
    activation = prerequisite_minimum(gate, graph.prereq_mask)
    if progress_kind == 'distance':
        if approach_radius is not None:
            raise ValueError('distance progress cannot use approach blending')
        # Diagnostic blend weight zero: paid progress is exactly the supplied
        # distance score, NOT self-gated or pinned. Prerequisite activation stays.
        progress_blend = torch.zeros_like(progress)
    elif progress_kind == 'direction':
        if approach_radius is None:
            raise ValueError('direction progress requires approach_radius')
        if edge_distance_xy is None or edge_distance_xy.shape != phi_prev.shape:
            raise ValueError('Edge approach progress requires XY distances with shape [N,E]')
        if edge_distance_xy.device != phi_prev.device or edge_distance_xy.dtype != phi_prev.dtype:
            raise ValueError('Edge XY distances must share phi device and dtype')
        if validate and (not torch.isfinite(edge_distance_xy).all() or (edge_distance_xy < 0).any()):
            raise ValueError('Edge XY distances must be finite and nonnegative')
        # Every edge uses the same distance-based approach blend. State-derived
        # gates remain solely in prerequisite activation, not self-pinning.
        progress_blend = approach_satisfaction(edge_distance_xy, approach_radius)
    else:
        raise ValueError('Unsupported progress kind: ' + str(progress_kind))
    pinned_progress = progress if progress_kind == 'distance' else pin_progress(progress, progress_blend)
    reward_mask = graph.edge_mask.to(phi_prev.dtype)
    state = state_weight * activation * phi_next * reward_mask
    progress_reward = progress_weight * activation * pinned_progress * reward_mask
    satisfied = phi_next >= satisfaction_threshold
    if saturate_edge_rewards and saturate_edge_rewards_while_current:
        raise ValueError('Latched and current edge saturation are mutually exclusive')
    current_success_state = torch.zeros_like(done_prev)
    if saturate_edge_rewards_while_current or current_success_reward > 0:
        if current_saturation_z_tolerance is None:
            raise ValueError('Current edge saturation or current success reward requires a Z tolerance')
        current_success_state = current_at_success(
            satisfied, graph, at_z_error, current_saturation_z_tolerance)
    target_success = None
    if require_current_target_prerequisites:
        target_success = current_target_success(satisfied, graph, at_z_error, success_z_tolerance)
    achieved, done, valid, first = advance_relation_history(
        satisfied, achieved_prev, done_prev, graph, target_success)
    raw_state, raw_progress = state, progress_reward
    saturation_active = (done if saturate_edge_rewards else current_success_state
                         if saturate_edge_rewards_while_current else torch.zeros_like(done))
    if saturate_edge_rewards or saturate_edge_rewards_while_current:
        # Override only paid rewards, AFTER prerequisite gating. Actual state,
        # gates, progress and observations remain untouched. The current mode
        # drops this override immediately when At/Z leaves the success state.
        saturated_edges = saturation_active[:, graph.edge_owner]
        state = torch.where(saturated_edges, state_weight * reward_mask, state)
        progress_reward = torch.where(saturated_edges, progress_weight * reward_mask, progress_reward)
    agent = torch.zeros_like(done_prev, dtype=phi_prev.dtype)
    agent.scatter_add_(1, graph.edge_owner.unsqueeze(0).expand(phi_prev.shape[0], -1), state + progress_reward)
    first_bonus = success_bonus * first.to(phi_prev.dtype)
    current_bonus = current_success_reward * current_success_state.to(phi_prev.dtype)
    bonus = first_bonus + current_bonus
    return dict(agent_task_reward=agent + bonus, edge_reward=state + progress_reward,
                activation=activation, pinned_progress=pinned_progress, progress_blend=progress_blend,
                state_component=state,
                raw_state_component=raw_state, raw_progress_component=raw_progress,
                progress_component=progress_reward, success_bonus=bonus,
                first_success_bonus=first_bonus, current_success_reward=current_bonus,
                first_success=first,
                achieved_next=achieved, done_next=done, valid_next=valid,
                gate_next=relation_gate(phi_next, beta, gate_center), satisfied_next=satisfied,
                current_success_state=current_success_state, saturation_active=saturation_active)


class RelationRuntime:
    """Owns history, reset seeding and the rollout-observation suffix."""

    def __init__(self, num_envs, graph, config, device):
        self.graph, self.config = graph, config
        self.phi = torch.zeros(num_envs, graph.edge_src.numel(), device=device)
        self.achieved = torch.zeros_like(self.phi, dtype=torch.bool)
        self.done = torch.zeros(num_envs, graph.subgoal_target.numel(), dtype=torch.bool, device=device)

    def reset(self, env_ids, phi, at_z_error=None):
        self.phi[env_ids] = phi
        satisfied = phi >= self.config.get('satisfaction_threshold', .9)
        # Root relations are seeded from initial state; targets use these initial parents.
        from utils.ontop_task_spec import graph_for_envs
        graph = graph_for_envs(self.graph, env_ids)
        roots = ~graph.prereq_mask.any(-1)
        seeded = satisfied & (roots.unsqueeze(0) if roots.ndim == 1 else roots)
        success = self.config.get('success', {})
        target_success = None
        if success.get('require_current_target_prerequisites', False):
            target_success = current_target_success(
                satisfied, graph, at_z_error, success.get('z_tolerance'))
        achieved, done, _, _ = advance_relation_history(
            satisfied, seeded, torch.zeros_like(self.done[env_ids]), graph, target_success)
        self.achieved[env_ids], self.done[env_ids] = achieved, done

    def step(self, phi, progress, edge_distance_xy=None, at_z_error=None):
        c = self.config
        result = relation_step(self.phi, phi, progress, self.achieved, self.done, self.graph,
            c.get('state_reward_weight', .2), c.get('progress_reward_weight', .2),
            c.get('subgoal_success_bonus', 0.), c.get('soft_gate', {}).get('beta', 30.),
            c.get('soft_gate', {}).get('center', .8), c.get('satisfaction_threshold', .9),
            validate=c.get('diagnostics', {}).get('validate_tensors', False),
            edge_distance_xy=edge_distance_xy,
            approach_radius=c.get('progress', {}).get('approach_radius'),
            require_current_target_prerequisites=c.get('success', {}).get('require_current_target_prerequisites', False),
            at_z_error=at_z_error,
            success_z_tolerance=c.get('success', {}).get('z_tolerance'),
            saturate_edge_rewards=c.get('success', {}).get('saturate_edge_rewards', False),
            progress_kind=c.get('progress', {}).get('kind', 'distance'),
            saturate_edge_rewards_while_current=c.get('success', {}).get(
                'saturate_edge_rewards_while_current', False),
            current_saturation_z_tolerance=c.get('success', {}).get(
                'current_saturation_z_tolerance'),
            current_success_reward=c.get('success', {}).get('current_success_reward', 0.))
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
