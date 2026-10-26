"""Paired PUSH initial states. Pure NumPy; never changes the training sampler."""
import numpy as np


MODES = ('direct', 'around')


def make_start_plan(base, mode, seed):
    """Place the human on the goal axis and face the box in both conditions.

    Preserve box/goal/loco DOFs and use the same radius for each paired scene.
    Radius is U(1,10), rejecting starts within .65m of the box's radial surface.
    Rotate the entire human state, including root/body velocities, about its root.
    """
    if mode not in MODES:
        raise ValueError(mode)
    plan = {k: v.copy() for k, v in base.items()}
    root = base['humanoid_root_state'][:, 0]
    box = base['physical_boxes'][:, 0]
    delta = base['goals'][:, 0, :2] - box[:, :2]
    direction = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
    q = box[:, 3:7]
    yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.stack((c*direction[:, 0]+s*direction[:, 1],
                      -s*direction[:, 0]+c*direction[:, 1]), -1)
    surface = np.min(base['physical_sizes'][:, 0, :2]/2 / np.maximum(np.abs(local), 1e-9), axis=-1)
    rng = np.random.RandomState(seed)
    radius = rng.uniform(1, 10, len(root))
    pending = radius < surface + .65
    while pending.any():
        radius[pending] = rng.uniform(1, 10, pending.sum())
        pending = radius < surface + .65
    sign = -1 if mode == 'direct' else 1
    new_xy = box[:, :2] + sign*radius[:, None]*direction
    facing = -sign*direction
    old_q = root[:, 3:7].astype(np.float64)
    old_q /= np.linalg.norm(old_q, axis=-1, keepdims=True)
    heading = np.arctan2(2*(old_q[:, 3]*old_q[:, 2]+old_q[:, 0]*old_q[:, 1]),
                         1-2*(old_q[:, 1]**2+old_q[:, 2]**2))
    turn = np.arctan2(facing[:, 1], facing[:, 0]) - heading

    def transform(states):
        shape = (len(root),) + (1,)*(states.ndim-2)
        c, s = np.cos(turn).reshape(shape), np.sin(turn).reshape(shape)
        old_origin = root[:, :2].reshape(shape+(2,))
        new_origin = new_xy.reshape(shape+(2,))
        xy = states[..., :2] - old_origin
        states[..., :2] = np.stack((c*xy[..., 0]-s*xy[..., 1], s*xy[..., 0]+c*xy[..., 1]), -1) + new_origin
        for start in (7, 10):
            xy = states[..., start:start+2].copy()
            states[..., start:start+2] = np.stack((c*xy[..., 0]-s*xy[..., 1], s*xy[..., 0]+c*xy[..., 1]), -1)
        q = states[..., 3:7].astype(np.float64)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        c, s = np.cos(turn/2).reshape(shape), np.sin(turn/2).reshape(shape)
        states[..., 3:7] = np.stack((c*q[..., 0]-s*q[..., 1], c*q[..., 1]+s*q[..., 0],
                                     c*q[..., 2]+s*q[..., 3], c*q[..., 3]-s*q[..., 2]), -1)

    transform(plan['humanoid_root_state'])
    transform(plan['kinematic_humanoid'])
    plan['push_start_mode'] = np.array(mode)
    plan['push_start_radius_m'] = radius.astype(np.float32)
    plan['push_start_surface_clearance_m'] = (radius-surface).astype(np.float32)
    validate_start_plan(plan, mode)
    return plan


def validate_start_plan(plan, mode):
    if str(plan['push_start_mode']) != mode:
        raise ValueError('Initial plan start mode does not match the evaluation')
    root = plan['humanoid_root_state'][:, 0]
    box = plan['physical_boxes'][:, 0, :2]
    direction = plan['goals'][:, 0, :2]-box
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    offset = root[:, :2]-box
    radius = np.linalg.norm(offset, axis=-1)
    sign = -1 if mode == 'direct' else 1
    # Isaac Gym's grid uses large world coordinates; float32 subtraction can
    # lose ~1e-4m even though the relative placement is exactly on the axis.
    np.testing.assert_allclose(offset/radius[:, None], sign*direction, atol=5e-4)
    q = root[:, 3:7]
    heading = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
    np.testing.assert_allclose(np.stack((np.cos(heading), np.sin(heading)), -1), -sign*direction, atol=2e-5)
    assert (plan['push_start_surface_clearance_m'] >= .65-1e-5).all()
    assert ((radius >= 1-1e-3) & (radius <= 10+1e-3)).all()
