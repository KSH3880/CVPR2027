"""CPU-only geometry summaries for opt-in PUSH evaluation.

Hand face labels are XY proximity proxies, not physical contact measurements.
"""
import math


def local_xy(point, box):
    x, y, z, w = box[3:7]
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = point[0]-box[0], point[1]-box[1]
    return c*dx+s*dy, -s*dx+c*dy


def target_region(box, goal, size):
    """Tangential position of the opposite-goal ray hit, normalized to half-size."""
    x, y = local_xy(goal, box)
    x, y = abs(x)/(size[0]/2), abs(y)/(size[1]/2)
    ratio = min(x, y)/max(x, y) if max(x, y) > 1e-6 else 0.
    return 'face_center' if ratio <= .5 else 'corner' if ratio >= .8 else 'oblique'


def hand_face(point, box, size, tolerance=.18):
    x, y = local_xy(point, box)
    hx, hy = size[0]/2, size[1]/2
    distances = [math.hypot(x-sign*hx, y-max(-hy, min(hy, y))) for sign in (-1, 1)]
    distances += [math.hypot(x-max(-hx, min(hx, x)), y-sign*hy) for sign in (-1, 1)]
    order = sorted(range(4), key=distances.__getitem__)
    if distances[order[0]] > tolerance:
        return None
    # Corners have no unique face; avoid turning a tie into a two-face claim.
    if distances[order[1]]-distances[order[0]] < .02:
        return 'ambiguous'
    return order[0]


def hand_layout(hands, box, size):
    faces = [hand_face(p, box, size) for p in hands]
    if None in faces:
        return 'not_both_near'
    if 'ambiguous' in faces:
        return 'ambiguous_corner'
    return 'same_face' if faces[0] == faces[1] else 'different_faces'


def inverse_rotate(vector, quaternion):
    """World vector to box coordinates (xyzw quaternion)."""
    x, y, z, w = quaternion
    q = [-x, -y, -z]
    cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    uv = cross(q, vector)
    uuv = cross(q, uv)
    return [vector[i]+2*(w*uv[i]+uuv[i]) for i in range(3)]


def side_face_3d(point, box, size, tolerance=.18):
    """Closest finite side face of the tilted OBB; exclude top/bottom contacts."""
    p = inverse_rotate([point[i]-box[i] for i in range(3)], box[3:7])
    half = [v/2 for v in size]
    distances = []
    for axis in range(3):
        for sign in (-1, 1):
            nearest = [max(-h, min(h, v)) for h, v in zip(half, p)]
            nearest[axis] = sign*half[axis]
            distances.append(math.dist(p, nearest))
    order = sorted(range(6), key=distances.__getitem__)
    face = order[0]
    if distances[face] > tolerance:
        return None
    if distances[order[1]]-distances[face] < .02:
        return 'ambiguous'
    return face if face < 4 else 'top_bottom'


def physical_layout(hands, box, size, hand_forces=None, box_force=None):
    faces = [side_face_3d(p, box, size) for p in hands]
    if None in faces:
        return 'not_both_near', False
    if 'ambiguous' in faces:
        return 'ambiguous_corner', False
    if 'top_bottom' in faces:
        return 'top_bottom', False
    layout = 'same_face' if faces[0] == faces[1] else 'different_faces'
    contact = hand_forces is not None and box_force is not None
    if contact:
        # Net body forces are a contact proxy, not a collision-pair measurement.
        for force, face in zip(hand_forces, faces):
            local_force = inverse_rotate(force, box[3:7])
            contact &= local_force[face//2]*(-1 if face % 2 == 0 else 1) >= .5
        contact &= math.hypot(box_force[0], box_force[1]) >= .5
    return layout, bool(contact)


def accumulate(stats, layout, tilt, distance_gain, speed):
    row = stats.setdefault(layout, dict(steps=0, tilt_sum_deg=0., max_tilt_deg=0.,
        over10_steps=0, over15_steps=0, goal_distance_gain_m=0., speed_sum_mps=0.))
    row['steps'] += 1
    row['tilt_sum_deg'] += tilt
    row['max_tilt_deg'] = max(row['max_tilt_deg'], tilt)
    row['over10_steps'] += tilt > 10
    row['over15_steps'] += tilt > 15
    row['goal_distance_gain_m'] += distance_gain
    row['speed_sum_mps'] += speed


class PushFaceAudit:
    def __init__(self, boxes, goals, size, dt):
        self.size, self.dt = size, dt
        self.rows = []
        for env, (bs, gs) in enumerate(zip(boxes, goals)):
            for agent, (box, goal) in enumerate(zip(bs, gs)):
                self.rows.append(dict(env=env, agent=agent, initial_region=target_region(box, goal, size),
                    initial_box_xy=box[:2], initial_goal_xy=goal[:2], steps=0,
                    final_success=False, ever_success=False, first_success_seconds=None,
                    goal_distance_initial=math.dist(box[:2], goal[:2]), goal_distance_final=None,
                    layout_steps={}, moving_unsuccessful_layout_steps={},
                    obb_layout_stats={}, force_proxy_layout_stats={},
                    before_first_success_moving_obb_stats={},
                    before_first_success_moving_force_stats={}))

    def step(self, boxes, goals, hands, success, alive, hand_forces=None, box_forces=None):
        for row in self.rows:
            i, j = row['env'], row['agent']
            if not alive[i]:
                continue
            box, goal = boxes[i][j], goals[i][j]
            layout = hand_layout(hands[i][j], box, self.size)
            # Older callers/tests supply XY points only; retain their legacy audit.
            if all(len(p) >= 3 for p in hands[i][j]):
                obb_layout, contact = physical_layout(hands[i][j], box, self.size,
                    hand_forces[i][j] if hand_forces is not None else None,
                    box_forces[i][j] if box_forces is not None else None)
                tilt = math.degrees(math.acos(max(-1., min(1., 1-2*(box[3]**2+box[4]**2)))))
                distance = math.dist(box[:2], goal[:2])
                previous = row['goal_distance_final']
                gain = (row['goal_distance_initial'] if previous is None else previous)-distance
                speed = math.sqrt(sum(v*v for v in box[7:10]))
                accumulate(row['obb_layout_stats'], obb_layout, tilt, gain, speed)
                if contact:
                    accumulate(row['force_proxy_layout_stats'], obb_layout, tilt, gain, speed)
                if not row['ever_success'] and speed > .02 and not success[i][j]:
                    accumulate(row['before_first_success_moving_obb_stats'], obb_layout, tilt, gain, speed)
                    if contact:
                        accumulate(row['before_first_success_moving_force_stats'], obb_layout, tilt, gain, speed)
            row['steps'] += 1
            row['layout_steps'][layout] = row['layout_steps'].get(layout, 0)+1
            current = bool(success[i][j])
            if math.sqrt(sum(v*v for v in box[7:10])) > .02 and not current:
                counts = row['moving_unsuccessful_layout_steps']
                counts[layout] = counts.get(layout, 0)+1
            row['final_success'] = current
            if current and not row['ever_success']:
                row['first_success_seconds'] = row['steps']*self.dt
            row['ever_success'] |= current
            row['goal_distance_final'] = math.dist(box[:2], goal[:2])

    def result(self):
        return dict(definition='Initial opposite-goal ray hit: face_center ratio<=0.5; corner ratio>=0.8; otherwise oblique. '
                    'Hand layouts use closest XY face within 0.18m; distance ties within 0.02m are ambiguous. '
                    'No hand height/force condition. Layout-conditioned counts are observational, not causal.',
                    physical_definition='OBB side-face labels use full quaternion, finite 3D faces within 0.18m, '
                    'ties within 0.02m ambiguous; top/bottom excluded. Force proxy additionally requires '
                    'each hand net force along its own outward face normal >=0.5N and box horizontal net force >=0.5N; '
                    'sampled at control boundaries, not exact collision pairs or physics-tick OR. '
                    'Pre-success moving excludes current/previous AT success and requires box speed >0.02m/s. '
                    'Observational groups do not establish causality.',
                    agents=self.rows)
