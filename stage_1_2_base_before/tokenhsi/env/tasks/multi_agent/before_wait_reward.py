"""Goal-relative waiting and yielding; no stored waypoint or action override."""
import math
import torch
from utils.edge_ontop_spec import batched, ON_TOP


def validate_wait_clear(option):
    positive = ('human_radius', 'clearance', 'safe_margin', 'wait_band')
    nonnegative = ('blocking_weight', 'motion_weight')
    if not isinstance(option, dict) or set(option) != set(positive + nonnegative):
        raise ValueError('wait_clear requires the six documented parameters')
    for key in positive + nonnegative:
        value = option[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (key in positive and value == 0)):
            raise ValueError('Invalid wait_clear parameter: ' + key)


def segment_distance(point, start, end):
    """XY distance to a finite segment; zero-length segments become disks."""
    delta = end - start
    t = ((point-start)*delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)
    nearest = start + t.clamp(0, 1)[..., None]*delta
    return (point-nearest).norm(dim=-1)


def smoothstep(value):
    x = value.clamp(0, 1)
    return x*x*(3-2*x)


def wait_clear_penalties(roots, boxes, sizes, goals, graph, raw_success, reward_cfg, option):
    """Return per-person penalties, using physical state and current AT success.

    Only dependent followers receive penalties. Each yields to its own leader's
    human->goal and payload->goal capsules. ON_TOP also checks its own payload.
    Box bounding spheres conservatively cover any orientation. No RNG or state.
    """
    batch, people = roots.shape[:2]
    block = roots.new_zeros(batch, people)
    motion = torch.zeros_like(block)
    active_out = torch.zeros_like(block)
    wait_out = torch.zeros_like(block)
    infringement = torch.zeros_like(block)
    radius_out = torch.zeros_like(block)
    def output():
        return dict(blocking=block, waiting_motion=motion, waiting_active=active_out,
                    waiting_weight=wait_out, corridor_intrusion=infringement,
                    waiting_radius=radius_out)
    prereq = graph.prereq_mask
    if prereq.ndim == 2:
        prereq = prereq.unsqueeze(0).expand(batch, -1, -1)
    valid = batched(graph.edge_valid, batch)
    rows, edges = (valid & prereq.any(-1)).nonzero(as_tuple=True)
    if not len(rows):
        return output()
    predecessor = prereq[rows, edges].long().argmax(-1)
    owner = batched(graph.edge_owner, batch)
    src = batched(graph.edge_src, batch)
    dst = batched(graph.edge_dst, batch)
    relation = batched(graph.edge_relation, batch)
    follower = owner[rows, edges]
    leader = owner[rows, predecessor]
    payload = src[rows, predecessor] - people
    goal_id = dst[rows, predecessor] - people - graph.num_objects
    goal = goals[rows, goal_id, :2]
    human = roots[rows, follower]
    leader_human = roots[rows, leader, :2]
    leader_box = boxes[rows, payload, :2]
    leader_radius = sizes[rows, payload].norm(dim=-1)/2
    top = relation[rows, edges] == ON_TOP
    own_payload = (src[rows, edges]-people).clamp(0, graph.num_objects-1)
    own_box = boxes[rows, own_payload]
    own_radius = sizes[rows, own_payload].norm(dim=-1)/2
    human_radius = torch.full_like(leader_radius, option['human_radius'])

    def clearance_and_width(point, point_radius):
        human_width = point_radius + human_radius + option['clearance']
        box_width = point_radius + leader_radius + option['clearance']
        human_clear = segment_distance(point, leader_human, goal) - human_width
        box_clear = segment_distance(point, leader_box, goal) - box_width
        return torch.minimum(human_clear, box_clear), torch.maximum(human_width, box_width)

    human_clear, human_width = clearance_and_width(human[:, :2], human_radius)
    payload_clear, payload_width = clearance_and_width(own_box[:, :2], own_radius)
    clear = torch.where(top, torch.minimum(human_clear, payload_clear), human_clear)
    human_violation = (-human_clear/human_width).clamp(0, 1).square()
    payload_violation = (-payload_clear/payload_width).clamp(0, 1).square()
    violation = torch.where(top, torch.maximum(human_violation, payload_violation), human_violation)
    unfinished = (~raw_success[rows, predecessor]).to(roots.dtype)

    # Keep a feasible waiting annulus beyond the goal-end capsule: its center
    # cannot be inside the forbidden disk, even for large ON_TOP payloads.
    target = (dst[rows, edges]-people).clamp(0, graph.num_objects-1)
    support_radius = sizes[rows, target, :2].norm(dim=-1)/2
    buffers = reward_cfg['progress']['bbox_buffers']
    buffer = torch.where(top, float(buffers['ON_TOP']),
        torch.where(relation[rows, edges] == 9, float(buffers['SIT']), float(buffers['CLIMB'])))
    reference = torch.where(top[:, None], own_box[:, :2], human[:, :2])
    width = torch.where(top, payload_width, human_width)
    radius = torch.maximum(support_radius+buffer, width) + option['safe_margin']
    distance = (reference-goal).norm(dim=-1)
    near = smoothstep(1-(distance-radius).abs()/option['wait_band'])
    safe = smoothstep(clear/option['safe_margin'])
    weight = unfinished*near*safe
    speed2 = human[:, 7:9].square().sum(-1)
    speed2 = speed2 + torch.where(top, own_box[:, 7:9].square().sum(-1), 0.)
    block[rows, follower] = -option['blocking_weight']*unfinished*violation
    motion[rows, follower] = -option['motion_weight']*weight*speed2
    active_out[rows, follower] = unfinished
    wait_out[rows, follower] = weight
    infringement[rows, follower] = unfinished*violation
    radius_out[rows, follower] = radius
    return output()
