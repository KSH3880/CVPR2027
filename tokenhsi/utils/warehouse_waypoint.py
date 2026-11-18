"""Shared inference/training geometry and a replayable fixed-radius state machine."""
import torch
from qualitative.warehouse_waypoint import ONTOP_TEMPLATE, door_box, waypoint_scene, worker_spawns
from utils.edge_context_spec import HOLDING
from utils.sensor_stage2_spec import PUSH, WAREHOUSE


OBS_WIDTH = 5
ROLE_OBS_WIDTH = 7
BOX_ONLY_VERSION = 3


def role_eligibility(scene_ids, before_pairs, predecessor, templates):
    paired = predecessor >= 0
    eligible = ((scene_ids == WAREHOUSE) & ~before_pairs
                & (paired.sum(-1) == 1) & ((templates == PUSH).sum(-1) == 2))
    follower = paired.long().argmax(-1)
    leader = predecessor.gather(1, follower[:, None]).squeeze(1).clamp_min(0)
    ontop = templates.gather(1, follower[:, None]).squeeze(1) == ONTOP_TEMPLATE
    return torch.stack((leader, follower), -1), torch.stack((eligible, eligible & ontop), -1)


def advance_phase(phase, root_xy, payload_xy, door_xy, holding, radius):
    """-1 disabled, 0 inbound, 1 pickup, 2 outbound, 3 final; monotonic."""
    phase = torch.where((phase == 0) & ((root_xy-door_xy).norm(dim=-1) <= radius), 1, phase)
    phase = torch.where((phase == 1) & holding, 2, phase)
    return torch.where((phase == 2) & ((payload_xy-door_xy).norm(dim=-1) <= radius), 3, phase)


def role_holding(runtime, ids, people):
    graph = runtime.graph
    held_edges = graph.edge_valid[ids] & (graph.edge_relation[ids] == HOLDING) & runtime.own_success[ids]
    return (held_edges[:, None] & (graph.edge_owner[ids, None] == people[:, :, None])).any(-1)


def advance_box_only_phase(phase, root_xy, door_xy, holding, radius):
    """Switch the box once per episode and latch the first observed pickup."""
    reached = (root_xy-door_xy).norm(dim=-1) <= radius
    phase = torch.where((phase == 0) & reached, 1, phase)
    return torch.where(((phase == 0) | (phase == 1)) & holding, 3, phase)
