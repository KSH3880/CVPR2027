"""Shared task graph storage, goal markers and resume identity."""

from dataclasses import dataclass
import torch

HOLDING, AT = (6, 7)
PUSH_CONTACT = 11


@dataclass(frozen=True)
class EdgeContextGraph:
    ids: tuple
    num_agents: int
    num_objects: int
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_relation: torch.Tensor
    edge_owner: torch.Tensor
    edge_valid: torch.Tensor
    required_goal: torch.Tensor
    prereq_mask: torch.Tensor
    term_index: torch.Tensor

    def to(self, device):
        from dataclasses import fields

        return EdgeContextGraph(
            **{
                f.name: (
                    getattr(self, f.name).to(device)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )


def at_goal_marker_slots(graph):
    at_edges = graph.edge_valid & (graph.edge_relation == AT)
    slots = (graph.edge_dst - graph.num_agents - graph.num_objects).clamp(
        0, graph.num_agents - 1
    )
    active = torch.zeros(
        (at_edges.shape[0], graph.num_agents), device=slots.device, dtype=torch.long
    )
    active.scatter_add_(1, slots, at_edges.long())
    return (at_edges, slots, active.bool())


def task_instance(spec, num_agents, num_objects):
    return dict(graph=spec, num_agents=num_agents, num_objects=num_objects)


def check_task_resume(weights, spec, num_agents, num_objects):
    if weights.get("relation_task_instance") != task_instance(
        spec, num_agents, num_objects
    ):
        raise ValueError(
            "Training resume requires identical task graph and entity counts"
        )
