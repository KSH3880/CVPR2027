import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from torch import nn
import yaml

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.before_spec import checkpoint_contract
from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.before_network import transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def check_at_progress(task, env_ids=None):
    if task.cfg['env']['before'].get('progress_target') != 'predecessor_at_goal':
        return {}
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from utils.edge_ontop_spec import select_graph
    ids = torch.arange(task.num_envs, device=task.device) if env_ids is None else env_ids
    phi, diag = task._evaluate_ontop_context(env_ids)
    native_phi, native = HumanoidMACarry._evaluate_ontop_context(task, env_ids)
    ordinary = task.relation_runtime.graph.edge_relation[ids] != 12
    torch.testing.assert_close(phi[ordinary], native_phi[ordinary], rtol=0, atol=0)
    for key in native:
        if key != 'progress':
            torch.testing.assert_close(diag[key][ordinary], native[key][ordinary], rtol=0, atol=0)
    graph = select_graph(task.relation_runtime.graph, ids)
    dependent = graph.prereq_mask.any(-1)
    assert torch.equal(diag['progress'][~dependent & ordinary], native['progress'][~dependent & ordinary])
    rows, edges = dependent.nonzero(as_tuple=True)
    es, follower = ids[rows], graph.edge_owner[rows, edges]
    leader = task._before_predecessor[es, follower]
    assert (leader >= 0).all()
    goal = task._scenario_goal(es, leader)
    bodies = (task._rigid_body_pos if env_ids is None
              else task._kinematic_humanoid_rigid_body_states[ids, ..., :3])
    root = bodies[rows, follower, 0, :2]
    source = task._scenario_physical_object(es, follower)
    top = task._templates[es, follower] == 4
    point = torch.where(top[:, None], task._box_states[es, source, :2], root)
    distance = (point-task._tar_pos[es, goal, :2]).norm(dim=-1)
    support = task._scenario_physical_object(es, leader)
    radius = task._box_size[es, support, :2].norm(dim=-1)/2 + torch.where(top, .1, .3)
    expected = 1/(1+(distance-radius).clamp_min(0))
    torch.testing.assert_close(diag['progress'][rows, edges], expected)
    return dict(dependent_edges=len(rows), progress_goal_verified=True,
                state_success_geometry_unchanged=True, other_progress_unchanged=True)
