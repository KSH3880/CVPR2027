from check_before_pickup_carry_rsi import main
from check_before_virtual_object import check_virtual
import torch

from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from utils.edge_ontop_spec import select_graph


def check_progress(task, ids=None):
    phi, diag = task._evaluate_ontop_context(ids)
    option = task.cfg['env']['before']
    enabled = option['hard_gate_progress']
    option['hard_gate_progress'] = False
    try:
        old_phi, old = task._evaluate_ontop_context(ids)
    finally:
        option['hard_gate_progress'] = enabled
    assert torch.equal(phi, old_phi)
    for key in diag:
        if key != 'progress' and isinstance(diag[key], torch.Tensor):
            assert torch.equal(diag[key], old[key])
    graph = task.relation_runtime.graph if ids is None else select_graph(task.relation_runtime.graph, ids)
    raw = interaction_own_success(phi, diag['z_error'], diag['feet_height_error'],
        graph, task._relation_cfg, diag['region_error'])
    dependent = graph.prereq_mask.any(-1)
    row, edge = dependent.nonzero(as_tuple=True)
    predecessor = graph.prereq_mask[row, edge].long().argmax(-1)
    goals = task._tar_pos if ids is None else task._tar_pos[ids]
    bodies = task._rigid_body_pos if ids is None else task._kinematic_humanoid_rigid_body_states[ids, ..., :3]
    boxes = task._logical_box_values(task._box_states, ids)
    sizes = task._logical_box_values(task._box_size, ids)
    source = graph.edge_src[row, edge]
    top = graph.edge_relation[row, edge] == 8
    xy = torch.where(top[:, None], boxes[row, (source-task.num_agents).clamp(0, task.num_objects-1), :2],
        bodies[row, source.clamp(0, task.num_agents-1), 0, :2])
    goal = graph.edge_dst[row, predecessor]-task.num_agents-task.num_objects
    distance = (xy-goals[row, goal, :2]).norm(dim=-1)
    radius = sizes[row, graph.edge_dst[row, edge]-task.num_agents, :2].norm(dim=-1)/2
    radius += torch.where(top, .1, .3)
    failed = ~raw[row, predecessor]
    expected = torch.where(failed & (distance <= option.get('hard_gate_radius', radius)), 0., 1/(1+(distance-radius).clamp_min(0)))
    torch.testing.assert_close(diag['progress'][row, edge], expected, rtol=0, atol=0)
    assert torch.equal(diag['progress'][~dependent], old['progress'][~dependent])
    if ids is None:
        args = (diag['z_error'], diag['feet_height_error'], diag['region_error'], graph)
        a = task.relation_runtime.components(phi, diag['progress'], *args)
        b = task.relation_runtime.components(phi, old['progress'], *args)
        assert torch.equal(a['raw_own_success'][row, predecessor], raw[row, predecessor])
        assert torch.equal(a['prerequisite_gate'][row, edge], ~failed)
        for key in ('state_component', 'success_component', 'own_success', 'reward_saturated'):
            assert torch.equal(a[key], b[key])
        assert torch.equal(a['total'][~dependent], b['total'][~dependent])
    return raw


def check_switch(task):
    graph = task.relation_runtime.graph
    row, edge = graph.prereq_mask.any(-1).nonzero(as_tuple=True)
    predecessor = graph.prereq_mask[row, edge].long().argmax(-1)
    if not len(row):
        return
    m = task.num_agents
    boxes = task._logical_box_values(task._box_states)
    source = graph.edge_src[row, edge]
    top = graph.edge_relation[row, edge] == 8
    xy = torch.where(top[:, None], boxes[row, (source-m).clamp(0, task.num_objects-1), :2],
        task._rigid_body_pos[row, source.clamp(0, m-1), 0, :2])
    goal = graph.edge_dst[row, predecessor]-m-task.num_objects
    physical = task._logical_box_order[row, graph.edge_src[row, predecessor]-m]
    values = (task._box_states, task._tar_pos, task.relation_runtime.own_success, task.relation_runtime.achieved)
    saved = [v.clone() for v in values]
    last_diag = task._ontop_last_diag
    try:
        task._tar_pos[row, goal, :2] = xy
        for phase in (0, 1, 0):
            expected = (torch.arange(len(row), device=task.device)+phase) % 2 == 0
            task._box_states[row, physical, :3] = task._tar_pos[row, goal]
            task._box_states[row, physical, 0] += (~expected)*.3
            task.relation_runtime.own_success[row, predecessor] = ~expected
            task.relation_runtime.achieved.fill_(True)
            raw = check_progress(task)
            assert torch.equal(raw[row, predecessor], expected)
    finally:
        for value, before in zip(values, saved):
            value.copy_(before)
        task._ontop_last_diag = last_diag


def inspect(task):
    check_virtual(task, task._compute_clean_scene_obs())
    check_progress(task)
    check_progress(task, torch.tensor([3, 1], device=task.device))
    check_switch(task)
    return dict(hard_gate_progress=True, same_frame_at_switch=True, state_success_holding_unchanged=True,
        virtual_xyz_real_quaternion=True, physical_state_original_graph_unchanged=True)


def step_inspector(task, obs):
    check_virtual(task, obs)
    check_progress(task)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    Path(sys.argv[sys.argv.index('--output_path')+1]).mkdir(parents=True, exist_ok=True)
    preset = sys.argv[sys.argv.index('--task_graph')+1]
    cases = (preset,) if preset in ('before_sit', 'before_climb') else (
        'random_scenario', 'before_stack', 'one_pair_stack', 'all_solo', 'push')
    main(inspector=inspect, presets=[(p, 'loco') for p in cases], step_inspector=step_inspector)
