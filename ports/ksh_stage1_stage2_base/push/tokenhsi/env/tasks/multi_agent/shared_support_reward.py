import torch
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent, inner_xy_region_error


def evaluate_shared(task, ids, live):
    from utils.edge_ontop_spec import select_graph
    graph = select_graph(task.relation_runtime.graph, ids)
    bodies = task._rigid_body_pos[ids] if live else task._kinematic_humanoid_rigid_body_states[ids, ..., :3]
    hands, feet = bodies[:, :, task._key_body_ids[:2]], bodies[:, :, task._key_body_ids[-2:]]
    objects = task._logical_box_values(task._box_states, ids)
    sizes = task._logical_box_values(task._box_size, ids)
    phi, diag = evaluate_interaction_edges(hands, feet, bodies[:, :, 0], objects, sizes,
        task._tar_pos[ids], graph, task._relation_cfg, task._char_h)
    owner = graph.edge_owner
    shared = task._shared[ids].gather(1, owner) & graph.edge_valid & (graph.edge_relation != 6)
    row = torch.arange(len(ids), device=ids.device)[:, None]
    physical = task._support_physical[ids].gather(1, owner).clamp_min(0)
    support, size = task._box_states[ids][row, physical], task._box_size[ids][row, physical]
    top = support[..., 2]+vertical_extent(support[..., 3:7], size/2)
    rel = graph.edge_relation
    source = objects[row, (graph.edge_src-task.num_agents).clamp(0, task.num_objects-1)]
    source_size = sizes[row, (graph.edge_src-task.num_agents).clamp(0, task.num_objects-1)]
    point = torch.where((rel == 8)[..., None], source[..., :3], bodies[:, :, 0][row, owner])
    height = torch.where(rel == 8, vertical_extent(source[..., 3:7], source_size/2),
                        torch.where(rel == 9, task._relation_cfg['sit']['pelvis_clearance'], task._char_h))
    target = diag['target'].clone()
    target[..., 2] = top+height
    delta = point-target
    diag['target'] = torch.where(shared[..., None], target, diag['target'])
    phi = torch.where(shared, torch.exp(-10*delta.square().sum(-1)), phi)
    diag['z_error'] = torch.where(shared, delta[..., 2], diag['z_error'])
    diag['distance'] = torch.where(shared, delta.norm(dim=-1), diag['distance'])
    foot_z = feet[..., 2].mean(-1)[row, owner]
    diag['feet_height_error'] = torch.where(shared & (rel == 10), (foot_z-top).abs(), diag['feet_height_error'])
    if task.cfg['env']['sharedSupport']['success_region'] == 'physical':
        margin = task._relation_cfg['ontop']['success_inner_margin_fraction']
        region = inner_xy_region_error(point[..., :2], support, size, margin)
        diag['region_error'] = torch.where(shared, region, diag['region_error'])
    climb = shared & (rel == 10)
    ontop = shared & (rel == 8)
    diag['support_top_z'] = torch.where(ontop, top, diag['support_top_z'])
    diag['support_extent'] = torch.where(ontop, top-support[..., 2], diag['support_extent'])
    diag['signed_gap'] = torch.where(ontop, delta[..., 2], diag['signed_gap'])
    diag['z_surface'] = torch.where(climb, top, diag['z_surface'])
    feet_max = (feet[..., 2][row, owner]-top[..., None]).abs().amax(-1)
    diag['feet_individual_max_error'] = torch.where(climb, feet_max, diag['feet_individual_max_error'])
    return phi, diag
