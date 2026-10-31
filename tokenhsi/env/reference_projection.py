"""Project a reference scene into each independent TokenHSI policy's view."""
import torch

NATIVE_TASK_PROBS = (0., 1/3, 1/3, 1/3)


def amp_to_tokenhsi(obs):
    x = obs.reshape(-1, 10, 133)
    family = x[..., -4:].argmax(-1)
    task = x.new_tensor([2, 1, 3, 4], dtype=torch.long)[family]
    label = torch.nn.functional.one_hot(task, 5).to(x.dtype)
    return torch.cat((x[..., :129], label), -1).flatten(1)


def project(task, reset_ids=None):
    from env.tasks.humanoid import compute_humanoid_observations_max
    from env.tasks.basic_interaction_skills.humanoid_carry import compute_location_observations as carry_obs
    from env.tasks.basic_interaction_skills.humanoid_sit import compute_location_observations as sit_obs
    from env.tasks.basic_interaction_skills.humanoid_climb import compute_location_observations as climb_obs
    from push_reference.utils.edge_scenario_spec import classify_templates
    from push_reference.env.tasks.multi_agent.edge_ontop_reward import ontop_geometry, vertical_extent
    n = task.num_envs
    scene = torch.arange(n, device=task.device).repeat_interleave(2)
    owner = torch.arange(2, device=task.device).repeat(n)
    templates = classify_templates(task.relation_runtime.graph, scene, owner)
    body = torch.cat((task._rigid_body_pos, task._rigid_body_rot,
                      task._rigid_body_vel, task._rigid_body_ang_vel), -1)
    if reset_ids is not None:
        body[reset_ids] = task._kinematic_humanoid_rigid_body_states[reset_ids]
    pos, rot = body[..., :3].flatten(0,1), body[..., 3:7].flatten(0,1)
    vel, ang = body[..., 7:10].flatten(0,1), body[..., 10:13].flatten(0,1)
    # Match native TokenHSI teacher observation convention, not the GTA input.
    human = compute_humanoid_observations_max(pos, rot, vel, ang, False, False)
    root = task._humanoid_root_states.flatten(0, 1)
    boxes = task._box_states[scene, owner]
    size = task._box_size[scene, owner]
    corners = task._box_bps[scene, owner]
    goals = task._tar_pos[scene, owner].clone()
    ontop = templates == 4
    if ontop.any():
        _, geometry = ontop_geometry(boxes[ontop], task._box_states[scene[ontop], owner[ontop]+2], size[ontop], task._box_size[scene[ontop], owner[ontop]+2])
        goals[ontop] = geometry['target']
    top = boxes[:, 2]+vertical_extent(boxes[:, 3:7], size/2)
    obs = human.new_zeros(len(human), 397)
    obs[:, :223] = human
    # Materialize only the active task token; inactive tokens stay zero.
    for template, task_id in ((1, 1), (2, 3), (3, 2), (4, 2), (5, 4)):
        mask = templates == template
        if not mask.any():
            continue
        if task_id in (2, 4):
            start = 281 if task_id == 2 else 350
            obs[mask, start:start+42] = carry_obs(root[mask], boxes[mask], corners[mask], goals[mask], True)
        elif task_id == 1:
            target = boxes[mask, :3].clone()
            target[:, 2] = top[mask]+task._relation_cfg['sit']['pelvis_clearance']
            facing = target.new_zeros(len(target), 3); facing[:, 0] = 1
            obs[mask, 243:281] = sit_obs(root[mask], target, boxes[mask], corners[mask], facing)
        else:
            target = boxes[mask, :3].clone(); target[:, 2] = top[mask]+task._char_h
            obs[mask, 323:350] = climb_obs(root[mask], boxes[mask], corners[mask], target)
        obs[mask, 392+task_id] = 1
    if (obs[:, -5:].sum(-1) != 1).any():
        raise ValueError('Unexpected reference task (HOLDING/TRAJ is disabled)')
    return obs
