import json
from pathlib import Path

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from env.tasks.multi_agent.humanoid_ma_room import HumanoidMARoom
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from env.tasks.multi_agent.bridge_geometry import compute_agent_cpa_collision_penalty
from env.tasks.multi_agent.room_geometry import compute_wall_cpa_collision_penalty
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.edge_scenario_spec import agent_goal_indices, classify_templates
from utils.edge_context_spec import AT


def simulate(task, steps=1):
    for _ in range(steps):
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(29)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['seed'] = 29
    task = HumanoidMARoom(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                          args.device, args.device_id, True)
    env = VecTaskPythonWrapperMA(task, args.rl_device)
    ids = torch.arange(task.num_envs, device=task.device)
    origin = task._env_origins
    report = {}
    try:
        min_body_clearance = float('inf')
        min_object_clearance = float('inf')
        min_sensor_hits = 1024
        active_goals = 0
        template_counts = torch.zeros(4, dtype=torch.long, device=task.device)
        pickup_counts = torch.zeros(4, dtype=torch.long, device=task.device)
        shuffled_goals = 0
        for _ in range(8):
            obs = env.reset()
            assert obs.shape == (task.num_envs, 2655) and torch.isfinite(obs).all()
            body = task._kinematic_humanoid_rigid_body_states[..., :3]
            clearance = task._stage_half_size[:, None, None] - (body[..., :2] - origin[:, None, None, :2]).abs()
            min_body_clearance = min(min_body_clearance, clearance.min().item())
            assert clearance.min() > 0
            boxes = task._box_states[..., :3] - origin[:, None]
            radius = task._box_size[..., :2].norm(dim=-1) / 2
            clearance = task._stage_half_size[:, None] - boxes[..., :2].abs() - radius[..., None]
            min_object_clearance = min(min_object_clearance, clearance.min().item())
            assert clearance.min() >= task._room['object_margin'] - 1e-4
            torch.testing.assert_close(boxes[..., 2] - task._box_size[..., 2] / 2,
                torch.full_like(boxes[..., 2], .11), atol=1e-4, rtol=0)
            poses = obs[:, 538:587].reshape(task.num_envs, 7, 7)
            torch.testing.assert_close(poses[:, :2, :3], body[..., 0, :] - origin[:, None], atol=1e-4, rtol=0)
            torch.testing.assert_close(poses[:, 2:5, :3], task._logical_box_values(task._box_states)[..., :3] - origin[:, None], atol=1e-4, rtol=0)
            sensor = obs[:, 607:].reshape(-1, 2, 1024)
            hits = (sensor > .4).sum(-1)
            min_sensor_hits = min(min_sensor_hits, hits.min().item())
            assert hits.min() > 0
            if task._supports_holding_at:
                envs, agents = task._expand_slots(ids)
                kinds = classify_templates(task.relation_runtime.graph, envs, agents, True)
                template_counts += torch.bincount(kinds, minlength=4)
                own_xy = task._box_states[envs, task._scenario_physical_object(envs, agents), :2] - origin[envs, :2]
                start = body[envs, agents, 0, :2] - origin[envs, :2]
                cross = start[:, 0] * own_xy[:, 0] < 0
                assert not cross[(kinds == 1) | (kinds == 2)].any()
                t = -start[cross, 0] / (own_xy[cross, 0] - start[cross, 0])
                y = start[cross, 1] + t * (own_xy[cross, 1] - start[cross, 1])
                assert (y.abs() < task._wall_half_size[envs[cross], 4, 1]).all()
                pickup_counts += torch.bincount(kinds[cross], minlength=4)
                active = kinds == 3
                e, a = envs[active], agents[active]
                slots = agent_goal_indices(task.relation_runtime.graph, e, a)
                physical = task._scenario_physical_object(e, a)
                goals = task._tar_pos[e, slots] - origin[e]
                own = task._box_states[e, physical, :3] - origin[e]
                assert (goals[:, 0] * own[:, 0] < 0).all()
                assert (goals[:, :2].abs() + .8 <= task._stage_half_size[e] + 1e-4).all()
                torch.testing.assert_close(goals[:, 2], task._box_size[e, physical, 2] / 2 + task._cliff['height'], atol=1e-4, rtol=0)
                y_crossing = own[:, 1] - own[:, 0] / (goals[:, 0] - own[:, 0]) * (goals[:, 1] - own[:, 1])
                assert (y_crossing.abs() < task._wall_half_size[e, 4, 1]).all()
                assert (boxes[..., 0].abs() - radius > task._room['wall_thickness'] / 2).all()
                torch.testing.assert_close(poses[:, 5:7, :3], task._tar_pos - origin[:, None], atol=1e-4, rtol=0)
                phi, diag = task._evaluate_relations(ids)
                at = (task.relation_runtime.graph.edge_relation == AT) & task.relation_runtime.graph.edge_valid
                assert phi[at].max() < .01
                active_goals += len(e)
                shuffled_goals += int((slots != a).sum())
        report['reset_audit'] = {'environments': task.num_envs, 'resets': 8,
            'agents_inside': task.num_envs * 8 * 2, 'objects_inside': task.num_envs * 8 * 3,
            'minimum_body_center_wall_clearance_m': min_body_clearance,
            'minimum_object_footprint_wall_clearance_m': min_object_clearance,
            'minimum_wall_sensor_samples_per_agent': min_sensor_hits,
            'observation_shape': list(obs.shape), 'graph_entities': 7}
        if task._supports_holding_at:
            if task._room['partition'].get('pickup_cross_probability', 0.) == .5:
                assert (pickup_counts[[0, 3]].float() / template_counts[[0, 3]] - .5).abs().max() < .1
            report['partition_goals'] = {'active': active_goals, 'different_goal_slot_from_owner': shuffled_goals,
                'all_straight_paths_blocked': True, 'template_counts': template_counts.cpu().tolist(),
                'pickup_cross_counts': pickup_counts.cpu().tolist(), 'passage_width_m': task._passage_width}

        task._humanoid_root_states[..., 7:10] = torch.tensor([1., 0., 0.], device=task.device)
        roots, velocity = task._humanoid_root_states[..., :3], task._humanoid_root_states[..., 7:10]
        agent_risk = compute_agent_cpa_collision_penalty(roots, velocity, .7, task.dt, .99)
        wall_risk = compute_wall_cpa_collision_penalty(roots, velocity, task._wall_starts, task._wall_ends, .7, task.dt, task._wall_collision_ttc_discount)
        task._compute_reward(None)
        collision = task.extras['reward_terms'][:, 4].view(-1, 2)
        torch.testing.assert_close(collision, -.5 * torch.maximum(agent_risk, wall_risk))
        assert wall_risk.min() > 0 and agent_risk.max() == 0
        report['single_collision_term'] = {'wall_only_nonzero_agents': int((wall_risk > 0).sum()),
            'collision_range': [collision.min().item(), collision.max().item()]}

        env.reset()
        wall_body_ids = torch.tensor([[task.gym.get_actor_rigid_body_index(e, h, 0, gymapi.DOMAIN_SIM)
            for h in handles] for e, handles in zip(task.envs, task._wall_handles)], device=task.device)
        wall_actor_ids = torch.tensor([[task.gym.get_actor_index(e, h, gymapi.DOMAIN_SIM)
            for h in handles] for e, handles in zip(task.envs, task._wall_handles)], device=task.device)
        torch.testing.assert_close(task._root_states[wall_actor_ids, :3], task._wall_centers, atol=1e-4, rtol=0)
        contacts = gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim))
        initial_walls = task._wall_centers.clone()
        side = ids % 4
        axis = side // 2
        sign = torch.where(side % 2 == 0, 1., -1.)
        task._box_states[:, 0, :3] = origin
        task._box_states[:, 0, 2] += task._cliff['height'] + task._box_size[:, 0, 2] / 2 + .01
        task._box_states[:, 0, 3:7] = torch.tensor([0., 0., 0., 1.], device=task.device)
        task._box_states[:, 0, 7:13] = 0
        task._box_states[ids, 0, axis] += sign * (task._stage_half_size[ids, axis] - .6)
        task._box_states[ids, 0, 7 + axis] = sign * 3
        task._humanoid_root_states[..., 2] += 10
        task._box_states[:, 1:, 2] += 10
        task._reset_env_tensors(ids)
        hit = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        for _ in range(40):
            simulate(task)
            hit |= contacts[wall_body_ids[ids, side]].norm(dim=-1) > .1
        assert hit.all()
        torch.testing.assert_close(task._root_states[wall_actor_ids, :3], initial_walls, atol=1e-4, rtol=0)
        box_local = task._box_states[:, 0, :2] - origin[:, :2]
        assert (box_local.abs() < task._stage_half_size).all()
        report['physical_wall_collision'] = {'tested': task.num_envs, 'wall_contacts': int(hit.sum()),
            'each_wall_tested': torch.bincount(side, minlength=4).cpu().tolist(),
            'boxes_stopped_inside': int((box_local.abs() < task._stage_half_size).all(-1).sum())}

        if task._supports_holding_at:
            env.reset()
            sign = torch.where(ids % 2 == 0, 1., -1.)
            task._box_states[:, 0, :3] = origin
            task._box_states[:, 0, 2] += task._cliff['height'] + task._box_size[:, 0, 2] / 2 + .01
            task._box_states[:, 0, 3:7] = torch.tensor([0., 0., 0., 1.], device=task.device)
            task._box_states[:, 0, 7:13] = 0
            task._box_states[:, 0, 0] += sign * .8
            task._box_states[:, 0, 7] = -sign * 3
            task._humanoid_root_states[..., 2] += 10
            task._box_states[:, 1:, 2] += 10
            task._reset_env_tensors(ids)
            hit.zero_()
            for _ in range(30):
                simulate(task)
                hit |= contacts[wall_body_ids[:, 4]].norm(dim=-1) > .1
            assert hit.all()
            assert (sign * (task._box_states[:, 0, 0] - origin[:, 0]) > .19).all()
            report['partition_physical_collision'] = {'both_sides_tested': task.num_envs,
                'contacts': int(hit.sum()), 'no_box_crossed_partition': True}

        env.reset()
        task._humanoid_root_states[:] = task._initial_humanoid_root_states
        task._humanoid_root_states[..., :2] = origin[:, None, :2]
        task._humanoid_root_states[..., 2] += task._cliff['height']
        task._humanoid_root_states[:, 0, 0] = origin[:, 0] + task._stage_half_size[:, 0] - .25
        task._humanoid_root_states[:, 1, 0] = origin[:, 0] - 1
        task._humanoid_root_states[..., 7:13] = 0
        task._dof_pos[:] = task._initial_dof_pos
        task._dof_vel.zero_()
        task._box_states[..., 2] += 10
        task._reset_env_tensors(ids)
        actions = ((task._dof_pos - task._pd_action_offset) / task._pd_action_scale).flatten(0, 1)
        contact_alive = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        for _ in range(5):
            env.step(actions)
            wall_contact = task._contact_forces[:, 0, :, :2].norm(dim=-1).amax(-1) > 1.
            upright = task._rigid_body_pos[:, 0, 0, 2] - origin[:, 2] > .7
            contact_alive |= wall_contact & upright & ~task._terminate_buf.view(-1, 2)[:, 0].bool() & (task.progress_buf > 1)
        assert contact_alive.all()
        report['human_wall_contact_without_termination'] = int(contact_alive.sum())
        report['wall_geometry_centers'] = initial_walls[0].cpu().tolist()
        report['sensor_metadata'] = task.sensor_metadata
        out = Path(args.output_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'physics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
