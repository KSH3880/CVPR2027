import json
from pathlib import Path

import isaacgym
from isaacgym import gymapi, gymtorch
import torch

from env.tasks.multi_agent.humanoid_ma_mixed_sensor import HumanoidMAMixedSensor, SCENARIOS
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from env.tasks.multi_agent.cliff_geometry import terrain_fall
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.edge_scenario_spec import agent_goal_indices, classify_templates


def simulate(task, steps=1):
    for _ in range(steps):
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()


def close(a, b):
    torch.testing.assert_close(a, b, atol=2e-4, rtol=0)


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    set_seed(37)
    cfg['env']['motion_file'] = args.motion_file
    task = HumanoidMAMixedSensor(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                                 args.device, args.device_id, True)
    env = VecTaskPythonWrapperMA(task, args.rl_device)
    ids = torch.arange(task.num_envs, device=task.device)
    origins = task._env_origins
    report = {}
    static_handles = [task._stage_handles[0], task._bridge_stage_handles[0], task._far_stage_handle] + task._room_handles[0]
    roots = task._root_states.view(task.num_envs, -1, 13)
    static_states = roots[:, static_handles].clone()
    try:
        counts = torch.zeros(3, 4, device=task.device, dtype=torch.long)
        pickup_counts = torch.zeros(4, 2, device=task.device, dtype=torch.long)
        transitions = torch.zeros(3, 3, device=task.device, dtype=torch.long)
        previous = None
        goals_checked = 0
        for _ in range(24):
            obs = env.reset()
            scene = task._scenario_ids
            if previous is not None:
                transitions += torch.bincount(previous * 3 + scene, minlength=9).view(3, 3)
            previous = scene.clone()
            assert obs.shape == (task.num_envs, 2655) and torch.isfinite(obs).all()
            e, a = task._expand_slots(ids)
            kinds = classify_templates(task.relation_runtime.graph, e, a, True)
            counts += torch.bincount(scene[e] * 4 + kinds, minlength=12).view(3, 4)
            poses = obs[:, 538:587].reshape(-1, 7, 7)
            body = task._kinematic_humanoid_rigid_body_states
            close(poses[:, :2, :3], body[..., 0, :3] - origins[:, None])
            close(poses[:, 2:5, :3], task._logical_box_values(task._box_states)[..., :3] - origins[:, None])
            close(poses[:, 5:7, :3], task._tar_pos - origins[:, None])
            # H223 root height must stay relative to the active support, including partial resets.
            human = obs[:, :446].reshape(-1, 2, 223)
            close(human[..., 0], body[..., 0, 2] - origins[:, None, 2] - task._floor_height[:, None])
            box_base = task._box_states[..., 2] - origins[:, None, 2] - task._box_size[..., 2] / 2
            expected_base = torch.where(scene == 0, 0., task._floor_height)
            close(box_base, expected_base[:, None].expand(-1, 3) + .01)
            centers, _, _ = task._sensor_boxes(ids)
            assert (centers[scene == 2, 0, 2] == -1000).all()
            assert (centers[scene != 1, 4:6, 2] == -1000).all()
            assert (centers[scene != 2, 6:, 2] == -1000).all()
            close(task._room_states[..., :3], task._room_centers)
            close(roots[:, static_handles], static_states)
            close(origins, task._scene_origins[ids, scene])
            xy = task._box_states[..., :2] - origins[:, None, :2]
            radius = task._box_size[..., :2].norm(dim=-1) / 2
            cliff = scene == 0
            assert ((xy[cliff].abs() - task._stage_half_size[cliff, None] - radius[cliff, :, None]).amax(-1) >= .8 - 2e-4).all()
            room = scene == 2
            assert (xy[room].abs() + radius[room, :, None] + .2 <= task._room_half_size[room, None] + 2e-4).all()
            bridge = scene == 1
            far = task._far_stage_states[bridge, :2] - origins[bridge, :2]
            assert ((xy[bridge] - far[:, None]).abs() + radius[bridge, :, None] <= task._far_stage_half_size[bridge, None] + 2e-4).all()
            room_slots = scene[e] == 2
            re, ra, rk = e[room_slots], a[room_slots], kinds[room_slots]
            own = task._box_states[re, task._scenario_physical_object(re, ra), :2] - origins[re, :2]
            start = body[re, ra, 0, :2] - origins[re, :2]
            cross = start[:, 0] * own[:, 0] < 0
            assert not cross[(rk == 1) | (rk == 2)].any()
            t = -start[cross, 0] / (own[cross, 0] - start[cross, 0])
            y = start[cross, 1] + t * (own[cross, 1] - start[cross, 1])
            assert (y.abs() < task._room_half_size[re[cross], 1] - task._room['partition']['passage_width']).all()
            pickup_counts[:, 0] += torch.bincount(rk, minlength=4)
            pickup_counts[:, 1] += torch.bincount(rk[cross], minlength=4)
            e, a = e[kinds == 3], a[kinds == 3]
            g = agent_goal_indices(task.relation_runtime.graph, e, a)
            b = task._scenario_physical_object(e, a)
            goal = task._tar_pos[e, g] - origins[e]
            own = task._box_states[e, b, :3] - origins[e]
            close(goal[:, 2], task._floor_height[e] + task._box_size[e, b, 2] / 2)
            room = scene[e] == 2
            assert (goal[room, 0] * own[room, 0] < 0).all()
            assert (goal[~room, :2].abs() + .8 <= task._stage_half_size[e[~room]] + 2e-4).all()
            assert ((goal - own).norm(dim=-1) > .8).all()
            goals_checked += len(e)
            simulate(task)
        frequencies = counts.float() / counts.sum()
        assert (frequencies.sum(-1) - 1 / 3).abs().max() < .06
        assert (counts.float() / counts.sum(-1, keepdim=True) - .25).abs().max() < .06
        assert (transitions > 0).all()
        if task._room['partition'].get('pickup_cross_probability', 0.) == .5:
            assert (pickup_counts[[0, 3], 1].float() / pickup_counts[[0, 3], 0] - .5).abs().max() < .1
        report['room_pickup_counts_total_and_cross'] = pickup_counts.cpu().tolist()
        report['random_resets'] = dict(resets=24, envs=task.num_envs, task_counts=counts.cpu().tolist(),
            scenario_transition_counts=transitions.cpu().tolist(), active_at_goals_checked=goals_checked,
            observation_shape=list(obs.shape), h223_floor_and_gta_coordinates=True)
        frozen = task._root_states.view(task.num_envs, -1, 13)[1::2].clone()
        scenes = task._scenario_ids[1::2].clone()
        task._reset_envs(ids[::2])
        close(task._root_states.view(task.num_envs, -1, 13)[1::2], frozen)
        assert torch.equal(task._scenario_ids[1::2], scenes)
        assert torch.isfinite(task.obs_buf).all()
        report['partial_reset_untouched_environments'] = len(scenes)

        # Actual PhysX contact after switching the same environments through every scene.
        contacts = gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim))
        wall_bodies = torch.tensor([[task.gym.get_actor_rigid_body_index(e, h, 0, gymapi.DOMAIN_SIM)
            for h in handles[1:]] for e, handles in zip(task.envs, task._room_handles)], device=task.device)
        physical = []
        for scene in (2, 0, 1, 2):
            task._scenario_probabilities.zero_(); task._scenario_probabilities[scene] = 1
            env.reset()
            before_stage = task._stage_states[:, :3].clone()
            before_room = task._room_states[..., :3].clone()
            task._humanoid_root_states[..., 2] += 10
            task._box_states[:, 1:, 2] += 10
            task._box_states[:, 0, :3] = origins
            task._box_states[:, 0, 2] += task._floor_height + task._box_size[:, 0, 2] / 2 + .01
            task._box_states[:, 0, 3:7] = torch.tensor([0., 0., 0., 1.], device=task.device)
            task._box_states[:, 0, 7:13] = 0
            if scene == 2:
                sign = torch.where(ids % 2 == 0, -1., 1.)
                task._box_states[:, 0, 0] += sign * (.2 + task._box_size[:, 0, 0] / 2 + .3)
                task._box_states[:, 0, 7] = -sign * 3
            task._reset_env_tensors(ids)
            hit = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            for _ in range(40):
                simulate(task)
                hit |= contacts[wall_bodies[:, 4]].norm(dim=-1) > .1
            floor_error = (task._box_states[:, 0, 2] - origins[:, 2] - task._box_size[:, 0, 2] / 2 - task._floor_height).abs()
            assert floor_error.max() < .025, (scene, floor_error.max().item())
            stage_drift = (task._stage_states[:, :3] - before_stage).norm(dim=-1).max().item()
            wall_drift = (task._room_states[..., :3] - before_room).norm(dim=-1).max().item()
            assert max(stage_drift, wall_drift) < .001, (scene, stage_drift, wall_drift)
            if scene == 2:
                assert hit.all()
                assert (sign * (task._box_states[:, 0, 0] - origins[:, 0]) > .19).all()
            physical.append(dict(scenario=SCENARIOS[scene], supported_boxes=task.num_envs,
                partition_contacts=int(hit.sum()), stage_drift_m=stage_drift, room_drift_m=wall_drift))
        report['physical_transitions'] = physical
        close(roots[:, static_handles], static_states)
        report['fixed_terrain_actors_per_environment'] = len(static_handles)

        task._scenario_probabilities[:] = 1 / 3
        env.reset()
        task._humanoid_root_states[..., 7:10] = torch.tensor([1., 0., 0.], device=task.device)
        task._compute_reward(None)
        close(task.extras['reward_terms'][:, 4].view(-1, 2), -.5 * task._cpa_collision_risk(task._humanoid_root_states[..., :3], .7))
        task._scenario_ids[:] = ids % 3
        # Standing feet on the external floor fail for both stage scenes, but not inside the room.
        points = origins[:, None, None].expand(-1, 2, 3, -1).clone()
        points[..., 0] += 12
        points[..., 2] += torch.tensor([1., .05, .05], device=task.device)
        ground = task._terrain_height(points[..., :2])
        forces = torch.zeros_like(points); forces[..., 1:, 2] = 10
        falls = terrain_fall(forces, points, torch.tensor([1, 2], device=task.device), ground,
            torch.full((3,), .15, device=task.device), torch.full_like(ids, 5), origins[:, 2])
        assert falls[ids % 3 != 2].all() and not falls[ids % 3 == 2].any()
        report['floor_failure_masks'] = True
        report['single_cpa_reward'] = True
        out = Path(args.output_path); out.mkdir(parents=True, exist_ok=True)
        (out / 'physics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
