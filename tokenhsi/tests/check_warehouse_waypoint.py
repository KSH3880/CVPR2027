import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch

from qualitative.warehouse_waypoint import update_switch
from qualitative.warehouse_waypoint_policy import WaypointPolicy
from qualitative.warehouse_waypoint_viewer import environment, DEFAULT_CHECKPOINT
from env.tasks.multi_agent import square_sensor_scene as scene
from utils.before_spec import task_objects
from utils.sensor_stage2_spec import WAREHOUSE


@torch.no_grad()
def main():
    opts = SimpleNamespace(checkpoint=DEFAULT_CHECKPOINT, out='output_etc/check/warehouse_waypoint/wall_clearance12/geometry',
                           seed=42, envs=32, steps=900, headless=True)
    torch.set_num_threads(4)
    task, env, output = environment(opts)
    try:
        source = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
        policy = WaypointPolicy(task, source)
        for key, value in policy.net.state_dict().items():
            torch.testing.assert_close(value.cpu(), source['model']['a2c_network.'+key], rtol=0, atol=0)
        for key, value in policy.rms.state_dict().items():
            torch.testing.assert_close(value.cpu(), source['running_mean_std'][key], rtol=0, atol=0)
        assert not any(p.requires_grad for p in policy.net.parameters())
        ids = torch.arange(task.num_envs, device=task.device)
        records = []
        families = set()
        for reset in range(3):
            obs = env.reset()
            leader, follower = task.waypoint_people(ids)
            virtual, sizes = task.virtual_box(ids, leader)
            yaw = task._structure_yaw[:, WAREHOUSE]
            shift = task._structure_shift[:, WAREHOUSE]
            origin = task._env_origins
            half = task._surface_half[:, WAREHOUSE, 0, 0]

            def local(xy):
                result = scene.rotate_xy(xy-origin[:, None, :2]-shift[:, None], -yaw[:, None])
                result[..., 0] += half[:, None]
                return result

            roots = local(task._humanoid_root_states[..., :2])
            primary = task._agent_box_assignment
            payload = local(task._box_states[ids[:, None], primary, :2])
            goals = local(task._tar_pos[..., :2])
            p = roots[ids, leader]
            eps = .001
            assert ((p[:, 0] >= -3.5-eps) & (p[:, 0] <= -2.5+eps)).all()
            assert ((p[:, 1].abs() >= 2.4-eps) & (p[:, 1].abs() <= 3.2+eps)).all()
            assert (p[:, 1]*payload[ids, leader, 1] >= 0).all()
            source = payload[ids, leader]
            extent = task._box_size[ids, primary[ids, leader], :2]/2
            half_y = task._surface_half[:, WAREHOUSE, 0, 1]
            wall_gap = torch.stack((source[:, 0]-extent[:, 0],
                2*half-source[:, 0]-extent[:, 0], half_y-source[:, 1].abs()-extent[:, 1]), -1)
            assert (wall_gap >= 1.2-eps).all()
            push = task._templates == 5
            assert push.sum(-1).eq(2).all()
            assert (task._before_predecessor >= 0).sum(-1).eq(1).all()
            torch.testing.assert_close(roots[..., 0][push], torch.full_like(roots[..., 0][push], -1.85), rtol=0, atol=eps)
            torch.testing.assert_close(roots[..., 1][push].abs(), torch.full_like(roots[..., 1][push], .9), rtol=0, atol=eps)
            separation = torch.cdist(roots, roots)+torch.eye(4, device=task.device)[None]*100
            assert (separation >= .85-eps).all()
            slots = torch.cat((primary, (12+torch.arange(4, device=task.device)).expand(len(ids), -1),
                               task._bridge_slot[:, None]), -1)
            boxes = local(task._box_states[ids[:, None], slots, :2])
            dimensions = task._box_size[ids[:, None], slots, :2].clone()
            dimensions[:, -1] = dimensions[:, -1, [1, 0]]
            valid = task_objects(task._before_predecessor, task._templates)
            valid[:, :4] &= ~push
            valid = torch.cat((valid, torch.ones(len(ids), 1, device=task.device, dtype=torch.bool)), -1)
            for person in (leader, follower):
                gap = ((roots[ids, person, None]-boxes).abs()-dimensions/2).clamp_min(0).norm(dim=-1)
                assert ((gap >= .5-eps) | ~valid).all()
            first = (p-payload[ids, leader]).norm(dim=-1)+(payload[ids, leader]-goals[ids, leader]).norm(dim=-1)
            family = task._templates[ids, follower]
            families.update(family.cpu().tolist())
            target = torch.where((family < 3)[:, None], goals[ids, leader], payload[ids, follower])
            distance = (roots[ids, follower]-target).norm(dim=-1)
            assert ((distance >= 1-eps) & (distance <= torch.where(family < 3, 10., 5.)+eps)).all()
            route = distance+torch.where(family == 4, (payload[ids, follower]-goals[ids, leader]).norm(dim=-1), 0.)
            assert (route <= first+eps)[task._before_near_pairs[ids, follower]].all()
            assert not task._warehouse_before_pairs.any()
            assert task._warehouse_start_mode.eq(0).all()
            assert task._warehouse_before_cache is None
            rng = torch.cuda.get_rng_state()
            reference = policy.net.eval_actor(policy.rms(obs[:, :policy.width]))[0].clamp(-1, 1)
            active = ids % 2 == 0
            torch.cuda.set_rng_state(rng)
            guided = policy.actions(obs, leader, virtual-origin, active).view(len(ids), 4, -1)
            keep = torch.ones(len(ids), 4, device=task.device, dtype=torch.bool)
            keep[ids[active], leader[active]] = False
            torch.testing.assert_close(guided[keep], reference.view_as(guided)[keep], rtol=0, atol=0)
            changed = (guided[ids[active], leader[active]]-reference.view_as(guided)[ids[active], leader[active]]).abs()
            assert (changed.amax(-1) > 1e-5).all()
            target = ids % 4 == 0
            torch.cuda.set_rng_state(rng)
            outbound = policy.actions(obs, leader, virtual-origin, active, target=target).view_as(guided)
            torch.testing.assert_close(outbound[keep], reference.view_as(guided)[keep], rtol=0, atol=0)
            torch.testing.assert_close(outbound[~target], guided[~target], rtol=0, atol=0)
            target_change = (outbound[ids[target], leader[target]]-reference.view_as(guided)[ids[target], leader[target]]).abs()
            assert (target_change.amax(-1) > 1e-5).all()
            runtime = task.relation_runtime
            own_success = runtime.own_success.clone()
            runtime.own_success.zero_()
            assert not task.leader_holding(leader).any()
            holding_edges = (runtime.graph.edge_relation == 6) & (runtime.graph.edge_owner == leader[:, None]) & runtime.graph.edge_valid
            runtime.own_success.copy_(holding_edges)
            assert task.leader_holding(leader).all()
            runtime.own_success.copy_(own_success)
            torch.cuda.set_rng_state(rng)
            restored = policy.actions(obs, leader, virtual-origin, torch.zeros_like(active))
            torch.testing.assert_close(restored, reference, rtol=0, atol=0)
            sensor_changed = obs.clone()
            sensor_changed[:, policy.width:] = 100.
            torch.cuda.set_rng_state(rng)
            ignored = policy.actions(sensor_changed, leader, virtual-origin, torch.zeros_like(active))
            torch.testing.assert_close(ignored, reference, rtol=0, atol=0)
            switched, _, _ = update_switch(task._humanoid_root_states[ids, leader, :2], virtual[:, :2],
                                           sizes, 10., torch.zeros_like(active))
            assert switched.all()
            final = task._tar_pos.clone()
            before_boxes = task._box_states.clone()
            policy.actions(obs, leader, virtual-origin, ~switched)
            torch.testing.assert_close(task._box_states, before_boxes, rtol=0, atol=0)
            for _ in range(3):
                actions = policy.actions(obs, leader, virtual-origin, active, target=target)
                obs, reward, _, _ = env.step(actions)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            torch.testing.assert_close(task._tar_pos, final, rtol=0, atol=0)
            records.append(dict(reset=reset, min_human_distance=float(separation.min()),
                min_leader_box_wall_gap=float(wall_gap.min()),
                leader_action_change_max=float(changed.max())))
        assert families == {1, 2, 4}
        result = dict(envs=opts.envs, resets=3, original_weights_and_rms_exact=True,
            nonguided_actions_exact=True, switched_actions_exact=True, sensor_ignored=True,
            mixed_payload_target_actions_preserved=True, holding_detection_passed=True,
            physical_boxes_and_goals_preserved=True, spawn_constraints_passed=True,
            follower_families=sorted(families), records=records)
        (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
