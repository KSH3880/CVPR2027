import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from torch import nn

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.bridge_spec import predecessor_relations
from utils.task_role_spec import task_graph_packet
from env.tasks.multi_agent.push_reward import face_xy_distances
from learning.multi_agent.bridge_network import BridgeBuilder
from learning.multi_agent.stage2_network_builder import transfer_stage1
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd


def check_layout(task):
    b, m, n, slot = task._bridge_cfg, task.num_agents, task.num_envs, task._bridge_slot
    rows = torch.arange(n, device=task.device)
    origin = task._env_origins
    crosser = task._crosser.long().argmax(-1)
    pusher = 1 - crosser
    scene = task._bridge_scene
    side = task._bridge_size
    torch.testing.assert_close(task._stage_height, side[:, 2] + b['top_drop'])
    torch.testing.assert_close(task._bridge_gap, side[:, 0] + 2 * b['clearance'])
    body = task._kinematic_humanoid_rigid_body_states[..., :3]
    assert (body[rows, crosser, 0, 2] - origin[:, 2] > task._stage_height + .5).all()
    floor = task._terrain_height(body[rows, pusher, 0, :2][:, None], rows)[:, 0] - origin[:, 2]
    assert (floor.abs() < 1e-5).all() and (body[rows, pusher, :, 2] - origin[:, None, 2] < 2.).all()
    bridge = task._bridge_states[:, :3] - origin
    landing = task._bridge_goal - origin
    dependent = scene == 0
    torch.testing.assert_close(bridge[~dependent], landing[~dependent], atol=1e-4, rtol=0)
    sign = bridge[:, 1].sign()
    end = torch.maximum(task._stage_half_size[:, 1], sign * (task._far_stage_states[:, 1] - origin[:, 1])
                        + task._far_stage_half_size[:, 1])
    assert (bridge[dependent, 1].abs() - side[dependent, 1] / 2 >= end[dependent] - 1e-4).all()
    torch.testing.assert_close(bridge[:, 2], side[:, 2] / 2, atol=1e-4, rtol=0)
    # Pushers start in the PUSH contact pose: both hands near the back face, no body inside the box.
    pushing = task._pushing[rows, pusher]
    payload = task._push_payload[rows, pusher]
    box, size = task._box_states[rows, payload], task._box_size[rows, payload]
    hands = body[rows, pusher][:, task._key_body_ids[:2]]
    reach = face_xy_distances(hands, box, size).amax(-1)
    assert (reach[pushing] <= .18).all()
    q = box[:, None, 3:7]
    offset = body[rows, pusher] - box[:, None, :3]
    cross = 2 * torch.cross(q[..., :3].expand_as(offset), offset, dim=-1)
    local = offset - q[..., 3:] * cross + torch.cross(q[..., :3].expand_as(offset), cross, dim=-1)
    inside = (local.abs() < size[:, None] / 2 - .02).all(-1).any(-1)
    assert not inside[pushing].any()
    packet = task_graph_packet(task.relation_runtime.graph, n).reshape(n, m, 5).long()
    assert torch.equal(packet[rows, pusher, 1] == 4, pushing)
    relations = predecessor_relations(task._before_predecessor.float(), m)
    assert torch.equal(relations[rows, crosser, pusher] == 2, dependent)
    mapping = task._logical_box_order
    assert ((mapping == slot).sum(-1) == 1).all()
    assert torch.equal((mapping == slot + 1).any(-1), scene != 1)
    return dict(scenes=torch.bincount(scene, minlength=3).tolist(), push_reach_max_m=float(reach[pushing].max()) if pushing.any() else None,
                push_distance_m=(bridge[dependent, :2] - landing[dependent, :2]).norm(dim=-1).tolist())


def rollout(task, env, net, rms, obs, steps):
    for _ in range(steps):
        mu, _ = net.eval_actor(rms(obs)); value = net.eval_critic(rms(obs))
        obs, reward, done, _ = env.step(mu)
        if not all(torch.isfinite(x).all() for x in (mu, value, obs, reward)):
            raise RuntimeError('Non-finite PUSH bridge rollout')
    return obs


def push_study(task, env, net, rms, steps):
    # Dependent scenes only. The crosser starts on stage B, so the push is measured without its falls.
    task._task_graph_preset = 'bridge_push'
    task.reset_buf[:] = 1
    obs = env.reset()
    n = task.num_envs
    rows = torch.arange(n, device=task.device)
    crosser = task._crosser.long().argmax(-1)
    far = task._far_stage_states[:, :2]
    delta = torch.zeros(n, task.num_agents, 2, device=task.device)
    delta[rows, crosser] = far + torch.tensor([.5, 0.], device=task.device) - task._humanoid_root_states[rows, crosser, :2]
    task._humanoid_root_states[..., :2] += delta
    task._kinematic_humanoid_rigid_body_states[..., :2] += delta[:, :, None]
    task._reset_env_tensors(rows)
    task._refresh_sim_tensors()
    start = task._bridge_states[:, :2].clone()
    yaw0 = 2 * torch.atan2(task._bridge_states[:, 5], task._bridge_states[:, 6])
    initial = (start - task._bridge_goal[:, :2]).norm(dim=-1)
    best = initial.clone()
    alive = torch.ones(n, dtype=torch.bool, device=task.device)
    connected = torch.zeros_like(alive)
    contact = torch.zeros(n, device=task.device)
    stuck = torch.zeros(n, device=task.device)
    jammed = torch.zeros_like(alive)
    yaw_error = torch.zeros(n, device=task.device)
    for _ in range(steps):
        obs = rollout(task, env, net, rms, obs, 1)
        alive &= ~task.reset_buf.bool() & (task.progress_buf > 0)
        distance = (task._bridge_states[:, :2] - task._bridge_goal[:, :2]).norm(dim=-1)
        best = torch.where(alive, torch.minimum(best, distance), best)
        connected |= alive & (distance < .2)
        hit = task._push_hits.any(-1) & alive & ~connected
        contact += hit.float()
        slow = task._bridge_states[:, 7:9].norm(dim=-1) < .02
        stuck = torch.where(hit & slow, stuck + 1, torch.zeros_like(stuck))
        jammed |= stuck >= 30
        yaw = 2 * torch.atan2(task._bridge_states[:, 5], task._bridge_states[:, 6])
        error = (torch.remainder(yaw - yaw0 + math.pi, 2 * math.pi) - math.pi).abs()
        yaw_error = torch.where(alive & ~connected, torch.maximum(yaw_error, error), yaw_error)
    survived = alive | connected
    return dict(envs=n, steps=steps, clearance_m=task._bridge_cfg['clearance'],
                connected_fraction=float(connected.float().mean()),
                pusher_survived_fraction=float(survived.float().mean()),
                mean_progress_fraction=float(((initial - best) / initial).clamp(0, 1).mean()),
                initial_distance_m=[float(initial.min()), float(initial.mean()), float(initial.max())],
                contact_step_fraction=float((contact / steps).mean()),
                jammed_fraction=float(jammed.float().mean()),
                max_yaw_error_deg=float(torch.rad2deg(yaw_error).max()),
                mean_yaw_error_deg=float(torch.rad2deg(yaw_error).mean()))


def main():
    torch.set_num_threads(1)
    args = get_args()
    args.test, args.eval = True, True
    args.task_graph = 'random_scenario'
    cfg, train, _ = load_cfg(args)
    if os.environ.get('BRIDGE_CLEARANCE'):
        cfg['env']['bridge']['clearance'] = float(os.environ['BRIDGE_CLEARANCE'])
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    set_seed(42)
    task, env = parse_task(args, cfg, train['params']['config'], parse_sim_params(args, cfg, train))
    builder = BridgeBuilder(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=task.num_agents,
        num_objects=task.num_objects, observation_mode='clean_scene',
        scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=7,
        scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
        relation_graph_spec=task._relation_graph_spec, device=task.device).to(task.device)
    model = nn.Module(); model.a2c_network = net
    checkpoint = torch.load(cfg['experiment']['transfer']['checkpoint'], map_location=task.device, weights_only=False)
    report = transfer_stage1(model, checkpoint)
    rms = SceneRunningMeanStd([223, 30, 1], [task.num_agents, task.num_objects, task.num_agents],
        normalized_sizes=[223, 30, 0], kinematic_size=7,
        extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True); rms.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    results = []
    with torch.no_grad():
        if os.environ.get('BRIDGE_STUDY_ONLY') != '1':
            for preset in ('bridge_push', 'bridge_present', 'decoy_push', 'random_scenario'):
                task._task_graph_preset = preset
                task.reset_buf[:] = 1
                obs = env.reset()
                layout = check_layout(task)
                phi, diag = task._evaluate_relations()
                result = task.relation_runtime.step(phi, diag['progress'], diag['z_error'], diag['feet_height_error'],
                                                    diag['region_error'])
                dependent = task.relation_runtime.graph.prereq_mask.any(-1)
                if result['prerequisite_gate'][dependent].any():
                    raise RuntimeError('Bridge gate starts open')
                demo = task.fetch_amp_obs_demo(256)
                if demo.shape[-1] != 10 * 133 or not torch.isfinite(demo).all():
                    raise RuntimeError('AMP demos must be finite with four families')
                families = task._amp_family()
                obs = rollout(task, env, net, rms, obs, int(os.environ.get('BRIDGE_STEPS', '32')))
                untouched = task._root_states.view(task.num_envs, -1, 13)[1:].clone()
                task._reset_envs(torch.tensor([0], device=task.device))
                if not torch.equal(untouched, task._root_states.view(task.num_envs, -1, 13)[1:]):
                    raise RuntimeError('Partial reset changed another environment')
                results.append(dict(preset=preset, amp_families=torch.bincount(families, minlength=4).tolist(), **layout))
        study = push_study(task, env, net, rms, int(os.environ.get('BRIDGE_PUSH_STEPS', '600')))
    if any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()):
        raise RuntimeError('Validation changed weights')
    output = Path(args.output_path); output.mkdir(parents=True, exist_ok=True)
    report.update(physics=results, push_study=study, training_executed=False, obs_size=task.get_obs_size(),
                  amp_obs_size=task.get_num_amp_obs(), physical_objects=task.num_physical_objects)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PUSH bridge verification complete:', output / 'validation.json')
    print(json.dumps(study))


if __name__ == '__main__':
    main()
