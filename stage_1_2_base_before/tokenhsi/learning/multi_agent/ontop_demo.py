"""Select real ON_TOP/CLIMB rollouts; never alter policy, actions, or physical states."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from utils.edge_ontop_spec import batched
from utils.evaluation_protocol import spatial_success
from utils.task_role_spec import task_graph_packet
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent, inner_xy_region_error


def qualifying_wait(distance, speed, holding, gate, valid, dt, seconds=1.0, radius=2.0, positions=None, max_extent=None):
    """Return an uninterrupted pre-first-gate wait interval, inclusive endpoints."""
    first = np.flatnonzero(gate)
    if not len(first):
        return None
    stop = int(first[0])
    count = 0
    # Each sample describes one completed control step, not an initial pose.
    required = int(math.ceil(seconds / dt - 1e-6))
    for i in range(stop):
        good = valid[i] and distance[i] <= radius and holding[i]
        if max_extent is None:
            good = good and speed[i] <= .20
        count = count + 1 if good else 0
        if count >= required:
            start = i - required + 1
            if max_extent is not None:
                xy = positions[start:i+1]
                diameter = np.linalg.norm(xy[:,None]-xy[None,:], axis=-1).max()
                if diameter > max_extent:
                    continue
            return start, i, stop
    return None


def center_above_support(payload, support, support_size):
    """Demo-only broad geometry: center inside full footprint and above top."""
    return ((inner_xy_region_error(payload[..., :2],support,support_size,0.)<=0)
            & (payload[...,2]>=support[...,2]+vertical_extent(support[...,3:7],support_size/2)))


def holding_through_gate(holding, gate):
    """Only keep the continuous holding segment reaching the first gate."""
    result = np.zeros_like(holding, dtype=bool)
    first = np.flatnonzero(gate)
    if len(first):
        stop = int(first[0])
        result[:stop+1] = np.logical_and.accumulate(holding[:stop+1][::-1])[::-1]
    return result


@torch.no_grad()
def search(player):
    task = player.env.task
    assert task.headless and task._task_graph_preset in ('before_stack', 'before_climb', 'before_sit')
    climbing = task._task_graph_preset == 'before_climb'
    sitting = task._task_graph_preset == 'before_sit'
    interaction = climbing or sitting
    task_name = 'CLIMB' if climbing else ('SIT' if sitting else 'ON_TOP')
    broad_success=os.environ.get('ONTOP_DEMO_SUCCESS','strict')=='center_above'
    if broad_success and interaction: raise ValueError('center_above is ON_TOP only')
    n, m = task.num_envs, task.num_agents
    out = Path(task.cfg['args'].output_path)
    out.mkdir(parents=True, exist_ok=True)
    horizon = task.max_episode_length
    player._reward_debug = False
    required_success = math.ceil(.5 / task.dt - 1e-6)
    radius = float(os.environ.get('ONTOP_DEMO_RADIUS', '2'))
    require_carry = os.environ.get('ONTOP_DEMO_REQUIRE_CARRY', '0') == '1'
    if interaction and require_carry:
        raise ValueError('CLIMB follower does not carry a payload')
    wait_seconds=float(os.environ.get('ONTOP_DEMO_WAIT_SECONDS','1'))
    if wait_seconds<=0:raise ValueError('Wait seconds must be positive')
    stable_carry=os.environ.get('ONTOP_DEMO_STABLE_CARRY','1' if require_carry else '0')=='1'
    wait_extent=float(os.environ.get('ONTOP_DEMO_WAIT_EXTENT','0'))
    require_walk = os.environ.get('ONTOP_DEMO_REQUIRE_WALK', '0') == '1'
    if not 0 < radius <= 4:
        raise ValueError('Search radius must be in (0, 4] metres')
    results = []
    for batch_id in range(int(os.environ.get('ONTOP_DEMO_BATCHES', '20'))):
        obs = player.env_reset()
        player.get_batch_size(obs['obs'], 1)
        graph = task.relation_runtime.graph
        packet = task_graph_packet(graph, n).reshape(n, m, 5).long().clone()
        relation, owner, valid_edges = [batched(getattr(graph, key), n) for key in
                                       ('edge_relation', 'edge_owner', 'edge_valid')]
        prereq = graph.prereq_mask
        if prereq.dim() == 2:
            prereq = prereq[None].expand(n, -1, -1)
        rows, edges = (valid_edges & (relation == (10 if climbing else (9 if sitting else 8))) & prereq.any(-1)).nonzero(as_tuple=True)
        followers = owner[rows, edges]
        predecessor_edges = prereq[rows, edges].long().argmax(-1)
        leaders = owner[rows, predecessor_edges]
        holding_edges = ((owner[rows] == followers[:, None]) & (relation[rows] == 6)
                         & valid_edges[rows]).long().argmax(-1)
        payload_slots = task._logical_box_order[rows, packet[rows, followers, 4 if interaction else 3]-m]
        support_slots=task._logical_box_order[rows,packet[rows,followers,4]-m]
        goal_slots = task._scenario_goal(rows, leaders)
        goals = task._tar_pos[rows, goal_slots].clone()
        pair_count = len(rows)
        alive = torch.ones(n, device=task.device, dtype=torch.bool)
        sizes = task._logical_box_values(task._box_size).clone()
        initial = dict(packet=packet.cpu(), logical_box_order=task._logical_box_order.cpu().clone(),
                       sizes_physical=task._box_size.cpu().clone(), goals=task._tar_pos.cpu().clone(),
                       origin=task._env_origins.cpu().clone(),
                       roots=task._root_states.reshape(n, -1, 13).cpu().clone(),
                       dofs=task._dof_state.reshape(n, m, -1, 2).cpu().clone())
        traces = {k: [] for k in ('roots', 'dofs', 'bodies', 'distance', 'speed', 'holding',
                                  'native_holding', 'payload_bottom', 'hand_distance', 'payload_speed',
                                  'gate', 'valid', 'success', 'spatial', 'center_above', 'leader_spatial', 'follower_spatial')}
        # Reject predecessors already successful at reset, including a brief transient.
        initial_gate = task.relation_runtime.own_success[rows, predecessor_edges].cpu().numpy()
        for step in range(horizon):
            action = player.get_action(obs, True)
            obs, _, done, info = player.env_step(player.env, action)
            if not isinstance(obs, dict):
                obs = {'obs': obs}
            raw = task.relation_runtime.last_result['raw_own_success']
            pair_alive = alive[rows] & ~task._terminate_buf.reshape(n, m).bool()[rows].any(-1)
            roots = task._humanoid_root_states
            payload = task._box_states[rows, payload_slots]
            bottom = payload[:,2]-vertical_extent(payload[:,3:7],task._box_size[rows,payload_slots]/2)
            hands = task._rigid_body_pos[rows,followers][:,task._key_body_ids[[0,1]]].mean(1)
            hand_distance = (hands-payload[:,:3]).norm(dim=-1)
            spatial = spatial_success(packet, roots[..., :3],
                task._logical_box_values(task._box_states), sizes, task._tar_pos)[..., 1]
            values = dict(
                roots=task._root_states.reshape(n, -1, 13),
                dofs=task._dof_state.reshape(n, m, -1, 2),
                bodies=task._rigid_body_state.reshape(n, -1, 13)[:, :m*task.num_bodies].reshape(n,m,task.num_bodies,13),
                distance=(roots[rows, followers, :2]-goals[:, :2]).norm(dim=-1),
                speed=roots[rows, followers, 7:9].norm(dim=-1),
                holding=(bottom > .15) & (hand_distance < .55),
                payload_speed=payload[:,7:10].norm(dim=-1),
                native_holding=torch.zeros_like(raw[rows, holding_edges]) if interaction else raw[rows, holding_edges], payload_bottom=bottom, hand_distance=hand_distance,
                gate=raw[rows, predecessor_edges], valid=pair_alive,
                success=task.relation_runtime.own_success[rows, edges] & raw[rows, predecessor_edges],
                spatial=spatial[rows, followers] & spatial[rows, leaders],
                center_above=center_above_support(payload,task._box_states[rows,support_slots],task._box_size[rows,support_slots]),
                leader_spatial=spatial[rows,leaders], follower_spatial=spatial[rows,followers])
            for key, value in values.items():
                traces[key].append(value.cpu().numpy().copy())
            alive &= ~done.reshape(n, m).bool().any(-1)
            if (step+1) % 200 == 0:
                print(f'[demo search] batch={batch_id} step={step+1} active={int(alive.sum())}', flush=True)
            if step >= 399 and (step+1) % 100 == 0 and not require_carry and not require_walk:
                # Finish recording early once a <=2m example has a stable successful ending.
                recent = min(len(traces['success']),required_success+math.ceil(1/task.dt))
                stable = np.stack(traces['success'][-recent:]) & np.stack(traces['spatial'][-recent:]) & np.stack(traces['valid'][-recent:])
                for candidate in np.flatnonzero(stable.all(0)):
                    if initial_gate[candidate]: continue
                    early_wait = qualifying_wait(
                        np.array(traces['distance'])[:,candidate],np.array(traces['speed'])[:,candidate],
                        np.ones(step+1,dtype=bool),np.array(traces['gate'])[:,candidate],
                        np.array(traces['valid'])[:,candidate],task.dt,seconds=wait_seconds)
                    if early_wait is not None:
                        print('[demo search] stable <=2m candidate captured; selecting saved cases',flush=True)
                        break
                else:
                    continue
                break
            if not alive.any():
                break
        traces = {k: np.stack(v) for k, v in traces.items()}
        if os.environ.get('ONTOP_DEMO_DIAGNOSTICS') == '1':
            np.savez_compressed(out/f'batch_{batch_id}_diagnostics.npz',
                **{k:v for k,v in traces.items() if k not in ('roots','dofs','bodies')},
                initial_gate=initial_gate,dt=task.dt,rows=rows.cpu().numpy(),
                followers=followers.cpu().numpy(),leaders=leaders.cpu().numpy())
        summary = dict(batch=batch_id, pairs=pair_count, waiting=0, successful=0, qualified=0)
        for p in range(pair_count):
            ok = traces['success'][:,p] & traces['spatial'][:,p] & traces['valid'][:,p]
            if broad_success:
                ok=traces['center_above'][:,p] & traces['gate'][:,p] & traces['leader_spatial'][:,p] & traces['valid'][:,p]
            summary['successful'] += int(ok.any())
            if initial_gate[p]:
                continue
            carry_mask = traces['holding'][:,p] if require_carry else np.ones(len(ok),dtype=bool)
            if stable_carry:
                carry_mask=carry_mask & holding_through_gate(traces['holding'][:,p],traces['gate'][:,p])
                if not wait_extent:
                    carry_mask=carry_mask & (traces['payload_speed'][:,p]<=.25)
            if require_walk:
                e,f=int(rows[p]),int(followers[p])
                origin_xy=initial['roots'][e,f,:2].numpy()
                initial_distance=float(np.linalg.norm(origin_xy-goals[p,:2].cpu().numpy()))
                displacement=np.linalg.norm(traces['roots'][:,e,f,:2]-origin_xy,axis=-1)
                carry_mask=carry_mask & (displacement>=1.) & ((initial_distance-traces['distance'][:,p])>=.75)
            inputs = (traces['distance'][:,p],traces['speed'][:,p],carry_mask,traces['gate'][:,p],traces['valid'][:,p])
            wait_options=dict(positions=traces['roots'][:,int(rows[p]),int(followers[p]),:2], max_extent=wait_extent or None)
            wait = qualifying_wait(*inputs, task.dt, seconds=wait_seconds, **wait_options)
            if wait is None and radius > 2:
                wait = qualifying_wait(*inputs, task.dt, radius=radius, seconds=wait_seconds, **wait_options)
            if wait is None:
                continue
            summary['waiting'] += 1
            start, end, gate_step = wait
            if (broad_success or interaction) and traces['follower_spatial'][:gate_step,p].any():
                continue
            if stable_carry and not traces['holding'][start:gate_step+1,p].all():
                continue
            count = 0
            success_step = None
            for t in range(gate_step+1, len(ok)):
                count = count+1 if ok[t] else 0
                if count >= required_success:
                    success_step = t
                    break
            if success_step is None:
                continue
            summary['qualified'] += 1
            env_id, follower, leader = int(rows[p]), int(followers[p]), int(leaders[p])
            checkpoint = Path(task.cfg['args'].checkpoint)
            meta = dict(protocol='selected_real_rollout_v1', batch=batch_id, env_id=env_id,
                follower=follower, leader=leader, seed=task.cfg['args'].seed,
                dt=task.dt, wait_start_step=start+1, wait_end_step=end+1,
                first_predecessor_success_step=gate_step+1, success_step=success_step+1,
                wait_duration=(end-start+1)*task.dt,
                wait_required_seconds=wait_seconds,
                wait_extent_m=wait_extent or None,
                search_radius=radius,
                require_carry_while_waiting=require_carry,
                stable_carry_while_waiting=stable_carry,
                require_walk_before_wait=require_walk,
                walk_displacement_m=float(displacement[start]) if require_walk else None,
                approach_distance_m=float(initial_distance-traces['distance'][start,p]) if require_walk else None,
                wait_max_distance=float(traces['distance'][start:end+1,p].max()),
                wait_max_xy_speed=float(traces['speed'][start:end+1,p].max()),
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                num_agents=m, num_physical_objects=task.num_physical_objects,
                hand_body_ids=task._key_body_ids[[0,1]].cpu().tolist(),
                carrying_definition='payload bottom >0.15m and hand midpoint within 0.55m of payload center',
                success_mode='center_above' if broad_success else 'strict',
                success_definition=('payload center inside support full XY footprint and above top; native AT and spatial AT, continuous >=0.5s' if broad_success else f'native gated {task_name} and AT plus independent spatial region success, continuous >=0.5s'),
                task_name=task_name,
                replay_box_ids=list(dict.fromkeys([int(task._logical_box_order[env_id,packet[env_id,leader,3]-m]), int(payload_slots[p])])),
                preset=task._task_graph_preset, replay='recorded states only; no new physics or policy rollout')
            directory = out/f'case_b{batch_id}_e{env_id}_f{follower}'
            directory.mkdir()
            # Include one second after confirmed success, only while episode remains valid.
            last = len(ok) if broad_success else min(len(ok), success_step+1+math.ceil(1/task.dt))
            invalid = np.flatnonzero(~traces['valid'][success_step+1:last,p])
            if len(invalid): last = success_step+1+int(invalid[0])
            np.savez_compressed(directory/'trajectory.npz',
                **{k: v[:last,env_id] for k,v in traces.items() if k in ('roots','dofs','bodies')},
                **{k: v[:last,p] for k,v in traces.items() if k not in ('roots','dofs','bodies')},
                **{('initial_'+k if k in ('roots','dofs') else k): v[env_id].numpy() for k,v in initial.items()},
                pair_goal=goals[p].cpu().numpy())
            # Avoid ambiguous duplicate initial/trace keys; initial full states are a separate artifact.
            torch.save({k:v[env_id] for k,v in initial.items()}, directory/'initial.pt')
            (directory/'metadata.json').write_text(json.dumps(meta,indent=2))
            print('[demo search] FOUND '+str(directory), flush=True)
        results.append(summary)
        (out/'search_summary.json').write_text(json.dumps(results,indent=2))
        print('[demo search] '+json.dumps(summary), flush=True)
        if summary['qualified']:
            break
    torch.cuda.synchronize()
    print('[demo search] complete', flush=True)
