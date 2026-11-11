"""Select successful, diverse measured episodes; never synthesize trajectories."""
import argparse
import json
from pathlib import Path
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
QUOTAS = {'enable_sit': 2, 'enable_climb': 1, 'enable_ontop': 2,
          'coupled_2': 2, 'coupled_3': 1, 'coupled_4': 2}
NAMES = ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')
DESCRIPTIONS = {
    'SIT': 'Approach the support and sit on its upper surface.',
    'CLIMB': 'Approach the support and climb onto its upper surface.',
    'AT': 'Approach, grasp and carry the assigned object to its fixed center-position goal.',
    'ON_TOP': 'Approach, grasp and carry the assigned payload, then place it on the support.',
    'PUSH': 'Approach the assigned region of the shared box and push toward the shared center goal.'}


def vec(value):
    return np.asarray(value).astype(float).round(6).tolist()


def yaw(state):
    x, y, z, w = state[3:7]
    return float(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def completion_definition(kind):
    if kind == 'AT':
        return 'First control frame with the carried physical object center within 0.20 m (3D Euclidean) of its assigned fixed goal, while the agent is valid.'
    offset = {'SIT': '0.10 m', 'CLIMB': '0.94 m', 'ON_TOP': 'half the payload height'}[kind]
    point = 'payload center' if kind == 'ON_TOP' else 'humanoid root position'
    return (f'First control frame with {point} projected inside the full oriented XY footprint of the actual support '
            f'and vertical error <=0.20 m relative to support upper surface plus {offset}, while the agent is valid. '
            'This is the existing formal geometric task-completion proxy; it does not establish hand contact or force.')


def candidate(path, initial, data, meta, env):
    success = first(data['group_success'][1:, env])
    if success is None:
        return None
    success += 1
    # Retain native group-success semantics, with stricter intact-team selection for examples.
    if not data['alive'][:success+1, env].all():
        return None
    m = data['roots'].shape[2]
    coupled = meta['setting'].startswith('coupled_')
    events = {}
    for a in range(m):
        if coupled:
            ready = first((data['surface_distances'][:success+1, env, a] < .12).all(-1) & data['alive'][:success+1, env, a])
            if ready is None:
                return None
            events[f'A{a}_ready'] = ready
            events[f'A{a}_complete'] = success
        else:
            complete = first(data['task_success'][:success+1, env, a])
            if complete is None:
                return None
            events[f'A{a}_complete'] = complete
    goals = np.asarray(initial['goals'][env])
    anchor = goals[0, :2]
    feature = np.concatenate([(data['roots'][0, env, :, :2]-anchor).reshape(-1),
                              (goals[:, :2]-anchor).reshape(-1)])
    return dict(path=path, env=env, success=success, events=events, feature=feature)


def build_demo(item, initial, data, meta, controller_id, index, stride):
    e, end = item['env'], item['success']
    dt = meta['control_dt_s']
    packet = np.asarray(initial['packet'][e])
    m = len(packet)
    mapping = np.asarray(initial['logical_box_order'][e]).astype(int)
    assignment = np.asarray(initial['assignment'][e]).astype(int)
    physical_ids = sorted(set(mapping.tolist()) | set(assignment.tolist()))
    roots = data['roots'][0, e]
    physical = data['physical_boxes'][0, e]
    coupled = meta['setting'].startswith('coupled_')
    objid = lambda b: f'object_{int(b)}'
    agents = [dict(agent_id=f'A{a}', root_position_m=vec(roots[a, :3]), root_yaw_rad=round(yaw(roots[a]), 6),
                   root_orientation_xyzw=vec(roots[a, 3:7]), root_linear_velocity_mps=vec(roots[a, 7:10]),
                   task_id=f'task_{a}') for a in range(m)]
    objects = [dict(object_id=objid(b), position_m=vec(physical[b, :3]),
                    orientation_xyzw=vec(physical[b, 3:7]), yaw_rad=round(yaw(physical[b]), 6),
                    dimensions_m=vec(initial['physical_sizes'][e, b]),
                    linear_velocity_mps=vec(physical[b, 7:10])) for b in physical_ids]
    tasks, definitions = [], {}
    for a, row in enumerate(packet):
        _, kind, actor, payload, target = map(int, row)
        name = NAMES[kind]
        t = dict(task_id=f'task_{a}', agent_id=f'A{a}', task_label=name,
                 description=DESCRIPTIONS[name], completion_event_id=f'A{a}_complete')
        if coupled:
            t.update(object_id=objid(assignment[a]), readiness_event_id=f'A{a}_ready',
                     object_goal_position_m=vec(initial['center_goals'][e, a]),
                     assigned_region=dict(center_world_m=vec(data['logical_boxes'][0,e,a,:3]),
                        orientation_xyzw=vec(data['logical_boxes'][0,e,a,3:7]),
                        dimensions_m=vec(initial['logical_sizes'][e,a])))
            definitions[f'A{a}_ready'] = (f'First valid control frame when both A{a} hand centers have absolute signed '
                '3D distance <0.12 m to the oriented surface of its assigned physical shared box. '
                'This measures surface proximity, not assigned hand-point distance or contact force.')
            definitions[f'A{a}_complete'] = ('First frame the shared physical box center is within 0.20 m (3D) '
                'of the shared center goal and at least one original participant is valid; native coupled group success.')
        else:
            if kind >= 2:
                t['object_id'] = objid(mapping[payload-m])
            if name == 'AT':
                t['object_goal_position_m'] = vec(initial['goals'][e, target-m-len(mapping)])
                t['enabling_event_id'] = f'A{a}_complete'
            else:
                support = mapping[target-m]
                t['support_object_id'] = objid(support)
                leader = int(initial['predecessor'][e,a])
                leader_target = int(packet[leader,4])-m-len(mapping)
                t['support_intended_final_center_m'] = vec(initial['goals'][e,leader_target])
                t['dependency_sensitive_event_id'] = f'A{a}_complete'
                t['dependency_sensitive_event_note'] = ('Explicitly formal task completion, not a separate approach/readiness event. '
                    'No root-arrival or hand-contact time is substituted for completion.')
            definitions[f'A{a}_complete'] = completion_definition(name)
        tasks.append(t)
    if coupled:
        constraints = [dict(type='align_events', event_ids=[f'A{a}_ready' for a in range(m)])]
    else:
        follower = int(np.flatnonzero(np.asarray(initial['predecessor'][e]) >= 0)[0])
        leader = int(initial['predecessor'][e,follower])
        constraints = [dict(type='enabling', enabling_event_id=f'A{leader}_complete', dependent_event_id=f'A{follower}_complete')]
    sample_frames = set(range(0, end+1, stride)) | {0,end}
    for frame in item['events'].values():
        sample_frames.update(range(max(0,frame-1), min(end,frame+1)+1))
    sample_frames = sorted(sample_frames)
    columns = ['episode_time_s', 'root_x_m','root_y_m','root_z_m','root_yaw_rad',
               'root_vx_mps','root_vy_mps','root_vz_mps',
               'left_hand_x_m','left_hand_y_m','left_hand_z_m',
               'right_hand_x_m','right_hand_y_m','right_hand_z_m',
               'left_hand_assigned_object_surface_distance_m','right_hand_assigned_object_surface_distance_m']
    trajectories = {}
    for a in range(m):
        rows = []
        for f in sample_frames:
            r = data['roots'][f,e,a]
            rows.append([float(f*dt)] + vec(r[:3]) + [round(yaw(r),6)] + vec(r[7:10]) +
                        vec(data['hands'][f,e,a].reshape(-1)) + vec(data['surface_distances'][f,e,a]))
        trajectories[f'A{a}'] = rows
    object_trajectories = {objid(b): [[float(f*dt)] + vec(data['physical_boxes'][f,e,b,:7]) for f in sample_frames] for b in physical_ids}
    example_id = f'demo_{index:02d}_{meta["setting"]}'
    return dict(example_id=example_id, scene_id=example_id, controller_id=controller_id,
        coordination_type='coupled' if coupled else 'enabling',
        initial_observation=dict(agents=agents, objects=objects, obstacles=[],
            obstacle_note='Other active physical boxes are included in objects; flat ground is implicit.',
            task_object_tokens=[dict(token_id=i, physical_object_id=objid(mapping[i]),
                position_m=vec(data['logical_boxes'][0,e,i,:3]), orientation_xyzw=vec(data['logical_boxes'][0,e,i,3:7]),
                dimensions_m=vec(initial['logical_sizes'][e,i])) for i in range(len(mapping))]),
        tasks=tasks, event_definitions=definitions, coordination_constraints=constraints,
        executed_release_times_s={f'A{a}':0.0 for a in range(m)},
        trajectory_columns=columns, agent_trajectories=trajectories,
        object_trajectory_columns=['episode_time_s','x_m','y_m','z_m','qx','qy','qz','qw'],
        object_trajectories=object_trajectories,
        measured_events=[dict(agent_id=event.split('_')[0],event_id=event,reached=True,
            episode_time_s=float(frame*dt),elapsed_since_release_s=float(frame*dt)) for event,frame in item['events'].items()],
        observation_end_time_s=float(end*dt),
        outcomes=dict(group_success=True,first_group_success_time_s=float(end*dt),
            all_participants_valid_through_success=True, selected_success_only=True,
            control_dt_s=dt, recorded_horizon_s=float((meta['frames']-1)*dt),
            note='Prompt trajectories truncated at first group success; complete raw episode retained.'),
        provenance=dict(raw_directory=str(item['path'].resolve().relative_to(ROOT)),environment_index=e,
            checkpoint_sha256=controller_id.split(':',1)[1],external_freeze=False,
            policy='Stage2Distill900V3_00002000 including learned coordination; not original Stage1 SCT'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=30, help='About 1 s, plus event-neighbor frames')
    args = parser.parse_args()
    snapshot = json.loads((ROOT/'teacher_runtime/snapshot.json').read_text())
    controller_id = 'sha256:' + snapshot['checkpoint_sha256']
    pools, loaded, summary = {k:[] for k in QUOTAS}, {}, []
    for raw_root in args.raw_roots:
        for path in sorted(raw_root.glob('*')):
            if not (path/'recording.json').exists():
                continue
            meta = json.loads((path/'recording.json').read_text())
            initial = torch.load(path/'record_initial.pt', map_location='cpu', weights_only=False)
            with np.load(path/'dense_trajectories.npz') as archive:
                data = {k:archive[k] for k in archive.files}
            loaded[path] = (initial,data,meta)
            formal = json.loads((path/'result.json').read_text())
            label = 'Enabling Interaction Group Success Rate' if meta['setting'].startswith('enable_') else 'Coupled Interaction Group Success Rate'
            successes = int(data['group_success'][1:].any(0).sum())
            if successes != formal['interaction_group_scores']['region'][label]['success']:
                raise ValueError('Dense recording success disagrees with original evaluator')
            accepted = []
            for e in range(data['roots'].shape[1]):
                item = candidate(path,initial,data,meta,e)
                if item:
                    accepted.append(item)
            pools[meta['setting']].extend(accepted)
            summary.append(dict(path=str(path),setting=meta['setting'],trials=data['roots'].shape[1],
                formal_successes=successes, eligible_successes=len(accepted)))
    selected = []
    for setting,count in QUOTAS.items():
        pool = pools[setting]
        if len(pool) < count:
            raise ValueError(f'{setting}: need {count}, eligible {len(pool)}; collect more episodes. Summary: {summary}')
        # Greedy farthest-point selection over initial geometry, never fabricate/retime a trajectory.
        chosen = [pool.pop(0)]
        while len(chosen) < count:
            item = max(pool, key=lambda x:min(np.linalg.norm(x['feature']-y['feature']) for y in chosen))
            chosen.append(item)
            pool = [x for x in pool if x is not item]
        selected.extend(chosen)
    demos = [build_demo(item,*loaded[item['path']],controller_id,i,args.stride) for i,item in enumerate(selected,1)]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    env = dict(units=dict(position='meters',time='seconds',angle='radians'),
        coordinate_system=dict(frame='shared world frame within each independent recorded scene',up_axis='z',
            heading='yaw counterclockwise from +x',quaternion='xyzw'),
        controller=dict(description='Fixed Stage2Distill900V3_00002000 pretrained policy, including learned coordination.',
            scheduler_control='One initial release time per agent.', before_release='Physically frozen.',
            after_release='Autonomous policy; no external FSM or runtime gate.'),
        trajectory_format=dict(timestamps='Episode seconds including initial waiting; actual control_dt times, not rounded event labels.',
            sampling='Approximately 1-second samples plus exact detector frame and adjacent frames; full raw rollout retained.',
            missing_values='null means unavailable.'), task_glossary=DESCRIPTIONS,
        calibration_selection=dict(success_only=True,enabling=5,coupled=5,
            limitations=['Success-conditioned evidence cannot estimate unconditional failure rates.',
                'All demonstrations release agents at zero; changed release-time effects are unmeasured.',
                'Enabling dependency-sensitive event is explicitly formal task completion, not separate readiness.',
                'Coupled readiness uses actual box surface proximity, not assigned hand target points.',
                'These are Stage2 demonstrations and must not be passed as same-controller evidence for a Stage1 SCT.']))
    for name,value in [('demonstrations.json',demos),('environment_spec.json',env),('selection_report.json',summary)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    for demo in demos:
        (output/(demo['example_id']+'.json')).write_text(json.dumps(demo,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))
    print('Exported',len(demos),'measured successful demonstrations to',output)


if __name__ == '__main__':
    main()
