"""Static shared-support feasibility using PhysX poses and the actual MJCF shapes.

This is a geometry measurement, not a policy rollout or a stability test. ON_TOP
means two payloads or a payload and a person share O0; carriers are excluded.
Optional dependencies are isolated from the training environment with --deps.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tokenhsi'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps', type=Path, help='isolated python-fcl/numpy directory')
parser.add_argument('--output', type=Path, default=ROOT / 'output/shared_support_measurement')
parser.add_argument('--max-side', type=float, default=1.5)
parser.add_argument('--frames', type=int, default=3)
# The training engine has contact_offset=0.02 on each shape: use 0.04 for a pair.
parser.add_argument('--clearance', type=float, default=.04)
parser.add_argument('--heights', type=float, nargs='+', default=[.3,.35,.4,.45,.5,.55,.6])
parser.add_argument('--payload-sides', type=float, nargs='+', default=[.4,.6])
parser.add_argument('--payload-height', type=float, help='fixed payload Z; default 0.30m for sides <=0.40m, else 0.50m')
parser.add_argument('--validate-only', action='store_true')
args = parser.parse_args()
if args.frames < 1 or args.max_side < .6 or not math.isfinite(args.max_side):
    parser.error('--frames must be positive and --max-side must be finite and at least 0.60m')
if not math.isfinite(args.clearance) or args.clearance < 0:
    parser.error('--clearance must be finite and nonnegative')
if .5 not in args.heights or any(not math.isfinite(h) or h <= 0 for h in args.heights):
    parser.error('--heights must be positive finite values and include 0.50m for the report')
if any(not math.isfinite(s) or s <= 0 for s in args.payload_sides):
    parser.error('--payload-sides must be positive finite values')
if args.payload_height is not None and (not math.isfinite(args.payload_height) or args.payload_height <= 0):
    parser.error('--payload-height must be positive and finite')
args.payload_sides = sorted(set(args.payload_sides))
if args.deps:
    sys.path.insert(0, str(args.deps))

# Isaac Gym must precede torch (also imported by poselib).
from isaacgym import gymapi
import numpy as np
# Poselib still uses aliases removed in NumPy 1.24. These are local to this probe.
np.float = float
np.int = int
import torch
import yaml
import fcl
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from utils.motion_lib import MotionLib

ASSET = ROOT / 'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'
CFG = ROOT / 'tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_klclimb50.yaml'
DATA = ROOT / 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml'
config = yaml.safe_load(CFG.read_text())['env']['relationReward']


def yaw_matrix(yaw):
    return Rotation.from_euler('z', yaw).as_matrix()


def reference_poses():
    """Three seated/standing frames per clip, selected without pair-size results."""
    entries = yaml.safe_load(DATA.read_text())['motions']
    groups = {'SIT': [], 'CLIMB': []}
    converter = MotionLib.__new__(MotionLib)
    converter._device = 'cpu'
    converter._num_dof = 32
    converter._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
    converter._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 20, 23, 26, 29, 32]
    for skill in groups:
        for entry in entries[skill.lower()]:
            motion = SkeletonMotion.from_file(str(DATA.parent / entry['file']))
            pos = motion.global_translation.numpy()
            foot_z = pos[:, [11, 14], 2]
            if skill == 'SIT':
                eligible = np.flatnonzero(pos[:, 0, 2] <= pos[:, 0, 2].min() + .015)
            else:
                standing = ((np.abs(foot_z[:, 0] - foot_z[:, 1]) <= .07) &
                            (pos[:, 0, 2] - foot_z.mean(1) >= .69))
                if not standing.any():
                    raise ValueError('No level standing pose: ' + entry['file'])
                max_height = foot_z.mean(1)[standing].max()
                eligible = np.flatnonzero(standing & (foot_z.mean(1) >= max_height - .05))
            if not len(eligible):
                raise ValueError('No terminal pose: ' + entry['file'])
            frames = np.unique(eligible[np.linspace(0, len(eligible)-1, args.frames).astype(int)])
            dof = converter._local_rotation_to_dof(motion.local_rotation[frames]).numpy()
            poses = []
            for k, frame in enumerate(frames):
                quat = motion.global_rotation[frame, 0].numpy().copy()
                joints = dof[k].copy()
                if skill == 'CLIMB':
                    # A final standing posture, rather than tilted feet mid-jump.
                    # Keep measured upper-body DOFs; standardize the legs and root.
                    joints[14:] = 0
                    matrix = Rotation.from_quat(quat).as_matrix()
                    quat = Rotation.from_euler('z',math.atan2(matrix[1,0],matrix[0,0])).as_quat()
                poses.append(dict(frame=int(frame), root=pos[frame, 0].copy(),
                                  quat=quat, dof=joints))
            groups[skill].append(dict(file=entry['file'], poses=poses))
    return groups


def physx_forward_kinematics(groups):
    """Materialize reference DOFs in the real asset, in a separate CPU simulator."""
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1e-4
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.num_threads = 4
    params.physx.contact_offset = .02
    params.physx.rest_offset = 0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('CPU PhysX creation failed')
    options = gymapi.AssetOptions()
    options.disable_gravity = True
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset = gym.load_asset(sim, str(ASSET.parent.parent), 'mjcf/phys_humanoid_v3.xml', options)
    names = gym.get_asset_rigid_body_names(asset)
    handles = []
    for skill_groups in groups.values():
        for group in skill_groups:
            for pose in group['poses']:
                env = gym.create_env(sim, gymapi.Vec3(-3,-3,0), gymapi.Vec3(3,3,3), 10)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(*pose['root'])
                transform.r = gymapi.Quat(*pose['quat'])
                actor = gym.create_actor(env, asset, transform, 'human', len(handles), 1)
                state = np.zeros(32, dtype=gymapi.DofState.dtype)
                state['pos'] = pose['dof']
                gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
                handles.append((env, actor, pose))
    gym.prepare_sim(sim)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    root_error = 0.
    for env, actor, pose in handles:
        state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
        p = np.stack([state['pose']['p'][axis] for axis in ('x','y','z')], -1).astype(float)
        q = np.stack([state['pose']['r'][axis] for axis in ('x','y','z','w')], -1).astype(float)
        if not np.isfinite(q).all() or (np.linalg.norm(q,axis=1)<.1).any():
            raise RuntimeError('Invalid PhysX states: '+repr(state[:2])+' quaternion norms '+repr(np.linalg.norm(q,axis=1)))
        root_error = max(root_error, float(np.linalg.norm(p[0] - pose['root'])))
        r0 = Rotation.from_quat(q[0]).as_matrix()
        heading = math.atan2(r0[1,0], r0[0,0])
        pose['heading'] = heading
        undo = yaw_matrix(-heading)
        pose['p'] = (p - p[0]) @ undo.T
        pose['r'] = np.einsum('ij,bjk->bik', undo, Rotation.from_quat(q).as_matrix())
    gym.destroy_sim(sim)
    if root_error > .001:
        raise AssertionError('Unexpected PhysX root drift: ' + str(root_error))
    return names, root_error


def shapes_from_asset(names):
    """MJCF spheres/capsules and convex hulls of the actual foot meshes."""
    xml = ET.parse(ASSET).getroot()
    meshes = {m.attrib['name']: ASSET.parent / m.attrib['file'] for m in xml.findall('asset/mesh')}
    shapes = []
    for body in xml.findall('.//worldbody//body'):
        body_id = names.index(body.attrib['name'])
        for geom in body.findall('geom'):
            kind = geom.attrib.get('type', 'capsule')
            local = np.array([float(v) for v in geom.attrib.get('pos','0 0 0').split()])
            rot = np.eye(3)
            vertices = None
            if kind == 'sphere':
                radius = float(geom.attrib['size'])
                shape = fcl.Sphere(radius)
                lo, hi = local-radius, local+radius
            elif kind == 'capsule':
                ends = np.array([float(v) for v in geom.attrib['fromto'].split()]).reshape(2,3)
                local = ends.mean(0)
                axis = ends[1] - ends[0]
                length = np.linalg.norm(axis)
                z = axis / length
                x = np.cross([0,1,0], z); x /= np.linalg.norm(x)
                rot = np.column_stack([x, np.cross(z,x), z])
                radius = float(geom.attrib['size'])
                shape = fcl.Capsule(radius, length)
                lo, hi = ends.min(0)-radius, ends.max(0)+radius
            elif kind == 'mesh':
                vertices = np.array([[float(v) for v in line.split()[1:4]]
                                     for line in meshes[geom.attrib['mesh']].read_text().splitlines()
                                     if line.startswith('v ')])
                hull = ConvexHull(vertices)
                triangles = hull.simplices.copy()
                normals = np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],
                                   vertices[triangles[:,2]]-vertices[triangles[:,0]])
                reverse = (normals*hull.equations[:,:3]).sum(1) < 0
                triangles[reverse] = triangles[reverse][:,[0,2,1]]
                faces = np.column_stack([np.full(len(triangles),3), triangles]).flatten()
                shape = fcl.Convex(vertices, len(hull.simplices), faces)
                lo, hi = vertices.min(0), vertices.max(0)
            else:
                raise ValueError('Unsupported MJCF collision geometry: ' + kind)
            shapes.append(dict(body=body_id, shape=shape, local=local, rot=rot,
                               vertices=vertices, lo=lo, hi=hi, name=geom.attrib['name']))
    return shapes


def manager(objects):
    result = fcl.DynamicAABBTreeCollisionManager()
    result.registerObjects(objects)
    result.setup()
    return result


def distance(a, b):
    data = fcl.DistanceData(fcl.DistanceRequest())
    a.distance(b, data, fcl.defaultDistanceCallback)
    return float(data.result.min_distance)


def transformed(group, shapes, yaw):
    rotate = yaw_matrix(yaw)
    result = []
    for pose in group['poses']:
        centers, rotations, bounds, foot_vertices = [], [], [], []
        for shape in shapes:
            body = shape['body']
            br = rotate @ pose['r'][body]
            center = rotate @ pose['p'][body] + br @ shape['local']
            centers.append(center)
            rotations.append(br @ shape['rot'])
            # Bounds from actual capsule endpoints/sphere radii, or foot vertices.
            if shape['vertices'] is not None:
                verts = shape['vertices'] @ br.T + rotate @ pose['p'][body]
                bounds.append((verts.min(0), verts.max(0)))
                foot_vertices.append(verts)
            else:
                # Transform the local bounding box conservatively for floor checking.
                corners = np.array(list(itertools.product(*zip(shape['lo'],shape['hi']))))
                corners = corners @ br.T + rotate @ pose['p'][body]
                bounds.append((corners.min(0), corners.max(0)))
        result.append(dict(centers=np.array(centers), rotations=rotations,
                           bounds=bounds, feet=np.concatenate(foot_vertices),
                           foot_joints=(pose['p'][[11,14]] @ rotate.T)))
    return result


def place_human(skill, transformed_poses, shapes, side, height, direction, support):
    margin = config[skill.lower()]['success_inner_margin_fraction']
    lo = np.full(2, -(0.5-margin)*side + 1e-7)
    hi = -lo
    if skill == 'CLIMB':
        feet = np.concatenate([p['feet'] for p in transformed_poses])
        # Both actual foot footprints must fit on the support, stronger than reward.
        lo = np.maximum(lo, -side/2 - feet[:,:2].min(0))
        hi = np.minimum(hi, side/2 - feet[:,:2].max(0))
    if np.any(lo > hi):
        return None
    xy = np.where(direction > 0, hi, np.where(direction < 0, lo, (lo+hi)/2))
    result = []
    offsets = []
    for pose in transformed_poses:
        if skill == 'SIT':
            # Search the real success-height interval for a floor/support-safe pose.
            heights = np.arange(.05, .19001, .005) + height
        else:
            heights = [height - .001 - pose['feet'][:,2].min()]
        valid = None
        for z in heights:
            shift = np.r_[xy,z]
            if min(b[0][2] + z for b in pose['bounds']) < -.002:
                continue
            if skill == 'CLIMB':
                if abs(z-height-.94) > config['climb']['root_height_tolerance']:
                    continue
                if abs(pose['foot_joints'][:,2].mean()+z-height) > config['climb']['feet_height_tolerance']:
                    continue
            objects = [fcl.CollisionObject(s['shape'], fcl.Transform(r, c+shift))
                       for s,r,c in zip(shapes, pose['rotations'], pose['centers'])]
            bad = False
            nearest = float('inf')
            for obj in objects:
                collision = fcl.CollisionResult()
                fcl.collide(obj, support, fcl.CollisionRequest(num_max_contacts=16, enable_contact=True), collision)
                if any(c.penetration_depth > .002 for c in collision.contacts):
                    bad = True
                    break
                nearest = min(nearest, fcl.distance(obj, support, fcl.DistanceRequest(), fcl.DistanceResult()))
            if not bad and nearest <= .015:
                valid = objects
                offsets.append(shift.tolist())
                break
        if valid is None:
            return None
        result.extend(valid)
    return dict(manager=manager(result), objects=result, roots=offsets)


def payload_z(payload):
    return args.payload_height if args.payload_height is not None else (.3 if payload <= .4 else .5)


def place_payload(payload, side, height, direction, containment):
    limit = (.5-config['ontop']['success_inner_margin_fraction'])*side-1e-7
    if containment:
        limit = np.minimum(limit, (side-payload)/2)
    if np.any(limit < 0):
        return None
    payload_height = payload_z(payload)
    center = np.r_[direction*limit, height+payload_height/2]
    obj = fcl.CollisionObject(fcl.Box(payload,payload,payload_height), fcl.Transform(center))
    return dict(manager=manager([obj]), objects=[obj], roots=[center.tolist()])


def sweep(groups, shapes):
    pairs = [('ON_TOP','ON_TOP'), ('ON_TOP','SIT'), ('ON_TOP','CLIMB'),
             ('SIT','SIT'), ('SIT','CLIMB'), ('CLIMB','CLIMB')]
    directions = [np.array(x,dtype=float) for x in [(1,0),(0,1),(1,1),(1,-1)]]
    # Face outward, or along either tangent. Opposite participant faces opposite.
    layouts = [(d, math.atan2(d[1],d[0])+offset) for d in directions
               for offset in [0, math.pi/2, -math.pi/2]]
    rotated = {}
    for skill in groups:
        for i,g in enumerate(groups[skill]):
            for layout,(d,yaw) in enumerate(layouts):
                for sign in (1,-1):
                    rotated[skill,i,layout,sign] = transformed(g, shapes, yaw+(sign<0)*math.pi)
    rows, witnesses = [], []
    for containment in (False,True):
        mode = 'payload_fully_supported' if containment else 'reward_center_only'
        for height in args.heights:
            human_cache = {}
            for payload in args.payload_sides:
                for a,b in pairs:
                    if a != 'ON_TOP' and payload != args.payload_sides[-1]:
                        continue  # Human-only results do not depend on payload size.
                    if a != 'ON_TOP' and not containment:
                        continue
                    na = 1 if a == 'ON_TOP' else len(groups[a])
                    nb = 1 if b == 'ON_TOP' else len(groups[b])
                    pending = set(itertools.product(range(na),range(nb)))
                    minima = {}
                    coverage = {}
                    for side in np.round(np.arange(.4,args.max_side+.001,.05),2):
                        support = fcl.CollisionObject(fcl.Box(side,side,height),
                                                      fcl.Transform(np.array([0.,0.,height/2])))
                        cache = {}
                        def placed(skill,index,layout,sign):
                            key = skill,index,layout,sign
                            if key not in cache:
                                direction = layouts[layout][0]*sign
                                if skill == 'ON_TOP':
                                    cache[key] = place_payload(payload,side,height,direction,containment)
                                else:
                                    shared_key = key+(float(side),)
                                    if shared_key not in human_cache:
                                        human_cache[shared_key] = place_human(skill,rotated[key],shapes,side,height,direction,support)
                                    cache[key] = human_cache[shared_key]
                            return cache[key]
                        successful = set()
                        for i,j in itertools.product(range(na),range(nb)):
                            for layout in range(len(layouts)):
                                left,right = placed(a,i,layout,1),placed(b,j,layout,-1)
                                if left is None or right is None:
                                    continue
                                gap = distance(left['manager'],right['manager'])
                                if gap + 1e-8 >= args.clearance:
                                    successful.add((i,j))
                                    if (i,j) not in minima:
                                        minima[i,j] = float(side)
                                        witnesses.append(dict(mode=mode,pair=a+'+'+b,height=height,payload=payload,
                                            clips=[i,j], side=float(side),layout=layout, clearance=gap,
                                            roots=[left['roots'],right['roots']]))
                                        pending.discard((i,j))
                                    break
                        coverage[float(side)] = len(successful)/(na*nb)
                    values = sorted(minima.values())
                    total = na*nb
                    all_min = next((s for s,c in coverage.items() if c == 1),None)
                    p95 = next((s for s,c in coverage.items() if c >= .95),None)
                    row = dict(mode=mode,pair=a+'+'+b,height=height,payload=payload if a=='ON_TOP' else None,
                               combinations=total, feasible=len(values),
                               min_side_any=min(values) if values else None,
                               min_side_95pct=p95,min_side_all=all_min,
                               coverage_at_060=coverage[.6],coverage_by_side=coverage,
                               missing_clips=[list(x) for x in sorted(pending)])
                    rows.append(row)
                    print(json.dumps(row),flush=True)
    return rows,witnesses


def validate_physx(groups, report):
    """Confirm six measured layouts and deliberately overlapping controls in PhysX."""
    selected = list(report.get('selected_layouts', []))
    for row in (() if 'selected_layouts' in report else report['results']):
        if row['mode'] != 'payload_fully_supported' or row['height'] != .5:
            continue
        threshold = row['min_side_all'] or row['min_side_95pct'] or row['min_side_any']
        if threshold is None:
            continue
        candidates = [w for w in report['witnesses'] if w['mode']==row['mode'] and
                      w['height']==.5 and w['pair']==row['pair'] and
                      w['payload']==(args.payload_sides[-1] if row['payload'] is None else row['payload'])]
        selected.append(min(candidates,key=lambda w:abs(w['side']-threshold)))
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1e-4
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0,0,0)
    params.physx.use_gpu = False
    params.use_gpu_pipeline = False
    params.physx.num_threads = 4
    params.physx.contact_offset = .02
    params.physx.rest_offset = 0
    sim = gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
    options = gymapi.AssetOptions()
    options.disable_gravity = True
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    human_asset = gym.load_asset(sim,str(ASSET.parent.parent),'mjcf/phys_humanoid_v3.xml',options)
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    scenes = []
    directions = [(1,0),(0,1),(1,1),(1,-1)]
    for witness in selected:
        skills = witness['pair'].split('+')
        pose_ids = [range(1) if s=='ON_TOP' else range(len(groups[s][witness['clips'][k]]['poses']))
                    for k,s in enumerate(skills)]
        for indices in itertools.product(*pose_ids):
            for overlap in (False,True):
                env_id = len(scenes)
                env = gym.create_env(sim,gymapi.Vec3(-3,-3,0),gymapi.Vec3(3,3,4),10)
                support_size = witness.get('support_xyz')
                if support_size is None:
                    support_size = [witness['side'],witness['side'],witness['height']]
                support_asset = gym.create_box(sim,*support_size,fixed)
                support_pose = gymapi.Transform()
                support_pose.p.z = witness['height']/2
                gym.create_actor(env,support_asset,support_pose,'support',env_id,0)
                occupants = []
                for k,skill in enumerate(skills):
                    transform = gymapi.Transform()
                    shift = np.array(witness['roots'][k][indices[k]])
                    if overlap:
                        shift[:2] = 0
                    transform.p = gymapi.Vec3(*shift)
                    if skill == 'ON_TOP':
                        payload_height = witness.get('payload_height',payload_z(witness['payload']))
                        asset = gym.create_box(sim,witness['payload'],witness['payload'],payload_height,options)
                    else:
                        pose = groups[skill][witness['clips'][k]]['poses'][indices[k]]
                        direction = directions[witness['layout']//3]
                        yaw = math.atan2(direction[1],direction[0])+[0,math.pi/2,-math.pi/2][witness['layout']%3]+k*math.pi
                        quat = (Rotation.from_euler('z',yaw-pose['heading'])*Rotation.from_quat(pose['quat'])).as_quat()
                        transform.r = gymapi.Quat(*quat)
                        asset = human_asset
                    # Match Rescue YAML: enableSelfCollisionDetection=true -> mask 0.
                    actor = gym.create_actor(env,asset,transform,'occupant_'+str(k),env_id,0)
                    if skill != 'ON_TOP':
                        state = np.zeros(32,dtype=gymapi.DofState.dtype)
                        state['pos'] = pose['dof']
                        gym.set_actor_dof_states(env,actor,state,gymapi.STATE_ALL)
                    occupants.append(set(gym.get_actor_rigid_body_index(env,actor,i,gymapi.DOMAIN_ENV)
                                         for i in range(gym.get_actor_rigid_body_count(env,actor))))
                scenes.append((env,witness,indices,overlap,occupants))
    gym.prepare_sim(sim)
    gym.simulate(sim)
    gym.fetch_results(sim,True)
    checks = []
    for env,w,indices,overlap,occupants in scenes:
        contacts = gym.get_env_rigid_contacts(env)
        overlap_field = next((k for k in contacts.dtype.names if k.replace('_','').lower()=='initialoverlap'),None)
        if overlap_field is None:
            raise ValueError('Contact dtype fields: '+repr(contacts.dtype.names))
        cross = [c for c in contacts if (int(c['body0']) in occupants[0] and int(c['body1']) in occupants[1]) or
                 (int(c['body0']) in occupants[1] and int(c['body1']) in occupants[0])]
        depth = max([float(c[overlap_field]) for c in cross]+[0.])
        checks.append(dict(pair=w['pair'],side=w.get('side'),support_xyz=w.get('support_xyz'),height=w['height'],pose_indices=list(indices),
                           deliberately_overlapping=overlap,cross_contact_count=len(cross),contact_initial_overlap_max=depth,
                           body_pairs=[[int(c['body0']),int(c['body1'])] for c in cross]))
    gym.destroy_sim(sim)
    result = dict(checks=checks, measured_layouts=len([c for c in checks if not c['deliberately_overlapping']]),
                  measured_layouts_with_cross_contacts=sum(c['cross_contact_count']>0 for c in checks if not c['deliberately_overlapping']),
                  controls_detected=sum(c['cross_contact_count']>0 for c in checks if c['deliberately_overlapping']))
    (args.output/'physx_contacts.json').write_text(json.dumps(result,indent=2))
    print('PhysX contact confirmation:',{k:v for k,v in result.items() if k!='checks'},flush=True)
    if result['measured_layouts_with_cross_contacts']:
        raise AssertionError('PhysX disagrees with the static geometry; inspect physx_contacts.json')
    if result['controls_detected'] != result['measured_layouts']:
        raise AssertionError('Overlapping controls did not generate cross-actor collisions')
    return result


def write_readable_report(report, confirmation):
    rows = [r for r in report['results'] if r['mode']=='payload_fully_supported' and
            r['height']==.5]
    pose_count = sum(len(m['frames']) for clips in report['motions'].values() for m in clips)
    lines = ['# 공유 받침 공간 측정', '',
        '실제 humanoid MJCF를 별도 CPU PhysX에서 자세로 만든 뒤 충돌 형상 간 거리를 측정했다. '
        f"정사각 받침 X=Y 0.40~{report['max_side']:.2f}m·5cm 간격, "
        f"높이 {min(report['support_heights']):.2f}~{max(report['support_heights']):.2f}m, "
        f"SIT {len(report['motions']['SIT'])}모션·CLIMB {len(report['motions']['CLIMB'])}모션의 "
        f'총 {pose_count}개 자세와 12개 자리/방향 배치를 사용했다.', '',
        '아래 표는 **받침 높이 0.50m**, ON_TOP 상자 크기는 각 행에 표시하며, '
        f"**형상 사이 {report['clearance']*100:g}cm 여유**를 기준으로 한다. "
        '상자 바닥 전체와 CLIMB의 양쪽 발 형상을 상판 안에 둔다. '
        '비율은 해당 크기에서 자리를 선택하면 모든 선택 프레임이 조건을 만족하는 모션 쌍의 비율이며, 정책 성공률이 아니다.', '',
        '| 조합 | ON_TOP 상자 X=Y/Z | 표에 사용할 정사각 받침 X=Y | 자세 조합 통과 | 검사한 자세 조합 모두 통과한 크기 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        coverage = {float(k):v for k,v in row['coverage_by_side'].items()}
        side = row['min_side_all'] or row['min_side_95pct']
        if side is None:
            side = max(coverage,key=coverage.get)
        all_side = '찾지 못함' if row['min_side_all'] is None else f"{row['min_side_all']:.2f}m"
        payload = '—' if row['payload'] is None else f"{row['payload']:.2f}/{payload_z(row['payload']):.2f}m"
        lines.append(f"| {row['pair']} | {payload} | {side:.2f}m | {coverage[side]*100:.1f}% | {all_side} |")
    lines += ['', 'SIT+SIT은 검사한 크기에서 95% 또는 전체 통과 조건을 찾지 못했다. '
              '고정 참조 자세·제한된 자리 배치에서 실패한 것이므로, 더 큰 상자나 다른 자세가 불가능하다는 뜻은 아니다. '
              '높이 0.30~0.40m에서 SIT 참조 자세는 바닥/받침 간섭으로 더 많이 탈락했다.', '',
              '**크기만 늘려서는 무충돌이 보장되지 않는다.** 성공 전 dense reward는 중심을 선호하지만, '
              '현재 plane 성공 영역 안에서는 state/progress 보상이 최대값으로 포화된다. '
              '강제 자리 배정 없이도 분리된 위치에서 최대 보상을 받을 수 있으며 접근 경로 검증이 필요하다. '
              '직사각 받침의 최소 X/Y 조합은 이번 정사각 격자 측정 범위에 포함하지 않았다.', '',
              'CLIMB은 기록된 상체 관절을 유지하고 root를 세우며 다리 관절을 0으로 둔 최종 직립 자세다. '
              'SIT은 앉은 참조 자세를 유지하며 현재 성공 높이 범위에서 바닥·상판 간섭을 피할 높이를 선택했다. '
              '최종 자세만 검사했다. pth rollout, 운반자/HOLDING 성공, 접근·운반·앉기·오르기 경로, '
              '중력하 균형·마찰·지속 안정성은 검사하지 않았다.', '',
              f"PhysX 교차 확인: 대표 최종 배치 {confirmation['measured_layouts']}건에서 "
              f"다른 점유자와의 접촉 {confirmation['measured_layouts_with_cross_contacts']}건. "
              f"의도적으로 겹친 대조 배치 {confirmation['controls_detected']}건은 모두 접촉을 검출했다.", '',
              '세부 조건·프레임·좌표: [results.json](results.json), 크기별 요약: [summary.csv](summary.csv), '
              '엔진 접촉 확인: [physx_contacts.json](physx_contacts.json).', '',
              '![높이 0.50m에서 크기별 최종 자세 통과 비율](coverage.png)', '']
    (args.output/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sizes = sorted(float(k) for k in rows[0]['coverage_by_side'])
    values = [[r['coverage_by_side'].get(s, r['coverage_by_side'].get(str(s))) for s in sizes] for r in rows]
    fig,ax = plt.subplots(figsize=(12,4.2))
    chart=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['pair'] + ('' if r['payload'] is None else f" ({r['payload']:.2f}m)") for r in rows])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{s:.2f}' for s in sizes],rotation=45,ha='right')
    ax.set_xlabel('Square support X = Y (m); support height 0.50 m')
    ax.set_title(f"Final pose feasibility: {report['clearance']*100:g} cm clearance, full payload/foot support\n"
                 'SIT reference poses; CLIMB standardized standing legs; not policy success rate')
    fig.colorbar(chart,ax=ax,label='Fraction of motion pairs with a feasible layout')
    fig.tight_layout()
    fig.savefig(args.output/'coverage.png',dpi=180)
    plt.close(fig)


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    groups = reference_poses()
    names,root_error = physx_forward_kinematics(groups)
    shapes = shapes_from_asset(names)
    if args.validate_only:
        report=json.loads((args.output/'results.json').read_text())
        confirmation=validate_physx(groups,report)
        write_readable_report(report,confirmation)
        return
    print('PhysX poses:',sum(len(g['poses']) for gs in groups.values() for g in gs),
          'geometries per pose:',len(shapes),'root error:',root_error,flush=True)
    rows,witnesses = sweep(groups,shapes)
    report = dict(method='CPU PhysX articulation poses; FCL sphere/capsule/convex-foot clearance',
        config=str(CFG.relative_to(ROOT)), clearance=args.clearance, support_side_step=.05,
        max_side=args.max_side, support_heights=args.heights,
        payload_sizes=[[s,s,payload_z(s)] for s in args.payload_sides],
        pose_selection='SIT: root height within 15mm of minimum; CLIMB: high and level feet',
        climb_pose_adjustment='Root made upright; hip/knee/ankle DOFs zero (standing); recorded upper-body DOFs retained.',
        pose_union='Every selected pose must clear every pose of the other clip at one layout',
        layouts=12, actual_asset=str(ASSET.relative_to(ROOT)), physx_root_error=root_error,
        constraints=dict(other_occupant_clearance=args.clearance, own_support_penetration_max=.002,
                         floor_penetration_max=.002, support_contact_distance_max=.015,
                         climb_actual_footprints_fully_supported=True),
        limitations=['Static final poses, not policy rollouts or approach/transport trajectories.',
                     'No balance, friction, load-bearing, or settlement guarantee.',
                     'ON_TOP carriers and HOLDING success excluded; placement geometry only.',
                     'Finite reference poses and layouts; failure does not prove impossibility.',
                     'Feet use convex hulls of MJCF meshes; PhysX cooked hull may differ slightly.'],
        motions={s:[dict(file=g['file'],frames=[p['frame'] for p in g['poses']]) for g in gs]
                 for s,gs in groups.items()}, results=rows,witnesses=witnesses)
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k not in ('missing_clips','coverage_by_side')])
        writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k not in ('missing_clips','coverage_by_side')} for r in rows)
    confirmation=validate_physx(groups,report)
    write_readable_report(report,confirmation)


if __name__ == '__main__':
    main()
