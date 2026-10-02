"""Finite static layout audit of carryable rectangular shared supports.

Reuses CPU PhysX articulation poses and actual MJCF collision shapes. This is
not a policy, transport, balance or trajectory guarantee.
"""
import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps', type=Path, required=True)
parser.add_argument('--output', type=Path, default=ROOT/'output/shared_scenario_audit/rectangles')
parser.add_argument('--supports-file', type=Path, help='JSON list of XYZ supports selected by a transport measurement')
parser.add_argument('--payload-sides', type=float, nargs='+', default=[.3,.35,.4])
parser.add_argument('--payload-heights', type=float, nargs='+', default=[.3])
parser.add_argument('--clearances', type=float, nargs='+', default=[.04])
parser.add_argument('--verify-results', type=Path, help='verify saved 4cm witness layouts in separate CPU PhysX simulation')
parser.add_argument('--verify-supports-file', type=Path, help='JSON XYZ support list for verify-results; default recommended two sizes')
args = parser.parse_args()
if any(not math.isfinite(v) or v <= 0 for v in args.payload_sides+args.payload_heights):
    parser.error('payload sides/heights must be positive and finite')
if any(not math.isfinite(v) or v < 0 for v in args.clearances):
    parser.error('clearances must be nonnegative and finite')
supports = json.loads(args.supports_file.read_text()) if args.supports_file else [
    [x,y,z] for x,y in ((.6,.7),(.6,.75),(.6,.8)) for z in (.45,.5)]
if not supports or any(len(s)!=3 or any(not math.isfinite(v) or v<=0 for v in s) for s in supports):
    parser.error('supports must be a nonempty list of positive finite XYZ sizes')
args.payload_sides = sorted(set(args.payload_sides))
saved = sys.argv
sys.argv = ['measure_shared_support.py', '--deps', str(args.deps), '--payload-height', '0.30']
spec = importlib.util.spec_from_file_location('geometry', Path(__file__).with_name('measure_shared_support.py'))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
sys.argv = saved
np, fcl = g.np, g.fcl
groups = g.reference_poses()
names, root_error = g.physx_forward_kinematics(groups)
shapes = g.shapes_from_asset(names)
if args.verify_results:
    records = json.loads(args.verify_results.read_text())['results']
    verify_supports = json.loads(args.verify_supports_file.read_text()) if args.verify_supports_file else [[.55,.65,.5],[.6,.65,.5]]
    selected = []
    for row in records:
        if row['support'] not in verify_supports or row['clearance_m']!=.04 or row['witness'] is None:
            continue
        if row['payload'] is not None and row['payload'] != [.3,.3,.45]:
            continue
        selected.append(dict(row['witness'],pair=row['pair'],height=row['support'][2],
            support_xyz=row['support'],payload=.3,payload_height=.45))
    if not selected:
        raise RuntimeError('No recommended witness layouts in verify-results')
    args.output.mkdir(parents=True,exist_ok=True)
    g.args.output = args.output
    g.validate_physx(groups,dict(selected_layouts=selected))
    sys.exit(0)
directions = [np.array(x, dtype=float) for x in ((1,0),(0,1),(1,1),(1,-1))]
layouts = [(d, math.atan2(d[1],d[0])+off) for d in directions for off in (0,math.pi/2,-math.pi/2)]
rotated = {(skill,i,layout,sign): g.transformed(clip,shapes,yaw+(sign<0)*math.pi)
           for skill,clips in groups.items() for i,clip in enumerate(clips)
           for layout,(d,yaw) in enumerate(layouts) for sign in (1,-1)}
pairs = [('ON_TOP','ON_TOP'),('ON_TOP','SIT'),('ON_TOP','CLIMB'),
         ('SIT','SIT'),('SIT','CLIMB'),('CLIMB','CLIMB')]
def contact_only(left, right):
    # FCL distance=-1 includes exact touching. Zero-clearance layouts permit
    # contact, but not material interpenetration; 1e-7m is numerical tolerance.
    data = fcl.CollisionData(fcl.CollisionRequest(num_max_contacts=4096,enable_contact=True))
    left.collide(right,data,fcl.defaultCollisionCallback)
    return max([float(c.penetration_depth) for c in data.result.contacts] or [0.]) <= 1e-7

# Calibrate touching versus overlapping before measuring any zero-gap layout.
calibration = {}
for label,sep in [('touching',.3),('overlapping',.29)]:
    a=g.manager([fcl.CollisionObject(fcl.Box(.3,.3,.3),fcl.Transform(np.array([0.,0.,.15])))])
    b=g.manager([fcl.CollisionObject(fcl.Box(.3,.3,.3),fcl.Transform(np.array([sep,0.,.15])))])
    calibration[label]=contact_only(a,b)
if calibration != {'touching':True,'overlapping':False}:
    raise RuntimeError('FCL touching/penetration calibration failed: '+str(calibration))
results = []
for x,y,height in supports:
    side = np.array([x,y])
    support = fcl.CollisionObject(fcl.Box(x,y,height), fcl.Transform(np.array([0.,0.,height/2])))
    human_cache = {}
    for payload,payload_height in itertools.product(args.payload_sides,args.payload_heights):
        g.args.payload_height = payload_height
        for a,b in pairs:
            if a != 'ON_TOP' and (payload != args.payload_sides[-1] or payload_height != args.payload_heights[-1]):
                continue
            na = 1 if a=='ON_TOP' else len(groups[a])
            nb = 1 if b=='ON_TOP' else len(groups[b])
            cache = {}
            def placed(skill,index,layout,sign):
                key = skill,index,layout,sign
                if key not in cache:
                    direction = layouts[layout][0]*sign
                    if skill=='ON_TOP':
                        cache[key] = g.place_payload(payload,side,height,direction,True)
                    else:
                        if key not in human_cache:
                            human_cache[key] = g.place_human(skill,rotated[key],shapes,side,height,direction,support)
                        cache[key] = human_cache[key]
                return cache[key]
            counts = {c:0 for c in args.clearances}
            witnesses = {c:None for c in args.clearances}
            for i,j in itertools.product(range(na),range(nb)):
                best_gap = -1.
                best_layout = None
                best_pair = None
                for layout in range(len(layouts)):
                    left,right = placed(a,i,layout,1),placed(b,j,layout,-1)
                    if left is not None and right is not None:
                        gap = g.distance(left['manager'],right['manager'])
                        if gap < 0 and 0.0 in args.clearances and contact_only(left['manager'],right['manager']):
                            gap = 0.
                        if gap > best_gap:
                            best_gap,best_layout,best_pair = gap,layout,(left,right)
                        if gap+1e-8 >= max(args.clearances):
                            break
                for clearance in args.clearances:
                    if best_gap+1e-8 >= clearance:
                        counts[clearance] += 1
                        if witnesses[clearance] is None:
                            witnesses[clearance] = dict(clips=[i,j],layout=best_layout,clearance_m=best_gap,
                                roots=[best_pair[0]['roots'],best_pair[1]['roots']])
            for clearance in args.clearances:
                count = counts[clearance]
                row = dict(support=[x,y,height],payload=[payload,payload,payload_height] if a=='ON_TOP' else None,
                           pair=a+'+'+b,clearance_m=clearance,combinations=na*nb,feasible=count,
                           coverage=count/(na*nb),witness=witnesses[clearance])
                results.append(row)
                print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results_partial.json').write_text(json.dumps(results,indent=2))
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'results.json').write_text(json.dumps(dict(
    method='CPU PhysX articulation FK; MJCF sphere/capsule/convex-foot FCL; 12 layouts',
    physx_root_error=root_error,clearances_m=args.clearances,results=results,
    touching_calibration=calibration,contact_numerical_tolerance_m=1e-7,
    limitations=['Finite static layouts; failure does not prove impossibility.',
                 'No carrier, transport, approach, balance or stability guarantee.']),indent=2))
