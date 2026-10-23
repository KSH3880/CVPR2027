"""Build discrete reference RSI geometry masks on CPU, preserving original motions."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tokenhsi')]
import argparse
import json
import itertools
import numpy as np
import torch
import yaml
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from utils.reference_rsi import REFERENCE_SKILLS, digest, collision_shapes, geometry_digest


def rotate(q, v):
    t = 2*np.cross(q[..., :3], v)
    return v+q[..., 3:4]*t+np.cross(q[..., :3], t)


def slerp(a, b, t):
    dot = (a*b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    den = np.maximum(np.sin(angle), 1e-8)
    q = np.sin((1-t)*angle)/den*a+np.sin(t*angle)/den*b
    q = np.where(dot > .9995, (1-t)*a+t*b, q)
    return q/np.linalg.norm(q, axis=-1, keepdims=True)


def screen(bp, br, names, op, oq, sizes, skill, shapes, penetration=.01):
    """Sphere/capsule screen plus foot-origin floor check; feet meshes need probe."""
    safe = np.ones((len(sizes), len(bp)), dtype=bool)
    semantic = safe.copy()*False
    hand_ids = [names.index(k) for k in ('right_hand', 'left_hand')]
    feet_ids = [names.index(k) for k in ('right_foot', 'left_foot')]
    inv = oq.copy(); inv[:, :3] *= -1
    world_shapes = [(radius, bp[:, names.index(body), None]+
                     rotate(br[:, names.index(body), None], line[None]))
                    for body, radius, line in shapes]
    for s, size in enumerate(sizes):
        center = op.copy()
        center[:, 2] = (size[2]/2 if skill in ('sit', 'climb')
                        else np.maximum(center[:, 2], size[2]/2))
        half = size/2
        for radius, points in world_shapes:
            local = rotate(inv[:, None], points-center[:, None])
            v = np.abs(local)-half
            sdf = np.linalg.norm(np.maximum(v, 0), axis=-1)+np.minimum(v.max(-1), 0)
            safe[s] &= (radius-sdf).max(-1) <= penetration
        safe[s] &= bp[:, feet_ids, 2].min(-1) >= -.02
        feet = rotate(inv[:, None], bp[:, feet_ids]-center[:, None])
        hands = rotate(inv[:, None], bp[:, hand_ids]-center[:, None])
        root = rotate(inv, bp[:, 0]-center)
        hand_v = np.abs(hands)-half
        hand_gap = np.linalg.norm(np.maximum(hand_v, 0), axis=-1)
        hand_near = (hand_gap <= .10).all(-1)
        if skill == 'climb':
            # Progress is physical foot placement, never elapsed motion phase.
            supported = ((np.abs(feet[..., :2]) <= half[:2]+.03).all(-1) &
                         (feet[..., 2] >= half[2]-.07) &
                         (feet[..., 2] <= half[2]+.30))
            semantic[s] = supported.any(-1)
        elif skill == 'sit':
            semantic[s] = ((np.abs(root[:, :2]) <= half[:2]+.10).all(-1) &
                           (np.abs(root[:, 2]-(half[2]+.12)) <= .25))
        elif skill == 'putDown':
            semantic[s] = hand_near & (center[:, 2]-half[2] <= .15)
        else:
            semantic[s] = hand_near & (center[:, 2]-half[2] >= .05)
    return safe, semantic


def build(args):
    dataset = ROOT/args.dataset
    env = yaml.safe_load((ROOT/args.config).read_text())['env']
    box = env['box']['build']
    axes = [np.arange(*[box['scaleRange'+a][0],
             box['scaleRange'+a][1]+box['scaleSampleInterval']/2,
             box['scaleSampleInterval']])*box['baseSize'][i] for i,a in enumerate('XYZ')]
    sizes = np.array(list(itertools.product(*axes)), dtype=np.float32)
    asset = ROOT/env['asset']['assetRoot']/env['asset']['assetFileName']
    shapes = collision_shapes(asset)
    entries = yaml.safe_load(dataset.read_text())['motions']
    meta = {'schema': 1, 'dataset_sha256': digest(dataset), 'asset_sha256': digest(asset),
            'geometry_sha256': geometry_digest(asset),
            'files': {}, 'objects': {}, 'motion_sha256': {}, 'object_sha256': {},
            'coverage': {}, 'subframes': args.subframes,
            'screen': 'sphere/capsule conservative centerline plus foot mesh vertices; no dynamics; runtime all-box/all-agent screen',
            'penetration_m': args.penetration}
    arrays = {'sizes': sizes}
    for skill in REFERENCE_SKILLS:
        chunks = {k: [] for k in ('times','motion_ids','phase','allowed','safe','semantic')}
        meta['files'][skill] = []; meta['objects'][skill] = []
        meta['motion_sha256'][skill] = []; meta['object_sha256'][skill] = []
        for clip, entry in enumerate(entries[skill]):
            path = dataset.parent/entry['file']; objpath = dataset.parent/entry['obj_file']
            motion = SkeletonMotion.from_file(str(path))
            n = len(motion.tensor)
            frame = np.arange(0, n-1, 1/args.subframes)
            allowed = np.ones(len(frame), dtype=bool)
            # Exact same exclusion conversion as MotionLib.sample_time_rsi.
            skip = entry.get('rsi_skipped_range', [])
            if skip:
                lo, hi = np.array(skip)*(n-1)/n
                allowed = (frame < lo) | (frame > hi)
            if not len(frame):
                raise ValueError('No RSI allowed frames: '+str(path))
            i = frame.astype(int); j = np.minimum(i+1, n-1); t = (frame-i)[:, None]
            local = motion.local_rotation.numpy(); root = motion.root_translation.numpy()
            state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree,
                torch.from_numpy(slerp(local[i],local[j],t[:,None])).float(),
                torch.from_numpy(root[i]*(1-t)+root[j]*t).float(), is_local=True)
            obj = np.load(objpath)
            if obj.ndim == 1:
                obj = np.repeat(obj[None], n, axis=0)
            elif len(obj) == 1:
                obj = np.repeat(obj, n, axis=0)
            if len(obj) != n:
                raise ValueError('Object/human frame mismatch: '+str(objpath))
            op = obj[i,:3]*(1-t)+obj[j,:3]*t
            oq = slerp(obj[i,3:7],obj[j,3:7],t)
            safe, semantic = screen(state.global_translation.numpy(),state.global_rotation.numpy(),
                motion.skeleton_tree.node_names,op,oq,sizes,skill,shapes,args.penetration)
            chunks['times'].append((frame/motion.fps).astype(np.float32))
            chunks['phase'].append((frame/(n-1)).astype(np.float32))
            chunks['allowed'].append(allowed)
            chunks['motion_ids'].append(np.full(len(frame),clip,dtype=np.int64))
            chunks['safe'].append(safe); chunks['semantic'].append(semantic)
            meta['files'][skill].append(entry['file']); meta['objects'][skill].append(entry['obj_file'])
            meta['motion_sha256'][skill].append(digest(path)); meta['object_sha256'][skill].append(digest(objpath))
        for k, values in chunks.items():
            arrays[skill+'_'+k] = np.concatenate(values, axis=1 if k in ('safe','semantic') else 0)
        safe = arrays[skill+'_safe']; phase = arrays[skill+'_phase']
        semantic = arrays[skill+'_semantic']
        allowed = arrays[skill+'_allowed'][None]
        general = (safe & allowed).sum(-1)
        preferred_allowed = np.ones_like(allowed) if skill == 'climb' else allowed
        late = (safe & preferred_allowed & ((phase>=.65)&(phase<=.98))[None]).sum(-1)
        sem = (safe & preferred_allowed & semantic).sum(-1)
        meta['coverage'][skill] = {'general_counts': general.tolist(), 'time_late_counts': late.tolist(),
                                 'semantic_counts': sem.tolist()}
        print(skill, 'candidates',len(phase),'covered_sizes',int((general>0).sum()),
              'time_late_sizes',int((late>0).sum()),'semantic_sizes',int((sem>0).sum()),flush=True)
    arrays['metadata'] = np.array(json.dumps(meta))
    output = Path(args.output).resolve()
    if ROOT.parent not in output.parents:
        raise ValueError('Bank output must remain inside koo_cvpr')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps({'sizes': sizes.tolist(), **meta},indent=2))
    print('BANK', output, digest(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='tokenhsi/data/dataset_loco_sit_carry_climb.yaml')
    parser.add_argument('--config',default='tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_random_binding.yaml')
    parser.add_argument('--output',default='../runs/reference_rsi/reference_bank_mesh.npz')
    parser.add_argument('--subframes',type=int,default=2)
    parser.add_argument('--penetration',type=float,default=.01)
    args = parser.parse_args()
    if args.subframes < 1 or not 0 <= args.penetration <= .02:
        parser.error('Invalid sampling resolution/penetration')
    build(args)
