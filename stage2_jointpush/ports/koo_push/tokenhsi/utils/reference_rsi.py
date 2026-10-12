"""Size-conditioned sampling of original reference states, without teacher rollouts.

The bank is a discrete geometry screen, not a guarantee of dynamic stability.
Missing preferred stages fall back to the same-size general pool. Missing skills
are reported to the caller; sizes are never resampled.
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch

REFERENCE_SKILLS = ('sit', 'climb', 'pickUp', 'carryWith', 'putDown')


def collision_shapes(asset):
    """Sphere/capsule envelopes and foot mesh vertices from the actual MJCF.

    Capsule samples are inflated by half their spacing. Mesh vertices are a
    penetration screen; triangle-only intersections still require a physics probe.
    """
    tree = ET.parse(asset)
    meshes = {m.get('name'): Path(asset).parent/m.get('file') for m in tree.iter('mesh')}
    result = []
    for body in tree.iter('body'):
        for g in body.findall('geom'):
            kind = g.get('type', 'capsule')
            pos = np.fromstring(g.get('pos', '0 0 0'), sep=' ')
            if kind == 'mesh':
                with meshes[g.get('mesh')].open() as f:
                    points = np.array([list(map(float,line.split()[1:4])) for line in f if line.startswith('v ')])
                points = points+pos
                result.append((body.get('name'), 0., points))
                continue
            if kind not in ('sphere', 'capsule'):
                raise ValueError('Unsupported collision geometry: '+kind)
            radius = float(g.get('size').split()[0])
            if g.get('fromto'):
                ends = np.fromstring(g.get('fromto'), sep=' ').reshape(2, 3)
                points = ends[0]+np.linspace(0,1,17)[:,None]*(ends[1]-ends[0])
                radius += np.linalg.norm(ends[1]-ends[0])/32
            else:
                points = pos[None]
            result.append((body.get('name'), radius, points))
    return result


class SceneGeometryScreen:
    def __init__(self, asset, body_names, device, penetration=.01):
        shapes = collision_shapes(asset)
        self.ids = torch.tensor([body_names.index(name) for name,_,points in shapes
                                 for _ in points], device=device)
        self.points = torch.tensor(np.concatenate([p for _,_,p in shapes]),
                                   device=device,dtype=torch.float32)
        self.radii = torch.tensor([radius for _,radius,points in shapes for _ in points],
                                  device=device,dtype=torch.float32)
        self.penetration = penetration

    @staticmethod
    def rotate(q,v):
        cross = 2*torch.cross(q[...,:3].expand_as(v),v,dim=-1)
        return v+q[...,3:4]*cross+torch.cross(q[...,:3].expand_as(v),cross,dim=-1)

    def invalid(self, bodies, boxes, sizes):
        q = bodies[:,:,self.ids,3:7]
        points = bodies[:,:,self.ids,:3]+self.rotate(q,self.points.expand_as(q[...,:3]))
        bad = torch.zeros(len(boxes),device=boxes.device,dtype=torch.bool)
        for obj in range(boxes.shape[1]):
            inverse = boxes[:,obj,3:7].clone(); inverse[:,:3] *= -1
            local = self.rotate(inverse[:,None,None],points-boxes[:,obj,None,None,:3])
            distance = local.abs()-sizes[:,obj,None,None]/2
            sdf = distance.clamp_min(0).norm(dim=-1)+distance.amax(-1).clamp_max(0)
            bad |= (self.radii[None,None]-sdf>self.penetration).flatten(1).any(-1)
        return bad


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_digest(asset):
    h = hashlib.sha256(Path(asset).read_bytes())
    for mesh in ET.parse(asset).iter('mesh'):
        h.update((Path(asset).parent/mesh.get('file')).read_bytes())
    return h.hexdigest()


def validate_reference_rsi(spec):
    if spec is None:
        return
    if spec.get('mode') == 'hybrid_late':
        from utils.hybrid_late_rsi import validate_hybrid_rsi
        validate_hybrid_rsi(spec)
        return
    if spec.get('mode') == 'student_candidates':
        from utils.student_rsi import validate_student_rsi
        validate_student_rsi(spec)
        return
    required = {'mode', 'bank', 'late_fraction', 'phase_range', 'biased_skills',
                'empty_skill', 'restore_box_velocity'}
    if set(spec) != required or spec['mode'] not in ('time_late', 'semantic_late'):
        raise ValueError('Invalid referenceRsi fields/mode')
    if (not isinstance(spec['bank'], str) or not spec['bank'] or
            isinstance(spec['late_fraction'], bool) or not 0 <= spec['late_fraction'] <= 1 or
            len(spec['phase_range']) != 2 or
            not 0 <= spec['phase_range'][0] < spec['phase_range'][1] <= 1 or
            not spec['biased_skills'] or not set(spec['biased_skills']) <= set(REFERENCE_SKILLS) or
            spec['empty_skill'] != 'loco' or spec['restore_box_velocity'] is not True):
        raise ValueError('Invalid referenceRsi values')


def interpolate_object_velocity(lib, ids, times):
    """World-frame derivative of the same position/slerp used by MotionLib.

    Segment derivatives are exact away from knots. At a knot use the following
    segment, and at the last frame use the previous segment.
    """
    lengths = lib._motion_lengths[ids]
    frames = lib._motion_num_frames[ids]
    dt = lib._motion_dt[ids]
    i = (times / dt).floor().long().clamp_min(0)
    i = torch.minimum(i, frames - 2)
    lo = i.to(times.dtype) * dt
    hi = torch.minimum(lo + dt, lengths)
    p0, q0 = lib.get_obj_motion_state(ids, lo)
    p1, q1 = lib.get_obj_motion_state(ids, hi)
    span = (hi - lo).clamp_min(1e-8)
    linear = (p1 - p0) / span[:, None]
    # q_delta = q1 * conjugate(q0), xyzw convention, shortest world rotation.
    a, aw = q1[:, :3], q1[:, 3:4]
    b, bw = -q0[:, :3], q0[:, 3:4]
    xyz = aw*b + bw*a + torch.cross(a, b, dim=-1)
    w = aw*bw - (a*b).sum(-1, keepdim=True)
    xyz = torch.where(w < 0, -xyz, xyz)
    w = w.abs()
    norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2*torch.atan2(norm, w)
    angular = xyz * (angle / norm.clamp_min(1e-8)) / span[:, None]
    return linear, angular


class ReferenceRSIBank:
    def __init__(self, spec, motion_libs, dataset, asset, device):
        validate_reference_rsi(spec)
        self.spec = spec
        self.device = device
        self.sha256 = digest(spec['bank'])
        self.data = {}
        self.plans = {}
        self.stats = {'attempts': 0, 'preferred_requests': 0,
                      'preferred_fallback': 0, 'empty_skill': 0}
        with np.load(spec['bank'], allow_pickle=False) as bank:
            meta = json.loads(str(bank['metadata']))
            if (meta['schema'] != 1 or meta['dataset_sha256'] != digest(dataset) or
                    meta['asset_sha256'] != digest(asset)):
                raise ValueError('RSI bank dataset/asset fingerprint mismatch')
            if meta.get('geometry_sha256',geometry_digest(asset)) != geometry_digest(asset):
                raise ValueError('RSI bank mesh content mismatch')
            self.sizes = torch.tensor(bank['sizes'], device=device)
            for skill in REFERENCE_SKILLS:
                lib = motion_libs[skill]
                expected = [str(Path(p).resolve()) for p in lib._motion_files]
                saved = [str(Path(dataset).parent.joinpath(p).resolve())
                         for p in meta['files'][skill]]
                if expected != saved:
                    raise ValueError('RSI bank motion order mismatch: ' + skill)
                for p, checksum in zip(expected, meta['motion_sha256'][skill]):
                    if digest(p) != checksum:
                        raise ValueError('RSI bank motion content mismatch: ' + skill)
                for p, checksum in zip(meta['objects'][skill], meta['object_sha256'][skill]):
                    if digest(Path(dataset).parent / p) != checksum:
                        raise ValueError('RSI bank object content mismatch: ' + skill)
                self.data[skill] = {k: bank[skill+'_'+k].copy()
                    for k in ('times', 'motion_ids', 'phase', 'allowed', 'safe', 'semantic')}
                self.data[skill]['weights'] = lib._motion_weights.detach().cpu().numpy()
        self.coverage = meta['coverage']

    def _plan(self, skill, preferred):
        key = (skill, preferred, self.spec['mode'])
        if key in self.plans:
            return self.plans[key]
        d = self.data[skill]
        mask = d['safe'].copy()
        # Legacy late-CLIMB intentionally overrides the ordinary skipped range.
        # Other skills and general sampling always preserve their original ranges.
        if not preferred or skill != 'climb':
            mask &= d['allowed'][None]
        if preferred:
            if self.spec['mode'] == 'time_late':
                lo, hi = self.spec['phase_range']
                mask &= ((d['phase'] >= lo) & (d['phase'] <= hi))[None]
            else:
                mask &= d['semantic']
        nsize, _ = mask.shape
        nclip = len(d['weights'])
        starts = np.zeros((nsize, nclip), dtype=np.int64)
        counts = starts.copy()
        chunks = []
        cursor = 0
        for size in range(nsize):
            for clip in range(nclip):
                ids = np.flatnonzero(mask[size] & (d['motion_ids'] == clip))
                starts[size, clip] = cursor
                counts[size, clip] = len(ids)
                chunks.append(ids)
                cursor += len(ids)
        flat = np.concatenate(chunks) if cursor else np.zeros(1, dtype=np.int64)
        probabilities = (counts > 0) * d['weights'][None]
        plan = tuple(torch.tensor(a, device=self.device) for a in
                     (starts, counts, flat, probabilities))
        self.plans[key] = plan
        return plan

    def sample(self, skill, sizes):
        delta = (sizes[:, None] - self.sizes[None]).abs().amax(-1)
        error, size_ids = delta.min(-1)
        if (error > 1e-5).any():
            raise ValueError('Physical size not covered by RSI bank')
        n = len(sizes)
        preferred = ((torch.rand(n, device=self.device) < self.spec['late_fraction'])
                     if skill in self.spec['biased_skills'] else
                     torch.zeros(n, dtype=torch.bool, device=self.device))
        requests = preferred.clone()
        general = self._plan(skill, False)
        late = self._plan(skill, True)
        general_ok = general[3][size_ids].sum(-1) > 0
        late_ok = late[3][size_ids].sum(-1) > 0
        preferred &= late_ok
        available = general_ok | preferred
        picked = torch.zeros(n, dtype=torch.long, device=self.device)
        for want, plan in ((False, general), (True, late)):
            dest = ((preferred == want) & available).nonzero().flatten()
            if not len(dest):
                continue
            s = size_ids[dest]
            clip = torch.multinomial(plan[3][s].float(), 1).flatten()
            count = plan[1][s, clip]
            offset = (torch.rand(len(dest), device=self.device)*count).long()
            picked[dest] = plan[2][plan[0][s, clip]+offset]
        self.stats['attempts'] += n
        self.stats['preferred_requests'] += int(requests.sum())
        self.stats['preferred_fallback'] += int((requests & ~late_ok & general_ok).sum())
        self.stats['empty_skill'] += int((~available).sum())
        for key, value in (('attempts', n), ('empty_skill', int((~available).sum()))):
            tag = key+'/'+skill
            self.stats[tag] = self.stats.get(tag, 0)+value
        d = self.data[skill]
        ids = torch.tensor(d['motion_ids'], device=self.device)[picked]
        times = torch.tensor(d['times'], device=self.device)[picked]
        return ids.long(), times.float(), ~available
