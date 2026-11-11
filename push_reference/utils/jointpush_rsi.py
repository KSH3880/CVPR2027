import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import yaml

from push_reference.utils.jointpush_rsi_geometry import geometry


class PushBank:
    def __init__(self, env, libraries):
        spec = env['jointPush']['rsi']
        payload = json.loads(Path(spec['bank_file']).read_text())
        source = payload['contract']
        asset = Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        meshes = [asset.parent/m.get('file') for m in ET.parse(asset).findall('./asset/mesh')]
        manifest = Path(env['motion_file'])
        motions = yaml.safe_load(manifest.read_text())['motions']['push']
        hashes = [digest(manifest.parent/r['file']) for r in motions]
        if (source['asset'] != digest(asset) or source['meshes'] != [digest(p) for p in meshes]
                or source['motions'] != hashes):
            raise ValueError('KOO RSI bank motion/character geometry mismatch')
        physical = env['jointPush']['box']
        reference_mass = spec.get('reference_mass', physical.get('mass'))
        if (source['box']['mass'] != reference_mass or source['box']['friction'] != physical['friction']
                or source['box']['size_profiles'] != physical['profiles']
                or payload['screen'] != 'PD_0.1s_all_profiles'):
            raise ValueError('KOO RSI bank physics mismatch')
        self.spec = spec
        lib = libraries['push']
        self.device = lib.gts.device
        self.geometry = geometry(asset, lib._motions[0].skeleton_tree.node_names, self.device)
        rows = payload['candidates']['push']
        self.weights = torch.zeros(len(rows), device=self.device)
        self.late = torch.zeros_like(self.weights)
        for mid in sorted({r['motion_id'] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r['motion_id'] == mid]
            if mid in spec['excluded_motion_ids']:
                raise ValueError('Excluded clip in PUSH bank')
            weight = lib._motion_weights[mid]
            self.weights[indices] = weight/len(indices)
            tail = indices[int(len(indices)*(1-spec['late_fraction'])):]
            self.late[tail] = weight/len(tail)
        self.ids = torch.tensor([r['motion_id'] for r in rows], device=self.device, dtype=torch.long)
        self.times = torch.tensor([r['time'] for r in rows], device=self.device)
        if not torch.isfinite(self.times).all() or (self.times < .3).any() or (
                self.times > lib._motion_lengths[self.ids]+1e-5).any():
            raise ValueError('Invalid PUSH RSI times')

    def sample(self, count):
        chosen = torch.multinomial(self.weights, count, replacement=True)
        late = torch.multinomial(self.late, count, replacement=True)
        requested = torch.rand(count, device=self.device) < self.spec['late_probability']
        chosen = torch.where(requested, late, chosen)
        return self.ids[chosen], self.times[chosen]
