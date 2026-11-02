"""Read-only, control-rate departure instrumentation for formal evaluation."""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ksh_jointpush.scripts.coupled_base_run import install
install()
import numpy as np
import torch
from learning.multi_agent import formal_eval
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.coupled_push_spec import inverse_rotate, group_contact_gate


class Capture:
    def __init__(self, task):
        self.task = task
        self.samples = []

    @torch.no_grad()
    def sample(self):
        t = self.task
        hands = t._rigid_body_pos[:, :, t._key_body_ids[:2]]
        boxes = t._assigned_box_values(t._box_states)
        sizes = t._assigned_box_values(t._box_size)
        local = inverse_rotate(boxes[..., None, 3:7].expand(*hands.shape[:-1], 4),
                               hands - boxes[..., None, :3])
        dx = (local[..., 0].abs() - sizes[..., None, 0]/2).clamp_min(0)
        dz = (local[..., 2].abs() - sizes[..., None, 2]/2).clamp_min(0)
        distance = (dx.square() + dz.square() +
                    (local[..., 1] + sizes[..., None, 1]/2).square()).sqrt()
        tolerance = t.cfg['env']['stage2JointPush']['gate_face_tolerance']
        near = (distance <= tolerance) & (local[..., 1] <= 0)
        same = t._groups[:, :, None] == t._groups[:, None, :]
        ready = (near.all(-1)[:, None, :] | ~same).all(-1)
        joint = same.sum(-1) > 1
        reference = group_contact_gate(hands, boxes, sizes, torch.zeros_like(boxes[..., :3]),
                                       t._groups, tolerance=tolerance, goal_saturation=False)
        assert torch.equal(ready[joint], reference[joint])
        assert bool(torch.isfinite(boxes).all() and torch.isfinite(hands).all())
        self.samples.append((near.cpu().numpy().copy(), boxes[..., :2].cpu().numpy().copy(),
                             boxes[..., 7:9].cpu().numpy().copy()))
        if len(self.samples) == 1:
            print('DEPARTURE_CAPTURE_INITIALIZED native contact gate equivalence verified', flush=True)

    def save(self):
        t = self.task
        out = Path(t.cfg['args'].output_path)
        near, xy, velocity = map(np.stack, zip(*self.samples))
        np.savez_compressed(out/'departure.npz', near=near, xy=xy, velocity=velocity,
                            groups=t._groups.cpu().numpy(), templates=t._templates_by_owner.cpu().numpy(),
                            dt=np.asarray(t.dt))
        (out/'departure_metadata.json').write_text(json.dumps(dict(
            samples=len(self.samples), includes_initial=True, sample_rate='control', dt=t.dt,
            tolerance=t.cfg['env']['stage2JointPush']['gate_face_tolerance'],
            readiness='both hands within rear face tolerance, local y <= 0; no force/aligned/goal exception',
            native_gate_equivalence_verified=True), indent=2))


original_reset = MAPlayerContinuous.env_reset
original_step = MAPlayerContinuous.env_step
original_eval = formal_eval.run_formal_eval


def reset(self, *args, **kwargs):
    obs = original_reset(self, *args, **kwargs)
    if getattr(self, '_departure_capture_enabled', False):
        self._departure_capture = Capture(self.env.task)
        self._departure_capture.sample()
    return obs


def step(self, *args, **kwargs):
    result = original_step(self, *args, **kwargs)
    if getattr(self, '_departure_capture_enabled', False):
        self._departure_capture.sample()
    return result


def evaluate(player):
    player._departure_capture_enabled = True
    result = original_eval(player)
    player._departure_capture.save()
    return result


MAPlayerContinuous.env_reset = reset
MAPlayerContinuous.env_step = step
formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'ksh_jointpush/coupled_base/tokenhsi/run.py'), run_name='__main__')
