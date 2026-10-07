"""CPU checks using the real MotionLib and AMP observation builder; no training."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tokenhsi')]
from isaacgym import gymapi  # Must precede torch.
import argparse
import hashlib
import json
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from utils.motion_lib import MotionLib
from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations
from tokenhsi.scripts.multi_agent.convert_bones_amp import DOF_BODIES, DOF_OFFSETS, joint_bounds


def amp_observations(states):
    root, rotation, dof, velocity, angular, dof_velocity, key = states
    return build_amp_observations(root,rotation,velocity,angular,dof,dof_velocity,key,True,False,72,DOF_OFFSETS)


def validate(manifest, skill):
    torch.set_num_threads(1)
    lib = MotionLib(str(manifest), skill, DOF_BODIES, DOF_OFFSETS, [5, 8, 11, 14], 'cpu')
    paths = yaml.safe_load(manifest.read_text())['motions'][skill]
    results = []
    for i, entry in enumerate(paths):
        motion = lib.get_motion(i)
        output = (manifest.parent/entry['file']).parent
        audit = json.loads((output/'conversion_report.json').read_text())
        assert hashlib.sha256((ROOT/audit['source']).read_bytes()).hexdigest() == audit['source_sha256']
        quats = motion.local_rotation.numpy()
        np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-5)
        assert motion.fps == audit['fps'] and quats.shape == (audit['frames'], 15, 4)
        assert motion.skeleton_tree.node_names == audit['target_skeleton']
        rv = Rotation.from_quat(quats.reshape(-1, 4)).as_rotvec().reshape(-1, 15, 3)
        values = np.concatenate([rv[:, b] if hi-lo==3 else rv[:, b, 1:2]
                 for b,lo,hi in zip(DOF_BODIES,DOF_OFFSETS[:-1],DOF_OFFSETS[1:])], -1)
        lo, hi = joint_bounds(motion.skeleton_tree)
        assert np.all(values >= lo-1e-5) and np.all(values <= hi+1e-5)
        np.testing.assert_allclose(quats[:, [5,8]], np.broadcast_to([0,0,0,1], (len(quats),2,4)), atol=1e-5)
        assert audit['foot_mesh_floor_min_m'] >= 0
        assert entry['weight'] == (0.0 if audit['review_flags'] else 1.0)
        # Check the serialized target mesh geometry, not only the audit value.
        for side,b in [('right',11),('left',14)]:
            mesh=ROOT/'tokenhsi/data/assets/smpl_foot_meshes'/f'{side}_foot.obj'
            vertices=np.array([[float(x) for x in line.split()[1:4]]
                              for line in mesh.read_text().splitlines() if line.startswith('v ')])
            matrices=Rotation.from_quat(motion.global_rotation[:,b].numpy()).as_matrix()
            z=np.einsum('fi,vi->fv',matrices[:,2],vertices)+motion.global_translation[:,b,2].numpy()[:,None]
            assert z.min() >= .0049
        assert np.abs(np.diff(values,axis=0)*motion.fps).max() <= 12.01
        # All frame boundaries AND off-grid samples exercise interpolation/velocity.
        times = torch.linspace(0, lib.get_motion_length(torch.tensor([i])).item(), 2*len(quats)-1)
        ids = torch.full((len(times),), i, dtype=torch.long)
        states = lib.get_motion_state(ids, times)
        assert states[2].shape == (len(times),32)
        assert all(torch.isfinite(v).all() for v in states)
        # Legacy float32 quat_to_angle_axis rounds near-identity angles (~1e-3 rad).
        np.testing.assert_allclose(states[2][::2].numpy(), values, atol=2e-3)
        obs = amp_observations(states)
        assert obs.shape == (len(times),129) and torch.isfinite(obs).all()
        # The 10-frame history used by current experiments.
        anchor = max(9/30, float(lib.get_motion_length(torch.tensor([i])).item())/2)
        history_times = (torch.tensor(anchor)-torch.arange(10)/30).clamp_min(0)
        history = amp_observations(lib.get_motion_state(torch.full((10,),i,dtype=torch.long),history_times)).reshape(1,-1)
        assert history.shape == (1,1290) and torch.isfinite(history).all()
        results.append(dict(clip=output.parent.name, frames=len(quats), amp_frame_width=129,
                            amp_history_width=1290, review_flags=audit['review_flags']))
    sampled=lib.sample_motions(4096)
    assert all(paths[i]['weight'] > 0 for i in sampled.tolist())
    return dict(manifest=str(manifest.relative_to(ROOT)), skill=skill, clips=len(results),
                checks='SkeletonMotion/limits/fixed hands/full-clip interpolation/32 DOF/AMP/history', results=results)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected',action='store_true')
    parser.add_argument('--dataset',choices=['all','door','push'],default='all')
    args=parser.parse_args()
    reports=[]
    for profile,skill in [('door','doorOpen'),('push','push')]:
        if args.dataset not in ('all',profile): continue
        suffix='_selected' if args.selected else ''
        reports.append(validate(ROOT/'tokenhsi/data'/f'dataset_bones_{profile}_amp{suffix}.yaml',skill))
    output=ROOT/'output/bones_amp_conversion_check'
    output.mkdir(parents=True,exist_ok=True)
    path=output/('validation_selected.json' if args.selected else 'validation.json')
    path.write_text(json.dumps(reports,indent=2)+'\n')
    print('Validation passed:',path,flush=True)
