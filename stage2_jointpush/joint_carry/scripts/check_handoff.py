"""Verify corrected paired RSI handoff without importing Gym or running physics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'joint_carry'
SKILLS=('pickUp','carryWith','putDown')
DEPENDENCIES=(
    'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',
    'tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml',
    'tokenhsi/utils/size_rsi.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payloads():
    paths=[FOLDER/'README.md',FOLDER/'HANDOFF.md']
    paths+=sorted((FOLDER/'scripts').glob('*.py'))
    for directory in ('carry_joint_corrected','carry_corrected_physics'):
        paths+=sorted(p for p in (FOLDER/directory).rglob('*')
                      if p.is_file() and p.suffix not in ('.zip','.pyc'))
    return paths


def verify_data():
    report=json.loads((FOLDER/'carry_corrected_physics/report.json').read_text())
    for path,expected in report['input_hashes'].items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Audit input changed: '+path)
    expected_counts=json.loads((FOLDER/'carry_joint_corrected/selected_frames.json').read_text())
    counts={}
    for skill in SKILLS:
        p=FOLDER/'carry_joint_corrected'/skill
        with np.load(p/'screened_rsi_snapshots.npz',allow_pickle=False) as s, \
             np.load(p/'cooperative_rsi_candidate.npz',allow_pickle=False) as ref:
            ids=s['source_frame'];n=len(ids)
            assert n==expected_counts[skill] and n>0
            assert s['root_state'].shape==(n,2,13)
            assert s['box_state'].shape==(n,1,13)
            assert s['dof_position'].shape==s['dof_velocity'].shape==(n,2,32)
            assert len(np.unique(ids))==n and ref['source_rsi_allowed'][ids].all()
            assert s['short_rsi_screen_pass'].all() and float(s['screen_seconds'])==.1
            assert not bool(s['training_loader_integrated']) and not bool(s['continuous_motion_validated'])
            np.testing.assert_allclose(s['box_size'],[.52,.8,.4],atol=1e-12)
            np.testing.assert_array_equal(s['dof_position'],ref['dof_position'][ids])
            np.testing.assert_array_equal(s['dof_velocity'],ref['dof_velocity'][ids])
            roots=np.concatenate([ref['root_position'],ref['root_rotation_xyzw'],
                                  ref['body_linear_velocity'][:,:,0],ref['body_angular_velocity'][:,:,0]],axis=-1)
            boxes=np.concatenate([ref['box_position'],ref['box_rotation_xyzw'],
                                  ref['box_linear_velocity'],ref['box_angular_velocity']],axis=-1)
            np.testing.assert_array_equal(s['root_state'],roots[ids])
            np.testing.assert_array_equal(s['box_state'],boxes[ids,None])
            np.testing.assert_allclose(np.linalg.norm(s['root_state'][...,3:7],axis=-1),1,atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(s['box_state'][...,3:7],axis=-1),1,atol=1e-6)
            assert all(np.isfinite(s[k]).all() for k in s.files if s[k].dtype.kind in 'fc')
            audit=FOLDER/'carry_corrected_physics'/skill/'cooperative'
            with np.load(audit/'screen_masks.npz') as masks:
                assert masks['conservative_rsi_screen_pass'][ids].all()
            if skill=='carryWith':
                with (audit/'frames.csv').open() as stream:rows=list(csv.DictReader(stream))
                assert all(float(rows[i]['all_carriers_hand_contact_step_fraction'])>=.5 for i in ids)
            counts[skill]=n
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true',help='Explicitly refresh checksums after intentional edits; does not rerun physics.')
    parser.add_argument('--package',action='store_true',help='Build corrected-only handoff ZIP after verification.')
    args=parser.parse_args()
    counts=verify_data()
    manifest_path=FOLDER/'handoff_manifest.json'
    if args.write_manifest:
        manifest=dict(schema=1,scope='Corrected paired RSI snapshots; short 0.1s screen only',counts=counts,
                      files={str(p.relative_to(ROOT)):digest(p) for p in payloads()},
                      external_dependencies={p:digest(ROOT/p) for p in DEPENDENCIES})
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    manifest=json.loads(manifest_path.read_text())
    assert manifest['counts']==counts
    for path,expected in {**manifest['files'],**manifest['external_dependencies']}.items():
        if digest(ROOT/path)!=expected:raise ValueError('Handoff checksum mismatch: '+path)
    if args.package:
        dest=FOLDER/'joint_carry_handoff.zip'
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in manifest['files']:archive.write(ROOT/path,path)
            archive.write(manifest_path,str(manifest_path.relative_to(ROOT)))
        with zipfile.ZipFile(dest) as archive:assert archive.testzip() is None
        print('Package:',dest.relative_to(ROOT))
    print(json.dumps(dict(status='OK',counts=counts,files_verified=len(manifest['files']),
                          dependency_hashes_verified=len(manifest['external_dependencies']),
                          physics_rerun=False,training_loader_integrated=False),ensure_ascii=False))


if __name__=='__main__':
    main()
