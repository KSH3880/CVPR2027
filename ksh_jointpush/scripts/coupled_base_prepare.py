"""Prepare the pinned stage1_2_base coupled project without touching active runs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '2979913dbfb3bd823da324e4bc0b493444782437'
SOURCE = ROOT/'ksh_jointpush/ports/stage1_2_base_2979913'
PROJECT = ROOT/'ksh_jointpush/coupled_base'
DEFAULT_DATA = Path('/home/hwanhee/koo_cvpr/stage1_2_jointpush_st/tokenhsi/data')
DEFAULT_CHECKPOINT = Path('/home/hwanhee/koo_cvpr/stage1_2_jointpush_st/pth/Stage1JointPushDensity3Cone90Finetune_00021000.pth')


def prepare(checkpoint=DEFAULT_CHECKPOINT, data=DEFAULT_DATA):
    checkpoint = Path(checkpoint).resolve(strict=True)
    data = Path(data).resolve(strict=True)
    if not PROJECT.exists():
        shutil.copytree(SOURCE, PROJECT, symlinks=True)
    def link(path, target):
        target = target.resolve(strict=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if path.resolve() != target:
                raise ValueError('Existing link differs: '+str(path))
        elif path.exists():
            raise ValueError('Refusing to replace existing path: '+str(path))
        else:
            path.symlink_to(target, target_is_directory=target.is_dir())
    for name in ('dataset_amass_loco','dataset_amass_climb','dataset_sit','dataset_carry','jointpush_motions'):
        link(PROJECT/'tokenhsi/data'/name, data/name)
    link(PROJECT/'body_models', ROOT/'stage1_combined/body_models')
    for name, source in (('Stage1PushFinetune_00021000.pth', checkpoint),
                         ('Stage1_00078000.pth', ROOT/'stage1_combined/checkpoints/Stage1V2Fixed_00078000.pth')):
        dest = PROJECT/'checkpoints'/name
        dest.parent.mkdir(exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise ValueError('Checkpoint differs; use a separate project')
        else:
            shutil.copy2(source, dest)
    missing = []
    motion_count = 0
    for name in ('dataset_loco_sit_carry_climb.yaml','dataset_stage1_jointpush.yaml'):
        manifest = PROJECT/'tokenhsi/data'/name
        for rows in yaml.safe_load(manifest.read_text())['motions'].values():
            for row in rows:
                motion_count += 1
                for key in ('file','obj_file'):
                    if row.get(key) and not (manifest.parent/row[key]).is_file():
                        missing.append(str(manifest.parent/row[key]))
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    # Validate all PUSH motion/character bytes against the bank's contract.
    bank = json.loads((PROJECT/'tokenhsi/data/stage1_jointpush_rsi_bank.json').read_text())['contract']
    manifest = PROJECT/'tokenhsi/data/dataset_stage1_jointpush.yaml'
    motions = yaml.safe_load(manifest.read_text())['motions']['push']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert [digest(manifest.parent/row['file']) for row in motions] == bank['motions']
    asset = PROJECT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'
    import xml.etree.ElementTree as ET
    assert digest(asset) == bank['asset']
    assert [digest(asset.parent/m.get('file')) for m in ET.parse(asset).findall('./asset/mesh')] == bank['meshes']
    report = dict(commit=COMMIT,project=str(PROJECT),checkpoint_source=str(checkpoint),
                  checkpoint_sha256=digest(checkpoint),motion_rows=motion_count,push_clips=len(motions),
                  motion_paths_valid=True,push_bank_hashes_valid=True,training_started=False)
    out=PROJECT/'output_etc/check/branch_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    args=parser.parse_args()
    print(json.dumps(prepare(args.checkpoint,args.data_root),indent=2))
