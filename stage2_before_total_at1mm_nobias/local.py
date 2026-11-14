"""Prepare local assets or verify the import without importing torch/Isaac Gym."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DATASETS = ('dataset_sit', 'dataset_carry', 'dataset_amass_climb', 'dataset_amass_loco')
CHECKPOINT = 'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
PUSH = 'tokenhsi/data/jointpush_motions'


def setup(args):
    data = args.data_root.resolve()
    for name in DATASETS:
        source = data / name / 'motions'
        if not source.is_dir():
            raise ValueError('Missing motion directory: ' + str(source))
        target = ROOT / 'tokenhsi/data' / name / 'motions'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise ValueError('Refusing to overwrite: ' + str(target))
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
    for rel, source in ((CHECKPOINT, args.checkpoint), (PUSH, args.jointpush_motions)):
        target = ROOT / rel
        if target.exists():
            continue
        if source is None or not source.exists():
            raise ValueError('Provide the source asset for ' + rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel == PUSH:
            shutil.copytree(source, target, symlinks=False)
        else:
            shutil.copy2(source, target)
    print('Local assets prepared. Run: python local.py check')


def check():
    import yaml

    manifest = json.loads((ROOT / 'IMPORT_MANIFEST.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            failures.append('Missing: ' + entry['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append('SHA256 mismatch: ' + entry['path'])
    for name in DATASETS:
        path = ROOT / 'tokenhsi/data' / name / 'motions'
        if not path.is_dir():
            failures.append('Missing dataset: ' + str(path))
    # Check every motion and object motion listed by the actual training recipe.
    motions = yaml.safe_load((ROOT / 'tokenhsi/data/dataset_stage1_jointpush.yaml').read_text())
    paths = []
    for entries in motions['motions'].values():
        for entry in entries:
            for key in ('file', 'obj_file'):
                if key in entry:
                    path = ROOT / 'tokenhsi/data' / entry[key]
                    paths.append(path)
                    if not path.is_file():
                        failures.append('Missing motion: ' + str(path))
    # Runtime dependencies may use shared baseline data, but never the ksh checkout.
    for path in [ROOT / CHECKPOINT, ROOT / PUSH] + paths:
        if '/home/hwanhee/ksh/' in str(path.resolve()):
            failures.append('Runtime dependency points into ksh: ' + str(path))
    if failures:
        raise ValueError('\n'.join(failures))
    print('PASS: {} imported file hashes, {} motion paths; no ksh runtime dependency.'.format(
        len(manifest['files']), len(paths)))
    print('GPU/simulator execution has not been tested by this command.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    prepare = sub.add_parser('setup')
    prepare.add_argument('--data-root', type=Path, required=True)
    prepare.add_argument('--checkpoint', type=Path)
    prepare.add_argument('--jointpush-motions', type=Path)
    args = parser.parse_args()
    try:
        setup(args) if args.command == 'setup' else check()
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
