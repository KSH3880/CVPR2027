"""Prepare motion data or verify this scratch-training import without GPU imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DATASETS = ('dataset_sit', 'dataset_carry', 'dataset_amass_climb', 'dataset_amass_loco')
PUSH = 'tokenhsi/data/jointpush_motions'


def setup(args):
    sources = [(args.data_root.resolve() / name / 'motions',
                ROOT / 'tokenhsi/data' / name / 'motions') for name in DATASETS]
    for source, target in sources:
        if not source.is_dir():
            raise ValueError('Missing motion directory: ' + str(source))
        if (target.exists() or target.is_symlink()) and not (
                target.is_symlink() and target.resolve() == source):
            raise ValueError('Refusing to overwrite: ' + str(target))
    target = ROOT / PUSH
    if target.is_symlink():
        raise ValueError('PUSH motions must be a local copy: ' + str(target))
    if not target.exists() and (args.jointpush_motions is None or not args.jointpush_motions.is_dir()):
        raise ValueError('Provide --jointpush-motions /path/to/jointpush_motions')
    for source, link in sources:
        if not link.is_symlink():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)
    if not target.exists():
        shutil.copytree(args.jointpush_motions, target, symlinks=False)
    print('Local motion data prepared. Run: python local.py check')


def check():
    import yaml

    manifest = json.loads((ROOT / 'IMPORT_MANIFEST.json').read_text())
    errors = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            errors.append('Missing: ' + entry['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append('SHA256 mismatch: ' + entry['path'])
    for name in DATASETS:
        path = ROOT / 'tokenhsi/data' / name / 'motions'
        if not path.is_dir():
            errors.append('Missing dataset: ' + str(path))
    motions = yaml.safe_load((ROOT / 'tokenhsi/data/dataset_stage1_jointpush.yaml').read_text())
    paths = []
    for entries in motions['motions'].values():
        for entry in entries:
            for key in ('file', 'obj_file'):
                if key in entry:
                    path = ROOT / 'tokenhsi/data' / entry[key]
                    paths.append(path)
                    if not path.is_file():
                        errors.append('Missing motion: ' + str(path))
    for path in [ROOT / PUSH] + paths:
        if '/home/hwanhee/ksh/' in str(path.resolve()):
            errors.append('Runtime dependency points into ksh: ' + str(path))
    if errors:
        raise ValueError('\n'.join(errors))
    print('PASS: {} imported file hashes, {} motion paths; no ksh runtime dependency.'.format(
        len(manifest['files']), len(paths)))
    print('Scratch training needs no initial checkpoint. GPU/simulator execution was not tested.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    prepare = sub.add_parser('setup')
    prepare.add_argument('--data-root', type=Path, required=True)
    prepare.add_argument('--jointpush-motions', type=Path)
    args = parser.parse_args()
    try:
        setup(args) if args.command == 'setup' else check()
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
