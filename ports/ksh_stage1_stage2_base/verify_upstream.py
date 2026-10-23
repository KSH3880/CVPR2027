"""Verify the immutable import; optionally verify the initial PUSH working copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-push', action='store_true',
                        help='Also require push/ to match the imported baseline')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    manifest = json.loads((base/'UPSTREAM_MANIFEST.json').read_text())
    expected = {entry['path'] for entry in manifest['files']}
    errors = []
    for name in ('upstream', 'push') if args.check_push else ('upstream',):
        root = base/name
        actual = set()
        for directory, dirs, files in os.walk(root, followlinks=False):
            directory = Path(directory)
            if name == 'upstream' and directory.stat().st_mode & 0o222:
                errors.append(f'{directory}: writable directory')
            for child in dirs + files:
                path = directory/child
                if path.is_symlink() or path.is_file():
                    actual.add(path.relative_to(root).as_posix())
        if actual != expected:
            errors.append(f'{name}: added={sorted(actual-expected)}, missing={sorted(expected-actual)}')
        for entry in manifest['files']:
            path = root/entry['path']
            try:
                if entry['git_mode'] == '120000':
                    data = os.fsencode(os.readlink(path))
                else:
                    if path.is_symlink():
                        raise ValueError('regular file replaced by symlink')
                    data = path.read_bytes()
                    executable = bool(path.stat().st_mode & 0o111)
                    if executable != (entry['git_mode'] == '100755'):
                        raise ValueError('executable mode changed')
                    if name == 'upstream' and path.stat().st_mode & 0o222:
                        raise ValueError('writable baseline file')
                if hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError('content differs')
            except (OSError, ValueError) as error:
                errors.append(f'{name}/{entry["path"]}: {error}')
        print(f'{name}: checked {len(expected)} entries against {manifest["commit"]}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('OK: imported contents and upstream read-only permissions verified')


if __name__ == '__main__':
    main()
