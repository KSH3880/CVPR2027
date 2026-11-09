"""Verify immutable delivery files without importing torch or Isaac Gym."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'BUNDLE_SHA256.json').read_text())
    errors = []
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file():
            errors.append('missing: ' + name)
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            errors.append('changed: ' + name)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('BUNDLE_OK: {} files'.format(len(manifest)))


if __name__ == '__main__':
    main()
