"""Verify copied code, immutable artifacts, and shared data links without GPU."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'PORT_MANIFEST.json').read_text())
expected = dict(manifest['source_file_sha256'])
expected.update({row['path']: row['port_sha256'] for row in manifest['local_adaptations']})
expected.update({row['path']: row['sha256'] for row in manifest['copied_artifacts']})
errors = []
for name, digest in expected.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append(name)
for row in manifest['data_links']:
    path = root / row['path']
    if not path.is_symlink() or not path.exists() or str(path.resolve()) != row['target']:
        errors.append(row['path'])
if errors:
    raise SystemExit('Verification failed: ' + ', '.join(errors))
print(f"Verified {len(expected)} files and {len(manifest['data_links'])} shared data links")
