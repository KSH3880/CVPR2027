"""PUSH task id and the RSI bank of the PUSH Stage1 (TokenHSI_stage1_push)."""

import hashlib
import json
from pathlib import Path

PUSH = 11


def load_bank(options, manifest):
    import yaml
    path = Path(manifest)
    rows = yaml.safe_load(path.read_text())['motions']['push']
    bank = json.loads(Path(options['rsi']['bank']).read_text())
    expected_hashes = bank['contract']['motion_hashes']['push']
    if len(rows) != len(expected_hashes) or any(
        hashlib.sha256((path.parent / row['file']).read_bytes()).hexdigest() != digest
        for row, digest in zip(rows, expected_hashes)
    ):
        raise ValueError('PUSH RSI motion hashes/order differ from the manifest')
    if bank['contract']['push_box'] != options['box']:
        raise ValueError('PUSH box physics differs from the RSI bank')
    return bank
