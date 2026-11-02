"""Small file protocol for switching native viewers without a blank reload gap."""
import json
import os
from pathlib import Path

KINDS = ('1+3', '2+2', '4')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def select_case(cases, current_file, code):
    """Return None for a missing category; preserve selection across report growth."""
    cases = sorted(cases, key=lambda c: (KINDS.index(c['category']),
                   not c.get('simultaneous', False), not c.get('final_all', False)))
    if not cases:
        return None
    if code == 44:
        current = next((i for i, c in enumerate(cases) if c['file'] == current_file), -1)
        return cases[(current + 1) % len(cases)]
    if 41 <= code <= 43:
        return next((c for c in cases if c['category'] == KINDS[code-41]), None)
    return next((c for c in cases if c['file'] == current_file), cases[0])
