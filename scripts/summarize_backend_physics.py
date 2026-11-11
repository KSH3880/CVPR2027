"""Summarize PUSH parity and controlled native warmup/world-origin experiments."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1] / 'output_etc/check/backend_parity'


def load(name):
    return torch.load(ROOT / name, map_location='cpu', weights_only=True)


def delta(a, b):
    x = (a.float() - b.float()).abs()
    return dict(max=x.max().item(), mean=x.mean().item())


reference = load('trace_worker.pt')['records']
fixed = load('trace_shared_fixed_rng.pt')['records']
report = {'reference': {}, 'native': {}}
for label in ('shared_before', 'shared_fixed_rng'):
    other = {x['stage']: x for x in load('trace_' + label + '.pt')['records']}
    rows = []
    for x in reference:
        if not x['stage'].startswith('step_'):
            continue
        y = other[x['stage']]
        rows.append(dict(stage=x['stage'], position=delta(x['root'][..., :3], y['root'][..., :3]),
                         **{k: delta(x[k], y[k]) for k in ('root', 'dof', 'body', 'obs', 'reward')}))
    report['reference'][label] = rows

fixed_native = [x for x in fixed if x['stage'] == 'constructed' or x['stage'].startswith(('step_', 'reset_native'))]
for label, name in [('original', 'native_worker_original.pt'), ('warmup', 'native_worker_warmup.pt'),
                    ('warmup_and_same_origins', 'native_worker_warmup_offset.pt')]:
    records = load(name)
    assert len(records) == len(fixed_native)
    rows = []
    for x, y in zip(records, fixed_native):
        z = y['native_state']
        rows.append(dict(stage=y['stage'], position=delta(x['_root_states'][..., :3], z['_root_states'][..., :3]),
                         **{k: delta(x[k], z[k]) for k in ('_root_states', '_dof_state', '_rigid_body_state', 'obs_buf')}))
    report['native'][label] = rows

(ROOT / 'physics_summary.json').write_text(json.dumps(report, indent=2) + '\n')
for label, rows in report['reference'].items():
    print('reference', label, 'max position', max(x['position']['max'] for x in rows),
          'max reward', max(x['reward']['max'] for x in rows))
for label, rows in report['native'].items():
    steps = [x for x in rows if x['stage'].startswith('step_')]
    print('native', label, 'max position', max(x['position']['max'] for x in steps),
          'max obs', max(x['obs_buf']['max'] for x in steps))
