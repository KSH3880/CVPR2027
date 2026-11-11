"""Export the matched learning curves and measured physics errors."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1] / 'output_etc/check/backend_parity'
summary = json.loads((ROOT / 'training_comparison.json').read_text())
physics = json.loads((ROOT / 'physics_summary.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for name, label, color, style in [
    ('worker_2048', 'Worker, 2048 + 2048', '#1765ab', '-'),
    ('shared_fixed_2048', 'Shared fixed, 2048 + 2048', '#e87619', '-'),
    ('historical_worker_2048', 'Historical worker, 2048 + 2048', '#777777', ':'),
    ('shared_buggy_4096', 'Shared before fix, 4096 + 2048*', '#bb3542', '--'),
]:
    if name not in summary:
        continue
    acc = EventAccumulator(str(Path(summary[name]['path']) / 'summaries'), size_guidance={'scalars': 0})
    acc.Reload()
    points = [x for x in acc.Scalars('reward_push/iter') if x.step <= 60]
    axes[0].plot([x.step for x in points], [x.value for x in points], style, color=color, label=label)
axes[0].set(xlabel='Training iteration', ylabel='Logged PUSH episode reward',
            title='KD coefficient 0.001, seed 42, fresh initialization')
axes[0].legend(fontsize=8, loc='upper left')
axes[0].text(.01, -.23, '*Historical buggy run has a different PUSH environment count.',
             transform=axes[0].transAxes, fontsize=8)
for key, label, color in [('shared_before', 'Before fix', '#bb3542'),
                           ('shared_fixed_rng', 'After fix', '#1765ab')]:
    rows = physics['reference'][key]
    axes[1].plot(range(1, len(rows) + 1), [r['position']['max'] for r in rows],
                 'o-', color=color, label=label)
axes[1].set(xlabel='Control step (fixed actions)', ylabel='Maximum position difference vs worker (m)',
            title='PUSH physics: identical reset RNG and actions')
axes[1].legend()
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(ROOT / 'backend_comparison.png', dpi=160)
fig.savefig(ROOT / 'backend_comparison.svg')
print(ROOT / 'backend_comparison.png')
