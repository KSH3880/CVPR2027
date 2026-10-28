"""Render Stage1/Stage2 QK, bias, sum logits and actual attention separately."""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
report = json.loads((out / 'verification.json').read_text())
states = json.loads((out / 'states.json').read_text())
gallery = ['<!doctype html><meta charset="utf-8"><title>Partial13000 attention</title>',
           '<style>body{font:16px sans-serif;margin:24px}img{max-width:95vw;width:1000px}a{display:block;margin:18px 0}</style>',
           '<h1>Partial13000: Stage1 encoder / Stage2 coordination</h1>',
           '<p>Columns: QK before bias | additive bias | QK+bias logits | softmax(QK+bias). Rows: layer/head. Actor only, unchanged checkpoint/RMS. Stage1 is the frozen encoder within this Stage2 model. Later QK includes effects from earlier biased layers.</p>',
           '<p>Stage1 H: humans; B: logical object slots (not necessarily distinct physical boxes); G: goals. Stage2 contains human tokens only. Colors are shared across logit panels within each branch; probability range is 0–1. Invalid masked entries are grey. Each scene stops contributing after its first termination.</p>']
verification = {}
for branch in ('stage1', 'stage2'):
    data = np.load(out / (branch + '_attention.npz'))
    qk, bias, logits, attention = [data[k] for k in ('qk','bias','logits','attention')]
    finite = np.isfinite(logits)
    assert np.allclose((qk + bias)[finite], logits[finite])
    stable = logits - logits.max(-1, keepdims=True)
    expected = np.exp(stable); expected /= expected.sum(-1, keepdims=True)
    err = float(np.abs(expected - attention).max())
    assert err < 2e-6
    verification[branch] = dict(shape=list(qk.shape), softmax_max_error=err)
    bound = max(float(np.abs(a[np.isfinite(a)]).max()) for a in (qk,bias,logits))
    labels = report[branch + '_labels']
    gallery.append('<h2>' + branch + '</h2>')
    for frame, state in enumerate(states):
        for scene, active in enumerate(state['active']):
            if not active: continue
            fig, axes = plt.subplots(8, 4, figsize=(20, 31), constrained_layout=True)
            for layer in range(4):
                for head in range(2):
                    row = layer * 2 + head
                    for col, (name, values) in enumerate(zip(('QK / sqrt(d), before bias','Additive bias','QK / sqrt(d) + bias (logits)','softmax(QK + bias)'), (qk,bias,logits,attention))):
                        ax = axes[row, col]
                        image = np.ma.masked_invalid(values[frame, scene, layer, head])
                        cmap = plt.get_cmap('viridis' if col == 3 else 'RdBu_r').copy(); cmap.set_bad('#bbbbbb')
                        im = ax.imshow(image, vmin=0 if col == 3 else -bound, vmax=1 if col == 3 else bound, cmap=cmap)
                        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
                        ax.set_yticks(range(len(labels)), labels, fontsize=7)
                        ax.set_title(f'L{layer+1} head{head+1}: {name}', fontsize=10)
                        ax.set_xlabel('key'); ax.set_ylabel('query')
                        fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
            assignment = state['assignment'][scene]
            fig.suptitle(f'PartialRawCuboid 13000 | {branch} | scene {scene}, step {state["step"]}\nHuman physical box assignment: {assignment} | QK/bias/logits shared scale ±{bound:.3g}', fontsize=16)
            name = f'{branch}_scene{scene}_step{state["step"]:04d}.png'
            fig.savefig(out/name, dpi=110); plt.close(fig)
            gallery.append(f'<a href="{name}">{branch} scene {scene}, step {state["step"]}<br><img loading="lazy" src="{name}"></a>')
(out/'index.html').write_text('\n'.join(gallery))
(out/'render_verification.json').write_text(json.dumps(verification, indent=2))
print(json.dumps(verification))
