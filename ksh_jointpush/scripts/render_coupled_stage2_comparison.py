"""Compare relation bias, raw QK and softmax on three real actor rollouts."""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ksh_jointpush/coupled_base/tokenhsi')]
import numpy as np
import torch
from ksh_jointpush.ksh_torch_compat import install_attention_compat
install_attention_compat()
from ksh_jointpush.ksh_stage2_attention import canonical_coordination_maps
from learning.multi_agent.human_coordination import HumanCoordination
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATTERNS = [('1plus3', '1 solo + 3 coupled', [0, 1, 1, 1]),
            ('2plus2', '2 coupled + 2 coupled', [0, 0, 2, 2]),
            ('4', '4 coupled', [0, 0, 0, 0]),
            ('4solo', '4 independent solo PUSH', [0, 1, 2, 3])]
COLORS = ['#6ebf24', '#3984d3', '#d48b22', '#ad42bc']
COMPONENTS = ['relation_bias', 'raw_qk_scores', 'attention']


def load_cases(out):
    cases, sources = {}, {}
    checkpoint = None
    for key, title, groups in PATTERNS:
        source = out/key
        if key == '4solo' and not (source/'capture.json').exists():
            continue
        meta = json.loads((source/'capture.json').read_text())
        assert meta['groups'] == groups and meta['optimizer_steps'] == 0
        assert meta['model_and_rms_unchanged'] and meta['unchanged_action_max_error'] == 0
        sources[key] = str(source.resolve())
        actual_hash = hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()
        assert actual_hash == meta['checkpoint_sha256']
        if checkpoint is None:
            checkpoint = torch.load(meta['checkpoint'], map_location='cpu')
            expected_hash = actual_hash
            branch = HumanCoordination(num_layers=meta['layer_count'], heads=meta['head_count'],
                                       shuffle_human_order=True, relations=4)
            prefix = 'a2c_network.coordination.'
            branch.load_state_dict({name[len(prefix):]: value for name, value in checkpoint['model'].items()
                                    if name.startswith(prefix)}, strict=True)
        assert actual_hash == expected_hash, 'All cases must use the same checkpoint'
        data = torch.load(source/'actor_coordination_inputs.pt', map_location='cpu')
        states = json.loads((source/'states.json').read_text())
        assert len(states) == meta['frames'] == len(data['humans'])
        assert all(s['groups'] == groups for s in states)
        assert torch.equal(data['groups'], data['groups'].new_tensor(groups).expand_as(data['groups']))
        trace = canonical_coordination_maps(branch, data['humans'], data['packet'], data['groups'])
        arrays = {name: trace[name].numpy() for name in COMPONENTS + ['qk_scores']}
        assert all(np.isfinite(a).all() for a in arrays.values())
        head_dim = branch.blocks[0].attention.head_dim
        replay_error = float((trace['output']-data['output']).abs().max())
        logits = arrays['raw_qk_scores']/math.sqrt(head_dim) + arrays['relation_bias']
        weights = np.exp(logits-logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        softmax_error = float(np.abs(weights-arrays['attention']).max())
        row_error = float(np.abs(arrays['attention'].sum(-1)-1).max())
        assert replay_error < 3e-5 and softmax_error < 2e-6 and row_error < 2e-6
        group_array = np.asarray(groups)
        relation = np.where(np.eye(4, dtype=bool), 1,
                            np.where(group_array[:, None] == group_array[None, :], 3, 0))
        per_edge = {name: float(arrays['attention'][..., relation == index].mean())
                    if (relation == index).any() else None
                    for index, name in [(1, 'SELF'), (3, 'COUPLED peer'), (0, 'NONE cross-group')]}
        info = dict(pattern=key, title=title, groups=groups, frames=len(states),
                    episodes=meta['episodes'], source=sources[key], head_dim=head_dim,
                    canonical_replay_output_max_error=replay_error,
                    mha_reference_max_error=trace['mha_reference_max_error'],
                    raw_qk_softmax_max_error=softmax_error, row_sum_max_error=row_error,
                    mean_per_edge=per_edge, model_and_rms_unchanged=True,
                    unchanged_action_max_error=meta['unchanged_action_max_error'], optimizer_steps=0)
        target = out/'analysis'/key
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target/'stage2_attention_raw.npz', **arrays)
        (target/'summary.json').write_text(json.dumps(info, indent=2)+'\n')
        labels = [f'H{i} '+('solo' if groups.count(g) == 1 else f'team {sorted(set(groups)).index(g)+1}')
                  for i, g in enumerate(groups)]
        cases[key] = dict(info=info, arrays=arrays, mean={k: a.mean(0) for k, a in arrays.items()},
                          states=states, labels=labels)
        print('CASE_VERIFIED', key, json.dumps(info), flush=True)
    summary = dict(checkpoint=meta['checkpoint'], checkpoint_sha256=expected_hash,
                   checkpoint_epoch=checkpoint['epoch'], head_dim=head_dim,
                   formula='softmax(Q K^T / sqrt(head_dim) + relation_bias, axis=key)',
                   array_layout='frame, layer, head, canonical query agent, canonical key agent',
                   raw_qk_scores='Unscaled projected Q K^T', qk_scores='Q K^T / sqrt(head_dim)',
                   mean_note='Each component is averaged over sampled frames separately; mean(softmax(logits)) is not softmax(mean(logits)).',
                   sampling_note='Three rollouts per layout, sampled before actions every 10 steps, horizon 600 or environment termination; no success filtering. Captures may reuse prior verified rollouts or come from separate player processes; scenes are not paired. See per-layout capture metadata.',
                   cases={key: c['info'] for key, c in cases.items()})
    return cases, summary


def draw_map(ax, values, case, component, bound):
    probability = component == 'attention'
    ax.imshow(values, cmap='viridis' if probability else 'coolwarm',
              vmin=0 if probability else -bound, vmax=1 if probability else bound)
    ax.set_xticks(range(4), case['labels'], rotation=12, fontsize=8)
    ax.set_yticks(range(4), case['labels'], fontsize=8)
    for i in range(4):
        ax.get_xticklabels()[i].set_color(COLORS[i])
        ax.get_yticklabels()[i].set_color(COLORS[i])
        if i < 3 and case['info']['groups'][i] != case['info']['groups'][i+1]:
            ax.axhline(i+.5, color='#aaa', lw=2)
            ax.axvline(i+.5, color='#aaa', lw=2)
        for j in range(4):
            value = float(values[i, j])
            light = value < .6 if probability else abs(value) > .7*bound
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if light else '#172032')
    ax.set_xlabel('Key agent'); ax.set_ylabel('Query agent')


def make_pngs(out, cases, summary, bounds):
    layers, heads = next(iter(cases.values()))['mean']['attention'].shape[:2]
    for scaled in (False, True):
        names = ['relation_bias', 'qk_scores' if scaled else 'raw_qk_scores', 'attention']
        titles = ['Relation bias', 'QK / sqrt(d)' if scaled else 'Raw QK (unscaled)', 'Final softmax']
        suffix = '_scaled' if scaled else ''
        for key, case in cases.items():
            fig, axes = plt.subplots(layers*heads, 3, figsize=(14, 4.1*layers*heads), constrained_layout=True)
            for layer in range(layers):
                for head in range(heads):
                    for col, name in enumerate(names):
                        ax = axes[layer*heads+head, col]
                        draw_map(ax, case['mean'][name][layer, head], case, name, bounds[name])
                        ax.set_title(f'L{layer+1} H{head+1} | {titles[col]}\n'
                                     + ('color 0..1' if name == 'attention' else f'color +/- {bounds[name]:.3f}'), fontsize=11)
            fig.suptitle(f"{summary['checkpoint_epoch']} iter | {case['info']['title']} | {case['info']['frames']} sampled frames\n"
                         'Each column has one common color scale across all available layouts; rows = query, columns = key\n'
                         f"Final weights = softmax(raw QK / sqrt({summary['head_dim']}) + bias); displayed means are averaged separately", fontsize=13)
            fig.savefig(out/f'{key}_mean{suffix}.png', dpi=150)
            plt.close(fig)
        fig, axes = plt.subplots(len(cases), 3, figsize=(14, 4.5*len(cases)), constrained_layout=True)
        for row, (key, case) in enumerate(cases.items()):
            for col, name in enumerate(names):
                ax = axes[row, col]
                draw_map(ax, case['mean'][name][-1, 0], case, name, bounds[name])
                ax.set_title(case['info']['title']+' | '+titles[col]+ '\n'
                             + ('color 0..1' if name == 'attention' else f'color +/- {bounds[name]:.3f}'), fontsize=11)
        fig.suptitle(f"Stage2 {summary['checkpoint_epoch']} iter | Layer {layers}, Head 1 | sampled-frame means\n"
                     f"softmax(raw QK / sqrt({summary['head_dim']}) + bias); common scale per column across layouts", fontsize=14)
        fig.savefig(out/f'comparison_mean{suffix}.png', dpi=170)
        plt.close(fig)


def main():
    out = Path(sys.argv[1]).resolve()
    cases, summary = load_cases(out)
    bounds = {name: max(max(float(np.abs(c['arrays'][name]).max()) for c in cases.values()), 1e-4)
              for name in COMPONENTS + ['qk_scores']}
    summary['color_bounds'] = bounds
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    make_pngs(out, cases, summary, bounds)
    payload = dict(summary=summary, bounds=bounds, cases={key: dict(info=c['info'], labels=c['labels'],
                   states=c['states'], arrays={k: a.round(7).tolist() for k, a in c['arrays'].items()},
                   mean={k: a.round(7).tolist() for k, a in c['mean'].items()}) for key, c in cases.items()})
    template = Path(__file__).with_name('stage2_attention_comparison.html').read_text()
    (out/'index.html').write_text(template.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':'), allow_nan=False)))
    (out/'README.md').write_text('Stage2 attention comparison: 1+3, 2+2, 4 coupled, optional 4 solo\n\n'
        'Open index.html. Default view: all available layouts, layer 4/head 1, sampled-frame means. '
        'Choose any/all layers and heads or inspect the same episode/step across layouts. '
        'Each row shows relation bias, unscaled raw QK, and final softmax side by side. '
        'The scale checkbox switches raw QK to the actual content logits QK/sqrt(d). '
        'Color bounds are shared across layouts within each column; optional common logit scale shares bias/QK bounds. '
        'Raw arrays: analysis/<layout>/stage2_attention_raw.npz. Shapes: frame/layer/head/query/key. '
        'raw_qk_scores is unscaled QK; qk_scores includes sqrt(d) scaling. '
        'Source captures are preserved. See summary.json for original paths and independent numerical checks. '
        'Means of logits and softmax are computed separately. No success filtering or paired-scene claim.\n')
    print('COMPARISON_RENDERED', out, flush=True)


if __name__ == '__main__':
    main()
