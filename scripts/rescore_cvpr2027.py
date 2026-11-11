"""CPU-only 1109 rescoring of archived single-skill trajectories; preserves originals."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]
from learning.cvpr2027_eval import metrics_frame
from learning.cvpr2027_scoring import ever_from_trace, trial_minimum_distances, error_statistics, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='One task/seed directory containing result.json and raw traces')
    p.add_argument('--output', required=True, help='New JSON report path; originals are never overwritten')
    args = p.parse_args()
    src, output = Path(args.input).resolve(), Path(args.output).resolve()
    if output.exists():
        p.error('Output already exists')
    report = json.loads((src/'result.json').read_text())
    initial = np.load(src/'initial.npz', allow_pickle=False)
    roots = np.load(src/'roots.npy', mmap_mode='r')
    boxes = np.load(src/'boxes.npy', mmap_mode='r')
    falls = np.load(src/'fallen.npy', mmap_mode='r')[:report['frames']]
    metrics = [metrics_frame(report['task'], roots[i], boxes[i], initial['sizes'], initial['goals'], 1., report.get('scoring_mode', 'protocol_region'))
               for i in range(report['frames'])]
    ever, valid = ever_from_trace(np.stack([x[0] for x in metrics]), falls)
    minimum = trial_minimum_distances(np.stack([x[1] for x in metrics]), valid)
    selected = np.ones(len(ever), dtype=bool)
    result = dict(protocol='cvpr2027_single_1109', xy_fraction=1., task=report['task'], seed=report['seed'],
                  scoring_mode=report.get('scoring_mode', 'protocol_region'), source_directory=str(src), source_protocol=report['protocol'],
                  source_result_sha256=hashlib.sha256((src/'result.json').read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256((ROOT/'cvpr2027_evaluation_protocol_1109.md').read_bytes()).hexdigest(),
                  frames=report['frames'], scenes=len(ever), checkpoint=report['checkpoint'],
                  point=counts(ever[:,0],selected), region=counts(ever[:,1],selected),
                  errors=error_statistics(minimum,ever[:,1]), non_paper_smoke=report.get('non_paper_smoke',False),
                  scope='Rescoring only: source physics, objects and trajectories are unchanged')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
