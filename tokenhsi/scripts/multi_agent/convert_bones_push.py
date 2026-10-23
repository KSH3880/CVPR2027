"""Convert selected BONES front two-hand PUSH references; retain the legacy pair."""
from pathlib import Path
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from isaacgym import gymapi  # Must precede torch/Poselib.
from tokenhsi.scripts.multi_agent.convert_bones_door import ROOT, convert

CLIPS = {'push_obstacle_180_101__A341': 12.2,
         'push_obstacle_180_003__A342': 11.733333}

def convert_selected(item):
    name, crop = item
    paths = list((ROOT / 'bones_seed/soma_uniform/bvh').rglob(name + '.bvh'))
    if len(paths) != 1:
        raise RuntimeError('Extract exactly one selected original BVH first: ' + name)
    output = ROOT / 'tokenhsi/data/dataset_bones_push180/motions' / name / 'phys_humanoid_v3'
    # A completed audited output can be reused; incomplete outputs are rebuilt.
    report = output / 'conversion_report.json'
    if report.is_file() and (output / 'ref_motion.npy').is_file():
        audit = json.loads(report.read_text())
        import hashlib
        if (audit['source_sha256'] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
                and audit['source_crop_start_s'] == crop['start']
                and audit['source_crop_end_s'] == crop['end']):
            return name
    convert(paths[0], output, start=crop['start'], end=crop['end'], profile='push')
    return name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded-180', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.expanded_180:
        selected = json.loads((ROOT / 'tokenhsi/data/bones_push180_selection.json').read_text())['clips']
        assert len(selected) == 21 and all('_180_' in n for n in selected)
        pending = [(n, c) for n, c in selected.items() if n not in CLIPS]
        for name in CLIPS:
            if not (ROOT / 'tokenhsi/data/dataset_bones_push/motions' / name / 'phys_humanoid_v3/ref_motion.npy').is_file():
                raise RuntimeError('Legacy converted clip missing: ' + name)
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for name in pool.map(convert_selected, pending):
                print('CONVERTED', name, flush=True)
    else:
        for name, end in CLIPS.items():
            paths = list((ROOT / 'bones_seed/soma_uniform/bvh').rglob(name + '.bvh'))
            if len(paths) != 1:
                raise RuntimeError('Extract the selected original BVH first: ' + name)
            output = ROOT / 'tokenhsi/data/dataset_bones_push/motions' / name / 'phys_humanoid_v3'
            convert(paths[0], output, start=0., end=end, profile='push')
