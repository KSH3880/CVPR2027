"""Build CPU-only, unscaled source AMP caches from the selected BONES BVHs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tokenhsi'))
from utils.raw_push_motion import FORMAT, read_source, SourceAmpMotion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True,help='Directory containing original BONES .bvh files')
    parser.add_argument('--output',type=Path,default=ROOT.parents[2]/'runs/jointpush/raw_push_amp')
    parser.add_argument('--manifest',type=Path,default=ROOT/'tokenhsi/data/partition_push_raw_motions.yaml')
    args=parser.parse_args()
    selected=json.loads((ROOT/'tokenhsi/data/bones_push180_selection.json').read_text())['clips']
    base=yaml.safe_load((ROOT/'tokenhsi/data/partition_push_motions.yaml').read_text())
    sources={}
    for path in args.source_root.rglob('*.bvh'):
        if path.stem in selected: sources.setdefault(path.stem,[]).append(path)
    for name in selected:
        if len(sources.get(name,[]))!=1: raise ValueError('Need exactly one original BVH: '+name)
    args.output.mkdir(parents=True,exist_ok=True)
    records=[];push=[]
    for entry in base['motions']['push']:
        name=Path(entry['file']).parents[1].name
        crop=selected[name];source=sources[name][0]
        arrays=read_source(source,crop['start'],crop['end'])
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        target=args.output/(name+'.npz')
        np.savez_compressed(target,**arrays,source_sha256=np.array(sha))
        loaded=SourceAmpMotion.from_file(target)
        assert len(loaded.global_translation)==len(arrays['global_translation'])
        records.append(dict(clip=name,source=str(source.resolve()),source_sha256=sha,
            output=str(target.resolve()),frames=len(loaded.tensor),fps=loaded.fps,
            crop_start=crop['start'],crop_end=crop['end'],scale=1.,ik=False,
            joint_limit_clamp=False,ground_shift_m=0.))
        push.append(dict(file=os.path.relpath(target.resolve(),args.manifest.resolve().parent),weight=entry['weight']))
        print('RAW_SOURCE_AMP',name,len(loaded.tensor),flush=True)
    base['motions']['push']=push
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(yaml.safe_dump(base,sort_keys=False))
    (args.output/'audit.json').write_text(json.dumps(dict(format=FORMAT,clips=records,
        notes=['Source metre positions and segment lengths retained.',
               'Only units/axes, anatomical AMP joint representation, crop and 30fps sampling change.',
               'No target humanoid sizing, IK, joint limits or ground adjustment.',
               'AMP only; locomotion initialization still uses its existing reference.']),indent=2)+'\n')
    print('MANIFEST',args.manifest.resolve(),flush=True)


if __name__=='__main__': main()
