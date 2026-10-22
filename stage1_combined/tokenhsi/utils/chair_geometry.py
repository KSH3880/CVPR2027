"""Compact chair descriptor: routing flag, backrest direction XY, seat XYZ."""
from pathlib import Path
import json
import numpy as np
EXTRA=6

def descriptor(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text())
    facing=np.asarray(cfg['facing'][:2],dtype=np.float32)
    norm=np.linalg.norm(facing)
    if norm<=0:raise ValueError('Chair facing must have a nonzero XY direction')
    seat=np.asarray(cfg['tarSitPos'],dtype=np.float32);seat[2]-=.1
    # Extent is target-region metadata only; it is not sent into the tokenizer.
    full=json.loads((Path(__file__).resolve().parents[2]/'chair_assets/catalog.json').read_text())
    entry=next(e for split in ['train','test'] for e in full[split] if e['id']==folder.name)
    feature=np.concatenate([[1.],-facing/norm,seat]).astype(np.float32)
    return feature,seat,np.asarray(entry['seat_extent'],dtype=np.float32),cfg
