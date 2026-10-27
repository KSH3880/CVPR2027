"""Publish only physically accepted Table RSI poses, preserving donor provenance."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def merge(output, banks):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    manifests=[json.loads((b/'manifest.json').read_text()) for b in banks]
    assets=manifests[0]['assets'];report=[]
    for entry in assets:
        pieces=[];provenance=[]
        for bank in banks:
            path=bank/entry['id']/'carryWith.npz'
            if not path.exists():continue
            d=dict(np.load(path));valid=d['physics_pass'].astype(bool)
            if not valid.any():continue
            pieces.append({k:v[valid] for k,v in d.items()});provenance.extend([dict(bank=str(bank),index=int(i),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for i in np.flatnonzero(valid)])
        n=sum(len(d['root_state']) for d in pieces)
        if n:
            folder=output/entry['id'];folder.mkdir();np.savez_compressed(folder/'carryWith.npz',**{k:np.concatenate([d[k] for d in pieces]) for k in pieces[0]})
            (folder/'provenance.json').write_text(json.dumps(provenance,indent=2))
        report.append(dict(id=entry['id'],split=entry['split'],accepted=n))
    manifest=manifests[0];manifest.update(physics_validated=True,donor_banks=[str(b) for b in banks],coverage=report,physics_seconds=.3,missing_policy='explicit locomotion fallback; no synthetic/unscreened RSI')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('banks',nargs='+',type=Path);a=p.parse_args();merge(a.output,[b.resolve() for b in a.banks])
