"""Preliminary reference-time shortlist and probe-only replay. No training policy."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shortlist(source, output, bank, limit=2., preferred=1.):
    if not 0 < preferred <= limit:
        raise ValueError('Require 0 < preferred <= limit')
    source, output, bank = map(Path, (source, output, bank))
    root = Path(__file__).resolve().parents[3]
    if root not in output.resolve().parents:
        raise ValueError('Output must be inside koo_cvpr')
    summary = json.loads((source.parent/'summary.json').read_text())
    if summary.get('mode') != 'impact_sweep':
        raise ValueError('Expected original-reference impact_sweep results')
    metadata = json.loads(bank.with_suffix('.json').read_text())
    candidates, counts, seen = [], Counter(), set()
    for line in source.open():
        r = json.loads(line)
        skill = r['skill']
        if skill not in ('sit','climb','pickUp','carryWith','putDown'):
            continue
        counts['examined'] += 1
        if 'motion_id' not in r:
            raise ValueError('Source lacks original motion IDs; cannot replay')
        if not r['scene_finite'] or not .65 <= r['phase'] <= .98:
            continue
        if r['first_two_steps_delta_v'] > limit:
            continue
        key = (tuple(r['size_cm']),skill,r['motion_id'],r['phase'])
        if key in seen:
            continue
        seen.add(key)
        tier = 'preferred' if r['first_two_steps_delta_v'] <= preferred else 'conditional'
        candidate = {'id':len(candidates),'size_cm':r['size_cm'],'skill':skill,
                     'motion_id':r['motion_id'],'phase':r['phase'],'tier':tier,
                     'source_delta_v':r['first_two_steps_delta_v'],
                     'source_round':r['round'],'source_env':r['env'],'source_agent':r['agent']}
        candidates.append(candidate)
        counts[tier] += 1
        counts[skill] += 1
    if not candidates:
        raise ValueError('No candidates meet preliminary delta-v limit')
    payload = {'schema':1,'stage':'preliminary_not_training_safe',
        'source_sha256':sha(source),'source_summary':summary,
        'reference_metadata':metadata,'preferred_delta_v':preferred,'max_delta_v':limit,
        'counts':dict(counts),'candidates':candidates,
        'note':'Single observed teacher transition. No vertical/rotation/grasp safety inferred. '
               'Scene termination is not filtered because it may concern the teammate.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError('Choose a new shortlist output path')
    output.write_text(json.dumps(payload,ensure_ascii=False))
    output.with_suffix('.summary.json').write_text(json.dumps(
        {k:v for k,v in payload.items() if k not in ('candidates','reference_metadata')},indent=2))
    return payload


class ProbeCandidateReplay:
    def __init__(self, path, task, dataset, asset):
        import torch
        self.torch = torch
        self.payload = json.loads(Path(path).read_text())
        if self.payload.get('schema') != 1:
            raise ValueError('Unknown candidate schema')
        meta = self.payload['reference_metadata']
        if sha(dataset) != meta['dataset_sha256'] or sha(asset) != meta['asset_sha256']:
            raise ValueError('Candidate reference dataset/asset mismatch')
        for skill in meta['files']:
            for section, hashes in [('files','motion_sha256'),('objects','object_sha256')]:
                for filename, expected in zip(meta[section][skill],meta[hashes][skill]):
                    if sha(Path(dataset).parent/filename) != expected:
                        raise ValueError('Reference motion/object has changed')
        self.task = task
        self.sha256 = sha(path)
        self.data = {k:{} for k in ('sit','climb','pickUp','carryWith','putDown')}
        self.stats = {'candidate_draws':0,'missing_size_skill_draws':0}
        self.groups = defaultdict(list)
        for c in self.payload['candidates']:
            lib = task._motion_lib[c['skill']]
            if not 0 <= c['motion_id'] < lib.num_motions():
                raise ValueError('Candidate motion ID outside library')
            self.groups[(tuple(c['size_cm']),c['skill'])].append(c)

    def sample(self, name, sizes):
        t = self.torch
        lib = self.task._motion_lib[name]
        ids = lib.sample_motions(len(sizes))
        times = lib.sample_time_rsi(ids)
        for size in t.unique((sizes*100).round().long(),dim=0):
            mask = ((sizes*100).round().long()==size).all(-1)
            group = self.groups.get((tuple(size.cpu().tolist()),name),[])
            n = int(mask.sum())
            if not group:
                # Original same-skill RSI, same physical size; explicitly unmatched.
                self.stats['missing_size_skill_draws'] += n
                continue
            mids = t.tensor([c['motion_id'] for c in group],device=sizes.device)
            clips = mids.unique()
            chosen_clip = clips[t.multinomial(lib._motion_weights[clips],n,replacement=True)]
            selected = []
            for clip in chosen_clip.cpu().tolist():
                options = [c for c in group if c['motion_id']==clip]
                weights = t.tensor([2. if c['tier']=='preferred' else 1. for c in options],device=sizes.device)
                selected.append(options[int(t.multinomial(weights,1))])
            chosen_ids = t.tensor([c['motion_id'] for c in selected],device=sizes.device)
            phases = t.tensor([c['phase'] for c in selected],device=sizes.device)
            ids[mask] = chosen_ids
            times[mask] = phases*lib.get_motion_length(chosen_ids)
            self.stats['candidate_draws'] += n
        return ids,times,t.zeros(len(sizes),device=sizes.device,dtype=t.bool)

    def match(self, size, name, motion_id, phase):
        group = self.groups.get((tuple(size),name),[])
        found = [c for c in group if c['motion_id']==motion_id and abs(c['phase']-phase)<2e-6]
        return min(found,key=lambda c:abs(c['phase']-phase))['id'] if found else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reference-bank',required=True)
    p.add_argument('--max-delta-v',type=float,default=2.)
    p.add_argument('--preferred-delta-v',type=float,default=1.)
    a=p.parse_args()
    result=shortlist(a.source,a.output,a.reference_bank,a.max_delta_v,a.preferred_delta_v)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
