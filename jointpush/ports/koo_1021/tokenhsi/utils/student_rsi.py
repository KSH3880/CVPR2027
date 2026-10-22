"""Original-reference RSI selected by fixed-size repeated student observations."""
from collections import defaultdict
import json
from pathlib import Path

import torch

from utils.reference_rsi import REFERENCE_SKILLS, digest, geometry_digest


def validate_student_rsi(spec):
    keys={'mode','bank','min_observations','max_delta_v','carry_max_hand_increase',
          'late_fraction','phase_range','biased_skills','min_late_candidates',
          'empty_skill','restore_box_velocity'}
    if set(spec)!=keys or spec['mode']!='student_candidates':
        raise ValueError('Invalid student referenceRsi fields')
    if (not isinstance(spec['bank'],str) or not spec['bank'] or
        type(spec['min_observations']) is not int or spec['min_observations']<3 or
        type(spec['min_late_candidates']) is not int or spec['min_late_candidates']<1 or
        type(spec['max_delta_v']) not in (int,float) or not 0<spec['max_delta_v']<=2 or
        type(spec['carry_max_hand_increase']) not in (int,float) or not 0<spec['carry_max_hand_increase']<=.2 or
        type(spec['late_fraction']) not in (int,float) or not 0<=spec['late_fraction']<=1 or
        len(spec['phase_range'])!=2 or not .65<=spec['phase_range'][0]<spec['phase_range'][1]<=.98 or
        not spec['biased_skills'] or not set(spec['biased_skills'])<=set(REFERENCE_SKILLS) or
        spec['empty_skill']!='original' or spec['restore_box_velocity'] is not True):
        raise ValueError('Invalid student referenceRsi values')


class StudentRSIBank:
    def __init__(self,spec,motion_libs,dataset,asset,device):
        validate_student_rsi(spec)
        self.spec=spec; self.device=device; self.sha256=digest(spec['bank'])
        payload=json.loads(Path(spec['bank']).read_text())
        if payload.get('schema')!=1 or payload.get('stage')!='student_repeated_low_impact_not_safety_certified':
            raise ValueError('Expected repeated student reference candidates')
        summary=payload['source_student_summary']
        if (summary.get('measurement_version')!=2 or summary.get('policy_role')!='student' or
            summary.get('nonfinite_scenes')!=0 or not summary.get('partial_reset_checked')):
            raise ValueError('Invalid student measurement provenance')
        meta=payload['reference_metadata']
        if digest(dataset)!=meta['dataset_sha256'] or digest(asset)!=meta['asset_sha256']:
            raise ValueError('Student RSI reference dataset/asset mismatch')
        if 'geometry_sha256' in meta and geometry_digest(asset)!=meta['geometry_sha256']:
            raise ValueError('Student RSI mesh content mismatch')
        for skill in REFERENCE_SKILLS:
            lib=motion_libs[skill]
            files=[str(Path(f).resolve()) for f in lib._motion_files]
            expected=[str((Path(dataset).parent/f).resolve()) for f in meta['files'][skill]]
            if files!=expected:
                raise ValueError('Student RSI motion ordering mismatch: '+skill)
            for section,hashes in [('files','motion_sha256'),('objects','object_sha256')]:
                for filename,sha in zip(meta[section][skill],meta[hashes][skill]):
                    if digest(Path(dataset).parent/filename)!=sha:
                        raise ValueError('Student RSI reference changed: '+filename)
        self.data={skill:{} for skill in REFERENCE_SKILLS}
        groups=defaultdict(list)
        excluded=0
        for c in payload['candidates']:
            skill=c['skill']; mid=c['motion_id']; phase=c['phase']
            if skill not in self.data or not 0<=mid<motion_libs[skill].num_motions() or not .65<=phase<=.98:
                raise ValueError('Invalid original student RSI motion/time')
            if (c['student_n']<spec['min_observations'] or
                c['student_first2_delta_v_max']>spec['max_delta_v'] or
                (skill=='carryWith' and c['student_hand_box_distance_increase_max_m']>spec['carry_max_hand_increase'])):
                excluded+=1;continue
            groups[(skill,tuple(c['size_cm']))].append(c)
        for (skill,size),cs in groups.items():
            lo,hi=spec['phase_range']
            late=[c for c in cs if lo<=c['phase']<=hi]
            earlier=[c for c in cs if c['phase']<lo]
            general=(earlier if skill in spec['biased_skills'] and
                     len(late)<spec['min_late_candidates'] and earlier else cs)
            self.data[skill][size]=(self._pool(general),self._pool(late) if len(late)>=spec['min_late_candidates'] else None)
        self.libs=motion_libs
        self.tables={skill:self._table(skill) for skill in REFERENCE_SKILLS}
        self.stats={'attempts':0,'late_requests':0,'late_selected':0,'same_size_general_fallback':0,
                    'missing_size_skill_original':0,'excluded_candidates':excluded}

    def _pool(self,cs):
        mids=torch.tensor([c['motion_id'] for c in cs])
        phases=torch.tensor([c['phase'] for c in cs])
        # Weight actual student observations, not inherited teacher tiers.
        quality=torch.tensor([2. if c['student_first2_delta_v_max']<=1 else 1. for c in cs])
        return mids,phases,quality

    @staticmethod
    def size_keys(cm):
        return cm[:,0]*1000000+cm[:,1]*1000+cm[:,2]

    def _table(self,skill):
        # Pad once at startup; resets draw all rows on GPU without per-size loops.
        groups=sorted(self.data[skill].items())
        width=max((len(pool[0]) for _,pair in groups for pool in pair if pool is not None),default=1)
        keys=self.size_keys(torch.tensor([size for size,_ in groups],dtype=torch.long).reshape(-1,3))
        ids=torch.zeros(len(groups),2,width,dtype=torch.long)
        phases=torch.zeros(len(groups),2,width)
        weights=torch.zeros_like(phases)
        late_ok=torch.zeros(len(groups),dtype=torch.bool)
        skill_weights=self.libs[skill]._motion_weights.detach().cpu()
        for row,(_,pair) in enumerate(groups):
            late_ok[row]=pair[1] is not None
            for kind,pool in enumerate(pair):
                if pool is None:continue
                mids,ps,quality=pool;n=len(mids)
                probs=torch.zeros_like(quality)
                for mid in mids.unique():
                    mask=mids==mid
                    probs[mask]=skill_weights[mid]*quality[mask]/quality[mask].sum()
                if not (probs.sum()>0):raise ValueError('No positive original motion weights')
                ids[row,kind,:n]=mids;phases[row,kind,:n]=ps;weights[row,kind,:n]=probs
        return tuple(t.to(self.device) for t in (keys,ids,phases,weights,late_ok))

    def sample(self,skill,sizes):
        lib=self.libs[skill];n=len(sizes)
        previous_missing=self.stats['missing_size_skill_original']
        previous_general=self.stats['same_size_general_fallback']
        ids=lib.sample_motions(n);times=lib.sample_time_rsi(ids)
        # Missing pools retain SAME size and SAME skill's original allowed RSI.
        # False means caller must not replace the skill with loco.
        empty=torch.zeros(n,device=sizes.device,dtype=torch.bool)
        cm=(sizes*100).round().long()
        requested=(torch.rand(n,device=sizes.device)<self.spec['late_fraction']) if skill in self.spec['biased_skills'] else empty
        keys,table_ids,phases,weights,late_ok=self.tables[skill]
        if len(keys):
            query=self.size_keys(cm);rows=torch.searchsorted(keys,query).clamp_max(len(keys)-1)
            present=keys[rows]==query
            preferred=requested&present&late_ok[rows]
            source_rows=rows[present];kind=preferred[present].long()
            if present.any():
                choices=torch.multinomial(weights[source_rows,kind],1).flatten()
                selected=table_ids[source_rows,kind,choices]
                ids[present]=selected
                times[present]=phases[source_rows,kind,choices]*lib.get_motion_length(selected)
            self.stats['same_size_general_fallback']+=int((requested&present&~preferred).sum())
            self.stats['late_selected']+=int(preferred.sum())
            self.stats['missing_size_skill_original']+=int((~present).sum())
        else:
            self.stats['missing_size_skill_original']+=n
        self.stats['attempts']+=n;self.stats['late_requests']+=int(requested.sum())
        self.stats['draws_'+skill]=self.stats.get('draws_'+skill,0)+n
        for label,change in [('missing_original_'+skill,self.stats['missing_size_skill_original']-previous_missing),
                             ('general_fallback_'+skill,self.stats['same_size_general_fallback']-previous_general)]:
            self.stats[label]=self.stats.get(label,0)+change
        return ids,times,empty
