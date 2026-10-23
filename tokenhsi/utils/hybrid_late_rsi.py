"""Fixed-size physical RSI pools refined by the current policy's reset rollouts.

Success never changes size, task, clip weights or candidate admission. Only early
physical instability rejects a candidate. Pools are ordered by reference time.
"""
import hashlib
import json
import math

import torch

from utils.reference_rsi import digest


MODE = 'hybrid_late'
BANDS = ((.80, .95), (.70, .80), (.60, .70), (.50, .60))
CARRY_SKILLS = ('pickUp', 'carryWith', 'putDown', 'putDownOnTop')


def validate_hybrid_rsi(spec):
    required = {'mode', 'student_bank', 'cache_directory', 'phase_bands',
                'samples_per_band', 'validated_fraction', 'min_observations',
                'refresh_epochs', 'empty_skill', 'restore_box_velocity', 'screen'}
    if set(spec) != required or spec['mode'] != MODE:
        raise ValueError('Invalid hybrid RSI fields')
    if tuple(map(tuple, spec['phase_bands'])) != BANDS:
        raise ValueError('Hybrid RSI requires 80..95,70..80,60..70,50..60 percent')
    if (not all(isinstance(spec[k], str) and spec[k] for k in
                ('student_bank', 'cache_directory')) or
            type(spec['samples_per_band']) is not int or spec['samples_per_band'] < 2 or
            type(spec['min_observations']) is not int or spec['min_observations'] < 3 or
            type(spec['refresh_epochs']) is not int or spec['refresh_epochs'] < 1 or
            type(spec['validated_fraction']) not in (int, float) or
            not 0 <= spec['validated_fraction'] <= 1 or
            spec['empty_skill'] != 'loco' or spec['restore_box_velocity'] is not True):
        raise ValueError('Invalid hybrid RSI values')
    limits = spec['screen']
    expected = {'seconds', 'root_velocity_change', 'box_speed', 'root_displacement',
                'box_displacement', 'box_delta_v', 'carry_hand_increase'}
    if set(limits) != expected or any(type(v) not in (int, float) or
            not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError('Invalid hybrid RSI screen')


def phase_samples(spec, band):
    """Disjoint bands: boundaries .8/.7/.6 belong to the later band only."""
    lo, hi = spec['phase_bands'][band]
    count = spec['samples_per_band']
    if band:
        return [lo + (hi-lo)*i/count for i in range(count)]
    return [lo + (hi-lo)*i/(count-1) for i in range(count)]


def profile_key(skill, size, support=None):
    cm = tuple(round(float(v)*100) for v in size)
    return (skill, cm, tuple(round(float(v)*100) for v in support)) \
        if skill == 'putDownOnTop' else (skill, cm)


def encoded_keys(sizes, supports=None):
    cm = (sizes * 100).round().long()
    keys = (cm[:, 0]*100 + cm[:, 1])*100 + cm[:, 2]
    if supports is not None:
        keys = keys*1000000 + encoded_keys(supports)
    return keys


class HybridLateRSIBank:
    def __init__(self, spec, libs, dataset, asset, device, records, fingerprint):
        from utils.student_rsi import StudentRSIBank
        validate_hybrid_rsi(spec)
        self.spec, self.libs, self.device = spec, libs, device
        student_spec = dict(mode='student_candidates', bank=spec['student_bank'],
            min_observations=3, max_delta_v=2., carry_max_hand_increase=.2,
            late_fraction=.5, phase_range=[.9, .98],
            biased_skills=['pickUp', 'carryWith', 'putDown'],
            min_late_candidates=3, empty_skill='original', restore_box_velocity=True)
        self.student = StudentRSIBank(student_spec, libs, dataset, asset, device)
        self.sha256 = hashlib.sha256((fingerprint + self.student.sha256 +
            json.dumps(spec, sort_keys=True)).encode()).hexdigest()
        self.data = {skill: {} for skill in ('sit', 'climb', 'pickUp', 'carryWith', 'putDown')}
        self.tables, self.descriptors, self.stats = {}, [], {}
        for skill in CARRY_SKILLS:
            profiles = sorted((key, value) for key, value in records.items() if key[0] == skill)
            width = max((len(value) for _, value in profiles), default=1)
            size_values = torch.tensor([key[1] for key, _ in profiles], dtype=torch.float).reshape(-1, 3)/100
            support_values = torch.tensor([key[2] for key, _ in profiles], dtype=torch.float).reshape(-1, 3)/100 \
                if skill == 'putDownOnTop' else None
            keys = encoded_keys(size_values, support_values)
            order = keys.argsort()
            ids = torch.zeros(len(profiles), width, dtype=torch.long)
            phases = torch.zeros(len(profiles), width)
            bands = torch.full_like(ids, 4)
            weights = torch.zeros_like(phases)
            tokens = torch.full_like(ids, -1)
            for row, (key, candidates) in enumerate(profiles):
                for col, candidate in enumerate(candidates):
                    mid, phase, band = candidate
                    ids[row, col], phases[row, col], bands[row, col] = mid, phase, band
                    weights[row, col] = libs['putDown' if skill == 'putDownOnTop' else skill]._motion_weights[mid].item()
                    tokens[row, col] = len(self.descriptors)
                    self.descriptors.append((key, mid, phase, band))
            self.tables[skill] = tuple(t[order].to(device) for t in (keys, ids, phases, bands, weights, tokens))
        self.counts = torch.zeros(len(self.descriptors), dtype=torch.long, device=device)
        # Resume observations must belong to exactly the same ordered candidates.
        self.sha256 = hashlib.sha256((self.sha256 + json.dumps(
            self.descriptors, separators=(',', ':'))).encode()).hexdigest()
        self.failures = torch.zeros_like(self.counts)
        self.is_carry = torch.tensor([d[0][0] == 'carryWith' for d in self.descriptors],
                                     device=device, dtype=torch.bool)
        self.last_tokens = None
        self.last_epoch = 0
        self.generation = 0

    def on_epoch(self, epoch):
        generation = epoch // self.spec['refresh_epochs']
        if generation > self.generation:
            self.counts.zero_(); self.failures.zero_()
            self.generation = generation
            self.stats['policy_windows'] = generation
        self.last_epoch = epoch

    def observe(self, tokens, valid):
        mask = tokens >= 0
        ids = tokens[mask]
        self.counts.index_add_(0, ids, torch.ones_like(ids))
        self.failures.index_add_(0, ids, (~valid[mask]).long())
        self.stats['policy_observations'] = self.stats.get('policy_observations', 0) + int(mask.sum())
        self.stats['policy_failed_observations'] = self.stats.get('policy_failed_observations', 0) + int((~valid[mask]).sum())

    def sample(self, skill, sizes):
        if skill in ('sit', 'climb'):
            result = self.student.sample(skill, sizes)
            self.last_tokens = torch.full((len(sizes),), -1, device=self.device, dtype=torch.long)
            return result
        return self.sample_context(skill, sizes)

    def sample_context(self, skill, sizes, supports=None):
        """Return empty=True for legal loco fallback; never draw another size."""
        keys, ids, phases, bands, base, tokens = self.tables[skill]
        query = encoded_keys(sizes, supports)
        n = len(query)
        out_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        out_times = torch.zeros(n, device=self.device)
        self.last_tokens = torch.full_like(out_ids, -1)
        empty = torch.ones(n, dtype=torch.bool, device=self.device)
        self.stats['attempts'] = self.stats.get('attempts', 0) + n
        if len(keys) and len(self.descriptors):
            rows = torch.searchsorted(keys, query).clamp_max(len(keys)-1)
            present = keys[rows] == query
            t = tokens[rows].clamp_min(0)
            observed = self.counts[t] >= self.spec['min_observations']
            rejected = observed & (self.failures[t] > 0)
            usable = (base[rows] > 0) & ~rejected & present[:, None]
            first = bands[rows].masked_fill(~usable, 4).min(-1).values
            usable &= bands[rows] == first[:, None]
            stable = usable & observed & (self.failures[t] == 0)
            prefer = (torch.rand(n, device=self.device) < self.spec['validated_fraction']) & stable.any(-1)
            mask = torch.where(prefer[:, None], stable, usable)
            # Preserve clip weights independently of the number of admitted times.
            probabilities = torch.zeros_like(base[rows])
            for mid in range(self.libs['putDown' if skill == 'putDownOnTop' else skill].num_motions()):
                same_clip = mask & (ids[rows] == mid)
                probabilities += same_clip * base[rows] / same_clip.sum(-1, keepdim=True).clamp_min(1)
            active = probabilities.sum(-1) > 0
            if active.any():
                chosen = torch.multinomial(probabilities[active], 1).flatten()
                r = rows[active]
                out_ids[active] = ids[r, chosen]
                lib = self.libs['putDown' if skill == 'putDownOnTop' else skill]
                out_times[active] = phases[r, chosen] * lib.get_motion_length(out_ids[active])
                self.last_tokens[active] = tokens[r, chosen]
                empty[active] = False
                for band in range(4):
                    name = 'band_' + str(band)
                    self.stats[name] = self.stats.get(name, 0) + int((first[active] == band).sum())
                self.stats['validated_draws'] = self.stats.get('validated_draws', 0) + int(prefer[active].sum())
        name = 'loco_fallback_' + skill
        self.stats[name] = self.stats.get(name, 0) + int(empty.sum())
        return out_ids, out_times, empty

    def state_dict(self):
        return dict(sha256=self.sha256, counts=self.counts.cpu(), failures=self.failures.cpu(),
                    generation=self.generation, last_epoch=self.last_epoch, stats=self.stats.copy())

    def load_state_dict(self, state):
        if state['sha256'] != self.sha256 or state['counts'].shape != self.counts.shape:
            raise ValueError('Hybrid RSI candidate identity differs on resume')
        self.counts.copy_(state['counts']); self.failures.copy_(state['failures'])
        self.generation, self.last_epoch = state['generation'], state['last_epoch']
        self.stats = state['stats'].copy()
