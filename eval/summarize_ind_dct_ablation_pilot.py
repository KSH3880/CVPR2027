import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from run_ind_dct_ablation_pilot import ROOT, OUT, REFERENCE
sys.path.insert(0, str(ROOT/'tokenhsi'))
import stage2_xy100
from run_stage2_scalability import verify


def main():
    stage2_xy100.install()
    torch.set_num_threads(1)
    source = ROOT/'output_etc/check/ind_approach_diagnosis_v4_2000/analyze.py'
    spec = importlib.util.spec_from_file_location('carry_diagnosis', source)
    diagnosis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnosis)
    checks, summary, trials = {}, {}, []
    for mode in ('none', 'self_only', 'bypass'):
        path = OUT/mode
        checks[mode] = verify(path, REFERENCE, 2, 64)
        result = json.loads((path/'result.json').read_text())
        proof = result['dct_ablation']
        for key in ('first_policy_input_exact', 'disabled_action_exact', 'observation_unchanged',
                    'rng_restored', 'weights_exact_before_after', 'normalization_exact_before_after'):
            if not proof[key]:
                raise ValueError(mode+'/'+key)
        calls = result['simulated_steps'] + 3
        if result['encoder_attention_checks'] != dict(actor=calls, critic=calls):
            raise ValueError('Unexpected encoder call counts')
        if result['distractor_attention'] != 'all_observed_tokens':
            raise ValueError('SCT observation masking changed')
        data = np.load(path/'trace.npz')
        initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
        skills = data['packet'][..., 1]
        ever = data['ever'][..., 1]
        by_skill = {}
        for i, name in enumerate(('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')):
            mask = skills == i
            by_skill[name] = dict(n=int(mask.sum()), success=int(ever[mask].sum()))
        rows, _ = diagnosis.collect(path, 'independent', mode, 1, None)
        for row in rows:
            e, a = row['env'], row['slot']
            payload = int(data['packet'][e, a, 3])-2
            root = initial['roots'][e, a].numpy().astype(float)
            obj = initial['boxes'][e, payload, :2].numpy().astype(float)
            x, y, z, w = root[3:7]/np.linalg.norm(root[3:7])
            yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            bearing = np.arctan2(*(obj-root[:2])[::-1])
            angle = abs(np.arctan2(np.sin(bearing-yaw), np.cos(bearing-yaw)))
            row['initial_angle_deg'] = float(np.degrees(angle))
            row['rear'] = bool(angle > np.pi/2)
            step = round(3/result['control_dt'])-1
            valid = step < len(data['roots']) and bool(data['valid'][step, e, a])
            row['valid_at_3s'] = valid
            end = data['roots'][min(step, len(data['roots'])-1), e, a, :2].astype(float)
            displacement = end-root[:2]
            moving = np.linalg.norm(displacement) > .1
            direction = np.arctan2(displacement[1], displacement[0])
            row['movement_vs_target_deg_3s'] = float(np.degrees(abs(np.arctan2(
                np.sin(direction-bearing), np.cos(direction-bearing))))) if valid and moving else None
            row['initial_target_distance_reduction_3s'] = float(np.linalg.norm(obj-root[:2])-np.linalg.norm(obj-end)) if valid else None
        groups = {}
        for name, group in (('all_carry', rows), ('rear_carry', [r for r in rows if r['rear']]),
                            ('front_carry', [r for r in rows if not r['rear']])):
            groups[name] = diagnosis.stats(group)
            groups[name]['valid_at_3s'] = sum(r['valid_at_3s'] for r in group)
            for key in ('movement_vs_target_deg_3s', 'initial_target_distance_reduction_3s'):
                values = [r[key] for r in group if r[key] is not None]
                groups[name][key] = float(np.median(values)) if values else None
        summary[mode] = dict(success=int(ever.sum()), n=int(ever.size), falls=result['falls'],
                             skills=by_skill, carry=groups, ablation=proof)
        trials.extend(rows)
    lookup = {(r['model'], r['env'], r['slot']): r for r in trials}
    paired = {}
    baseline = [r for r in trials if r['model'] == 'none']
    for mode in ('self_only', 'bypass'):
        paired[mode] = {}
        for name, group in (('all_carry', baseline), ('rear_carry', [r for r in baseline if r['rear']])):
            others = [lookup[(mode, r['env'], r['slot'])] for r in group]
            paired[mode][name] = dict(
                original_success_to_failure=sum(r['success'] and not s['success'] for r, s in zip(group, others)),
                original_failure_to_success=sum(not r['success'] and s['success'] for r, s in zip(group, others)),
                new_no_approach=sum(r['success'] and s['stage']=='not_approached' for r, s in zip(group, others)))
    new_failures = [r for r in baseline if r['success'] and
                    lookup[('bypass', r['env'], r['slot'])]['stage'] == 'not_approached']
    case_fields = ('success', 'stage', 'rear', 'prefall', 'initial_angle_deg', 'final_distance',
                   'movement_vs_target_deg_3s', 'initial_target_distance_reduction_3s')
    cases = [dict(env=r['env'], actor=r['slot'], task=r['task'], modes={
        mode: {key: lookup[(mode, r['env'], r['slot'])][key] for key in case_fields}
        for mode in summary}) for r in new_failures]
    with (OUT/'carry_trials.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader(); writer.writerows(trials)
    (OUT/'verification.json').write_text(json.dumps(checks, indent=2))
    (OUT/'summary.json').write_text(json.dumps(dict(modes=summary, paired=paired,
        original_success_bypass_no_approach_cases=cases), indent=2))
    lines = ['# IND DCT inference ablation pilot', '',
        '2H, 64 environments, seed1, 900 steps. Shared canonical scene; unchanged checkpoint, SCT encoder, action head and normalization.', '',
        '| Metric | Original | Self-only DCT | DCT bypass |', '| --- | --- | --- | --- |']
    def line(name, values):
        lines.append('| '+name+' | '+' | '.join(str(v) for v in values)+' |')
    def ratio(g):
        return f"{g['success']}/{g['n']} ({100*g['success']/g['n']:.2f}%)" if g['n'] else '0/0'
    values = list(summary.values())
    line('All SR', [ratio(v) for v in values])
    for skill in ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH'):
        line(skill+' SR', [ratio(v['skills'][skill]) for v in values])
    for group in ('rear_carry', 'front_carry'):
        line(group+' SR', [ratio(v['carry'][group]) for v in values])
        line(group+' approach within 1m', [f"{v['carry'][group]['approached']}/{v['carry'][group]['n']}" for v in values])
        line(group+' no-approach failures', [v['carry'][group]['stages']['not_approached'] for v in values])
    line('All falls', [v['falls'] for v in values])
    line('Rear carry: median movement vs target at 3s (degrees)',
         [round(v['carry']['rear_carry']['movement_vs_target_deg_3s'], 2) for v in values])
    line('Rear carry: median initial-target distance reduction at 3s (m)',
         [round(v['carry']['rear_carry']['initial_target_distance_reduction_3s'], 2) for v in values])
    lines += ['', '## 짝지은 접근 실패', '',
        '원본 성공 → bypass 접근 전 실패로 바뀐 모든 사례다. 세 사례 모두 뒤쪽 대상이며, 원본과 self-only는 성공했고 bypass는 낙상 없이 대상에서 멀어졌다.', '',
        '| env / actor (0-based) | Skill | Initial angle | Original movement vs target at 3s | Bypass movement vs target at 3s | Bypass final object distance |',
        '| --- | --- | --- | --- | --- | --- |']
    for case in cases:
        original, bypass = case['modes']['none'], case['modes']['bypass']
        lines.append(f"| {case['env']} / {case['actor']} | {case['task']} | "
            f"{original['initial_angle_deg']:.1f}° | {original['movement_vs_target_deg_3s']:.1f}° | "
            f"{bypass['movement_vs_target_deg_3s']:.1f}° | {bypass['final_distance']:.2f}m |")
    lines += ['', '이번 표본에서 타인 attention 차단은 뒤쪽 운반의 성공/실패 구성을 바꾸지 않았다. 반면 DCT 특징 변환 우회는 뒤쪽 접근 실패를 추가로 만들었다. 따라서 현재 학습된 정책에서 자기 특징을 변환하는 경로가 목표 방향으로 이동하는 데 기여한다는 해석을 지지한다. Bypass는 CLIMB/PUSH와 전체 낙상도 악화시켰으므로, 뒤돌기만 담당하는 모듈이라고 해석하지 않는다.']
    lines += ['', 'Self-only masks every off-diagonal human attention edge in actor DCT. Bypass replaces actor DCT output with its SCT input while retaining the trained Ours action head. Original DCT computation still runs in bypass to preserve random-number consumption.', '',
        'A bypass performance drop establishes dependence of this trained policy on DCT features; it cannot alone distinguish task-specific steering from an action-head input distribution mismatch. This pilot is a single evaluation seed, not a retrained architecture ablation.', '',
        'Carry failure phases use the existing analysis at `output_etc/check/ind_approach_diagnosis_v4_2000/analyze.py`. Approach means root-to-payload XY distance <=1m before success/fall; success uses official region scoring. Rear means initial heading-to-payload bearing >90 degrees. Three-second directional statistics exclude agents invalid by that time; see per-trial CSV for denominators.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()
