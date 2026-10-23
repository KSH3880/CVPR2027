"""Summarize the Y1.8m high-contact probe; validate saved contact traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
args=parser.parse_args();out=args.output
contacts=[]
for folder in ['contacts','contacts_boundary']:
    report=json.loads((out/folder/'audit.json').read_text())
    rows=report['cases'];trace=np.load(out/folder/'trace.npz')['trace']
    assert trace.shape[1]==len(rows)
    np.testing.assert_allclose(trace[:,:,0].max(0),[r['max_tilt_deg'] for r in rows],atol=1e-5,rtol=1e-6)
    np.testing.assert_allclose(trace[-1,:,1],[r['final_forward_m'] for r in rows],atol=1e-5,rtol=1e-6)
    assert all(abs(r['actual_mass_kg']-r['mass'])<.001 for r in rows)
    contacts.extend(rows)
human=json.loads((out/'humanoid/audit.json').read_text())['cases']
assert all(r['initial_collider_clear'] for r in human)
masses={r['mass'] for r in contacts+human};assert len(masses)==1
mass=next(iter(masses))
assert all(abs(r['actual_mass_kg']-mass)<.001 for r in human)

def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        shape='Y1.8' if abs(row['size'][1]-1.8)<1e-6 else 'X=Y'
        height='AMP' if row.get('height_mode')=='amp_reference' else f"{row['command_midpoint_z_m']:.2f}"
        groups[shape,height].append(row)
    result=[]
    for (shape,height),items in sorted(groups.items()):
        result.append(dict(shape=shape,height_m=height,n=len(items),max_tilt_deg=max(r['max_tilt_deg'] for r in items),
            tilt_over_45_count=sum(r['max_tilt_deg']>45 for r in items),
            final_forward_min_m=min(r['final_forward_m'] for r in items),final_forward_max_m=max(r['final_forward_m'] for r in items),
            stable_2deg_move_count=sum(r.get('stable_2deg_move_50cm',r.get('stable_2deg_move_10cm',False)) for r in items),
            stable_5deg_move_count=sum(r.get('stable_5deg_move_50cm',r.get('stable_5deg_move_10cm',False)) for r in items),
            supported_hand_push_count=sum(r.get('valid_supported_hand_push',False) for r in items)))
    return result
c=aggregate([r for r in contacts if r['cap_over_mg']]);h=aggregate(human)
summary=dict(contact_trace_verified=True,proxy_cases=len(contacts),humanoid_cases=len(human),contacts=c,humanoid=h,
    limitations='Proxy results isolate box/contact physics. Humanoid uses experimental PD/IK and selected AMP initial frames, not a trained policy. Unsupported or nonhand-contact cases cannot certify hand pushing. Finite5s conditions do not guarantee all learned actions.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Y1.8m 상자 높은 손 위치 전도 검사','',
f'{mass:g}kg·Z1.3m·X1.2~1.5m/5cm, Y1.8m와 X=Y 대조. 상자 마찰0.6·바닥1.0. 각 조건5초.',
'접촉체: 유한힘 양손 구체·하중한도1/2mg·대칭/힘불균형/높이차 세 조건, 0.9m 이동 명령. 상자에 직접 외력을 가하지 않았다.','',
'## 접촉체 결과','', '| 상자 | 손 높이(m) | 조건 수 | 최대 기울기(°) | >45° | ≤5°·≥0.5m 이동 | 최종 전진거리 범위(m) |',
'| --- | --- | ---: | ---: | ---: | ---: | --- |']
for r in c:
    lines.append(f"| {r['shape']} | {r['height_m']} | {r['n']} | {r['max_tilt_deg']:.3f} | {r['tilt_over_45_count']} | {r['stable_5deg_move_count']} | {r['final_forward_min_m']:.3f}~{r['final_forward_max_m']:.3f} |")
lines+=['','## 실제 humanoid 참고 결과','',
'21 AMP 선택초기자세·PD/중점IK·하중1/2mg·고정0.65/0.95/1.10/1.20m 및 원본 AMP 높이. 초기 collider/바닥 무관통 보정. 학습된 정책 평가가 아니다.',
'유효 손 밀기는 관절/높이/균형/양손접촉/발지지 및 비손접촉 배제를 모두 만족해야 한다. 전도 전체에는 유효하지 않은 몸 충돌/균형실패도 포함될 수 있다.','',
'| 상자 | 손 높이(m) | 조건 수 | 유효 손 밀기 | 그중 ≤5°·≥0.1m 이동 | 전체 >45° |',
'| --- | --- | ---: | ---: | ---: | ---: |']
for r in h:
    lines.append(f"| {r['shape']} | {r['height_m']} | {r['n']} | {r['supported_hand_push_count']} | {r['stable_5deg_move_count']} | {r['tilt_over_45_count']} |")
lines+=['','높은점이 항상 안전하다는 보장은 아니다. 접촉체와 실제 humanoid/정책의 결과를 구분한다. raw cases·trace와 검증 집계는 각 하위 폴더 및 summary.json을 따른다.','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(out/'REPORT.md')
