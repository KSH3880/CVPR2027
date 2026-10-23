"""Summarize completed probes without treating proxy motion as policy success."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(rows, proxy):
    if proxy:
        rows=[r for r in rows if r['cap_over_mg']>0]
    return dict(cases=len(rows),
        passed_2deg=sum(r['stable_2deg_move_50cm' if proxy else 'stable_2deg_move_10cm'] for r in rows),
        max_tilt_deg=max(r['max_tilt_deg'] for r in rows),
        min_forward_m=min(r['final_forward_m'] for r in rows),
        max_forward_m=max(r['final_forward_m'] for r in rows),
        **({} if proxy else dict(valid_supported_hand_push=sum(r['valid_supported_hand_push'] for r in rows),
            initial_height_within_3cm=sum(r['initial_midpoint_z_error_m']<=.03 for r in rows),
            root_above_65cm_entire_run=sum(r['min_root_height_m']>=.65 for r in rows))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();base=args.output
    names=['contacts','contacts_high','contacts_dense_z065','humanoid','humanoid_z065']
    reports={name:json.loads((base/name/'audit.json').read_text()) for name in names}
    summary={name:summarize(report['cases'],name.startswith('contacts')) for name,report in reports.items()}
    # Independently reconstruct tilt/movement from saved physical traces.
    validation={}
    for name in names:
        rows=reports[name]['cases'];checks=0
        if name.startswith('contacts'):
            with np.load(base/name/'trace.npz') as f:
                trace=f['trace'];sample_max=trace[:,:,0].max(0)
                assert np.all(sample_max<=np.array([r['max_tilt_deg'] for r in rows])+1e-4)
                if name=='contacts_dense_z065':
                    np.testing.assert_allclose(sample_max,[r['max_tilt_deg'] for r in rows],atol=1e-4)
                    np.testing.assert_allclose(trace[-1,:,1],[r['final_forward_m'] for r in rows],atol=1e-5)
                    assert trace.shape[0]==300
                checks=len(rows)
        else:
            for path in sorted((base/name).glob('trace_h*_load*.npz')):
                setting=float(path.stem.split('_')[1][1:]);load=float(path.stem.split('_')[2][4:])
                group=[r for r in rows if abs(r.get('height_setting',r['height_fraction'])-setting)<1e-5 and r['load_command_over_mg']==load]
                with np.load(path) as f:
                    trace=f['trace'];q=trace[:,:,3:7];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
                    x,y,z,w=np.moveaxis(q,-1,0)
                    tilt=np.degrees(np.arctan2(np.hypot(2*(x*z+w*y),2*(y*z-w*x)),1-2*(x*x+y*y)))
                    assert len(group)==trace.shape[1] and trace.shape[0]==300
                    assert np.all(tilt.max(0)<=np.array([r['max_tilt_deg'] for r in group])+1e-3)
                    np.testing.assert_allclose(trace[-1,:,1]-f['start_box_xyz'][:,1],[r['final_forward_m'] for r in group],atol=1e-4)
                    checks+=len(group)
        assert all(abs(r['actual_mass_kg']-r['mass'])<.001 for r in rows)
        validation[name]=dict(trace_cases_checked=checks,mass_check=True,audit_sha256=hashlib.sha256((base/name/'audit.json').read_bytes()).hexdigest())
    heights=[]
    for name in ('contacts','contacts_high'):
        rows=reports[name]['cases']
        for h in sorted(set(r['height_fraction'] for r in rows if r['cap_over_mg']>0)):
            heights.append(dict(height_fraction=h,**summarize([r for r in rows if r['height_fraction']==h and r['cap_over_mg']>0],True)))
    masses=[dict(mass=m,**summarize([r for r in reports['contacts']['cases'] if r['mass']==m],True)) for m in (10,15,20,25,30)]
    result=dict(summary=summary,proxy_height_comparison=heights,proxy_mass_comparison=masses,validation=validation,
        recommendation=dict(status='provisional physics starting point; humanoid stable pushing unvalidated',mass_kg=20,
            point='rear-face centerline, mean of both hands at 0.65m above the initial floor',
            box_xyz_m=[1.2,1.5],box_friction=.6,floor_friction=1.,tilt_priority_deg=2,
            reason='20kg retains previous baseline; proxy tests do not identify an optimal mass. Avoid high points; no tilt guarantee.'))
    (base/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (base/'proxy_height_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(heights[0]));writer.writeheader();writer.writerows(heights)
    lines=['# PUSH 직육면체·동일 질량 물리 검사','',
        '기울기 최소화가 우선이다. 판정은 yaw를 제외한 상자 수직축의 기울기를 전체 5초 동안 측정해 최대 2° 이하로 둔다. 0° 보장이나 학습 정책 성공률 검사가 아니다.','',
        '## 조건','',
        '- XYZ는 각각 독립으로 1.20/1.35/1.50m(27조합). 10/15/20/25/30kg를 크기와 독립적으로 적용한다. 같은 질량 후보 안에서는 모든 크기의 질량이 같다.',
        '- 상자 마찰0.6·바닥1.0, 중앙 COM, 실제 생성 질량 오차1g 미만 확인. GPU6, 60Hz·substeps2. 내부 substep 순간 최대각은 계측하지 않는다.',
        '- 손 목표는 양손의 중점이다. 양손 간격·각 손 높이가 같아야 한다는 조건은 부과하지 않는다.',
        '- 접촉체: 1kg 구체 두 개, 유한 Cartesian 구동(합산 수평 cap=mg 또는2mg). 대칭/75:25 힘배분/손 높이±0.1m·간격 변경을 시험한다. 상자에 직접 힘을 주지 않는다. 3초 전진명령0.9m 후2초 유지.',
        '- humanoid: 21개 PUSH AMP clip의 선택 초기자세 각1개 + 전신 PD/중점 Jacobian 제어. 학습된 정책은 사용하지 않는다. 다른 상자·두 번째 사람은 멀리 둔다. 모션 전체 재생이나 모든 RSI 후보 검증이 아니다.','',
        '## 접촉체 결과','',
        '| 손 중점 높이 / 상자 Z | 2° 이하 + 0.5m 이동 | 최대 기울기 |',
        '| --- | ---: | ---: |']
    for row in heights:
        lines.append(f"| {row['height_fraction']:.0%} | {row['passed_2deg']}/{row['cases']} | {row['max_tilt_deg']:.3f}° |")
    d=summary['contacts_dense_z065']
    lines+=['',f"고정 높이0.65m·20kg 추가 검사: XYZ각0.05m 간격의343조합×2힘한도×3양손조건 = {d['cases']}회 중 {d['passed_2deg']}회 통과. 최대각 {d['max_tilt_deg']:.4f}°, 실제 최종 전진 {d['min_forward_m']:.3f}~{d['max_forward_m']:.3f}m. 별도 무구동343대조 포함. 이 결과는 균형 제약 없는 접촉체 결과다.",'',
        '## 실제 humanoid 결과','',
        '| 검사 | 회수 | 유효한 손 밀기 | 유효 + 2° 이하 + 0.1m 이동 |',
        '| --- | ---: | ---: | ---: |']
    for name in ('humanoid','humanoid_z065'):
        r=summary[name];lines.append(f"| {name} | {r['cases']} | {r['valid_supported_hand_push']} | {r['passed_2deg']} |")
    lines+=['',
        '유효 조건: 초기 관절 한계 위반 없음·초기 중점 Z 오차≤3cm·전 구간 root높이≥0.65m·양손 접촉≥15tick·발 지지≥6tick·접촉 중 평균 중점 오차≤10cm·비손 신체의 상자 근접접촉 없음. 접촉힘은 net reaction과 기하로 판정하며 pair 전용 센서는 아니다. 몸통/발 충돌로 움직인 상자는 성공에서 제외한다. 임시 PD 제어의 균형 실패가 많아 이 결과로 학습 정책의 성공/실패를 단정할 수 없다.','',
        '## 권고와 한계','',
        '**시험 시작값은 20kg·뒷면 중앙선의 바닥 기준0.65m 지점**을 권고한다. 20kg는 기존값 유지이며 최적 질량으로 검증된 값이 아니다. 10~30kg 모두 낮은 접촉점의 접촉체 검사에 통과해 질량 우열을 결정할 근거가 부족하다. 실제 humanoid가 균형을 유지하며 잘 미는 설정은 아직 검증되지 않았다.',
        '중앙 COM에 수평으로 천천히 미는 단순 모델에서 Fslide=μmg, Ftip=mg·Y/(2h)이므로 두 임계힘의 비율은 질량과 무관하다. 무게를 늘리는 것만으로 전도를 막을 수 없고 필요한 추진력도 커진다. 고정h=0.65m·최소Y=1.2m일 때 명목비율은1.538이다. 충격·수직힘·비대칭 하중은 이 계산의 가정 밖이다.',
        '높이를 Z의55% 이상으로 올리면 일부 형상/힘 조건에서90° 이상 전도했다. 크기에 비례해 높이를 올리지 않는 편을 우선 검토한다. 유한 격자·5초·약0.8m 시험이므로 임의의 연속 크기·1~5m 장거리·임의 추진력·학습 중 기울기까지 보장하지 않는다.',
        '이번에는 분석 코드와 결과만 추가했다. 실제 학습 YAML·reward·RSI bank·원본 보관본·기존 학습은 변경하지 않았다. 점 기반 CONTACT와 가변 XYZ를 본학습에 적용한 것은 아니다.','',
        '## 산출물과 검증','',
        '- `summary.json`, `proxy_height_comparison.csv`, `stability.png`; 각 검사 폴더의 `audit.json`, `cases.csv`, `trace*.npz`.',
        '- 저장 trace에서 기울기·최종변위를 독립 재계산했다. 초기 접촉체 두 sweep의 trace는10Hz, dense와humanoid는60Hz; 최댓값 판정 자체는 모두60Hz. humanoid의 초기 settling1tick은 audit 최대각에 포함되나 trace에서는 빠져 있어 trace최대≤audit최대를 확인했다.',
        '- `humanoid_quick*`는 제어기 개발 중 실패 probe이며 본표에서 제외하고 원자료는 보존했다. AMP21clip hold sweep과 단순 접촉체 결과를 합쳐 정책 성공률로 표현하지 않는다.','']
    (base/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot([r['height_fraction'] for r in heights],[r['max_tilt_deg'] for r in heights],'o-')
    axes[0].axhline(2,color='r',linestyle='--',label='2 degree target')
    axes[0].set(xlabel='Command midpoint height / box Z',ylabel='Worst box tilt (degrees)',title='Contact proxies: high points can topple')
    axes[0].legend()
    axes[1].bar([str(r['mass']) for r in masses],[r['max_tilt_deg'] for r in masses])
    axes[1].set(xlabel='Box mass (kg)',ylabel='Worst box tilt (degrees)',title='Low points (30-50% Z): proxies only')
    fig.savefig(base/'stability.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
