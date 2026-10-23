"""Summarize finite-grid contact, humanoid and wrench height experiments."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--contacts',type=Path,required=True);p.add_argument('--pd',type=Path,required=True)
p.add_argument('--forces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
contacts,pd,forces=[json.loads((x/'audit.json').read_text()) for x in (a.contacts,a.pd,a.forces)]
a.output.mkdir(parents=True,exist_ok=True)
sizes=sorted({r['side_m'] for r in contacts['cases']})

def intervals(values,step=.025):
    result=[]
    for x in sorted(values):
        if not result or x-result[-1][1]>step+1e-6:result.append([x,x])
        else:result[-1][1]=x
    return result

summary=[]
categories=[('horizontal_1to2mg',forces['profiles'][:3],2),
            ('horizontal_1to5mg',forces['profiles'][:3],5),
            ('vertical20_asymmetry_1to2mg',forces['profiles'],2),
            ('vertical20_asymmetry_1to3mg',forces['profiles'],3)]
global_sets={k:None for k,_,_ in categories};global_sets['physical_hand_contacts']=None
for s in sizes:
    cr=[r for r in contacts['cases'] if r['side_m']==s and r['drive_limit_over_mg']>=1]
    hs=sorted({r['command_z_over_side'] for r in cr})
    good=[h for h in hs if all(r['moved_10cm_max_tilt_le5'] for r in cr if r['command_z_over_side']==h)]
    pr=[r for r in pd['cases'] if r['side_m']==s]
    supported=[r for r in pr if r['supported_no_fall_success']]
    row=dict(side_m=s,mass_kg=20*(s/1.3)**3,
        physical_hand_contact_passing_z_over_H=good,
        physical_hand_contact_passing_z_m=[round(s*h,4) for h in good],
        humanoid_trials=len(pr),humanoid_moving_le5=sum(r['moved_10cm_max_tilt_le5'] for r in pr),
        humanoid_supported_no_fall=len(supported),humanoid_hands_only_proxy=sum(r['hands_only_proxy_success'] for r in pr))
    global_sets['physical_hand_contacts']=set(good) if global_sets['physical_hand_contacts'] is None else global_sets['physical_hand_contacts']&set(good)
    for name,profiles,maxforce in categories:
        fr=[r for r in forces['cases'] if r['side_m']==s and r['profile'] in profiles and 1<=r['force_over_mg']<=maxforce]
        good=[h for h in sorted({r['midpoint_z_over_side'] for r in fr}) if all(r['moved_10cm_max_tilt_le5'] for r in fr if r['midpoint_z_over_side']==h)]
        row[name+'_z_over_H']=good
        global_sets[name]=set(good) if global_sets[name] is None else global_sets[name]&set(good)
    # Example is illustrative, not a safe interval or causal hand-force estimate.
    examples=[r for r in supported if .35<=r['paired_contact_mean_midpoint_z_m']/s<=.55]
    row['humanoid_example']=min(examples,key=lambda r:r['max_tilt_deg']) if examples else (min(supported,key=lambda r:r['max_tilt_deg']) if supported else None)
    summary.append(row)
common={k:sorted(v) for k,v in global_sets.items()}
core=[r for r in contacts['cases'] if .45<=r['command_z_over_side']<=.5 and r['drive_limit_over_mg']>=1]
assert all(r['moved_10cm_max_tilt_le5'] for r in core)
core_summary=dict(trials=len(core),max_tilt_deg=max(r['max_tilt_deg'] for r in core),
    forward_m_range=[min(r['final_forward_m'] for r in core),max(r['final_forward_m'] for r in core)],
    actual_horizontal_contact_peak_N=max(r['peak_horizontal_contact_force_N'] for r in core),
    actual_contact_height_mean_error_max_m=max(abs(r['mean_contact_midpoint_z_m']-r['command_midpoint_z_m']) for r in core))
result=dict(sizes=summary,common_passing_grid_points_over_H=common,
    common_segments_over_H={k:intervals(v) for k,v in common.items()},physical_contact_core_045to050H=core_summary,
    sources=dict(contacts=str(a.contacts),humanoid=str(a.pd),known_wrenches=str(a.forces)),
    limitations='Passing discrete grid points under recorded finite duration/controller/forces do not establish a universal interval. Physical hand proxies are not a humanoid; humanoid nonhand-near-box proxies are conservative, not pair contacts. Hand midpoint alone cannot guarantee<=5degrees.')
(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 상자 크기별 PUSH 손 중점 높이 분석','',
    'GPU6에서 상자1.0~1.5m/0.05m간격·동일밀도20kg/1.3³m³·상자마찰0.6·바닥1.0을 시험했다. 학습/정책/RSI bank는 변경하지 않았다.', '',
    '**절대 안 넘어지는 높이는 확인할 수 없다.** 높이만 제한하면 힘 방향·크기·양손 분담·충격에 따라5°를 넘을 수 있다. 아래는 유한 격자/시간/제어조건의 결과이며 시험점 사이 연속구간 보장은 아니다.', '',
    '## 1. 두 물리 손 접촉체가 실제로 밀기','',
    f"총{contacts['num_cases']}조건, 실제 두 구 접촉체(반경5cm·각1kg)가0.3m/s로3초 전진한 뒤2초 목표위치를 유지했다. 상자 외력 없음. 제어기수평구동상한1/1.5/2/3/5mg와 3가지 양손분담(동일/75:25/높이차0.2H+75:25)을 시험했다. 각손 수직구동상한은 수평구동상한의20%이며 실제 수직 접촉력이 정해진 비율이라는 뜻은 아니다.", '',
    f"모든크기/구동상한/분담에서 통과한 시험점은 z/H={common['physical_hand_contacts']}. 0.45~0.50H의{core_summary['trials']}시험은 최대기울기{core_summary['max_tilt_deg']:.3f}°·실제이동{core_summary['forward_m_range'][0]:.3f}~{core_summary['forward_m_range'][1]:.3f}m였다. 평균실제높이 목표오차 최대{core_summary['actual_contact_height_mean_error_max_m']:.4f}m.", '',
    '| 정육면체 한 변/높이 | 질량 | 접촉체 통과 시험점 범위 | 우선 검토할 목표 높이0.45~0.50H |',
    '|---|---:|---:|---:|']
for r in summary:
    s=r['side_m'];zs=r['physical_hand_contact_passing_z_m']
    lines.append(f"| {s:.2f}m | {r['mass_kg']:.2f}kg | {min(zs):.3f}~{max(zs):.3f}m | {s*.45:.3f}~{s*.5:.3f}m |")
lines+=['','이 범위는 제어된 물리 접촉체의 결과다. 실제 humanoid의 팔 도달범위·발 지지·학습정책에 대한 보장은 아니다. 수평구동상한과 실제접촉력/충격peak도 다르다. `cases.csv`에 실제힘·실제높이를 기록했다.','',
    '## 2. 실제 humanoid PD 밀기','',
    f"{pd['num_cases']}조건(초기 모션4개·목표높이0.4~1.3m 중 상자 이내·수평load명령100/200/350/500N). 외부상자힘 없이 native 관절PD와 whole-body Jacobian 힘bias로 밀었다. 4초 전구간60Hz에서 상자최대기울기≤5°·최종전진≥10cm·양손접촉≥6tick 기준이다. 명령높이/명령힘과 실제접촉높이/힘은 다르다.", '',
    '| 크기 | 상자 이동≥10cm·≤5° | 지지·root저하 제외 | 손 외 접촉 의심도 제외 | 지지된 예시: 실제중점Z / 이동 / 최대각 |',
    '|---|---:|---:|---:|---|']
for r in summary:
    ex=r['humanoid_example'];example=f"{ex['paired_contact_mean_midpoint_z_m']:.3f}m / {ex['final_forward_m']:.3f}m / {ex['max_tilt_deg']:.3f}°" if ex else '없음'
    lines.append(f"| {r['side_m']:.2f}m | {r['humanoid_moving_le5']}/{r['humanoid_trials']} | {r['humanoid_supported_no_fall']} | {r['humanoid_hands_only_proxy']} | {example} |")
lines+=['','지지는 양발반력/골반높이>0.45m가6tick이상, root<0.35m인 실행은 제외했다. 이는 전구간 균형/자연스러운 보행 성공이 아니다. 손 외 몸체가 상자 근처에 있고 수평반력이 있으면 보수적으로 손 외 접촉 의심으로 분류했다. 실제 pair 센서가 아니므로 오탐 가능성이 있다. 손 외 접촉까지 엄격히 제외하면 모든 크기에서 순수 손밀기 범위를 확정하지 못했다. 해당 예시들은 가능 사례이며 범위·성공률 보장이 아니다.','',
    '## 3. 손 힘 크기·방향을 정확히 지정한 보조 실험','',
    f"{forces['num_cases']}조건에서 상자에 두 점의 알려진 합력/토크를3초 가하고2초 해제했다. 이는 실제 humanoid 접촉이 아닌 외력 실험이다. step/ramp/pulse 수평력과 높은손75%·높이차0.2H·수직성분±20%를 시험했다.", '',
    '| 알려진 힘 조건 | 모든 크기에 공통으로 통과한 z/H 시험점 구간 |','|---|---|']
labels=dict(horizontal_1to2mg='수평력1~2mg·동일 높이/동일 분담',horizontal_1to5mg='수평력1~5mg·동일 높이/동일 분담',vertical20_asymmetry_1to2mg='수직±20%·불균형 포함·수평력1~2mg',vertical20_asymmetry_1to3mg='수직±20%·불균형 포함·수평력1~3mg')
for key,label in labels.items():
    seg=intervals(common[key]);lines.append(f"| {label} | {seg if seg else '공통 통과 시험점 없음'} |")
lines+=['','따라서 접촉체의 넓은 높이범위를 임의의 손힘에 그대로 적용하면 안 된다. 예를 들어1.3m/20kg의2mg는392.4N이며3mg는588.6N이다. 현재 PD effort에는 실질적 유한상한이 없어 관절설정만으로 손힘이 이 범위 안에 있다고 보장할 수 없다.','',
    '## 적용 판단','',
    '높이 목표 후보는 우선0.45~0.50H(1.3m면0.585~0.650m)로 검토할 수 있다. 단, 이는 수평 위주로 제어한 접촉 시험의 후보이며 절대 안전구간은 아니다. 임의의 양손 높이차/힘분담·위로 드는 힘까지 허용하면서 중점만 이 점에 맞추는 보상은5° 제한을 보장하지 않는다. 실제정책의 힘방향/힘차이/손높이차와 상자기울기를 함께 제약·평가해야 한다. 현재 정책/보상/RSI는 수정하지 않았다.','',
    '## 검증과 재현','',
    '영힘 대조군의 이동<1mm·기울기<0.1°를 확인했다. 실제 접촉체 초기중점높이가 목표와1e-5m이내인지 검사한 수정실행만 사용했다. 초기 tensor동기화가 누락된 예비실행 `output/push_height_hand_contacts_20261018`은 INVALIDATED.json으로 제외했다.','',
    'physics tensor는1/60초마다 계측하고 내부substep2를 사용했다. 내부substep 순간peak는 직접 관측하지 못하므로 무한시간/임의외란/절대안전 보장이 아니다. 시험한상자는 중심COM 정육면체이며 다른가로세로높이/COM에는 재계산이 필요하다.','',
    '입력: '+str(a.contacts)+' / '+str(a.pd)+' / '+str(a.forces), '',
    '재현 명령은 `tokenhsi/scripts/multi_agent/analyze_push_height.sh`, 요약 JSON은 `summary.json`, 그림은 `height_limits.png`를 참조한다.']
(a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
for ax,mode in zip(axes,('contacts','wrenches')):
    hs=sorted({r['command_z_over_side'] for r in contacts['cases']}) if mode=='contacts' else sorted({r['midpoint_z_over_side'] for r in forces['cases']})
    grid=[]
    for h in hs:
        line=[]
        for s in sizes:
            if mode=='contacts':g=[r for r in contacts['cases'] if r['side_m']==s and r['command_z_over_side']==h and r['drive_limit_over_mg']>=1]
            else:g=[r for r in forces['cases'] if r['side_m']==s and r['midpoint_z_over_side']==h and 1<=r['force_over_mg']<=2]
            line.append(max(r['max_tilt_deg'] for r in g))
        grid.append(line)
    im=ax.imshow(np.array(grid),origin='lower',aspect='auto',extent=[.975,1.525,min(hs)-.0125,max(hs)+.0125],vmin=0,vmax=10,cmap='RdYlGn_r')
    ax.set_xlabel('Cube side / height H (m)');ax.set_ylabel('Command midpoint height z/H')
    ax.set_title('Physical hand contacts: all profiles' if mode=='contacts' else 'Known wrenches: vertical +/-20%, F=1..2mg')
    ax.axhline(.45,color='black',ls='--',lw=.8);ax.axhline(.5,color='black',ls='--',lw=.8)
fig.colorbar(im,ax=axes,label='Worst sampled tilt (deg), clipped at 10; criterion <=5')
fig.savefig(a.output/'height_limits.png',dpi=180)
print(json.dumps(dict(core=core_summary,common=result['common_segments_over_H']),indent=2))
