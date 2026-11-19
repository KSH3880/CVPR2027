import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from common import OUT as BASE, SETTINGS, DELAYS, read, write, sha
from run_tokenhsi import OUT, CHECKPOINT, sources
import analyze


def report(rows):
    labels=dict(tokenhsi='TokenHSI40.5k',sct='SCT26k',fsm='SCT+FSM26k',coopft='Coop.FT7k',ours='Ours V4 2k')
    lines=['# TokenHSI 추가 지연 평가','',
        'GPU6·64환경·seed1·35초·지연0/2/4초. TokenHSI6조건을 새로 평가하고 기존4모델24조건을 재사용했다. '
        '초기 장면·속도0·reset 몸체 관측·정지 방식·성공 조건은 같다. '
        'TokenHSI는 원래397D 자기 상태/담당 과제 입력을 사용하며 타인·distractor token을 직접 관측하지 않는다.','']
    for setting in SETTINGS:
        enabling=setting.startswith('enable')
        tail='AT SR(%) | 준비율(%) | 점유율(%) | Post-AT(s)' if enabling else 't_coop(%) | 합류(s) | 합류 전 변위(m) | 합류/64'
        lines += ['## '+setting,'',f'| 모델 | 지연(s) | 성공/64 | SR35s(%) | SR30s(%) | 성공 완료(s) | {tail} |',
                  '|---|'+'---:|'*9]
        for model,label in labels.items():
            for delay in DELAYS:
                row=next(r for r in rows if (r['setting'],r['model'],r['delay'])==(setting,model,delay))
                w=row['windows']['35']
                keys=('at_sr_percent','prep_percent','occupancy_percent','post_at_s') if enabling else (
                    'tcoop_percent','late_arrival_s','box_displacement_before_late_m','late_arrivals')
                values=[w['group_sr_percent'],row['windows']['30']['group_sr_percent'],w['completion_success_s'],*[w[k] for k in keys]]
                lines.append(f'| {label} | {delay} | {w["successes"]} | '+' | '.join('N/A' if v is None else f'{v:.2f}' for v in values)+' |')
        lines.append('')
    lines += ['준비율·점유율은 AT 성공 표본, Post-AT·완료 시간은 Group 성공 표본의 조건부 평균이다. '
        '점유율은 root 반경0.35m/초기 yaw footprint+margin0.10m 근사다. '
        '합류는 지연 대상의 최초 유효 양손 표면거리12cm 미만이며, 합류 시점/이전 변위는 합류한 표본만 평균낸다. '
        't_coop은 기존20cm 시작·전체 기록 접촉 누적·NaN/상한 clipping 규약이다. '
        '단일 seed이며 성공 표본 선택 및 입력 구조 차이를 함께 해석해야 한다.','',
        '[전체 통계·시간창](summary.json) · [CSV](summary.csv) · [검증](completion_verification.json) · [그래프](behavior.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,model in enumerate(('tokenhsi','ours')):
        source=OUT if model=='tokenhsi' else BASE
        for row,setting,key,label in [(0,'enable_sit','follower_goal_distance_m','Follower-target distance (m)'),
                                      (1,'coupled_3','box_displacement_m','Box displacement (m)')]:
            for d in DELAYS:
                with np.load(source/f'analysis/{setting}_{model}_d{d}_curves.npz') as z:
                    values=z[key]
                axes[row,col].plot(np.arange(1,len(values)+1)/30,values.mean(1),label=f'{d}s')
            axes[row,col].set(title=model+' / '+setting,xlabel='Time (s)',ylabel=label)
            axes[row,col].legend()
    fig.tight_layout()
    fig.savefig(OUT/'behavior.png',dpi=150)
    fig.savefig(OUT/'behavior.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args()
    folder='development' if args.development else 'rollouts'
    delays=(4,) if args.development else DELAYS
    rows=[]
    hashes={}
    (OUT/'analysis').mkdir(exist_ok=True)
    for setting,people in SETTINGS.items():
        paths=[OUT/f'{folder}/{setting}/tokenhsi_d{d}' for d in delays]
        analyze.verify_initial([BASE/f'rollouts/{setting}/ours_d0',*paths])
        first=None
        for path in paths:
            native=torch.load(path/'tokenhsi_first_input.pt',map_location='cpu',weights_only=False)
            scene=torch.load(path/'first_policy_input.pt',map_location='cpu',weights_only=False)
            if not torch.equal(native['scene'],scene) or native['raw'].shape!=(64*people,397):
                raise ValueError('Actual native input mismatch')
            if first is None:
                first=native
            for key,value in native.items():
                if not torch.isfinite(value).all() or not torch.equal(value,first[key]):
                    raise ValueError('Native input/first action differs across delays: '+key)
            row,trials,curves=analyze.score(path)
            proof=read(path/'policy_verification.json')
            if proof['epoch']!=40500 or proof['action_calls']!=row['steps'] or proof['other_policy_actions']!=0:
                raise ValueError('Native policy execution mismatch')
            if not proof['weights_unchanged'] or not proof['normalization_unchanged']:
                raise ValueError('Native policy changed')
            if row['steps']!=(135 if args.development else 1050) or row['trials']!=64:
                raise ValueError('Unexpected horizon/cohort')
            if not read(path/'result.json')['protocol_limitations']:
                raise ValueError('Missing397D input limitation')
            rows.append(row)
            for name in ('initial.pt','trace.npz','freeze_trace.npz','freeze.json','table2_trials.npz',
                         'table2.json','policy_verification.json','cooperation.npz','tokenhsi_first_input.pt'):
                if (path/name).exists():
                    hashes[str(path/name)]=sha(path/name)
            if not args.development:
                name=setting+'_'+path.name
                np.savez_compressed(OUT/f'analysis/{name}_trials.npz',**trials)
                np.savez_compressed(OUT/f'analysis/{name}_curves.npz',**curves)
    if sources()!=read(OUT/'manifest.json')['source_sha256']:
        raise ValueError('Source/checkpoint/reference/baseline changed')
    proof=dict(completed=True,runs=len(rows),initial_states_match_original=True,shared_scene_inputs_exact=True,
        native397d_inputs_and_first_actions_equal_across_delays=True,delay_frames_exact=True,
        cpu_group_events_exact=True,weights_normalization_unchanged=True,sources_unchanged=True,
        source_sha256=hashes,checkpoint=str(CHECKPOINT))
    if args.development:
        write(OUT/'development_verification.json',proof)
    else:
        combined=read(BASE/'summary.json')['runs']+rows
        write(OUT/'summary.json',dict(runs=combined,new_runs=6,reused_runs=24))
        flat=[dict(setting=r['setting'],model=r['model'],delay_seconds=r['delay'],window_seconds=s,**w)
              for r in combined for s,w in r['windows'].items()]
        with (OUT/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
            writer.writeheader();writer.writerows(flat)
        report(combined)
        plot()
        write(OUT/'completion_verification.json',proof)
    print({k:v for k,v in proof.items() if k!='source_sha256'},flush=True)


if __name__=='__main__':
    main()
