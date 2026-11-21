import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from common import ROOT, OUT, MODELS, SETTINGS, DELAYS, read, write, sha
sys.path[:0] = [str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import stage2_xy100
from utils.evaluation_protocol import spatial_success
from coupled_cooperation_metrics import counters, ratios, surface_contact
from table2_metrics import surface_distance
from analyze_coordination_behavior import behavior_arrays

stage2_xy100.install()
torch.set_num_threads(1)


def first(mask):
    return np.where(mask.any(0), mask.argmax(0)+1, -1)


def mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def score(path):
    initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
    initial = {k:v.numpy() for k,v in initial.items()}
    with np.load(path/'trace.npz') as z:
        a = {k:z[k] for k in ('roots','boxes','valid','fall_step')}
    with np.load(path/'freeze_trace.npz') as z:
        frozen = {k:z[k] for k in z.files}
    info, freeze = read(path/'table2.json'), read(path/'freeze.json')
    steps, n, m = a['valid'].shape
    ids = np.arange(n)
    selected = np.array(freeze['selected'])
    expected = (np.arange(steps)[:,None,None] < freeze['delay_steps']) & a['valid']
    expected &= np.arange(m)[None,None,:] == selected[None,:,None]
    np.testing.assert_array_equal(frozen['delayed'], expected)
    base = np.broadcast_to(initial['roots'][None,...,:3], a['roots'].shape)
    np.testing.assert_array_equal(a['roots'][expected], base[expected])
    if freeze['max_frozen_root_error'] or freeze['max_frozen_dof_error']:
        raise ValueError('Freeze verification failed')
    if steps not in (135,1050) and not ((a['fall_step'] > 0) & (a['fall_step'] <= steps)).all():
        raise ValueError('Unexpected early stop')
    current = []
    for start in range(0,steps,30):
        stop=min(start+30,steps)
        width=stop-start
        value=spatial_success(torch.from_numpy(initial['packet']).repeat(width,1,1),
            torch.from_numpy(a['roots'][start:stop]).reshape(width*n,m,3),
            torch.from_numpy(a['boxes'][start:stop]).reshape(width*n,-1,7),
            torch.from_numpy(initial['sizes']).repeat(width,1,1),
            torch.from_numpy(initial['goals']).repeat(width,1,1))[...,1]
        current.append(value.reshape(width,n,m).numpy())
    current=np.concatenate(current)&a['valid']
    enabling=info['setting'].startswith('enable')
    if enabling:
        group=current.all(-1)
        follower=(initial['predecessor']>=0).argmax(-1)
        leader=initial['predecessor'][ids,follower]
        at=first(current[:,ids,leader])
    else:
        with np.load(path/'cooperation.npz') as z:
            pose,sizes,hands=(z[k] for k in ('physical_boxes','physical_sizes','hands'))
        goal=initial['shared_center_goals'][:,0].astype(float)
        group=(np.linalg.norm(pose[...,:3].astype(float)-goal,axis=-1)<=.2)&a['valid'].any(-1)
        contacts=surface_contact(hands,pose,sizes)
        distances=surface_distance(torch.from_numpy(hands).reshape(steps*n,m,2,3).double(),
            torch.from_numpy(pose).reshape(steps*n,7).double(),
            torch.from_numpy(np.broadcast_to(sizes,(steps,*sizes.shape)).copy()).reshape(steps*n,3).double())
        near=(distances.reshape(steps,n,m,2).numpy()<.12).all(-1)&a['valid']
        late_first=first(near[:,ids,selected])
    first_group=first(group)
    with np.load(path/'table2_trials.npz') as z:
        np.testing.assert_array_equal(first_group,z['first_success'])
    if int((first_group>0).sum())!=info['successes']:
        raise ValueError('CPU group SR differs')
    result=dict(model=info['model'],setting=info['setting'],delay=info['delay_seconds'],steps=steps,
        source=str(path),trials=n,freeze_frames_exact=True,group_events_exact=True,windows={})
    trials=dict(first_group=first_group,selected=selected)
    if enabling:
        box=initial['packet'][ids,leader,3]-m
        q=initial['boxes'][ids,box,3:7].astype(float)
        yaw=np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
        positions=np.concatenate((initial['roots'][ids,follower,:2][None],a['roots'][:,ids,follower,:2])).astype(float)
        goal=initial['goals'][ids,leader,:2].astype(float)
        half=initial['sizes'][ids,box,:2].astype(float)/2
        behavior=behavior_arrays(positions,goal,half,yaw,at,first_group)
        trials.update(first_at=at,**{k:behavior[k] for k in ('prep_percent','occupancy_percent','post_at_s')})
        distance=np.linalg.norm(positions-goal,axis=-1)
        speed=np.linalg.norm(np.diff(positions,axis=0),axis=-1)*30
        curves=dict(follower_goal_distance_m=distance[1:],follower_speed_m_s=speed)
    else:
        displacement=np.linalg.norm(pose[...,:2].astype(float)-initial['shared_centers'][None,:,0,:2],axis=-1)
        before=(np.arange(steps)[:,None] < (late_first-1)[None]) & (late_first[None]>0)
        max_before=np.where(late_first>0,np.where(before,displacement,0).max(0),np.nan)
        trials.update(late_first=late_first,box_displacement_before_late_m=max_before)
        curves=dict(box_displacement_m=displacement,late_hand_distance_m=distances.reshape(steps,n,m,2).numpy()[:,ids,selected].max(-1),
            all_hands_near=near.all(-1),box_goal_distance_m=np.linalg.norm(pose[...,:3].astype(float)-goal,axis=-1))
    for seconds in sorted(set((30,35,30+info['delay_seconds']))):
        length=min(steps,seconds*30)
        success=(first_group>0)&(first_group<=length)
        row=dict(successes=int(success.sum()),group_sr_percent=100*float(success.mean()),
            completion_success_s=mean(first_group[success]/30),group_failures=int((~success).sum()))
        if enabling:
            at_ok=(at>0)&(at<=length)
            row.update(at_successes=int(at_ok.sum()),at_sr_percent=100*float(at_ok.mean()),
                at_without_group=int((at_ok&~success).sum()),post_at_s=mean((first_group[success]-at[success])/30),
                prep_percent=mean(behavior['prep_percent'][at_ok]),occupancy_percent=mean(behavior['occupancy_percent'][at_ok]))
        else:
            clipped_first=np.where(success,first_group,-1)
            start,begin,end=counters(pose[:length],initial['shared_centers'][:,0],clipped_first,.2)
            raw,ratio=ratios(contacts[:length].sum(0),begin,end)
            arrived=(late_first>0)&(late_first<=length)
            row.update(tcoop_percent=100*mean(ratio) if np.isfinite(ratio).any() else None,
                tcoop_nan=int(np.isnan(ratio).sum()),tcoop_clipped=int((raw>1).sum()),
                late_arrivals=int(arrived.sum()),late_arrival_s=mean(late_first[arrived]/30),
                box_displacement_before_late_m=mean(max_before[arrived]))
            trials[f'tcoop_{seconds}s']=ratio
        result['windows'][str(seconds)]=row
    return result,trials,curves


def verify_initial(paths):
    reference=None
    for path in paths:
        initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
        obs=torch.load(path/'first_policy_input.pt',map_location='cpu',weights_only=False)
        if reference is None:
            reference=(initial,obs)
        for k,v in initial.items():
            if not torch.equal(reference[0][k],v):
                raise ValueError('Initial state mismatch '+str(path)+' '+k)
        if not torch.equal(reference[1],obs):
            raise ValueError('Shared first observation differs '+str(path))


def paired_changes(rows, episodes):
    changes=[]
    for setting in SETTINGS:
        for model in MODELS:
            base=episodes[(setting,model,0)]
            for delay in DELAYS[1:]:
                perturbed=episodes[(setting,model,delay)]
                for horizon in (30,35):
                    a=(base['first_group']>0)&(base['first_group']<=horizon*30)
                    b=(perturbed['first_group']>0)&(perturbed['first_group']<=horizon*30)
                    common=a&b
                    changes.append(dict(setting=setting,model=model,delay=delay,horizon=horizon,
                        sr_change_pp=100*float(b.mean()-a.mean()),lost=int((a&~b).sum()),
                        gained=int((~a&b).sum()),common_successes=int(common.sum()),
                        completion_shift_common_success_s=mean((perturbed['first_group'][common]-base['first_group'][common])/30)))
    return changes


def report(rows):
    labels=dict(sct='SCT26k',fsm='SCT+FSM26k',coopft='Coop. FT7k',ours='Ours V4 2k')
    lines=['# Delay response pilot','',
        'GPU6·64환경·seed1·35초·지연0/2/4초. 4모델×Enabling SIT/3인 Coupled×3지연=24회. '
        '모든 사람의 초기 속도를0으로 통일한 새 공통 장면이다. 단일 seed 파일럿이며 평균±seed std를 제시하지 않는다.','']
    for setting in SETTINGS:
        lines += ['## '+setting,'', '| 모델 | 지연(s) | SR30s (%) | SR35s (%) | 성공/64 (35s) | 성공 완료(s) | '+
                  ('AT SR(%) | Post-AT(s) | 준비율(%) | 점유율(%) |' if setting.startswith('enable') else 't_coop35s (%) | 늦은 팀원 근접(s) | 근접 전 상자변위(m) |'),
                  '|---|'+'---:|'*(9 if setting.startswith('enable') else 8)]
        for model in MODELS:
            for delay in DELAYS:
                r=next((r for r in rows if (r['setting'],r['model'],r['delay'])==(setting,model,delay)),None)
                if r is None: continue
                w=r['windows']['35']
                keys=('at_sr_percent','post_at_s','prep_percent','occupancy_percent') if setting.startswith('enable') else (
                    'tcoop_percent','late_arrival_s','box_displacement_before_late_m')
                cells=[r['windows']['30']['group_sr_percent'],w['group_sr_percent'],w['successes'],w['completion_success_s'],*[w[k] for k in keys]]
                lines.append('| '+labels[model]+f' | {delay} | '+' | '.join('N/A' if x is None else f'{x:.2f}' for x in cells)+' |')
    lines += ['', '## 시간 여유 보정', '',
        '| 장면 | 모델 | 지연(s) | 기준 30s SR(%) | 지연 조건 30+지연s SR(%) | 차이(%p) |',
        '|---|---|---:|---:|---:|---:|']
    for setting in SETTINGS:
        for model in MODELS:
            base=next(r for r in rows if (r['setting'],r['model'],r['delay'])==(setting,model,0))['windows']['30']['group_sr_percent']
            for delay in DELAYS[1:]:
                r=next(r for r in rows if (r['setting'],r['model'],r['delay'])==(setting,model,delay))
                value=r['windows'][str(30+delay)]['group_sr_percent']
                lines.append(f'| {setting} | {labels[model]} | {delay} | {base:.2f} | {value:.2f} | {value-base:+.2f} |')
    lines += ['', 'Enabling: AT 직전 준비율/사전 점유율은 AT 성공 표본, Post-AT는 Group 성공 표본. '
        '점유는 root 반경0.35m·초기 yaw 상자 영역+margin0.10m의 근사다.',
        'Coupled: t_coop은 기존20cm 시작·손 표면12cm·기록 전체 누적·NaN/clip 규약을 각 시간창에 적용한다. '
        '늦은 팀원 근접은 양손 모두12cm 미만인 최초 유효 프레임이며 실제 접촉력 판정이 아니다. '
        '늦은 팀원이 끝까지 근접하지 않은 건수는 JSON에 저장하고 합류 전 변위 평균에서 제외한다.',
        '정지는 FSM/controller.py의 substep 상태 투영을 사용하며 실제 kinematic actor가 아니다. '
        'FSM 비교군은 여기에 원래 관계별 대기/일회 해제 규칙을 함께 적용한다. '
        '먼저 움직이거나 오래 기다리는 것 자체를 성공/실패로 판정하지 않는다.', '',
        '[전체 통계](summary.json) · [동일 장면 성공 변화](paired_changes.json) · [검증](completion_verification.json) · [행동 곡선](behavior.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
    for col,model in enumerate(MODELS):
        for setting,ax,key in [('enable_sit',axes[0,col],'follower_goal_distance_m'),
                              ('coupled_3',axes[1,col],'box_displacement_m')]:
            for delay in DELAYS:
                with np.load(OUT/f'analysis/{setting}_{model}_d{delay}_curves.npz') as z:
                    values=z[key]
                ax.plot(np.arange(1,len(values)+1)/30,values.mean(1),label=f'{delay}s')
            ax.set_title(model+' / '+setting)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Follower-target distance (m)' if setting.startswith('enable') else 'Box displacement (m)')
            ax.legend()
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
    episodes={}
    (OUT/'analysis').mkdir(exist_ok=True)
    for setting in SETTINGS:
        paths=[OUT/f'{folder}/{setting}/{model}_d{d}' for d in delays for model in MODELS]
        verify_initial(paths)
        for path in paths:
            row,trials,curves=score(path)
            rows.append(row)
            episodes[(setting,row['model'],row['delay'])]=trials
            name=setting+'_'+path.name
            if not args.development:
                np.savez_compressed(OUT/f'analysis/{name}_trials.npz',**trials)
                np.savez_compressed(OUT/f'analysis/{name}_curves.npz',**curves)
    hashes={str(p):sha(p) for p in set(MODELS.values())}
    manifest=read(OUT/'manifest.json')
    for model,path in MODELS.items():
        if hashes[str(path)]!=manifest['checkpoints_sha256'][model]:
            raise ValueError('Checkpoint changed')
    proof=dict(completed=True,runs=len(rows),initial_states_exact=True,shared_first_observations_exact=True,
        delay_frames_exact=True,frozen_roots_exact=True,cpu_group_success_exact=True,checkpoints_unchanged=True)
    if args.development:
        write(OUT/'development_verification.json',proof)
    else:
        for path,digest in read(OUT/'formal_sources.json').items():
            if sha(Path(path)) != digest:
                raise ValueError('Formal source changed '+path)
        proof['formal_sources_unchanged'] = True
        write(OUT/'summary.json',dict(runs=rows))
        write(OUT/'paired_changes.json',paired_changes(rows,episodes))
        write(OUT/'completion_verification.json',proof)
        report(rows)
        plot(rows)
    print(proof,flush=True)


if __name__=='__main__':
    main()
