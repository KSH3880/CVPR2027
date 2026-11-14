"""Protocol v1 postprocessing of recorded LLM rollouts; preserves legacy files."""
from pathlib import Path
import argparse,json,sys,statistics,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'protocol_v1/checks'))
import reference_metrics as metrics
from verify import verify as common_verify

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def stats(v):
    return dict(mean=statistics.mean(v),std=statistics.stdev(v) if len(v)>1 else 0) if v and all(x is not None for x in v) else dict(mean=None,std=None)
def average(v):return float(np.mean(v)) if len(v) else None

def process(path):
    old=json.loads((path/'table2.json').read_text());tr=np.load(path/'table2_trials.npz');first=tr['first_success'];n=len(first)
    result=dict(protocol='cvpr2027-evaluation-v1.1.0',setting=old['setting'],trials=n,successes=int((first>=1).sum()),group_success_rate_percent=float((first>=1).mean()*100),completion_time_s=average(first[first>=1].astype(float)/30))
    if not old['setting'].startswith('coupled_'):
        save(path/'protocol_v1.json',result);return result
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
    c=np.load(path/'cooperation.npz');pose=c['physical_boxes'];sizes=c['physical_sizes'];hands=c['hands'];valid=np.load(path/'trace.npz')['valid'];T,B=pose.shape[:2]
    assert B==n
    all_counts=[];ratios=[];jerks=[];starts=[];aa=[];bb=[];raw=[]
    for i in range(B):
        expected=metrics.first_coupled(pose[:,i,:3],initial['shared_center_goals'][i,0].numpy(),valid[:,i])
        assert expected==first[i],('first success',i)
        s,a,b=metrics.start_counters(pose[:,i,:3],initial['shared_centers'][i,0].numpy(),int(first[i]))
        # Match float32 surface arithmetic; never add alive filtering to t_coop.
        delta=torch.from_numpy(hands[:,i].copy())-torch.from_numpy(pose[:,i,:3].copy())[:,None,None,:]
        q=torch.from_numpy(pose[:,i,3:7].copy());xyz=-q[:,:3,None].transpose(1,2)[:,None]
        uv=torch.cross(xyz.expand_as(delta),delta,dim=-1)
        local=delta+2*(q[:,None,None,3:4]*uv+torch.cross(xyz.expand_as(delta),uv,dim=-1))
        excess=local.abs()-torch.from_numpy(sizes[i])[None,None,None,:]/2
        distance=(excess.clamp_min(0).norm(dim=-1)+excess.amax(-1).clamp_max(0)).abs()
        count=int((distance<.12).all(-1).all(-1).sum())
        value=metrics.cooperative_ratio(count,a,b)
        points=metrics.world_rim(pose[:,i],sizes[i]);jerk=metrics.mean_absolute_jerk(points,a,b)
        with np.errstate(divide='ignore',invalid='ignore'):unclipped=float(np.float32(count)/np.float32(b-a))
        starts.append(s);aa.append(a);bb.append(b);all_counts.append(count);ratios.append(value);jerks.append(jerk);raw.append(unclipped)
    ratios=np.asarray(ratios);jerks=np.asarray(jerks);raw=np.asarray(raw)
    np.savez_compressed(path/'protocol_v1_trials.npz',first_success=first,start=starts,start_counter=aa,end_counter=bb,cooperative_frames_all_episode=all_counts,tcoop=ratios,jerk=jerks)
    result.update(tcoop_percent=average(ratios[~np.isnan(ratios)])*100 if (~np.isnan(ratios)).any() else None,jerk_m_s3=average(jerks[jerks!=0]),tcoop_valid=int((~np.isnan(ratios)).sum()),jerk_valid=int((jerks!=0).sum()),no_start=int((np.asarray(starts)<0).sum()),nonpositive_denominator=int((np.asarray(bb)-aa<=0).sum()),tcoop_nan=int(np.isnan(ratios).sum()),upper_clipped=int((raw>1).sum()),lower_clipped=int((raw<0).sum()),jerk_zero=int((jerks==0).sum()),recorded_steps=T)
    save(path/'protocol_v1.json',result);return result

def add_coupled_seed_summary(result):
    overall=result['coupled_overall']
    seed_values={}
    fields={'group_success_rate_percent':'group_success_rate_percent',
            'tcoop_seed_percent':'tcoop_percent','jerk_seed_m_s3':'jerk_m_s3'}
    for output,source in fields.items():
        values=[]
        for seed in result['seeds']:
            parts=[result['runs'][f'coupled_{n}/general_seed{seed}'][source] for n in [2,3,4]]
            values.append(statistics.mean(parts) if all(x is not None for x in parts) else None)
        overall[output]=stats(values)
        seed_values[output]={str(seed):value for seed,value in zip(result['seeds'],values)}
    overall['seed_values']=seed_values
    overall['aggregation']='equal weight for 2/3/4 agents within each seed; seed mean and sample std (ddof=1)'


def add_all_completion_times(result, gpu):
    """Failed trials receive the full 900-step episode budget (30 seconds)."""
    for key,row in result['runs'].items():
        first=np.load(gpu/'eval/table2'/key/'table2_trials.npz')['first_success']
        times=np.where(first>=1,first.astype(float)/30.,30.)
        row['completion_time_all_s']=float(times.mean())
        row['failure_time_penalty_s']=30.
        save(gpu/'eval/table2'/key/'protocol_v1.json',row)
    for name,row in result['groups'].items():
        row['completion_time_all_s']=stats([result['runs'][name+f'/general_seed{s}']['completion_time_all_s'] for s in result['seeds']])
    for prefix,key in [('enable','enable_overall'),('coupled','coupled_overall')]:
        names=['enable_sit','enable_climb','enable_ontop'] if prefix=='enable' else ['coupled_2','coupled_3','coupled_4']
        for field in ['completion_time_s','completion_time_all_s']:
            values=[]
            for seed in result['seeds']:
                parts=[result['runs'][name+f'/general_seed{seed}'][field] for name in names]
                values.append(statistics.mean(parts) if all(v is not None for v in parts) else None)
            result[key][field]=stats(values)
    result['time_definition']='Succ: successful trials only; All: actual success time or 30s for failure. Per-setting then equal-task within seed, seed mean and sample std.'


def write_current_report(base, result):
    """Keep the suite report layout, with current protocol metrics as default."""
    import shutil
    report=base/'report.md'
    backup=base/'report_legacy_metrics.md'
    if not backup.exists() and report.exists():
        shutil.copyfile(report,backup)
    prefix=backup.read_text().split('## table2 /',1)[0].rstrip()
    prefix=prefix.replace('성공률은 seed 평균±표본 표준편차. 제외·실패는 status 및 API audit 참조.',
        '프로토콜: cvpr2027-evaluation-v1.1.0. 성공률·Time·t_coop·Jerk는 seed 평균±표본표준편차(ddof=1).\n64환경 탐색 평가이며 정식512환경 결과와 구분. Mixed 내부 solo는 Independent 전용 결과가 아님.\n제외·실패는 status 및 API audit 참조.')
    lines=[prefix,'']
    for name in sorted(result['groups']):
        row=result['groups'][name]
        metrics={'group_success_rate_percent':row['group_success_rate_percent']}
        metrics['time_succ_s']=row['completion_time_s']
        metrics['time_all_s']=row['completion_time_all_s']
        if name.startswith('enable_'):
            metrics['completion_time_s']=row['completion_time_s']
        else:
            metrics.update(tcoop_percent=row['tcoop_seed_percent'],mean_absolute_jerk_m_s3=row['jerk_seed_m_s3'])
        lines += ['## table2 / '+name,'','검증 seed: [1, 6, 7], 평가 환경: 192','',
                  '```json',json.dumps(metrics,indent=2,ensure_ascii=False),'```','']
    lines += ['## Table 2 요약','','| 설정 | Group SR (%) | Time (Succ. / All) (s) ↓ | t_coop (%) | Jerk (m/s³) |',
              '|---|---:|---:|---:|---:|']
    def fmt(x):
        return 'N/A' if x['mean'] is None else f"{x['mean']:.2f} ± {x['std']:.2f}"
    overall=result['enable_overall']
    lines.append('| Enable 전체 | '+fmt(overall['group_success_rate_percent'])+' | '+fmt(overall['completion_time_s'])+' / '+fmt(overall['completion_time_all_s'])+' | — | — |')
    for name in ['coupled_2','coupled_3','coupled_4']:
        row=result['groups'][name]
        lines.append('| '+name+' | '+fmt(row['group_success_rate_percent'])+' | '+fmt(row['completion_time_s'])+' / '+fmt(row['completion_time_all_s'])+' | '+fmt(row['tcoop_seed_percent'])+' | '+fmt(row['jerk_seed_m_s3'])+' |')
    row=result['coupled_overall']
    lines.append('| Coupled 전체 Avg. | '+fmt(row['group_success_rate_percent'])+' | '+fmt(row['completion_time_s'])+' / '+fmt(row['completion_time_all_s'])+' | '+fmt(row['tcoop_seed_percent'])+' | '+fmt(row['jerk_seed_m_s3'])+' |')
    lines += ['','Enable 전체: seed별 SIT·CLIMB·ON_TOP 동일 가중 평균 후 seed 평균±표본표준편차. 성공한 쌍의 완료 시각만 사용하며 초기 대기 포함.',
              'Time (Succ. / All): Succ.는 성공 trial만 평균. All은 성공 시 실제 완료 시각, 실패 시 최대 episode 30초를 부여해 모든 trial 평균. 각 값은 seed 평균±표본표준편차.',
              'Coupled: 20cm 시작·전체 기록 양손12cm 근접·공용 counter/clip/NaN 및 Jerk 0제외 규칙 적용. 모든 표시값은 seed 평균±표본표준편차이며 pooled 값은 protocol_v1_summary.json에 별도 보존.',
              'Coupled 전체 Avg.: 각 seed에서 2·3·4명 동일 가중 평균을 먼저 계산한 뒤 seed 평균±표본표준편차. 인원별 표준편차의 평균이 아님.','']
    text='\n'.join(lines)
    report.write_text(text)
    (base/'protocol_v1_report.md').write_text(text)

def main(base):
    torch.set_num_threads(1);status=json.loads((base/'status.json').read_text());gpu=Path(status['gpu_output']);assert len(status['verified'])==30
    checks=common_verify(metrics);runs={}
    for p in sorted((gpu/'eval/table2').glob('*/general_seed*')):
        runs[p.parent.name+'/'+p.name]=process(p);print('processed',p.parent.name,p.name,flush=True)
    result=dict(protocol='cvpr2027-evaluation-v1.1.0',exploratory_envs_per_seed=64,seeds=[1,6,7],common_checks_passed=checks,formal_512_envs=False,independent_only_evaluation=False,runs=runs,groups={})
    for name in ['enable_sit','enable_climb','enable_ontop','coupled_2','coupled_3','coupled_4']:
        rows=[runs[name+f'/general_seed{s}'] for s in [1,6,7]]
        summary=dict(group_success_rate_percent=stats([r['group_success_rate_percent'] for r in rows]),completion_time_s=stats([r['completion_time_s'] for r in rows]))
        if name.startswith('coupled_'):
            archives=[np.load(gpu/'eval/table2'/name/f'general_seed{s}'/'protocol_v1_trials.npz') for s in [1,6,7]]
            t=np.concatenate([a['tcoop'] for a in archives]);j=np.concatenate([a['jerk'] for a in archives]);summary.update(tcoop_pooled_percent=average(t[~np.isnan(t)])*100 if (~np.isnan(t)).any() else None,jerk_pooled_m_s3=average(j[j!=0]),tcoop_seed_percent=stats([r['tcoop_percent'] for r in rows]),jerk_seed_m_s3=stats([r['jerk_m_s3'] for r in rows]),tcoop_valid=int((~np.isnan(t)).sum()),jerk_valid=int((j!=0).sum()))
        result['groups'][name]=summary
    esr=[];etime=[]
    for s in [1,6,7]:
        rows=[runs[n+f'/general_seed{s}'] for n in ['enable_sit','enable_climb','enable_ontop']]
        esr.append(statistics.mean(r['group_success_rate_percent'] for r in rows));etime.append(statistics.mean(r['completion_time_s'] for r in rows) if all(r['completion_time_s'] is not None for r in rows) else None)
    result['enable_overall']=dict(group_success_rate_percent=stats(esr),completion_time_s=stats(etime),aggregation='per-task equal weight within seed; then seed mean and sample std')
    cs=[result['groups'][n] for n in ['coupled_2','coupled_3','coupled_4']]
    result['coupled_overall']=dict(tcoop_pooled_percent=average([r['tcoop_pooled_percent'] for r in cs]),jerk_pooled_m_s3=average([r['jerk_pooled_m_s3'] for r in cs]))
    add_coupled_seed_summary(result)
    add_all_completion_times(result, gpu)
    save(base/'protocol_v1_summary.json',result)
    lines=['Protocol cvpr2027-evaluation-v1.1.0; 64-env exploratory run; legacy freeze.', 'Independent-only scenes not evaluated; Mixed solo is not Table1 Ind.', '', '\n'.join(['| Enable setting | Group SR (%) | Successful completion time (s) |','|---|---:|---:|']+[f"| {name} | {row['group_success_rate_percent']['mean']:.2f} ± {row['group_success_rate_percent']['std']:.2f} | {row['completion_time_s']['mean']:.2f} ± {row['completion_time_s']['std']:.2f} |" for name,row in [(n,result['groups'][n]) for n in ['enable_sit','enable_climb','enable_ontop']]+[('Enable overall',result['enable_overall'])]]), '', '| Coupled group | SR (%) | t_coop (%) | Jerk (m/s^3) |','|---|---:|---:|---:|']
    for n in ['coupled_2','coupled_3','coupled_4']:
        r=result['groups'][n];lines.append('| '+n+' | '+' | '.join(f"{r[k]['mean']:.2f} ± {r[k]['std']:.2f}" for k in ['group_success_rate_percent','tcoop_seed_percent','jerk_seed_m_s3'])+' |')
    lines.append('All Coupled columns: seed mean ± sample standard deviation (ddof=1), seeds 1/6/7; pooled values remain separate in JSON.')
    old_summary=json.loads((base/'summary.json').read_text())
    table1=['','| Mixed agents | Solo within Mixed (%) | Enable (%) | Coupled (%) | Mixed (%) |','|---|---:|---:|---:|---:|']
    labels=['Independent Interaction Success Rate','Enabling Interaction Group Success Rate','Coupled Interaction Group Success Rate','Mixed Interaction Group Success Rate']
    for n in [2,4,8,16]:
        scores=old_summary['settings']['table1'][f'mixed_{n}h']['interaction_group_scores']['region']
        table1.append('| '+str(n)+' | '+' | '.join(f"{scores[k]['mean']*100:.2f} ± {scores[k]['std']*100:.2f}" for k in labels)+' |')
    lines[2:2]=table1
    save(base/'protocol_v1_verification.json',dict(common_checks=checks,recorded_surface_reference_smoke='output_etc/check/protocol_v1_metric_smoke_20261111',first_policy_inputs_passed=all(json.loads((gpu/'eval'/key/'first_policy_input.json').read_text())['passed'] for key in status['verified']),raw_suite_cpu_verification_runs=len(status['verified']),protocol_files_sha256={str(p.relative_to(ROOT/'protocol_v1')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'protocol_v1').rglob('*') if p.is_file() and '__pycache__' not in str(p)},postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    write_current_report(base, result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path);a=p.parse_args();main(a.suite.resolve())
