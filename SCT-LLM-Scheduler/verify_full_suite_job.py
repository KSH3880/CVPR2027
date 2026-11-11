"""Read saved traces; check original scoring, Table2 metrics and actual releases."""
import argparse
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'teacher_runtime/tokenhsi'),str(ROOT/'teacher_runtime/eval')]
import torch
from rescore import rescore
from evaluate import summarize
import stage2_xy100
from table2_metrics import replay
stage2_xy100.install();torch.set_num_threads(1)


def save(path,value):
    path.write_text(json.dumps(value,indent=2))


def verify(path,reference,schedules,envs,steps,preflight):
    result=json.loads((path/'result.json').read_text())
    excluded={x['environment'] for x in result['llm_scheduler']['excluded']}
    n=result['num_envs'];m=result['num_agents']
    assert n==envs-len(excluded) and result['num_objects']==2*m
    assert result['horizon']==steps and result['simulated_steps']==steps
    proof=result['stage1_policy']
    assert all(proof[k] for k in ('native_actor_exact','normalization_exact','weights_unchanged'))
    assert proof['coordination_calls']==0
    assert all(json.loads((path/'llm_initial_match.json').read_text()).values())
    assert result['llm_scheduler']['first_step']['max_displacement_m']<.5
    initial=torch.load(path/('initial_full.pt' if excluded else 'initial.pt'),map_location='cpu',weights_only=False)
    expected=torch.load(reference/'initial.pt',map_location='cpu',weights_only=False)
    audit=json.loads((path/'release_audit.json').read_text())
    frozen=torch.tensor([[row[f'A{i+1:02d}']>0 for i in range(m)]
                         for row in audit['quantized_release_times_s']],dtype=torch.bool)
    for key,value in expected.items():
        if key=='boxes':continue
        value=value.clone()
        if key=='roots':value[...,7:][frozen]=0
        if key=='dof_vel':value[frozen]=0
        assert torch.equal(value,initial[key]),key
    parked=torch.load(path/'physics_only_reset.pt',map_location='cpu',weights_only=False)
    assert torch.equal(expected['boxes'],parked['observed'])
    assert torch.equal(expected['boxes'][~parked['mask']],initial['boxes'][~parked['mask']])
    assert torch.equal(torch.load(reference/'observed_scene.pt',map_location='cpu',weights_only=False),parked['observed_scene'])
    physics=json.loads((path/'physics_only_verification.json').read_text())
    assert physics['checked_frames']==steps+1
    score=rescore(path)
    assert all(score[k] for k in ('finite','cpu_gpu_exact','json_counts_exact','errors_match_recorded'))
    if (path/'table2.json').exists():save(path/'table2_recomputed.json',replay(path))
    audit=json.loads((path/'release_audit.json').read_text());checked=0;past_horizon=0
    dt=result['control_dt']
    for i in range(envs):
        if i in excluded:
            assert not audit['release_events'][i];continue
        if preflight:continue
        response=json.loads((schedules/f'env_{i:04d}'/'response.raw.txt').read_text())
        events={r['agent_id']:r for r in audit['release_events'][i]}
        for row in response['schedule']:
            agent=row['agent_id'];tick=int((Decimal(str(row['release_time_s']))/Decimal(str(dt))).to_integral_value(rounding=ROUND_CEILING))
            assert abs(audit['quantized_release_times_s'][i][agent]-tick*dt)<1e-9
            if tick<steps:
                assert events[agent]['release_frame']==tick
                assert abs(events[agent]['actual_release_time_s']-tick*dt)<1e-9
            else:
                assert agent not in events;past_horizon+=1
            checked+=1
    save(path/'suite_verification.json',dict(passed=True,preflight=preflight,simulated_envs=envs,
        evaluated_envs=n,excluded=sorted(excluded),steps=steps,agents=m,
        cpu_gpu_exact=True,initial_states_exact=True,table2_exact=(path/'table2.json').exists(),
        actual_release_matches_raw=True,agents_checked=checked,release_beyond_horizon=past_horizon))


def aggregate(base):
    status=json.loads((base/'status.json').read_text());gpu=Path(status['gpu_output'])
    manifest=json.loads((base/'manifest.json').read_text());out={};total=0
    text=['# Stage1 + LLM · 64환경 전체 평가','',
          'Independent 전용 제외. Mixed 2/4/8/16명 + Table2 6설정, seed 1/6/7.',
          '실행 가능한 원본 스케줄 사용, API 자동 재시도/스케줄 보정 없음.',
          '성공률은 seed 평균±표본 표준편차. 제외·실패는 status 및 API audit 참조.','']
    for table in ('table1','table2'):
        out[table]={}
        for name in sorted({j['name'] for j in manifest['jobs'] if j['table']==table}):
            directory=gpu/'eval'/table/name
            paths=[directory/f'general_seed{s}' for s in (1,6,7)
                   if (directory/f'general_seed{s}'/'suite_verification.json').exists()]
            if not paths:continue
            # Only verified results enter the aggregate; never hide a failed run.
            aggregate_dir=base/'verified_results'/table/name;aggregate_dir.mkdir(parents=True,exist_ok=True)
            for p in paths:(aggregate_dir/p.name).symlink_to(p.resolve(),target_is_directory=True)
            summary=summarize(aggregate_dir)
            records=[json.loads((p/'result.json').read_text()) for p in paths]
            count=sum(r['num_envs'] for r in records);total+=count
            summary.update(evaluated_envs=count,verified_seeds=[r['seed'] for r in records])
            text += ['## '+table+' / '+name,'',f'검증 seed: {summary["verified_seeds"]}, 평가 환경: {count}','']
            if table=='table1':
                for label,row in summary.get('interaction_group_scores',{}).get('region',{}).items():
                    text.append(f'- {label}: {row["mean"]*100:.2f}% ± {(row["std"] or 0)*100:.2f}%')
            else:
                metrics=[json.loads((p/'table2.json').read_text()) for p in paths]
                stats={}
                for field in ('group_success_rate','completion_time_s'):
                    values=[r[field] for r in metrics if r[field] is not None]
                    stats[field]=dict(mean=statistics.mean(values) if values else None,
                        std=statistics.stdev(values) if len(values)>1 else None,valid_seeds=len(values))
                if name.startswith('coupled'):
                    stats['cooperative_time_ratio_percent']={}
                    for cm in ('6','8','12'):
                        values=[r['cooperative_time_ratio_percent'][cm] for r in metrics]
                        stats['cooperative_time_ratio_percent'][cm]=dict(mean=statistics.mean(values),
                            std=statistics.stdev(values) if len(values)>1 else None)
                summary['table2']=stats
                text.append('```json\n'+json.dumps(stats,indent=2)+'\n```')
            text.append('');out[table][name]=summary
    usage=dict(calls=0,prompt_tokens=0,completion_tokens=0,total_tokens=0)
    for p in (base/'api').glob('*/*/general_seed*/env_*/inference.json'):
        row=json.loads(p.read_text());usage['calls']+=row['calls']
        for k in ('prompt_tokens','completion_tokens','total_tokens'):usage[k]+=(row.get('usage') or {}).get(k,0)
    save(base/'api_usage.json',usage)
    save(base/'summary.json',dict(settings=out,evaluated_envs=total,api_usage=usage,failures=status['failures']))
    (base/'report.md').write_text('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path,nargs='?');p.add_argument('reference',type=Path,nargs='?')
    p.add_argument('--schedules',type=Path);p.add_argument('--envs',type=int);p.add_argument('--steps',type=int)
    p.add_argument('--preflight',action='store_true');p.add_argument('--summarize',type=Path)
    a=p.parse_args()
    if a.summarize:aggregate(a.summarize)
    else:verify(a.path,a.reference,a.schedules,a.envs,a.steps,a.preflight)
