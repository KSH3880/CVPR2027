import argparse
import fcntl
import json
from pathlib import Path
import statistics

METRICS = ['Independent Interaction Success Rate','Enabling Interaction Group Success Rate',
           'Coupled Interaction Group Success Rate','Mixed Interaction Group Success Rate']


def stats(values):
    if not values or any(v is None for v in values):
        return dict(mean=None,std=None,values=values)
    return dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0.,values=values)


def summarize(output):
    rows = {}
    for table in ('table1','table2'):
        for manifest in sorted((output/table).glob('*/manifest.json')):
            config = json.loads(manifest.read_text())
            if not config['completed']:
                continue
            runs = [json.loads((manifest.parent/f'general_seed{seed}/result.json').read_text()) for seed in config['seeds']]
            checks = [json.loads((manifest.parent/f'general_seed{seed}/verification.json').read_text()) for seed in config['seeds']]
            item = dict(table=table,setting=manifest.parent.name,seeds=config['seeds'],runs=runs,verification=checks,
                success_rates_percent={metric:stats([r['interaction_group_scores']['region'][metric]['rate']*100
                    if r['interaction_group_scores']['region'][metric]['rate'] is not None else None for r in runs]) for metric in METRICS})
            if table=='table2':
                item['completion_time_s'] = stats([r['table2']['completion_time_s'] for r in runs])
                if item['setting'].startswith('coupled_'):
                    item['provisional_tcoop_percent'] = {key:stats([r['table2']['cooperative_time_ratio_percent'][key] for r in runs])
                                                          for key in ('6','8','12')}
            rows[table+'/'+item['setting']] = item
    def fmt(s):
        return 'N/A' if s['mean'] is None else f"{s['mean']:.2f}±{s['std']:.2f}"
    lines = ['# SCT26k + FSM 평가','','XY100%·M=2N·distractor 관측/attention 유지·물리 제거. 평균±표본표준편차.',
        'Coupled는 각 참여자의 양손12cm 도착을 누적하고 마지막 도착 때 전원 일회 해제한다.',
        't_coop 시작 조건은 미정이며 아래 값은 기존20cm 변위 시작 정의의 참고값이다. 원시 손/상자 궤적을 보존한다.','',
        '| Table 1 설정 | '+' | '.join(METRICS)+' |','| --- | ---: | ---: | ---: | ---: |']
    for key,item in rows.items():
        if item['table']=='table1':
            lines.append('| '+item['setting']+' | '+' | '.join(fmt(item['success_rates_percent'][k]) for k in METRICS)+' |')
    lines += ['','| Table 2 설정 | Group Success Rate (%) | 성공 Completion Time (s) | 참고 t_coop 6cm (%) | 8cm | 12cm |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,item in rows.items():
        if item['table']=='table2':
            metric = METRICS[1] if item['setting'].startswith('enable_') else METRICS[2]
            contact = [fmt(item['provisional_tcoop_percent'][k]) for k in ('6','8','12')] if 'provisional_tcoop_percent' in item else ['—']*3
            completion = fmt(item['completion_time_s']) if item['setting'].startswith('enable_') else '—'
            lines.append('| '+item['setting']+' | '+' | '.join([fmt(item['success_rates_percent'][metric]),completion,*contact])+' |')
    averages = {}
    for name,settings,metric in (('enabling',['enable_sit','enable_climb','enable_ontop'],METRICS[1]),
                                 ('coupled',['coupled_2','coupled_3','coupled_4'],METRICS[2])):
        items = [rows.get('table2/'+s) for s in settings]
        if all(items):
            if any(i['seeds']!=items[0]['seeds'] for i in items):
                raise ValueError('Table2 seed mismatch')
            count = len(items[0]['seeds'])
            averages[name] = dict(success_rate_percent=stats([statistics.mean(i['success_rates_percent'][metric]['values'][s] for i in items) for s in range(count)]))
            if name=='enabling':
                averages[name]['completion_time_s'] = stats([statistics.mean(i['completion_time_s']['values'][s] for i in items)
                    if all(i['completion_time_s']['values'][s] is not None for i in items) else None for s in range(count)])
            lines += ['',name+' 동일 가중 평균 Group Success Rate: '+fmt(averages[name]['success_rate_percent'])+'%.']
            if name=='enabling':
                lines.append('Enabling 성공 Completion Time: '+fmt(averages[name]['completion_time_s'])+'s.')
    lines += ['','## Seed별 성공률','','| 설정 | Seed | '+' | '.join(METRICS)+' |','| --- | --- | ---: | ---: | ---: | ---: |']
    for key,item in rows.items():
        for seed,r in zip(item['seeds'],item['runs']):
            values = [r['interaction_group_scores']['region'][m]['rate'] for m in METRICS]
            lines.append('| '+key+' | '+str(seed)+' | '+' | '.join('N/A' if v is None else f'{v*100:.2f}' for v in values)+' |')
    complete = len(rows)==14
    data = dict(completed=complete,completed_settings=len(rows),expected_settings=14,results=rows,table2_averages=averages)
    (output/'summary.json').write_text(json.dumps(data,indent=2))
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'FSM summary: {len(rows)}/14 settings complete',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    output = parser.parse_args().output.resolve()
    with (output/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summarize(output)
