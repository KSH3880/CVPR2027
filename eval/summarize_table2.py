import argparse
import json
from pathlib import Path
import statistics

import torch

from table2_sampling import SETTINGS


def stats(values):
    return dict(mean=statistics.mean(values) if None not in values else None,
                std=statistics.stdev(values) if None not in values and len(values)>1 else None,
                seeds=values)


def summarize(output, seeds):
    torch.set_num_threads(1)
    results, checks = {}, []
    for model in ('stage1','stage2'):
        results[model] = {}
        for setting in SETTINGS:
            folder = output/model/setting
            manifest = json.loads((folder/'manifest.json').read_text())
            if not manifest['completed'] or manifest['seeds'] != seeds:
                raise ValueError('Incomplete Table2 evaluation: '+str(folder))
            verification = json.loads((folder/'verification.json').read_text())
            if len(verification) != len(seeds) or not all(row['cpu_gpu_table2_exact'] for row in verification):
                raise ValueError('Missing Table2 verification')
            rows = [json.loads((folder/('general_seed'+str(seed))/'table2.json').read_text()) for seed in seeds]
            item = dict(success_rate_percent=stats([row['group_success_rate']*100 for row in rows]),
                completion_time_s=stats([row['completion_time_s'] for row in rows]),
                successes=[row['successes'] for row in rows], trials=[row['trials'] for row in rows], runs=rows)
            if setting.startswith('coupled_'):
                item['cooperative_time_ratio_percent'] = {key:stats([row['cooperative_time_ratio_percent'][key] for row in rows])
                                                           for key in ('6','8','12')}
            results[model][setting] = item
        enable = [results[model][key] for key in ('enable_sit','enable_climb','enable_ontop')]
        coupled = [results[model]['coupled_'+str(k)] for k in (2,3,4)]
        for name, source in (('enabling_average',enable),('coupled_average',coupled)):
            results[model][name] = dict(success_rate_percent=stats([
                statistics.mean(row['success_rate_percent']['seeds'][i] for row in source) for i in range(len(seeds))]))
        results[model]['enabling_average']['completion_time_s'] = stats([
            statistics.mean(row['completion_time_s']['seeds'][i] for row in enable)
            if all(row['completion_time_s']['seeds'][i] is not None for row in enable) else None for i in range(len(seeds))])
        results[model]['coupled_average']['cooperative_time_ratio_percent'] = {key:stats([
            statistics.mean(row['cooperative_time_ratio_percent'][key]['seeds'][i] for row in coupled)
            for i in range(len(seeds))]) for key in ('6','8','12')}
    for setting in SETTINGS:
        for seed in seeds:
            name = 'general_seed'+str(seed)
            left = torch.load(output/'stage1'/setting/name/'initial.pt', map_location='cpu', weights_only=False)
            right = torch.load(output/'stage2'/setting/name/'initial.pt', map_location='cpu', weights_only=False)
            if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
                raise ValueError('Stage1/Stage2 initial scene mismatch: '+setting+' '+str(seed))
            checks.append(dict(setting=setting, seed=seed, initial_states_exact=True))
    data = dict(seeds=seeds, results=results, xy_extent_percent=100, contact_threshold_cm=6,
                comparison_checks=checks, success_time_trials='successful_only', contact_trials='all')
    (output/'summary.json').write_text(json.dumps(data, indent=2))

    def fmt(item):
        return '{:.2f}±{:.2f}'.format(item['mean'],item['std']) if item['mean'] is not None else 'N/A'

    lines = ['# Table 2 ·Stage1 26k / Stage2 V2 2k', '',
        '512환경 × 임시 seed '+', '.join(map(str,seeds))+' ·900step·XY100%·M=2N.',
        '같은 초기 장면·물리·성공 판정, distractor 관측 유지/물리 제거. 평균±표본표준편차.', '',
        '## (a) Enabling', '', '| 모델 | AT→SIT SR (%) | AT→CLIMB SR (%) | AT→ON_TOP SR (%) | Avg. (%) | Time (s) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model,label in (('stage1','SCT (Stage1 26k)'),('stage2','Ours (Stage2 V2 2k)')):
        result = results[model]
        values = [fmt(result[key]['success_rate_percent']) for key in ('enable_sit','enable_climb','enable_ontop','enabling_average')]
        lines.append('| '+label+' | '+' | '.join(values+[fmt(result['enabling_average']['completion_time_s'])])+' |')
    lines += ['', 'Time은 성공한 쌍의 최초 동시 성공 시간이다. 행동별 평균 후 동일 가중 평균하며 실패는 제외한다.', '',
        '## (b) Coupled', '', '| 모델 | 2H SR (%) | 3H SR (%) | 4H SR (%) | Avg. (%) | t_coop 6cm (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model,label in (('stage1','SCT (Stage1 26k)'),('stage2','Ours (Stage2 V2 2k)')):
        result = results[model]
        values = [fmt(result[key]['success_rate_percent']) for key in ('coupled_2','coupled_3','coupled_4','coupled_average')]
        lines.append('| '+label+' | '+' | '.join(values+[fmt(result['coupled_average']['cooperative_time_ratio_percent']['6'])])+' |')
    lines += ['', 't_coop는 상자 초기 수평 변위20cm부터 최초 성공까지(실패는 종료까지) 전원 양손이 실제 공유 상자 표면6cm 미만인 프레임 비율이다. 실패 포함·미시작0%, 최초 참가자 유지·낙상 이후 접촉0, 시작/끝 프레임 포함·30Hz다.', '',
        '## 거리 문턱별 t_coop (%)', '', '| 모델 | 문턱 | 2H | 3H | 4H | Avg. |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for model in ('stage1','stage2'):
        for key in ('6','8','12'):
            values = [fmt(results[model][setting]['cooperative_time_ratio_percent'][key])
                      for setting in ('coupled_2','coupled_3','coupled_4','coupled_average')]
            lines.append('| '+model+' | '+key+'cm | '+' | '.join(values)+' |')
    lines += ['', '## Seed별 결과', '', '| 모델 | 설정 | Seed | 성공/시도 | SR (%) | 성공 Time (s) | t_coop 6cm (%) |',
              '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for model in ('stage1','stage2'):
        for setting in SETTINGS:
            for seed,row in zip(seeds,results[model][setting]['runs']):
                time = '{:.2f}'.format(row['completion_time_s']) if row['completion_time_s'] is not None else 'N/A'
                ratio = '{:.2f}'.format(row['cooperative_time_ratio_percent']['6']) if setting.startswith('coupled_') else '—'
                lines.append('| {} | {} | {} | {}/{} | {:.2f} | {} | {} |'.format(
                    model,setting,seed,row['successes'],row['trials'],row['group_success_rate']*100,time,ratio))
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:24]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[1,3,6])
    args = parser.parse_args()
    summarize(args.output.resolve(),args.seeds)
