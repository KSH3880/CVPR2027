"""Export mass-dependent success and conditional tilt for both PUSH policies."""
import csv
import json
from pathlib import Path
import sys


def export(root):
    root = Path(root)
    fields = ['policy', 'mode', 'group', 'mass_kg', 'trials', 'successful_trials',
              'success_pct', 'upright_success_pct', 'first_success_tilt_mean_deg',
              'first_success_tilt_median_deg', 'first_success_tilt_p90_deg',
              'max_success_tilt_mean_deg', 'max_success_tilt_p90_deg']
    per_size, bins = [], []
    for policy in ['gpu4_policy', 'gpu6_policy']:
        data = json.loads((root / policy / 'audit.json').read_text())
        for mode in ['training_reset_mix', 'loco']:
            for grouping, target in [('per_size', per_size), ('size_bins', bins)]:
                for group, v in data['summary'][mode][grouping].items():
                    if v is None:
                        continue
                    target.append(dict(zip(fields, [policy, mode, group,
                        v['mass_kg_mean'], v['trials'], v['successful_trials'],
                        100*v['ever_success_rate'], 100*v['ever_upright_success_rate'],
                        v['first_success_tilt_deg_mean'], v['first_success_tilt_deg_median'],
                        v['first_success_tilt_deg_p90'], v['max_success_tilt_deg_mean'],
                        v['max_success_tilt_deg_p90']])))
    for name, rows in [('per_mass.csv', per_size), ('mass_bins.csv', bins)]:
        with (root / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    def fmt(value):
        return 'N/A' if value is None else f'{value:.2f}'
    lines = ['# PUSH 무게별 성공률·성공 시 기울기 비교', '',
        '성공률: 에피소드 중 AT에 한 번 이상 도달한 agent 비율. 기울기: 박스 수직축과 월드 수직축 사이 각도(도), yaw 제외.',
        '첫 성공 기울기는 성공 에피소드마다 첫 성공 순간 1회 집계. 성공 중 최대 기울기는 각 성공 에피소드의 성공 프레임들에서 최대값을 구한 뒤 집계. 실패 에피소드는 두 통계에서 제외(N/A).',
        '각 정책의 원래 config·물성·정규화 사용. 같은 seed와 크기별 표본 수를 사용하지만 문 유무와 정책 계약이 달라 초기 장면 완전 일치는 보장하지 않음.',
        '크기와 질량이 함께 변하므로 질량만의 인과효과가 아님. training_reset_mix는 학습 RSI 혼합, loco는 보행 시작.', '',
        '|정책|시작|크기군|평균 kg|표본|성공 %|수직성공 %|첫 성공 평균°|첫 성공 P90°|성공 중 최대 평균°|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in bins:
        lines.append('|'+'|'.join(str(r[k]) if k in ['policy','mode','group','trials'] else fmt(r[k]) for k in [
            'policy','mode','group','mass_kg','trials','success_pct','upright_success_pct',
            'first_success_tilt_mean_deg','first_success_tilt_p90_deg','max_success_tilt_mean_deg'])+'|')
    lines += ['', '31개 크기별 상세: `per_mass.csv`; 3개 크기군: `mass_bins.csv`; 원시 기록: 각 정책 폴더의 `trials.json`·`audit.json`.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(f'Results: {root}/REPORT.md')


if __name__ == '__main__':
    export(sys.argv[1])
