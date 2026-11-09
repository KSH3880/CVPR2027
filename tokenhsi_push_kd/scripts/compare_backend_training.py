"""Summarize the matched worker/shared validation and historical controls."""
import json
import statistics
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output_etc/check/backend_parity'
RUNS = {
    'worker_2048': OUT / 'train_worker',
    'shared_fixed_2048': OUT / 'train_shared',
    'historical_worker_2048': ROOT / 'output/push100_solo_random42_seed42_20261105_210745',
    'shared_buggy_4096': ROOT / 'output_etc/check/push4096_kd0p001_fresh_seed42_20261106_140658',
}
report = {}
contracts = {}
for label, path in RUNS.items():
    directories = sorted(path.glob('*/summaries'))
    if not directories:
        continue
    if len(directories) != 1:
        raise ValueError('Ambiguous run: ' + str(path))
    acc = EventAccumulator(str(directories[0]), size_guidance={'scalars': 0})
    acc.Reload()
    tags = acc.Tags()['scalars']
    if 'reward_push/iter' not in tags:
        continue
    rewards = acc.Scalars('reward_push/iter')
    result = dict(path=str(directories[0].parent), latest_iter=rewards[-1].step,
                  latest_reward=rewards[-1].value, reward_windows={})
    for lo, hi in [(1, 20), (21, 40), (41, 60)]:
        values = [x.value for x in rewards if lo <= x.step <= hi]
        if values:
            result['reward_windows'][f'{lo}-{hi}'] = dict(count=len(values), mean=statistics.mean(values))
    epochs = {x.step: int(x.value) for x in acc.Scalars('info/epochs')}
    result['update_metrics_21_60'] = {}
    for tag in ('losses/a_loss', 'losses/c_loss', 'losses/disc_loss',
                'distill/teacher_kl', 'distill/coefficient', 'performance/play_time',
                'performance/update_time', 'samples/push_rollout', 'samples/kd_rollout'):
        values = [x.value for x in acc.Scalars(tag) if 21 <= epochs.get(x.step, -1) <= 60]
        if values:
            result['update_metrics_21_60'][tag] = statistics.mean(values)
    contract = json.loads((directories[0].parent / 'method_contract.json').read_text())
    result['configuration'] = dict(kd_coef=contract['kd_coef'], env=contract['env'])
    if label in ('worker_2048', 'shared_fixed_2048'):
        contracts[label] = contract
        log = (OUT / ('train_worker.log' if label == 'worker_2048' else 'train_shared.log')).read_text(errors='replace')
        result['completed_60'] = 'ITER 60/60' in log and 'MAX EPOCHS NUM!' in log
    report[label] = result
if len(contracts) == 2:
    a, b = contracts.values()
    different_env = {k: [a['env'].get(k), b['env'].get(k)] for k in a['env'].keys() | b['env'].keys()
                     if a['env'].get(k) != b['env'].get(k)}
    other = [k for k in a.keys() | b.keys() if k != 'env' and a.get(k) != b.get(k)]
    report['contract_differences'] = dict(environment=different_env, other=other)
(OUT / 'training_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
for key, value in report.items():
    if 'latest_iter' in value:
        print(key, 'ITER', value['latest_iter'], 'reward', round(value['latest_reward'], 4),
              'windows', {k: round(v['mean'], 4) for k, v in value['reward_windows'].items()})
    else:
        print(key, value)
