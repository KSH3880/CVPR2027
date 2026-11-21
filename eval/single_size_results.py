import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval')]
import torch
from rescore import rescore
import stage2_xy100
from single_carry_sizes import CARRY_MODE, NINE_SIZES

TASKS = ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')
MODELS = ('tokenhsi', 'sct', 'coopft', 'ours')
SEEDS = (1, 6, 7)
RANGES = (((.25, .8), (.25, .8), (.35, .5)),
          ((.4, .8), (.4, .8), (.2, .7)),
          ((.2, .6),)*3, ((.2, .6),)*3,
          ((1.6, 2.2), (1.6, 2.2), (1.45, 1.7)))
COMMON = ((.4, .6), (.4, .6), (.35, .5))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def within(values, bounds):
    bounds = values.new_tensor(bounds)
    return bool(((values >= bounds[:, 0]-.002) & (values <= bounds[:, 1]+.002)).all())


def geometry(path, variant):
    data = load(path/'initial.pt')
    packet = data['packet'][:, 0].long()
    types = packet[:, 1]
    primary = torch.where(types >= 2, packet[:, 3], packet[:, 4])-1
    batch = torch.arange(len(types))
    sizes = data['sizes'][batch, primary]
    boxes = data['boxes'][batch, primary, :3]
    roots = data['roots'][:, 0, :3]
    target = data['goals'][:, 0].clone()
    ontop = types == 3
    target[ontop] = data['boxes'][batch[ontop], packet[ontop, 4]-1, :3]
    distances = torch.stack(((roots[:, :2]-boxes[:, :2]).norm(dim=-1),
                             (target[:, :2]-boxes[:, :2]).norm(dim=-1)), -1)
    if len(types) != 512 or data['predecessor'].ne(-1).any():
        raise ValueError('Expected 512 independent single-person episodes')
    rows = {}
    for kind, name in enumerate(TASKS):
        mask = types == kind
        v = sizes[mask]
        bounds = COMMON if variant == 'common' and kind in (0, 1, 2) else RANGES[kind]
        nine = CARRY_MODE == 'original9' and kind in (2, 3)
        matches = (v[:, None]-v.new_tensor(NINE_SIZES)[None]).abs().amax(-1) < .0001
        if nine and (not matches.any(-1).all() or not matches.any(0).all()):
            raise ValueError('Expected all nine original carry sizes: '+name)
        if not len(v) or (not nine and not within(v, bounds)):
            raise ValueError('Invalid '+name+' size range')
        if kind != 4 and not nine and ((v*20-(v*20).round()).abs() > .0001).any():
            raise ValueError('Non-PUSH size is off the 5cm grid')
        if kind == 4 and (v[:, :2].min(-1).values < 1.1*v[:, 2]-.002).any():
            raise ValueError('Invalid PUSH profile')
        d = distances[mask]
        lo = 2. if kind == 4 else 1.
        if not within(d[:, :1], ((lo, 5.),)):
            raise ValueError('Invalid '+name+' approach distance: '+str(d[:, 0].min().item())+'..'+str(d[:, 0].max().item()))
        if kind >= 2 and not within(d[:, 1:], ((lo, 5.),)):
            raise ValueError('Invalid '+name+' transport distance')
        rows[name] = dict(trials=int(mask.sum()), size_min=v.min(0).values.tolist(),
            size_max=v.max(0).values.tolist(), approach_min=float(d[:, 0].min()),
            approach_max=float(d[:, 0].max()), transport_min=float(d[:, 1].min()),
            transport_max=float(d[:, 1].max()))
        if nine:
            rows[name]['original9_counts'] = matches.sum(0).tolist()
    support = data['sizes'][batch[ontop], packet[ontop, 4]-1]
    if not within(support, COMMON if variant == 'common' else ((.4, .8), (.4, .8), (.25, .5))):
        raise ValueError('Invalid ON_TOP support range')
    rows['ON_TOP']['support_min'] = support.min(0).values.tolist()
    rows['ON_TOP']['support_max'] = support.max(0).values.tolist()
    return rows


def verify(path, reference, variant, model, steps):
    stage2_xy100.install()
    data = read(path/'result.json')
    if data.get('carry_sizes', 'table1') != CARRY_MODE:
        raise ValueError('Incorrect carry size protocol')
    if (data['num_envs'], data['num_agents'], data['num_objects'], data['horizon']) != (512, 1, 2, steps):
        raise ValueError('Incorrect evaluation dimensions')
    if data['simulated_steps'] != steps and data['falls'] != 512:
        raise ValueError('Incomplete rollout')
    before, after = load(reference/'initial.pt'), load(path/'initial.pt')
    for key in before:
        if key != 'boxes' and not torch.equal(before[key], after[key]):
            raise ValueError('Initial state mismatch: '+key)
    parked = load(path/'physics_only_reset.pt')
    mask = parked['mask']
    if not torch.equal(before['boxes'], parked['observed']) or not torch.equal(before['boxes'][~mask], after['boxes'][~mask]):
        raise ValueError('Initial object mismatch')
    expected = load(reference/'observed_scene.pt')
    if not torch.equal(expected, parked['observed_scene']) or not torch.equal(expected, load(path/'first_policy_input.pt')):
        raise ValueError('First observation mismatch')
    physics = read(path/'physics_only_verification.json')
    if physics['checked_frames'] != data['simulated_steps']+1:
        raise ValueError('Missing physical distractor checks')
    policy = data['policy_verification']
    if policy['epoch'] != dict(tokenhsi=40500, sct=26000, coopft=7000, ours=2000)[model]:
        raise ValueError('Wrong model epoch')
    if not policy['weights_unchanged'] or not policy['normalization_unchanged']:
        raise ValueError('Policy changed')
    score = rescore(path)
    if not all(score[k] for k in ('finite', 'cpu_gpu_exact', 'json_counts_exact', 'errors_match_recorded')):
        raise ValueError('CPU rescoring failed')
    if sum(data['scores']['region'][k]['trials'] for k in TASKS) != 512:
        raise ValueError('Wrong trial denominator')
    return dict(initial_state_exact=True, first_input_exact=True, cpu_sr_error_exact=True,
        policy_preserved=True, physical_distractor_checks=physics['checked_frames'],
        geometry=geometry(reference, variant))


def summarize(output, verified):
    rows = []
    lines = ['# 1명 Independent · 일반/Enable 공통 크기 · '+CARRY_MODE, '',
        f'검증 완료 {len(verified)}/24회. 512환경 × seed1/6/7 × 900step. SR %, seed 평균±표본std.', '',
        '| 크기 | 모델 | 전체 SR | SIT | CLIMB | AT | ON_TOP | PUSH |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for variant in ('native', 'common'):
        for model in MODELS:
            keys = [f'{variant}/{model}/seed{s}' for s in SEEDS]
            if not all(k in verified for k in keys):
                continue
            results = [read(output/'rollouts'/variant/model/f'general_seed{s}'/'result.json') for s in SEEDS]
            stats = {}
            for task in ('overall', *TASKS):
                values = [100*(sum(d['scores']['region'][k]['success'] for k in TASKS)/512 if task == 'overall'
                              else d['scores']['region'][task]['rate']) for d in results]
                stats[task] = dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)
            rows.append(dict(variant=variant, model=model, metrics=stats,
                counts={t:sum(d['scores']['region'][t]['trials'] for d in results) for t in TASKS}))
            lines.append('| '+variant+' | '+model+' | '+' | '.join(f'{v["mean"]:.2f}±{v["std"]:.2f}' for v in stats.values())+' |')
    sizes_note = ('AT와 ON_TOP payload는 두 조건 모두 TokenHSI 원본9종. 공통 크기는 SIT/CLIMB/ON_TOP 받침에 적용.'
                  if CARRY_MODE == 'original9' else 'ON_TOP payload·PUSH는 기존 크기 유지.')
    lines += ['', '일반은 Table 1 행동별 크기를 기본으로 한다. 공통 범위는 X/Y .40–.60m, Z .35–.50m.',
        sizes_note+' PUSH는 기존 크기 유지. 행동별20% 무작위 배정이며 전체SR은 개인 성공/전체 시도다.',
        '각 크기/seed의 초기 장면을 네 모델이 공유한다. 크기가 바뀌면 배치 재시도도 달라질 수 있어 조건 간 완전히 동일한 기하는 보장하지 않는다.',
        'TokenHSI는 기존397D 자기 상태/담당 과제 입력을 유지하며 distractor attention을 지원하지 않는다.',
        '', '[원시 집계](summary.json) · [실행·출처](manifest.json) · [상태](status.json)']
    write(output/'summary.json', dict(completed=len(verified)==24, verified=len(verified), carry_sizes=CARRY_MODE, rows=rows))
    (output/'report.md').write_text('\n'.join(lines)+'\n')
