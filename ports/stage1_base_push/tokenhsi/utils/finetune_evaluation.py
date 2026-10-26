"""Position-only benchmark scoring; no simulator imports or training reward changes."""
import json
from pathlib import Path
import numpy as np

TASKS = ('push_at', 'holding_at', 'holding_ontop', 'sit', 'climb')
AT_SIZES = [(s, s, s) for s in (.22, .27, .32, .37, .42, .47, .52, .57)] + [(.30, .30, .40)]


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def classify(task, roots, source, support, source_size, support_size, goals, dtype=np.float64):
    """Inputs end with XYZ / xyz+xyzw. Two criteria on exactly the same frame."""
    roots, source, support, source_size, support_size, goals = [
        np.asarray(x, dtype=dtype) for x in (roots, source, support, source_size, support_size, goals)]
    if task in ('holding_at', 'push_at'):
        distance = np.linalg.norm(source[..., :3] - goals, axis=-1)
        success = distance <= .20
        return success, success.copy()
    x, y, z, w = np.moveaxis(support[..., 3:7], -1, 0)
    zrow = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
    top = support[..., 2] + (np.abs(zrow)*support_size/2).sum(-1)
    target = support[..., :3].copy()
    target[..., 2] = top + (source_size[..., 2]/2 if task == 'holding_ontop'
                            else .10 if task == 'sit' else .94)
    point = source[..., :3] if task == 'holding_ontop' else roots
    point_success = np.linalg.norm(point-target, axis=-1) <= .20
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    delta = point[..., :2] - support[..., :2]
    lx = np.cos(yaw)*delta[..., 0] + np.sin(yaw)*delta[..., 1]
    ly = -np.sin(yaw)*delta[..., 0] + np.cos(yaw)*delta[..., 1]
    region = ((np.abs(lx) <= .4*support_size[..., 0]) &
              (np.abs(ly) <= .4*support_size[..., 1]) &
              (np.abs(point[..., 2]-target[..., 2]) <= .20))
    return point_success, region


def summarize_rows(rows):
    n = len(rows)
    out = {'agent_trials': n}
    for name in ('point_ever', 'region_ever', 'point_final', 'region_final', 'strict_ever', 'terminated'):
        count = sum(bool(r[name]) for r in rows)
        out[name+'_count'] = count
        out[name+'_rate'] = count/n if n else None
    for key in ('initial_box_goal_distance_m', 'initial_agent_box_distance_m',
                'final_box_goal_distance_m', 'progress_fraction', 'best_progress_fraction'):
        out['mean_'+key] = sum(r[key] for r in rows)/n if n else None
    times = [r['first_point_success_seconds'] for r in rows if r['first_point_success_seconds'] is not None]
    out['mean_first_point_success_seconds'] = float(np.mean(times)) if times else None
    return out


def distance_bins(rows):
    """Actual initial box-centre→fixed-goal XY distance. Left-closed bins."""
    out = {}
    edges = [0., 1.] + list(map(float, range(2, 21))) + [float('inf')]
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if lo <= r['initial_box_goal_distance_m'] < hi]
        label = f'{lo:g}-{hi:g}m' if np.isfinite(hi) else '20m+'
        out[label] = summarize_rows(selected)
    assert sum(x['agent_trials'] for x in out.values()) == len(rows)
    return out


def verify_archive(directory):
    """Independently rescore saved first-episode trajectories, ignoring reset episodes."""
    directory=Path(directory)
    result=json.loads((directory/'results.json').read_text())
    with np.load(directory/'initial.npz') as initial, np.load(directory/'trace.npz') as trace:
        src, sup=initial['source'], initial['support']
        batch=np.arange(len(src))[:,None]
        point,region=classify(result['task'],trace['root'],trace['boxes'][:,batch,src],
            trace['boxes'][:,batch,sup],initial['sizes'][batch,src][None],
            initial['sizes'][batch,sup][None],trace['goals'])
        active=trace['active'][...,None]
        for key,flags in [('point_ever',point),('region_ever',region)]:
            ever=(flags & active).any(0)
            expected=np.array([r[key] for r in result['agents']]).reshape(ever.shape)
            np.testing.assert_array_equal(ever,expected)
        assert result['num_scene_trials']==len(src)
    return {'saved_trajectory_rescore_match':True}


def summarize_batch(root):
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text()) if (root/'run_manifest.json').exists() else {}
    planned = manifest.get('planned_runs', 15)
    records = [json.loads(p.read_text()) for p in sorted(root.glob('*/results.json'))]
    result = {'completed_runs': len(records), 'tasks': {},
              'caveat': 'Two-agent/four-box policy environment. Agents in a scene are correlated. '
                        'Not the one-person single_match benchmark; no comparison of identical initial scenes claimed.'}
    for task in TASKS:
        runs = [r for r in records if r['task'] == task]
        if not runs:
            continue
        rows = [a for r in runs for a in r['agents']]
        item = {'completed_seeds': [r['seed'] for r in runs], 'pooled': summarize_rows(rows),
                'per_seed': {str(r['seed']): summarize_rows(r['agents']) for r in runs}}
        for metric in ('point_ever_rate', 'region_ever_rate'):
            values = [r[metric] for r in item['per_seed'].values()]
            item[metric+'_mean'] = float(np.mean(values))
            item[metric+'_std_ddof1'] = float(np.std(values, ddof=1)) if len(values)>1 else None
        item['per_owner'] = {str(owner): summarize_rows([r for r in rows if r['owner']==owner]) for owner in (0, 1)}
        if task == 'push_at':
            item['by_initial_box_goal_distance'] = distance_bins(rows)
            item['distance_by_seed'] = {str(r['seed']): distance_bins(r['agents']) for r in runs}
        result['tasks'][task] = item
    atomic_json(root/'summary.json', result)
    lines = ['# PUSH fine-tune 평가 중간/최종 결과', '',
             f'완료 {len(records)}/{planned} 실행. 2인·4상자 환경이며 1인 single_match와 직접 동등 비교하지 않는다.', '',
             '| 과제 | 완료 seed | agent 시도 | 점 ever | 영역 ever |', '|---|---|---:|---:|---:|']
    for task, item in result['tasks'].items():
        p=item['pooled']
        lines.append(f"| {task} | {item['completed_seeds']} | {p['agent_trials']} | {p['point_ever_rate']*100:.2f}% | {p['region_ever_rate']*100:.2f}% |")
    if 'push_at' in result['tasks']:
        lines += ['', '## PUSH 초기 상자→목표 XY 거리별', '',
                  '| 거리 | 시도 | 점 ever 성공 | 성공률 | 평균 종료 잔여거리 | 평균 진행률 |', '|---|---:|---:|---:|---:|---:|']
        for label, p in result['tasks']['push_at']['by_initial_box_goal_distance'].items():
            if p['agent_trials']:
                lines.append(f"| {label} | {p['agent_trials']} | {p['point_ever_count']} | {p['point_ever_rate']*100:.2f}% | {p['mean_final_box_goal_distance_m']:.3f}m | {p['mean_progress_fraction']*100:.2f}% |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result
