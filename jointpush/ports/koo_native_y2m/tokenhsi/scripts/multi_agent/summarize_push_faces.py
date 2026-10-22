"""Summarize opt-in face evaluation JSON or completed repeats from its log."""
import argparse
import ast
import json
from pathlib import Path


def physical_summary(rows):
    phases = ('obb_layout_stats', 'force_proxy_layout_stats',
              'before_first_success_moving_obb_stats', 'before_first_success_moving_force_stats')
    result = {}
    for phase in phases:
        groups = {}
        for row in rows:
            for layout, stats in row.get(phase, {}).items():
                target = groups.setdefault(layout, dict(agent_trials=0, steps=0, tilt_sum_deg=0.,
                    max_tilt_deg=0., over10_steps=0, over15_steps=0, goal_distance_gain_m=0., speed_sum_mps=0.))
                target['agent_trials'] += 1
                for key, value in stats.items():
                    target[key] = max(target[key], value) if key == 'max_tilt_deg' else target[key]+value
        for stats in groups.values():
            n = max(1, stats['steps'])
            stats['mean_tilt_deg'] = stats.pop('tilt_sum_deg')/n
            stats['over10_fraction'] = stats.pop('over10_steps')/n
            stats['over15_fraction'] = stats.pop('over15_steps')/n
            stats['mean_speed_mps'] = stats.pop('speed_sum_mps')/n
        result[phase] = groups
    # Assign episode groups using only movement before first success, avoiding
    # post-success hand release. Keep mixed/low evidence episodes explicit.
    for kind in ('obb', 'force'):
        phase = 'before_first_success_moving_'+kind+'_stats'
        classified = {}
        for row in rows:
            stats = row.get(phase, {})
            counts = {k: stats.get(k, {}).get('steps', 0) for k in ('same_face', 'different_faces')}
            total = sum(v.get('steps', 0) for v in stats.values())
            dominant = max(counts, key=counts.get)
            label = dominant if counts[dominant] >= 15 and counts[dominant]/max(1,total) >= .6 else 'mixed_or_insufficient'
            for geometry in ('all', row['initial_region']):
                group = classified.setdefault(geometry, {}).setdefault(label, [])
                group.append(row)
        tables = {}
        for geometry, groups in classified.items():
            tables[geometry] = {}
            for label, selected in groups.items():
                n = len(selected)
                maxima = [max((s['max_tilt_deg'] for s in r.get('obb_layout_stats', {}).values()), default=0.) for r in selected]
                tables[geometry][label] = dict(agent_trials=n,
                    ever_success_count=sum(r['ever_success'] for r in selected),
                    final_success_count=sum(r['final_success'] for r in selected),
                    ever_at_success=sum(r['ever_success'] for r in selected)/n,
                    final_at_success=sum(r['final_success'] for r in selected)/n,
                    mean_episode_max_tilt_deg=sum(maxima)/n,
                    mean_goal_distance_reduction_m=sum(r['goal_distance_initial']-r['goal_distance_final'] for r in selected)/n)
        result['episode_dominant_'+kind] = tables
    return result


def summarize(results):
    rows = [row for result in results.values()
            for row in result.get('push_face_audit', {}).get('agents', [])]
    groups = {}
    for region in ('face_center', 'oblique', 'corner'):
        selected = [r for r in rows if r['initial_region'] == region]
        if not selected:
            continue
        count = len(selected)
        layout, moving = {}, {}
        for row in selected:
            for key, value in row['layout_steps'].items():
                layout[key] = layout.get(key, 0)+value
            for key, value in row['moving_unsuccessful_layout_steps'].items():
                moving[key] = moving.get(key, 0)+value
        reached = [r['first_success_seconds'] for r in selected if r['ever_success']]
        groups[region] = dict(agent_trials=count,
            ever_at_success=sum(r['ever_success'] for r in selected)/count,
            final_at_success=sum(r['final_success'] for r in selected)/count,
            first_success_seconds_reached_only=sum(reached)/len(reached) if reached else None,
            mean_goal_distance_reduction_m=sum(r['goal_distance_initial']-r['goal_distance_final'] for r in selected)/count,
            layout_step_fractions={k:v/max(1, sum(layout.values())) for k,v in layout.items()},
            moving_unsuccessful_layout_step_fractions={k:v/max(1, sum(moving.values())) for k,v in moving.items()})
    return dict(completed_scene_trials=sum(r.get('num_scene_trials', 1) for r in results.values()),
                agent_trials=len(rows), by_initial_goal_geometry=groups,
                physical_layout_comparison=physical_summary(rows),
                caveat='Random layouts; two agents share each scene. Face labels are XY proxies, not force contacts. '
                       'Small groups and observational layout fractions do not establish causal superiority.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.path.suffix == '.json':
        results = json.loads(args.path.read_text())
    else:
        prefix = '[edge context evaluation] '
        results = {}
        for line in args.path.read_text().splitlines():
            if line.startswith(prefix):
                results[str(len(results))] = ast.literal_eval(line[len(prefix):])
    print(json.dumps(summarize(results), indent=2))
