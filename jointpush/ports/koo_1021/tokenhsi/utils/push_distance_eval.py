"""CPU summaries of independent first episodes, grouped by initial box-goal XY distance."""
import math


def summarize_distance_results(results):
    rows = [dict(row, repeat=repeat) for repeat, result in results.items()
            for row in result.get('push_distance_audit', {}).get('agents', [])]

    def summarize(selected):
        n = len(selected)
        reached = [r['first_success_seconds'] for r in selected if r['ever_success']]
        return dict(agent_trials=n,
                    ever_success_count=sum(r['ever_success'] for r in selected),
                    final_success_count=sum(r['final_success'] for r in selected),
                    ever_success_rate=sum(r['ever_success'] for r in selected)/n if n else None,
                    final_success_rate=sum(r['final_success'] for r in selected)/n if n else None,
                    mean_first_success_seconds=sum(reached)/len(reached) if reached else None,
                    mean_final_goal_distance_m=sum(r['final_goal_distance_m'] for r in selected)/n if n else None,
                    mean_initial_agent_box_distance_m=sum(r['initial_agent_box_distance_m'] for r in selected)/n if n else None)

    bins = {}
    for lo, hi in ((3., 3.5), (3.5, 4.), (4., 4.5), (4.5, 5.)):
        selected = [r for r in rows if lo <= r['initial_goal_distance_m'] and
                    (r['initial_goal_distance_m'] < hi or hi == 5. and r['initial_goal_distance_m'] <= hi)]
        bins[f'{lo:g}-{hi:g}m'] = summarize(selected)
    outside = [r for r in rows if not 3. <= r['initial_goal_distance_m'] <= 5.]
    if any(not math.isfinite(r['initial_goal_distance_m']) for r in rows):
        raise ValueError('Nonfinite initial goal distance')
    return dict(completed_scene_trials=sum(v.get('num_scene_trials', 0) for v in results.values()),
                overall=summarize(rows), by_initial_box_goal_xy_distance=bins,
                outside_3to5m=summarize(outside),
                caveat='Locomotion starts; each scene contains two correlated agent trials. '
                       'Ever means reached at least once before first scene termination; final means at termination. '
                       'Distance bins use actual initial box-to-goal XY distance, not humanoid approach distance.')
