"""Pure aggregation: completed episodes for rates, all frames for diagnostics."""


def summarize_retreat(records):
    def mean(values):
        return sum(values) / len(values) if values else None
    eligible = [r for r in records if r['completed'] and r['retreat_steps'] > 0]
    result = dict(
        completed_episodes=sum(r['completed'] for r in records),
        retreat_episodes=len(eligible),
        censored_episodes=sum(not r['completed'] for r in records),
        retreat_reach_ratio=mean([r['reached'] for r in eligible]),
        retreat_stop_ratio=mean([r['stopped'] for r in eligible]),
        retreat_fall_ratio=mean([r['fall'] for r in eligible]),
        endpoint_drift_mae=mean([x for r in records for x in r['endpoint_changes']]),
        retreat_path_mae=mean([x for r in records for x in r['path_errors']]),
        collision_proxy_ratio=(sum(r['collision_steps'] for r in records) /
                               max(1, sum(r['retreat_steps'] for r in records))),
    )
    for field in ('min_box_gap', 'min_agent_gap'):
        result[field] = mean([r[field] for r in records if r[field] is not None])
    # Descriptive natural-condition subsets, not controlled intervention tests.
    for condition in ('approaching', 'stationary'):
        subset = [r for r in eligible if r[condition + '_steps'] > 0]
        result[condition + '_episodes'] = len(subset)
        result[condition + '_reach_ratio'] = mean([r['reached'] for r in subset])
        result[condition + '_collision_proxy_ratio'] = (
            sum(r['collision_steps'] for r in subset) / sum(r['retreat_steps'] for r in subset)
            if subset else None)
    return result
