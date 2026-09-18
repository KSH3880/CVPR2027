# Relation diagnostics: TensorBoard pins and episode traces

These diagnostics do not change rewards, success conditions, observations,
termination, RSI, or checkpoint compatibility. They become available on the next
process launch, including a normal resume with the same reward configuration.
Already-running processes and historical event files are not modified.

## Common placement criterion

Every experiment uses the same diagnostic predicate, independent of its reward
success predicate: box-to-goal XY distance <= 0.10 m AND absolute Z error <=
0.001 m. This is a geometric proxy, not a contact, grasp, or release detector.

One completed **agent-episode** is one statistical sample. When a scene ends,
each agent is finalized using the last simulated state, before reset overwrites
it. Timeout and fall terminations both count. Unfinished episodes carry across
rollouts and are not counted until completion. Forced resets without a terminal
event discard the unfinished episode. Reset-resampling attempts do not count.

An agent that already satisfies placement at reset is excluded from the main
placement metrics and counted separately. This avoids scoring a favorable RSI
reset as newly achieved placement. Episodes that start near but outside the
placement tolerance remain eligible; the metrics do not prove full-distance
transport or continuous grasping.

## Recommended pins (10 total)

| TensorBoard tag | Scale and meaning |
| --- | --- |
| `relation/holding/satisfied` | 0–1, rollout step fraction with Holding phi >= 0.9. |
| `relation/at/satisfied` | 0–1, rollout step fraction with At phi >= 0.9; no Holding prerequisite. |
| `relation/placement/episode_ever_rate` | 0–1, reached placement at least once / eligible completed agent-episodes. |
| `relation/placement/episode_final_rate` | 0–1, placed at termination / eligible completed agent-episodes. |
| `relation/placement/post_first_retention` | 0–1, mean per-episode placement occupancy after first placement, reached eligible episodes only. |
| `relation/placement/first_seconds` | Seconds to first placement, reached eligible episodes only; lower is faster. |
| `relation/placement/longest_hold_seconds` | Mean longest uninterrupted geometric placement duration, reached eligible episodes only; NOT duration of grasping. |
| `relation/placement/eligible_completed_count` | Number of eligible agent-episodes completed during this logging window. |
| `relation/placement/reached_count` | Number of those episodes that reached placement; denominator for conditional metrics. |
| `relation/placement/initially_placed_count` | Completed agent-episodes excluded because they started placed. |

All new `placement/` metrics aggregate **completed episodes since the previous
logging call**, not every rollout frame and not lifetime totals. Counts reset at
each logging call. At 2048 environments with two agents, a single scene finish
can contribute two agent-episodes. The existing `holding/satisfied` and
`at/satisfied` retain their rollout-time averaging semantics.

For retention, each reached episode's fraction is calculated separately and
then averaged with equal episode weight. The first placed frame is included.
For example, placement at step 400 in a 600-step episode gives a 201-frame
post-first window. If placement holds for 100 of those frames, its retention is
100/201. An episode reaching placement only on its final frame has retention 1
but only one frame of evidence. Use `longest_hold_seconds` and the additional
`relation/placement/post_first_seconds` (mean observed post-first window length)
to distinguish long maintenance from a late lucky hit.

If no eligible episodes completed, ever/final rates are omitted. If none of the
eligible completed episodes reached placement, conditional retention/time tags
are omitted, not logged as artificial zeros. Counts still appear. TensorBoard
may connect points across these missing intervals: check `reached_count` at the
same frame and use smoothing 0 when debugging. A lower first-time metric alone
does not prove improvement if reach rate also falls.

Additional tags: `relation/placement/completed_count`,
`relation/placement/initially_placed_fraction`, and
`relation/placement/initially_placed_final_rate` identify the excluded group.

## How to interpret the combination

- High ever rate, low final rate/retention: placement is reached but not kept.
- Low ever rate and high XY error: acquisition/transport/approach is still failing.
- High retention but short longest duration/post-first window: possibly only a
  brief placement near termination, not demonstrated long-term stability.
- Low Holding step fraction is not alone proof of failure: placing and releasing
  can legitimately reduce Holding after the task is complete.

`relation/done` and `at/achieved` remain history-flag **step averages**. They are
not episode success rates. Their success criteria can differ between reward
experiments, so do not use them as the common placement comparison.

## Per-step timeline CSV

Files: `<run>/diagnostics/relation_timeline_<session-id>.csv`.
The default traces all agents in environment 0, every simulated step, on the
first and then every 100th reset episode, up to 600 steps per sampled episode.
This is a bounded sample, not an unbiased estimate over all environments.
Existing `relation_samples.csv` periodic sampling remains unchanged.

Timeline columns include environment/agent/episode/step/time, actual Holding and
At phi/gates, XY/Z errors, raw direction progress, the prerequisite and blend
actually used for reward computation, raw and paid state/progress rewards,
success event/current-valid/history flags, common placement, initially placed,
and saturation-active flags. `gate` is the current gate; `prerequisite_used`
captures the preceding-state gate used by the dense reward. Actual phi/gates are
never overwritten by saturation. `task_relation_total` excludes external
power/collision/box-speed penalties and AMP.

Rows are buffered and flushed periodically, at sampled episode termination,
and at training logging calls. An abrupt process kill can lose the remaining
unflushed rows. Each process creates a separate CSV to avoid mixing episode IDs
or appending different column layouts to older runs.

Optional settings under `env.relationReward.diagnostics`:

```yaml
enabled: true
timeline_enabled: true
timeline_sample_envs: 1
timeline_every_episodes: 100
timeline_max_steps: 600
```

These defaults work without config edits. Set `timeline_every_episodes: 1` for
a short, focused debugging run; avoid tracing many environments during long
training. Set `timeline_enabled: false` to disable only the timeline CSV.
Setting diagnostics `enabled: false` disables the new metrics and traces too.
