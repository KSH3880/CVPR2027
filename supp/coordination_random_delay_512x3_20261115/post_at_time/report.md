# Post-AT Completion Time: Group success only

(first simultaneous Group success step − first AT success step) / 30, in seconds.
AT-only and failed trials are excluded; no failure-time substitution. Zero successful trials = N/A.
Within each seed: successful-episode mean per type → equal-weight mean of three types. Then seed mean ± sample std.
If a type has zero successes, its equal-weight family mean is N/A; pooled successful-trial means are also saved.
Condition order: 0 → random_1_5; seeds=[1, 6, 7].

## enable_climb

| Model | Post-AT (s) | Group success/total |
|---|---:|---:|
| SCT | 3.692 ± 0.360 → 2.925 ± 0.282 | 230/512 → 174/512 |
| SCT+FSM | 6.769 ± 0.538 → 7.130 ± 0.431 | 255/512 → 254/512 |
| CoopFT7k | 5.253 ± 0.382 → 3.989 ± 0.086 | 248/512 → 255/512 |
| DCT | 4.501 ± 0.383 → 3.671 ± 0.136 | 417/512 → 426/512 |
| TokenHSI | 1.829 ± 0.168 → 3.067 ± 1.902 | 30/512 → 19/512 |

## enable_ontop

| Model | Post-AT (s) | Group success/total |
|---|---:|---:|
| SCT | 4.235 ± 0.647 → 3.556 ± 0.080 | 145/512 → 122/512 |
| SCT+FSM | 7.195 ± 0.230 → 7.099 ± 0.336 | 147/512 → 141/512 |
| CoopFT7k | 5.753 ± 0.109 → 5.582 ± 0.446 | 265/512 → 265/512 |
| DCT | 5.515 ± 0.150 → 5.176 ± 0.164 | 412/512 → 387/512 |
| TokenHSI | 2.999 ± 1.478 → 2.815 ± 0.851 | 55/512 → 26/512 |

## enable_sit

| Model | Post-AT (s) | Group success/total |
|---|---:|---:|
| SCT | 4.444 ± 0.920 → 4.182 ± 0.694 | 112/512 → 43/512 |
| SCT+FSM | 7.578 ± 0.184 → 7.325 ± 0.203 | 254/512 → 261/512 |
| CoopFT7k | 5.746 ± 0.313 → 4.557 ± 0.309 | 329/512 → 333/512 |
| DCT | 4.832 ± 0.191 → 3.982 ± 0.112 | 439/512 → 423/512 |
| TokenHSI | 2.488 ± 0.458 → 2.427 ± 1.472 | 77/512 → 23/512 |

## enable_average

| Model | Post-AT (s) | Group success/total |
|---|---:|---:|
| SCT | 4.124 ± 0.418 → 3.554 ± 0.302 | 487/1536 → 339/1536 |
| SCT+FSM | 7.181 ± 0.316 → 7.185 ± 0.322 | 656/1536 → 656/1536 |
| CoopFT7k | 5.584 ± 0.246 → 4.709 ± 0.234 | 842/1536 → 853/1536 |
| DCT | 4.950 ± 0.107 → 4.276 ± 0.040 | 1268/1536 → 1236/1536 |
| TokenHSI | 2.439 ± 0.395 → 2.770 ± 1.089 | 162/1536 → 68/1536 |

