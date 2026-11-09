"""Per-stage numbers for the slide: PUSH signal-to-noise per minibatch, KD vs PUSH-signal size and direction.

Model: each minibatch PUSH gradient = s + noise. With two halves of h minibatches each, the half-sum cosine is
c = h|s|^2 / (h|s|^2 + sigma^2), so sigma^2/|s|^2 = h(1/c - 1). The signal share of the full sum (2h minibatches)
is sqrt(2) / sqrt(1 + 1/c).
"""
import json
import math
import sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    g = r['groups']['all']
    h = r['minibatches'] / 2
    c = g['half_consistency']['ppo']
    n = g['norm']
    keys = ['kd_sit', 'kd_carry', 'kd_climb']
    pair = g['cos_kd_pairs']
    sq = sum(n[k] ** 2 for k in keys)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sq += 2 * n[a] * n[b] * pair[a + '|' + b]
    kd = math.sqrt(sq)
    if c > 0:
        snr = 1 / math.sqrt(h * (1 / c - 1))
        share = math.sqrt(2) / math.sqrt(1 + 1 / c)
        signal = share * n['ppo']
        true_cos = g['cos_ppo_kd_total'] / share
        print(f"{path}: mb={r['minibatches']} c={c:.3f} SNR=1/{1 / snr:.1f} signal_share={share:.2f} "
              f"KD/signal={kd / signal:.3f} cos_tot={g['cos_ppo_kd_total']:+.3f} (signal-corrected {true_cos:+.3f}) "
              f"KD effect on PUSH progress={kd / signal * true_cos:+.4f}")
    else:
        print(f"{path}: mb={r['minibatches']} c={c:.3f} -> signal not measurable")
