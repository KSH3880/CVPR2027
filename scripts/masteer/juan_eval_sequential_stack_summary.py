import os
import sys

import numpy as np


tag, path, envs = sys.argv[1], sys.argv[2], int(sys.argv[3])
box_grid = int(sys.argv[4]) != 0
m = np.load(path)
if m.ndim != 2 or m.shape[1] != 55:
    raise SystemExit(f"Juan metric shape (N, 55) expected, got {m.shape}")

by_row = {}
for row in m:
    by_row.setdefault(int(row[0]), []).append(row)
episodes = []
missing = []
for env in range(envs):
    a1 = by_row.get(2 * env, [])
    a2 = by_row.get(2 * env + 1, [])
    if min(len(a1), len(a2)) < 3:
        missing.append((env, len(a1), len(a2)))
        continue
    for index in (1, 2):
        episodes.append((a1[index], a2[index]))
if missing:
    raise SystemExit(
        "per-env warm-up + 2 measured episodes missing: " + repr(missing[:10]))
if len(episodes) != 2 * envs:
    raise SystemExit(f"expected {2 * envs} episodes, got {len(episodes)}")

a1 = np.stack([pair[0] for pair in episodes])
a2 = np.stack([pair[1] for pair in episodes])
success = a1[:, 40] > 0.5
terminate_reason = a1[:, 54].astype(np.int64)
root_sum = a1[:, 26] + a2[:, 26]
box_sum = a1[:, 27] + a2[:, 27]
steps = np.maximum(a1[:, 30] + a2[:, 30], 1.0)


def mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def success_stats(mask):
    if not np.any(mask):
        return (" success_root_cum=NA success_box_cum=NA"
                " success_root_mae=NA success_box_mae=NA")
    return (
        f" success_root_cum={mean(root_sum[mask]):.4f}"
        f" success_box_cum={mean(box_sum[mask]):.4f}"
        f" success_root_mae={mean(root_sum[mask] / steps[mask]):.4f}"
        f" success_box_mae={mean(box_sum[mask] / steps[mask]):.4f}"
    )


parts = [
    f"SEQ_STACK_EVAL tag={tag}",
    f"success_mode={os.environ['STACK_EVAL_SUCCESS_MODE']}",
    f"episodes={len(episodes)}",
    f"success_n={int(success.sum())}",
    f"success_rate={success.mean():.4f}",
    "warmup=per_env_discarded",
    f"root_cum={mean(root_sum):.4f}",
    f"box_cum={mean(box_sum):.4f}",
    f"root_mae={mean(root_sum / steps):.4f}",
    f"box_mae={mean(box_sum / steps):.4f}",
]
for name, source, col in (("place", a1, 42),
                          ("retreat", a1, 45),
                          ("a2", a2, 48)):
    count = np.maximum(source[:, col + 2], 1.0)
    parts.extend((
        f"{name}_root_cum={mean(source[:, col]):.4f}",
        f"{name}_box_cum={mean(source[:, col + 1]):.4f}",
        f"{name}_root_mae={mean(source[:, col] / count):.4f}",
        f"{name}_box_mae={mean(source[:, col + 1] / count):.4f}",
    ))
parts.extend((
    f"terminate_n={int(np.sum(terminate_reason != 0))}",
    f"terminate_rate={mean(terminate_reason != 0):.4f}",
    f"term_fall_a1_n={int(np.sum(terminate_reason == 1))}",
    f"term_fall_a2_n={int(np.sum(terminate_reason == 2))}",
    f"term_fall_both_n={int(np.sum(terminate_reason == 3))}",
    f"term_bottom_displaced_n={int(np.sum(terminate_reason == 4))}",
    f"term_multiple_n={int(np.sum(terminate_reason == 5))}",
    f"term_unknown_n={int(np.sum(terminate_reason == 6))}",
))
print(" ".join(parts) + success_stats(success))


def size_label(row):
    return "x".join(f"{value:.2f}" for value in row[51:54])


combo_keys = sorted({(size_label(a1[i]), size_label(a2[i]))
                     for i in range(len(episodes))})
if box_grid:
    expected = len(episodes) // 9
    combo_counts = {
        key: sum(size_label(a1[i]) == key[0]
                 and size_label(a2[i]) == key[1]
                 for i in range(len(episodes)))
        for key in combo_keys
    }
    if len(combo_keys) != 9 or any(
            count != expected for count in combo_counts.values()):
        raise SystemExit(
            f"unbalanced 3x3 box grid: expected 9 x {expected}, "
            f"got {combo_counts}")
for bottom, top in combo_keys:
    mask = np.array([
        size_label(a1[i]) == bottom and size_label(a2[i]) == top
        for i in range(len(episodes))
    ])
    reason = terminate_reason[mask]
    print(
        f"SEQ_STACK_BOX_COMBO tag={tag} bottom={bottom} top={top} "
        f"episodes={int(mask.sum())} success_n={int(success[mask].sum())} "
        f"success_rate={mean(success[mask]):.4f} "
        f"root_mae={mean(root_sum[mask] / steps[mask]):.4f} "
        f"box_mae={mean(box_sum[mask] / steps[mask]):.4f} "
        f"terminate_rate={mean(reason != 0):.4f}"
    )
