#!/usr/bin/env python3
"""Derives every setting that has to move when num_envs moves.

    python scripts/env_config.py 8192
    python scripts/env_config.py 8192 --samples 400e6
    python scripts/env_config.py 8192 --apply runs/gen_cfgs/steer/f2.yaml

Changing num_envs alone is not a speed knob, it is a different experiment.
The batch grows, so rl_games runs more minibatch steps on each rollout before
collecting fresh data, and the policy drifts further from the one that gathered
it. Holding that step count fixed is what makes a bigger num_envs the same
learning at a different wall-clock, and that means minibatch_size has to grow
with it. Everything below follows from that one requirement.

See docs/ENV_SCALING.md for the measurements this is based on.
"""

import argparse
import re
import sys
from pathlib import Path

# The 2048-env recipe every steer result so far was produced with. Ratios are
# preserved from here; the absolute numbers come from TokenHSI's own configs.
REF = {"num_envs": 2048, "minibatch_size": 16384, "horizon_length": 32,
       "mini_epochs": 6, "learning_rate": 5e-5}

# num_envs -> max_gpu_contact_pairs that runs with NO PhysX overflow warning.
# This is the knob that matters, not default_buffer_size_multiplier: raising the
# multiplier from 10 to 5000 changed nothing, while raising contact pairs cleared
# the warning immediately. Bigger is NOT safer -- an oversized buffer costs real
# throughput (4096 runs 28 % slower on 128M than on 32M), so use the smallest
# value that stays clean. Each row is measured, not extrapolated.
PHYSX_CONTACT_PAIRS = {2048: 8 << 20, 4096: 32 << 20, 8192: 128 << 20}
PHYSX_UNTESTED = "no verified contact-pair setting; run scripts/steer/env_sweep.sh first"


def derive(num_envs, samples=None, lr_rule="sqrt"):
    scale = num_envs / REF["num_envs"]
    batch = num_envs * REF["horizon_length"]
    mb = int(REF["minibatch_size"] * scale)
    if batch % mb:
        raise SystemExit(f"batch {batch} is not divisible by minibatch {mb}; "
                         f"use a power-of-two num_envs")
    steps = (batch // mb) * REF["mini_epochs"]
    ref_steps = ((REF["num_envs"] * REF["horizon_length"]) //
                 REF["minibatch_size"]) * REF["mini_epochs"]
    # Larger minibatches mean less gradient noise per step, which is the only
    # reason to touch the learning rate at all. Square-root scaling is the
    # conservative choice; linear is the aggressive one and needs a warmup.
    lr = REF["learning_rate"] * (scale ** (0.5 if lr_rule == "sqrt" else 1.0))
    out = {
        "num_envs": num_envs,
        "batch_size": batch,
        "minibatch_size": mb,
        "minibatches_per_epoch": batch // mb,
        "sequential_updates_per_rollout": steps,
        "reference_updates_per_rollout": ref_steps,
        "learning_rate": lr,
        "max_gpu_contact_pairs": PHYSX_CONTACT_PAIRS.get(num_envs),
    }
    if samples:
        out["max_iterations"] = int(round(samples / batch))
        out["total_samples"] = int(samples)
    return out


def apply_to(cfg_path, d):
    """Rewrite minibatch_size and learning_rate in an rl_games train cfg."""
    p = Path(cfg_path)
    text = p.read_text()
    for key, val in (("minibatch_size", d["minibatch_size"]),
                     ("learning_rate", f"{d['learning_rate']:.3g}")):
        text, n = re.subn(rf"^(\s*){key}: .*$", rf"\g<1>{key}: {val}", text,
                          flags=re.M)
        if n != 1:
            raise SystemExit(f"expected exactly one '{key}:' in {p}, found {n}")
    p.write_text(text)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("num_envs", type=int)
    ap.add_argument("--samples", type=float, default=None,
                    help="total samples to match, e.g. 400e6")
    ap.add_argument("--lr-rule", choices=("sqrt", "linear", "none"), default="sqrt")
    ap.add_argument("--apply", metavar="TRAIN_CFG",
                    help="rewrite minibatch_size and learning_rate in this cfg")
    a = ap.parse_args()

    d = derive(a.num_envs, a.samples, a.lr_rule)
    if a.lr_rule == "none":
        d["learning_rate"] = REF["learning_rate"]

    w = max(len(k) for k in d)
    for k, v in d.items():
        print(f"  {k:<{w}}  {v}")

    if d["sequential_updates_per_rollout"] != d["reference_updates_per_rollout"]:
        print("\n  !! sequential updates differ from the 2048 reference - this is "
              "a different learning problem, not just a faster one")
    if d["max_gpu_contact_pairs"] is None:
        print(f"\n  !! num_envs={a.num_envs}: {PHYSX_UNTESTED}")
    else:
        print(f"\n  physx: set sim.physx.max_gpu_contact_pairs to "
              f"{d['max_gpu_contact_pairs']} ({d['max_gpu_contact_pairs'] >> 20}M)")
        if a.num_envs != 4096:
            print("  note: 4096 measured fastest (34,956 samples/s vs 26,652 at 2048, "
                  "25,478 at 8192)")
    if a.apply:
        apply_to(a.apply, d)


if __name__ == "__main__":
    sys.exit(main())
