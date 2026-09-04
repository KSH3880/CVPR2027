#!/bin/bash
# Re-score finished runs on extra path draws until every one has n=4.
#
#   bash scripts/steer/repeat_evals.sh [prefix] [n_repeats]
#
# One measurement per run is what made the F4 table unreadable: the L sweep came
# out non-monotone and nobody could say whether that was the axis or the draw.
# Repeating the eval costs 5 minutes against a 3.3 h retrain and pins each policy
# to +/-0.004, which is enough to separate arms 0.08 apart. Training variance
# still needs new runs; this settles everything that does not.
set -u
ROOT=/home/hwanhee/CVPR2027
cd "$ROOT"
PREFIX=${1:-f4_}
N=${2:-3}

free_gpu() {
    for x in 0 1 2 3 4 5 6 7; do
        [ -n "$(nvidia-smi -i $x --query-compute-apps=pid --format=csv,noheader | head -1)" ] && continue
        [ -e "$ROOT/runs/queue/gpu_locks/gpu$x" ] && continue
        echo $x; return
    done
}

# Drive off the training logs, not results.tsv: a run that finished but has not
# been scored once yet still needs all N measurements.
for f in runs/queue/logs/${PREFIX}*.log; do
    tag=$(basename "$f" .log)
    case "$tag" in eval_*|rep_*|reval_*) continue;; esac
    grep -q "MAX EPOCHS NUM" "$f" 2>/dev/null || continue
    for r in $(seq 0 "$N"); do
        out=$tag; [ "$r" != "0" ] && out=${tag}_r${r}
        [ -f "runs/steer/${out}.npz" ] && continue
        while :; do
            g=$(free_gpu)
            [ -n "$g" ] && break
            sleep 20
        done
        # r=0 is the plain eval, under the bare tag -- passing a literal 0 would
        # name it <tag>_r0 and split the run across two names in the table.
        seed_arg=""; [ "$r" != "0" ] && seed_arg=$r
        nohup bash scripts/steer/eval_one.sh "$tag" "$g" 512 $seed_arg \
            > "runs/queue/logs/rep_${tag}_r${r}.log" 2>&1 &
        echo "$tag repeat $r -> gpu$g"
        sleep 25          # let it claim the card before the next scan
    done
done
wait
echo "repeat_evals done"
