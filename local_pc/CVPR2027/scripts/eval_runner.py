#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path

from queue_docs import ROOT, atomic_write, load_track, now_text


def main(track_name, tag, gpu, source, checkpoint, lock_name):
    track = load_track(track_name)
    lock = Path(lock_name)
    failure = ROOT / "runs/queue/eval_failures" / track_name / f"{tag}.json"
    launcher_log = ROOT / "runs/queue/logs" / f"eval_{tag}.launcher.log"
    env = os.environ.copy()
    env["QUEUE_EVAL_CKPT"] = checkpoint
    env["QUEUE_EVAL_SOURCE"] = source
    rc = 125
    try:
        with launcher_log.open("a") as log:
            rc = subprocess.run(
                ["bash", str(ROOT / track["eval"]), tag, gpu, "512"],
                cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            failure.unlink(missing_ok=True)
        else:
            atomic_write(failure, json.dumps({
                "track": track_name, "tag": tag, "rc": rc,
                "failed_at": now_text(), "retry_after_minutes": 30,
            }, indent=2, ensure_ascii=False) + "\n")
    except Exception as error:
        atomic_write(failure, json.dumps({
            "track": track_name, "tag": tag, "rc": rc, "detail": str(error),
            "failed_at": now_text(), "retry_after_minutes": 30,
        }, indent=2, ensure_ascii=False) + "\n")
    finally:
        lock.unlink(missing_ok=True)
        with (ROOT / "runs/queue/gpumap.log").open("a") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/queue_status.py"), "--auto-write"],
                           stdout=subprocess.DEVNULL, stderr=log)
    return rc


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit("usage: eval_runner.py <track> <tag> <gpu> <source> <checkpoint> <lock>")
    raise SystemExit(main(*sys.argv[1:]))
