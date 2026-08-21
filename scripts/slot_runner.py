#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path

from queue_docs import ROOT, atomic_write, exit_path, load_track, now_text


def main(track_name, tag, gpu, lock_name):
    track = load_track(track_name)
    lock = Path(lock_name)
    env_path = ROOT / "runs/queue/logs" / f"{tag}.env.json"
    log_path = ROOT / "runs/queue/logs" / f"{tag}.log"
    record = {"track": track_name, "tag": tag, "gpu": int(gpu)}
    rc = 125
    try:
        values = json.loads(env_path.read_text())
        expected = {int(value) for value in values.pop("QUEUE_EXPECT_RC", "0").split(",")}
        env = os.environ.copy()
        env.update({key: str(value) for key, value in values.items()})
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env[track["tag_env"]] = tag
        with log_path.open("w") as log:
            rc = subprocess.run(["bash", str(ROOT / track["train"])], cwd=str(ROOT),
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record.update({"status": "training_complete" if rc in expected else "failed",
                       "rc": rc, "expected_rc": sorted(expected)})
    except Exception as error:
        record.update({"status": "failed", "rc": rc, "detail": str(error)})
    finally:
        record["finished_at"] = now_text()
        path = exit_path(track, tag)
        atomic_write(path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        lock.unlink(missing_ok=True)
        with (ROOT / "runs/queue/gpumap.log").open("a") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/queue_status.py"), "--auto-write"],
                           stdout=subprocess.DEVNULL, stderr=log)
    return rc


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: slot_runner.py <track> <tag> <gpu> <lock>")
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
