#!/usr/bin/env python3
"""Keeps the GPUs busy with queued experiments.

Jobs are json files dropped into runs/queue/pending/. Each is claimed as soon as
enough GPUs are free, run with CUDA_VISIBLE_DEVICES pinned, and moved to done/
or failed/ with its exit code and log path recorded. The runner owns nothing
else: it never kills a process it did not start, so it coexists with whatever
else is on the box.

  {"name": "s0_lookahead_1.2", "gpus": 1, "cmd": "bash scripts/steer/s0.sh 1.2"}

Optional keys: "priority" (lower runs first), "after" (name that must be done
first), "timeout" (seconds).
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/hwanhee/CVPR2027")
Q = ROOT / "runs/queue"
LOGS = ROOT / "runs/queue/logs"
POLL = 15
MEM_FREE_MIB = 2000  # a gpu with less than this in use counts as free


def dirs():
    for d in ("pending", "running", "done", "failed"):
        (Q / d).mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def gpu_busy():
    """GPU indices with a compute process on them, per nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set(range(8))
    busy = set()
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",")]
        if int(used) >= MEM_FREE_MIB:
            busy.add(int(idx))
    return busy


def load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def done_names():
    return {load(p).get("name") for p in (Q / "done").glob("*.json") if load(p)}


def write_status(running):
    status = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "running": [{"name": j["name"], "gpus": j["gpus"],
                     "elapsed_s": int(time.time() - j["t0"])} for j in running],
        "pending": len(list((Q / "pending").glob("*.json"))),
        "done": len(list((Q / "done").glob("*.json"))),
        "failed": len(list((Q / "failed").glob("*.json"))),
    }
    (Q / "status.json").write_text(json.dumps(status, indent=2))


def launch(job, gpus):
    name = job["name"]
    log = LOGS / f"{name}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env.update(job.get("env", {}))
    cmd = f'source /home/hwanhee/anaconda3/etc/profile.d/conda.sh && conda activate tokenhsi && {job["cmd"]}'
    fh = open(log, "w")
    fh.write(f"# {name}\n# gpus={gpus}\n# cmd={job['cmd']}\n# start={time.strftime('%F %T')}\n\n")
    fh.flush()
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=job.get("cwd", str(ROOT)),
                         env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return {"name": name, "gpus": gpus, "proc": p, "fh": fh, "log": str(log),
            "t0": time.time(), "job": job}


def finish(run, code):
    run["fh"].close()
    rec = dict(run["job"])
    rec.update({"exit_code": code, "log": run["log"],
                "seconds": int(time.time() - run["t0"]),
                "finished": time.strftime("%F %T")})
    dest = Q / ("done" if code == 0 else "failed") / f"{run['name']}.json"
    dest.write_text(json.dumps(rec, indent=2))
    print(f"[{time.strftime('%T')}] {run['name']} exit={code} "
          f"({rec['seconds']}s) -> {dest.parent.name}", flush=True)


def main():
    dirs()
    running = []
    print(f"[{time.strftime('%T')}] runner up", flush=True)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))

    while not stop["v"]:
        for run in list(running):
            code = run["proc"].poll()
            to = run["job"].get("timeout")
            if code is None and to and time.time() - run["t0"] > to:
                os.killpg(os.getpgid(run["proc"].pid), signal.SIGKILL)
                code = 124
            if code is not None:
                finish(run, code)
                running.remove(run)

        held = {g for r in running for g in r["gpus"]}
        free = [g for g in range(8) if g not in gpu_busy() and g not in held]

        if free:
            jobs = []
            for p in sorted((Q / "pending").glob("*.json")):
                j = load(p)
                if j:
                    jobs.append((j.get("priority", 100), p, j))
            jobs.sort(key=lambda x: (x[0], x[1].name))
            finished = done_names()
            for _, p, j in jobs:
                need = int(j.get("gpus", 1))
                if need > len(free):
                    continue
                if j.get("after") and j["after"] not in finished:
                    continue
                take, free = free[:need], free[need:]
                p.unlink()
                (Q / "running" / p.name).write_text(json.dumps(j, indent=2))
                running.append(launch(j, take))
                print(f"[{time.strftime('%T')}] start {j['name']} gpus={take}", flush=True)
                if not free:
                    break

        write_status(running)
        time.sleep(POLL)

    for run in running:
        run["proc"].terminate()


if __name__ == "__main__":
    sys.exit(main())
