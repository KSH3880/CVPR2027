#!/usr/bin/env python3
"""Puts the next queued experiment on a GPU that has been idle long enough.

    python scripts/autofill.py --track steer --loop
    python scripts/autofill.py --track ma --loop
    python scripts/autofill.py --track steer            # one pass, no loop

One instance per track. Tracks are independent: their own queue file, their own
state file, their own train script, their own output tree. Neither knows the
other exists, and neither is pinned to particular GPUs.

Placement is by load, not ownership:

    1. a GPU running nothing        <- preferred, but only once it has settled
    2. a GPU under MAX_PER_GPU      <- lower-memory cards first, so a large
                                       sequence usually stays alone

Sharing costs per-slot speed (measured: 35.2k -> 18.8k fps, docs/ENV_SCALING.md
section 7) and buys only 7% total throughput, so an empty card always wins. But
refusing to share would stall a track completely whenever the other one happens
to hold every GPU, which is worse than slow.

The settle wait applies only to case 1. A run takes 30-60 s to actually exit --
checkpoint write, atexit dumps, PhysX teardown -- and launching into that window
collides with the previous job's cleanup. A card that already has a live job has
no teardown to collide with, so it is available immediately.

Slots are taken with an O_EXCL create before anything starts, because the idle
test alone is a race between two instances: both poll, both see gpu6 free, both
launch. Each running job holds one gpu<N>.<pid> file; dead holders are reaped, so
a killed job frees its slot without bookkeeping. Hand-run probes take slots the
same way through scripts/ma/gpu.sh.

One slot per pass. Filling every free GPU at once would put several jobs through
their memory-hungry startup simultaneously, and it also removes the chance to
look at what the first one does before committing the rest.
"""

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from queue_docs import (exit_path, load_exit, load_state, load_track as load_track_config,
                        log_summary, now_text, read_queue_rows, save_state, update_queue)
from queue_results import refresh as refresh_results, result_tags

ROOT = Path("/home/hwanhee/CVPR2027")
LOGS = ROOT / "runs/queue/logs"
LOCKS = ROOT / "runs/queue/gpu_locks"
IDLE_MIN = 5.0
MEM_FREE_MIB = 2000        # below this the card counts as running nothing
JOB_MIB = 26000            # one slot (measured 25.7 GB); also the headroom test
MAX_PER_GPU = 2            # 3 fits in memory but sits at 79-99% util already
EVAL_RETRY_MIN = 30
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
TAG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_track(name):
    return load_track_config(name)


def slot_load(track=None):
    """Live slots per GPU, reaping holders that died.

    With `track`, counts only that track's slots -- that is what caps a track to
    one card at a time (`max_concurrent`) without pinning it to a fixed card.

    Memory alone cannot answer this: a job takes ~30 s of IsaacGym startup before
    it allocates anything, so a poll landing in that window sees the card as
    empty and stacks another job on top of it.
    """
    LOCKS.mkdir(parents=True, exist_ok=True)
    load = {}
    for f in LOCKS.glob("gpu*.*"):
        idx = f.name[3:].split(".")[0]
        if not idx.isdigit():
            continue                       # eval_<tag> claims live here too
        try:
            lines = f.read_text().split("\n")
            pid = int(lines[0].strip())
        except Exception:
            f.unlink(missing_ok=True)
            continue
        if not Path(f"/proc/{pid}").exists():
            f.unlink(missing_ok=True)      # holder died: release
            continue
        owner = lines[1].strip() if len(lines) > 1 else ""
        if track is not None and owner != track:
            continue
        load[int(idx)] = load.get(int(idx), 0) + 1
    return load


def claim(gpu, max_per, track):
    """Take one slot on the GPU, or return False if it filled up first.

    O_EXCL on a pid-named file is what makes two instances safe against each
    other; counting and then writing would leave the same window the idle test
    already has. The daemon's own pid goes in first because it is certainly
    alive -- the job's pid replaces it once the job exists.
    """
    LOCKS.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"gpu{gpu}.", dir=LOCKS)
    os.write(fd, f"{os.getpid()}\n{track}".encode())
    os.close(fd)
    path = Path(name)
    if slot_load().get(gpu, 0) > max_per:
        path.unlink(missing_ok=True)       # both instances grabbed it: back off
        return None
    return path


def gpu_slots(max_per_gpu, allow=None, job_mib=JOB_MIB):
    """(idle, shared) -- cards running nothing, and cards with room to share.

    Empty cards are ordered by index. Shared cards are ordered by job count and
    then used memory, so a large one-job run is chosen only after smaller runs.

    `allow` is the track's GPU permission (track cfg "gpus"). Reserving cards for
    a track used to mean emptying the other tracks' queues, which threw the queue
    away to express a scheduling decision. A permission says the same thing
    without touching the queue: the rows stay, they just wait.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    process_count = {}
    for line in apps.strip().splitlines():
        uuid, _ = [value.strip() for value in line.split(",", 1)]
        process_count[uuid] = process_count.get(uuid, 0) + 1
    load = slot_load()
    idle, shared = [], []
    for line in out.strip().splitlines():
        idx_text, uuid, used_text, total_text = [value.strip() for value in line.split(",")]
        idx, used, total = int(idx_text), int(used_text), int(total_text)
        if allow is not None and idx not in allow:
            continue
        # Compute processes count actual jobs. Locks cover the startup window
        # before a process appears; memory catches an untracked non-compute user.
        by_mem = 0 if used < MEM_FREE_MIB else 1
        n = max(load.get(idx, 0), process_count.get(uuid, 0), by_mem)
        if n == 0:
            idle.append(idx)
        elif n < max_per_gpu and (total - used) > job_mib:
            shared.append((n, used, idx))
    return sorted(idle), [g for _, _, g in sorted(shared)]


def _row_values(t, row):
    env = "" if row.get("env") in (None, "—", "-") else row["env"]
    values = parse_env(t, row["tag"], env)
    base = parse_env(t, "base_env", t.get("base_env", ""))
    base.update(values)
    return base


def requires_eval(t, row):
    """Whether this queue row requires a post-training evaluation."""
    if not t.get("eval"):
        return False
    values = _row_values(t, row)
    override = values.get("QUEUE_EVAL", "auto").lower()
    if override in ("1", "true", "yes"):
        return True
    if override in ("0", "false", "no"):
        return False
    if override != "auto":
        raise ValueError(f"{t['queue']}: {row['tag']} has invalid QUEUE_EVAL={override}")
    mode_env = t.get("eval_mode_env")
    modes = t.get("eval_modes")
    return not mode_env or not modes or values.get(mode_env) in modes


def eval_summary(t, tag):
    if t["name"] == "steer" and tag in result_tags():
        return "자동 평가와 원시 결과 저장 완료"
    prefix = t.get("eval_summary_prefix")
    path = LOGS / f"eval_{tag}.log"
    if not prefix or not path.exists():
        return ""
    training_exit = exit_path(t, tag)
    if t["name"] == "ma" and training_exit.exists() and path.stat().st_mtime < training_exit.stat().st_mtime:
        return ""
    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines()
             if line.startswith(prefix)]
    return lines[-1] if lines else ""


def _training_finished(t, tag):
    record = load_exit(t, tag)
    if record and record.get("status") == "training_complete":
        return True
    summary = log_summary(t, tag)
    if summary:
        match = re.search(r"\brc=(\d+)", summary)
        return not match or int(match.group(1)) == 0
    path = LOGS / f"{tag}.log"
    return path.exists() and "MAX EPOCHS NUM" in path.read_text(errors="ignore")


def needs_eval(t):
    """Finished training runs that have not been evaluated yet.

    A run is only a result once it has been evaluated and scored, so this is
    checked BEFORE placing new training: a finished run whose numbers are never
    read is worse than one that never started, because it looks like progress.

    Tracks may restrict final evaluation to training modes. Infrastructure
    probes and explicit QUEUE_EVAL=0 rows finish from their normal summary.
    """
    if not t.get("eval"):
        return []
    out = []
    rows = read_queue_rows(t["queue"])
    processes = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
    required = {row["tag"]: row for row in rows if requires_eval(t, row)}
    # Steer predates the markdown queue lifecycle. Keep finding its historical
    # finished runs until their append-only result rows exist.
    if t["name"] == "steer":
        for log in LOGS.glob(t["log_glob"]):
            tag = log.stem
            if TAG_NAME.fullmatch(tag):
                required.setdefault(tag, {"tag": tag, "env": "—"})
    for tag in sorted(required):
        if _tag_is_live(t, tag, processes):
            continue
        if not _training_finished(t, tag):
            continue
        if eval_summary(t, tag):
            continue
        if (LOCKS / f"eval_{tag}").exists():
            continue
        failure = ROOT / "runs/queue/eval_failures" / t["name"] / f"{tag}.json"
        if failure.exists() and time.time() - failure.stat().st_mtime < EVAL_RETRY_MIN * 60:
            continue
        ck = sorted((ROOT / t["output_dir"] / tag).glob("*/nn/Humanoid.pth"))
        if ck:
            out.append((tag, ck[0], tag))
    for source in t.get("watch_checkpoints", []):
        if not TAG_NAME.fullmatch(source):
            raise ValueError(f"invalid watch_checkpoints tag: {source!r}")
        for ck in sorted((ROOT / t["output_dir"] / source).glob("*/nn/Humanoid_[0-9]*.pth")):
            match = re.fullmatch(r"Humanoid_0*([0-9]+)\.pth", ck.name)
            if not match or time.time() - ck.stat().st_mtime < 60:
                continue
            tag = f"{source}_e{int(match.group(1))}"
            if eval_summary(t, tag) or (LOCKS / f"eval_{tag}").exists():
                continue
            failure = ROOT / "runs/queue/eval_failures" / t["name"] / f"{tag}.json"
            if failure.exists() and time.time() - failure.stat().st_mtime < EVAL_RETRY_MIN * 60:
                continue
            if not (LOGS / f"{source}.env").exists():
                continue
            out.append((tag, ck, source))
    return out


def launch_eval(t, gpu, tag, ck, source, lock):
    """Evaluate, score, and append one line to the results table.

    Goes through the track's eval script, which takes the eval_<tag> claim
    atomically before starting anything. Doing the claim here instead would still
    leave a hand-run eval able to collide with this one, and that is exactly what
    scrambled f3_L0.6's log and npz the second time.
    """
    proc = subprocess.Popen([
        sys.executable, str(ROOT / "scripts/eval_runner.py"), t["name"], tag,
        str(gpu), source, str(ck), str(lock)], start_new_session=True)
    lock.write_text(f"{proc.pid}\n{t['name']}")
    print(f"[{t['name']}] gpu{gpu} <- EVAL {tag}", flush=True)
    subprocess.Popen([sys.executable, str(ROOT / "scripts/queue_status.py"), "--auto-write"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def parse_env(t, tag, env):
    values = {}
    allowed = set(t.get("allowed_env", []))
    forbidden = set(t.get("forbidden_env", []))
    for token in env.split():
        if "=" not in token:
            raise ValueError(f"{t['queue']}: {tag} has invalid env token {token!r}")
        name, value = token.split("=", 1)
        if not ENV_NAME.fullmatch(name) or not value:
            raise ValueError(f"{t['queue']}: {tag} has invalid env token {token!r}")
        if any(char in value for char in ";&|`$<>(){}\\\""):
            raise ValueError(f"{t['queue']}: {tag} has unsafe env value for {name}")
        if name in values:
            raise ValueError(f"{t['queue']}: {tag} repeats {name}")
        if name in forbidden:
            raise ValueError(f"{t['queue']}: {tag} uses forbidden env {name}")
        if allowed and name not in allowed:
            raise ValueError(f"{t['queue']}: {tag} uses unknown env {name}")
        values[name] = value
    return values


def read_queue(t):
    """Pending rows as (tag, env-overrides, note). Started tags are skipped.

    The queue lives in the markdown table between the QUEUE markers in the
    track's plan file rather than in a separate list, so that reading the plan
    and reading what will actually run cannot disagree. Editing the table edits
    the queue.
    """
    started = (set(json.loads(t["state"].read_text()).get("started", []))
               if t["state"].exists() else set())
    if not t["queue"].exists():
        print(f"[{t['name']}] QUEUE FILE MISSING: {t['queue']}", flush=True)
        return [], started
    rows = []
    tags = set()
    seed_groups = {}
    seed_env = t.get("seed_env")
    max_seeds = int(t.get("max_seeds", 2))
    defaults = parse_env(t, "base_env", t.get("base_env", ""))
    for row in read_queue_rows(t["queue"]):
        tag = row["tag"]
        if not TAG_NAME.fullmatch(tag) or tag in tags:
            raise ValueError(f"{t['queue']}: invalid or duplicate tag: {tag!r}")
        tags.add(tag)
        env = row["env"]
        env = "" if env in ("—", "-") else env
        values = parse_env(t, tag, env)
        redundant = sorted(name for name, value in values.items()
                           if defaults.get(name) == value)
        if redundant:
            raise ValueError(
                f"{t['queue']}: {tag} repeats base_env values: {', '.join(redundant)}")
        output_dir = t.get("output_dir")
        # 재개(MA_RESUME=1)는 **자기 출력 디렉터리를 다시 쓰는 것이 목적**이다.
        # 그 외에는 기존 디렉터리를 덮어쓰는 사고를 막는다.
        # 트랙마다 이름이 다르다 (MA_RESUME=1 / STEER_RESUME=<체크포인트 경로>).
        resuming = any(k.endswith("_RESUME") and v not in ("", "0") for k, v in values.items())
        if (tag not in started and output_dir and not resuming
                and (ROOT / output_dir / tag).exists()):
            raise ValueError(
                f"{t['queue']}: {tag} already has an output directory; use a new tag")

        if seed_env:
            config = tuple(sorted((k, v) for k, v in values.items() if k != seed_env))
            seed = values.get(seed_env, "<default>")
            group = seed_groups.setdefault(config, {})
            if seed in group:
                raise ValueError(f"{t['queue']}: {tag} duplicates seed {seed} from {group[seed]}")
            group[seed] = tag
            if len(group) > max_seeds:
                raise ValueError(
                    f"{t['queue']}: setting exceeds max_seeds={max_seeds}: {', '.join(group.values())}")

        if tag not in started:
            rows.append((tag, env, row["note"]))
    return rows, started


def _tag_is_live(t, tag, processes):
    tag_env = t.get("tag_env")
    return ((tag_env and f"{tag_env}={tag}" in processes) or
            f"slot_runner.py {t['name']} {tag} " in processes or
            f"output/{t['name']}/{tag}" in processes)


def _terminal(t, row, started, processes):
    tag = row["tag"]
    if tag not in started:
        return None
    eval_claim = ROOT / "runs/queue/gpu_locks" / f"eval_{tag}"
    if eval_claim.exists() or _tag_is_live(t, tag, processes):
        return None
    if requires_eval(t, row):
        summary = eval_summary(t, tag)
        if summary:
            return {"status": "완료", "summary": summary}
        exit_record = load_exit(t, tag)
        if exit_record and exit_record.get("status") == "failed":
            return {"status": "실패", "detail": exit_record.get("detail") or
                    f"training script rc={exit_record.get('rc')}"}
        training_summary = log_summary(t, tag)
        if training_summary:
            match = re.search(r"\brc=(\d+)", training_summary)
            rc = int(match.group(1)) if match else 0
            values = _row_values(t, row)
            expected = {int(value) for value in values.get("QUEUE_EXPECT_RC", "0").split(",")}
            if rc not in expected:
                return {"status": "실패", "summary": training_summary,
                        "detail": f"training script rc={rc}, expected={sorted(expected)}"}
        return None
    summary = log_summary(t, tag)
    if summary:
        match = re.search(r"\brc=(\d+)", summary)
        rc = int(match.group(1)) if match else 0
        values = parse_env(t, tag, "" if row["env"] in ("—", "-") else row["env"])
        expected = {int(value) for value in values.get("QUEUE_EXPECT_RC", "0").split(",")}
        accepted = rc in expected
        return {"status": "완료" if accepted else "실패", "summary": summary,
                "detail": "" if accepted else f"training script rc={rc}, expected={sorted(expected)}"}
    exit_record = load_exit(t, tag)
    if exit_record and exit_record.get("status") == "failed":
        return {"status": "실패", "detail": exit_record.get("detail") or
                f"training script rc={exit_record.get('rc')}"}
    log = LOGS / f"{tag}.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    if t["name"] == "steer" and "MAX EPOCHS NUM" in text:
        return None
    if log.exists() and time.time() - log.stat().st_mtime > 120:
        return {"status": "실패", "detail": "프로세스가 사라졌고 완료 표식이 없음"}
    return None


def consume_terminal_rows(t, state):
    rows = read_queue_rows(t["queue"])
    started = set(state.get("started", []))
    processes = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                               text=True).stdout
    terminal = {}
    for row in rows:
        record = _terminal(t, row, started, processes)
        if record:
            terminal[row["tag"]] = (row, record)
    if not terminal:
        return False
    jobs = state.setdefault("jobs", {})
    for tag, (row, record) in terminal.items():
        previous = jobs.get(tag, {})
        jobs[tag] = {**previous, **record, "env": row["env"], "note": row["note"],
                     "finished_at": previous.get("finished_at") or now_text()}
    save_state(t, state)
    try:
        refresh_results(t["name"])
    except Exception as error:
        print(f"[{t['name']}] AUTO RESULTS ERROR: {error}; queue rows kept", flush=True)
        return False
    removed = update_queue(t["queue"], remove_tags=terminal)
    for row in removed:
        print(f"[{t['name']}] consumed {row['tag']} -> {t['auto_results']}", flush=True)
    if removed:
        try:
            with open(ROOT / "runs/queue/gpumap.log", "a") as log:
                subprocess.Popen([sys.executable, str(ROOT / "scripts/queue_status.py"), "--auto-write"],
                                 stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
        except Exception as error:
            print(f"[{t['name']}] status refresh failed: {error}", flush=True)
    return bool(removed)


def launch(t, gpu, tag, env, note, lock):
    """Start one training slot, and record what it was started with.

    The sidecar is not optional bookkeeping. Evaluation loads the checkpoint
    into a freshly constructed env, so anything the run relied on that is not
    replayed comes from today's defaults instead of the run's own. That has
    already produced both failure modes: STEER_K=3 crashes on a shape mismatch,
    which is loud, and a changed STEER_TRAJ default silently scored five v1
    policies on v2 paths, which is not. eval_one.sh sources this file.

    The wrapper outlives the training run and drops the GPU lock when it ends,
    and start_new_session detaches it so restarting this daemon does not take the
    training down with it.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    exit_path(t, tag).unlink(missing_ok=True)
    values = parse_env(t, "base_env", t.get("base_env", ""))
    values.update(parse_env(t, tag, env))
    (LOGS / f"{tag}.env").write_text(
        "\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items()) + "\n")
    (LOGS / f"{tag}.env.json").write_text(json.dumps(values, indent=2) + "\n")
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/slot_runner.py"),
                             t["name"], tag, str(gpu), str(lock)], start_new_session=True)
    try:
        lock.write_text(f"{proc.pid}\n{t['name']}")
    except Exception as error:
        print(f"[{t['name']}] lock owner update failed for {tag}: {error}", flush=True)
    print(f"[{t['name']}] gpu{gpu} <- {tag}   {note}", flush=True)
    # 시작 예약을 즉시 표시한다. 실제 GPU 프로세스가 올라온 뒤에는 주기 갱신이
    # starting 표식을 실시간 실행 상태로 바꾼다. 실패해도 학습에는 영향이 없어야 한다.
    try:
        with open(ROOT / "runs/queue/gpumap.log", "a") as log:
            subprocess.Popen([sys.executable, str(ROOT / "scripts/queue_status.py"), "--auto-write"],
                             stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
    except Exception as e:
        print(f"[{t['name']}] gpumap launch failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="steer")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--idle-min", type=float, default=IDLE_MIN)
    ap.add_argument("--max-per-gpu", type=int, default=None)
    a = ap.parse_args()
    daemon_lock_path = ROOT / "runs/queue" / f"autofill_{a.track}.daemon.lock"
    daemon_lock = daemon_lock_path.open("a+")
    try:
        fcntl.flock(daemon_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[{a.track}] another daemon already holds {daemon_lock_path}", flush=True)
        return 3
    idle_since = {}
    last_config = None
    while True:
        try:
            t = load_track(a.track)
            max_per_gpu = int(a.max_per_gpu if a.max_per_gpu is not None
                              else t.get("max_per_gpu", MAX_PER_GPU))
            max_conc = int(t.get("max_concurrent", 99))
            allow = set(t["gpus"]) if t.get("gpus") is not None else None
            idle_min = a.idle_min
            if a.idle_min == IDLE_MIN and t.get("idle_min") is not None:
                idle_min = float(t["idle_min"])
            # 2 is the standing limit: sharing costs 46 % of per-job speed and
            # buys 8 % of card throughput, so a third slot is never worth it for
            # TRAINING. 3 is allowed only because evaluation is a different
            # animal -- 512 envs, ~4 GB against a trainer's 25 GB -- and with all
            # eight cards held by another track a backlog of scored checkpoints
            # otherwise never starts at all. User authorised it 2026-08-21;
            # put it back to 2 once the backlog clears.
            if not 1 <= max_per_gpu <= 3:
                raise ValueError(f"max_per_gpu must be 1..3, got {max_per_gpu}")
            if max_conc < 1 or idle_min < 0:
                raise ValueError("max_concurrent must be positive and idle_min must be non-negative")
            if int(t.get("max_seeds", 2)) > 2:
                raise ValueError(f"max_seeds exceeds the default limit 2: {t.get('max_seeds')}")
            if allow is not None and any(g < 0 or g > 7 for g in allow):
                raise ValueError(f"gpus must contain indices 0..7: {sorted(allow)}")
            parse_env(t, "base_env", t.get("base_env", ""))
            state = load_state(t)
            consumed = consume_terminal_rows(t, state)
            if consumed:
                state = load_state(t)
            rows, started = read_queue(t)
            processes = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                                       text=True).stdout
            adopted = [(tag, env, note) for tag, env, note in rows
                       if _tag_is_live(t, tag, processes)]
            if adopted:
                jobs = state.setdefault("jobs", {})
                for tag, env, note in adopted:
                    started.add(tag)
                    jobs[tag] = {"status": "실행 중", "env": env, "note": note,
                                 "started_at": jobs.get(tag, {}).get("started_at") or now_text()}
                state["started"] = sorted(started)
                save_state(t, state)
                rows = [row for row in rows if row[0] not in {tag for tag, _, _ in adopted}]
        except Exception as e:
            print(f"[{a.track}] CONFIG ERROR: {e}", flush=True)
            if not a.loop:
                return 2
            time.sleep(a.interval)
            continue

        config = (tuple(sorted(allow)) if allow is not None else None,
                  max_per_gpu, max_conc, idle_min, str(t["queue"]))
        if config != last_config:
            print(f"[{t['name']}] gpus={sorted(allow) if allow is not None else 'all'} "
                  f"max_per_gpu={max_per_gpu} max_concurrent={max_conc} "
                  f"idle_min={idle_min} queue={t['queue']}", flush=True)
            last_config = config

        mine = sum(slot_load(t["name"]).values())
        # 트랙마다 잡 하나의 GPU 메모리가 다르다. MA_NOFREEZE 런은 36GB 를 쓰는데
        # 기본 26GB 가정으로는 한 장에 둘을 얹어 72GB 가 되어 OOM 으로 죽는다.
        job_mib = int(t.get("job_mib", JOB_MIB))
        idle, shared = ([], []) if mine >= max_conc else gpu_slots(max_per_gpu, allow, job_mib)
        now = time.time()

        for g in list(idle_since):
            if g not in idle:
                del idle_since[g]        # busy again: the clock restarts
        for g in idle:
            idle_since.setdefault(g, now)

        # an empty card wins, but only after its predecessor has finished tearing
        # down; a card already running something has no teardown to wait on
        settled = [g for g in idle if (now - idle_since[g]) / 60.0 >= idle_min]
        target = settled[0] if settled else (shared[0] if shared else None)

        pending_eval = needs_eval(t)
        if target is not None and (pending_eval or rows):
            lock = claim(target, max_per_gpu, t["name"])
            if lock is None:
                pass                                 # filled up first: next pass
            elif pending_eval:
                # evaluation first: it takes ~5 min against a 3.3 h training slot,
                # and an unevaluated run is not a result
                tag, ck, source = pending_eval[0]
                launch_eval(t, target, tag, ck, source, lock)
                idle_since.pop(target, None)
            else:
                tag, env, note = rows[0]
                previous_job = state.setdefault("jobs", {}).get(tag)
                state["started"] = sorted(started | {tag})
                state["jobs"][tag] = {
                    "status": "실행 중", "started_at": now_text(), "gpu": target,
                    "env": env, "note": note,
                }
                try:
                    save_state(t, state)
                    launch(t, target, tag, env, note, lock)
                except Exception as e:
                    lock.unlink(missing_ok=True)
                    state["started"] = sorted(started)
                    if previous_job is None:
                        state["jobs"].pop(tag, None)
                    else:
                        state["jobs"][tag] = previous_job
                    try:
                        save_state(t, state)
                    except Exception as rollback_error:
                        print(f"[{t['name']}] state rollback failed: {rollback_error}", flush=True)
                    print(f"[{t['name']}] gpu{target} launch failed: {e}", flush=True)
                idle_since.pop(target, None)
        elif (pending_eval or rows) and idle:
            waits = ", ".join(f"gpu{g} {(now - idle_since[g])/60:.1f}m" for g in sorted(idle))
            print(f"[{t['name']}] free but not settled: {waits}  (need {idle_min}m)", flush=True)
        elif not (pending_eval or rows):
            print(f"[{t['name']}] queue empty, {len(idle)} idle / {len(shared)} sharable",
                  flush=True)

        try:
            with open(ROOT / "runs/queue/gpumap.log", "a") as log:
                subprocess.Popen([sys.executable, str(ROOT / "scripts/queue_status.py"),
                                  "--auto-write"], stdout=subprocess.DEVNULL, stderr=log,
                                 start_new_session=True)
        except Exception as error:
            print(f"[{t['name']}] periodic status refresh failed: {error}", flush=True)

        if not a.loop:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
