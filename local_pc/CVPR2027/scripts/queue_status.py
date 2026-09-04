#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from queue_docs import (ROOT, atomic_write, load_exit, load_state, load_track, locked,
                        log_summary, md_escape, read_queue_rows, replace_block, update_queue)
from queue_results import result_tags


def gpu_snapshot():
    memory = {}
    jobs = {index: [] for index in range(8)}
    process_count = {index: 0 for index in range(8)}
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30).stdout
    for line in out.splitlines():
        index, used = [int(value.strip()) for value in line.split(",")]
        memory[index] = used
    for index in range(8):
        pids = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30).stdout.split()
        process_count[index] = len(pids)
        for pid in pids:
            args = subprocess.run(["ps", "-p", pid, "-o", "args="], capture_output=True,
                                  text=True).stdout
            match = re.search(r"output_path\s+(\S+)", args)
            if not match:
                continue
            raw = Path(match.group(1)).name
            tag = raw[5:] if raw.startswith("_lab_") else raw
            if raw.startswith("_lab_"):
                track = "lab"
            elif "output/steer/" in args:
                track = "steer"
            elif "output/ma/" in args:
                track = "ma"
            else:
                track = "?"
            jobs[index].append({"tag": tag, "track": track, "eval": "--test" in args,
                                "reserved": False})
    return memory, jobs, process_count


def add_reservations(jobs, tracks):
    known = {item["tag"] for entries in jobs.values() for item in entries}
    for track in tracks:
        for tag, record in load_state(track).get("jobs", {}).items():
            gpu = record.get("gpu")
            if (tag in known or record.get("status") not in ("시작 중", "실행 중") or
                    not isinstance(gpu, int) or load_exit(track, tag)):
                continue
            jobs[gpu].append({"tag": tag, "track": track["name"], "eval": False,
                              "reserved": True})


def fingerprint(jobs, tracks, statuses):
    payload = {
        "jobs": [[index, item["track"], item["tag"], item["eval"], item["reserved"]]
                 for index, entries in sorted(jobs.items())
                 for item in sorted(entries, key=lambda value: (value["track"], value["tag"]))],
        "pending": {track["name"]: pending(track) for track in tracks},
        "attention": attention_rows(tracks, jobs),
        "queues": {
            track["name"]: [[row["tag"], row["note"], statuses[track["name"]].get(row["tag"], ""),
                              row["env"]] for row in read_queue_rows(track["queue"])]
            for track in tracks
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def queue_statuses(track, where):
    state = load_state(track)
    started = set(state.get("started", []))
    jobs = state.get("jobs", {})
    results = result_tags() if track["name"] == "steer" else set()
    statuses = {}
    for row in read_queue_rows(track["queue"]):
        tag = row["tag"]
        if tag in where:
            gpu, is_eval = where[tag]
            statuses[tag] = f"gpu{gpu} · {'평가 중' if is_eval else '실행 중'}"
        elif tag not in started:
            statuses[tag] = "대기"
        elif tag in results:
            statuses[tag] = "완료 처리 중"
        elif (ROOT / "runs/queue/gpu_locks" / f"eval_{tag}").exists():
            statuses[tag] = "평가 중"
        elif load_exit(track, tag):
            statuses[tag] = "완료 처리 중"
        elif log_summary(track, tag):
            statuses[tag] = "완료 기록 대기"
        else:
            log = ROOT / "runs/queue/logs" / f"{tag}.log"
            text = log.read_text(errors="ignore") if log.exists() else ""
            if track["name"] == "steer" and "MAX EPOCHS NUM" in text:
                statuses[tag] = "평가 대기"
            elif jobs.get(tag, {}).get("status") in ("시작 중", "실행 중"):
                statuses[tag] = "시작 중"
            else:
                statuses[tag] = "확인 필요"
    return statuses


def pending(track):
    started = set(load_state(track).get("started", []))
    return [row["tag"] for row in read_queue_rows(track["queue"]) if row["tag"] not in started]


def attention_rows(tracks, gpu_jobs=None):
    rows = []
    queue_tags = {track["name"]: {row["tag"] for row in read_queue_rows(track["queue"])}
                  for track in tracks}
    for track in tracks:
        for tag, record in load_state(track).get("jobs", {}).items():
            if record.get("status") == "실패" and not record.get("acknowledged"):
                rows.append((track["name"], tag, record.get("detail") or record.get("summary") or "원인 확인 필요"))
    if gpu_jobs:
        for entries in gpu_jobs.values():
            for job in entries:
                track = job["track"]
                if (track in queue_tags and not job["eval"] and not job["reserved"] and
                        job["tag"] not in queue_tags[track]):
                    rows.append((track, job["tag"], "실행 중이지만 트랙 활성 큐에 없음"))
    return rows


def dashboard_block(memory, jobs, process_count, tracks):
    lines = [
        "<!-- GPUMAP -->",
        "",
        f"`{datetime.now().strftime('%m-%d %H:%M')}` 기준 · 실행 시작/종료 때 자동 갱신",
        "",
        "| GPU | 메모리 | 프로세스 | 트랙 | 올라가 있는 것 |",
        "|---|---|---|---|---|",
    ]
    for index in range(8):
        entries = sorted(jobs[index], key=lambda item: (item["track"], item["tag"]))
        track_names = " ".join(sorted(set(item["track"] for item in entries))) or "—"
        labels = " ".join(f"`{item['tag']}`" +
                          ("(eval)" if item["eval"] else "(starting)" if item["reserved"] else "")
                          for item in entries)
        lines.append(f"| gpu{index} | {memory.get(index, 0)} M | {process_count[index]} | "
                     f"{track_names} | {labels or '_(빈 카드)_'} |")
    lines.extend(["", "| 트랙 | 대기 | 다음 |", "|---|---|---|"])
    for track in tracks:
        queued = pending(track)
        next_tags = " ".join(f"`{tag}`" for tag in queued[:4])
        if len(queued) > 4:
            next_tags += " …"
        lines.append(f"| {track['name']} | {len(queued)} | {next_tags or '—'} |")
    attention = attention_rows(tracks, jobs)
    lines.extend(["", "### NEEDS ATTENTION", ""])
    if attention:
        lines.extend(["| 트랙 | tag | 상태 |", "|---|---|---|"])
        for track, tag, detail in attention[-10:]:
            lines.append(f"| {track} | `{tag}` | {detail.replace('|', '/')} |")
    else:
        lines.append("없음.")
    lines.extend(["", "<!-- /GPUMAP -->"])
    return "\n".join(lines)


def queue_summary_block(tracks, statuses):
    lines = [
        "<!-- QUEUE_SUMMARY -->",
        "",
        "| 트랙 | tag | 질문·판정 기준 | 상태/GPU | 환경변수 |",
        "|---|---|---|---|---|",
    ]
    for track in tracks:
        rows = read_queue_rows(track["queue"])
        if not rows:
            lines.append(f"| [{track['name']}](PLAN_{track['name']}.md) | — | 활성 큐 없음 | — | — |")
            continue
        for row in rows:
            env = "—" if row["env"] in ("", "-", "—") else f"`{md_escape(row['env'])}`"
            status = statuses[track["name"]].get(row["tag"], row["status"] or "—")
            lines.append(
                f"| [{track['name']}](PLAN_{track['name']}.md) | `{row['tag']}` | "
                f"{md_escape(row['note'])} | {md_escape(status)} | {env} |")
    lines.extend(["", "<!-- /QUEUE_SUMMARY -->"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--auto-write", action="store_true")
    args = parser.parse_args()
    write_mode = args.write or args.auto_write
    context = locked("status") if write_mode else nullcontext()
    with context:
        tracks = [load_track(name) for name in ("steer", "ma", "lab")]
        memory, jobs, process_count = gpu_snapshot()
        add_reservations(jobs, tracks)
        where = {}
        for index, entries in jobs.items():
            for item in entries:
                where[item["tag"]] = (index, item["eval"])
        print(f"{'GPU':5s} {'메모리':7s} {'잡':4s} 올라가 있는 것")
        print("-" * 76)
        for index in range(8):
            labels = " ".join(f"{item['track']}:{item['tag']}" +
                              ("(starting)" if item["reserved"] else "") for item in jobs[index])
            print(f"gpu{index:<1d} {memory.get(index, 0):5d}M  {process_count[index]:<4d} {labels}")
        print()
        for track in tracks:
            queued = pending(track)
            print(f"{track['name']:<6s} 대기 {len(queued)} {' '.join(queued[:4])}")
        if not write_mode:
            return
        statuses = {track["name"]: queue_statuses(track, where) for track in tracks}
        current_fingerprint = fingerprint(jobs, tracks, statuses)
        fingerprint_path = ROOT / "runs/queue/dashboard_fingerprint.json"
        if (args.auto_write and fingerprint_path.exists() and
                fingerprint_path.read_text() == current_fingerprint):
            return
        for track in tracks:
            update_queue(track["queue"], statuses=statuses[track["name"]])
        layout = json.loads((ROOT / "runs/queue/layout.json").read_text())
        master = ROOT / layout["master_plan"]
        replace_block(master, "<!-- GPUMAP -->", "<!-- /GPUMAP -->",
                      dashboard_block(memory, jobs, process_count, tracks))
        replace_block(master, "<!-- QUEUE_SUMMARY -->", "<!-- /QUEUE_SUMMARY -->",
                      queue_summary_block(tracks, statuses))
        atomic_write(fingerprint_path, current_fingerprint)


if __name__ == "__main__":
    main()
