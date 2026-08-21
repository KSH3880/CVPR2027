#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from pathlib import Path

from queue_docs import (ROOT, atomic_write, load_state, load_track, locked, md_escape,
                        replace_block)


RESULTS = ROOT / "runs/queue/results.tsv"
HEADER = "run\tsuccess\tapp_xt\tapp_R2\tcar_xt\tcar_R2\tpick\tdeliv"


def _read_results():
    rows = {}
    if not RESULTS.exists():
        return rows
    for line in RESULTS.read_text().splitlines():
        fields = line.split("\t")
        if not fields or fields[0] == "run" or len(fields) != 8:
            continue
        rows.setdefault(fields[0], []).append(fields)
    return rows


def result_tags():
    return set(_read_results())


def append(fields):
    if len(fields) != 8:
        raise ValueError("result row must have eight columns")
    line = "\t".join(fields)
    added = False
    with locked("results"):
        existing = _read_results()
        if fields[0] in existing:
            old = ["\t".join(row) for row in existing[fields[0]]]
            if line not in old:
                raise ValueError(f"result for {fields[0]} already exists with different values")
        else:
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            if not RESULTS.exists() or not RESULTS.read_text().strip():
                RESULTS.write_text(HEADER + "\n")
            with RESULTS.open("a") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            added = True
        refresh("steer", results_locked=True)
    return added


def _finished_time(track, tag, record):
    value = record.get("finished_at", "")
    if value:
        return value.replace("T", " ")[:16]
    prefix = "eval_" if track == "steer" else ""
    log = ROOT / "runs/queue/logs" / f"{prefix}{tag}.log"
    if log.exists():
        return datetime.fromtimestamp(log.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M")
    return "—"


def _finished_sort_key(track, tag, record):
    value = record.get("finished_at", "")
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    prefix = "eval_" if track == "steer" else ""
    log = ROOT / "runs/queue/logs" / f"{prefix}{tag}.log"
    return log.stat().st_mtime if log.exists() else float("-inf")


def _steer_summary(fields):
    return (f"success={fields[1]}, app_xt={fields[2]}, car_xt={fields[4]}, "
            f"pick={fields[6]}, deliv={fields[7]}")


def refresh(name, results_locked=False):
    if name == "steer" and not results_locked:
        with locked("results"):
            return refresh(name, results_locked=True)
    track = load_track(name)
    path = track.get("auto_results")
    if not path:
        return
    state = load_state(track)
    jobs = state.get("jobs", {})
    rows = []
    seen = set()
    if name == "steer":
        for tag, result_rows in _read_results().items():
            record = jobs.get(tag, {})
            for index, fields in enumerate(result_rows, 1):
                status = "완료" if len(result_rows) == 1 else f"중복 원본 {index}/{len(result_rows)}"
                rows.append((_finished_sort_key(name, tag, record),
                             _finished_time(name, tag, record), status, tag,
                             _steer_summary(fields), record.get("note", "—")))
            seen.add(tag)
    for tag, record in jobs.items():
        if tag in seen:
            continue
        status = record.get("status", "완료")
        if status not in ("완료", "실패"):
            continue
        rows.append((_finished_sort_key(name, tag, record),
                     _finished_time(name, tag, record), status, tag,
                     record.get("summary") or record.get("detail") or "—",
                     record.get("note", "—")))
    rows.sort(key=lambda row: row[0], reverse=True)
    source = ("`runs/queue/results.tsv`와 트랙 상태 JSON" if name == "steer" else
              "트랙 상태 JSON과 실행 로그의 종료 요약")
    lines = [
        f"# {name} 자동 결과 원장",
        "",
        f"> 자동 생성 파일이다. 직접 수정하지 않는다. 원시 기준은 {source}이다.",
        "> 연구적 비교와 확정 결론은 해당 `EXPERIMENTS.md`에 따로 기록한다.",
        "",
        "| 완료 시각 | 상태 | tag | 자동 요약 | 큐 질문 |",
        "|---|---|---|---|---|",
    ]
    for _, finished, status, tag, summary, note in rows:
        lines.append(f"| {md_escape(finished)} | {md_escape(status)} | `{md_escape(tag)}` | "
                     f"{md_escape(summary)} | {md_escape(note)} |")
    if not rows:
        lines.append("| — | — | — | 아직 전환 이후 완료된 결과 없음 | — |")
    atomic_write(path, "\n".join(lines) + "\n")
    curated = track.get("curated_results")
    if curated:
        curated = ROOT / curated
        block = [
            "<!-- AUTO_RESULTS -->",
            "",
            "## 자동 결과 (최신순)",
            "",
            "> 자동 생성 영역이다. 최신 결과가 위, 가장 오래된 결과가 아래다.",
            "> 비교·채택·기각 해석은 이 블록 아래의 수동 기록에 남긴다.",
            "",
            *lines[5:],
            "",
            "<!-- /AUTO_RESULTS -->",
        ]
        replace_block(curated, "<!-- AUTO_RESULTS -->", "<!-- /AUTO_RESULTS -->",
                      "\n".join(block))


def main():
    names = sys.argv[1:] or ["steer", "ma", "lab"]
    for name in names:
        refresh(name)


if __name__ == "__main__":
    main()
