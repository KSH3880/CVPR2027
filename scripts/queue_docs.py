#!/usr/bin/env python3

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/hwanhee/CVPR2027")
QUEUE_START = "<!-- QUEUE -->"
QUEUE_END = "<!-- /QUEUE -->"


def load_track(name):
    path = ROOT / "runs/queue/tracks" / f"{name}.json"
    cfg = json.loads(path.read_text())
    cfg["config_path"] = path
    for key in ("queue", "state", "auto_results"):
        if cfg.get(key):
            cfg[key] = ROOT / cfg[key]
    return cfg


@contextmanager
def locked(name="docs"):
    path = ROOT / "runs/queue" / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o664
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
        directory = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def replace_block(path, start, end, block):
    path = Path(path)
    with locked("docs"):
        original = path.read_text()
        if original.count(start) != 1 or original.count(end) != 1:
            raise ValueError(f"{path}: {start} / {end} markers must appear exactly once")
        before, tail = original.split(start, 1)
        _, after = tail.split(end, 1)
        updated = before + block.rstrip("\n") + after
        if updated != original:
            if path.read_text() != original:
                raise RuntimeError(f"{path}: changed during automatic block update; retry later")
            atomic_write(path, updated)


def _tag(cell):
    cell = cell.strip()
    if len(cell) >= 2 and cell[0] == "`" and cell[-1] == "`":
        return cell[1:-1]
    return cell


def _queue_body(text, path):
    if text.count(QUEUE_START) != 1 or text.count(QUEUE_END) != 1:
        raise ValueError(f"{path}: QUEUE markers must appear exactly once")
    before, tail = text.split(QUEUE_START, 1)
    body, after = tail.split(QUEUE_END, 1)
    return before, body, after


def read_queue_rows(path):
    path = Path(path)
    _, body, _ = _queue_body(path.read_text(), path)
    rows = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        tag = _tag(cells[0]) if cells else ""
        if tag.lower() == "tag" or (tag and set(tag) <= set("-: ")):
            continue
        if len(cells) != 4:
            raise ValueError(f"{path}: {tag or '<empty>'} must have exactly four columns")
        rows.append({"tag": tag, "note": cells[1], "status": cells[2], "env": _tag(cells[3])})
    return rows


def update_queue(path, statuses=None, remove_tags=None):
    path = Path(path)
    statuses = statuses or {}
    remove_tags = set(remove_tags or [])
    removed = []
    with locked("docs"):
        original = path.read_text()
        before, body, after = _queue_body(original, path)
        output = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                output.append(line)
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            tag = _tag(cells[0]) if cells else ""
            if len(cells) != 4:
                raise ValueError(f"{path}: {tag or '<empty>'} must have exactly four columns")
            if tag.lower() == "tag":
                cells[1:] = ["질문·판정 기준", "상태/GPU", "환경변수"]
            elif tag and set(tag) <= set("-: "):
                cells[2] = "---"
            elif tag in remove_tags:
                removed.append({"tag": tag, "note": cells[1], "status": cells[2],
                                "env": _tag(cells[3])})
                continue
            elif tag in statuses:
                cells[2] = statuses[tag]
            output.append("| " + " | ".join(cells) + " |" if stripped.startswith("|") else line)
        new_body = "\n".join(output)
        if not new_body.endswith("\n"):
            new_body += "\n"
        updated = before + QUEUE_START + new_body + QUEUE_END + after
        if updated != original:
            if path.read_text() != original:
                raise RuntimeError(f"{path}: changed during automatic queue update; retry later")
            atomic_write(path, updated)
    return removed


def load_state(track):
    path = track["state"]
    state = json.loads(path.read_text()) if path.exists() else {"started": []}
    state.setdefault("started", [])
    state.setdefault("jobs", {})
    return state


def save_state(track, state):
    atomic_write(track["state"], json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def exit_path(track, tag):
    return ROOT / "runs/queue/exits" / track["name"] / f"{tag}.json"


def load_exit(track, tag):
    path = exit_path(track, tag)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"status": "failed", "rc": 125, "detail": f"invalid exit record: {path}"}


def log_summary(track, tag):
    path = ROOT / "runs/queue/logs" / f"{tag}.log"
    if not path.exists():
        return ""
    prefixes = {
        "ma": ("MA_METRIC ", "MA_SUMMARY "),
        "lab": ("LAB tag=",),
    }.get(track["name"], ())
    found = []
    for line in path.read_text(errors="ignore").splitlines():
        if any(line.startswith(prefix) for prefix in prefixes):
            found.append(line.strip())
    return " · ".join(found[-2:])


def md_escape(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")
