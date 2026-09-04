#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/hwanhee/CVPR2027")
sys.path.insert(0, str(ROOT / "scripts"))

import autofill

errors = []
warnings = []


def require(ok, message):
    if not ok:
        errors.append(message)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        errors.append(f"{path}: {e}")
        return {}


layout_path = ROOT / "runs/queue/layout.json"
layout = load_json(layout_path)
require(layout.get("schema_version") == 2, "layout schema_version must be 2")
require(layout.get("operations") == "docs/OPERATIONS.md",
        "layout operations must be docs/OPERATIONS.md")
master_rel = layout.get("master_plan")
require(master_rel == "plan/PLAN.md", "layout master_plan must be plan/PLAN.md")
master = ROOT / master_rel if isinstance(master_rel, str) else None
require(master is not None and master.is_file(), f"master PLAN missing: {master}")
if master and master.is_file():
    text = master.read_text()
    require(text.count("<!-- GPUMAP -->") == 1, f"{master}: GPUMAP start marker must appear once")
    require(text.count("<!-- /GPUMAP -->") == 1, f"{master}: GPUMAP end marker must appear once")
    require(text.count("<!-- QUEUE_SUMMARY -->") == 1,
            f"{master}: QUEUE_SUMMARY start marker must appear once")
    require(text.count("<!-- /QUEUE_SUMMARY -->") == 1,
            f"{master}: QUEUE_SUMMARY end marker must appear once")

entries = {
    ROOT / "AGENTS.md": "docs/OPERATIONS.md",
    ROOT / "CLAUDE.md": "docs/OPERATIONS.md",
    ROOT / "TokenHSI-steer/AGENTS.md": "../docs/OPERATIONS.md",
    ROOT / "TokenHSI-steer/CLAUDE.md": "../docs/OPERATIONS.md",
    ROOT / "TokenHSI-ma/AGENTS.md": "../docs/OPERATIONS.md",
    ROOT / "TokenHSI-ma/CLAUDE.md": "../docs/OPERATIONS.md",
}
require((ROOT / "docs/OPERATIONS.md").is_file(), "docs/OPERATIONS.md missing")
for path, link in entries.items():
    require(path.is_file(), f"entry document missing: {path}")
    if path.is_file():
        require(link in path.read_text(), f"{path}: missing OPERATIONS link {link}")

queue_paths = set()
auto_paths = set()
expected_paths = {
    "steer": ("plan/PLAN_steer.md", "experiments/EXPERIMENTS_steer.md"),
    "ma": ("plan/PLAN_ma.md", "experiments/EXPERIMENTS_ma.md"),
    "lab": ("plan/PLAN_lab.md", "experiments/EXPERIMENTS_lab.md"),
}
for name in ("steer", "ma", "lab"):
    cfg_path = ROOT / f"runs/queue/tracks/{name}.json"
    cfg = load_json(cfg_path)
    if not cfg:
        continue
    queue_rel, curated_rel = expected_paths[name]
    require(cfg.get("queue") == queue_rel, f"{name}: queue must be {queue_rel}")
    require(cfg.get("curated_results") == curated_rel,
            f"{name}: curated_results must be {curated_rel}")
    try:
        track = autofill.load_track(name)
    except Exception as e:
        errors.append(f"{cfg_path}: {e}")
        continue

    require(track.get("name") == name, f"{cfg_path}: name must be {name}")
    require(track["queue"].is_file(), f"{name}: queue missing: {track['queue']}")
    require(track.get("auto_results") and track["auto_results"].is_file(),
            f"{name}: auto results missing: {track.get('auto_results')}")
    require(str(track.get("auto_results")) not in auto_paths,
            f"{name}: auto results path reused: {track.get('auto_results')}")
    auto_paths.add(str(track.get("auto_results")))
    require(str(track["queue"]) not in queue_paths, f"{name}: queue path reused: {track['queue']}")
    queue_paths.add(str(track["queue"]))
    require((ROOT / track["train"]).is_file(), f"{name}: train script missing: {track['train']}")
    if track.get("eval"):
        require((ROOT / track["eval"]).is_file(), f"{name}: eval script missing: {track['eval']}")
    if track.get("curated_results"):
        require((ROOT / track["curated_results"]).is_file(),
                f"{name}: curated results missing: {track['curated_results']}")

    max_per_gpu = int(track.get("max_per_gpu", autofill.MAX_PER_GPU))
    require(max_per_gpu == 2, f"{name}: max_per_gpu must be 2, got {max_per_gpu}")
    require(float(track.get("idle_min", autofill.IDLE_MIN)) == 5,
            f"{name}: idle_min must be 5")
    require(int(track.get("max_concurrent", 99)) >= 1, f"{name}: max_concurrent must be positive")
    require(int(track.get("max_seeds", 2)) <= 2, f"{name}: max_seeds exceeds the default limit 2")

    allowed = track.get("allowed_env", [])
    forbidden = track.get("forbidden_env", [])
    require(len(allowed) == len(set(allowed)), f"{name}: duplicate allowed_env entry")
    require("QUEUE_EXPECT_RC" in allowed, f"{name}: QUEUE_EXPECT_RC must be allowed")
    if track.get("eval"):
        require("QUEUE_EVAL" in allowed, f"{name}: QUEUE_EVAL must be allowed with eval")
    require(not set(allowed) & set(forbidden), f"{name}: env cannot be both allowed and forbidden")
    if track.get("seed_env"):
        require(track["seed_env"] in allowed, f"{name}: seed_env must be in allowed_env")
    parsed_defaults = {}
    try:
        parsed_defaults = autofill.parse_env(track, "base_env", track.get("base_env", ""))
    except Exception as e:
        errors.append(f"{name}: {e}")

    expected_defaults = {
        "steer": {"STEER_ENVS": "4096", "STEER_ENVSPACING": "5", "STEER_CP": "4"},
        "ma": {"MA_AGENTS": "2", "MA_ENVS": "2048", "MA_SPACING": "5", "MA_CP": "4"},
        "lab": {"LAB_ENVS": "4096", "LAB_SPACING": "5", "LAB_CP": "4"},
    }[name]
    require(all(parsed_defaults.get(key) == value for key, value in expected_defaults.items()),
            f"{name}: execution profile defaults do not match PLAN")

    gpus = track.get("gpus")
    require(isinstance(gpus, list) and gpus, f"{name}: gpus must be a non-empty list")
    if isinstance(gpus, list):
        require(len(gpus) == len(set(gpus)), f"{name}: duplicate GPU permission")
        require(all(isinstance(g, int) and 0 <= g <= 7 for g in gpus),
                f"{name}: gpus must contain unique indices 0..7")

    if track["queue"].is_file():
        try:
            autofill.read_queue(track)
        except Exception as e:
            errors.append(f"{name}: {e}")
        queue_text = track["queue"].read_text()
        if queue_text.count("<!-- QUEUE -->") == 1 and queue_text.count("<!-- /QUEUE -->") == 1:
            queue_body = queue_text.split("<!-- QUEUE -->", 1)[1].split("<!-- /QUEUE -->", 1)[0]
            table_lines = [line.strip() for line in queue_body.splitlines()
                           if line.strip().startswith("|")]
            header = ([cell.strip() for cell in table_lines[0].strip("|").split("|")]
                      if table_lines else [])
            require(header == ["tag", "질문·판정 기준", "상태/GPU", "환경변수"],
                    f"{name}: queue columns must be tag/question/status/env")

for path in (ROOT / "scripts/queue_docs.py", ROOT / "scripts/queue_results.py",
             ROOT / "scripts/queue_status.py", ROOT / "scripts/slot_runner.py",
             ROOT / "scripts/eval_runner.py"):
    require(path.is_file(), f"queue management script missing: {path}")

gpumap = ROOT / "scripts/gpumap.sh"
require("queue_status.py" in gpumap.read_text(), "gpumap.sh must delegate to queue_status.py")

deprecated = [
    "PLAN.md", "PLAN_steer.md", "PLAN_ma.md", "PLAN_lab.md",
    "TokenHSI-steer/EXPERIMENTS.md", "TokenHSI-ma/EXPERIMENTS.md",
    "docs/BENCHMARK.md", "docs/HANDOFF_ma.md", "docs/RESTRUCTURE_PLAN.md",
    "docs/archive", "scripts/archive",
    "scripts/steer/watch2.sh", "scripts/steer/watch_intermediate.sh",
]
for relative in deprecated:
    require(not (ROOT / relative).exists(), f"deprecated path must stay removed: {relative}")

active_docs = [
    ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / "REPOS.md",
    ROOT / "docs/OPERATIONS.md",
    ROOT / "plan/PLAN.md", ROOT / "plan/PLAN_steer.md",
    ROOT / "plan/PLAN_ma.md", ROOT / "plan/PLAN_lab.md",
    ROOT / "experiments/EXPERIMENTS.md",
    ROOT / "experiments/EXPERIMENTS_steer.md",
    ROOT / "experiments/EXPERIMENTS_ma.md",
    ROOT / "experiments/EXPERIMENTS_lab.md",
]
for path in active_docs:
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        target = target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        require((path.parent / target).resolve().exists(), f"{path}: broken link {target}")

for name in ("steer", "ma", "lab"):
    path = ROOT / f"experiments/EXPERIMENTS_{name}.md"
    text = path.read_text()
    require(text.count("<!-- AUTO_RESULTS -->") == 1,
            f"{path}: AUTO_RESULTS start marker must appear once")
    require(text.count("<!-- /AUTO_RESULTS -->") == 1,
            f"{path}: AUTO_RESULTS end marker must appear once")

results = ROOT / "runs/queue/results.tsv"
if results.is_file():
    seen = {}
    for number, line in enumerate(results.read_text().splitlines(), 1):
        fields = line.split("\t")
        require(len(fields) == 8, f"results.tsv:{number}: expected 8 columns, got {len(fields)}")
        if not fields or fields[0] == "run":
            continue
        old = seen.setdefault(fields[0], line)
        if old != line:
            warnings.append(f"results.tsv: conflicting duplicate tag {fields[0]}")

if errors:
    for error in errors:
        print(f"CHECK_DOCS ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)

for warning in sorted(set(warnings)):
    print(f"CHECK_DOCS WARN: {warning}", file=sys.stderr)

print(f"CHECK_DOCS OK: master={master_rel}, tracks=steer,ma,lab")
