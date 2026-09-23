#!/usr/bin/env python3
"""Re-run C5 CEM on exactly the states stored by a prior oracle run."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-masteer"))

from coordinator.schema import CoordinatorState
from coordinator_c5_v2 import (
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    MANEUVER_SLOW,
)


from coordinator_c5_v2.quality_teacher import QualityCEMConfig, QualityCEMTeacher

MODES = (MANEUVER_LEFT, MANEUVER_RIGHT, MANEUVER_SLOW)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu")
    old = source["summary"]
    teacher = QualityCEMTeacher(config=QualityCEMConfig(
        population=old["population"],
        elites=old["elites"],
        iterations=old["iterations"],
    ))
    generator = torch.Generator().manual_seed(args.seed)
    saved = []
    safe_parts = []
    valid_parts = []
    role_parts = []
    for start in range(0, len(source["batches"]), 3):
        state = CoordinatorState.from_mapping(source["batches"][start]["state"])
        role = source["batches"][start]["role"]
        role_parts.append(role)
        mode_safe = []
        mode_valid = []
        for mode in MODES:
            maneuver = torch.full_like(role, mode)
            result = teacher.search(state, role, maneuver, generator=generator)
            elite_safe = result.diagnostics["safe"] & result.diagnostics["valid"]
            mode_safe.append(result.found_safe)
            mode_valid.append(result.found_valid)
            saved.append({
                "state": state.as_dict(),
                "role": result.role,
                "maneuver": result.maneuver,
                "action": result.action,
                "score": result.score,
                "elite_safe": elite_safe,
            })
        safe_parts.append(torch.stack(mode_safe).any(dim=0))
        valid_parts.append(torch.stack(mode_valid).any(dim=0))

    safe = torch.cat(safe_parts)
    valid = torch.cat(valid_parts)
    role = torch.cat(role_parts)
    summary = dict(old)
    summary.update({
        "seed": args.seed,
        "oracle_safe": int(safe.sum()),
        "oracle_safe_rate": float(safe.float().mean()),
        "unresolved_nonstop": int((~safe).sum()),
        "valid_but_unsafe": int((valid & ~safe).sum()),
        "no_valid_plan": int((~valid).sum()),
        "role0_safe_rate": float(safe[role == 0].float().mean()),
        "role1_safe_rate": float(safe[role == 1].float().mean()),
        "source": str(args.source),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"summary": summary, "batches": saved}, args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
