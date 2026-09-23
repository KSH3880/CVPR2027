#!/usr/bin/env python3
"""Measure and save the non-stop C5 teacher oracle."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-masteer"))

from coordinator_c5_v2 import (
    MANEUVER_LEFT,
    MANEUVER_RIGHT,
    MANEUVER_SLOW,
)
from coordinator_c5_v2.quality_teacher import QualityCEMConfig, QualityCEMTeacher
from coordinator_c5_v2.scenarios import sample_crossing_states


MODES = (MANEUVER_LEFT, MANEUVER_RIGHT, MANEUVER_SLOW)


def duplicate_roles(state):
    index = torch.arange(state.batch_size).repeat_interleave(2)
    return state.index(index), torch.arange(2).repeat(state.batch_size)


def run_batch(teacher, state, role, generator):
    mode_results = []
    found_safe = []
    found_valid = []
    for mode in MODES:
        maneuver = torch.full_like(role, mode)
        result = teacher.search(state, role, maneuver, generator=generator)
        mode_results.append(result)
        found_safe.append(result.found_safe)
        found_valid.append(result.found_valid)
    safe = torch.stack(found_safe).any(dim=0)
    valid = torch.stack(found_valid).any(dim=0)
    return mode_results, safe, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.scenarios <= 0 or args.batch_size <= 0:
        raise ValueError("scenarios and batch-size must be positive")

    generator = torch.Generator().manual_seed(args.seed)
    state, role = duplicate_roles(sample_crossing_states(args.scenarios, generator))
    teacher = QualityCEMTeacher(config=QualityCEMConfig(
        population=args.population,
        elites=args.elites,
        iterations=args.iterations,
    ))
    safe_parts = []
    valid_parts = []
    saved = []
    for start in range(0, state.batch_size, args.batch_size):
        stop = min(start + args.batch_size, state.batch_size)
        batch_state = state.index(slice(start, stop))
        batch_role = role[start:stop]
        results, safe, valid = run_batch(
            teacher, batch_state, batch_role, generator
        )
        safe_parts.append(safe)
        valid_parts.append(valid)
        if args.output:
            for result in results:
                saved.append({
                    "state": batch_state.as_dict(),
                    "role": result.role,
                    "maneuver": result.maneuver,
                    "action": result.action,
                    "score": result.score,
                    "elite_safe": (
                        result.diagnostics["safe"]
                        & result.diagnostics["valid"]
                    ),
                })

    safe = torch.cat(safe_parts)
    valid = torch.cat(valid_parts)
    summary = {
        "base_scenarios": args.scenarios,
        "state_role_pairs": int(safe.numel()),
        "oracle_safe": int(safe.sum().item()),
        "oracle_safe_rate": float(safe.float().mean().item()),
        "unresolved_nonstop": int((~safe).sum().item()),
        "valid_but_unsafe": int((valid & ~safe).sum().item()),
        "no_valid_plan": int((~valid).sum().item()),
        "role0_safe_rate": float(safe[role == 0].float().mean().item()),
        "role1_safe_rate": float(safe[role == 1].float().mean().item()),
        "population": args.population,
        "elites": args.elites,
        "iterations": args.iterations,
        "seed": args.seed,
        "speed_floor_mps": 0.375,
        "stop_action": False,
    }
    print(json.dumps(summary, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"summary": summary, "batches": saved}, args.output)


if __name__ == "__main__":
    main()
