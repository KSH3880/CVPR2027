"""Relabel visited closed-loop states and split strictly by episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import save_split
from .oracle import solve_oracle
from .schema import SCHEMA_VERSION, PlannerState


def _load(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != "masteer-dagger-state-v1":
        raise ValueError("not a masteer DAgger state file")
    PlannerState.from_mapping(payload["state"])
    return payload


def _subset(payload, indices, oracle):
    state = {key: value[indices] for key, value in payload["state"].items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "count": len(indices),
        "seed": -1,
        "state": state,
        "scenario": payload["scenario"][indices],
        # Group identity is retained as scene_seed so subsequent tooling can
        # audit that one episode never crosses train/validation boundaries.
        "scene_seed": payload["episode_key"][indices],
        **{key: value[indices] for key, value in oracle.as_dict().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../runs/trajectory_predictor/dagger"))
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    payload = _load(args.input)
    state = PlannerState.from_mapping(payload["state"])
    parts = []
    for lo in range(0, state.batch_size, args.batch_size):
        parts.append(solve_oracle(state.index(slice(lo, min(lo + args.batch_size, state.batch_size)))))
    oracle = type(parts[0])(**{
        key: torch.cat([getattr(part, key) for part in parts])
        for key in parts[0].as_dict()
    })

    episodes = payload["episode_key"].long()
    # Stable integer hash; every state from an episode gets the same split.
    bucket = ((episodes * 1103515245 + 12345) & 0x7FFFFFFF) % 10_000
    val = bucket < int(args.val_fraction * 10_000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in (("train", ~val), ("val", val)):
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if len(indices) == 0:
            raise ValueError(f"DAgger {name} split is empty; collect more episodes")
        path = args.output_dir / f"dagger_r{args.round}_{name}.pt"
        save_split(path, _subset(payload, indices, oracle))
        valid = oracle.oracle_valid[indices].float().mean().item()
        print(f"saved {len(indices)} {name} states to {path} (oracle_valid={valid:.4f})")


if __name__ == "__main__":
    main()
