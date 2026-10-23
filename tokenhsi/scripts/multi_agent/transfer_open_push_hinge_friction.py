"""Explicitly transfer revision5 training state to the screened friction0.5 variant.

Run from the repository root through transfer_open_push_hinge_friction.sh.
This changes physics metadata, not learned parameters or optimizer state.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from utils.open_push_rsi import bank_contract

SOURCE_CONFIG = Path("tokenhsi/data/cfg/multi_agent/open_push_late_rsi.yaml")
TARGET_CONFIG = Path("tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction.yaml")


def transfer_contract(contract, source, target, old_bank, new_bank, old_hash, new_hash):
    expected = copy.deepcopy(source)
    expected["door"]["hinge"]["friction"] = 0.5
    expected["openPushRSI"]["bank_file"] = target["openPushRSI"]["bank_file"]
    if source["door"]["hinge"]["friction"] != 0.0 or expected != target:
        raise ValueError("Only hinge friction0->0.5 and bank path may change")
    expected_bank = copy.deepcopy(old_bank["contract"])
    expected_bank["door"]["hinge"]["friction"] = 0.5
    if (expected_bank != new_bank["contract"] or
            old_bank["candidates"] != new_bank["candidates"] or
            old_bank.get("screen") != "PD_12_steps" or
            new_bank.get("screen") != "PD_12_steps"):
        raise ValueError("Screened banks must differ only in hinge friction")
    if (contract.get("revision") != 5 or
            contract.get("open_reward_contract") != "rsi_baseline_plus_hand_angle_v1" or
            contract.get("push_reward_contract") != "rsi_baseline_plus_hand_distance_midpoint_xy_v2" or
            contract.get("rsi") != source["openPushRSI"] or
            contract.get("rsi_bank_sha256") != old_hash or
            contract.get("door") != source["door"] or
            contract.get("hold") != source["doorHold"] or
            contract.get("push_box") != source["pushBox"] or
            contract.get("skill") != source["skill"] or
            contract.get("init") != source["skillInitProb"] or
            contract.get("expert_mapping") != source["ampExpertMapping"] or
            contract.get("spawn_distance_range") != source["doorSpawnDistanceRange"]):
        raise ValueError("Checkpoint must match the original revision5 late RSI experiment")
    result = copy.deepcopy(contract)
    result["door"] = copy.deepcopy(target["door"])
    result["rsi"] = copy.deepcopy(target["openPushRSI"])
    result["rsi_bank_sha256"] = new_hash
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Output must be a new file; source checkpoint is preserved")
    source = yaml.safe_load(SOURCE_CONFIG.read_text())["env"]
    target = yaml.safe_load(TARGET_CONFIG.read_text())["env"]
    banks = []
    hashes = []
    for env in (source, target):
        raw = Path(env["openPushRSI"]["bank_file"]).read_bytes()
        bank = json.loads(raw)
        runtime_env = dict(env, motion_file="tokenhsi/data/dataset_open_push.yaml")
        if bank["contract"] != bank_contract(runtime_env):
            raise ValueError("Bank does not match config/reference files")
        banks.append(bank)
        hashes.append(hashlib.sha256(raw).hexdigest())
    weights = torch.load(args.checkpoint, map_location="cpu")
    if weights.get("amp_enabled") is not True:
        raise ValueError("AMP-enabled source checkpoint required")
    for key in ("model", "optimizer", "epoch", "frame", "running_mean_std", "amp_input_mean_std"):
        if key not in weights:
            raise ValueError("Training state missing: " + key)
    weights["open_push_contract"] = transfer_contract(
        weights["open_push_contract"], source, target, *banks, *hashes)
    weights["physics_transfer"] = dict(
        source_checkpoint=str(args.checkpoint.resolve()), source_epoch=int(weights["epoch"]),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        change="door.hinge.friction: 0.0 -> 0.5; identical screened RSI candidates",)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as out:
        torch.save(weights, out)
    print(json.dumps(dict(output=str(args.output), epoch=int(weights["epoch"]),
        frame=int(weights["frame"]), transfer=weights["physics_transfer"]), indent=2))


if __name__ == "__main__":
    main()
