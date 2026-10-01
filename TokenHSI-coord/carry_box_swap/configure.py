"""Generate identical original-Carry configuration for probe groups."""

import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source"); p.add_argument("output")
    p.add_argument("--envs", type=int, required=True)
    p.add_argument("--skill", choices=["loco_carry", "carryWith"], required=True)
    p.add_argument("--steps", type=int, default=600)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.source).read_text())
    env = cfg["env"]
    env.update(numAgents=2, numEnvs=args.envs, envSpacing=5, episodeLength=args.steps,
               enableDebugVis=False, enableIET=True)
    env["taskInitProb"] = [float(name == "carry") for name in env["task"]]
    env["carry"].update(skill=[args.skill], skillInitProb=[1.0], enableIET=False)
    env["carry"]["box"]["reset"]["randomHeight"] = False
    Path(args.output).write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__": main()
