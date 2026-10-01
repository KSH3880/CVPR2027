"""Generate a carryWith-only scene config for stop/continue experiments."""

import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=360)
    args = parser.parse_args()
    if args.envs < 1 or args.steps < 30:
        parser.error("envs must be positive and steps must be at least 30")
    cfg = yaml.safe_load(Path(args.source).read_text())
    env = cfg["env"]
    env.update(numAgents=2, numEnvs=args.envs, envSpacing=5, episodeLength=args.steps,
               enableDebugVis=False, enableIET=False)
    env["taskInitProb"] = [1.0 if name == "carry" else 0.0 for name in env["task"]]
    carry = env["carry"]
    carry.update(skill=["carryWith"], skillInitProb=[1.0], enableIET=False)
    carry.setdefault("eval", {}).update(skill=["carryWith"], skillInitProb=[1.0], enableIET=False)
    Path(args.output).write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
