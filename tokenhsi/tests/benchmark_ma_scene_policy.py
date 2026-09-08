"""CUDA actor-forward microbenchmark for legacy, old-Geo, and GTA scene policies.

Run from the repository root, for example:

    PYTHONPATH=tokenhsi python tokenhsi/tests/benchmark_ma_scene_policy.py
"""

import argparse
import copy
from pathlib import Path

import torch
import yaml

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder


def _build_network(mode, num_agents, num_objects, input_width, device):
    config_path = Path(__file__).resolve().parents[1] \
        / "data/cfg/train/rlg/amp_ma_carry.yaml"
    with config_path.open() as stream:
        config = copy.deepcopy(yaml.safe_load(stream)["params"]["network"])

    transformer = config["transformer"]
    transformer["gta"]["diagnostics_first_forward"] = False
    if mode == "geo":
        transformer["gta"]["enable"] = False
        transformer["geometry"]["enable"] = True

    observation_mode = "legacy_multirow" if mode == "legacy" else "clean_scene"
    pose_size = 13 if mode == "geo" else 7
    builder = AMPMultiAgentBuilder()
    builder.load(config)
    return builder.build(
        "{}_benchmark".format(mode),
        actions_num=32,
        input_shape=(input_width,),
        amp_input_shape=(1290,),
        value_size=1,
        num_agents=num_agents,
        num_objects=num_objects,
        humanoid_obs_size=230,
        object_obs_size=39,
        goal_obs_size=6,
        observation_mode=observation_mode,
        scene_entity_sizes=[223, 30, 1],
        scene_kinematic_size=pose_size,
        scene_arena_scale=5.0,
        device=device,
    ).to(device).eval()


def _input(mode, scenes, num_agents, num_objects, device):
    tokens = 2 * num_agents + num_objects
    if mode == "legacy":
        width = num_agents * 230 + num_objects * 39 + num_agents * 6
        return torch.randn(scenes * num_agents, width, device=device), width

    node_width = num_agents * 223 + num_objects * 30 + num_agents
    pose_size = 13 if mode == "geo" else 7
    width = node_width + tokens * pose_size
    obs = torch.randn(scenes, width, device=device)
    obs[:, num_agents * 223 + num_objects * 30:node_width] = 1.0
    records = obs[:, node_width:].view(scenes, tokens, pose_size)
    records[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    return obs, width


def _measure(mode, scenes, num_agents, num_objects, warmup, iterations, device):
    torch.cuda.empty_cache()
    obs, width = _input(mode, scenes, num_agents, num_objects, device)
    network = _build_network(mode, num_agents, num_objects, width, device)

    with torch.inference_mode():
        for _ in range(warmup):
            network.eval_actor(obs)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            network.eval_actor(obs)
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations
    peak_mib = (torch.cuda.max_memory_allocated() - baseline) / (1024 ** 2)
    del network, obs
    torch.cuda.empty_cache()
    return elapsed_ms, peak_mib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    device = "cuda:0"
    print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print("scenes={} warmup={} iterations={}".format(
        args.scenes, args.warmup, args.iterations))
    print("M O legacy_ms geo_ms gta_ms legacy_peak_MiB geo_peak_MiB gta_peak_MiB")
    for num_agents, num_objects in ((1, 1), (2, 3), (3, 4), (4, 5)):
        results = [
            _measure(mode, args.scenes, num_agents, num_objects,
                     args.warmup, args.iterations, device)
            for mode in ("legacy", "geo", "gta")
        ]
        print("{} {} {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f}".format(
            num_agents, num_objects,
            results[0][0], results[1][0], results[2][0],
            results[0][1], results[1][1], results[2][1]))


if __name__ == "__main__":
    main()
