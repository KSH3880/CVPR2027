import argparse
import sys
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "tokenhsi"))
    from utils.mixed_carry_spec import sample_graph
    from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
    from env.tasks.multi_agent.joint_carry_reward import (
        apply_joint_geometry,
        JointCarryRuntime,
        joint_cpa_release,
    )

    config_path = args.code_root / "tokenhsi/data/cfg/multi_agent/stage2.yaml"
    if not config_path.exists():
        config_path = (args.code_root /
            "tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml")
    env = yaml.safe_load(config_path.read_text())["env"]
    config = env["relationReward"]
    torch.set_num_threads(1)
    torch.manual_seed(37)
    results = {}
    for preset in ("joint_carry_at", "joint_carry_ontop", "independent"):
        n = 32
        graph = sample_graph(n, env["relationGraph"], preset=preset)
        objects = torch.randn(n, 4, 13) * 0.1
        objects[:, :, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        sizes = torch.rand(n, 4, 3) * 0.3 + 0.3
        sizes[:, 0] = torch.tensor([0.52, 0.8, 0.4])
        roots = torch.randn(n, 2, 3)
        hands = objects[:, 0, None, None, :3].expand(-1, 2, 2, -1).clone()
        hands[:, 0, :, 1] -= 0.25
        hands[:, 1, :, 1] += 0.25
        hands[n // 2 :] += 0.5
        feet = torch.randn(n, 2, 2, 3)
        goals = torch.randn(n, 2, 3)
        phi, diag = evaluate_interaction_edges(
            hands, feet, roots, objects, sizes, goals, graph, config, 0.89
        )
        phi, gate = apply_joint_geometry(
            phi, diag, hands, objects, sizes, graph, config
        )
        runtime = JointCarryRuntime(n, graph, config, "cpu")
        runtime.coupled_holding.copy_(gate)
        runtime.reset(
            torch.arange(n),
            phi,
            diag["z_error"],
            diag["feet_height_error"],
            diag["region_error"],
        )
        result = runtime.step(
            phi,
            diag["progress"],
            diag["z_error"],
            diag["feet_height_error"],
            diag["region_error"],
        )
        root_states = torch.zeros(n, 2, 13)
        root_states[:, :, :3] = roots
        results[preset] = dict(
            phi=phi,
            diag=diag,
            gate=gate,
            result=result,
            release=joint_cpa_release(root_states, objects, sizes, graph, 1.0),
        )
    if args.record:
        torch.save(results, args.reference)
        return
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)

    def compare(a, b, path=""):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise RuntimeError("Environment fields changed: " + path)
            for key in a:
                compare(a[key], b[key], path + "/" + key)
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=path)
        elif a != b:
            raise RuntimeError("Environment value changed: " + path)

    compare(results, reference)
    print(
        "AT/ON_TOP/independent: geometry, holding gate, reward, reset and CPA release identical"
    )


if __name__ == "__main__":
    main()
