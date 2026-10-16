import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "tokenhsi"))
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    from learning.multi_agent.grasp_view import make_grasp_views
    from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
    from learning.multi_agent.stage2_transfer import transfer_stage1_weights

    torch.set_num_threads(1)
    torch.manual_seed(42)
    obs = torch.randn(8, 634) * 0.2
    poses = obs[:, 568:624].reshape(8, 8, 7)
    poses[..., 3:7] = torch.nn.functional.normalize(poses[..., 3:7], dim=-1)
    obs[:, -10:] = torch.tensor([1, 2, 0, 2, 6, 1, 2, 1, 2, 6])
    obs[2:4, -10:] = torch.tensor([1, 3, 0, 2, 4, 1, 3, 1, 2, 4])
    obs[4:, -10:] = torch.tensor([1, 2, 0, 2, 6, 1, 3, 1, 3, 5])
    stage1 = torch.load(
        args.checkpoint_root
        / "stage1"
        / "ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth",
        map_location="cpu",
        weights_only=False,
    )
    trained_path = sorted(
        (args.checkpoint_root / "output").glob("*sa4*/*/nn/*00008000.pth")
    )[0]
    trained = torch.load(trained_path, map_location="cpu", weights_only=False)
    results = {}
    suffix = "locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform"
    for grasp, checkpoint in ((False, stage1), (True, stage1), (False, trained)):
        name = suffix + ("_graspview" if grasp else "") + "_task_embedding"
        cfg = args.code_root / "tokenhsi/data/cfg"
        short = "stage2_grasp" if grasp else "stage2"
        env_path = cfg / "multi_agent" / (short + ".yaml")
        train_path = cfg / "train/rlg" / (short + ".yaml")
        if not env_path.exists():
            env_path = cfg / "multi_agent" / ("approach_stage2_joint_carry_mixed80_" + name + ".yaml")
            train_path = cfg / "train/rlg" / ("amp_ma_stage2_joint_carry_mixed80_" + name + ".yaml")
        env = yaml.safe_load(env_path.read_text())["env"]
        params = yaml.safe_load(train_path.read_text())["params"]["network"]
        torch.manual_seed(91)
        builder = AMPMultiAgentBuilder()
        builder.load(params)
        net = builder.build(
            "amp",
            actions_num=32,
            input_shape=(1268 if grasp else 634,),
            amp_input_shape=(1320,),
            value_size=1,
            num_agents=2,
            num_objects=4,
            humanoid_obs_size=230,
            object_obs_size=39,
            goal_obs_size=6,
            observation_mode="clean_scene",
            scene_entity_sizes=[223, 30, 1],
            scene_kinematic_size=7,
            scene_arena_scale=5.0,
            relation_reward_mode=env["relationReward"]["mode"],
            relation_graph_spec=env["relationGraph"],
            device="cpu",
        )
        model = torch.nn.Module()
        model.add_module("a2c_network", net)
        if checkpoint is stage1:
            transfer_stage1_weights(model, checkpoint)
        else:
            model.load_state_dict(checkpoint["model"], strict=True)
        rms = SceneRunningMeanStd(
            entity_sizes=[223, 30, 1],
            entity_counts=[2, 4, 2],
            normalized_sizes=[223, 30, 0],
            kinematic_size=7,
            extra_passthrough_size=10,
            num_views=2 if grasp else 1,
        ).eval()
        rms.load_state_dict(checkpoint["running_mean_std"], strict=True)
        net.actor_encoder.gta_diagnostics_first_forward = False
        net.critic_encoder.gta_diagnostics_first_forward = False
        net.eval()
        inputs = (
            make_grasp_views(obs, torch.tensor([[-1.0, 1.0]]).expand(8, -1), 0.15)
            if grasp
            else obs
        )
        with torch.no_grad():
            normalized = rms(inputs)
            action, sigma = net.eval_actor(normalized)
            value = net.eval_critic(normalized)
        key = ("grasp" if grasp else "sa") + (
            "_transfer" if checkpoint is stage1 else "_trained"
        )
        results[key] = dict(
            action=action,
            sigma=sigma,
            value=value,
            shapes={k: tuple(v.shape) for k, v in model.state_dict().items()},
        )
    if args.record:
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, args.reference)
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        report = {}
        for key, result in results.items():
            if result["shapes"] != reference[key]["shapes"]:
                raise RuntimeError(
                    "Checkpoint parameter names or shapes changed: " + key
                )
            report[key] = {}
            for field in ("action", "sigma", "value"):
                torch.testing.assert_close(
                    result[field], reference[key][field], rtol=1e-5, atol=2e-6
                )
                report[key][field + "_max_error"] = (
                    (result[field] - reference[key][field]).abs().max().item()
                )
        print(json.dumps(report, indent=2))
        args.reference.with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
