import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import joblib
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../dataset_cfg.yaml"))
    args = parser.parse_args()

    # load yaml
    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # input/output dirs
    omomo_dir = cfg["omomo_dir"]
    output_dir = os.path.join(os.path.dirname(__file__), "motions")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = {
        "omomo": cfg["motions"]["omomo"],
    }

    # load dataset
    train_data = joblib.load(os.path.join(omomo_dir, "train_diffusion_manip_seq_joints24.p"))
    test_data = joblib.load(os.path.join(omomo_dir, "test_diffusion_manip_seq_joints24.p"))
    
    seq_to_id = {}
    for k, v in train_data.items():
        seq_to_id[v["seq_name"]] = {
            "set": "train",
            "id": k
        }
    for k, v in test_data.items():
        seq_to_id[v["seq_name"]] = {
            "set": "test",
            "id": k
        }

    for k, v in candidates.items():
        skill = k
        output_dir_skill = os.path.join(output_dir, skill)
        os.makedirs(output_dir_skill, exist_ok=True)

        for seq_name in v:

            # load raw params from AMASS dataset
            if seq_to_id[seq_name]["set"] == "train":
                raw_params = train_data[seq_to_id[seq_name]["id"]]
            else:
                raw_params = test_data[seq_to_id[seq_name]["id"]]

            poses = np.concatenate([raw_params["root_orient"], raw_params["pose_body"], np.zeros((raw_params["root_orient"].shape[0], 6))], axis=-1)
            trans = raw_params["trans"]

            required_params = {}
            required_params["poses"] = poses
            required_params["trans"] = trans
            required_params["fps"] = 30.0

            save_path = os.path.join(output_dir_skill, f"OMOMO+__+{seq_name}", "smpl_params.npy")
            print("saving {}".format(save_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, required_params)
