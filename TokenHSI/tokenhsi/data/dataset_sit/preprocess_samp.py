import sys
sys.path.append("./")

import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import torch
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../dataset_cfg.yaml"))
    args = parser.parse_args()

    # load yaml
    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # input/output dirs
    samp_pkl_dir = cfg["samp_pkl_dir"]
    output_dir = os.path.join(os.path.dirname(__file__), "motions")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    motion_cfg = {}
    for data in cfg["motions"]["sit"]:
        motion_cfg[data["file"]] = {
            "start": data["start"],
            "end": data["end"],
            "stateInitRange": data["stateInitRange"],
        }

    motion_file = osp.join(osp.dirname(__file__), "dataset_sit.yaml")
    with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
        motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        candidates = {}
        for data in motion_config["motions"]["sit"]:
            candidates[data['file'].split("/")[2].split("+__+")[-1]] = data["rsi_skipped_range"][0]

    pbar = tqdm(list(motion_cfg.keys()))
    for seq in pbar:
        pbar.set_description(seq)

        # only consider motions in candidates (dataset_sit.yaml)
        if seq not in list(candidates.keys()):
            continue

        save_dir = osp.join(osp.dirname(__file__), "motions/sit/")
        os.makedirs(save_dir, exist_ok=True)

        # read smplx parameters from SAMP dataset
        with open(osp.join(samp_pkl_dir, seq + ".pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
            source_fps = data["mocap_framerate"]
            full_poses = torch.tensor(data["pose_est_fullposes"], dtype=torch.float32)
            full_trans = torch.tensor(data["pose_est_trans"], dtype=torch.float32)

        # downsample from source_fps (120Hz) to target_fps (30Hz)
        target_fps = 30
        skip = int(source_fps // target_fps)
        full_poses = full_poses[::skip]
        full_trans = full_trans[::skip]

        # crop
        full_poses = full_poses[motion_cfg[seq]["start"]:motion_cfg[seq]["end"]]
        full_trans = full_trans[motion_cfg[seq]["start"]:motion_cfg[seq]["end"]]
        
        end_init_frame = int(len(full_poses) * motion_cfg[seq]["stateInitRange"])
        assert end_init_frame == candidates[seq] - 1, f"{seq}: end_init_frame: {end_init_frame}, data_read_from_cfg: {candidates[seq]}, it should be: {end_init_frame + 1}"

        # extract useful joints
        joints_to_use = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
        )
        joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
        full_poses = full_poses[:, joints_to_use]
    
        required_params = {}
        required_params["poses"] = full_poses.numpy()
        required_params["trans"] = full_trans.numpy()
        required_params["fps"] = target_fps

        save_path = os.path.join(save_dir, f"SAMP+__+Subject+__+{seq}", "smpl_params.npy")
        print("saving {}".format(save_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, required_params)
