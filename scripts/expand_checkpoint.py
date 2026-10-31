"""Turn the TokenHSI stage-1 checkpoint (traj, sit, carry, climb) into the starting point for the PUSH fine-tune
(HumanoidTrajSitCarryClimbPush, five tasks).

    python scripts/expand_ckpt_push.py --src output/tokenhsi/ckpt_stage1.pth --dst output/push/ckpt/push_init.pth

The policy obs grow from 223 + 127 task + 4 one-hot to 223 + 127 + 42 (PUSH) + 5 one-hot, and every
discriminator step from 129 + 4 to 129 + 5. Everything that exists in stage 1 is copied unchanged:
  - the new PUSH tokenizer starts as a copy of the carry tokenizer (same 42-dim layout: box state, corners, target),
    and its obs normaliser starts from the carry statistics;
  - the critic and discriminator get zero weights on the new inputs, so their outputs for the old tasks are unchanged;
  - optimizer state, epoch and frame counters start fresh.
"""
import argparse
import os

import torch

SELF = 223
OLD_TASK = [20, 38, 42, 27] # traj, sit, carry, climb
PUSH = 42
AMP_STEP = 129
AMP_STEPS = 10


def obs_index_map():
    """For every new obs index, the old index whose statistics it starts from, and whether its weights are copied."""
    old_task_tot = sum(OLD_TASK)
    carry0 = SELF + OLD_TASK[0] + OLD_TASK[1]
    src, copy = [], []
    for i in range(SELF + old_task_tot): # self obs + old task obs
        src.append(i); copy.append(True)
    for j in range(PUSH): # push obs <- carry obs stats
        src.append(carry0 + j); copy.append(False)
    for k in range(4): # old one-hot
        src.append(SELF + old_task_tot + k); copy.append(True)
    src.append(SELF + old_task_tot + 2); copy.append(False) # push one-hot <- carry one-hot stats
    return torch.tensor(src), torch.tensor(copy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="output/tokenhsi/ckpt_stage1.pth")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu")
    m = ck["model"]
    src, copy = obs_index_map()
    n_old = SELF + sum(OLD_TASK) + 4
    assert m["a2c_network.critic_mlp.0.weight"].shape[1] == n_old

    new = dict(m)
    # policy obs normaliser
    rms = ck["running_mean_std"]
    new_rms = {"running_mean": rms["running_mean"][src].clone(), "running_var": rms["running_var"][src].clone(), "count": rms["count"].clone()}

    # critic: same weights on old inputs, zero on new ones
    w = m["a2c_network.critic_mlp.0.weight"]
    nw = torch.zeros(w.shape[0], len(src))
    nw[:, copy] = w[:, src[copy]]
    new["a2c_network.critic_mlp.0.weight"] = nw

    # PUSH tokenizer <- carry tokenizer
    for k in list(m.keys()):
        if k.startswith("a2c_network.task_encoder.2."):
            new[k.replace("task_encoder.2.", "task_encoder.4.")] = m[k].clone()

    pe = m["a2c_network.pos_embed"]
    new["a2c_network.pos_embed"] = torch.cat([pe, torch.zeros(1, 1, pe.shape[-1])], dim=1) # unused (use_pos_embed False)

    # discriminator: every step gets a fifth one-hot input with zero weight
    dw = m["a2c_network._disc_mlp.0.weight"]
    old_step = AMP_STEP + 4
    assert dw.shape[1] == old_step * AMP_STEPS
    ndw = torch.zeros(dw.shape[0], (old_step + 1) * AMP_STEPS)
    for s in range(AMP_STEPS):
        ndw[:, s * (old_step + 1):s * (old_step + 1) + old_step] = dw[:, s * old_step:(s + 1) * old_step]
    new["a2c_network._disc_mlp.0.weight"] = ndw
    amp = ck["amp_input_mean_std"]
    new_amp = {"running_mean": torch.cat([amp["running_mean"], torch.zeros(1)]),
               "running_var": torch.cat([amp["running_var"], torch.ones(1)]), "count": amp["count"].clone()}

    opt = ck["optimizer"]
    n_params = len(opt["param_groups"][0]["params"]) + 6 # one more tokenizer: 3 linear layers
    pg = {k: v for k, v in opt["param_groups"][0].items() if k != "params"}
    pg["params"] = list(range(n_params))

    out = {"model": new, "running_mean_std": new_rms, "reward_mean_std": ck["reward_mean_std"], "amp_input_mean_std": new_amp,
           "epoch": 0, "frame": 0, "last_mean_rewards": -100500, "env_state": None,
           "optimizer": {"state": {}, "param_groups": [pg]}}
    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    torch.save(out, a.dst)
    print("wrote %s: obs %d -> %d, disc %d -> %d, %d optimizer params" % (a.dst, n_old, len(src), dw.shape[1], ndw.shape[1], n_params))


if __name__ == "__main__":
    main()
