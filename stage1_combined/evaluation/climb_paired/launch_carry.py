"""Frozen Chair carry policies on one common dynamic-chair evaluation scene."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
import torch
# Bundled fused attention does not support RTX5090 with this torch version.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from learning.multi_agent.ma_players import MAPlayerContinuous
from evaluation.hold_at.public_policy import PublicChairPolicy
from evaluation.climb_paired.continuous_carry import run_continuous

restore_original=MAPlayerContinuous.restore
action_original=MAPlayerContinuous.get_action


def snapshot(player):
    return {name:{key:value.detach().cpu().clone() for key,value in module.state_dict().items()}
            for name,module in [('model',player.model),('running_mean_std',player.running_mean_std)]}


def restore(player,path):
    result=restore_original(player,path)
    weights=torch.load(path,map_location='cpu',weights_only=False)
    player._carry_frozen=snapshot(player)
    for name,values in player._carry_frozen.items():
        assert set(values)==set(weights[name])
        for key,value in values.items():assert torch.equal(value,weights[name][key]),key
    player.model.requires_grad_(False);player.model.eval();player.running_mean_std.eval()
    player._carry_epoch=int(weights['epoch'])
    return result


def run_eval(player):
    cfg=player.env.task.cfg['env']['holdAtFT'].get('public_policy')
    if cfg:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            player.public_chair_policy=PublicChairPolicy(player.env.task,player,cfg)
        public_before={key:value.detach().cpu().clone() for key,value in player.public_chair_policy.net.state_dict().items()}
        public_rms_before={key:value.detach().cpu().clone() for key,value in player.public_chair_policy.rms.state_dict().items()}
    run_continuous(player)
    current=snapshot(player)
    for name,values in player._carry_frozen.items():
        for key,value in values.items():assert torch.equal(value,current[name][key]),key
    if cfg:
        for module,values in [(player.public_chair_policy.net,public_before),(player.public_chair_policy.rms,public_rms_before)]:
            for key,value in module.state_dict().items():assert torch.equal(value.detach().cpu(),values[key]),key
    path=Path(player.env.task.cfg['args'].output_path)/'results.json'
    result=json.loads(path.read_text())
    result.update(scene_checkpoint_exact=True,scene_model_and_rms_unchanged=True,
        policy=player.public_chair_policy.metadata if cfg else dict(source_epoch=player._carry_epoch,strict_model_tensors=len(current['model']),weights_frozen=True),
        public_model_and_rms_unchanged=True if cfg else None)
    path.write_text(json.dumps(result,indent=2))


def action(player,obs,is_determenistic=False):
    if hasattr(player,'public_chair_policy'):
        return player.public_chair_policy.action(reset_env_ids=player._public_reset_env_ids)
    return action_original(player,obs,is_determenistic)

MAPlayerContinuous.restore=restore
MAPlayerContinuous.run_eval=run_eval
MAPlayerContinuous.get_action=action
import run as app
try:app.main()
finally:
    if torch.cuda.is_initialized():torch.cuda.synchronize()
