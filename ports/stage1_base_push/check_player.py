"""CPU evaluation entrypoint and checkpoint roundtrip using an untrained policy."""
import isaacgym  # Must precede torch.
import json
from pathlib import Path
import torch
import run as entry
from utils.config import get_args, load_cfg, set_seed
from utils.stage1_push_spec import checkpoint_contract
from utils.relation_task_spec import checkpoint_metadata
from env.tasks.multi_agent.collision_reward import agent_collision_config

args=get_args()
assert args.headless and args.device=='cpu' and not args.use_gpu
cfg,train,_=load_cfg(args)
cfg['env']['motion_file']=args.motion_file
torch.set_num_threads(1)
set_seed(42)
train['params']['seed']=42
train['params']['config']['seed']=42
train['params']['config']['device']='cpu'
train['params']['config']['device_name']='cpu'
train['params']['config']['num_actors']=cfg['env']['numEnvs']
entry.args,entry.cfg,entry.cfg_train=args,cfg,train
runner=entry.build_alg_runner(entry.RLGPUAlgoObserver())
runner.load(train)
runner.reset()
player=runner.create_player()
task=player.env.task
obs=player.env_reset()
player.get_batch_size(obs['obs'],1)
assert task.get_obs_size()==634 and task.num_objects==4
out=Path(args.output_path)
out.mkdir(parents=True,exist_ok=True)
checkpoint=out/'validation_only_untrained.pth'
torch.save(dict(model=player.model.state_dict(),
    running_mean_std=player.running_mean_std.state_dict(),
    amp_input_mean_std=player._amp_input_mean_std.state_dict(),
    relation_metadata=checkpoint_metadata(task._relation_cfg),
    agent_collision_config=agent_collision_config(task.cfg['env']),
    stage1_push_contract=checkpoint_contract(task.cfg['env']),
    validation_only=True),checkpoint)
player.restore(str(checkpoint))
resets=[]
for _ in range(24):
    obs=player.env_reset(resets)
    action=player.get_action(obs,True)
    obs,reward,done,info=player.env_step(player.env,action)
    observation=obs['obs'] if isinstance(obs,dict) else obs
    assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
    resets=done.nonzero().flatten()
assert not task._rsi_interaction.any()
report=dict(untrained_policy=True, cpu=True, observation=634, tokens=8,
            amp_input=task.get_num_amp_obs(), player_steps=24,
            model_and_rms_checkpoint_restore=True, loco_start=True)
(out/'player_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
task.gym.destroy_sim(task.sim)
