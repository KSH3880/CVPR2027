"""CPU-only simulator, reset, physical asset and original-network connection audit."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
import yaml
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from utils.task_role_spec import task_graph_packet

args = get_args()
assert args.headless and args.device == 'cpu' and not args.use_gpu
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env']['motion_file'] = args.motion_file
cfg['env']['enableTrackInitState'] = True
args.eval = False  # Exercise training RSI and training buffers without an optimizer.
set_seed(42)
torch.set_num_threads(1)
sim = parse_sim_params(args, cfg, train)
sim.physx.num_threads = 1
task = HumanoidMAStage1Push(cfg, sim, args.physics_engine, 'cpu', 0, True)
ids = torch.arange(task.num_envs)
task._reset_envs(ids)
assert task.get_obs_size() == 634 and task.get_scene_num_tokens() == 8
assert task.num_objects == 4
for env in range(task.num_envs):
    for box in range(4):
        handle=task._box_handles[env*4+box]
        body=task.gym.get_actor_rigid_body_properties(task.envs[env],handle)
        shape=task.gym.get_actor_rigid_shape_properties(task.envs[env],handle)
        assert abs(sum(b.mass for b in body)-20.)<1e-3
        assert all(abs(s.friction-.6)<1e-6 for s in shape)
builder = AMPMultiAgentBuilder()
builder.load(train['params']['network'])
network = builder.build('amp', actions_num=task.num_dof, input_shape=(634,),
    amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=2, num_objects=4,
    humanoid_obs_size=task.get_humanoid_obs_size(), object_obs_size=task.get_object_obs_size(),
    goal_obs_size=task.get_goal_obs_size(), observation_mode='clean_scene',
    scene_entity_sizes=task.get_scene_entity_sizes(), scene_kinematic_size=7,
    scene_arena_scale=task.get_scene_arena_scale(), relation_reward_mode=task._relation_cfg['mode'],
    relation_graph_spec=task._relation_graph_spec, device='cpu')
network.eval()
interactions, late, trials = 0, 0, 0
for _ in range(250):
    task._reset_envs(ids)
    interactions += int(task._rsi_interaction.sum())
    late += int((task._rsi_late & task._rsi_interaction).sum())
    trials += 2*len(ids)
    packet = task_graph_packet(task.relation_runtime.graph, task.num_envs).reshape(-1, 2, 5)
    assert (packet[..., 1] == 4).all()
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
before = [x[1].clone() for x in (task._box_states, task._humanoid_root_states, task._tar_pos, task._amp_obs_buf)]
task._reset_envs(ids[:1])
for tensor, expected in zip((task._box_states,task._humanoid_root_states,task._tar_pos,task._amp_obs_buf), before):
    torch.testing.assert_close(tensor[1], expected)
with torch.no_grad():
    actor = network.eval_actor(task.obs_buf)
    critic = network.eval_critic(task.obs_buf)
    expert = task.fetch_amp_obs_demo(32)
    disc = network.eval_disc(expert)
    assert all(torch.isfinite(x).all() for x in (*actor, critic, disc))
    saved={key:value.clone() for key,value in network.state_dict().items()}
    network.load_state_dict(saved,strict=True)
    for _ in range(24):
        actions = network.eval_actor(task.obs_buf)[0]
        task.pre_physics_step(actions.reshape(-1, task.num_dof))
        task._physics_step()
        task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
sampling=task._consume_sampling_diagnostics()
assert abs(float(sampling['sampling/template_push_at'])-1.)<1e-6
assert not float(sampling['sampling/shared_primary_object'])
assert abs(float(sampling['sampling/relation_push_contact'])-1.)<1e-6
task._is_eval = True
task._success_buf = torch.zeros(task.num_envs*2, dtype=torch.long)
task._precision_buf = torch.zeros(task.num_envs*2)
task._reset_envs(ids)
assert not task._rsi_interaction.any()
report = dict(observation=634, tokens=8, people=2, boxes=4, policy_tasks=5,
              amp_input=task.get_num_amp_obs(), reset_agent_trials=trials,
              interaction_fraction=interactions/trials, late_fraction=late/max(1,interactions),
              physics_steps=24, partial_reset=True, loco_eval=True,
              strict_restore=True, mass_kg=20, friction=.6,
              box_sizes=task._box_size.tolist())
out=Path(args.output_path)
out.mkdir(parents=True,exist_ok=True)
(out/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
task.gym.destroy_sim(task.sim)
