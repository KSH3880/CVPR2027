import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.stage2_jointpush import inverse_rotate


def check(player):
    task = player.env.task
    directory = Path(task.cfg['args'].output_path)
    directory.mkdir(parents=True,exist_ok=True)
    ids = torch.arange(task.num_envs,device=task.device)
    report = dict(environments=task.num_envs,training_steps=0,
                  transfer_parameters=len(player._stage2_transfer_report['copied']),resets=[])
    for preset in os.environ.get('JOINTPUSH_CHECK_PRESETS','all_coupled,all_solo,sit,climb,holding_at,holding_ontop,push,random_scenario').split(','):
        task._task_graph_preset = preset
        torch.cuda.synchronize()
        start = time.perf_counter()
        obs = player.env_reset()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        player.get_batch_size(obs['obs'],task.num_envs)
        assert set(task._reset_ref_slots) == {'loco'}
        assert not task._ontop_scene_infeasible(ids).any()
        boxes = task._assigned_box_values(task._box_states)
        roots = task._humanoid_root_states[...,:3]
        delta = roots-boxes[...,:3]
        distance = delta[...,:2].norm(dim=-1)
        goal_delta = task._tar_pos-boxes[...,:3]
        goal_distance = goal_delta[...,:2].norm(dim=-1)
        push = task._templates_by_owner == 5
        shared = task._members > 1
        assert ((distance[push] >= 2-1e-4)&(distance[push] <= 10+1e-4)).all()
        assert ((goal_distance[push] >= 2-1e-4)&(goal_distance[push] <= 10+1e-4)).all()
        local = inverse_rotate(boxes[...,3:7],delta)
        local_goal = inverse_rotate(boxes[...,3:7],goal_delta)
        assert (local[...,1][shared] < 0).all()
        assert (local_goal[...,1][shared] >= local_goal[...,0][shared].abs()-1e-4).all()
        torch.testing.assert_close(obs['obs'][:,1136:],task.obs_buf[:,1136:])
        assert torch.isfinite(obs['obs']).all() and obs['obs'].shape == (task.num_envs,1272)
        active = torch.zeros(task.num_envs,24,device=task.device,dtype=torch.bool)
        active.scatter_(1,task._logical_box_order,True)
        parked = task._box_states[...,:2]-task._env_origins[:,None,:2]
        assert (parked[~active].norm(dim=-1)>900).all()
        report['resets'].append(dict(preset=preset,seconds=elapsed,active_min=int(active.sum(-1).min()),
            active_max=int(active.sum(-1).max()),coupled_agents=int(shared.sum())))
        print('RESET',report['resets'][-1],flush=True)
    report['properties'] = []
    for parent in range(16,24):
        handle = task._box_handles[parent]
        mass = sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[0],handle))
        shapes = task.gym.get_actor_rigid_shape_properties(task.envs[0],handle)
        expected = [20,20,20,20,40,40,60,80][parent-16]
        assert abs(mass-expected)<1e-3
        assert len(shapes)==1 and abs(shapes[0].friction-.4)<1e-6
        report['properties'].append(dict(parent=parent,mass=mass,friction=shapes[0].friction))
    goals = task._tar_pos.clone()
    start = time.perf_counter()
    for _ in range(16):
        action = player.get_action(obs if isinstance(obs,dict) else {'obs':obs},True)
        obs,reward,_,_ = player.env_step(player.env,action)
        state = obs['obs'] if isinstance(obs,dict) else obs
        assert torch.isfinite(state).all() and torch.isfinite(reward).all()
        torch.testing.assert_close(task._tar_pos,goals)
    torch.cuda.synchronize()
    report['policy_step_ms'] = (time.perf_counter()-start)*1000/16
    if task.viewer is not None:
        task.gym.write_viewer_image_to_file(task.viewer,str(directory/'viewer.png'))
        assert (directory/'viewer.png').exists()
    before = task._box_states.clone()
    player.env_reset(ids[::2]*task.num_agents)
    torch.testing.assert_close(task._box_states[1::2],before[1::2])
    demo = task._fetch_conditioned_amp_demo(4096)
    assert demo.shape == (4096,1330) and torch.isfinite(demo).all()
    labels = demo.reshape(4096,10,133)[..., -4:]
    assert (labels.sum(-1)==1).all() and len(labels[:,0].argmax(-1).unique())==4
    report['amp_shape'] = list(demo.shape)
    report['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
    report['viewer'] = task.viewer is not None
    (directory/'simulator_verification.json').write_text(json.dumps(report,indent=2))
    print('PASS stage2_jointpush:',json.dumps(report),flush=True)


def check_agent(agent):
    task = agent.vec_env.env.task
    ids = torch.arange(task.num_envs,device=task.device)
    with torch.no_grad():
        task.reset(ids)
        assert set(task._reset_ref_slots) == {'loco'} and not task._is_eval
        assert not task._ontop_scene_infeasible(ids).any()
        normalized = agent.running_mean_std(task.obs_buf)
        action,_ = agent.model.a2c_network.eval_actor(normalized)
        value = agent.model.a2c_network.eval_critic(normalized)
        assert torch.isfinite(action).all() and torch.isfinite(value).all()
        assert all(not p.requires_grad for p in agent.model.a2c_network.actor_encoder.parameters())
        assert not agent.running_mean_std.training
        assert not agent.optimizer.state
        report = dict(environments=task.num_envs,optimizer_steps=0,training_initialization='loco',
            action_shape=list(action.shape),value_shape=list(value.shape),scene_minibatch=agent.minibatch_size//4,
            learning_rate=agent.optimizer.param_groups[0]['lr'])
        weights = agent.get_stats_weights()
        from utils.stage2_jointpush import check_checkpoint
        check_checkpoint(weights,task.cfg['env'])
        Path(agent.experiment_dir,'initialization_verification.json').write_text(json.dumps(report,indent=2))
        print('PASS stage2_jointpush training initialization:',json.dumps(report),flush=True)


if __name__ == '__main__':
    MAPlayerContinuous.run = check
    MAPlayerContinuous.run_eval = check
    if os.environ.get('JOINTPUSH_CHECK_AGENT') == '1':
        from learning.multi_agent.ma_agent import MAAgent
        MAAgent.train = check_agent
    if os.environ.get('JOINTPUSH_CHECK_TRAIN_INIT') == '1':
        original = entry.get_args
        def get_training_init_args():
            args = original()
            args.eval = False
            return args
        entry.get_args = get_training_init_args
    entry.main()
