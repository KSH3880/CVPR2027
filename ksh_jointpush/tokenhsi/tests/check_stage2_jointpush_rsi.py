import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi import HumanoidMAStage2JointPushRsi
from utils.stage2_jointpush import inverse_rotate


def check(player):
    task = player.env.task
    ids = torch.arange(task.num_envs,device=task.device)
    report = dict(environments=task.num_envs,evaluation=task._is_eval,optimizer_steps=0,
        eval_push_rsi=task._eval_push_rsi,
        legacy_size_fixture=os.environ.get('RSI_CHECK_LEGACY_FIXTURE') == '1',resets=[])
    presets = os.environ.get('RSI_CHECK_PRESETS','all_coupled,push').split(',')
    for preset in presets:
        task._task_graph_preset = preset
        obs = player.env_reset()
        player.get_batch_size(obs['obs'],task.num_envs)
        assert torch.isfinite(obs['obs']).all()
        assert not task._ontop_scene_infeasible(ids).any()
        push = task._templates_by_owner == 5
        boxes = task._assigned_box_values(task._box_states)
        delta = task._tar_pos-boxes[...,:3]
        goal_distance = delta[...,:2].norm(dim=-1)[push]
        assert ((goal_distance >= 2-1e-4)&(goal_distance <= 10+1e-4)).all()
        loco = push & ~task._interaction
        distance = (task._humanoid_root_states[...,:2]-boxes[...,:2]).norm(dim=-1)
        assert ((distance[loco] >= 2-1e-4)&(distance[loco] <= 10+1e-4)).all()
        same = task._groups[:,:,None] == task._groups[:,None]
        assert ((task._interaction[:,:,None] == task._interaction[:,None]) | ~same).all()
        if task._is_eval and task._eval_push_rsi:
            torch.testing.assert_close(task._interaction,push)
            assert set(task._reset_ref_slots) <= {'loco','push'}
            assert task._bank is not None and not hasattr(task,'_size_rsi_cache')
        elif task._is_eval:
            assert not task._interaction.any() and set(task._reset_ref_slots) == {'loco'}
            assert task._bank is None and not hasattr(task,'_size_rsi_cache')
        if (not task._is_eval or task._eval_push_rsi) and push.any():
            assert task._interaction.any()
            if not task._is_eval:
                assert loco.any()
            es,owners = task._reset_ref_slots['push']
            mids,times = task._reset_ref_motion_ids['push'],task._reset_ref_motion_times['push']
            reference = task._motion_lib['push'].get_motion_state(mids,times)[2]
            torch.testing.assert_close(task._dof_pos[es,owners],reference)
            labels = task._hist_amp_obs_buf[es,owners,...,-4:]
            assert (labels.argmax(-1) == 3).all()
        if not task._is_eval:
            for skill,slots in task._reset_ref_slots.items():
                if skill in ('loco','push'):
                    continue
                es,owners = slots
                mids,times = task._reset_ref_motion_ids[skill],task._reset_ref_motion_times[skill]
                dofs = task._motion_lib[skill].get_motion_state(mids,times)[2]
                torch.testing.assert_close(task._dof_pos[es,owners],dofs)
            if preset in ('sit','climb','holding_at','holding_ontop'):
                assert len(task._reset_ref_slots) > 1
            if task._size_rsi_cache.cache is not None:
                cache = task._size_rsi_cache.cache
                report['solo_cache'] = str(cache.path)
                assert len(cache.entries) > 0
        shared = task._members > 1
        local_goal = inverse_rotate(boxes[...,3:7],delta)
        assert (local_goal[...,1][shared] >= local_goal[...,0][shared].abs()-1e-4).all()
        report['resets'].append(dict(preset=preset,interaction_people=int(task._interaction.sum()),
            push_people=int(push.sum()),skills={k:len(v[0]) for k,v in task._reset_ref_slots.items()}))
        print('RSI RESET',report['resets'][-1],flush=True)
        goals = task._tar_pos.clone()
        for _ in range(16):
            action = player.get_action(obs if isinstance(obs,dict) else {'obs':obs},True)
            obs,reward,_,_ = player.env_step(player.env,action)
            state = obs['obs'] if isinstance(obs,dict) else obs
            assert torch.isfinite(state).all() and torch.isfinite(reward).all()
            torch.testing.assert_close(task._tar_pos,goals)
    before = task._box_states.clone()
    player.env_reset(ids[::2]*4)
    torch.testing.assert_close(task._box_states[1::2],before[1::2])
    directory = Path(task.cfg['args'].output_path)
    directory.mkdir(parents=True,exist_ok=True)
    if task.viewer is not None:
        task.gym.write_viewer_image_to_file(task.viewer,str(directory/'viewer.png'))
    (directory/'verification.json').write_text(json.dumps(report,indent=2))
    print('PASS stage2_jointpush_rsi',json.dumps(report),flush=True)


def check_agent(agent):
    task = agent.vec_env.env.task
    with torch.no_grad():
        task.reset(torch.arange(task.num_envs,device=task.device))
        assert not task._is_eval and len(task._reset_ref_slots) > 1
        assert task._interaction.any()
        assert not agent.optimizer.state
        assert not agent.running_mean_std.training
        assert all(not p.requires_grad for p in agent.model.a2c_network.actor_encoder.parameters())
        normalized = agent.running_mean_std(task.obs_buf)
        action,_ = agent.model.a2c_network.eval_actor(normalized)
        value = agent.model.a2c_network.eval_critic(normalized)
        assert torch.isfinite(action).all() and torch.isfinite(value).all()
        from utils.stage2_jointpush import check_checkpoint
        check_checkpoint(agent.get_stats_weights(),task.cfg['env'])
        report = dict(environments=task.num_envs,optimizer_steps=0,
            legacy_size_fixture=os.environ.get('RSI_CHECK_LEGACY_FIXTURE') == '1',
            skills={k:len(v[0]) for k,v in task._reset_ref_slots.items()},
            action_shape=list(action.shape),learning_rate=agent.optimizer.param_groups[0]['lr'])
        Path(agent.experiment_dir,'initialization_verification.json').write_text(json.dumps(report,indent=2))
        print('PASS stage2_jointpush_rsi training initialization:',json.dumps(report),flush=True)


if __name__ == '__main__':
    if os.environ.get('RSI_CHECK_LEGACY_FIXTURE') == '1':
        from utils import stage2_jointpush as spec
        original_bank = spec.asset_bank
        def fixture_bank(count,env,device):
            sizes,masses = original_bank(count,env,device)
            sizes[:,:4] = sizes.new_tensor([[.55,.45,.45],[.55,.5,.55],[.55,.45,.45],[.55,.5,.55]])
            sizes[:,4:8] = sizes.new_tensor([[.55,.45,.45],[.8,.65,.4],[.55,.45,.45],[.8,.65,.4]])
            sizes[:,8:12] = sizes[:,4:8]
            sizes[:,12:16] = sizes.new_tensor([[.6,.8,.4],[.4,.5,.4],[.6,.8,.4],[.4,.5,.4]])
            masses[:,:16] = sizes[:,:16].prod(-1)*100.
            return sizes,masses
        spec.asset_bank = fixture_bank
        from env.tasks.multi_agent.jointpush_solo_rsi_cache import JointPushSoloCache
        original_profiles = JointPushSoloCache._profiles
        def require_cached_profiles(cache,kind,owner,pending):
            profiles = original_profiles(cache,kind,owner,pending)
            if profiles:
                raise RuntimeError('Verification fixture was not found in matching Stage1 cache: '+str(cache.path))
            return profiles
        JointPushSoloCache._profiles = require_cached_profiles
    original_load = entry.load_cfg
    def load_cfg(args):
        cfg,train,logdir = original_load(args)
        if not args.eval:
            cfg['env']['stage2JointPush']['solo_rsi_cache_directory'] = 'output_etc/check/stage2_jointpush_rsi/solo_cache'
        return cfg,train,logdir
    entry.load_cfg = load_cfg
    original_apply = HumanoidMAStage2JointPushRsi._apply_push_rsi
    def checked_apply(task):
        boxes,goals = task._box_states.clone(),task._tar_pos.clone()
        original_apply(task)
        torch.testing.assert_close(task._box_states,boxes)
        torch.testing.assert_close(task._tar_pos,goals)
    HumanoidMAStage2JointPushRsi._apply_push_rsi = checked_apply
    original_args = entry.get_args
    def get_args():
        args = original_args()
        if os.environ.get('RSI_CHECK_TRAIN_INIT') == '1':
            args.eval = False
        return args
    entry.get_args = get_args
    MAPlayerContinuous.run = check
    MAPlayerContinuous.run_eval = check
    if os.environ.get('JOINTPUSH_CHECK_AGENT') == '1':
        from learning.multi_agent.ma_agent import MAAgent
        MAAgent.train = check_agent
    entry.main()
