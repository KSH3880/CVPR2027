"""GPU evaluation reset regression: requested PUSH RSI must reach the simulator."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_five_task import HumanoidMAFiveTask

args = get_args()
assert args.eval and args.task_graph == 'push_at'
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env']['motion_file'] = args.motion_file
set_seed(42)
torch.set_num_threads(1)
task = HumanoidMAFiveTask(cfg, parse_sim_params(args, cfg, train),
                         args.physics_engine, args.device, args.device_id, True)
report = {}
try:
    ids = torch.arange(task.num_envs, device=task.device)
    for skill in ('push', 'loco'):
        # Unnormalised weights exercise the same CLI configuration contract.
        cfg['env']['eval']['skill'] = [skill]
        cfg['env']['eval']['skillInitProb'] = [2.0]
        count = 0
        for _ in range(4):
            task._reset_envs(ids)
            mask = task._push_rsi_interaction
            assert bool(mask.all()) if skill == 'push' else not bool(mask.any())
            if skill == 'push':
                rows, owners = mask.nonzero(as_tuple=True)
                sample = task._push_rsi_samples
                native = task._motion_lib['push'].get_motion_state(
                    sample['motion_id'][rows, owners], sample['time'][rows, owners])
                torch.testing.assert_close(task._dof_pos[rows, owners], native[2], atol=1e-6, rtol=0)
                torch.testing.assert_close(task._dof_vel[rows, owners], native[5], atol=1e-6, rtol=0)
                bodies = task._kinematic_humanoid_rigid_body_states[rows, owners]
                assert not task._push_reset_geometry().invalid(
                    bodies[..., :3], bodies[..., 3:7], task._box_states[rows, owners],
                    task._box_size[rows, owners]).any()
                distance = (task._tar_pos-task._box_states[:, :task.num_agents, :3]).norm(dim=-1)
                assert distance.min() >= .999 and distance.max() <= 5.001
                count += len(rows)
            assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        report[skill] = dict(resets=4, rsi_agent_states=count)
    cfg['env']['eval']['skill'] = ['loco', 'push']
    cfg['env']['eval']['skillInitProb'] = [.5, .5]
    count = 0
    for _ in range(8):
        task._reset_envs(ids)
        count += int(task._push_rsi_interaction.sum())
    assert 0 < count < 8*task.num_envs*task.num_agents
    report['mixed'] = dict(resets=8, rsi_agent_states=count)
    cfg['env']['eval']['skill'] = ['push']
    cfg['env']['eval']['skillInitProb'] = [1.0]
    task._reset_envs(ids)
    before = task._dof_pos[1:].clone()
    task._reset_envs(ids[:1])
    torch.testing.assert_close(task._dof_pos[1:], before, atol=0, rtol=0)
    assert task._push_rsi_interaction.all()
    task._reset_envs(ids)
    for _ in range(12):
        task.pre_physics_step(torch.zeros(task.num_envs*task.num_agents, task.num_actions, device=task.device))
        task._physics_step()
        task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
    report['partial_reset_and_12_steps'] = 'passed'
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out/'eval_rsi.json').write_text(json.dumps(report, indent=2)+'\n')
    print('EVAL_RSI_OK', json.dumps(report))
finally:
    if task.viewer is not None:
        task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
