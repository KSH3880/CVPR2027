"""Inspect actual Door reset across one physics step, without a policy/checkpoint."""
from isaacgym import gymapi
import os
import time
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor

args = get_args()
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env']['motion_file'] = args.motion_file
# Hold the sampled joint pose for the diagnostic physics step only.
cfg['env']['enableTrackInitState'] = True
set_seed(args.seed if args.seed >= 0 else 42)
task = HumanoidMADoor(cfg, parse_sim_params(args, cfg, train),
                      args.physics_engine, args.device, args.device_id, args.headless)
cycles = int(os.environ.get('DOOR_RESET_CYCLES', '3' if args.headless else '0'))
try:
    ids = torch.arange(task.num_envs, device=task.device)
    cycle = 0
    while cycles == 0 or cycle < cycles:
        task._reset_envs(ids)
        before = task._humanoid_root_states[..., 3:7].clone()
        task.pre_physics_step(torch.zeros(task.num_envs * task.num_agents,
                                          task.num_dof, device=task.device))
        for _ in range(task.control_freq_inv):
            task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task.post_physics_step()
        after = task._humanoid_root_states[..., 3:7]
        similarity = (before * after).sum(-1).abs()
        print('DOOR_RESET_CHECK', cycle + 1,
              'quat_similarity_min=', round(similarity.min().item(), 5),
              'root_z=', task._humanoid_root_states[..., 2].tolist(),
              'hinge_angle=', task.door_hinge_angle.tolist(),
              'root_handle_distance=', (task._humanoid_root_states[..., :2] - task.door_handle_position[..., :2]).norm(dim=-1).tolist(), flush=True)
        assert torch.isfinite(task.obs_buf).all()
        assert similarity.min() > .98, 'Reset orientation changed unexpectedly on first step'
        # Freeze this first physical frame for 3 seconds so reset is easy to inspect.
        if task.viewer is not None:
            for _ in range(90):
                task.render()
                time.sleep(1 / 30)
        cycle += 1
finally:
    if task.viewer is not None:
        task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)
