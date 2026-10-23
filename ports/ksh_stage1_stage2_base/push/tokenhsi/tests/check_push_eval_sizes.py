"""GPU check of physical mass/friction, size observations and loco-only resets."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_five_task import HumanoidMAFiveTask

args = get_args()
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env']['motion_file'] = args.motion_file
set_seed(42)
torch.set_num_threads(1)
task = HumanoidMAFiveTask(cfg, parse_sim_params(args, cfg, train),
                         args.physics_engine, args.device, args.device_id, True)
try:
    ids = torch.arange(task.num_envs, device=task.device)
    task._reset_envs(ids)
    assert task._push_eval_asset_mask.all()
    # Template probabilities include the original standalone HOLDING slot;
    # PUSH is the last template, while its policy task ID remains 4.
    assert (task._size_template_probabilities[..., -1] == 1).all()
    assert not task._push_rsi_interaction.any()
    torch.testing.assert_close(task._box_bps.abs().amax(-2)*2, task._box_size)
    torch.testing.assert_close(task._box_states[:, :2, 2], task._box_size[:, :2, 2]/2)
    offset = task.num_agents*task.get_clean_humanoid_obs_size()
    nodes = task.obs_buf[:, offset:offset+task.num_objects*30].reshape(task.num_envs, task.num_objects, 30)
    torch.testing.assert_close(nodes[..., 6:], task._logical_box_values(task._box_bps).flatten(-2))
    masses, frictions = [], []
    for row in range(task.num_envs):
        for owner in range(task.num_agents):
            handle = task._box_handles[row*task.num_physical_objects+owner]
            body = task.gym.get_actor_rigid_body_properties(task.envs[row], handle)
            shapes = task.gym.get_actor_rigid_shape_properties(task.envs[row], handle)
            mass = sum(b.mass for b in body)
            assert abs(mass-cfg['env']['pushBox']['mass']) < 1e-4
            assert all(abs(s.friction-cfg['env']['pushBox']['friction']) < 1e-5 for s in shapes)
            masses.append(mass)
            frictions.extend(s.friction for s in shapes)
    for _ in range(12):
        task.pre_physics_step(torch.zeros(task.num_envs*task.num_agents, task.num_actions, device=task.device))
        task._physics_step()
        task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
    report = dict(task._push_eval_size_report, masses_kg=masses, shape_frictions=frictions,
                  bbox_observations_match=True, grounded=True, push_assignment=True,
                  finite_12_steps=True, rsi=False)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out/'size_physics_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print('SIZE_PHYSICS_OK', flush=True)
finally:
    task.gym.destroy_sim(task.sim)
