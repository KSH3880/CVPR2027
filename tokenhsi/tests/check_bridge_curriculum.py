import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
import capture_bridge_chamfer as capture
import check_sensor_stage2 as checks
from isaacgym import gymapi
import numpy as np
from PIL import Image, ImageDraw
import torch

from env.tasks.multi_agent import square_sensor as sensor
from utils.sensor_stage2_spec import BRIDGE


def inspect(task, obs, stats):
    walk = task._bridge_walk
    ids = walk.nonzero().flatten()
    people = task._roles[ids, 1:]
    assert task._bridge_pushers[ids].sum(-1).eq(1).all()
    assert task._crosser_mask[ids].sum(-1).eq(3).all()
    assert task._before_predecessor[ids].eq(-1).all()
    assert task._templates[ids[:, None], people].gt(0).all()
    assert task._templates[ids[:, None], people].lt(5).all()
    assert not task._bridge_carry_from_start[ids].any()
    assert task._bridge_start_mode[ids].eq(0).all()
    assert task._bridge_slot[ids].eq(task._bridge_bank+task._bridge_span[ids]-1).all()
    boxes = task._box_states[ids, task._bridge_slot[ids]]
    torch.testing.assert_close(boxes[:, :2], task._bridge_landing_world[ids, :2], atol=.002, rtol=0)
    roots = task._humanoid_root_states[ids[:, None], people]
    local = torch.stack([checks.local_xy(task, roots[:, c, :2], ids, task._scene_ids[ids]) for c in range(3)], 1)
    phases = task._cross_phase[ids[:, None], people]
    direction = torch.sign(task._surface_center[ids, BRIDGE, task._start_surface[ids, people[:, 0]]-1, 0]-task._bridge_landing[ids, 0])
    depth = (local[..., 0]-task._bridge_landing[ids, None, 0])*direction[:, None]-task._bridge_gap[ids, None]/2
    assert (depth[phases == 0] >= .8-.003).all()
    on = phases > 0
    size = task._box_size[ids, task._bridge_slot[ids]]
    assert ((local[..., 0]-task._bridge_landing[ids, None, 0]).abs()[on] < size[:, None, 0].expand_as(on)[on]/2).all()
    human_dist = (local[:, :, None]-local[:, None]).norm(dim=-1)
    pair = torch.triu(torch.ones(3, 3, dtype=torch.bool, device=task.device), diagonal=1)
    assert (human_dist[:, pair] >= 1.2-.003).all()
    sources = task._agent_box_assignment[ids[:, None], people]
    for column in range(3):
        primary = task._box_states[ids, sources[:, column], :2]
        xy = checks.local_xy(task, primary, ids, task._scene_ids[ids])
        dest = task._surface[ids, people[:, column]]-1
        assert ((xy-task._surface_center[ids, BRIDGE, dest]).abs() <= task._surface_half[ids, BRIDGE, dest]).all()
    lowest = sensor.lowest_foot_height(task._kinematic_humanoid_rigid_body_states[ids], task._foot_body_ids, task._foot_vertices)
    expected = task._env_origins[ids, 2]+size[:, 2]+task._scene_cfg['foot_clearance']
    person = task._roles[ids, 3]
    error = lowest[torch.arange(len(ids), device=task.device), person]-expected
    assert (error >= -.003).all()
    assert (error <= .08).all()
    checks.check_sensor(task, obs, stats)
    return dict(walk=int(walk.sum()), bridge=int((task._scene_ids == BRIDGE).sum()),
        span_counts=torch.bincount(task._bridge_span[ids]-1, minlength=3).tolist(),
        phases=torch.bincount(phases.flatten(), minlength=3).tolist(),
        minimum_human_distance=float(human_dist[:, pair].min()), foot_error_max=float(error.max()))


def cameras(task):
    result = []
    for span in (1, 2, 3):
        for phase in (1, 2):
            last = task._cross_phase.gather(1, task._roles[:, 3:4]).squeeze(1)
            ids = (task._bridge_walk & (task._bridge_span == span) & (last == phase)).nonzero().flatten()
            if not len(ids):
                raise RuntimeError('Missing span/phase capture')
            e = int(ids[0])
            target = task._static_centers[e, BRIDGE, :2].mean(0).cpu().numpy()
            radius = float(task._bridge_gap[e]+2*task._bridge_half_a[e, 0]+2*task._bridge_half_b[e, 0])
            xy = capture.scene.rotate_xy(torch.tensor([[.15*radius, radius]]), task._structure_yaw[e, BRIDGE].cpu())[0].numpy()
            handle = capture.camera(task, e, target, np.array([*xy, radius]), 1280, 720)
            result.append((e, span, phase, handle))
    return result


def main():
    args = capture.get_args()
    cfg, train, _ = capture.load_cfg(args)
    cfg['env']['motion_file'], cfg['args'] = args.motion_file, args
    torch.set_num_threads(4)
    capture.set_seed(42)
    capture.BaseTask.__init__ = capture.camera_init
    task, env = capture.parse_task(args, cfg, train['params']['config'], capture.parse_sim_params(args, cfg, train))
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    stats = dict(resets=[])
    fixed = task._root_states[task._terrain_actor_records[:, -1]].clone()
    for iteration, update in enumerate((0, 2000, 3000, 0)):
        task._bridge_curriculum_update = update
        obs = env.reset()
        row = inspect(task, obs, stats)
        row['update'] = update
        stats['resets'].append(row)
        torch.testing.assert_close(fixed, task._root_states[task._terrain_actor_records[:, -1]], atol=0, rtol=0)
        if iteration == 0:
            handles = cameras(task)
            env.step(torch.zeros(task.num_envs*4, task.num_actions, device=task.device))
            task.gym.fetch_results(task.sim, True)
            task.gym.step_graphics(task.sim)
            task.gym.render_all_camera_sensors(task.sim)
            sheet = Image.new('RGB', (1920, 720))
            for i, (e, span, phase, handle) in enumerate(handles):
                frame = capture.pixels(task, e, handle, 1280, 720)
                if np.asarray(frame).std() < 2:
                    raise RuntimeError('Empty camera image')
                ImageDraw.Draw(frame).text((12, 12), f'WALK | span {span} | phase {phase} | 1 PUSH + 3 CROSS', fill='white')
                frame.save(output/f'span{span}_phase{phase}.png')
                sheet.paste(frame.resize((640, 360)), ((i//2)*640, (i%2)*360))
            sheet.save(output/'starts_overview.png')
        assert torch.isfinite(obs).all()
    before = task._root_states.clone()
    walk = task._bridge_walk.clone()
    env.reset(torch.tensor([0], device=task.device))
    actors = task.get_num_actors_per_env()
    torch.testing.assert_close(before[actors:], task._root_states[actors:], atol=0, rtol=0)
    assert torch.equal(walk[1:], task._bridge_walk[1:])
    diagnostics = task._consume_sampling_diagnostics()
    assert all(torch.isfinite(v).all() for v in diagnostics.values())
    stats.update(num_envs=task.num_envs, partial_reset=True, fixed_actors=True, finite=True, screenshots=6)
    (output/'physics.json').write_text(json.dumps(stats, indent=2))
    task.gym.destroy_sim(task.sim)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
