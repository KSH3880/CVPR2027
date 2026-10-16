import json
from pathlib import Path

from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
import torch

from utils.task_role_spec import task_graph_packet
from utils.group_carry_spec import entity_validity
from env.tasks.multi_agent.group_carry_reset import scene_invalid


def main():
    args = get_args()
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    set_seed(3077)
    task, env = parse_task(args, cfg, train, parse_sim_params(args, cfg, train))
    ids = torch.arange(task.num_envs, device=task.device)
    results = []
    for preset in ('two_pairs_at', 'two_pairs_ontop', 'one_pair_at', 'one_pair_ontop',
                   'all_solo_at', 'all_solo_ontop', 'mixed_groups'):
        for name in ('loco', 'pickUp', 'carryWith', 'putDown'):
            if name == 'putDown' and preset in ('two_pairs_ontop', 'one_pair_ontop'):
                continue
            task._task_graph_preset = preset
            task._skill_init_prob.zero_()
            task._skill_init_prob[task._skill.index(name)] = 1.
            task._reset_envs(ids)
            if scene_invalid(task, ids).any():
                raise RuntimeError('Invalid initial placement: '+preset+'/'+name)
            if not torch.isfinite(task.obs_buf).all() or not torch.isfinite(task._amp_obs_buf).all():
                raise RuntimeError('Nonfinite reset observations')
            joint = task._group_joint
            if joint.any():
                torch.testing.assert_close(task._hist_amp_obs_buf[joint],
                    task._curr_amp_obs_buf[joint][:, None].expand(-1, task._num_amp_obs_steps-1, -1))
            if name != 'loco' and joint.any():
                rows, humans = joint.nonzero(as_tuple=True)
                pool = task._joint_reset.pools[name]
                frame = task._group_frame[rows, humans]
                source = (frame[:, None] == pool['source_frame'][None]).long().argmax(-1)
                role = task._group_source_role[rows, humans]
                torch.testing.assert_close(task._dof_pos[rows, humans], pool['dof_position'][source, role])
                torch.testing.assert_close(task._dof_vel[rows, humans], pool['dof_velocity'][source, role])
                torch.testing.assert_close(task._humanoid_root_states[rows, humans, 7:10].norm(dim=-1),
                    pool['root_state'][source, role, 7:10].norm(dim=-1), atol=1e-5, rtol=1e-5)
            keep = ids[1::2]
            old = {key: getattr(task, key)[keep].clone() for key in
                   ('_humanoid_root_states', '_box_states', '_group_leader', '_group_skill', '_grasp_sides')
                   if hasattr(task, key)}
            old_packet = task_graph_packet(task.relation_runtime.graph, task.num_envs)[keep].clone()
            task._reset_envs(ids[::2])
            for key, value in old.items():
                torch.testing.assert_close(getattr(task, key)[keep], value, rtol=0, atol=0)
            torch.testing.assert_close(task_graph_packet(task.relation_runtime.graph, task.num_envs)[keep], old_packet)
            for _ in range(16):
                env.step(torch.zeros(task.num_envs*4, task.get_action_size(), device=task.device))
                if not torch.isfinite(task.obs_buf).all() or not torch.isfinite(task.rew_buf).all():
                    raise RuntimeError('Nonfinite simulation: '+preset+'/'+name)
            results.append({'preset': preset, 'skill': name, 'steps': 16, 'subset_reset': True})
            print('[group physics]', preset, name, 'passed', flush=True)
    task._task_graph_preset = 'two_pairs_at'
    task._skill_init_prob.zero_(); task._skill_init_prob[task._skill.index('loco')] = 1.
    probability = task._joint_loco_platform.spec['probability']
    task._joint_loco_platform.spec['probability'] = 1.
    task._reset_envs(ids)
    if not task._joint_loco_platform.active.all():
        raise RuntimeError('Both loco platforms must be active')
    task._joint_loco_platform.spec['probability'] = probability
    task._task_graph_preset = 'random_scenario'
    task._skill_init_prob[:] = .25
    task._reset_envs(ids)
    valid = entity_validity(task_graph_packet(task.relation_runtime.graph, task.num_envs), 4, 8)[:, 4:12]
    active = torch.zeros(task.num_envs, 12, device=task.device, dtype=torch.bool)
    active.scatter_(1, task._logical_box_order, valid)
    for _ in range(128):
        env.step(torch.zeros(task.num_envs*4, task.get_action_size(), device=task.device))
    if not torch.isfinite(task._root_states).all():
        raise RuntimeError('Inactive physical actor became nonfinite')
    distance = (task._box_states[..., :2]-task._env_origins[:, None, :2]).norm(dim=-1)
    if (distance[~active] < 20.).any():
        raise RuntimeError('Inactive actor entered the play area')
    path = Path(args.output_path)/'physics.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'cases': results, 'two_platforms': True,
        'random_steps': 128, 'physical_boxes': task._box_states.shape[1],
        'raw_obs': task._compute_clean_scene_obs(grasp_view=False).shape[-1]}, indent=2))
    print('[group physics] all checks passed:', path, flush=True)


if __name__ == '__main__':
    main()
