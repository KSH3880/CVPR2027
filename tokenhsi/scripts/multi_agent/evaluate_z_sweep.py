"""Five-task Z evaluation; installs evaluation hooks only in this process."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.z_sweep import TASKS
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    colors = dict(source_size_z='C0', support_size_z='C1', source_bottom_z='C0',
                  support_bottom_z='C1', goal_bottom_z='C2')
    for column, task in enumerate(TASKS):
        for row, kind in enumerate(('size_z', 'bottom_z')):
            ax = axes[row, column]
            selected = [r for r in summary if r['task'] == task and kind in r['axis']]
            for axis in dict.fromkeys(r['axis'] for r in selected):
                series = [r for r in selected if r['axis'] == axis]
                xs = [100*r['value'] for r in series]
                for field, style in [('ever_success', '--'), ('final_success', '-')]:
                    ax.plot(xs, [100*r[field+'_rate'] for r in series], style+'o',
                            color=colors[axis], label=axis.replace('_', ' ')+' / '+field.split('_')[0])
            ax.set_title(task.upper()); ax.set_ylim(0, 100); ax.grid(alpha=.25)
            ax.set_xlabel('Box thickness Z (cm)' if row == 0 else 'Bottom above ground (cm)')
            if column == 0: ax.set_ylabel('Success (%)')
            ax.legend(fontsize=7)
    fig.suptitle('Fixed X/Y; loco starts; dashed=ever, solid=final (agent success rates)')
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/student_rsi_z_sweep'))
    parser.add_argument('--x', type=float, default=.4)
    parser.add_argument('--y', type=float, default=.4)
    parser.add_argument('--base-z', type=float, default=.4)
    parser.add_argument('--scenes-per-condition', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--episode-length', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true', help='Two grid points and a short horizon; not a success benchmark')
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    if not options.checkpoint.is_file(): parser.error('checkpoint does not exist')
    if min(options.scenes_per_condition, options.repeats, options.episode_length) < 1:
        parser.error('trial counts and horizon must be positive')
    if options.output.exists(): parser.error('output already exists; use a new output directory')
    import yaml
    from utils.z_sweep import make_conditions, box_sizes, graph_rows, summarize
    conditions = make_conditions(options.x, options.y, options.base_z,
        sizes=[.2,.65] if options.smoke else None, offsets=[0.,.5] if options.smoke else None)
    sizes = box_sizes(conditions, options.scenes_per_condition)
    config = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_student_rsi.yaml').read_text())
    config['env']['evaluationBoxSizes'] = sizes
    config['env']['box']['build']['randomSize'] = False
    config['env']['box']['reset'].update(randomHeight=True, randomHeightProb=0.,
        randomRot=False, randomAssignment=False, groundStandaloneSitClimb=False)
    config['env']['debug']['reward'] = False
    config['env']['relationReward']['diagnostics']['enabled'] = False
    options.output.mkdir(parents=True)
    snapshot = options.output.resolve()/'evaluation_config.yaml'
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = dict(checkpoint=str(options.checkpoint.resolve()), conditions=conditions,
        scenes_per_condition=options.scenes_per_condition, repeats=options.repeats,
        seed=options.seed, episode_length=options.episode_length, smoke=options.smoke,
        start='loco', binding='canonical evaluation only', box_yaw=0,
        source_distance_xy=1.5, target_distance_from_source_xy=1.5,
        position_z_definition='box bottom offset above floor; actual collision support',
        density=100., mass_changes_with_volume=True,
        interval='Wilson for binary scene-all success; agents within a scene are correlated')
    (options.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
    print('Prepared', len(conditions), 'conditions;', len(sizes), 'scenes', flush=True)
    if options.prepare_only: return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Use evaluate_z_sweep.sh with TOKENHSI_GPU explicitly selected')
    import isaacgym  # Must precede torch.
    import torch
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
    from learning.multi_agent.ma_players import MAPlayerContinuous
    from env.tasks.multi_agent.edge_context_reward import owner_sum
    from utils.edge_scenario_spec import compose_graph
    from utils.edge_ontop_spec import select_graph, copy_graph_rows

    original_sample = SampledOnTopTaskMixin._sample_episode_graph
    original_configure = SampledOnTopTaskMixin._configure_ontop_targets

    def sample(task, ids):
        original_sample(task, ids)
        if not hasattr(task, '_z_sweep_graph'):
            templates, bindings = graph_rows(conditions, options.scenes_per_condition)
            task._z_sweep_graph = compose_graph(templates, bindings, device=task.device, num_objects=4)
        new = select_graph(task._z_sweep_graph, ids)
        copy_graph_rows(task.relation_runtime.graph, ids, new)
        task._edge_goal_owners[ids] = owner_sum((new.required_goal & new.edge_valid).float(), new)>0

    def configure(task, ids):
        original_configure(task, ids)
        condition_ids = (ids // options.scenes_per_condition).cpu().tolist()
        selected = [conditions[i] for i in condition_ids]
        source_offset = torch.tensor([c['source_bottom_z'] for c in selected], device=task.device)
        support_offset = torch.tensor([c['support_bottom_z'] for c in selected], device=task.device)
        goal_offset = torch.tensor([c['goal_bottom_z'] for c in selected], device=task.device)
        is_top = torch.tensor([c['task']=='ontop' for c in selected], device=task.device)
        is_at = torch.tensor([c['task']=='at' for c in selected], device=task.device)
        roots = task._humanoid_root_states[ids, :, :2]
        source_xy = roots + torch.tensor([0.,1.5], device=task.device)
        target_xy = roots + torch.tensor([0.,3.], device=task.device)
        unused_xy = roots + torch.tensor([0.,-1.5], device=task.device)
        task._box_states[ids, :2, :2] = source_xy
        task._box_states[ids, 2:, :2] = torch.where(is_top[:,None,None], target_xy, unused_xy)
        task._box_states[ids, :2, 2] = task._box_size[ids, :2, 2]/2 + source_offset[:,None]
        task._box_states[ids, 2:, 2] = task._box_size[ids, 2:, 2]/2 + support_offset[:,None]
        task._box_states[ids, :, 3:6] = 0.; task._box_states[ids, :, 6] = 1.
        task._box_states[ids, :, 7:13] = 0.
        task._tar_pos[ids, :, :2] = target_xy
        task._tar_pos[ids, :, 2] = task._box_size[ids, :2, 2]/2 + goal_offset[:,None]
        # Two source and two target supports already belong to the existing actor layout.
        task._platform_pos[ids] = task._platform_default_pos[ids]
        task._tar_platform_pos[ids] = task._tar_platform_default_pos[ids]
        elevated = source_offset > 0
        e = ids[elevated]
        task._platform_pos[e, :, :2] = source_xy[elevated]
        task._platform_pos[e, :, 2] = source_offset[elevated,None]-task._platform_height/2
        target_offset = torch.where(is_top, support_offset, goal_offset)
        elevated_target = (is_top|is_at) & (target_offset > 0)
        e = ids[elevated_target]
        task._tar_platform_pos[e, :, :2] = target_xy[elevated_target]
        task._tar_platform_pos[e, :, 2] = target_offset[elevated_target,None]-task._platform_height/2

    @torch.no_grad()
    def evaluate(player):
        task = player.env.task
        n, m = task.num_envs, task.num_agents
        rows = []
        for repeat in range(options.repeats):
            obs = player.env_reset(); player.get_batch_size(obs['obs'], 1)
            alive = torch.ones(n, dtype=torch.bool, device=task.device)
            initial = task.relation_runtime.done.clone()
            ever = initial.clone()
            initial_states = task._box_states.clone()
            initial_goals = task._tar_pos.clone()
            resets = []
            for step in range(task.max_episode_length+1):
                obs = player.env_reset(resets)
                action = player.get_action(obs, True)
                if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite action')
                obs, _, done, info = player.env_step(player.env, action)
                current = task.relation_runtime.done
                ever |= current & alive[:,None]
                ending = done.reshape(n,m).bool().any(-1)
                collect = ending & alive
                for e in collect.nonzero(as_tuple=False).flatten().cpu().tolist():
                    c_id = e//options.scenes_per_condition
                    for a in range(m):
                        source_z = float(initial_states[e,a,2])
                        support_z = float(initial_states[e,2+a,2])
                        rows.append(dict(condition=c_id, repeat=repeat, scene=e, agent=a,
                            task=conditions[c_id]['task'],
                            ever_success=int(ever[e,a]), final_success=int(current[e,a]),
                            initial_success=int(initial[e,a]),
                            terminated=int(info['terminate'].reshape(n,m)[e].bool().any()),
                            end_step=step+1, source_initial_center_z=source_z,
                            source_initial_bottom_z=source_z-float(task._box_size[e,a,2])/2,
                            support_initial_center_z=support_z,
                            support_initial_bottom_z=support_z-float(task._box_size[e,2+a,2])/2,
                            goal_initial_center_z=float(initial_goals[e,a,2])))
                alive &= ~ending
                if not alive.any(): break
                resets = ending.nonzero(as_tuple=False).flatten()*m
            if alive.any(): raise RuntimeError('Evaluation exceeded horizon')
            write_csv(options.output/'trials.csv', rows)
            print('Completed repeat', repeat+1, '/', options.repeats, 'agent trials',len(rows),flush=True)
        summary = summarize(rows, conditions)
        write_csv(options.output/'summary.csv', summary)
        (options.output/'summary.json').write_text(json.dumps(summary,indent=2))
        plot(summary, options.output/'success_by_z.png')
        print('Z sweep saved:', options.output.resolve(), flush=True)
        return summary

    SampledOnTopTaskMixin._sample_episode_graph = sample
    SampledOnTopTaskMixin._configure_ontop_targets = configure
    MAPlayerContinuous.run_eval = evaluate
    sys.argv = ['run.py', '--task','HumanoidMACarry',
        '--cfg_train','tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_shared_edge_encoder.yaml',
        '--cfg_env',str(snapshot),'--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml',
        '--checkpoint',str(options.checkpoint.resolve()),'--num_agents','2',
        '--num_envs',str(len(sizes)),'--num_objects','4','--eval_repeats',str(options.repeats),
        '--eval_skills','loco','--eval_skill_probs','1.0','--task_graph','random_scenario',
        '--episode_length',str(options.episode_length),'--seed',str(options.seed),
        '--output_path',str(options.output.resolve()),'--headless','--no_video','--test','--eval']
    import run
    run.main()


if __name__ == '__main__':
    main()
