"""One task/seed, first scene episodes; isolated from the running training process."""
import isaacgym  # Isaac Gym must be imported before torch.
from isaacgym import gymapi
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
import run as entry
from utils.config import get_args, load_cfg, set_seed
from utils.finetune_evaluation import AT_SIZES, TASKS, atomic_json, classify, distance_bins, summarize_rows, verify_archive
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def install_benchmark_assets():
    """Evaluation-process-only asset hook. Source YAML/checkpoint contracts stay intact."""
    original = HumanoidMAStage1Push._load_box_asset

    def load(self):
        original(self)
        task = self._task_graph_preset
        if task not in ('holding_at', 'sit', 'climb'):
            return
        if task == 'holding_at':
            sizes = self._box_size.new_tensor(AT_SIZES)
            for owner in (0, 1):
                indices = torch.randint(9, (self.num_envs,), device=self.device)
                if self.num_envs >= 9:
                    indices[:9] = torch.arange(9, device=self.device)
                    indices = indices[torch.randperm(self.num_envs, device=self.device)]
                self._box_size[:, owner] = sizes[indices]
        self._box_scale = self._box_size/self._box_size.new_tensor(self._build_base_size)
        self._box_density = torch.full_like(self._box_density, 100.)
        self._box_assets = []
        cache = {}
        for dims in self._box_size.flatten(0, 1).cpu().tolist():
            key = tuple(dims)
            if key not in cache:
                options = gymapi.AssetOptions()
                options.density = 100.
                options.angular_damping = .01
                options.linear_damping = .01
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                options.fix_base_link = task in ('sit', 'climb')
                cache[key] = self.gym.create_box(self.sim, *dims, options)
            self._box_assets.append(cache[key])
        # Gym owns the asset handles until destroy_sim; there is no destroy_asset API.
    HumanoidMAStage1Push._load_box_asset = load


def bindings(task, name):
    graph = task.relation_runtime.graph
    relation = {'push_at': 7, 'holding_at': 7, 'holding_ontop': 8, 'sit': 9, 'climb': 10}[name]
    source, support = [], []
    rows = torch.arange(task.num_envs, device=task.device)
    for owner in (0, 1):
        mask = graph.edge_valid & (graph.edge_owner == owner) & (graph.edge_relation == relation)
        assert (mask.sum(-1) == 1).all()
        edge = mask.long().argmax(-1)
        src = graph.edge_src[rows, edge]-task.num_agents
        dst = graph.edge_dst[rows, edge]-task.num_agents
        if name in ('sit', 'climb'):
            src = dst
        source.append(src)
        support.append(dst if name in ('sit', 'climb', 'holding_ontop') else src)
    return torch.stack(source, -1), torch.stack(support, -1)


def capture(task):
    tensors = dict(root=task._rigid_body_pos[:, :, 0],
                   feet=task._rigid_body_pos[:, :, task._key_body_ids[[2, 3]]],
                   boxes=task._logical_box_values(task._box_states),
                   goals=task._tar_pos,
                   strict=task.relation_runtime.done)
    return {k: v.detach().cpu().numpy().copy() for k, v in tensors.items()}


def main():
    args = get_args()
    assert args.test and args.eval and args.headless
    assert args.task_graph in TASKS and args.num_agents == 2 and args.num_objects == 4
    cfg, train, _ = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['relationReward']['diagnostics']['sample_envs'] = 0
    cfg['args'] = args
    seed = set_seed(args.seed)
    train['params']['seed'] = train['params']['config']['seed'] = seed
    train['params']['config']['device'] = args.rl_device
    train['params']['config']['device_name'] = args.rl_device
    train['params']['config']['num_actors'] = cfg['env']['numEnvs']
    torch.set_num_threads(1)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        raise FileExistsError('Completed evaluation exists; use a fresh output path')
    checkpoint_sha = sha(args.checkpoint)
    atomic_json(out/'manifest.json', dict(task=args.task_graph, seed=seed,
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
        num_envs=cfg['env']['numEnvs'], num_agents=2, num_objects=4,
        episode_length=cfg['env']['episodeLength'],
        protocol='push_finetune_two_agent_position_v1',
        source_sha256={str(p):sha(p) for p in [Path(__file__),
            Path('tokenhsi/utils/finetune_evaluation.py'),
            Path('tokenhsi/env/tasks/multi_agent/humanoid_ma_stage1_push.py')]}))
    install_benchmark_assets()
    entry.args, entry.cfg, entry.cfg_train = args, cfg, train
    runner = entry.build_alg_runner(entry.RLGPUAlgoObserver())
    runner.load(train)
    runner.reset()
    player = runner.create_player()
    player._reward_debug = False
    player.restore(args.checkpoint)
    task = player.env.task
    try:
        with torch.no_grad():
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], 1)
            assert not task._rsi_interaction.any(), 'Evaluation must start from locomotion'
            n = task.num_envs
            src, sup = bindings(task, args.task_graph)
            src, sup = src.cpu().numpy(), sup.cpu().numpy()
            batch = np.arange(n)[:, None]
            sizes = task._logical_box_values(task._box_size).cpu().numpy().copy()
            initial = capture(task)
            initial.update(sizes=sizes, source=src, support=sup,
                humanoid_root_state=task._humanoid_root_states.cpu().numpy(),
                dof_pos=task._dof_pos.cpu().numpy(), dof_vel=task._dof_vel.cpu().numpy(),
                logical_box_order=task._logical_box_order.cpu().numpy(),
                **{k:getattr(task.relation_runtime.graph,k).cpu().numpy() for k in
                   ('edge_valid','edge_owner','edge_relation','edge_src','edge_dst','required_goal')})
            np.savez_compressed(out/'initial.npz', **initial)
            resolved = dict(env=task.cfg['env'], sim=task.cfg['sim'],
                evaluation_overrides=dict(at_sizes=AT_SIZES if args.task_graph=='holding_at' else None,
                    fixed_box_assets=args.task_graph in ('sit','climb'),
                    people=2,objects=4,policy_deterministic=True))
            (out/'resolved_env.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
            alive = np.ones(n, dtype=bool)
            ever = np.zeros((n, 2), dtype=bool)
            region_ever = ever.copy(); strict_ever = ever.copy()
            final = ever.copy(); region_final = ever.copy()
            terminated = np.zeros(n, dtype=bool)
            first = np.full((n, 2), np.nan)
            d0 = np.linalg.norm(initial['boxes'][batch,src,:2]-initial['goals'][...,:2],axis=-1)
            approach = np.linalg.norm(initial['boxes'][batch,src,:2]-initial['root'][...,:2],axis=-1)
            remaining = d0.copy(); best = np.zeros_like(d0)
            lengths = np.zeros(n, dtype=int)
            traces = {k:[] for k in ('root','feet','boxes','goals','strict','active','done','terminate','point','region')}
            resets = []
            start = time.time()
            for step in range(task.max_episode_length+1):
                obs = player.env_reset(resets)
                action = player.get_action(obs, True)
                obs, reward, done, info = player.env_step(player.env, action)
                snap = capture(task)
                for key in ('root','feet','boxes','goals'):
                    assert np.isfinite(snap[key]).all(), 'Nonfinite rollout '+key
                source = snap['boxes'][batch,src]
                support = snap['boxes'][batch,sup]
                point, region = classify(args.task_graph, snap['root'], source, support,
                    sizes[batch,src], sizes[batch,sup], snap['goals'])
                p32, r32 = classify(args.task_graph, snap['root'], source, support,
                    sizes[batch,src], sizes[batch,sup], snap['goals'], dtype=np.float32)
                # Save both precisions' ever results, checked at end instead of frame equality.
                if step == 0:
                    ever32 = np.zeros_like(ever); region32 = np.zeros_like(ever)
                valid = alive[:,None]
                ever |= point & valid; region_ever |= region & valid
                ever32 |= p32 & valid; region32 |= r32 & valid
                strict_ever |= snap['strict'] & valid
                first[(np.isnan(first)) & point & valid] = (step+1)*task.dt
                dist = np.linalg.norm(source[...,:2]-snap['goals'][...,:2],axis=-1)
                progress = np.clip((d0-dist)/np.maximum(d0,1e-6),0,1)
                best = np.where(valid, np.maximum(best,progress),best)
                ending = done.reshape(n,2).bool().any(-1).cpu().numpy()
                term = info['terminate'].reshape(n,2).bool().any(-1).cpu().numpy()
                collect = alive & ending
                final[collect] = point[collect]; region_final[collect] = region[collect]
                terminated[collect] = term[collect]
                remaining[collect] = dist[collect]; lengths[collect] = step+1
                if args.task_graph in ('sit','climb'):
                    np.testing.assert_allclose(source[alive,:,:7], initial['boxes'][batch,src][alive,:,:7],atol=1e-6,rtol=0)
                for key in ('root','feet','boxes','goals','strict'):
                    traces[key].append(snap[key])
                for key,value in dict(active=alive.copy(),done=ending,terminate=term,point=point,region=region).items():
                    traces[key].append(value)
                alive &= ~ending
                resets = ending.nonzero()[0].tolist()
                # VecTask expects flattened actor slot indices, not scene indices.
                resets = torch.as_tensor(resets,device=task.device,dtype=torch.long)*2
                if step % 100 == 0:
                    atomic_json(out/'progress.json',dict(step=step+1,alive_scenes=int(alive.sum()),
                        elapsed_seconds=time.time()-start))
                    print(f'[eval] task={args.task_graph} seed={seed} step={step+1} alive={alive.sum()}',flush=True)
                if not alive.any():
                    break
            assert not alive.any(), 'Evaluation exceeded episode horizon'
            np.testing.assert_array_equal(ever,ever32)
            np.testing.assert_array_equal(region_ever,region32)
            rows=[]
            for env in range(n):
                for owner in (0,1):
                    index=(env,owner)
                    rows.append(dict(env=env,owner=owner,
                        point_ever=bool(ever[index]),region_ever=bool(region_ever[index]),
                        point_final=bool(final[index]),region_final=bool(region_final[index]),
                        strict_ever=bool(strict_ever[index]),terminated=bool(terminated[env]),
                        steps=int(lengths[env]),initial_box_goal_distance_m=float(d0[index]),
                        initial_agent_box_distance_m=float(approach[index]),
                        final_box_goal_distance_m=float(remaining[index]),
                        progress_fraction=float(np.clip((d0[index]-remaining[index])/max(d0[index],1e-6),0,1)),
                        best_progress_fraction=float(best[index]),
                        first_point_success_seconds=None if np.isnan(first[index]) else float(first[index])))
            np.savez_compressed(out/'trace.npz', **{k:np.stack(v) for k,v in traces.items()},dt=task.dt)
            results=dict(task=args.task_graph,seed=seed,num_scene_trials=n,num_agents=2,
                agents=rows,summary=summarize_rows(rows),first_episode_only=True,
                float32_float64_ever_match=True,checkpoint_sha256=checkpoint_sha,
                elapsed_seconds=time.time()-start)
            if args.task_graph=='push_at':
                results['by_initial_box_goal_distance']=distance_bins(rows)
            assert float(task._sampling_failures)==0
            results['sampling_failures']=0
            results['artifact_sha256']={p.name:sha(p) for p in
                (out/'initial.npz',out/'trace.npz',out/'resolved_env.yaml')}
            assert sha(args.checkpoint)==checkpoint_sha
            atomic_json(out/'results.json',results)
            atomic_json(out/'validation.json',verify_archive(out))
            print('[evaluation complete]',json.dumps(results['summary']),flush=True)
    finally:
        task.gym.destroy_sim(task.sim)


if __name__=='__main__':
    main()
