"""One person; exact saved policy parameters; common archived initial states."""
import isaacgym  # before torch
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import torch
import yaml
from utils.config import get_args, set_seed, parse_sim_params
from utils.single_benchmark import SingleBenchmarkEnv, save_initial_plan
from utils.finetune_evaluation import classify, atomic_json, summarize_rows, distance_bins, verify_archive
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class FrozenPolicy:
    def __init__(self, weights, task):
        saved=weights['relation_experiment_config']
        builder=AMPMultiAgentBuilder();builder.load(saved['train']['params']['network'])
        assert all(k.startswith('a2c_network.') for k in weights['model'])
        model={k[len('a2c_network.'):]:v for k,v in weights['model'].items()}
        amp_width=model['_disc_mlp.0.weight'].shape[-1]
        self.net=builder.build('amp',actions_num=task.num_dof,input_shape=(saved['observation_size'],),
            amp_input_shape=(amp_width,),value_size=1,num_agents=2,num_objects=4,
            humanoid_obs_size=task.get_humanoid_obs_size(),object_obs_size=task.get_object_obs_size(),
            goal_obs_size=task.get_goal_obs_size(),observation_mode='clean_scene',
            scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=7,
            scene_arena_scale=task.get_scene_arena_scale(),relation_reward_mode=task._relation_cfg['mode'],
            relation_graph_spec=saved['env']['relationGraph'],device=task.device)
        self.net.load_state_dict(model,strict=True)
        self.net.set_task_graph(task._relation_graph_spec)
        self.net.set_entity_counts(1,task.num_objects)
        assert all(torch.equal(p.detach().cpu(),model[name]) for name,p in self.net.named_parameters())
        self.net=self.net.to(task.device).eval()
        self.norm=SceneRunningMeanStd(task.get_scene_entity_sizes(),[1,task.num_objects,1],
            normalized_sizes=task.get_scene_normalized_entity_sizes(),kinematic_size=7,
            extra_passthrough_size=5).to(task.device)
        self.norm.load_state_dict(weights['running_mean_std'],strict=True);self.norm.eval()

    @torch.no_grad()
    def action(self, obs):
        return self.net.eval_actor(self.norm(obs))[0].clamp(-1,1)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--benchmark-task',choices=('holding_at','holding_ontop','sit','climb','push_at'),required=True)
    parser.add_argument('--reference-checkpoint',required=True)
    parser.add_argument('--initial-plan',required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--push-start-mode',choices=('direct','around'))
    parser.add_argument('--visualize',action='store_true')
    parser.add_argument('--plan-env-index',type=int)
    special,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining
    args=get_args()
    assert args.test and args.eval and args.num_agents==1
    assert args.headless or (special.visualize and args.num_envs==1)
    expected_objects=2 if special.benchmark_task=='holding_ontop' else 1
    assert args.num_objects==expected_objects
    reference=torch.load(special.reference_checkpoint,map_location='cpu',weights_only=False)
    assert reference['epoch']==78000
    saved=reference['relation_experiment_config']
    env=deepcopy(saved['env'])
    env.update(numEnvs=args.num_envs,numAgents=1,numObjects=expected_objects,
        episodeLength=args.episode_length,mode='test',motion_file=args.motion_file)
    env['eval']['skill']=['loco'];env['eval']['skillInitProb']=[1.]
    env['video']['enable']=False
    assert not env.get('enableTrackInitState',False), 'Policy evaluation must not hold initial DOFs'
    env['relationReward']['diagnostics']['sample_envs']=0
    args.task_graph='holding_at' if special.benchmark_task=='push_at' else special.benchmark_task
    cfg=dict(env=env,sim=deepcopy(saved['sim']),args=args,seed=args.seed,headless=args.headless,
        task={'randomize':False},device_type=args.device,device_id=args.device_id)
    seed=set_seed(args.seed);torch.set_num_threads(1)
    sim=parse_sim_params(args,cfg,saved['train'])
    plan_path=Path(special.initial_plan)
    plan=None
    if not special.prepare_only:
        with np.load(plan_path) as z:plan={k:z[k].copy() for k in z.files}
        if special.plan_env_index is not None:
            assert special.visualize and args.num_envs==1
            count=len(plan['physical_boxes']);index=special.plan_env_index
            assert 0<=index<count
            plan={k:(v[index:index+1].copy() if v.ndim and len(v)==count else v)
                  for k,v in plan.items()}
        assert len(plan['physical_boxes'])==args.num_envs
        if special.push_start_mode:
            assert special.benchmark_task=='push_at'
            from utils.push_start_evaluation import validate_start_plan
            validate_start_plan(plan,special.push_start_mode)
    task=SingleBenchmarkEnv(cfg,sim,args.physics_engine,args.device,args.device_id,args.headless,
        benchmark_task=special.benchmark_task,initial_plan=plan)
    wrapper=VecTaskPythonWrapperMA(task,args.rl_device,
        saved['train']['params'].get('clip_observations',np.inf),1.)
    try:
        with torch.no_grad():
            obs=wrapper.reset(torch.arange(task.num_envs,device=task.device))
            assert task.num_agents==1 and task.num_objects==expected_objects
            assert all(len(handles)==1 for handles in task.humanoid_handles_all)
            assert task.get_scene_num_tokens()==2+expected_objects
            assert task.get_obs_size()==(317 if expected_objects==2 else 280)
            if special.prepare_only:
                plan_path.parent.mkdir(parents=True,exist_ok=True)
                save_initial_plan(task,plan_path)
                print('[single initial plan]',special.benchmark_task,seed,task.num_agents,task.num_objects,flush=True)
                return
            out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
            if (out/'results.json').exists():raise FileExistsError(out/'results.json')
            weights=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            policy=FrozenPolicy(weights,task)
            atomic_json(out/'manifest.json',dict(protocol='single_match_region_v1',
                task=special.benchmark_task,seed=seed,people=1,objects=task.num_objects,
                num_envs=task.num_envs,episode_length=task.max_episode_length,
                observation_width=task.get_obs_size(),tokens=task.get_scene_num_tokens(),
                actual_humanoid_actors_per_env=1,track_initial_dofs=False,
                push_start_mode=special.push_start_mode,
                plan_env_index=special.plan_env_index,visualize=special.visualize,
                checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha(args.checkpoint),
                original_reference_sha256=sha(special.reference_checkpoint),initial_plan_sha256=sha(plan_path),
                policy_parameters_changed=False,entity_counts_only_retargeted=True,
                source_sha256={str(p):sha(p) for p in [Path(__file__),Path('tokenhsi/utils/single_benchmark.py'),
                    Path('tokenhsi/utils/finetune_evaluation.py')]}))
            (out/'resolved_env.yaml').write_text(yaml.safe_dump(dict(env=env,sim=cfg['sim']),sort_keys=False))
            n=task.num_envs;batch=np.arange(n)[:,None]
            g=task.relation_runtime.graph
            relation={'holding_at':7,'push_at':7,'holding_ontop':8,'sit':9,'climb':10}[special.benchmark_task]
            mask=g.edge_valid & (g.edge_relation==relation)
            assert (mask.sum(-1)==1).all()
            edge=mask.long().argmax(-1);rr=torch.arange(n,device=task.device)
            src=(g.edge_src[rr,edge]-1).cpu().numpy()[:,None]
            dst=(g.edge_dst[rr,edge]-1).cpu().numpy()[:,None]
            if special.benchmark_task in ('sit','climb'):src=dst
            sup=dst if special.benchmark_task in ('sit','climb','holding_ontop') else src
            sizes=task._logical_box_values(task._box_size).cpu().numpy().copy()
            def capture():
                return {k:v.cpu().numpy().copy() for k,v in dict(root=task._rigid_body_pos[:,:,0],
                    feet=task._rigid_body_pos[:,:,task._key_body_ids[[2,3]]],
                    boxes=task._logical_box_values(task._box_states),goals=task._tar_pos,
                    strict=task.relation_runtime.done).items()}
            initial=capture();initial.update(sizes=sizes,source=src,support=sup,**plan)
            # GPU rigid-body tensors are stale until the first simulate. The
            # actual reset root/kinematic archive is the initial state, not the
            # previous creation pose still present in the rigid-body buffer.
            initial['root']=task._humanoid_root_states[...,:3].cpu().numpy().copy()
            initial['feet']=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[[2,3]],:3].cpu().numpy().copy()
            # Standard scorer keys are logical arrays; physical arrays stay separate.
            np.savez_compressed(out/'initial.npz',**initial)
            d0=np.linalg.norm(initial['boxes'][batch,src,:2]-initial['goals'][...,:2],axis=-1)
            approach=np.linalg.norm(initial['boxes'][batch,src,:2]-initial['root'][...,:2],axis=-1)
            alive=np.ones(n,dtype=bool);ever=np.zeros((n,1),dtype=bool)
            region_ever=ever.copy();strict_ever=ever.copy();ever32=ever.copy();region32=ever.copy()
            final=ever.copy();region_final=ever.copy();terminated=np.zeros(n,dtype=bool)
            first=np.full((n,1),np.nan);lengths=np.zeros(n,dtype=int)
            remaining_distance=d0.copy();best=np.zeros_like(d0)
            traces={k:[] for k in ('root','feet','boxes','goals','strict','active','done','terminate','point','region')}
            start=time.time()
            for step in range(task.max_episode_length):
                action=policy.action(obs)
                obs,_,done,info=wrapper.step(action)
                if special.visualize:
                    task.render()
                snap=capture()
                assert all(np.isfinite(snap[k]).all() for k in ('root','feet','boxes','goals'))
                source=snap['boxes'][batch,src];support=snap['boxes'][batch,sup]
                arguments=(special.benchmark_task,snap['root'],source,support,sizes[batch,src],sizes[batch,sup],snap['goals'])
                point,region=classify(*arguments);p32,r32=classify(*arguments,dtype=np.float32)
                valid=alive[:,None]
                ever|=point&valid;region_ever|=region&valid;ever32|=p32&valid;region32|=r32&valid
                strict_ever|=snap['strict']&valid
                first[np.isnan(first)&point&valid]=(step+1)*task.dt
                distance=np.linalg.norm(source[...,:2]-snap['goals'][...,:2],axis=-1)
                progress=np.clip((d0-distance)/np.maximum(d0,1e-6),0,1)
                best=np.where(valid,np.maximum(best,progress),best)
                ending=done.bool().cpu().numpy();term=info['terminate'].bool().cpu().numpy()
                collect=alive&ending
                final[collect]=point[collect];region_final[collect]=region[collect]
                terminated[collect]=term[collect];lengths[collect]=step+1;remaining_distance[collect]=distance[collect]
                if special.benchmark_task in ('sit','climb'):
                    np.testing.assert_allclose(source[alive,:,:7],initial['boxes'][batch,src][alive,:,:7],atol=1e-5,rtol=0)
                for k in ('root','feet','boxes','goals','strict'):traces[k].append(snap[k])
                for k,v in dict(active=alive.copy(),done=ending,terminate=term,point=point,region=region).items():traces[k].append(v)
                alive &= ~ending
                if step%100==0:
                    atomic_json(out/'progress.json',dict(step=step+1,alive=int(alive.sum()),elapsed_seconds=time.time()-start))
                    print('[single eval]',special.benchmark_task,seed,step+1,int(alive.sum()),flush=True)
                if not alive.any():break
            assert not alive.any()
            np.testing.assert_array_equal(ever,ever32);np.testing.assert_array_equal(region_ever,region32)
            rows=[]
            for i in range(n):
                rows.append(dict(env=i,owner=0,point_ever=bool(ever[i,0]),region_ever=bool(region_ever[i,0]),
                    point_final=bool(final[i,0]),region_final=bool(region_final[i,0]),strict_ever=bool(strict_ever[i,0]),
                    terminated=bool(terminated[i]),steps=int(lengths[i]),initial_box_goal_distance_m=float(d0[i,0]),
                    initial_agent_box_distance_m=float(approach[i,0]),final_box_goal_distance_m=float(remaining_distance[i,0]),
                    progress_fraction=float(np.clip((d0[i,0]-remaining_distance[i,0])/max(d0[i,0],1e-6),0,1)),
                    best_progress_fraction=float(best[i,0]),first_point_success_seconds=None if np.isnan(first[i,0]) else float(first[i,0])))
            np.savez_compressed(out/'trace.npz',**{k:np.stack(v) for k,v in traces.items()},dt=task.dt)
            result=dict(task=special.benchmark_task,seed=seed,num_scene_trials=n,num_agents=1,agents=rows,
                summary=summarize_rows(rows),float32_float64_ever_match=True,first_episode_only=True,
                checkpoint_sha256=sha(args.checkpoint),initial_plan_sha256=sha(plan_path),elapsed_seconds=time.time()-start)
            if special.benchmark_task=='push_at':result['by_initial_box_goal_distance']=distance_bins(rows)
            atomic_json(out/'results.json',result);atomic_json(out/'validation.json',verify_archive(out))
            print('[single complete]',json.dumps(result['summary']),flush=True)
    finally:
        task.gym.destroy_sim(task.sim)


if __name__=='__main__':main()
