"""Evaluation-only bridge: frozen Action actor in native TokenHSI single-human tasks.

No training configuration/checkpoint mutation. ON_TOP is a separately labeled
extension. Run from repository root after sourcing runtime_env.sh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaacgym import gymapi, gymtorch  # must precede torch
import numpy as np
import torch
import yaml
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from env.tasks.humanoid import compute_humanoid_observations_max
from env.tasks.multi_agent.scene_features import build_gta_pose_records
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
from learning.multi_agent import amp_network_builder_ma as nb
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils import torch_utils
from isaacgym.torch_utils import quat_rotate, quat_mul


class BenchmarkTask(HumanoidTrajSitCarryClimb):
    def __init__(self, *args, extension=False, static_mode='original', evaluation_kind=None,
                 approach_max_distance=None, **kwargs):
        self.extension = extension
        self.static_mode = static_mode
        self.evaluation_kind = evaluation_kind
        self.approach_max_distance = approach_max_distance
        super().__init__(*args, **kwargs)

    def _sample_task_object_xy(self, ids, max_distance):
        if self.approach_max_distance is not None:
            max_distance = min(max_distance, self.approach_max_distance)
        return super()._sample_task_object_xy(ids, max_distance)

    def _load_object_asset(self, paths):
        # Preserve original URDF/VHACD options. Optional GPU-reset compatibility:
        # effectively immovable dynamic bodies let PhysX relocate collision shapes.
        if self.static_mode != 'relocatable':
            return super()._load_object_asset(paths)
        gym = self.gym
        class Proxy:
            def __getattr__(self, name):
                return getattr(gym, name)
            def load_asset(self, sim, root, path, opts):
                opts.fix_base_link = False
                opts.disable_gravity = True
                opts.density = 1.e10
                opts.max_linear_velocity = 1.e-8
                opts.max_angular_velocity = 1.e-8
                return gym.load_asset(sim, root, path, opts)
        self.gym = Proxy()
        try:
            return super()._load_object_asset(paths)
        finally:
            self.gym = gym

    def _load_platform_asset(self):
        if self.static_mode != 'relocatable':
            super()._load_platform_asset()
        else:
            opts = gymapi.AssetOptions()
            opts.fix_base_link = False
            opts.disable_gravity = True
            opts.density = 1.e10
            opts.max_angular_velocity = 0.
            self._platform_height = .02
            self._platform_asset = self.gym.create_box(self.sim, .4, .4, .02, opts)
        if self.extension:
            # Match current support-box size rules, independently sampled axes.
            sizes = torch.stack([torch.randint(lo, hi+1, (self.num_envs,), device=self.device)/20.
                                 for lo, hi in [(10,16),(10,16),(5,9)]], -1)
            self.support_sizes = sizes
            self.support_assets = []
            for size in sizes.cpu().tolist():
                opts = gymapi.AssetOptions()
                opts.density = 100.
                opts.linear_damping = opts.angular_damping = .01
                opts.max_angular_velocity = 100.
                self.support_assets.append(self.gym.create_box(self.sim, *size, opts))

    def _build_platforms(self, env_id, env_ptr):
        if self.static_mode == 'fixed_scene':
            pose=gymapi.Transform();pose.p=gymapi.Vec3(50.,50.,20.)
            self._platform_handles.append(self.gym.create_actor(env_ptr,self._platform_asset,pose,'platform',env_id,0,0))
            self._tar_platform_handles.append(self.gym.create_actor(env_ptr,self._platform_asset,pose,'target',env_id,0,0))
            return
        if not self.extension and self.static_mode=='original':
            return super()._build_platforms(env_id, env_ptr)
        # Never construct dynamic substitutes below the plane: their initial
        # depenetration velocity would persist when the original reset sets XYZ only.
        pose = gymapi.Transform(); pose.p.z = 20.
        source = self.gym.create_actor(env_ptr, self._platform_asset, pose, 'platform', env_id, 0, 0)
        pose.p.z = 21.
        asset=self.support_assets[env_id] if self.extension else self._platform_asset
        target = self.gym.create_actor(env_ptr, asset, pose, 'support', env_id, 0, 0)
        self._platform_handles.append(source); self._tar_platform_handles.append(target)

    def _build_sit_object(self, env_id, env_ptr):
        if self.static_mode != 'fixed_scene' or self.evaluation_kind=='sit':
            return super()._build_sit_object(env_id,env_ptr)
        pose=gymapi.Transform();pose.p=gymapi.Vec3(50.,50.,20.)
        self._sit_object_handles.append(self.gym.create_actor(env_ptr,
            self._sit_object_assets[self._sit_obj_lib._every_env_object_ids[env_id]],pose,'object',env_id,0,0))

    def _build_climb_object(self, env_id, env_ptr):
        if self.static_mode != 'fixed_scene' or self.evaluation_kind=='climb':
            return super()._build_climb_object(env_id,env_ptr)
        pose=gymapi.Transform();pose.p=gymapi.Vec3(50.,50.,20.)
        self._climb_object_handles.append(self.gym.create_actor(env_ptr,
            self._climb_object_assets[self._climb_obj_lib._every_env_object_ids[env_id]],pose,'object',env_id,0,0))

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id,env_ptr,humanoid_asset)
        if self.static_mode == 'relocatable':
            # URDF <mass> overrides AssetOptions.density; set actor masses too.
            for handle in (self._sit_object_handles[-1],self._climb_object_handles[-1]):
                props=self.gym.get_actor_rigid_body_properties(env_ptr,handle)
                for prop in props: prop.mass=1.e10
                self.gym.set_actor_rigid_body_properties(env_ptr,handle,props,True)

    def _reset_task_carry(self, ids):
        super()._reset_task_carry(ids)
        if self.extension:
            state = self._tar_platform_states
            state[ids, :2] = self._box_tar_pos[ids, :2]
            state[ids, 2] = self.support_sizes[ids, 2]/2
            angle = torch.rand(len(ids), device=self.device)*2*np.pi
            state[ids, 3:7] = 0.
            state[ids, 5] = torch.sin(angle/2); state[ids, 6] = torch.cos(angle/2)
            state[ids, 7:] = 0.
            self._box_tar_pos[ids, 2] = self.support_sizes[ids, 2] + self._box_lib._box_size[ids, 2]/2

    def _reset_env_tensors(self, ids):
        if self.static_mode == 'fixed_scene':
            # A common SE(2) transform keeps the native sampled relative scene
            # unchanged while placing its fixed mesh at the creation transform.
            # No moving collider or approximate high-mass mesh is needed.
            sit=self._eval_task=='sit'
            state=self._sit_object_states if sit else self._climb_object_states
            old=state[ids,:3].clone();rotation=state[ids,3:7].clone();rotation[:,:3]*=-1
            anchor=old.clone();anchor[:,0]=1. if sit else 10.;anchor[:,1]=0.
            def point(p):return quat_rotate(rotation,p-old)+anchor
            if not hasattr(self,'bridge_scene_transform'):
                self.bridge_scene_transform=torch.zeros(self.num_envs,10,device=self.device)
            inverse=rotation.clone();inverse[:,:3]*=-1
            self.bridge_scene_transform[ids]=torch.cat([anchor,old,inverse],-1)
            human=self._humanoid_root_states
            human[ids,:3]=point(human[ids,:3])
            human[ids,3:7]=quat_mul(rotation,human[ids,3:7])
            human[ids,7:10]=quat_rotate(rotation,human[ids,7:10])
            human[ids,10:13]=quat_rotate(rotation,human[ids,10:13])
            body=self._kinematic_humanoid_rigid_body_states[ids].clone()
            count=body.shape[1];rot=rotation[:,None].expand(-1,count,-1).reshape(-1,4)
            body[:,:,:3]=quat_rotate(rot,(body[:,:,:3]-old[:,None]).reshape(-1,3)).reshape(-1,count,3)+anchor[:,None]
            body[:,:,3:7]=quat_mul(rot,body[:,:,3:7].reshape(-1,4)).reshape(-1,count,4)
            for start in (7,10):body[:,:,start:start+3]=quat_rotate(rot,body[:,:,start:start+3].reshape(-1,3)).reshape(-1,count,3)
            self._kinematic_humanoid_rigid_body_states[ids]=body
            goal=self._sit_tar_pos if sit else self._climb_tar_pos
            goal[ids]=point(goal[ids])
            state[ids,:3]=anchor;state[ids,3:]=0.;state[ids,6]=1.
            inactive=self._climb_object_states if sit else self._sit_object_states
            for hidden in (inactive,self._platform_states,self._tar_platform_states,self._box_states):
                hidden[ids,:3]=hidden.new_tensor([50.,50.,20.]);hidden[ids,7:]=0.
        if self.static_mode == 'relocatable':
            for state in (self._platform_states,self._tar_platform_states):
                state[ids,7:]=0.
            self._platform_states[ids,3:7]=0.
            self._platform_states[ids,6]=1.
            if not self.extension:
                self._tar_platform_states[ids,3:7]=0.
                self._tar_platform_states[ids,6]=1.
            # Dynamic substitutes must not be hidden below the ground: contact
            # depenetration would propel those otherwise inactive mesh actors up.
            for name,state in [('sit',self._sit_object_states),('climb',self._climb_object_states)]:
                inactive=ids[self._task_indicator[ids]!=self.TaskUID[name].value]
                state[inactive,:3]=state.new_tensor([50.,50.,20.])
                state[inactive,7:]=0.
        super()._reset_env_tensors(ids)

    def _compute_metrics_evaluation_v3(self):
        if self.extension:
            support = self._tar_platform_states
            self._box_tar_pos[:, :2] = support[:, :2]
            self._box_tar_pos[:, 2] = support[:, 2] + vertical_extent(
                support[:, 3:7], self.support_sizes/2) + self._box_lib._box_size[:, 2]/2
        super()._compute_metrics_evaluation_v3()


def load_actor(checkpoint, device, object_count):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    env_path = Path('tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action.yaml')
    train_path = Path('tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_action.yaml')
    cfg = yaml.safe_load(env_path.read_text())['env']
    train = yaml.safe_load(train_path.read_text())
    builder = nb.AMPMultiAgentBuilder(); builder.load(train['params']['network'])
    net = builder.build('amp', actions_num=32, input_shape=(634,), amp_input_shape=(1320,), value_size=1,
        num_agents=2, num_objects=4, humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6,
        observation_mode='clean_scene', scene_entity_sizes=[223,30,1], scene_kinematic_size=7,
        scene_arena_scale=5., relation_reward_mode=cfg['relationReward']['mode'],
        relation_graph_spec=cfg['relationGraph'], device='cpu')
    net.load_state_dict({k[12:]:v for k,v in saved['model'].items() if k.startswith('a2c_network.')}, strict=True)
    norm = SceneRunningMeanStd([223,30,1], [2,4,2], [223,30,0], 7, 10)
    norm.load_state_dict(saved['running_mean_std'], strict=True)
    # Only counts change; all learned weights and normalizer statistics are retained.
    enc = net.actor_encoder
    enc.num_agents = 1; enc.num_objects = object_count
    enc.entity_counts = [1,object_count,1]; enc.suffix_width = 5
    enc.entity_types = nb.build_entity_type_ids(1,object_count)
    enc.rel_matrix = nb.build_relation_matrix(1,object_count); enc._set_context_background()
    enc.gta_diagnostics_first_forward = False
    norm.entity_counts = [1,object_count,1]; norm.extra_passthrough_size = 5
    net.to(device).eval(); norm.to(device).eval()
    return net,norm,dict(path=str(checkpoint), epoch=int(saved['epoch']), frame=int(saved['frame']),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        observation='original humanoid state; actual asset full bbox; no seat-specific goal feature',
        counts=[1,object_count,1], strict_weight_load=True)


def object_state(task, kind):
    if kind == 'sit':
        return task._sit_object_states[:,None], task._sit_obj_lib._every_env_object_bbox_lengths[:,None]
    if kind == 'climb':
        return task._climb_object_states[:,None], task._climb_obj_lib._every_env_object_bbox_lengths[:,None]
    state = task._box_states[:,None]; sizes = task._box_lib._box_size[:,None]
    if kind == 'ontop':
        state = torch.cat([state,task._tar_platform_states[:,None]],1)
        sizes = torch.cat([sizes,task.support_sizes[:,None]],1)
    return state,sizes


def observation(task, kind, initial=False):
    if initial:
        body = task._kinematic_humanoid_rigid_body_states
    else:
        body = torch.cat([task._rigid_body_pos,task._rigid_body_rot,
                          task._rigid_body_vel,task._rigid_body_ang_vel],-1)
    state,size = object_state(task,kind); n,o = state.shape[:2]
    if hasattr(task,'bridge_scene_transform'):
        # Return every model feature to the original sampled world frame. In
        # particular the neutral goal's identity frame must remain unchanged.
        transform=task.bridge_scene_transform
        def restore(value):
            value=value.clone();count=value.shape[1]
            rot=transform[:,None,6:10].expand(-1,count,-1).reshape(-1,4)
            value[:,:,:3]=quat_rotate(rot,(value[:,:,:3]-transform[:,None,:3]).reshape(-1,3)).reshape(n,count,3)+transform[:,None,3:6]
            value[:,:,3:7]=quat_mul(rot,value[:,:,3:7].reshape(-1,4)).reshape(n,count,4)
            for start in (7,10):value[:,:,start:start+3]=quat_rotate(rot,value[:,:,start:start+3].reshape(-1,3)).reshape(n,count,3)
            return value
        body=restore(body);state=restore(state)
    human = compute_humanoid_observations_max(body[:,:,:3],body[:,:,3:7],body[:,:,7:10],body[:,:,10:13],True,True)
    assert human.shape == (task.num_envs,223)
    signs = state.new_tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],
                             [1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]])
    bps = size[:,:,None]/2*signs[None,None]
    inv = state[:,:,3:7].clone(); inv[:,:,:3] *= -1
    objects = torch.cat([quat_rotate(inv.reshape(-1,4),state[:,:,7:10].reshape(-1,3)).reshape(n,o,3),
        quat_rotate(inv.reshape(-1,4),state[:,:,10:13].reshape(-1,3)).reshape(n,o,3),bps.flatten(2)],-1)
    goal = task._box_tar_pos[:,None] if kind=='carry' else state.new_zeros(n,1,3)
    heading = torch_utils.calc_heading_quat(body[:,0,3:7])[:,None]
    # All original task tensors are in each environment's local coordinate frame.
    poses = build_gta_pose_records(body[:,0,:3][:,None],heading,state[:,:,:3],state[:,:,3:7],
                                  goal,state.new_zeros(n,3),goal_rotation='identity')
    task_id = {'sit':0,'climb':1,'carry':2,'ontop':3}[kind]
    target = 2 if kind in ('carry','ontop') else 1
    packet = state.new_tensor([1,task_id,0,1 if task_id>=2 else 0,target]).expand(n,-1)
    raw = torch.cat([human,objects.flatten(1),state.new_ones(n,1),poses.flatten(1),packet],-1)
    assert torch.isfinite(raw).all()
    return raw


def distance(task,kind):
    if kind=='sit': return (task._humanoid_root_states[:,:3]-task._sit_tar_pos).norm(dim=-1)
    if kind=='climb': return (task._humanoid_root_states[:,:3]-task._climb_tar_pos).norm(dim=-1)
    return (task._box_states[:,:3]-task._box_tar_pos).norm(dim=-1)


def contact_probe(task):
    """Drop a box onto each relocated source plate; fail on collider/reset mismatch."""
    task.reset()
    task._humanoid_root_states[:,:2] = 50.
    task._platform_states[:,:] = 0.; task._platform_states[:,6] = 1.
    task._platform_pos[:,:2] = 0.; task._platform_pos[:,2] = .59
    task._box_states[:,:] = 0.; task._box_states[:,6] = 1.
    task._box_states[:,2] = .6 + task._box_lib._box_size[:,2]/2 + .05
    task._sit_object_states[:,:3] = 50.; task._climb_object_states[:,:3] = 50.
    task._tar_platform_states[:,:3] = 40.
    task.gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
    for _ in range(45):
        task.gym.simulate(task.sim); task.gym.fetch_results(task.sim,True)
    task._refresh_sim_tensors()
    error=(task._box_states[:,2]-task._box_lib._box_size[:,2]/2-.6).abs()
    result=dict(max_bottom_error_m=error.max().item(), passed=bool((error<.015).all()),
        box=task._box_states[:,:3].cpu().tolist(),plate=task._platform_states[:,:3].cpu().tolist())
    print('CONTACT_PROBE',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--task',choices=['carry','sit','climb','ontop'],required=True)
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--approach-max-distance',type=float,
                        help='Restrict initial human-object XY radius to [1, max] metres; goal sampling unchanged.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--static-mode',choices=['original','relocatable','fixed_scene'],default='original')
    parser.add_argument('--probe-only',action='store_true')
    opt=parser.parse_args(); opt.output.mkdir(parents=True,exist_ok=True)
    assert opt.checkpoint.is_file() and opt.num_envs>0 and opt.repeats>0
    if opt.approach_max_distance is not None and not (1. < opt.approach_max_distance <= 10.):
        parser.error('--approach-max-distance must be greater than 1 and at most 10 metres')
    assert opt.static_mode!='fixed_scene' or opt.task in ('sit','climb')
    sys.argv=[sys.argv[0],'--task','HumanoidTrajSitCarryClimb','--cfg_env',
        'tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml',
        '--cfg_train','tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml',
        '--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml','--test','--eval',
        '--eval_task', 'carry' if opt.task=='ontop' else opt.task,
        '--num_envs',str(opt.num_envs),'--seed',str(opt.seed),'--headless','--no_video']
    args=get_args(); cfg,train,_=load_cfg(args)
    cfg['env']['motion_file']=args.motion_file
    set_seed(opt.seed); torch.set_num_threads(2)
    print('CUDA_FREE',torch.cuda.mem_get_info(),flush=True)
    sim_params=parse_sim_params(args,cfg,train)
    task=BenchmarkTask(cfg=cfg,sim_params=sim_params,physics_engine=args.physics_engine,
        device_type=args.device,device_id=args.device_id,headless=True,
        extension=opt.task=='ontop',static_mode=opt.static_mode,evaluation_kind=opt.task,
        approach_max_distance=opt.approach_max_distance)
    if opt.probe_only:
        probe=contact_probe(task)
        (opt.output/'probe.json').write_text(json.dumps(probe,indent=2))
        if not probe['passed']: raise RuntimeError('Relocated platform collider failed contact probe')
        return
    net,norm,metadata=load_actor(opt.checkpoint,task.device,2 if opt.task=='ontop' else 1)
    manifest=dict(task=opt.task,seed=opt.seed,num_envs=opt.num_envs,repeats=opt.repeats,
        static_mode=opt.static_mode,checkpoint=metadata,
        protocol='Native TokenHSI single-human task/reset/metrics; Action observation bridge. '
                 'ON_TOP only: dynamic current-size support box replaces destination platform.',
        horizons=dict(carry=600,sit=300,climb=300,ontop=600),
        limitations=['Action sees bbox only, no mesh/seat point feature',
                     'one human rather than training-time two; reduced token count',
                     'local Climb test split has 35 assets; paper reports 26',
                     'small evaluation sample; no TokenHSI policy baseline'])
    manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['native_task_sha256']=hashlib.sha256(Path(
        'tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py').read_bytes()).hexdigest()
    native_max=5. if opt.task=='sit' else 10.
    approach_max=native_max if opt.approach_max_distance is None else min(native_max,opt.approach_max_distance)
    manifest['approach_distance_range_m']=[1.,approach_max]
    manifest['static_actor_implementation']={'original':'original fixed',
        'relocatable':'gravity-disabled massive dynamic substitutes; inactive meshes moved away',
        'fixed_scene':'original fixed mesh; common SE(2) scene transform preserves sampled relative positions/orientations'}[opt.static_mode]
    if opt.task in ('sit','climb'):
        lib=task._sit_obj_lib if opt.task=='sit' else task._climb_obj_lib
        manifest['asset_paths']=[lib._obj_urdfs[i] for i in lib._every_env_object_ids.cpu().tolist()]
    else: manifest['box_sizes']=task._box_lib._box_size.cpu().tolist()
    if opt.task=='ontop':manifest['support_sizes']=task.support_sizes.cpu().tolist()
    (opt.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[]
    with torch.inference_mode():
        for repeat in range(opt.repeats):
            task.reset(); n=task.num_envs
            state,size=object_state(task,opt.task)
            initial_distance=(state[:,0,:2]-task._humanoid_root_states[:,:2]).norm(dim=-1)
            assert ((initial_distance>=1.-1e-4)&(initial_distance<=approach_max+1e-4)).all()
            static_initial=state[:,0,:3].clone() if opt.task in ('sit','climb') else None
            static_max_drift=0.
            torch.save(dict(root=task._humanoid_root_states.cpu(),dof=task._dof_pos.cpu(),
                objects=state.cpu(),sizes=size.cpu(),target=(task._sit_tar_pos if opt.task=='sit' else
                task._climb_tar_pos if opt.task=='climb' else task._box_tar_pos).cpu(),
                initial_obs=observation(task,opt.task,True).cpu()),opt.output/f'initial_{repeat}.pt')
            alive=torch.ones(n,dtype=torch.bool,device=task.device)
            ever=torch.zeros_like(alive); final=torch.zeros_like(alive); ended=torch.zeros_like(alive)
            minimum=torch.full((n,),float('inf'),device=task.device)
            enddist=torch.zeros(n,device=task.device); duration=torch.zeros(n,device=task.device)
            trace=[]
            for step in range(task.max_episode_length+1):
                raw=observation(task,opt.task,step==0)
                mu=net.action_head(net.actor_encoder(norm(raw)))[:,0]
                assert mu.shape==(n,32) and torch.isfinite(mu).all()
                task.step(mu.clamp(-1,1))
                d=distance(task,opt.task); current=d<=.2
                assert torch.isfinite(d).all()
                ever|=current&alive; minimum=torch.where(alive,torch.minimum(minimum,d),minimum)
                # Independent distance/accumulator calculation must match original evaluator.
                assert torch.equal(ever[alive],(task._success_buf>0)[alive])
                trace.append(torch.stack([d,alive.float(),current.float(),task._terminate_buf.float(),
                    task._humanoid_root_states[:,2],task._box_states[:,2],
                    task._box_states[:,7:10].norm(dim=-1)],-1).cpu())
                if static_initial is not None:
                    moved=(object_state(task,opt.task)[0][:,0,:3]-static_initial).norm(dim=-1)
                    static_max_drift=max(static_max_drift,moved[alive].max().item())
                    assert static_max_drift<.005,('Static target drift',static_max_drift)
                collect=task.reset_buf.bool()&alive
                passing=collect&ever
                assert torch.allclose(minimum[passing],task._precision_buf[passing],atol=1e-6)
                final[collect]=current[collect];enddist[collect]=d[collect]
                ended[collect]=task._terminate_buf[collect].bool();duration[collect]=(step+1)*task.dt
                alive&=~collect
                if not alive.any():break
            assert not alive.any()
            row=dict(repeat=repeat,trials=n,successes=int(ever.sum()),success_rate=ever.float().mean().item(),
                final_successes=int(final.sum()),final_success_rate=final.float().mean().item(),
                success_precision_m=minimum[ever].mean().item() if ever.any() else None,
                terminated=int(ended.sum()),failure_terminated=int((ended&~ever).sum()),
                seconds_mean=duration.mean().item(),median_final_distance_m=enddist.median().item())
            row['static_max_drift_m']=static_max_drift
            row['initial_distance_min_m']=initial_distance.min().item()
            row['initial_distance_max_m']=initial_distance.max().item()
            torch.save(dict(trace=torch.stack(trace),minimum=minimum.cpu(),final_distance=enddist.cpu(),
                duration=duration.cpu()),opt.output/f'trace_{repeat}.pt')
            results.append(row);(opt.output/'results.json').write_text(json.dumps(results,indent=2))
            print('BENCHMARK_RESULT',json.dumps(row),flush=True)
    summary=dict(trials=sum(r['trials'] for r in results),successes=sum(r['successes'] for r in results),
        final_successes=sum(r['final_successes'] for r in results),
        success_rate_mean=float(np.mean([r['success_rate'] for r in results])),
        success_rate_std=float(np.std([r['success_rate'] for r in results],ddof=1)) if len(results)>1 else None,
        validation='strict weights/normalizer, finite observations/actions, native success accumulator and precision match')
    (opt.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('BENCHMARK_SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
