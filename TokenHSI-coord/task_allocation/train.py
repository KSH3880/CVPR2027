"""Semi-Markov PPO for periodic matching; ms18 is inference-only."""
import os
import sys
import json
from pathlib import Path
from isaacgym import gymapi  # before torch
import torch

COORD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COORD/'tokenhsi'))
import run as tokenhsi_run
import utils.parse_task as registry
from utils.config import get_args, load_cfg, set_seed
from task_allocation.env import HumanoidTaskAllocationMS18
from task_allocation.core import AllocationPolicy, SCHEMA, assignment_from_action, switch_cost, gae, nearest_initial_action
registry.HumanoidTaskAllocationMS18 = HumanoidTaskAllocationMS18
from task_allocation.four_env import HumanoidFourBoxAllocationMS18
from task_allocation.four_core import FourBoxPolicy, SCHEMA_FOUR
registry.HumanoidFourBoxAllocationMS18 = HumanoidFourBoxAllocationMS18


def option(name, default, kind=float):
    return kind(os.environ.get('ALLOC_'+name, default))


from task_allocation.learning import macro_step, ppo_update


def main():
    args=get_args()
    cfg,cfg_train,_=load_cfg(args)
    seed=set_seed(cfg_train['params'].get('seed',0),False)
    cfg_train['params']['seed']=seed
    cfg_train['params']['config'].update(seed=seed,train_dir=args.output_path)
    cfg['env']['motion_file']=args.motion_file
    tokenhsi_run.args,tokenhsi_run.cfg,tokenhsi_run.cfg_train=args,cfg,cfg_train
    runner=tokenhsi_run.build_alg_runner(tokenhsi_run.RLGPUAlgoObserver())
    runner.load(cfg_train); runner.reset()
    player=runner.create_player(); player.restore(args.checkpoint)
    player.model.eval().requires_grad_(False)
    player.env.reset()
    task=player.env.task
    boxes=option('BOXES','2',int)
    if boxes not in (2,4): raise ValueError('ALLOC_BOXES must be 2 or 4')
    if (boxes==4) != isinstance(task,HumanoidFourBoxAllocationMS18):
        raise ValueError('box count does not match physical environment')
    policy_class=FourBoxPolicy if boxes==4 else AllocationPolicy
    schema=SCHEMA_FOUR if boxes==4 else SCHEMA
    policy=policy_class(option('D_MODEL','64',int)).to(task.device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=option('LR','0.0003'))
    steps,horizon,iters=option('INTERVAL','30',int),option('HORIZON','16',int),option('ITERS','100',int)
    epochs,batch=option('EPOCHS','4',int),option('MINIBATCH','256',int)
    gamma,lam=option('GAMMA','0.999'),option('LAMBDA','0.995')
    costs=(option('TIME_COEF','1'),option('DELIVERY_COEF','10'),option('FAILURE_COEF','40'))
    switch=option('SWITCH_COEF','0.1')
    if boxes==4 and switch!=0: raise ValueError('four-box jobs are non-preemptive; switch cost must be zero')
    save_every=option('SAVE_EVERY','10',int)
    mode=os.environ.get('ALLOC_MODE','train')
    if mode not in ('train','eval') or min(steps,horizon,iters,epochs,batch,save_every)<=0 or not 0<gamma<=1 or not 0<=lam<=1 or min(*costs,switch)<0:
        raise ValueError('invalid allocation training configuration')
    contract=dict(interval=steps,gamma=gamma,lam=lam,costs=costs,switch_coef=switch,
                  d_model=option('D_MODEL','64',int),extent=option('EXTENT','3'),
                  clearance=option('CLEARANCE','1.2'),episode_steps=task.max_episode_length)
    baseline=os.environ.get('ALLOC_BASELINE','none')
    if baseline not in ('none','nearest_initial') or (mode=='train' and baseline!='none'):
        raise ValueError('baseline is evaluation-only')
    if boxes==4 and baseline!='none':
        raise ValueError('nearest_initial baseline applies only to two boxes')
    start=0
    initial=os.environ.get('ALLOC_INIT','')
    if initial:
        ck=torch.load(initial,map_location=task.device)
        if ck['schema']!=schema:
            raise ValueError('allocation checkpoint schema mismatch')
        if ck['config']['executor'] != str(Path(args.checkpoint).resolve()) or ck['config']['stage1'] != str(Path(args.hrl_checkpoint).resolve()):
            raise ValueError('executor checkpoint contract mismatch')
        for key,expected in contract.items():
            stored=ck['config'].get(key)
            if key=='costs': stored=tuple(stored) if stored is not None else None
            if stored!=expected:
                raise ValueError(f'allocation checkpoint configuration mismatch: {key}')
        policy.load_state_dict(ck['policy'])
        if mode=='train':
            optimizer.load_state_dict(ck['optimizer']); start=ck['iteration']
    elif mode=='eval':
        raise ValueError('eval requires ALLOC_INIT')
    out=Path(os.environ['ALLOC_OUTPUT'])
    config=dict(interval=steps,horizon=horizon,gamma=gamma,lam=lam,seed=seed,costs=costs,switch_coef=switch,
                d_model=option('D_MODEL','64',int),executor=str(Path(args.checkpoint).resolve()),
                stage1=str(Path(args.hrl_checkpoint).resolve()),mode=mode,
                extent=contract['extent'],clearance=contract['clearance'],episode_steps=task.max_episode_length,baseline=baseline)
    config['boxes']=boxes
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('ALLOCATION_SETUP '+json.dumps(dict(boxes=boxes,mps=option('MPS','1',int),schema=schema)),flush=True)
    policy.eval()  # no stochastic dropout; sample only the joint action
    verify=bool(option('VERIFY','0',int))
    executor_before={k:v.detach().cpu().clone() for k,v in player.model.state_dict().items()} if verify else None
    policy_before={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()} if verify else None
    for iteration in range(start+1,start+iters+1):
        records=[]; actions=[]; logps=[]; values=[]; rewards=[]; dones=[]; durations=[]; next_values=[]
        diagnostics=dict(delivered=0,success=0,failure=0,executed_steps=0,switches=0,held_box_steps=0,root_displacement=0.)
        for _ in range(horizon):
            obs=task.allocation_observation()
            with torch.no_grad():
                dist,value=policy(obs)
                action=dist.sample() if mode=='train' else dist.logits.argmax(-1)
                if baseline=='nearest_initial':
                    action=nearest_initial_action(obs,task.allocation_initialized)
                assignment=policy.assignment_from_action(action) if boxes==4 else assignment_from_action(action)
                previous=obs['assignment'] if boxes==4 else task.box_assignment
                changed=(assignment!=previous).any(-1)
                penalty=torch.zeros(task.num_envs,device=task.device) if boxes==4 else switch_cost(previous,assignment,task.allocation_initialized,switch)
                if boxes==4:
                    busy_changed=(previous>=0) & (assignment!=previous)
                    assert not busy_changed.any()
                    diagnostics.setdefault('job_starts',0)
                    diagnostics['job_starts']+=int(((previous<0)&(assignment>=0)).sum())
                else:
                    diagnostics['switches']+=int((changed & task.allocation_initialized).sum())
                ids=changed.nonzero(as_tuple=False).flatten()
                if len(ids):
                    if verify:
                        physical_box=(task._allocation_boxes if boxes==4 else task._box_states).clone()
                        physical_root=task._humanoid_root_states.clone()
                        physical_goal=(task._allocation_goals if boxes==4 else task._box_tar_pos).clone()
                    task.set_box_assignment(ids,assignment[ids])
                    if verify:
                        assert torch.equal(physical_box,task._allocation_boxes if boxes==4 else task._box_states)
                        assert torch.equal(physical_root,task._humanoid_root_states)
                        assert torch.equal(physical_goal,task._allocation_goals if boxes==4 else task._box_tar_pos)
                task.allocation_initialized[:]=True
                reward,done,duration,diag=macro_step(player,steps,gamma,*costs)
                following=task.allocation_observation()
                _,nv=policy(following)
                diag['root_displacement']=float(((following['agent'][..., :3]-obs['agent'][..., :3]).norm(dim=-1)*(~done)[:,None]).sum())
            records.append({k:v.detach().clone() for k,v in obs.items()})
            actions.append(action); logps.append(dist.log_prob(action)); values.append(value)
            rewards.append(reward+penalty); dones.append(done); durations.append(duration); next_values.append(nv)
            for k,v in diag.items(): diagnostics[k]+=v
        rewards,values,next_values,dones,durations=map(torch.stack,(rewards,values,next_values,dones,durations))
        advantage,returns=gae(rewards,values,next_values,dones,durations,gamma,lam)
        loss=ppo_update(policy,optimizer,records,torch.stack(actions),torch.stack(logps),values,
                        advantage,returns,epochs,batch) if mode=='train' else None
        metrics=dict(iteration=iteration,reward=float(rewards.mean()),loss=loss,**diagnostics)
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
        print('ALLOCATION '+json.dumps(metrics),flush=True)
        if mode=='train':
            ck=dict(schema=schema,policy=policy.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,config=config)
            temp=out/'allocation_latest.tmp'; torch.save(ck,temp); temp.replace(out/'allocation_latest.pth')
            if iteration % save_every==0:
                torch.save(ck,out/f'allocation_{iteration:06d}.pth')

    if verify:
        unchanged=all(torch.equal(v.detach().cpu(),executor_before[k]) for k,v in player.model.state_dict().items())
        changed=any(not torch.equal(v.detach().cpu(),policy_before[k]) for k,v in policy.state_dict().items())
        if not unchanged or (mode=='train' and not changed) or (mode=='eval' and changed):
            raise RuntimeError('allocation/executor update isolation verification failed')
        result=dict(executor_unchanged=unchanged,allocation_updated=changed,mode=mode)
        if mode=='train':
            saved=torch.load(out/'allocation_latest.pth',map_location=task.device)
            loaded=policy_class(config['d_model']).to(task.device)
            loaded.load_state_dict(saved['policy']); loaded.eval()
            observation=task.allocation_observation()
            with torch.no_grad():
                original_dist,original_value=policy(observation)
                loaded_dist,loaded_value=loaded(observation)
            torch.testing.assert_close(original_dist.probs,loaded_dist.probs)
            torch.testing.assert_close(original_value,loaded_value)
            result['checkpoint_roundtrip']=True
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ALLOCATION_VERIFY '+json.dumps(result),flush=True)


if __name__=='__main__': main()
