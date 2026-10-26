"""CPU-only all-profile PUSH RSI screen with fixed-goal distance assertions."""
from isaacgym import gymapi
import json
from pathlib import Path
from unittest.mock import patch
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.native_push_rsi import digest
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push

args = get_args()
assert args.headless and args.device == 'cpu' and not args.use_gpu
cfg, train, _ = load_cfg(args)
env = cfg['env']; cfg['args'] = args
env['motion_file'] = args.motion_file
env['enableTrackInitState'] = True
env['_native_geometry_check'] = True
env['relationGraph']['template_probabilities'] = dict(HOLDING=0.,SIT=0.,CLIMB=0.,HOLDING_AT=0.,HOLDING_ON_TOP=0.,PUSH_AT=1.)
assert args.num_envs == len(env['pushBox']['size_profiles'])
args.eval = False
args.task_graph = 'push_at'
set_seed(42); torch.set_num_threads(1)
sim = parse_sim_params(args,cfg,train); sim.physx.num_threads = max(1,args.num_threads)


def all_profiles(n, spec, device):
    dims = torch.tensor(spec['pushBox']['size_profiles'],device=device)
    sizes = dims[:,None].expand(-1,4,-1).clone()
    probs = torch.zeros(n,2,6,device=device); probs[...,5] = 1.
    return sizes, probs


with patch('env.tasks.multi_agent.humanoid_ma_stage1_push.sample_asset_sizes',all_profiles):
    task = HumanoidMAStage1Push(cfg,sim,args.physics_engine,'cpu',0,True)
bank = Path(env['openPushRSI']['bank_file']); before = digest(bank)
payload = json.loads(bank.read_text()); rows = payload['candidates']['push']
ids = torch.arange(task.num_envs)
out = Path(args.output_path); out.mkdir(parents=True,exist_ok=True)
failed = []; representative = []; targets_checked = 0
try:
    for scene in range(task.num_envs):
        for box in range(4):
            handle = task._box_handles[scene*4+box]
            props = task.gym.get_actor_rigid_body_properties(task.envs[scene],handle)
            shape = task.gym.get_actor_rigid_shape_properties(task.envs[scene],handle)
            assert abs(sum(p.mass for p in props)-env['pushBox']['mass']) < 1e-3
            assert all(abs(s.friction-env['pushBox']['friction'])<1e-6 for s in shape)
    # Cover every retained pose at every new asset size, both owner slots.
    for index,row in enumerate(rows):
        def sample_pose(skill, count):
            sample = {k:torch.full((count,),v,device='cpu') for k,v in row.items()}
            sample['late_requested'] = torch.zeros(count,dtype=torch.bool)
            return sample
        with patch.object(task._push_bank,'sample',side_effect=sample_pose):
            task.cfg['env']['openPushRSI']['interaction_probability'] = 1.
            task._reset_envs(ids)
        initial = task._humanoid_root_states[...,:3].clone()
        initial_boxes = task._box_states[...,:3].clone()
        goals = task._tar_pos.clone()
        source = task._push_initial_plan[:,:2,:3]
        original_distance = (goals-source).norm(dim=-1)
        remaining = (goals-task._box_states[:,:2,:3]).norm(dim=-1)
        torch.testing.assert_close(remaining,original_distance*(1-row['box_progress']),atol=2e-5,rtol=1e-5)
        targets_checked += 2*len(ids)
        stable = torch.ones(len(ids),2,dtype=torch.bool)
        for step in range(3):
            task.pre_physics_step(torch.zeros(2*len(ids),32))
            task._physics_step(); task.post_physics_step()
            assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
            torch.testing.assert_close(task._tar_pos,goals,rtol=0,atol=0)
            stable &= (task._humanoid_root_states[...,2]>.65)
            stable &= ((task._humanoid_root_states[...,:3]-initial).norm(dim=-1)<.25)
            stable &= ((task._box_states[:,:2,:3]-initial_boxes[:,:2]).norm(dim=-1)<.08)
            stable &= ~task.reset_buf[:,None].bool()
        if not stable.all():
            failed.append(dict(index=index,row=row,failed=(~stable).nonzero().tolist()))
        if index%100 == 0:
            print('FINETUNE_RSI',index,len(rows),'failed',len(failed),flush=True)
        if index in (0,len(rows)//2,len(rows)-1):
            representative.append(dict(row=row,initial_distance=original_distance.tolist(),remaining=remaining.tolist()))
    assert before == digest(bank)
    report = dict(device='cpu',physics_threads=sim.physx.num_threads,poses=len(rows),profiles=len(ids),agents=2,
        states=len(rows)*len(ids)*2,seconds=.1,failed=failed,
        fixed_goal_checks=targets_checked,examples=representative,physics_mass_kg=env['pushBox']['mass'],physics_friction=env['pushBox']['friction'],
        limitations='All candidates/new sizes/both owners, seeded square/cone paths; 0.1s initial PD, not policy success or long balance.')
    (out/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
    if failed:
        raise RuntimeError('New RSI screen failed; requested pool preserved and training remains blocked')
    if payload.get('screen') != 'PD_0.1s_all_profiles':
        payload['screen'] = 'PD_0.1s_all_profiles'; payload['physics_screen'] = report
        bank.write_text(json.dumps(payload,indent=2)+'\n')
    print('FINETUNE_RSI_DONE',report['states'],flush=True)
finally:
    task.gym.destroy_sim(task.sim)
