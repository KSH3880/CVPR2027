"""CPU audit of native Stage1 SUPPORT/distractor roles in mixed task scenes."""
import isaacgym
from pathlib import Path
import json
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from utils.edge_scenario_spec import classify_templates
from utils.task_role_spec import task_graph_packet
from utils.stage1_push_scene import support_push_paths_clear

args=get_args()
assert args.headless and args.device=='cpu' and not args.use_gpu and args.eval
cfg,train,_=load_cfg(args);cfg['args']=args
cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(123);torch.set_num_threads(1)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
task=HumanoidMAStage1Push(cfg,sim,args.physics_engine,'cpu',0,True)
ids=torch.arange(task.num_envs)
counts=torch.zeros(6,dtype=torch.long);previous=None;changed=0
for _ in range(40):
    task._reset_envs(ids)
    ee,aa=task._expand_slots(ids)
    template=classify_templates(task.relation_runtime.graph,ee,aa).reshape(task.num_envs,2)
    counts+=torch.bincount(template.flatten(),minlength=6)
    packet=task_graph_packet(task.relation_runtime.graph,task.num_envs).reshape(task.num_envs,2,5).long()
    for owner in range(2):
        active=template[:,owner]==4
        assert (packet[active,owner,4]-task.num_agents==owner+2).all()
    assert task.num_physical_objects==4 and task.get_obs_size()==634 and task.get_scene_num_tokens()==8
    assert not task._push_asset_mask[:,2:].any()
    push=template==5
    assert support_push_paths_clear(task._box_states,task._box_size,task._tar_pos,push).all()
    supports=task._box_states[:,2:,:2]
    unused=template!=4
    assert (supports.norm(dim=-1)[unused]<=task._box_spawn_radius+1e-4).all()
    if previous is not None:changed+=int(((supports-previous).norm(dim=-1)>1e-4).sum())
    previous=supports.clone()
    assert torch.isfinite(task.obs_buf).all()
assert counts[4]>0 and counts[5]>0 and counts[1]>0 and counts[2]>0 and counts[3]>0
mass_errors=[];support_frictions=[]
for en in range(task.num_envs):
    for obj in range(4):
        h=task._box_handles[en*4+obj]
        props=task.gym.get_actor_rigid_body_properties(task.envs[en],h)
        mass=sum(p.mass for p in props)
        expected=20. if task._push_asset_mask[en,obj] else float(task._box_size[en,obj].prod())*100.
        mass_errors.append(abs(mass-expected))
        if obj>=2:support_frictions.extend(p.friction for p in task.gym.get_actor_rigid_shape_properties(task.envs[en],h))
assert max(mass_errors)<1e-3
report=dict(cpu=True,gpu_executed=False,physical_boxes=4,observation=634,tokens=8,
    reset_agent_trials=int(counts.sum()),template_counts=counts.tolist(),support_role_ontop=True,
    unused_support_arena=True,support_positions_changed=changed,max_mass_error=max(mass_errors),
    support_friction=sorted(set(round(x,6) for x in support_frictions)),
    support_min=task._box_size[:,2:].amin((0,1)).tolist(),support_max=task._box_size[:,2:].amax((0,1)).tolist(),
    limitations='Mixed-task loco resets on CPU; training RSI/long-run mixed policy success not evaluated here.')
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
(out/'support_roles_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));task.gym.destroy_sim(task.sim)
