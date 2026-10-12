"""Policy-free actuator audit and ideal horizontal traction-envelope validation.

The force probes act on boxes directly. They validate box mechanics at the
agent's ideal foot-traction bound; they do NOT prove an agent pose attains it.
"""
from isaacgym import gymapi, gymtorch
import csv
import json
import os
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from utils.door_reference import rotate

torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
HUMANOID_MASS=50.299713373184204
FRICTIONS=[i/10 for i in range(1,11)]
CASES=[(mu,ratio,HUMANOID_MASS/((1+mu)/2)*ratio) for mu in FRICTIONS for ratio in (.8,1.,1.2)]
old_build=HumanoidMAOpenPush._build_box

def build(self,e,b,ptr):
    old_build(self,e,b,ptr)
    if b<2:return
    mu,ratio,mass=CASES[e//2]
    handle=self._box_handles[-1]
    props=self.gym.get_actor_rigid_body_properties(ptr,handle);props[0].mass=mass
    self.gym.set_actor_rigid_body_properties(ptr,handle,props,True)
    props=self.gym.get_actor_rigid_shape_properties(ptr,handle)
    for p in props:p.friction=mu
    self.gym.set_actor_rigid_shape_properties(ptr,handle,props)

HumanoidMAOpenPush._build_box=build
args=get_args();assert args.num_envs==60 and args.test and args.eval and args.headless
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True;cfg['_rsi_build']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
try:
    env=task.envs[0];handle=task.humanoid_handles_all[0][0]
    bodies=task.gym.get_actor_rigid_body_properties(env,handle)
    mass=sum(p.mass for p in bodies);assert abs(mass-HUMANOID_MASS)<1e-5
    names=task.gym.get_actor_dof_names(env,handle);dofs=task.gym.get_actor_dof_properties(env,handle)
    actuators=[dict(name=n,effort_Nm=float(dofs['effort'][i]),stiffness=float(dofs['stiffness'][i]),damping=float(dofs['damping'][i]),motor_gear_effort=float(task.motor_efforts[i])) for i,n in enumerate(names)]
    assert all(x['effort_Nm']>1e30 for x in actuators)
    foot_mu=[p.friction for p in task.gym.get_actor_rigid_shape_properties(env,handle)]
    assert all(abs(x-1)<1e-6 for x in foot_mu)
    assert abs(sim.gravity.z+9.81)<1e-5
    traction=mass*abs(sim.gravity.z)
    ids=torch.arange(60,device=task.device);task._task_graph_preset='push';task._reset_envs(ids)
    task._humanoid_root_states[:,:,:2]+=100.
    task._box_states[:,2:]=0.;task._box_states[:,2:,2]=.65;task._box_states[:,2:,6]=1.
    task._box_states[:,2:,0]=-4.;task._box_states[:,2,1]=-5.;task._box_states[:,3,0]=-10.;task._box_states[:,3,1]=5.
    task._reset_env_tensors(ids)
    actions=torch.zeros(120,32,device=task.device)
    for _ in range(15):task.pre_physics_step(actions);task._physics_step()
    initial=task._box_states[:,2,:3].clone()
    nb=task._rigid_body_state.view(60,-1,13).shape[1];box_body=2*task.num_bodies+2*task._door_body_count
    forces=torch.zeros(60,nb,3,device=task.device);positions=torch.zeros_like(forces)
    forced=ids[::2];forces[forced,box_body,1]=traction
    max_tilt=torch.zeros(60,device=task.device)
    for _ in range(30):
        task.pre_physics_step(actions)
        for _ in range(task.control_freq_inv):
            state=task._box_states[forced,2];q=state[:,3:7]
            positions[forced,box_body]=state[:,:3]+rotate(q,state.new_tensor([0.,-.65,-.15]).expand(len(forced),3))
            task.gym.apply_rigid_body_force_at_pos_tensors(task.sim,gymtorch.unwrap_tensor(forces),gymtorch.unwrap_tensor(positions),gymapi.ENV_SPACE)
            task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
            q=task._box_states[:,2,3:7]
            tilt=torch.acos((1-2*(q[:,0].square()+q[:,1].square())).clamp(-1,1))
            max_tilt=torch.maximum(max_tilt,tilt)
    rows=[]
    for index,(mu,ratio,mass_box) in enumerate(CASES):
        e=2*index;handle=task._box_handles[4*e+2]
        actual_mass=task.gym.get_actor_rigid_body_properties(task.envs[e],handle)[0].mass
        assert abs(actual_mass-mass_box)<1e-3
        shapes=task.gym.get_actor_rigid_shape_properties(task.envs[e],handle)
        assert all(abs(p.friction-mu)<1e-5 for p in shapes)
        rows.append(dict(box_friction=mu,assumed_average_contact_friction=(1+mu)/2,ratio_to_ideal_mass_bound=ratio,mass_kg=actual_mass,force_N=traction,initial_force_height_m=.5,duration_s=30*task.dt,forward_displacement_m=float(task._box_states[e,2,1]-initial[e,1]),zero_force_control_displacement_m=float(task._box_states[e+1,2,1]-initial[e+1,1]),max_tilt_deg=float(max_tilt[e]*180/torch.pi)))
    assert all(abs(x['zero_force_control_displacement_m'])<1e-3 for x in rows)
    assert all(x['forward_displacement_m']>.1 and x['max_tilt_deg']<20 for x in rows if x['ratio_to_ideal_mass_bound']==.8)
    assert all(abs(x['forward_displacement_m'])<.01 for x in rows if x['ratio_to_ideal_mass_bound']==1.2)
    report=dict(training=False,checkpoint_used=False,policy_used=False,gpu=os.environ['TOKENHSI_GPU'],humanoid_mass_kg=mass,gravity_m_s2=abs(sim.gravity.z),pd_effort_effectively_unlimited=True,actuators=actuators,foot_shape_friction=foot_mu,ideal_horizontal_traction_N=traction,
        assumptions='Quasistatic unbraced agent, only horizontal hand force, no vertical hand force, no impacts, floor/foot friction1. Average box/floor material friction is assumed and tested at80/100/120percent of mass bound.',
        limits='This is an ideal contact-mechanics upper bound. External box probes validate sliding thresholds only, not the attainability of the bound by an agent pose/controller. Current PD has no meaningful finite actuator torque cap.',
        ideal_mass_bounds=[dict(box_friction=mu,mass_kg=HUMANOID_MASS/((1+mu)/2)) for mu in FRICTIONS],force_probes=rows)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'force_probes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(humanoid_mass_kg=mass,ideal_horizontal_traction_N=traction,ideal_mass_bounds=report['ideal_mass_bounds'],verified=True),indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)
