"""Export 5 AT and 5 PUSH real single-agent records, with evidence provenance."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

def rounded(x):return np.round(np.asarray(x,dtype=float),6).tolist()
def first(v):
 ids=np.flatnonzero(v);return int(ids[0]) if len(ids) else None

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args();source=a.source.resolve();out=ROOT/'output_etc/check'/('fewshot_examples_'+source.name);out.mkdir(exist_ok=False)
 examples=[];audits={};dt=None
 for kind in ['AT','PUSH']:
  folder=source/kind;meta=json.loads((folder/'recording.json').read_text());assert meta['completed'] and meta['agents_per_episode']==1 and meta['native_weights_exact'];dt=meta['dt_s'];z=np.load(folder/'dense.npz');init=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);assert init['packet'].shape[1]==1
  candidates=[];all_rows=[]
  for e in range(meta['episodes']):
   completion=first(z['at_goal'][:,e,0]);contact=first(z['contact'][:,e,0]);endpoint=completion if kind=='AT' else contact
   row=dict(env=e,at_goal_step=completion,contact_step=contact,eligible=endpoint is not None and completion is not None,initial_root_to_object_m=float(np.linalg.norm(z['roots'][0,e,0,:2]-z['boxes'][0,e,0,:2])))
   all_rows.append(row)
   if row['eligible']:candidates.append(row)
  assert len(candidates)>=5,(kind,len(candidates))
  candidates.sort(key=lambda r:(r['initial_root_to_object_m'],r['env']));chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,5).round().astype(int)];assert len({r['env'] for r in chosen})==5
  audits[kind]=dict(all_candidates=all_rows,selected_environments=[r['env'] for r in chosen],eligible_count=len(candidates),selection='Five quantiles of initial XY root-to-object distance among native task-successful trials; PUSH must also have measured contact.')
  for j,r in enumerate(chosen):
   e=r['env'];end=r['at_goal_step'] if kind=='AT' else r['contact_step'];assert end>0
   origin=init['origins'][e].numpy();root=z['roots'][:end+1,e,0].copy();root[:,:3]-=origin;obj=z['boxes'][:end+1,e,0].copy();obj[:,:3]-=origin;hands=z['hands'][:end+1,e,0].copy()-origin;goal=init['goals'][e,0].numpy()-origin;dims=init['sizes'][e,0].numpy()
   # 0.5 s spacing + 0.1 s samples in the final second + exact endpoint.
   stride=max(1,round(.5/dt));fine=max(1,round(.1/dt));idx=sorted(set(range(0,end+1,stride))|set(range(max(0,end-round(1/dt)),end+1,fine))|{0,end})
   yaw=np.arctan2(2*(root[:,6]*root[:,5]+root[:,3]*root[:,4]),1-2*(root[:,4]**2+root[:,5]**2))
   agent_rows=[];object_rows=[]
   for t in idx:
    force=[None]*6 if t==0 else rounded(z['hand_forces'][t,e,0].reshape(-1))
    agent_rows.append([round(t*dt,6)]+rounded(root[t,:3])+[round(float(yaw[t]),6)]+rounded(root[t,7:10])+rounded(hands[t].reshape(-1))+force+[bool(z['contact'][t,e,0])])
    object_rows.append([round(t*dt,6)]+rounded(obj[t,:3])+rounded(obj[t,3:7])+rounded(obj[t,7:10]))
   event='A01_at_complete' if kind=='AT' else 'A01_push_contact'
   definition=('First control-step sample with object center within 0.20 m 3D Euclidean distance of its goal while the agent is valid; does not require hand release or object stabilization.' if kind=='AT' else 'First control step containing a native PUSH contact hit in at least one physics substep: both hands satisfy the configured proximity and directed-force thresholds and the box satisfies its directed-force threshold. See contact_detector_parameters. This is not task completion.')
   sample=dict(example_id=f'stage1_single_{kind.lower()}_{j+1:02}',controller_id='sha256:'+meta['checkpoint_sha256'],reference_type='single_agent_behavior',task_type=kind,execution=dict(agents=1,start='All actions execute from t=0.',scheduler_waiting=False,freeze=False,time_unit='seconds',timestamp_origin='episode start; elapsed active execution time'),initial_observation=dict(agents=[dict(agent_id='A01',root_position_m=rounded(root[0,:3]),root_orientation_xyzw=rounded(root[0,3:7]),root_yaw_rad=round(float(yaw[0]),6),root_linear_velocity_mps=rounded(root[0,7:10]),left_hand_position_m=rounded(hands[0,0]),right_hand_position_m=rounded(hands[0,1]))],objects=[dict(object_id='object_0',position_m=rounded(obj[0,:3]),orientation_xyzw=rounded(obj[0,3:7]),dimensions_m=rounded(dims),linear_velocity_mps=rounded(obj[0,7:10]))],coordinate_frame='Environment-local frame, world axes preserved, z up'),tasks=[dict(agent_id='A01',task_label=kind,description='Grasp and carry the object to its goal.' if kind=='AT' else 'Approach the box and push toward the object goal; this reference stops at first contact.',object_id='object_0',object_goal_position_m=rounded(goal))],event_definitions={event:definition},trajectory_columns=['elapsed_time_s','root_x_m','root_y_m','root_z_m','root_yaw_rad','root_vx_mps','root_vy_mps','root_vz_mps','left_hand_x_m','left_hand_y_m','left_hand_z_m','right_hand_x_m','right_hand_y_m','right_hand_z_m','left_hand_force_x_N','left_hand_force_y_N','left_hand_force_z_N','right_hand_force_x_N','right_hand_force_y_N','right_hand_force_z_N','native_push_contact_in_step'],agent_trajectories={'A01':agent_rows},object_trajectory_columns=['elapsed_time_s','x_m','y_m','z_m','qx','qy','qz','qw','vx_mps','vy_mps','vz_mps'],object_trajectories={'object_0':object_rows},measured_events=[dict(agent_id='A01',event_id=event,elapsed_time_s=round(end*dt,6),reached=True,control_step=end,measurement_resolution_s=dt)],observation_end_time_s=round(end*dt,6),outcomes=dict(recorded_endpoint_reached=True,selected_from_task_successful_rollouts=True),sampling=dict(coarse_interval_s=stride*dt,near_endpoint_interval_s=fine*dt,initial_force_values=None,initial_force_note='Unavailable before first physics step. Null in export.'),provenance=dict(source_run=str(folder),seed=meta['seed'],environment_index=e,policy_weights_exact=True,stage2_coordination=False))
   if kind=='PUSH':sample['contact_detector_parameters']=meta['contact_detector'];sample['contact_measurement_note']='Hit is aggregated across substeps. Forces in a sampled row are end-of-control-step values, so they need not reproduce an earlier substep hit.'
   if (folder/'geometry_audit.json').exists():
    geometry=json.loads((folder/'geometry_audit.json').read_text());sample['sampling_geometry']=dict(variant=geometry['variant'],root_to_box_xy_m=geometry['root_to_box_xy_m'][e],box_to_goal_xy_m=geometry['box_to_goal_xy_m'][e],push_geometry=geometry['push_geometry'],eval_distance_bounds=geometry['eval_distance_bounds'])
   (out/(sample['example_id']+'.json')).write_text(json.dumps(sample,indent=2,allow_nan=False));examples.append(sample)
 (out/'demonstrations.json').write_text(json.dumps(examples,indent=2,allow_nan=False));(out/'selection_audit.json').write_text(json.dumps(audits,indent=2));(out/'manifest.json').write_text(json.dumps(dict(source=str(source),count=len(examples),at=5,push_contact=5,agents_per_episode=1,seed=meta['seed'],checkpoint_sha256=meta['checkpoint_sha256'],endpoint_selection='AT completion, PUSH native contact; all selected rollouts task-successful',api_calls=0),indent=2))
 print(out)
 for x in examples:print(x['example_id'],x['provenance']['environment_index'],x['observation_end_time_s'],len(x['agent_trajectories']['A01']))
if __name__=='__main__':main()
