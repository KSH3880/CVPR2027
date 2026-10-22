"""Read-only attention probes alongside an unchanged frozen CPU rollout."""
import os,sys,runpy,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PORT=ROOT/'ports/koo_native_y2m'
sys.path.insert(0,str(PORT/'tokenhsi'))
import isaacgym
import torch
import numpy as np
if os.environ.get('PROBE_AGENTS','4')=='4':
 from native4_y2m import install
 install()
from learning.multi_agent.ma_players import MAPlayerContinuous
from learning.multi_agent.amp_network_builder_ma import apply_gta_transform
old_init=MAPlayerContinuous.__init__
def init(self,config):
 config=dict(config,device_name='cpu');old_init(self,config)
MAPlayerContinuous.__init__=init

def digest(self):
 h=hashlib.sha256()
 for d in (self.model.state_dict(),self.running_mean_std.state_dict()):
  for k,v in sorted(d.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def run(self):
 import copy
 task=self.env.task;assert (task.num_agents,task.num_objects)==(4,4)
 self.model.eval();self.running_mean_std.eval();self._eval=False;self._reward_debug=False
 before=digest(self);encoder=self.model.a2c_network.actor_encoder
 small=copy.deepcopy(encoder);small.set_entity_counts(2,2);small.eval()
 out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
 obs=self.env_reset();self.get_batch_size(obs['obs'],1)
 n=task.num_envs;active=torch.ones(n,dtype=torch.bool);frames=[];states=[];actions=[]
 input_error=masked_error=readout_error=0.
 def trace(net,values,count):
  layers=[]
  handles=[net.layers[0].register_forward_pre_hook(lambda module,args:layers.append(args[0][:,:count].detach().clone()))]
  for layer in net.layers:handles.append(layer.register_forward_hook(lambda module,args,result:layers.append(result[:,:count].detach().clone())))
  try:features=net(values,token_order=torch.arange(sum(net.entity_counts)))
  finally:
   for handle in handles:handle.remove()
  torch.testing.assert_close(layers[-1],features,rtol=0,atol=0)
  return torch.stack(layers,1)
 def measure():
  nonlocal input_error,masked_error,readout_error
  values=self._preproc_obs(obs['obs'])
  full=trace(encoder,values,4)
  if os.environ.get('JOINTPUSH_CAPTURE_AGENTWISE')=='1' and not (out/'initial_attention.npz').exists():
   from agentwise_attention_capture import capture
   recorded=capture(self,obs,out)
   torch.testing.assert_close(recorded,full[:,-1],rtol=0,atol=0)
  h=values[:,:892].reshape(n,4,223);b=values[:,892:1012].reshape(n,4,30)
  g=values[:,1012:1016].reshape(n,4,1);poses=values[:,1016:1100].reshape(n,12,7);packet=values[:,1100:].reshape(n,8,5)
  pairs=[]
  for start in (0,2):
   ids=torch.tensor([start,start+1,4+start,5+start,8+start,9+start])
   mapping=torch.full((12,),-1,dtype=torch.long);mapping[ids]=torch.arange(6)
   edge=packet[:,2*start:2*(start+2)].clone()
   edge[:,:,1]=mapping[edge[:,:,1].long()];edge[:,:,2]=mapping[edge[:,:,2].long()];edge[:,:,4]-=start
   two=torch.cat((h[:,start:start+2].flatten(1),b[:,start:start+2].flatten(1),g[:,start:start+2].flatten(1),poses[:,ids].flatten(1),edge.flatten(1)),1)
   assert two.shape==(n,570)
   pairs.append(trace(small,two,2))
  pair=torch.cat(pairs,2)
  groups=torch.tensor([0,0,1,1]*3)
  def mask(module,args):
   args=list(args);args[1]=args[1].masked_fill(groups[:,None]!=groups[None,:],float('-inf'));return tuple(args)
  handles=[layer.register_forward_pre_hook(mask) for layer in encoder.layers]
  try:masked=trace(encoder,values,4)
  finally:
   for handle in handles:handle.remove()
  input_error=max(input_error,float((pair[:,0]-full[:,0]).abs().max()))
  masked_error=max(masked_error,float((masked-pair).abs().max()))
  torch.testing.assert_close(pair[:,0],full[:,0],rtol=0,atol=0)
  torch.testing.assert_close(masked,pair,rtol=2e-5,atol=2e-5)
  head=self.model.a2c_network.action_head
  mu2,mu4,mum=head(pair[:,-1]),head(full[:,-1]),head(masked[:,-1])
  readout_error=max(readout_error,float((mu2-mum).abs().max()))
  return torch.stack((pair,full,masked),1).numpy(),torch.stack((mu2,mu4,mum),1).numpy()
 with torch.no_grad():
  for step in range(int(os.environ.get('PROBE_STEPS','300'))):
   if step in (0,15,30,60,120,180,299) and active.any():
    f,a=measure();frames.append(f);actions.append(a)
    states.append(dict(step=step,active=active.tolist(),assignment=task._agent_box_assignment.tolist(),humans=task._humanoid_root_states[...,:3].tolist(),boxes=task._logical_box_values(task._box_states)[...,:3].tolist(),goals=task._tar_pos.tolist()))
    print('FEATURE_SAMPLE',step,'active',int(active.sum()),flush=True)
   action=self.get_action(obs,True);assert torch.isfinite(action).all()
   obs,_,done,_=self.env_step(self.env,action)
   if not isinstance(obs,dict):obs=self.obs_to_torch(obs)
   assert torch.isfinite(obs['obs']).all()
   active &= ~done.reshape(n,4).any(-1)
   if not active.any():break
 assert before==digest(self)
 np.savez_compressed(out/'features.npz',features=np.stack(frames),actions=np.stack(actions))
 (out/'states.json').write_text(json.dumps(states))
 meta=dict(checkpoint=task.cfg['args'].checkpoint,seed=task.cfg['args'].seed,scenes=n,agents=4,boxes=4,goals=4,feature_dim=64,layers=['tokenizer','layer1','layer2','layer3','layer4'],conditions=['two_agent_pair','four_agent_full','four_agent_pair_mask'],feature_layout='frame,scene,condition,layer,agent,feature',action_layout='frame,scene,condition,agent,dof',same_tokenizer_human_max_error=input_error,pair_mask_feature_max_error=masked_error,pair_mask_action_max_error=readout_error,model_rms_unchanged=True,device='CPU PhysX and inference',sampling='4-agent unmasked trajectory, before scene first termination; two-agent counterfactual preserves retained human/box/goal observations and remaps edge indices only')
 (out/'capture.json').write_text(json.dumps(meta,indent=2));print('FEATURE_COMPARISON',json.dumps(meta),flush=True)
MAPlayerContinuous.run=run
runpy.run_path(str(PORT/'tokenhsi/run.py'),run_name='__main__')
