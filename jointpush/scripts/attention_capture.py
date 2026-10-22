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
 task=self.env.task;n,m,o=task.num_envs,task.num_agents,task.num_objects
 self.model.eval();self.running_mean_std.eval();self._eval=False;self._reward_debug=False
 encoder=self.model.a2c_network.actor_encoder
 assert not encoder.geometry_enabled
 before=digest(self);out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
 obs=self.env_reset();self.get_batch_size(obs['obs'],1)
 active=torch.ones(n,dtype=torch.bool);frames=[];states=[];errors=[];identity_error=None
 horizon=int(os.environ.get('PROBE_STEPS','300'));sample_steps={0,30,60,120,180,299}
 def probe():
  records=[];pending={};hooks=[]
  def pre(layer,args):
   x,bias,geo,msg,g,gi=args[:6];B,L,d=x.shape
   q,k,v=layer.qkv(x).view(B,L,3,layer.num_heads,layer.head_dim).permute(2,0,3,1,4)
   if g is not None:
    q=apply_gta_transform(g.transpose(-1,-2),q);k=apply_gta_transform(gi,k);v=apply_gta_transform(gi,v)
   qk=(q@k.transpose(-1,-2))*layer.scale
   bias=torch.zeros_like(qk) if bias is None else bias.expand_as(qk)
   attn=(qk+bias).softmax(-1)
   z=attn@v
   if g is not None:z=apply_gta_transform(g,z)
   z=z.transpose(1,2).reshape(B,L,d)
   z=layer.norm1(x+layer.proj(z));expected=layer.norm2(z+layer.linear2(layer.act(layer.linear1(z))))
   pending[id(layer)]=expected
   records.append([a.detach().numpy().copy() for a in (qk,bias,attn)])
  def post(layer,args,result):errors.append(float((result-pending.pop(id(layer))).abs().max()))
  for layer in encoder.layers:hooks.extend((layer.register_forward_pre_hook(pre),layer.register_forward_hook(post)))
  try:features=encoder(self._preproc_obs(obs['obs']),token_order=torch.arange(2*m+o))
  finally:
   for hook in hooks:hook.remove()
  return np.stack(records),features
 with torch.no_grad():
  for step in range(horizon):
   if step in sample_steps and active.any():
    records,features=probe()
    if identity_error is None:
     rng=torch.get_rng_state()
     original=encoder(self._preproc_obs(obs['obs']))
     torch.set_rng_state(rng)
     identity_error=float((self.model.a2c_network.action_head(features)-self.model.a2c_network.action_head(original)).abs().max())
     assert identity_error<1e-4
    frames.append(records);states.append(dict(step=step,active=active.tolist(),assignment=task._agent_box_assignment.tolist(),humans=task._humanoid_root_states[...,:3].tolist(),boxes=task._logical_box_values(task._box_states)[...,:3].tolist(),goals=task._tar_pos.tolist()))
   action=self.get_action(obs,True);assert torch.isfinite(action).all()
   obs,_,done,_=self.env_step(self.env,action)
   if not isinstance(obs,dict):obs=self.obs_to_torch(obs)
   assert torch.isfinite(obs['obs']).all()
   ended=done.reshape(n,m).any(-1)
   active &= ~ended
   if not active.any():break
 assert before==digest(self) and max(errors)<1e-6
 # frame,layer,component,scene,head,query,key -> frame,scene,layer,head,query,key
 arr=np.stack(frames)
 np.savez_compressed(out/'attention_raw.npz',**{key:arr[:,:,i].transpose(0,2,1,3,4,5) for i,key in enumerate(('qk_scores','relation_bias','attention'))})
 (out/'states.json').write_text(json.dumps(states))
 report=dict(agents=m,boxes=o,scenes=n,seed=task.cfg['args'].seed,checkpoint=task.cfg['args'].checkpoint,device='CPU PhysX and CPU inference',frames=len(frames),valid_samples=sum(sum(s['active']) for s in states),layer_reconstruction_max_error=max(errors),identity_order_action_max_error=identity_error,model_rms_unchanged=True,policy_intervention=False,definition='GTA-transformed QK/sqrt(d), relation bias, softmax(QK+bias); original actor forward; later QK includes prior-layer biases',first_termination_censoring=True)
 (out/'capture.json').write_text(json.dumps(report,indent=2));print('ATTENTION_CAPTURE',json.dumps(report),flush=True)
MAPlayerContinuous.run=run
runpy.run_path(str(PORT/'tokenhsi/run.py'),run_name='__main__')
