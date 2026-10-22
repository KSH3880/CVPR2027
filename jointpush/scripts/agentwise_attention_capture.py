"""Capture canonical attention without changing the original layer forwards."""
import json
from pathlib import Path
import numpy as np
import torch
from learning.multi_agent.amp_network_builder_ma import apply_gta_transform

def capture(player,obs,out):
 task=player.env.task;encoder=player.model.a2c_network.actor_encoder
 assert task.num_agents==4 and not encoder.geometry_enabled and not getattr(encoder,'use_relation_message',False)
 records=[];errors=[];pending={};hooks=[]
 def pre(layer,args):
  x,bias,geo,message,g,gi=args[:6];B,L,d=x.shape
  assert geo is None and message is None
  q,k,v=layer.qkv(x).view(B,L,3,layer.num_heads,layer.head_dim).permute(2,0,3,1,4)
  if g is not None:
   q=apply_gta_transform(g.transpose(-1,-2),q);k=apply_gta_transform(gi,k);v=apply_gta_transform(gi,v)
  qk=(q@k.transpose(-1,-2))*layer.scale
  bias=torch.zeros_like(qk) if bias is None else bias.expand_as(qk)
  attention=(qk+bias).softmax(-1)
  z=attention@v
  if g is not None:z=apply_gta_transform(g,z)
  z=z.transpose(1,2).reshape(B,L,d)
  z=layer.norm1(x+layer.proj(z));pending[id(layer)]=layer.norm2(z+layer.linear2(layer.act(layer.linear1(z))))
  records.append([a.detach().cpu().numpy().copy() for a in (qk,bias,attention)])
 def post(layer,args,result):
  error=float((result-pending.pop(id(layer))).abs().max());errors.append(error);assert error<2e-5
 for layer in encoder.layers:hooks.extend((layer.register_forward_pre_hook(pre),layer.register_forward_hook(post)))
 try:features=encoder(player._preproc_obs(obs['obs']),token_order=torch.arange(sum(encoder.entity_counts)))
 finally:
  for h in hooks:h.remove()
 array=np.stack(records)
 out=Path(out)
 np.savez_compressed(out/'initial_attention.npz',**{key:array[:,i].transpose(1,0,2,3,4) for i,key in enumerate(('qk_scores','relation_bias','attention'))})
 assignment=task._agent_box_assignment.detach().cpu()
 logical=task._logical_box_order.detach().cpu()
 assert torch.equal(logical[:,:4],assignment),'Logical object slots must be owner first'
 meta=dict(scenes=task.num_envs,agents=4,objects=task.num_objects,goals=4,seed=task.cfg['args'].seed,checkpoint=task.cfg['args'].checkpoint,layout='scene,layer,head,query,key',own_box_token_ids=[4+i for i in range(4)],own_goal_token_ids=[4+task.num_objects+i for i in range(4)],unassigned_box_token_ids=list(range(8,4+task.num_objects)),physical_box_assignment=assignment.tolist(),logical_box_order=logical.tolist(),layer_reconstruction_max_error=max(errors),mask=False,sampling='initial state; original biased attention; canonical token order')
 (out/'initial_attention.json').write_text(json.dumps(meta,indent=2))
 return features
