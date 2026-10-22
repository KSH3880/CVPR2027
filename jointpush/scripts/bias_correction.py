"""Inference-only, owner-specific additive logit penalty on selected actor layers."""
import os,json,hashlib,types
from pathlib import Path
import torch
import numpy as np
from learning.multi_agent.amp_network_builder_ma import apply_gta_transform

class BiasCorrection:
 def __init__(self,player,out):
  self.player=player;self.task=player.env.task;self.encoder=player.model.a2c_network.actor_encoder;self.out=Path(out)
  self.m=self.task.num_agents;self.o=self.task.num_objects;self.count=2*self.m+self.o
  assert self.o==self.m and self.m in (2,4) and not self.encoder.geometry_enabled
  self.target=os.environ.get('JOINTPUSH_BIAS_TARGET','none');assert self.target in ('none','humans','boxes','goals','all_foreign')
  self.hard_mask=os.environ.get('JOINTPUSH_BIAS_HARD_MASK','0')=='1'
  assert not self.hard_mask or self.target=='all_foreign'
  self.mask_probability_errors=[]
  self.strength=float(os.environ.get('JOINTPUSH_BIAS_LAMBDA','0'));assert self.strength in (0.,2.,4.,6.,8.,12.)
  layer_mode=os.environ.get('JOINTPUSH_BIAS_LAYERS','last2')
  aliases={'first2':(0,1),'last2':(2,3),'all':(0,1,2,3),**{f'layer{i}':(i-1,) for i in range(1,5)}}
  if layer_mode in aliases: selected=aliases[layer_mode]
  else:
   indices=[int(s) for s in layer_mode.split(',')]
   assert indices and len(indices)==len(set(indices)) and all(1<=i<=4 for i in indices)
   selected=tuple(i-1 for i in indices)
  self.scope=os.environ.get('JOINTPUSH_BIAS_SCOPE','human');assert self.scope in ('human','all_owner')
  assert self.scope=='human' or self.target in ('none','all_foreign')
  self.layers=set(selected);self.original_forward=self.encoder.forward;self.order=None
  self.capture=False;self.parts=[];self.frames=[];self.states=[];self.errors=[];self.pending={};self.initial_hash=None;self.bias_errors=[]
  def forward(encoder,obs,return_all=False,token_order=None):
   if token_order is None:
    token_order=torch.randperm(self.count,device=obs.device) if encoder.relation_graph_spec.get('shuffle_token_order',False) else torch.arange(self.count,device=obs.device)
   self.order=token_order
   return self.original_forward(obs,return_all=return_all,token_order=token_order)
  self.encoder.forward=types.MethodType(forward,self.encoder)
  self.handles=[]
  for index,layer in enumerate(self.encoder.layers):
   self.handles.append(layer.register_forward_pre_hook(lambda module,args,index=index:self.pre(index,module,args)))
   self.handles.append(layer.register_forward_hook(self.post))

 def begin(self,step,active):
  self.capture=step in (0,30,60,120,180,299) and bool(active[:2].any());self.parts=[]
  if self.initial_hash is None:
   h=hashlib.sha256()
   for value in (self.task._humanoid_root_states,self.task._box_states,self.task._tar_pos,self.task.obs_buf):h.update(value.detach().cpu().contiguous().numpy().tobytes())
   self.initial_hash=h.hexdigest()
  if self.capture:self.state=dict(step=step,active=active[:2].cpu().tolist(),assignment=self.task._agent_box_assignment[:2].cpu().tolist())

 def pre(self,index,layer,args):
  x,bias,geo,msg,g,gi=args[:6];B,L,D=x.shape;assert geo is None and msg is None
  if bias is None:bias=x.new_zeros(layer.num_heads,L,L)
  mask=torch.zeros(L,L,dtype=torch.bool,device=x.device)
  if index in self.layers and self.target!='none':
   if self.scope=='all_owner':
    # Canonical H_i/B_i/G_i all have owner i. All ordered cross-owner
    # query/key pairs receive one -lambda; all intra-owner pairs stay intact.
    owners=self.order % self.m
    mask=owners[:,None]!=owners[None,:]
    assert int(mask.sum())==9*self.m*(self.m-1)
    for q in range(L):assert torch.equal(~mask[q],owners==owners[q])
   else:
    offsets=(0,self.m,self.m+self.o) if self.target=='all_foreign' else ({'humans':0,'boxes':self.m,'goals':self.m+self.o}[self.target],)
    for offset in offsets:
     for i in range(self.m):
      for j in range(self.m):
       if i!=j:mask |= (self.order[:,None]==i)&(self.order[None,:]==offset+j)
    assert int(mask.sum())==self.m*(self.m-1)*len(offsets)
    if self.target=='all_foreign':
     for i in range(self.m):
      row=mask[self.order==i].squeeze(0)
      assert torch.equal(~row,(self.order==i)|(self.order==self.m+i)|(self.order==self.m+self.o+i))
  correction=mask.to(bias.dtype)*(-self.strength)
  effective=bias.masked_fill(mask,-torch.inf) if self.hard_mask else bias+correction
  if self.capture:
   if self.hard_mask:
    assert torch.isneginf(effective[...,mask]).all()
    error=0.
   else:
    change=effective-bias
    error=float((change-correction).abs().max())
   self.bias_errors.append(error);assert error<2e-6
   assert torch.equal(effective[...,~mask],bias[...,~mask])
   q,k,v=layer.qkv(x).view(B,L,3,layer.num_heads,layer.head_dim).permute(2,0,3,1,4)
   if g is not None:
    q=apply_gta_transform(g.transpose(-1,-2),q);k=apply_gta_transform(gi,k);v=apply_gta_transform(gi,v)
   qk=(q@k.transpose(-1,-2))*layer.scale;p=(qk+effective).softmax(-1)
   assert torch.isfinite(p).all()
   if self.hard_mask:
    err=float(p[...,mask].abs().max()) if mask.any() else 0.
    self.mask_probability_errors.append(err);assert err==0.
   z=p@v
   if g is not None:z=apply_gta_transform(g,z)
   z=z.transpose(1,2).reshape(B,L,D);z=layer.norm1(x+layer.proj(z));self.pending[id(layer)]=layer.norm2(z+layer.linear2(layer.act(layer.linear1(z))))
   inv=self.order.argsort()
   def canonical(value):return value[:2].index_select(-2,inv).index_select(-1,inv).detach().cpu().numpy().copy()
   self.parts.append([canonical(a) for a in (qk,bias.expand_as(qk),effective.expand_as(qk),p)])
  values=list(args);values[1]=effective;return tuple(values)

 def post(self,layer,args,result):
  if self.capture:
   error=float((result-self.pending.pop(id(layer))).abs().max());self.errors.append(error);assert error<2e-5

 def end(self):
  if self.capture:
   assert len(self.parts)==4;self.frames.append(np.stack(self.parts));self.states.append(self.state)
  self.capture=False

 def finish(self):
  arr=np.stack(self.frames)
  np.savez_compressed(self.out/'attention_raw.npz',**{key:arr[:,:,i].transpose(0,2,1,3,4,5) for i,key in enumerate(('qk_scores','relation_bias','effective_bias','attention'))})
  meta=dict(agents=self.m,boxes=self.o,scenes=self.task.num_envs,target=self.target,strength=None if self.hard_mask else self.strength,hard_mask=self.hard_mask,blocked_attention_max_error=max(self.mask_probability_errors,default=0.),layers=[i+1 for i in sorted(self.layers)],scope=self.scope,scope_definition='all_owner: every H/B/G query gets -lambda on every different-owner H/B/G key; human: human queries only. Within-owner entries unchanged. Actor only.',initial_state_sha256=self.initial_hash,layer_reconstruction_max_error=max(self.errors),bias_correction_max_error=max(self.bias_errors),frames=len(self.frames),valid_samples=sum(sum(s['active']) for s in self.states))
  (self.out/'capture.json').write_text(json.dumps(meta,indent=2));(self.out/'states.json').write_text(json.dumps(self.states))
  for h in self.handles:h.remove()
  self.encoder.forward=self.original_forward
  return meta
