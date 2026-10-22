"""Process-local key masking and faithful attention capture for frozen evaluation."""
import json,os,hashlib,types
from pathlib import Path
import numpy as np
import torch
from learning.multi_agent.amp_network_builder_ma import apply_gta_transform

class AttentionAblation:
    def __init__(self,player,out):
        self.player=player;self.task=player.env.task;self.out=Path(out)
        self.encoder=player.model.a2c_network.actor_encoder
        assert self.encoder.entity_counts==[4,8,4] and not self.encoder.geometry_enabled
        self.boxes=[int(x) for x in os.environ.get('JOINTPUSH_MASK_BOXES','').split(',') if x]
        assert len(self.boxes)==len(set(self.boxes)) and all(4<=i<8 for i in self.boxes)
        self.token_ids=[4+i for i in self.boxes]
        self.strict_owner=os.environ.get('JOINTPUSH_STRICT_OWNER_BOX','0')=='1'
        self.perspective=None
        self.owner_only=os.environ.get('JOINTPUSH_OWNER_BOX_ONLY','0')=='1'
        self.blocked_pairs=[(i,4+j) for i in range(4) for j in range(4) if i!=j] if self.owner_only else []
        self.frames=[];self.states=[];self.layers=[];self.errors=[];self.zero_errors=[]
        self.capture=False;self.pending={};self.order=None;self.handles=[]
        self.original_forward=self.encoder.forward
        def forward(encoder,obs,return_all=False,token_order=None):
            # Preserve exactly one original random permutation per forward.
            count=sum(encoder.entity_counts)
            if token_order is None:
                token_order=torch.randperm(count,device=obs.device) if encoder.relation_graph_spec.get('shuffle_token_order',False) else torch.arange(count,device=obs.device)
            self.order=token_order
            if not self.strict_owner:
                return self.original_forward(obs,return_all=return_all,token_order=token_order)
            assert not return_all, 'Strict owner mode reads per-owner human features only'
            features=[]
            for owner in range(4):
                self.perspective=owner
                features.append(self.original_forward(obs,token_order=token_order)[:,owner])
            self.perspective=None
            return torch.stack(features,1)
        self.encoder.forward=types.MethodType(forward,self.encoder)
        for layer in self.encoder.layers:
            self.handles.extend((layer.register_forward_pre_hook(self.pre),layer.register_forward_hook(self.post)))
        self.initial_hash=None

    def begin(self,step,active):
        self.capture=step in (0,30,60,120,180,299) and bool(active[:2].any())
        self.layers=[]
        if self.initial_hash is None:
            h=hashlib.sha256()
            for x in (self.task._humanoid_root_states,self.task._box_states,self.task._tar_pos,self.task.obs_buf):
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            self.initial_hash=h.hexdigest()
        if self.capture:
            self.state=dict(step=step,active=active[:2].cpu().tolist(),assignment=self.task._agent_box_assignment[:2].cpu().tolist(),humans=self.task._humanoid_root_states[:2,:,:3].cpu().tolist(),boxes=self.task._logical_box_values(self.task._box_states)[:2,:,:3].cpu().tolist(),goals=self.task._tar_pos[:2].cpu().tolist())

    def pre(self,layer,args):
        x,bias,geo,msg,g,gi=args[:6];B,L,d=x.shape
        assert self.order is not None and geo is None and msg is None
        hidden=torch.zeros(L,dtype=torch.bool,device=x.device)
        tokens=list(self.token_ids)
        if self.strict_owner:
            assert self.perspective is not None
            tokens += [4+i for i in range(4) if i!=self.perspective]
        for token in tokens:hidden |= self.order==token
        if bias is None:bias=x.new_zeros(layer.num_heads,L,L)
        blocked=hidden[None,:].expand(L,L).clone()
        for query,key in self.blocked_pairs:
            blocked |= (self.order[:,None]==query)&(self.order[None,:]==key)
        effective=bias.masked_fill(blocked[None,:,:],float('-inf'))
        if self.capture:
            q,k,v=layer.qkv(x).view(B,L,3,layer.num_heads,layer.head_dim).permute(2,0,3,1,4)
            if g is not None:
                q=apply_gta_transform(g.transpose(-1,-2),q);k=apply_gta_transform(gi,k);v=apply_gta_transform(gi,v)
            qk=(q@k.transpose(-1,-2))*layer.scale
            p=(qk+effective).softmax(-1)
            zero=float(p[...,blocked].abs().max()) if blocked.any() else 0.
            self.zero_errors.append(zero);assert zero==0.
            z=p@v
            if g is not None:z=apply_gta_transform(g,z)
            z=z.transpose(1,2).reshape(B,L,d)
            z=layer.norm1(x+layer.proj(z));expected=layer.norm2(z+layer.linear2(layer.act(layer.linear1(z))))
            self.pending[id(layer)]=expected
            inverse=self.order.argsort()
            def canonical(a):
                return a[:2].index_select(-2,inverse).index_select(-1,inverse).detach().cpu().numpy().copy()
            self.layers.append([canonical(a) for a in (qk,bias.expand_as(qk),p)])
        values=list(args);values[1]=effective
        return tuple(values)

    def post(self,layer,args,result):
        if self.capture:
            error=float((result-self.pending.pop(id(layer))).abs().max())
            self.errors.append(error);assert error<2e-5,error

    def end(self):
        if self.capture:
            assert len(self.layers)==(16 if self.strict_owner else 4)
            frame=np.stack(self.layers)
            if self.strict_owner:frame=frame.reshape(4,4,*frame.shape[1:])
            self.frames.append(frame);self.states.append(self.state)
        self.capture=False

    def finish(self):
        arr=np.stack(self.frames)
        if self.strict_owner:
            for owner in range(4):
                folder=self.out/f'perspective_H{owner}';folder.mkdir(exist_ok=True)
                view=arr[:,owner]
                np.savez_compressed(folder/'attention_raw.npz',**{k:view[:,:,i].transpose(0,2,1,3,4,5) for i,k in enumerate(('qk_scores','relation_bias','attention'))})
                (folder/'states.json').write_text(json.dumps(self.states))
            arr=arr[:,0]
        np.savez_compressed(self.out/'attention_raw.npz',**{k:arr[:,:,i].transpose(0,2,1,3,4,5) for i,k in enumerate(('qk_scores','relation_bias','attention'))})
        metadata=dict(agents=4,boxes=8,scenes=self.task.num_envs,seed=self.task.cfg['args'].seed,checkpoint=self.task.cfg['args'].checkpoint,masked_physical_boxes=self.boxes,masked_logical_boxes=self.boxes,masked_token_ids=self.token_ids,masked_key_probability_max=max(self.zero_errors),layer_reconstruction_max_error=max(self.errors),initial_state_sha256=self.initial_hash,strict_owner_box=self.strict_owner,owner_box_only=self.owner_only,blocked_pairs=self.blocked_pairs,policy_intervention=bool(self.boxes) or self.owner_only or self.strict_owner,mask_scope='All actor layers/heads: selected extra box keys blocked for all queries; if owner_box_only, H_i -> B_j (j != i) blocked. Direct attention only; indirect paths via other tokens remain. Physics/rewards/critic/weights retained.',frames=len(self.frames),valid_samples=sum(sum(s['active']) for s in self.states),device=str(self.task.device))
        if self.strict_owner:
            metadata['mask_scope']='Four owner-specific encoder passes. Every query/layer blocks foreign assigned box keys; action H_i reads only perspective H_i. Humans/goals retained; foreign-box token information has no attention path to the output.'
            metadata['stored_root_perspective']=0
            for owner in range(4):
                per=dict(metadata,stored_root_perspective=owner,masked_token_ids=self.token_ids+[4+i for i in range(4) if i!=owner])
                (self.out/f'perspective_H{owner}'/'capture.json').write_text(json.dumps(per,indent=2))
            metadata['masked_token_ids']=self.token_ids+[4+i for i in range(4) if i!=0]
        (self.out/'states.json').write_text(json.dumps(self.states));(self.out/'capture.json').write_text(json.dumps(metadata,indent=2))
        for h in self.handles:h.remove()
        self.encoder.forward=self.original_forward
        return metadata
