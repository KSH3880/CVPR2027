"""Opt-in, read-only actor attention probes and a local live dashboard."""
import atexit
import functools
import http.server
import json
import math
import os
from pathlib import Path
import threading
import time
import torch
import torch.nn.functional as F
from learning.multi_agent.amp_network_builder_ma import apply_gta_transform


def canonical_attention(probability, order):
    if order is None:
        return probability
    inverse = order.argsort()
    return probability.index_select(-2, inverse).index_select(-1, inverse)


@torch.no_grad()
def scene_attention(layer, x, bias, g, ginv):
    # Same projections, GTA and task bias as the actor's forward. Only env0.
    x=x[:1];batch,length,width=x.shape
    q,k,_=layer.qkv(x).view(batch,length,3,layer.num_heads,layer.head_dim).permute(2,0,3,1,4)
    if g is not None:
        q=apply_gta_transform(g[:1].transpose(-1,-2),q)
        k=apply_gta_transform(ginv[:1],k)
    logits=(q@k.transpose(-1,-2))*layer.scale
    if bias is not None:logits=logits+(bias.unsqueeze(0) if bias.ndim==3 else bias[:1])
    return logits.softmax(-1)[0]


@torch.no_grad()
def human_attention(module, inputs, kwargs):
    query,key,_=inputs
    width=module.embed_dim;heads=module.num_heads;dim=width//heads
    weight=module.in_proj_weight;bias=module.in_proj_bias
    q=F.linear(query[:1],weight[:width],None if bias is None else bias[:width])
    k=F.linear(key[:1],weight[width:2*width],None if bias is None else bias[width:2*width])
    q=q.reshape(1,-1,heads,dim).transpose(1,2)
    k=k.reshape(1,-1,heads,dim).transpose(1,2)
    logits=q@k.transpose(-1,-2)/math.sqrt(dim)
    mask=kwargs.get('attn_mask')
    if mask is not None:
        mask=mask if mask.ndim==2 else mask[:heads].unsqueeze(0)
        logits=logits.masked_fill(mask,float('-inf')) if mask.dtype==torch.bool else logits+mask
    return logits.softmax(-1)[0]


class AttentionProbe:
    def __init__(self, network):
        self.network=network;self.enabled=False;self.maps={};self.handles=[]
        encoder=network.actor_encoder
        encoder._capture_attention_order=True
        for index,layer in enumerate(encoder.layers):
            def scene_hook(module,inputs,kwargs,index=index):
                if self.enabled:
                    bias=inputs[1] if len(inputs)>1 else kwargs.get('rel_bias')
                    p=scene_attention(module,inputs[0],bias,kwargs.get('gta_g'),kwargs.get('gta_ginv'))
                    self.maps['gta_'+str(index)]=canonical_attention(p,encoder._attention_order).detach()
            self.handles.append(layer.register_forward_pre_hook(scene_hook,with_kwargs=True))
        coordination=getattr(network,'coordination',None)
        if coordination is not None:
            coordination._capture_attention_order=True
            for index,block in enumerate(coordination.blocks):
                def human_hook(module,inputs,kwargs,index=index):
                    if self.enabled:
                        p=human_attention(module,inputs,kwargs)
                        self.maps['human_'+str(index)]=canonical_attention(p,coordination._attention_order).detach()
                self.handles.append(block.attention.register_forward_pre_hook(human_hook,with_kwargs=True))

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        self.network.actor_encoder._capture_attention_order=False
        if hasattr(self.network,'coordination'):self.network.coordination._capture_attention_order=False


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()


class AttentionViewer:
    def __init__(self,player):
        self.task=player.env.task
        self.directory=Path(self.task.cfg['args'].output_path).resolve()/'attention'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.probe=AttentionProbe(player.model.a2c_network)
        self.interval=max(1,int(os.environ.get('ATTENTION_INTERVAL','10')))
        self.counter=0
        html=Path(__file__).with_name('attention_dashboard.html').read_text()
        html=html.replace('__VIEWER_PORT__',str(int(os.environ.get('PORT','6088'))))
        (self.directory/'index.html').write_text(html)
        handler=functools.partial(QuietHandler,directory=str(self.directory))
        port=int(os.environ.get('ATTENTION_PORT','6089'))
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',port),handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        atexit.register(self.close)
        print(f'[attention] http://localhost:{port} / env0 / actor GTA + human coordination',flush=True)

    def begin(self):
        self.probe.enabled=self.counter%self.interval==0
        if self.probe.enabled:self.probe.maps.clear()

    def publish(self,repeat,step):
        if self.probe.enabled:
            task=self.task;m=task.num_agents;o=task.num_objects
            names=[f'H{i}' for i in range(m)]+[f'O{i}' for i in range(o)]+[f'G{i}' for i in range(m)]
            packet=task.relation_runtime.suffix()[:1].reshape(1,m,5)[0].detach().cpu().tolist()
            mapping=task._logical_box_order[0].detach().cpu().tolist()
            labels=['SIT','CLIMB','AT','ON_TOP','PUSH']
            roles=[dict(human=int(row[2]),task=labels[int(row[1])],payload=int(row[3]),target=int(row[4])) for row in packet if row[0]]
            payload=dict(time=time.time(),repeat=repeat,step=step,phase='pre_action',
                checkpoint=str(task.cfg['args'].checkpoint),labels=names,human_labels=names[:m],
                logical_to_physical=mapping,roles=roles,
                predecessors=task._before_predecessor[0].detach().cpu().tolist(),
                scene=dict(humans=task._humanoid_root_states[0,:,:3].detach().cpu().tolist(),
                           objects=task._logical_box_values(task._box_states)[0,:,:3].detach().cpu().tolist(),
                           goals=task._tar_pos[0].detach().cpu().tolist()),
                maps={key:value.cpu().tolist() for key,value in self.probe.maps.items()})
            tmp=self.directory/'latest.tmp';tmp.write_text(json.dumps(payload,allow_nan=False));tmp.replace(self.directory/'latest.json')
            # One immutable sample per episode; live state is overwritten atomically.
            sample=self.directory/f'episode_{repeat:04d}.json'
            if not sample.exists():sample.write_text(json.dumps(payload,allow_nan=False))
        self.counter+=1

    def close(self):
        if getattr(self,'server',None) is not None:
            self.server.shutdown();self.server.server_close();self.server=None
        self.probe.close()
