"""World-Z stack geometry, ungated saturation, and same-scene task sharing."""
import torch
from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, batched, select_graph, graph_packet
from env.tasks.multi_agent.edge_context_reward import EdgeContextRuntime, edge_context


def vertical_extent(quaternion, half_size):
    # Isaac Gym xyzw quaternion. Third row of rotation matrix, absolute projection.
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x,y,z,w = q.unbind(-1)
    row = torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)
    return (row.abs() * half_size).sum(-1)


def ontop_geometry(source, support, source_size, support_size):
    hs = vertical_extent(source[...,3:7], source_size/2)
    ht = vertical_extent(support[...,3:7], support_size/2)
    bottom = source[...,2]-hs; top = support[...,2]+ht
    gap = bottom-top
    target = support[...,:3].clone(); target[...,2] += hs+ht
    delta = source[...,:3]-target
    dxy = delta[...,:2].norm(dim=-1)
    return delta, dict(signed_gap=gap, source_bottom_z=bottom, support_top_z=top,
                       source_extent=hs, support_extent=ht, target=target, distance_xy=dxy)


def mix_task_reward(local):
    if local.ndim != 2 or local.shape[1] != 2:
        raise ValueError('Reward sharing is defined only for same-scene [N,2] task rewards')
    return .9*local + .1*local.flip(-1)


def evaluate_ontop_edges(hands, roots, objects, sizes, goals, graph, config):
    n=roots.shape[0];m=graph.num_agents;o=graph.num_objects
    src=batched(graph.edge_src,n);dst=batched(graph.edge_dst,n)
    rel=batched(graph.edge_relation,n);valid=batched(graph.edge_valid,n)
    batch=torch.arange(n,device=roots.device)[:,None]
    src_box=objects[batch,(src-m).clamp(0,o-1)]
    dst_box=objects[batch,(dst-m).clamp(0,o-1)]
    src_size=sizes[batch,(src-m).clamp(0,o-1)]
    dst_size=sizes[batch,(dst-m).clamp(0,o-1)]
    top_delta, diag=ontop_geometry(src_box,dst_box,src_size,dst_size)
    hold=rel==HOLDING;at=rel==AT;top=rel==ON_TOP
    h=hands.mean(-2)[batch,src.clamp(0,m-1)]
    goal=goals[batch,(dst-m-o).clamp(0,m-1)]
    delta=torch.where(hold[...,None], h-dst_box[...,:3],
                      torch.where(at[...,None],src_box[...,:3]-goal,top_delta))
    progress_delta=torch.where(hold[...,None],roots[batch,src.clamp(0,m-1)]-dst_box[...,:3],delta)
    xy=progress_delta[...,:2].norm(dim=-1)
    phi=torch.exp(-10*delta.square().sum(-1))*valid
    p=config['progress'];progress=1/(1+(xy-p['delta']).clamp_min(0)/p['sigma'])
    diag['target']=torch.where(hold[...,None],dst_box[...,:3],torch.where(at[...,None],goal,diag['target']))
    for key in ('signed_gap','source_bottom_z','support_top_z','source_extent','support_extent'):
        diag[key]=diag[key]*top*valid
    diag.update(progress=progress*valid, z_error=delta[...,2], distance=delta.norm(dim=-1), distance_xy=xy)
    diag['support_speed']=dst_box[...,7:10].norm(dim=-1)*top
    diag['relative_speed']=(src_box[...,7:10]-dst_box[...,7:10]).norm(dim=-1)*top
    diag['source_tilt']=torch.acos((1-2*(src_box[...,3].square()+src_box[...,4].square())).clamp(-1,1))*top
    diag['support_tilt']=torch.acos((1-2*(dst_box[...,3].square()+dst_box[...,4].square())).clamp(-1,1))*top
    return phi,diag


class OnTopContextRuntime(EdgeContextRuntime):
    def suffix(self, ids=None):
        graph=self.graph if ids is None else select_graph(self.graph,ids)
        phi=self.phi if ids is None else self.phi[ids]
        return graph_packet(graph,edge_context(phi,graph))

    def step(self, phi, progress, z_error):
        result=super().step(phi,progress,z_error)
        result['local_task_reward']=result['agent_task_reward']
        result['agent_task_reward']=mix_task_reward(result['local_task_reward'])
        return result
