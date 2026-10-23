"""OPEN terms following Stage-1 state/progress/current-success saturation."""
import torch
from utils.edge_ontop_spec import batched
from utils.edge_stage1_spec import semantic_graph_packet
from env.tasks.multi_agent.edge_context_reward import owner_sum, goal_success, EdgeContextRuntime

def door_reward(roots, hands, handles, angles, graph, config, success_override=None):
    n = roots.shape[0]
    src = batched(graph.edge_src,n)
    obj = batched(graph.edge_dst,n)-graph.num_agents
    rows = torch.arange(n,device=roots.device)[:,None]
    root, hand, handle, angle = roots[rows,src], hands[rows,src], handles[rows,obj], angles[rows,obj]
    d=config['door']
    distance=(root[...,:2]-handle[...,:2]).norm(dim=-1)
    approach=1/(1+(distance-d['approach_delta']).clamp_min(0)/d['approach_sigma'])
    # Either hand may manipulate its own handle; no cross-human averaging.
    hand_distance=(hand-handle[:,:,None]).square().sum(-1).amin(-1)
    interaction=torch.exp(-d['hand_distance_scale']*hand_distance)
    progress=(angle/d['target_angle']).clamp(0,1)
    success=(angle >= d['target_angle']-d['angle_tolerance']) & batched(graph.edge_valid,n)
    if success_override is not None:
        success=success_override & batched(graph.edge_valid,n)
    valid=batched(graph.edge_valid,n)
    approach=torch.where(success,1.,approach)*valid*config['state_reward_weight']*.5
    interaction=torch.where(success,1.,interaction)*valid*config['state_reward_weight']*.5
    progress=torch.where(success,1.,progress)*valid*config['progress_reward_weight']
    bonus=success.float()*config['success_reward_weight']
    total=approach+interaction+progress+bonus
    local=owner_sum(total,graph)
    sharing=config['task_sharing']
    return dict(approach_component=approach,interaction_component=interaction,
        progress_component=progress,success_component=bonus,total=total,
        phi_raw=(angle/d['target_angle']).clamp(0,1),own_success=success,
        agent_task_reward=sharing['self']*local+sharing['teammate']*local.flip(-1),
        angle_error=(d['target_angle']-angle).clamp_min(0))

class DoorRuntime(EdgeContextRuntime):
    def step(self, roots, hands, handles, angles):
        result=door_reward(roots,hands,handles,angles,self.graph,self.config)
        self.phi.copy_(result['phi_raw'])
        self.own_success.copy_(result['own_success'])
        self.achieved |= self.own_success
        self.done.copy_(goal_success(self.own_success,self.graph))
        self.last_result=result
        return result

    def reset(self, ids, angles):
        self.phi[ids]=(angles/self.config['door']['target_angle']).clamp(0,1)
        self.own_success[ids]=angles >= self.config['door']['target_angle']-self.config['door']['angle_tolerance']
        self.achieved[ids]=self.own_success[ids]
        self.done[ids]=self.own_success[ids]

    def suffix(self, ids=None):
        n=self.phi.shape[0] if ids is None else len(ids)
        return semantic_graph_packet(self.graph,n).to(self.phi.device)
