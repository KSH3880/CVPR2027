"""Soft approach shaping; box-goal success saturates CONTACT/AT without hand gates."""
import torch
from env.tasks.multi_agent.jointpush_reward import Stage1PushRuntime
from env.tasks.multi_agent.edge_context_reward import owner_sum
from utils.edge_ontop_spec import batched
from utils.stage2_jointpush import inverse_rotate, group_sizes


def soft_group_gate(hands, boxes, sizes, goals, groups, sigma=.18):
    if sigma <= 0:
        raise ValueError('Gate sigma must be positive')
    local=inverse_rotate(boxes[...,None,3:7].expand(*hands.shape[:-1],4),hands-boxes[...,None,:3])
    dx=(local[...,0].abs()-sizes[...,None,0]/2).clamp_min(0)
    dz=(local[...,2].abs()-sizes[...,None,2]/2).clamp_min(0)
    distance2=dx.square()+dz.square()+(local[...,1]+sizes[...,None,1]/2).square()
    quality=torch.exp(-distance2.amax(-1)/(sigma*sigma))
    normal=torch.zeros_like(sizes);normal[...,1]=1
    from utils.jointpush_geometry import rotate
    normal=rotate(boxes[...,3:7],normal)
    delta=goals-boxes[...,:3]
    at_position=(delta.norm(dim=-1)<=.2)&(delta[...,2].abs()<=.001)
    aligned=(normal[...,:2]*delta[...,:2]).sum(-1)>0
    quality=quality*((local[...,1]<=0).all(-1)&(aligned|at_position))
    same=groups[:,:,None]==groups[:,None,:]
    score=(quality[:,None,:]*same).sum(-1)/same.sum(-1)
    return torch.where(group_sizes(groups)>1,score,torch.ones_like(score))


class SoftStage2JointPushRuntime(Stage1PushRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gate=torch.ones_like(self.done)
        self.soft_gate=torch.ones_like(self.phi[:,:self.done.shape[1]])

    def step(self,phi,progress,z_error,feet_error=None,region_error=None):
        result=super().step(phi,progress,z_error,feet_error,region_error)
        owner=batched(self.graph.edge_owner,phi.shape[0])
        at=batched(self.graph.edge_relation,phi.shape[0])==7
        factor=torch.where(at & ~result['own_success'],self.soft_gate.gather(1,owner),1.)
        result['state_component']*=factor
        result['progress_component']*=factor
        result['total']=result['state_component']+result['progress_component']+result['success_component']
        local=owner_sum(result['total'],self.graph)
        result['agent_task_reward']=local
        result['local_task_reward']=local
        result['group_gate']=torch.where(self.done,1.,self.soft_gate)
        result['soft_group_gate_raw']=self.soft_gate.clone()
        result['hard_group_gate']=self.gate.clone()
        return result


def install(env):
    spec=env['stage2JointPush'].get('reward_gate')
    if not spec:return
    if spec['mode']!='mean_hand_face_quality_v1' or spec['success']!='box_goal_position':
        raise ValueError('Unknown local soft-gate contract')
    import env.tasks.multi_agent.stage2_jointpush_reward as reward
    reward.Stage2JointPushRuntime=SoftStage2JointPushRuntime
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
    original=HumanoidMAStage2JointPush._evaluate_ontop_context
    def evaluate(self,env_ids=None):
        result=original(self,env_ids)
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        bodies=self._rigid_body_pos[ids] if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids,...,:3]
        _,_,hands,boxes,sizes=self._contact_geometry(ids,bodies)
        self.relation_runtime.soft_gate[ids]=soft_group_gate(hands,boxes,sizes,self._tar_pos[ids],self._groups[ids],spec['sigma'])
        return result
    HumanoidMAStage2JointPush._evaluate_ontop_context=evaluate
    print('KSH_JOINTPUSH_SOFT_GATE',spec,flush=True)
