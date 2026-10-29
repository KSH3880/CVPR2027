import torch
from env.tasks.multi_agent.before_reward import BeforeRuntime
from utils.edge_ontop_spec import batched
from env.tasks.multi_agent.edge_context_reward import owner_sum


class BeforePushRuntime(BeforeRuntime):
    def __init__(self,*args,contact_weights,**kwargs):
        super().__init__(*args,**kwargs)
        self.contact_weights=contact_weights
        self.contact=torch.zeros_like(self.done)
        self._reset_contact=None

    def _raw_success(self,phi,z_error,feet_error,region_error,graph):
        raw=super()._raw_success(phi,z_error,feet_error,region_error,graph)
        owner=batched(graph.edge_owner,len(phi))
        contact=self.contact if self._reset_contact is None else self._reset_contact
        return torch.where(batched(graph.edge_relation,len(phi))==12,contact.gather(1,owner),raw)&graph.edge_valid

    def reset(self,ids,*args,**kwargs):
        self.contact[ids]=False
        self._reset_contact=self.contact[ids]
        try:return super().reset(ids,*args,**kwargs)
        finally:self._reset_contact=None

    def components(self,phi,progress,z_error,feet_error,region_error,graph):
        result=super().components(phi,progress,z_error,feet_error,region_error,graph)
        push=batched(graph.edge_relation,len(phi))==12
        saturated=result['reward_saturated']
        gate=result['prerequisite_gate']
        for i,(name,raw) in enumerate((('state_component',phi),('progress_component',progress),('success_component',torch.zeros_like(phi)))):
            value=torch.where(saturated,1.,raw)*self.contact_weights[i]*graph.edge_valid
            if i==0:value=value*gate
            result[name]=torch.where(push,value,result[name])
        result['total']=result['state_component']+result['progress_component']+result['success_component']
        local=owner_sum(result['total'],graph)
        for key in ('local_task_reward','agent_task_reward'):result[key]=local
        return result
