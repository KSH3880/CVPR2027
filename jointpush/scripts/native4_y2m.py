"""Evaluation-only four-person extension; original source files stay untouched.

One PhysX environment contains four people and four boxes. Two independent
source two-person layouts are translated by +/-10m in X. All twelve tokens
participate in every original actor attention layer; no masks or new weights.
"""
import copy
import os
import inspect
import textwrap
from types import SimpleNamespace
import torch
from isaacgym import gymapi,gymtorch
from env.tasks.multi_agent import humanoid_ma_open_push as native
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent import open_push_reward as reward
from utils import open_push_spec
from utils.edge_context_spec import EdgeContextGraph,AT
from utils.edge_ontop_spec import expand_graph
from utils.push_scene import sample_random_at_scene

Native=native.HumanoidMAOpenPush
OriginalRuntime=reward.OpenPushRuntime
original_graph=open_push_spec.compile_graph
original_runtime_step=OriginalRuntime.step
original_runtime_graph=reward.graph_for_tasks
GROUP_OFFSET=10.
BOX_COUNT=int(os.environ.get("JOINTPUSH_BOXES", "4"))
assert BOX_COUNT in (4,8), "JOINTPUSH_BOXES must be 4 or 8"


def graph(spec,m,o,device=None):
    if (m,o)!=(4,BOX_COUNT):return original_graph(spec,m,o,device)
    assert spec.get('push_only') and spec['task_probabilities']=={'OPEN':0.,'PUSH':1.}
    a=torch.arange(4,device=device)
    src=torch.stack((a,a+4),-1).flatten();dst=torch.stack((a+4,a+4+BOX_COUNT),-1).flatten()
    rel=torch.tensor([open_push_spec.PUSH,AT],device=device).repeat(4)
    owner=a.repeat_interleave(2);valid=torch.ones(8,dtype=torch.bool,device=device)
    return EdgeContextGraph(tuple(f'edge{i}' for i in range(8)),4,BOX_COUNT,src,dst,rel,owner,valid,valid.clone(),torch.zeros(8,8,dtype=torch.bool,device=device),torch.full((8,),-1,dtype=torch.long,device=device))


def graph_for_tasks(spec,tasks):
    if tasks.shape[1]==2:return original_runtime_graph(spec,tasks)
    assert tasks.shape[1]==4 and (tasks==1).all()
    return expand_graph(graph(spec,4,BOX_COUNT,tasks.device),len(tasks))


def source_method(cls,name,namespace,replacements):
    """Retain original function math, changing only enumerated count expressions."""
    code=textwrap.dedent(inspect.getsource(getattr(cls,name)))
    for old,new in replacements:
        assert old in code,(name,old)
        code=code.replace(old,new)
    scope=dict(namespace)
    exec(compile(code,f'<native4_y2m:{name}>','exec'),scope)
    return scope[name]


class Runtime4(OriginalRuntime):
    def step(self,roots,hands,handles,angles,boxes,goals):
        # Exact source two-person reward, including source pair-local sharing.
        # This scalar reward does not affect frozen inference.
        parts=[]
        for start in (0,2):
            fields={}
            for key,value in vars(self).items():
                if torch.is_tensor(value) and value.ndim>=2:
                    if value.shape[1]==4:value=value[:,start:start+2]
                    elif value.shape[1]==8:value=value[:,2*start:2*(start+2)]
                fields[key]=value
            view=SimpleNamespace(**fields)
            values=[x[:,start:start+2] for x in (roots,hands,handles,angles,boxes,goals)]
            parts.append(original_runtime_step(view,*values))
        return {key:torch.cat([r[key] for r in parts],1) for key in parts[0]}


class Task4(Native):
    def get_open_push_contract(self):
        c=Native.get_open_push_contract(self)
        c['push_only']={'revision':1,'num_objects':BOX_COUNT,'entity_tokens':8+BOX_COUNT}
        c['scaling_evaluation']={'agents':4,'boxes':BOX_COUNT,'group_x_offset':GROUP_OFFSET,'task':'PUSH-only','training':False}
        return c

    def _build_agent_spawn_offsets(self,device_type,device_id):
        from env.tasks.multi_agent.humanoid_ma import HumanoidMA
        HumanoidMA._build_agent_spawn_offsets(self,device_type,device_id)
        anchors=self._agent_spawn_offsets.new_tensor(self._push_scene_spec['owner_anchor_xy'])
        values=torch.cat((anchors+anchors.new_tensor([-GROUP_OFFSET,0.]),anchors+anchors.new_tensor([GROUP_OFFSET,0.])))
        self._agent_spawn_offsets[:,:2]=values
        self._agent_spawn_offsets[:,2]=0.

    def _build_box_tensors(self):
        HumanoidMACarry._build_box_tensors(self)
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs,-1,3)
        self._box_forces=forces[:,4*self.num_bodies:4*self.num_bodies+4]
        self._no_door_angles=torch.zeros(self.num_envs,4,device=self.device)
        self._no_door_handles=torch.zeros(self.num_envs,4,3,device=self.device)

    def _build_assignment_tensors(self):
        HumanoidMACarry._build_assignment_tensors(self)
        self._agent_zone=torch.arange(4,device=self.device).expand(self.num_envs,4).clone()
        self._scene_generator=torch.Generator(device=self.device).manual_seed(torch.initial_seed()+209759)
        self._zone_generator=torch.Generator(device=self.device).manual_seed(torch.initial_seed()+104729)
        self._push_reset_states=torch.zeros(self.num_envs,4,13,device=self.device)
        self._push_reset_states[...,6]=1.
        self._push_goal_direction=torch.zeros(self.num_envs,4,3,device=self.device)
        self._push_goal_direction[...,1]=1.
        self._push_random_goals=torch.zeros(self.num_envs,4,3,device=self.device)
        self._push_goal_distance=torch.zeros(self.num_envs,4,device=self.device)

    def _zone_values(self,values,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        return self._gather_entity_slots(values[ids],self._agent_zone[ids])

    def _sample_box_assignments(self,ids):
        self._agent_zone[ids]=self._sample_zone_assignments(ids)
        self._agent_box_assignment[ids]=self._agent_zone[ids]
        extras=torch.arange(4,self.num_objects,device=self.device).expand(len(ids),-1)
        self._logical_box_order[ids]=torch.cat((self._agent_zone[ids],extras),-1)

    def _sample_zone_assignments(self,ids):
        identity=torch.arange(4,device=self.device).expand(len(ids),4)
        swap=torch.rand(len(ids),2,device=self.device,generator=self._zone_generator)<self._zone_swap_probability
        return torch.where(swap.repeat_interleave(2,1),identity^1,identity).clone()

    def _sample_tasks(self,ids):
        assert self._task_graph_preset=='push'
        return torch.ones(len(ids),4,dtype=torch.long,device=self.device)

    def _init_relation_runtime(self):
        self.relation_runtime=Runtime4(self.num_envs,self._relation_graph_spec,self._relation_cfg,self._hold_spec,self.dt,self.device,self._push_sizes(),self._manipulation_spec)

    def _reset_boxes(self,ids):
        self._box_states[ids]=self._initial_box_states[ids]
        self._box_states[ids,:,7:]=0.
        for start,shift in [(0,-GROUP_OFFSET),(2,GROUP_OFFSET)]:
            xy,q,goals=sample_random_at_scene(len(ids),self._viewer_geometry_env,self.device,self._scene_generator)
            xy[:,:,0]+=shift;goals[:,:,0]+=shift
            self._box_states[ids,start:start+2,:2]=xy
            self._box_states[ids,start:start+2,3:7]=q
            self._push_random_goals[ids,start:start+2,:2]=goals
        if BOX_COUNT==8:
            # Four unassigned boxes beside the four task lanes, outward from
            # the two groups. They remain real dynamic objects and visible tokens.
            self._box_states[ids,4:]=self._box_states[ids,:4]
            self._box_states[ids,4:6,0]-=4.
            self._box_states[ids,6:8,0]+=4.
            self._box_states[ids,4:,2]=self._box_size[ids,4:,2]/2
        self._push_random_goals[ids,:,2]=self._box_states[ids,:4,2]
        self._push_reset_states[ids]=self._box_states[ids,:4]

    def _apply_joint_rsi(self,*args,**kwargs):
        raise RuntimeError('Four-person extension is loco-start evaluation only')


def install():
    assert original_graph is open_push_spec.compile_graph,'Install only once per process'
    open_push_spec.compile_graph=graph
    reward.graph_for_tasks=graph_for_tasks
    runtime_ns=dict(vars(reward),OriginalRuntime=OriginalRuntime,compile_graph=graph,BOX_COUNT=BOX_COUNT)
    Runtime4.__init__=source_method(OriginalRuntime,'__init__',runtime_ns,[
        ('super().__init__','super(OriginalRuntime,self).__init__'),
        ('compile_graph(spec,2,2 if spec.get(\'push_only\',False) else 4,device)','compile_graph(spec,4,BOX_COUNT,device)'),
        ('(n,2','(n,4')])
    ns=dict(vars(native),Native=Native)
    Task4.__init__=source_method(Native,'__init__',ns,[
        ('super().__init__','super(Native,self).__init__'),
        ('2*self.num_dof','self.num_agents*self.num_dof'),
        ('self._box_states[:,self._push_slot_start:,:3].clone()', 'self._owned_push_states()[...,:3].clone()'),
        ('self.num_envs,2,','self.num_envs,self.num_agents,')])
    Task4._load_box_asset=source_method(Native,'_load_box_asset',ns,[
        ('(self.num_envs,2)','(self.num_envs,self.num_objects)')])
    Task4._build_box=source_method(Native,'_build_box',ns,[('a=object_id%2','a=object_id%self.num_objects')])
    Task4.pre_physics_step=source_method(Native,'pre_physics_step',ns,[
        ('super().pre_physics_step','super(Native,self).pre_physics_step'),
        ('2*self.num_dof','self.num_agents*self.num_dof')])
    Task4._physics_step=source_method(Native,'_physics_step',ns,[
        ('self.num_envs,2,-1,-1','self.num_envs,self.num_agents,-1,-1')])
    Task4._compute_reward=source_method(Native,'_compute_reward',ns,[('range(2)','range(self.num_agents)')])
    import env.tasks.multi_agent.humanoid_ma_carry as carry_module
    carry_ns=dict(vars(carry_module))
    HumanoidMACarry.__init__=source_method(HumanoidMACarry,'__init__',carry_ns,[
        ('super().__init__','super(HumanoidMACarry,self).__init__'),
        ("self.num_objects = 2 if cfg['env']['relationGraph'].get('push_only',False) else 4", "self.num_objects = cfg['env']['numObjects'] if cfg['env']['relationGraph'].get('push_only',False) else 4"),
        ('self._context_suffix_width = semantic_packet_size(4)',"self._context_suffix_width = semantic_packet_size(2*cfg['env']['numAgents'])")])
    native.HumanoidMAOpenPush=Task4
    from learning.multi_agent.ma_players import MAPlayerContinuous
    from utils.open_push_spec import checkpoint_contracts_match
    original_restore=MAPlayerContinuous.restore
    def restore(self,fn):
        task=self.env.task
        actual=task.get_open_push_contract()
        expected=copy.deepcopy(actual);expected.pop('scaling_evaluation')
        expected['push_only']={'revision':1,'num_objects':2,'entity_tokens':6}
        saved=torch.load(fn,map_location='cpu',weights_only=False)
        assert checkpoint_contracts_match(saved['open_push_contract'],expected),'Contract differs outside explicit counts/layout'
        # Let the unchanged original restore perform all metadata and strict
        # model/RMS loading against the validated source contract.
        task.get_open_push_contract=lambda:expected
        try:return original_restore(self,fn)
        finally:del task.get_open_push_contract
    MAPlayerContinuous.restore=restore


def validate_cpu(task):
    from utils.push_scene import random_at_goal_valid,segment_distance
    assert task.device=='cpu' or str(task.device)=='cpu'
    assert task.num_agents==4 and task.num_objects==BOX_COUNT
    assert task.gym.get_actor_count(task.envs[0])==4+BOX_COUNT
    assert task.gym.get_env_dof_count(task.envs[0])==4*task.num_dof
    assert task.gym.get_env_rigid_body_count(task.envs[0])==4*task.num_bodies+BOX_COUNT
    for handle in task._box_handles[:BOX_COUNT]:
        mass=sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[0],handle))
        assert abs(mass-20.)<1e-3
    ids=torch.arange(task.num_envs,device=task.device)
    samples=0;min_cross=float('inf')
    for _ in range(125):
        task.reset(ids)
        zones=task._agent_zone
        assert torch.equal(zones.sort(-1).values,torch.arange(4).expand(len(ids),-1))
        assert torch.equal(zones//2,torch.arange(4).expand(len(ids),-1)//2)
        assert torch.equal(task._agent_box_assignment,zones)
        torch.testing.assert_close(task._owned_push_states(),task._logical_box_values(task._box_states)[:,:4])
        torch.testing.assert_close(task._push_sizes(),task._logical_box_values(task._box_size)[:,:4])
        for start,shift in [(0,-GROUP_OFFSET),(2,GROUP_OFFSET)]:
            xy=task._push_reset_states[:,start:start+2,:2].clone();xy[:,:,0]-=shift
            goals=task._push_random_goals[:,start:start+2,:2].clone();goals[:,:,0]-=shift
            assert random_at_goal_valid(xy,task._push_reset_states[:,start:start+2,3:7],goals,task.cfg['env']).all()
        xy=task._push_reset_states[...,:2];goal=task._push_random_goals[...,:2]
        gap=segment_distance(xy[:,:2,None],goal[:,:2,None],xy[:,None,2:],goal[:,None,2:]).amin()
        min_cross=min(min_cross,float(gap));assert gap>=task.cfg['env']['pushScene']['path_separation']
        samples+=len(ids)*4
    if task.num_envs>=2:
        roots=task._humanoid_root_states[1].clone();boxes=task._box_states[1].clone();goals=task._tar_pos[1].clone()
        task.reset(ids[:1])
        torch.testing.assert_close(roots,task._humanoid_root_states[1],rtol=0,atol=0)
        torch.testing.assert_close(boxes,task._box_states[1],rtol=0,atol=0)
        torch.testing.assert_close(goals,task._tar_pos[1],rtol=0,atol=0)
    return dict(reset_agent_samples=samples,unique_box_assignment=True,partial_reset_isolated=True,box_mass_kg=20.,actor_count=4+BOX_COUNT,dofs=4*task.num_dof,rigid_bodies=4*task.num_bodies+BOX_COUNT,min_cross_group_path_distance_m=min_cross)


def compare_pair_readout(player,obs):
    import copy
    encoder=player.model.a2c_network.actor_encoder
    small=copy.deepcopy(encoder);small.set_entity_counts(2,2);small.eval()
    normalized=player._preproc_obs(obs['obs'])
    b=len(normalized)
    h=normalized[:,:892].reshape(b,4,223)
    objects=normalized[:,892:1012].reshape(b,4,30)
    goals=normalized[:,1012:1016].reshape(b,4,1)
    poses=normalized[:,1016:1100].reshape(b,12,7)
    packet=normalized[:,1100:].reshape(b,8,5)
    expected=[]
    with torch.no_grad():
        full=encoder(normalized,token_order=torch.arange(12,device=normalized.device))
        for start in (0,2):
            ids=torch.tensor([start,start+1,4+start,5+start,8+start,9+start],device=normalized.device)
            mapping=torch.full((12,),-1,dtype=torch.long,device=normalized.device)
            mapping[ids]=torch.arange(6,device=normalized.device)
            edges=packet[:,2*start:2*(start+2)].clone()
            edges[:,:,1]=mapping[edges[:,:,1].long()]
            edges[:,:,2]=mapping[edges[:,:,2].long()]
            edges[:,:,4]-=start
            small_obs=torch.cat((h[:,start:start+2].flatten(1),objects[:,start:start+2].flatten(1),goals[:,start:start+2].flatten(1),poses[:,ids].flatten(1),edges.flatten(1)),1)
            expected.append(small(small_obs,token_order=torch.arange(6,device=normalized.device)).reshape(b,2,64))
        expected=torch.cat(expected,1)
        groups=torch.tensor([0,0,1,1]*3,device=normalized.device)
        def mask(module,args):
            values=list(args)
            values[1]=values[1].masked_fill(groups[:,None]!=groups[None,:],-1e9)
            return tuple(values)
        handles=[layer.register_forward_pre_hook(mask) for layer in encoder.layers]
        try:paired=encoder(normalized,token_order=torch.arange(12,device=normalized.device))
        finally:
            for handle in handles:handle.remove()
        error=float((paired-expected).abs().max())
        action_error=float((player.model.a2c_network.action_head(paired)-player.model.a2c_network.action_head(expected)).abs().max())
        torch.testing.assert_close(paired,expected,rtol=2e-5,atol=2e-5)
    return dict(pair_mask_matches_two_source_encoders=True,max_feature_error=error,max_action_head_error=action_error,full_vs_pair_feature_rms=float((full-paired).square().mean().sqrt()),note='CPU same-state feature/readout identity check only; GPU rollout used full unmasked attention')
