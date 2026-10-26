"""Public TokenHSI 40k policy on the existing shared evaluation scene."""
from pathlib import Path
import ast,importlib.util,json,hashlib
import torch,yaml
from isaacgym.torch_utils import quat_rotate,quat_mul
from utils import torch_utils
from rl_games.algos_torch.running_mean_std import RunningMeanStd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'TokenHSI/tokenhsi'


def source_function(path,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];space=dict(torch=torch,torch_utils=torch_utils,quat_rotate=quat_rotate,quat_mul=quat_mul)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
    return space[name]


class PublicPolicy:
    def __init__(self,task,player,checkpoint):
        self.task,self.player=task,player;self.first=True
        self.location=source_function(BASE/'env/tasks/multi_task/humanoid_traj_sit_carry_climb.py','compute_location_observations')
        self.human=source_function(BASE/'env/tasks/humanoid.py','compute_humanoid_observations_max')
        spec=importlib.util.spec_from_file_location('public_tokenhsi_network',BASE/'learning/transformer/amp_network_builder_transformer.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        params=yaml.safe_load((BASE/'data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml').read_text())['params']['network']
        sizes=[20,38,42,27];index=[0,20,58,100,127];mask=torch.zeros((4,127),dtype=torch.bool,device=task.device)
        for i in range(4):mask[i,index[i]:index[i+1]]=True
        self.mask=mask;self.onehot=torch.zeros((task.num_envs,4),device=task.device);self.onehot[:,1 if task._action=='SIT' else 3]=1
        info=dict(onehot_size=4,tota_subtask_obs_size=127,each_subtask_obs_size=sizes,each_subtask_obs_indx=index,each_subtask_name=['traj','sit','carry','climb'],enable_task_mask_obs=True)
        builder=module.AMPTransformerMultiTaskBuilder();builder.load(params)
        weights=torch.load(checkpoint,map_location=task.device,weights_only=False)
        self.net=builder.build('amp',actions_num=32,input_shape=(354,),num_seqs=task.num_envs,value_size=1,amp_input_shape=(1330,),self_obs_size=223,task_obs_size=131,multi_task_info=info,device=task.device).to(task.device).eval()
        self.net.load_state_dict({k.removeprefix('a2c_network.') if hasattr(k,'removeprefix') else k[len('a2c_network.'):]:v for k,v in weights['model'].items()},strict=True)
        self.rms=RunningMeanStd((354,)).to(task.device).eval();self.rms.load_state_dict(weights['running_mean_std'])
        self.entry_surface=torch.tensor([e['seat'] if task._action=='SIT' or task._category=='Chair' else [0,0,e['metadata_height']] for e in task._evaluation_catalog],device=task.device,dtype=torch.float32)[task._evaluation_ids]
        signs=torch.tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],[1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]],device=task.device)
        self.bps=task._box_size[:,0,None]*signs[None]/2
        features=task._chair_features[:,0];self.facing=torch.cat([-features[:,1:3],torch.zeros_like(features[:,:1])],-1)
        if task._category!='Chair':self.facing[:]=torch.tensor([1.,0.,0.],device=task.device)
        self.metadata=dict(epoch=weights['epoch'],checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),observation_size=354,model_keys_strict=True,original_observation_functions=True)
        print('PUBLIC_POLICY',json.dumps(self.metadata),flush=True)

    def action(self):
        t=self.task;root=t._humanoid_root_states[:,0];obj=t._box_states[:,0]
        if self.first:
            body=t._kinematic_humanoid_rigid_body_states[:,0];pos,rot,vel,ang=(body[...,:3],body[...,3:7],body[...,7:10],body[...,10:13]);self.first=False
        else:pos,rot,vel,ang=(t._rigid_body_pos[:,0],t._rigid_body_rot[:,0],t._rigid_body_vel[:,0],t._rigid_body_ang_vel[:,0])
        human=self.human(pos,rot,vel,ang,False,False)
        target=obj[:,:3]+quat_rotate(obj[:,3:7],self.entry_surface);target[:,2]+=.1 if t._action=='SIT' else .94
        traj=root[:,None,:3].expand(-1,10,-1)
        loc=self.location(root,traj,target,obj,self.bps,self.facing,obj,self.bps,target,obj,self.bps,target,self.onehot,self.mask,True)
        raw=torch.cat([human,loc,self.onehot],-1);assert raw.shape==(t.num_envs,354) and torch.isfinite(raw).all()
        normalized=self.rms(raw)
        # Original TransPlayer restores the onehot after RMS. The transformer
        # only reads its argmax; verify this also for earlier equivalent runs.
        assert torch.equal(normalized[:,-4:].argmax(-1),self.onehot.argmax(-1))
        normalized[:,-4:]=normalized[:,-4:]*self.rms.running_var[-4:].sqrt()+self.rms.running_mean[-4:]
        mu,_=self.net.eval_actor(normalized,raw)
        return (self.player.actions_high-self.player.actions_low)/2*mu.clamp(-1,1)+(self.player.actions_high+self.player.actions_low)/2
