import isaacgym
import sys,json,tempfile
from pathlib import Path
r=Path(__file__).resolve().parents[2];sys.path[:0]=[str(r/'tokenhsi'),str(r/'tokenhsi/tests')]
import torch,yaml
from utils.unified_training import validate_unified_env,validate_stage1_network
from utils.hold_at_ft import transfer_78k,object_catalog
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from env.tasks.multi_agent.humanoid_chair_hold_at_ft import HumanoidChairHoldAtFT
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from isaacgym.torch_utils import quat_rotate
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet
torch.set_num_threads(1)
env=yaml.safe_load((r/'tokenhsi/data/cfg/multi_agent/chair_hold_at_full_ft.yaml').read_text())['env']
train=yaml.safe_load((r/'tokenhsi/data/cfg/train/rlg/chair_hold_at_full_ft.yaml').read_text())
validate_unified_env(env);validate_stage1_network(env,train)
base=yaml.safe_load((r/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
assert env['relationReward']==base['relationReward']
assert HumanoidChairHoldAtFT._compute_reward is HumanoidMACarry._compute_reward
ckpt=torch.load(r/'checkpoints/Stage1V2Fixed_00078000.pth',map_location='cpu',weights_only=False)
b=AMPMultiAgentBuilder();b.load(train['params']['network'])
# 223 human +30 object +1 goal +3*7 GTA +2*5 packet =280
net=b.build('amp',actions_num=32,input_shape=(280,),amp_input_shape=(1320,),value_size=1,num_agents=1,num_objects=1,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode=env['relationReward']['mode'],relation_graph_spec=env['relationGraph'],device='cpu')
model=torch.nn.Module();model.a2c_network=net
transfer_78k(model,ckpt)
assert all(torch.equal(v,ckpt['model'][k]) for k,v in model.state_dict().items())
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
rms=SceneRunningMeanStd([223,30,1],[1,1,1],[223,30,0],7,5)
rms.load_state_dict(ckpt['running_mean_std'],strict=True)
graph=compose_canonical_graph(torch.full((4,1),3),num_objects=1)
kin=torch.randn(4,3,7);kin[...,3:]=torch.nn.functional.normalize(kin[...,3:],dim=-1)
obs=torch.cat([torch.randn(4,254),kin.flatten(1),task_graph_packet(graph,4)],-1)
assert obs.shape[-1]==280,obs.shape
mu,sigma=net.eval_actor(obs);value=net.eval_critic(obs)
assert torch.isfinite(mu).all() and torch.isfinite(value).all()
# No optimizer/update. Validate partial reset and semantic front/target distribution.
t=HumanoidChairHoldAtFT.__new__(HumanoidChairHoldAtFT);t.device='cpu';t.num_envs=32
n=32;t._humanoid_root_states=torch.zeros(n,1,13);t._box_states=torch.full((n,1,13),-99.)
t._env_origins=torch.zeros(n,3);t._object_ground_z=torch.full((n,),.6);t._object_front=torch.tensor([[0.,-1.,0.]]).repeat(n,1)
t._platform_pos=torch.zeros(n,1,3);t._platform_default_pos=t._platform_pos.clone();t._tar_pos=torch.full((n,1,3),-99.)
t._tar_platform_pos=t._platform_pos.clone();t._tar_platform_default_pos=t._platform_pos.clone()
ids=torch.arange(0,n,2);t._reset_boxes(ids);t._reset_task(ids)
assert (t._box_states[1::2]==-99).all() and (t._tar_pos[1::2]==-99).all()
d=t._box_states[ids,0,:2].norm(dim=-1);assert ((d>=1)&(d<=5)).all()
front=quat_rotate(t._box_states[ids,0,3:7],t._object_front[ids]);target=-t._box_states[ids,0,:2];target/=target.norm(dim=-1,keepdim=True)
torch.testing.assert_close(front[:,:2],target,atol=1e-6,rtol=1e-6)
xy=t._tar_pos[ids,0,:2];assert (xy.abs()<=4.5).all() and (xy.square().sum(-1)>=2).all()
assert ((xy-t._box_states[ids,0,:2]).square().sum(-1)>=2).all()
try:object_catalog('/tmp/nonexistent_hold_at_dataset','train','ArmChair_Normal')
except FileNotFoundError:pass
else:raise AssertionError('Missing data must fail')
print(json.dumps(dict(source_epoch=ckpt['epoch'],strict_model_tensors=len(model.state_dict()),object_input=30,normalizer_strict=True,cpu_inference=True,reward_unchanged=True,partial_reset=True,front_alignment=True,original_target_distribution=True,gpu_used=False,optimizer_steps=0)))
