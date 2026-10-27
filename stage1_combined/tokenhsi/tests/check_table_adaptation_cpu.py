import isaacgym
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tokenhsi')]
import torch,yaml
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.table_adaptation_network import TableAdaptationBuilder
from learning.multi_agent.table_adaptation_agent import transfer_policy,ADAPTER_PREFIXES
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet

torch.set_num_threads(1);torch.manual_seed(42)
source=torch.load(ROOT/'checkpoints/Stage1V2Fixed_00078000.pth',map_location='cpu',weights_only=False)
kwargs=dict(actions_num=32,input_shape=(280,),amp_input_shape=(1320,),value_size=1,num_agents=1,num_objects=1,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode='state_relation_edge_stage1_v1',relation_graph_spec=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/table_tokenhsi_ft.yaml').read_text())['env']['relationGraph'],device='cpu')
base_params=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/table_tokenhsi_ft.yaml').read_text())['params']['network'];b=AMPMultiAgentBuilder();b.load(base_params);base=b.build('amp',**kwargs);base.load_state_dict({k.removeprefix('a2c_network.') if hasattr(k,'removeprefix') else k[len('a2c_network.'):]:v for k,v in source['model'].items()},strict=True)
kin=torch.randn(4,3,7);kin[...,3:]=torch.nn.functional.normalize(kin[...,3:],dim=-1)
graph=compose_canonical_graph(torch.full((4,1),3),num_objects=1)
obs=torch.cat((torch.randn(4,254),kin.flatten(1),task_graph_packet(graph,4)),-1)
base.eval()
with torch.no_grad():original=base.eval_actor(obs)[0].detach()
report={}
for mode in ('residual','tokenizer'):
 params=yaml.safe_load((ROOT/f'tokenhsi/data/cfg/train/rlg/table_adapt_{mode}.yaml').read_text())['params']['network'];b=TableAdaptationBuilder();b.load(params);net=b.build('amp',**kwargs)
 model=torch.nn.Module();model.a2c_network=net;transfer_policy(model,source);model.eval()
 with torch.no_grad():initial=net.eval_actor(obs)[0]
 torch.testing.assert_close(initial,original,rtol=1e-6,atol=1e-6)
 before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(ADAPTER_PREFIXES)}
 for n,p in model.named_parameters():p.requires_grad_(n.startswith(ADAPTER_PREFIXES))
 optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-5)
 norms=[]
 for step in range(2):
  optimizer.zero_grad();mu=net.eval_actor(obs)[0];mu.square().mean().backward()
  norms.append(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(ADAPTER_PREFIXES) and p.grad is not None));assert norms[-1]>0
  if mode=='residual' and step==1:assert net.action_residual[0].weight.grad.abs().sum()>0
  assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(ADAPTER_PREFIXES))
  optimizer.step()
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 assert not torch.equal(net.eval_actor(obs)[0],original)
 report[mode]=dict(initial_actor_allclose=True,initial_max_error=float((initial-original).abs().max()),two_backward_grads=norms,frozen_base_exact=True,policy_changed_after_adapter_update=True)
print('TABLE_ADAPTATION_CPU',json.dumps(report))
