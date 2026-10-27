"""CPU execution of the real full-state restore path using a 2-epoch GPU checkpoint."""
import isaacgym
from pathlib import Path
import sys,json
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tokenhsi')]
import torch,yaml
from learning.multi_agent.table_adaptation_network import TableAdaptationBuilder
from learning.multi_agent.table_adaptation_agent import TableAdaptationAgent
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from rl_games.algos_torch.running_mean_std import RunningMeanStd

torch.set_num_threads(1)
r=ROOT.parent/'runs/table_adaptation_compare_20261027/output_etc/check'
p=next((r/'residual').glob('*/nn/*.pth'));saved=torch.load(p,map_location='cpu')
params=saved['relation_experiment_config']['train']['params']['network'];env=saved['relation_experiment_config']['env']
b=TableAdaptationBuilder();b.load(params)
net=b.build('amp',actions_num=32,input_shape=(280,),amp_input_shape=(1320,),value_size=1,num_agents=1,num_objects=1,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode='state_relation_edge_stage1_v1',relation_graph_spec=env['relationGraph'],device='cpu')
a=TableAdaptationAgent.__new__(TableAdaptationAgent);a.model=torch.nn.Module();a.model.a2c_network=net
flags=json.loads((p.parents[1]/'adaptation_trainable.json').read_text())
for name,weight in a.model.named_parameters():weight.requires_grad_(flags[name])
a.optimizer=torch.optim.Adam([v for v in a.model.parameters() if v.requires_grad],lr=2e-5,eps=1e-8)
a.ppo_device='cpu';a.normalize_input=a.normalize_value=a._normalize_amp_input=True;a.has_central_value=a.mixed_precision=False
a.running_mean_std=SceneRunningMeanStd([223,30,1],[1,1,1],[223,30,0],7,5)
a._amp_input_mean_std=RunningMeanStd(tuple(saved['amp_input_mean_std']['running_mean'].shape))
a.value_mean_std=RunningMeanStd(tuple(saved['reward_mean_std']['running_mean'].shape))
a.experiment_dir=str(r/'resume_cpu');Path(a.experiment_dir).mkdir(exist_ok=True)
a.vec_env=SimpleNamespace(env=SimpleNamespace(task=SimpleNamespace(cfg={'env':env,'args':SimpleNamespace(test=False,eval=False)},_relation_cfg=env['relationReward'],_relation_graph_spec=env['relationGraph'],num_agents=1,num_objects=1)),set_env_state=lambda state:None)
a.restore(str(p))
assert a.epoch_num==2
print('RESUME_CPU_OK',json.loads((Path(a.experiment_dir)/'resume_restore.json').read_text()))
