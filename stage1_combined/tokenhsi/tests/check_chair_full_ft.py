import isaacgym
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tokenhsi'),str(root/'tokenhsi/tests')]
import torch,yaml
from learning.multi_agent.chair_ft_network import ChairFTBuilder,transfer
from stage1_test_utils import observations,all_pairs
cfg=yaml.safe_load((root/'tokenhsi/data/cfg/train/rlg/chair_sit_climb_full_ft.yaml').read_text())
env=yaml.safe_load((root/'tokenhsi/data/cfg/multi_agent/chair_sit_climb_full_ft.yaml').read_text())['env']
torch.set_num_threads(1)
b=ChairFTBuilder();b.load(cfg['params']['network'])
net=b.build('amp',actions_num=32,input_shape=(658,),amp_input_shape=(1320,),value_size=1,num_agents=2,num_objects=4,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',scene_entity_sizes=[223,36,1],scene_kinematic_size=7,scene_arena_scale=5.,relation_reward_mode=env['relationReward']['mode'],relation_graph_spec=env['relationGraph'],device='cpu')
model=torch.nn.Module();model.a2c_network=net
checkpoint=torch.load(root/'checkpoints/Stage1V2Fixed_00073000.pth',map_location='cpu',weights_only=False)
transfer(model,checkpoint);model.requires_grad_(True)
obs=observations(all_pairs());objects=obs[:,446:566].reshape(-1,4,30);features=torch.randn(len(obs),4,6);features[:,:,0]=1
obs=torch.cat([obs[:,:446],torch.cat([objects,features],-1).flatten(1),obs[:,566:]],-1)
optimizer=torch.optim.Adam(model.parameters(),lr=2e-5)
initial={n:p.detach().clone() for n,p in model.named_parameters()}
mu,sigma=net.eval_actor(obs);value=net.eval_critic(obs)
loss=mu.square().mean()+value.square().mean()+sigma.square().mean()+net.eval_disc(torch.randn(len(obs),1320)).square().mean()
loss.backward();optimizer.step()
for prefix in ['actor_encoder.tokenizers.1.chair','critic_encoder.tokenizers.1.chair','actor_encoder.layers','critic_encoder.layers','action_head','value_head','disc']:
 names=[n for n,p in model.named_parameters() if prefix in n]
 changed=[n for n in names if not torch.equal(initial[n],dict(model.named_parameters())[n])]
 print(prefix,'parameters',len(names),'updated',len(changed))
 assert changed,prefix
assert all(p.requires_grad for p in model.parameters())
print('ALL TRAINABLE',len(initial),'TRANSFER AND UPDATE OK')
