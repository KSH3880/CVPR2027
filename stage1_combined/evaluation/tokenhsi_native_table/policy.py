"""Frozen scene-policy observation bridge for the unmodified native carry task."""
from pathlib import Path
import torch
from isaacgym.torch_utils import quat_rotate
from env.tasks.humanoid import compute_humanoid_observations_max
from env.tasks.multi_agent.scene_features import build_gta_pose_records
from learning.multi_agent.table_adaptation_network import TableAdaptationBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.edge_scenario_spec import compose_canonical_graph
from utils.task_role_spec import task_graph_packet
from utils import torch_utils


def pack_scene(body_pos, body_rot, body_vel, body_ang_vel, box, bps, target, origins):
    n=body_pos.shape[0]
    human=compute_humanoid_observations_max(body_pos,body_rot,body_vel,body_ang_vel,True,True)
    inverse=box[:,3:7].clone();inverse[:,:3]*=-1
    objects=torch.cat((quat_rotate(inverse,box[:,7:10]),quat_rotate(inverse,box[:,10:13]),bps.reshape(n,24)),-1)
    heading=torch_utils.calc_heading_quat(body_rot[:,0])
    pose=build_gta_pose_records(body_pos[:,None,0],heading[:,None],box[:,None,:3],box[:,None,3:7],target[:,None],origins,goal_rotation='identity')
    graph=compose_canonical_graph(torch.full((n,1),3,dtype=torch.long,device=body_pos.device),num_objects=1)
    packet=task_graph_packet(graph,n)
    obs=torch.cat((human,objects,torch.ones(n,1,device=body_pos.device),pose.flatten(1),packet),-1)
    assert obs.shape==(n,280) and torch.isfinite(obs).all()
    return obs


class FrozenScenePolicy:
    def __init__(self,checkpoint,device):
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        assert saved['epoch']==10000
        config=saved['relation_experiment_config']
        b=TableAdaptationBuilder();b.load(config['train']['params']['network'])
        kw=dict(actions_num=32,input_shape=(280,),amp_input_shape=(1320,),value_size=1,
            num_agents=1,num_objects=1,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,
            observation_mode='clean_scene',scene_entity_sizes=[223,30,1],scene_kinematic_size=7,
            scene_arena_scale=5.,relation_reward_mode=config['env']['relationReward']['mode'],
            relation_graph_spec=config['env']['relationGraph'],device=device)
        self.net=b.build('amp',**kw).to(device)
        state={k[len('a2c_network.'):]:v for k,v in saved['model'].items()}
        self.net.load_state_dict(state,strict=True);self.net.requires_grad_(False);self.net.eval()
        self.rms=SceneRunningMeanStd([223,30,1],[1,1,1],[223,30,0],7,5).to(device).eval()
        self.rms.load_state_dict(saved['running_mean_std'],strict=True)
        assert all(torch.equal(v,self.net.state_dict()[k]) for k,v in state.items())
        assert all(torch.equal(v,self.rms.state_dict()[k]) for k,v in saved['running_mean_std'].items())
        self._model={k:v.clone() for k,v in self.net.state_dict().items()}
        self._rms={k:v.clone() for k,v in self.rms.state_dict().items()}

    @torch.no_grad()
    def action(self,obs):
        mu,_=self.net.eval_actor(self.rms(obs))
        assert torch.isfinite(mu).all()
        return mu.clamp(-1,1)

    def check_frozen(self):
        assert all(torch.equal(v,self.net.state_dict()[k]) for k,v in self._model.items())
        assert all(torch.equal(v,self.rms.state_dict()[k]) for k,v in self._rms.items())
