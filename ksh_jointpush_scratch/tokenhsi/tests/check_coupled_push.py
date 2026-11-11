"""Live coupled geometry, common policy, partial reset and AMP checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from torch import nn
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
from utils.coupled_push_spec import check_checkpoint,group_sizes,physical_box_count
from utils.coupled_push_scene import scene_infeasible
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.human_coordination import stage2_human_relations
from utils.task_role_spec import task_graph_packet


def inspect(task):
    n,m=task.num_envs,task.num_agents;ids=torch.arange(n,device=task.device)
    tolerance = max(3e-5,4*torch.finfo(task._box_states.dtype).eps*float(task._env_origins.abs().max()))
    groups=task._groups;members=group_sizes(groups);push=task._templates_by_owner==5
    assert (task._templates_by_owner[members>1]==5).all()
    assert torch.equal(members,task._members)
    mapping=task._agent_box_assignment
    whole=task._assigned_box_values(task._box_states)
    sizes=task._assigned_box_values(task._box_size)
    tokens=task._logical_box_values(task._box_states)[:,:m]
    cells=task._logical_box_values(task._box_size)[:,:m]
    torch.testing.assert_close(cells[...,0]*members,sizes[...,0])
    same=groups[:,:,None]==groups[:,None]
    assert (mapping[:,:,None]==mapping[:,None])[same].all()
    individual = task._individual_push_goals
    goals = task._center_tar_pos if individual else task._tar_pos
    assert (goals[:,:,None]==goals[:,None])[same].all()
    if individual:
        torch.testing.assert_close((task._tar_pos-tokens[..., :3])[push], (goals-whole[..., :3])[push], atol=tolerance, rtol=0)
    shared=members>1
    delta=goals[...,:2]-whole[...,:2]
    q=whole[...,3:7];forward=torch.stack((-2*q[...,2]*q[...,3],1-2*q[...,2].square()),-1)
    cosine=(delta*forward).sum(-1)/delta.norm(dim=-1).clamp_min(1e-8)
    assert (cosine[shared]>1-1e-5).all()
    assert (cosine[push&~shared]>=2**-.5-1e-5).all()
    distance=(whole[...,:2]-task._humanoid_root_states[...,:2]).norm(dim=-1)
    spec=task.cfg['env']['stage2JointPush']
    solo=dict(task.cfg['env']['jointPush']['scene'])
    for key in ('source', 'goal'):
        solo[key+'_distance'] = [solo[key+'_distance'][0],
            spec.get('solo_'+key+'_max_distance', solo[key+'_distance'][1])]
    for mask,config in ((shared,spec),(push&~shared,solo)):
        for values,key in ((distance,'source_distance'),(delta.norm(dim=-1),'goal_distance')):
            low,high=config.get(key,[2.,10.])
            assert ((values[mask]>=low-tolerance)&(values[mask]<=high+tolerance)).all()
    solo_distances = {}
    for kind, name in ((1, 'SIT'), (2, 'CLIMB'), (3, 'AT'), (4, 'ON_TOP')):
        mask = task._templates_by_owner == kind
        high = min(5. if kind == 1 else 10., spec.get('solo_source_max_distance', 10.))
        assert ((distance[mask] >= 1.-tolerance) & (distance[mask] <= high+tolerance)).all()
        solo_distances[name] = dict(count=int(mask.sum()),
            min=float(distance[mask].min()) if mask.any() else None,
            max=float(distance[mask].max()) if mask.any() else None)
    at = task._templates_by_owner == 3
    expected_z = (sizes[..., 2]/2 if task._ground_only else
                  task._tar_platform_pos[..., 2]+task._platform_height/2+sizes[..., 2]/2)
    torch.testing.assert_close(task._tar_pos[..., 2][at], expected_z[at], atol=1e-5, rtol=0)
    if task._ground_only:
        carry = at | (task._templates_by_owner == 4)
        torch.testing.assert_close(whole[..., 2][carry], sizes[..., 2][carry]/2, atol=tolerance, rtol=0)
        assert (task._platform_pos[ids, :, 2] == -20.).all()
        assert (task._tar_platform_pos[ids, :, 2] == -19.5).all()
    carry_distances = {}
    if 'solo_carry_goal_distance' in spec:
        packet = task_graph_packet(task.relation_runtime.graph,n).reshape(n,m,5).long()
        rows = torch.arange(n,device=task.device)[:,None]
        logical = task._logical_box_values(task._box_states)
        support = logical[rows,(packet[...,4]-m).clamp(0,task.num_objects-1),:2]
        for kind,name in ((3,'AT'),(4,'ON_TOP')):
            mask = task._templates_by_owner == kind
            target = task._tar_pos[...,:2] if kind == 3 else support
            travel = (target-whole[...,:2]).norm(dim=-1)[mask]
            low,high = spec['solo_carry_goal_distance']
            assert ((travel >= low-tolerance) & (travel <= high+tolerance)).all()
            gap = (target-task._humanoid_root_states[...,:2]).norm(dim=-1)[mask]
            assert (gap >= 1.-tolerance).all()
            carry_distances[name] = dict(count=int(mask.sum()),
                min=float(travel.min()) if len(travel) else None,
                max=float(travel.max()) if len(travel) else None,
                mean=float(travel.mean()) if len(travel) else None)
    else:
        assert ((task._tar_pos[..., :2]-task._env_origins[:, None, :2])[at].abs() <= 4.5+tolerance).all()
    assert not scene_infeasible(task,ids).any()
    end = physical_box_count(m)
    expected=task._box_size[:,4*m:end].prod(-1)*3
    torch.testing.assert_close(task._box_density.reshape(n,task.num_physical_objects)[:,4*m:end]*task._box_size[:,4*m:end].prod(-1),expected)
    assert task.get_obs_size()==238*m+37*task.num_objects+task.get_relation_suffix_size() and task.get_num_amp_obs()==1330
    obs=task._compute_clean_scene_obs()
    assert torch.equal(obs[:,-m:].long(),groups)
    if task._coupled_peer_mode != 'all':
        peers = obs[:, -2*m:-m].long()
        assert torch.equal(peers, task._single_peer_selector.peers)
        assert ((peers >= -1) & (peers < m)).all()
    packet=task_graph_packet(task.relation_runtime.graph,n)
    relation=stage2_human_relations(packet,groups)
    off=~torch.eye(m,device=task.device,dtype=torch.bool)[None]
    assert torch.equal(relation==3,same&shared[:,:,None]&off)
    # Scoring must use the whole box center, not a shifted participant cell.
    scored=task._evaluation_box_values(task._box_states)[:,:m]
    torch.testing.assert_close(scored[push],(tokens if individual else whole)[push])
    if individual:
        record=packet.reshape(n,m,5)
        expected=torch.arange(m,device=task.device)[None].expand(n,-1)+m+task.num_objects
        assert torch.equal(record[...,4][push],expected[push])
    family=task._amp_family().reshape(n,m)
    assert (family[push]==3).all()
    return dict(push_people=int(push.sum()),coupled_people=int(shared.sum()),
        group_sizes=sorted(members.unique().tolist()),straight_goals=True,physical_bank=task.num_physical_objects,
        virtual_cells=True,common_obs=task.get_obs_size(),amp=1330,whole_box_eval=not individual,
        float32_geometry_tolerance=tolerance,solo_source_distances=solo_distances,solo_carry_goal_distances=carry_distances,at_platform_height_preserved=True,
        elevated_at_goals=int((at & (task._tar_pos[..., 2] > sizes[..., 2]/2+.01)).sum()))


def main():
    torch.set_num_threads(1);args=get_args();args.test=args.eval=True
    cfg,train,_=load_cfg(args);cfg['env']['motion_file']=args.motion_file;set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    builder=Stage2PushBuilder();builder.load(train['params']['network'])
    net=builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),amp_input_shape=(task.get_num_amp_obs(),),
        value_size=1,num_agents=task.num_agents,num_objects=task.num_objects,observation_mode='clean_scene',scene_entity_sizes=task.get_scene_entity_sizes(),
        scene_kinematic_size=7,scene_arena_scale=task.get_scene_arena_scale(),relation_reward_mode=task._relation_cfg['mode'],relation_graph_spec=task._relation_graph_spec,device=task.device).to(task.device)
    model=nn.Module();model.a2c_network=net
    ckpt=torch.load(args.checkpoint,map_location=task.device,weights_only=False);check_checkpoint(ckpt,cfg['env'],inference=True)
    model.load_state_dict(ckpt['model'],strict=True);model.eval()
    rms=SceneRunningMeanStd([223,30,1],[task.num_agents,task.num_objects,task.num_agents],normalized_sizes=[223,30,0],kinematic_size=7,extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(ckpt['running_mean_std'],strict=True);rms.eval()
    weights={k:v.clone() for k,v in model.state_dict().items()};cases=[]
    with torch.no_grad():
        presets = (args.task_graph,) if (task.num_agents,task.num_objects) != (4,8) else ('random_scenario','all_coupled','all_solo','push','holding_at','holding_ontop','sit','climb')
        for preset in presets:
            task._task_graph_preset=preset;obs=env.reset();case=inspect(task)
            assert torch.isfinite(task.fetch_amp_obs_demo(512)).all()
            fixed_goals=task._tar_pos.clone()
            for _ in range(16):
                mu,_=net.eval_actor(rms(obs));value=net.eval_critic(rms(obs));obs,reward,done,info=env.step(mu)
                assert all(torch.isfinite(v).all() for v in (obs,reward,mu,value))
            assert torch.equal(fixed_goals,task._tar_pos)
            untouched=task._root_states.view(task.num_envs,-1,13)[1:].clone()
            saved_groups=task._groups[1:].clone()
            saved_goals=task._tar_pos[1:].clone()
            saved_centers=task._center_tar_pos[1:].clone() if task._individual_push_goals else None
            saved_peers=task._single_peer_selector.peers[1:].clone() if task._coupled_peer_mode != 'all' else None
            task._reset_envs(torch.tensor([0],device=task.device))
            assert torch.equal(untouched,task._root_states.view(task.num_envs,-1,13)[1:])
            assert torch.equal(saved_groups,task._groups[1:])
            assert torch.equal(saved_goals,task._tar_pos[1:])
            if saved_centers is not None:
                assert torch.equal(saved_centers,task._center_tar_pos[1:])
            if saved_peers is not None:
                assert torch.equal(saved_peers,task._single_peer_selector.peers[1:])
            cases.append(dict(preset=preset,finite=True,partial_reset=True,**case))
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in weights.items())
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    (out/'physics_validation.json').write_text(json.dumps(dict(weights_unchanged=True,cases=cases),indent=2))
    print('COUPLED PUSH CHECK COMPLETE',out,flush=True)

if __name__=='__main__':main()
