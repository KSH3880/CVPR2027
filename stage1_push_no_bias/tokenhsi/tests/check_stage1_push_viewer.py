"""Restore a Stage1 PUSH viewer at arbitrary entity counts and check live resets."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # Isaac Gym must precede torch.
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.task_role_spec import task_graph_packet
from utils.jointpush_scene import paths_clear, goal_angles_valid


@torch.no_grad()
def check(player):
    task=player.env.task
    before={key:value.clone() for key,value in player.model.state_dict().items()}
    obs=player.env_reset()
    player.get_batch_size(obs['obs'],1)
    ids=torch.arange(task.num_envs,device=task.device)
    templates=task._templates(ids)
    assert ((templates==4).sum(-1)<=task.num_objects-task.num_agents).all()
    packet=task_graph_packet(task.relation_runtime.graph,task.num_envs).reshape(task.num_envs,task.num_agents,5).long()
    for row in packet:
        supports=row[row[:,1]==3,4]
        assert len(supports.unique())==len(supports)
        assert ((supports>=2*task.num_agents)&(supports<task.num_agents+task.num_objects)).all()
    push=templates==5
    assert paths_clear(task._box_states,task._box_size,task._tar_pos,
                       task._humanoid_root_states[...,:3],push,task._joint['scene']).all()
    assert goal_angles_valid(task._box_states,task._tar_pos,push,task._joint['scene']).all()
    if task.num_envs>1:
        untouched=task._root_states.view(task.num_envs,-1,13)[1:].clone()
        obs=player.env_reset(ids[:1])
        assert torch.equal(untouched,task._root_states.view(task.num_envs,-1,13)[1:])
    done_ids=[]
    for _ in range(32):
        obs=player.env_reset(done_ids)
        assert torch.isfinite(obs['obs']).all()
        action=player.get_action(obs,True)
        assert torch.isfinite(action).all()
        obs,reward,done,_=player.env_step(player.env,action)
        assert torch.isfinite(reward).all() and torch.isfinite(task._root_states).all()
        done_ids=done.nonzero().flatten()[::task.num_agents]
    assert all(torch.equal(value,player.model.state_dict()[key]) for key,value in before.items())
    result=dict(agents=task.num_agents,objects=task.num_objects,envs=task.num_envs,
                preset=task._task_graph_preset,steps=32,finite=True,partial_reset=True,
                weights_unchanged=True,initial_tasks=templates.cpu().tolist())
    path=Path(task.cfg['args'].output_path)/'viewer_validation.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS viewer counts',json.dumps(result),flush=True)


if __name__=='__main__':
    MAPlayerContinuous.run=check
    MAPlayerContinuous.run_eval=check
    entry.main()
