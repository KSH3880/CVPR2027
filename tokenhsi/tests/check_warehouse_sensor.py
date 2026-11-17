import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.scene_validation import scene_infeasible
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from utils.before_spec import task_objects
from utils.sensor_stage2_spec import WAREHOUSE


def local(task, ids, xy):
    shape = (len(ids),)+(1,)*(xy.ndim-2)
    value = scene.rotate_xy(xy-task._env_origins[ids,:2].view(*shape,2)
        -task._structure_shift[ids,WAREHOUSE].view(*shape,2),-task._structure_yaw[ids,WAREHOUSE].view(shape))
    value[...,0] += task._surface_half[ids,WAREHOUSE,0,0].view(shape)
    return value


def inspect(task):
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    if not len(ids):
        return 0
    cfg = task._warehouse
    n,m = len(ids),task.num_agents
    rows = torch.arange(n,device=task.device)
    tasks,pre = task._templates[ids],task._before_predecessor[ids]
    pusher = task._roles[ids,0]
    assert ((tasks == 5).sum(-1) == 1).all()
    assert (tasks[rows,pusher] == 5).all() and (pre[rows,pusher] < 0).all()
    assert ((pre >= 0).sum(-1) <= 1).all()
    assert not task._unreachable[ids].any()
    assert (task._start_surface[ids] == 0).all()
    roots = local(task,ids,task._humanoid_root_states[ids,:,:2])
    assert (roots[...,0] < -cfg['wall_thickness']-.35).all()
    primary = task._agent_box_assignment[ids]
    boxes = task._box_states[ids[:,None],primary]
    sizes = task._box_size[ids[:,None],primary]
    xy = local(task,ids,boxes[...,:2])
    used = task_objects(pre,tasks)[:,:m] & (tasks != 5)
    half = task._surface_half[ids,WAREHOUSE,0]
    assert (xy[...,0][used] > sizes[...,0][used]/2).all()
    assert (xy[...,0][used]+sizes[...,0][used]/2 < 2*half[:,0,None].expand(-1,m)[used]).all()
    assert (xy[...,1].abs()[used]+sizes[...,1][used]/2 < half[:,1,None].expand(-1,m)[used]).all()
    assert (used.sum(-1) >= 2).all()
    assert (task._agent_box_assignment[ids,pusher] == task._bridge_slot[ids]).all()
    assert task._logical_box_order[ids].eq(task._bridge_slot[ids,None]).any(-1).all()
    obstacle = xy[rows,pusher]
    block = sizes[rows,pusher]
    expected = -cfg['wall_thickness']-cfg['wall_box_gap']-block[:,0]/2
    assert torch.allclose(obstacle[:,0],expected,atol=.002)
    mode = task._warehouse_start_mode[ids]
    if task._warehouse_start:
        required = block[:,1]-cfg['opening_overlap']+cfg['open_clearance']
        progress = obstacle[:,1].abs()/required
        assert (obstacle[:,1].abs()[mode == 0] < .002).all()
        low,high = task._warehouse_start['partial_open_progress']
        assert ((progress[mode == 1] >= low-.002) & (progress[mode == 1] <= high+.002)).all()
        torch.testing.assert_close(obstacle[mode == 2,1],local(task,ids,task._warehouse_landing[ids,:2])[mode == 2,1],atol=.002,rtol=0)
    else:
        assert obstacle[:,1].abs().max() < .002
    assert torch.allclose((roots[rows,pusher]-obstacle).norm(dim=-1),torch.full_like(expected,cfg['pusher_stand_off']),atol=.002)
    landing = local(task,ids,task._warehouse_landing[ids,:2])
    assert torch.allclose(landing[:,0],obstacle[:,0],atol=.002)
    assert (landing[:,1].abs() >= block[:,1]-cfg['opening_overlap']+cfg['open_clearance']-.002).all()
    assert not scene_infeasible(task,ids).any()
    assert not invalid_before_layout(task,ids).any()
    return n


def check_motion(task, env):
    from isaacgym import gymtorch
    env.reset()
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    active = task._active_physical(ids)
    initial = task._box_states[ids,:,:3].clone()
    action = torch.zeros(task.num_envs*task.num_agents,task.get_action_size(),device=task.device)
    for step in range(6):
        obs, reward, _, _ = env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        if step == 0:
            first = (task._box_states[ids,:,:3]-initial).norm(dim=-1)
            first_position = task._box_states[ids,:,:3].clone()
    moved = (task._box_states[ids,:,:3]-initial).norm(dim=-1)
    ranked = first.masked_fill(~active,-1).flatten().argsort(descending=True)[:5]
    examples = []
    for index in ranked.tolist():
        row,slot = divmod(index,task.num_physical_objects)
        e = int(ids[row])
        examples.append(dict(env=e,slot=slot,size=task._box_size[e,slot].tolist(),
            before=initial[row,slot].tolist(),after=first_position[row,slot].tolist()))
    env.reset()
    ids = (task._scene_ids == WAREHOUSE).nonzero().flatten()
    closed = task._sensor_observation(ids).clone()
    task._box_states[ids,task._bridge_slot[ids],:3] = task._warehouse_landing[ids]
    task._box_states[ids,task._bridge_slot[ids],7:] = 0
    task.gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
    task._refresh_sim_tensors()
    opened = task._sensor_observation(ids)
    assert (closed-opened).abs().max() > 0
    task._record_scene_outcomes(torch.zeros_like(task._templates,dtype=torch.bool),
        torch.zeros(task.num_envs,device=task.device,dtype=torch.bool))
    assert task._warehouse_opened[ids].all()
    return dict(first_step_box_max=float(first[active].max()),six_step_box_max=float(moved[active].max()),
        sensor_opening_max_change=float((closed-opened).abs().max()),opening_recorded=True,first_step_examples=examples)
