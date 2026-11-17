from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent.bridge_carry_rsi_cache import object_state
from utils import torch_utils
from utils.evaluation_protocol import spatial_success
from utils.task_role_spec import task_graph_packet


def transform(states, rotation, reference, position):
    result = states.clone()
    result[:, :3] = quat_rotate(rotation, states[:, :3]-reference)+position
    result[:, 3:7] = quat_mul(rotation, states[:, 3:7])
    for start in (7,10):
        result[:, start:start+3] = quat_rotate(rotation, states[:, start:start+3])
    return result


def candidate(task, env, people, mids, times, kind):
    lib = task._motion_lib['carryWith']
    root,q,dof,vel,ang,dof_vel,_ = lib.get_motion_state(mids,times)
    bp,bq,bv,ba = lib.get_motion_state_max(mids,times)
    body = torch.cat((bp,bq,bv,ba),-1)
    box = object_state(lib,mids,times)
    source = task._start_surface[env,people]-1
    center = task._surface_center[env,1,source,0]
    landing = task._bridge_landing[env]
    sign = torch.sign(center-landing[:,0])
    size = task._box_size[env,task._bridge_slot[env]]
    distance = .35+.3*torch.rand(len(env),device=task.device)
    local = landing.clone()
    normal = sign*(task._bridge_gap[env]/2+distance)
    on = sign*(size[:,0]/2-distance)
    local[:,0] += torch.where(kind == 1,normal,torch.where(kind == 2,on,-on))
    position = root.clone()
    position[:,:2] = scene.to_world(local,task._structure_yaw[env,1],task._structure_shift[env,1],task._env_origins[env])
    position[:,2] += task._env_origins[env,2]+task._surface_height[env,1]
    position[:,2] -= (kind > 1)*task._scene_cfg['bridge']['top_drop']
    lowest = sensor.lowest_foot_height(body,task._foot_body_ids,task._foot_vertices)
    position[:,2] += (task._scene_cfg['foot_clearance']-lowest).clamp_min(0.)
    yaw = task._structure_yaw[env,1]+torch.where(sign < 0,0.,torch.pi)
    axis = torch.zeros_like(root);axis[:,2] = 1.
    rotation = quat_from_angle_axis(yaw-torch_utils.calc_heading(q),axis)
    human = transform(torch.cat((root,q,vel,ang),-1),rotation,root,position)
    box = transform(box,rotation,root,position)
    repeated = rotation[:,None].expand(-1,task.num_bodies,-1).reshape(-1,4)
    moved = transform(body.flatten(0,1),repeated,
        root[:,None].expand(-1,task.num_bodies,-1).reshape(-1,3),
        position[:,None].expand(-1,task.num_bodies,-1).reshape(-1,3)).view_as(body)
    progress = torch.where(kind == 2,distance/size[:,0],torch.where(kind == 3,1-distance/size[:,0],0.))
    return human,moved,box,dof,dof_vel,progress


def valid(task, env, people, human, body, box):
    slot = task._agent_box_assignment[env,people]
    size = task._box_size[env,slot]
    good = task._boundary_supported(env,body)
    others = task._humanoid_root_states[env,:,:2]
    separation = (human[:,None,:2]-others).norm(dim=-1)
    separation[torch.arange(len(env),device=task.device),people] = torch.inf
    good &= (separation >= task._bridge_curriculum['person_clearance']).all(-1)
    active = task._active_physical(env)
    row = torch.arange(len(env),device=task.device)
    active[row,slot] = False;active[row,task._bridge_slot[env]] = False
    boxes = task._box_states[env]
    radius = task._box_size[env].norm(dim=-1)/2
    good &= ~(((human[:,None,:2]-boxes[...,:2]).norm(dim=-1) < radius+.5) & active).any(-1)
    good &= ~(((box[:,None,:3]-boxes[...,:3]).norm(dim=-1) < radius+size.norm(dim=-1)[:,None]/2+.05) & active).any(-1)
    box_separation = (box[:,None,:2]-others).norm(dim=-1)
    box_separation[row,people] = torch.inf
    good &= (box_separation >= size[:,:2].norm(dim=-1)[:,None]/2+.5).all(-1)
    corners = box.new_tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    points = quat_rotate(box[:,None,3:7].expand(-1,8,-1).reshape(-1,4),
        (corners[None]*size[:,None]/2).reshape(-1,3)).view(-1,8,3)+box[:,None,:3]
    good &= (points[...,2]-task._terrain_height(points[...,:2],env) >= -.005).all(-1)
    target = task._tar_pos[env,people]
    support = task._box_states[env,3*task.num_agents+people]
    target = torch.where((task._templates[env,people] == 4)[:,None],support[:,:3],target)
    good &= (human[:,:2]-target[:,:2]).norm(dim=-1) <= 9.9
    good &= (human[:,:2]-box[:,:2]).norm(dim=-1) <= 9.9
    packet = task_graph_packet(task.relation_runtime.graph,task.num_envs).reshape(task.num_envs,task.num_agents,5)[env]
    roots = task._humanoid_root_states[env,:,:3].clone();roots[row,people] = human[:,:3]
    logical = task._gather_entity_slots(task._box_states[env],task._logical_box_order[env]).clone()
    matches = task._logical_box_order[env] == slot[:,None]
    logical = torch.where(matches[...,None],box[:,None],logical)
    sizes = task._gather_entity_slots(task._box_size[env],task._logical_box_order[env])
    success = spatial_success(packet,roots,logical,sizes,task._tar_pos[env])
    good &= ~success[row,people].any(-1)
    return good & torch.isfinite(human).all(-1) & torch.isfinite(box).all(-1)
