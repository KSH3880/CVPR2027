"""Live checks of the square-sensor scenes: resets, placement, sensor values and physical stability."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import numpy as np
import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent import square_sensor_scene as scene
from utils.before_goal_sampling import route_distances
from utils.before_spec import task_objects
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.parse_task import parse_task
from utils.sensor_stage2_spec import BRIDGE, CLIFF, FLOOR, ROOM, SCENES, paired_people

EPS = 2e-3  # float32 root coordinates reach 4km with 1024 environments, about 0.5mm per value.


def local_xy(task, xy, ids, current):
    shape = (len(ids),)+(1,)*(xy.ndim-2)
    offset = xy-task._env_origins[ids, :2].view(*shape, 2)-task._structure_shift[ids, current].view(*shape, 2)
    return scene.rotate_xy(offset, -task._structure_yaw[ids, current].view(shape))


def inspect(task, stats):
    n, m = task.num_envs, task.num_agents
    ids = torch.arange(n, device=task.device)
    rows = ids[:, None]
    current = task._scene_ids
    people = torch.arange(m, device=task.device)[None].expand(n, -1)
    assert torch.equal(task._env_origins, task._scene_origins[ids, current])
    roots = task._humanoid_root_states[..., :3]
    templates, predecessor = task._templates, task._before_predecessor
    paired = paired_people(predecessor)
    local = local_xy(task, roots[..., :2], ids, current)
    height = task._surface_height[ids, current]
    origin_z = task._env_origins[ids, 2]

    for surface in (1, 2):
        mask = task._start_surface == surface
        center = task._surface_center[ids, current, surface-1]
        half = task._surface_half[ids, current, surface-1]
        inside = ((local-center[:, None]).abs() <= half[:, None]-task._scene_cfg['human_margin']+EPS).all(-1)
        bad = (mask & ~inside).nonzero().tolist()
        assert not bad, 'people must start inside their surface: {}'.format(
            [(e, a, SCENES[current[e]], int(templates[e, a]), int(predecessor[e, a]), local[e, a].tolist(), half[e].tolist(),
              bool(task._bridge_aligned[e]), int((task._roles[e] == a).nonzero())) for e, a in bad[:6]])
    raised = task._start_surface > FLOOR
    feet = sensor.lowest_foot_height(task._kinematic_humanoid_rigid_body_states, task._foot_body_ids, task._foot_vertices)
    support = (origin_z+height)[:, None]
    assert (feet[raised] >= (support.expand_as(feet))[raised]+task._scene_cfg['foot_clearance']-EPS).all()
    assert (roots[..., 2][raised] > (support.expand_as(feet))[raised]+.2).all()
    # Floor people keep the plain Stage2 height.
    assert (roots[..., 2][~raised] < 1.5).all()

    source, support_box = task._agent_box_assignment, 3*m+people
    valid = task_objects(predecessor, templates)
    boxes, sizes = task._box_states, task._box_size
    index = rows.expand(-1, m)
    on_surface = (task._surface > FLOOR) & ~task._unreachable
    for box, used in ((source, valid[:, :m]), (support_box, valid[:, m:])):
        z, top = boxes[index, box, 2], sizes[index, box, 2]/2
        lifted = used & on_surface
        assert ((z-(support+top)).abs()[lifted] < EPS).all(), 'boxes rest on their surface'
        grounded = used & ~on_surface & (box != task._bridge_slot[:,None]) & ~((templates == 5) & (current == BRIDGE)[:, None])
        assert ((z-(origin_z[:, None]+top)).abs()[grounded] < EPS).all(), 'other boxes rest on the floor'

    # Stage2 distance contract survives the rigid move.
    r, follower = (predecessor >= 0).nonzero(as_tuple=True)
    leader = predecessor[r, follower]
    if len(r):
        box_xy, goal_xy = boxes[r, source[r, leader], :2], task._tar_pos[r, leader, :2]
        payload_xy = boxes[r, source[r, follower], :2]
        first, second = route_distances(roots[r, leader, :2], box_xy, goal_xy, roots[r, follower, :2],
                                        payload_xy, templates[r, follower])
        assert ((roots[r, leader, :2]-box_xy).norm(dim=-1) < 5+EPS).all()
        assert ((box_xy-goal_xy).norm(dim=-1) < 5+EPS).all()
        assert ((box_xy-goal_xy).norm(dim=-1) > 1-EPS).all()
        near = task._before_near_pairs[r, follower]
        assert (second[near] < first[near]+EPS).all(), 'near pairs must keep the goal-route order'
        assert (((roots[r, follower, :2]-goal_xy).norm(dim=-1)) >= 1-EPS).all()
        assert (task._surface[r, follower] == task._surface[r, leader]).all()
        assert (templates[r, leader] == 3).all()
        assert (task._scene_ids[r] != CLIFF).any() or len(r) > 0
        assert (~task._unreachable[r, follower]).all() and (~task._unreachable[r, leader]).all()

    # Cliff people have their objects on the floor beyond the edge; no surface person sits on a wall.
    cliff = (current == CLIFF).nonzero().flatten()
    if len(cliff):
        un = task._unreachable[cliff]
        assert torch.equal(un, ~paired[cliff] & (templates[cliff] != 5))
        box_xy = local_xy(task, boxes[cliff[:, None].expand(-1, m), source[cliff], :2], cliff, current[cliff])
        half = task._surface_half[cliff, CLIFF, 0, None]
        outside = (box_xy.abs()-half).amax(-1)
        assert (outside[un] > 0).all(), 'cliff objects must stay off stage'
        distance = (boxes[cliff[:, None].expand(-1, m), source[cliff], :2]-roots[cliff, :, :2]).norm(dim=-1)
        assert ((distance[un] >= 1-EPS) & (distance[un] <= 5+EPS)).all()

    room = (current == ROOM).nonzero().flatten()
    if len(room):
        half_y = (task._room_half[room, 1]-task._scene_cfg['room']['passage_width'])[:, None]
        reach = task._scene_cfg['room']['wall_thickness']/2+scene.HUMAN_RADIUS
        inner = task._start_surface[room] > FLOOR
        people_local = local[room]
        blocked = ((people_local[..., 0].abs() < reach-EPS)
                   & (people_local[..., 1].abs() < half_y+scene.HUMAN_RADIUS-EPS))
        assert not (blocked & inner).any(), 'people start {:.4f}m inside the partition reach'.format(
            (reach-people_local[..., 0].abs())[blocked & inner].max().item() if (blocked & inner).any() else 0.)
        assert ((templates[room] != 5) | (task._start_surface[room] == FLOOR)).all()

    bridge = (current == BRIDGE).nonzero().flatten()
    if len(bridge):
        k = torch.arange(len(bridge), device=task.device)
        pusher, crosser = task._roles[bridge, 0], task._roles[bridge, 1]
        assert (templates[bridge, pusher] == 5).all()
        assert (task._start_surface[bridge, pusher] == FLOOR).all()
        assert ((task._surface[bridge, crosser]+task._start_surface[bridge, crosser]) == 3).all()
        assert (task._start_surface[bridge, crosser] > FLOOR).all()
        assert (predecessor[bridge, crosser] < 0).all() and (predecessor[bridge, pusher] < 0).all()
        assert not task.relation_runtime.graph.prereq_mask[bridge][
            (task.relation_runtime.graph.edge_owner[bridge] == crosser[:, None])].any()
        box = boxes[bridge, task._bridge_slot[bridge]]
        assert torch.equal(task._logical_box_order[bridge].eq(task._bridge_slot[bridge,None]).any(-1), torch.ones_like(k, dtype=torch.bool))
        assert (task._scenario_physical_object(bridge, pusher) == task._bridge_slot[bridge]).all(), 'pusher must own the bridge box'
        assert (task._agent_box_assignment[bridge, pusher] == task._bridge_slot[bridge]).all()
        assert ((box[:, 2]-sizes[bridge, task._bridge_slot[bridge], 2]/2).abs() < EPS).all()
        assert ((task._center_tar_pos[bridge, pusher, :2]-task._bridge_landing_world[bridge, :2]).norm(dim=-1) < EPS).all()
        stand_off = scene.rotate_xy(roots[bridge,pusher,:2]-box[:,:2],-task._structure_yaw[bridge,BRIDGE])[:,1].abs()
        assert ((stand_off-task._scene_cfg['bridge']['pusher_stand_off']).abs() < EPS).all()
        local_box = local_xy(task, boxes[bridge, source[bridge, crosser], :2], bridge, current[bridge])
        carry_from_start = task._bridge_carry_from_start[bridge]
        carry = (templates[bridge, crosser] == 3) | (templates[bridge, crosser] == 4)
        assert not (carry_from_start & ~carry).any()
        dest = task._surface[bridge, crosser]-1
        payload_stage = torch.where(carry_from_start, task._start_surface[bridge, crosser]-1, dest)
        center = task._surface_center[bridge, BRIDGE, payload_stage]
        half = task._surface_half[bridge, BRIDGE, payload_stage]
        assert ((local_box-center).abs() <= half).all(), 'crosser object must lie on its sampled pickup stage'
        for template, name in ((3, 'at'), (4, 'ontop')):
            chosen = templates[bridge, crosser] == template
            counts = stats.setdefault('bridge_carry', {}).setdefault(name, dict(total=0, from_start=0))
            counts['total'] += int(chosen.sum())
            counts['from_start'] += int((chosen & carry_from_start).sum())
            goal = task._tar_pos[bridge, crosser, :2] if template == 3 else boxes[bridge, support_box[bridge, crosser], :2]
            target = local_xy(task, goal, bridge, current[bridge])
            center, half = task._surface_center[bridge, BRIDGE, dest], task._surface_half[bridge, BRIDGE, dest]
            assert ((target-center).abs()[chosen] <= half[chosen]).all(), 'carry target must stay on the far stage'
        distance = (roots[bridge, crosser, :2]-boxes[bridge, source[bridge, crosser], :2]).norm(dim=-1)
        payload_radius = sizes[bridge, source[bridge, crosser], :2].norm(dim=-1)/2
        assert (distance[carry_from_start] >= (scene.HUMAN_RADIUS+payload_radius+scene.GROUP_PAD)[carry_from_start]-EPS).all()
    # Floor people, their goals and PUSH paths never enter the structure rectangle.
    radius = torch.full((n, m), scene.HUMAN_RADIUS, device=task.device)
    floor = task._start_surface == FLOOR
    floor &= ~task._bridge_pushers
    bounds = task._structure_bounds[ids, current]

    def inside(xy, disc):
        at = local_xy(task, xy, ids, current)
        pad = disc+task._scene_cfg['floor_margin']-EPS
        return ((at[..., 0] > bounds[:, 0, None]-pad) & (at[..., 0] < bounds[:, 1, None]+pad)
                & (at[..., 1] > bounds[:, 2, None]-pad) & (at[..., 1] < bounds[:, 3, None]+pad))
    assert not (inside(roots[..., :2], radius) & floor).any(), 'floor people entered a structure'
    own = sizes[index, source, :2].norm(dim=-1)/2
    start_xy = boxes[index, source, :2]
    moving = floor & ((templates == 3) | (templates == 5))
    for fraction in (0., .5, 1.):
        point = torch.where((templates == 5)[..., None], start_xy+fraction*(task._center_tar_pos[..., :2]-start_xy), task._tar_pos[..., :2])
        assert not (inside(point, own) & moving).any(), 'floor goals or PUSH paths entered a structure'
    other = (current != BRIDGE).nonzero().flatten()
    if len(other):
        assert ((boxes[other, task._bridge_slot[other], :2]-task._env_origins[other, :2]).norm(dim=-1) > 100).all()

    for index, name in enumerate(SCENES):
        chosen = current == index
        row = stats['scenes'][name]
        row['episodes'] += int(chosen.sum())
        row['before_pairs'] += int((predecessor[chosen] >= 0).sum())
        row['episodes_with_pair'] += int((predecessor[chosen] >= 0).any(-1).sum())
        row['unreachable'] += int(task._unreachable[chosen].sum())


def check_sensor(task, obs, stats):
    """Recompute some sensor rows with plain NumPy from the live states."""
    width, points = task._sensor_obs_width, task._sensor_points
    values = obs[:, -width:].reshape(task.num_envs, task.num_agents, points).cpu().numpy()
    grid = task._sensor_grid.cpu().numpy()
    worst = 0.
    raw_error, boundary_points = 0., 0
    for env in np.linspace(0, task.num_envs-1, 8).astype(int).tolist():
        current = int(task._scene_ids[env])
        boxes = task._box_states[env,task._logical_box_order[env]].cpu().numpy()
        sizes = task._box_size[env,task._logical_box_order[env]].cpu().numpy()
        centers = np.concatenate((task._static_centers[env, current].cpu().numpy(), boxes[:, :3]))
        quats = np.concatenate((task._static_quat[env, current].cpu().numpy(), boxes[:, 3:7]))
        halves = np.concatenate((task._static_half[env, current].cpu().numpy(), sizes/2))
        kin = task._kinematic_humanoid_rigid_body_states[env].cpu().numpy()
        for person in range(task.num_agents):
            root, q = kin[person, 0, :3], kin[person, 0, 3:7]
            heading = math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
            # calc_heading rotates +x by the root quaternion and takes its planar angle.
            x, y, z, w = q
            forward = np.array([1-2*(y*y+z*z), 2*(x*y+z*w)])
            heading = math.atan2(forward[1], forward[0])
            c, s = math.cos(heading), math.sin(heading)
            world = root[:2]+np.stack((c*grid[:, 0]-s*grid[:, 1], s*grid[:, 0]+c*grid[:, 1]), -1)
            floor = float(task._env_origins[env, 2])
            height = np.full(len(grid), floor)
            inner_height, outer_height = height.copy(), height.copy()
            for center, quat, half in zip(centers, quats, halves):
                x, y, z, w = quat
                rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                top = center[2]+(np.abs(rotation[2])*half).sum()
                offset = world-center[:2]
                ca, sa = math.cos(yaw), math.sin(yaw)
                local_x, local_y = ca*offset[:, 0]+sa*offset[:, 1], ca*offset[:, 1]-sa*offset[:, 0]
                inside = (np.abs(local_x) <= half[0]) & (np.abs(local_y) <= half[1])
                height = np.where(inside, np.maximum(height, top), height)
                inner = (np.abs(local_x) < half[0]-EPS) & (np.abs(local_y) < half[1]-EPS)
                outer = (np.abs(local_x) <= half[0]+EPS) & (np.abs(local_y) <= half[1]+EPS)
                inner_height = np.where(inner, np.maximum(inner_height, top), inner_height)
                outer_height = np.where(outer, np.maximum(outer_height, top), outer_height)
            clip = task._scene_cfg['sensor']['clip']
            expected = np.clip(height-root[2], -clip, clip)/clip
            lower = np.clip(inner_height-root[2], -clip, clip)/clip
            upper = np.clip(outer_height-root[2], -clip, clip)/clip
            observed = values[env, person]
            worst = max(worst, float(np.maximum(lower-observed, observed-upper).clip(0).max()))
            raw_error = max(raw_error, float(np.abs(expected-observed).max()))
            boundary_points += int((inner_height != outer_height).sum())
    assert worst < 2e-3, 'sensor is outside the NumPy edge-tolerance interval by {}'.format(worst)
    stats['sensor_numpy_raw_max_error'] = max(stats.get('sensor_numpy_raw_max_error', 0.), raw_error)
    stats['sensor_boundary_points'] = stats.get('sensor_boundary_points', 0)+boundary_points
    stats['sensor_edge_tolerance'] = EPS
    return worst


def main():
    args=get_args()
    torch.set_num_threads(4)
    cfg,train,_=load_cfg(args)
    cfg['env']['motion_file'],cfg['args']=args.motion_file,args
    set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    stats=dict(scenes={name:dict(episodes=0,before_pairs=0,episodes_with_pair=0,unreachable=0) for name in SCENES})
    obs=env.reset()
    records=task._terrain_actor_records
    roots=task._root_states.reshape(-1,13)
    fixed=roots[records[:,-1]].clone()
    counts=torch.zeros(3,device=task.device)
    for iteration in range(4):
        obs=env.reset()
        inspect(task,stats)
        check_sensor(task,obs,stats)
        ids=torch.arange(task.num_envs,device=task.device)
        bridge=(task._scene_ids==BRIDGE).nonzero().flatten()
        k=task._bridge_count[bridge]
        counts+=torch.bincount(k-1,minlength=3)
        assert torch.equal(task._bridge_pushers[bridge].sum(-1),k)
        sizes=task._box_size[bridge,task._bridge_slot[bridge]]
        assert torch.allclose(task._bridge_gap[bridge],sizes[:,0]+.2)
        for person in range(4):
            chosen=task._bridge_pushers[bridge,person]
            assert (task._agent_box_assignment[bridge[chosen],person]==task._bridge_slot[bridge[chosen]]).all()
        e,f=(task._before_predecessor>=0).nonzero(as_tuple=True)
        leader=task._before_predecessor[e,f]
        box=task._box_size[e,task._agent_box_assignment[e,leader]]
        limits=box.new_tensor(task._scene_cfg['before_leader_size'])
        assert ((box>=limits[:,0]-1e-6)&(box<=limits[:,1]+1e-6)).all()
        # Active sensor geometry must agree with the fixed physics actors, including variant yaw.
        e,kind,variant,slot,index=records.unbind(-1)
        active=(kind==task._scene_ids[e]) & ((kind!=BRIDGE)|(variant==task._bridge_count[e]-1))
        torch.testing.assert_close(roots[index[active],:3],task._static_centers[e[active],kind[active],slot[active]],atol=.002,rtol=0)
        torch.testing.assert_close(roots[index[active],3:7],task._static_quat[e[active],kind[active],slot[active]],atol=1e-6,rtol=0)
        assert torch.equal(fixed,roots[records[:,-1]])
    previous=roots.clone(); previous_ids=task._scene_ids.clone()
    env.reset(torch.tensor([0],device=task.device))
    actors=task.get_num_actors_per_env()
    assert torch.equal(previous[actors:],roots[actors:])
    assert torch.equal(previous_ids[1:],task._scene_ids[1:])
    stats['bridge_pusher_counts']=counts.cpu().tolist()
    stats['partial_reset']=True
    stats['fixed_actors']=True
    stats['num_envs']=task.num_envs
    path=Path(args.output_path);path.mkdir(parents=True,exist_ok=True)
    (path/'physics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats),flush=True)


if __name__=='__main__':
    main()
