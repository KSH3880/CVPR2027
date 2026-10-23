import math

from isaacgym import gymapi, gymtorch
import numpy as np
import torch
import yaml

from joint_carry.geometry import geometry
from utils.size_rsi import screen_states


def screen(data, support_size=None):
    cfg = yaml.safe_load(open('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml'))
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1/60
    params.substeps = cfg['sim']['substeps']
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = False
    for key, value in cfg['sim']['physx'].items():
        setattr(params.physx, key, value)
    params.physx.use_gpu = True
    params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Could not create the RSI physics screen')
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    plane.static_friction = plane.dynamic_friction = 1.
    gym.add_ground(sim, plane)
    options = gymapi.AssetOptions()
    options.angular_damping = .01
    options.max_angular_velocity = 100.
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset = cfg['env']['asset']
    human = gym.load_asset(sim, asset['assetRoot'], asset['assetFileName'], options)
    names = gym.get_asset_rigid_body_names(human)
    if names != data['joint_names'].tolist():
        raise ValueError('RSI rigid-body ordering differs from the native asset')
    properties = gym.get_asset_dof_properties(human)
    properties['driveMode'] = gymapi.DOF_MODE_POS
    options.density = 100.
    options.linear_damping = .01
    box_asset = gym.create_box(sim, *data['box_size'].tolist(), options)
    support_asset = gym.create_box(sim, *(support_size or [.4, .4, .3]), options)
    count = len(data['time'])
    origins = []
    for frame in range(count):
        env = gym.create_env(sim, gymapi.Vec3(-8, -8, 0), gymapi.Vec3(8, 8, 8), math.ceil(math.sqrt(count)))
        origins.append(gym.get_env_origin(env))
        for person in range(2):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(person*3, 0, 1)
            actor = gym.create_actor(env, human, pose, 'human'+str(person), frame, 0)
            gym.set_actor_dof_properties(env, actor, properties)
        for index, asset in enumerate((box_asset, support_asset)):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(5+index, 5, .2)
            gym.create_actor(env, asset, pose, 'box'+str(index), frame, 0)
    gym.prepare_sim(sim)
    roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).reshape(count, 4, 13)
    dofs = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).reshape(count, 2, 32, 2)
    bodies = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).reshape(count, 32, 13)
    contacts = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).reshape(count, 32, 3)
    origin = torch.tensor([[p.x, p.y, p.z] for p in origins])
    root_state = np.concatenate((data['root_position'], data['root_rotation_xyzw'],
                                 data['body_linear_velocity'][:, :, 0], data['body_angular_velocity'][:, :, 0]), -1)
    box_state = np.concatenate((data['box_position'], data['box_rotation_xyzw'],
                                data['box_linear_velocity'], data['box_angular_velocity']), -1)
    roots[:, :2] = torch.tensor(root_state, dtype=torch.float32)
    roots[:, :2, :3] += origin[:, None]
    roots[:, 2] = torch.tensor(box_state, dtype=torch.float32)
    roots[:, 2, :3] += origin
    roots[:, 3] = 0.
    roots[:, 3, 6] = 1.
    roots[:, 3, :3] = origin
    if support_size is not None:
        roots[:, 3, :2] += torch.tensor(data['terminal_box_xy'], dtype=torch.float32)
        roots[:, 3, 2] += support_size[2]/2
    else:
        roots[:, 3, 0] += 6.
        roots[:, 3, 2] += .15
    dofs[..., 0] = torch.tensor(data['dof_position'], dtype=torch.float32)
    dofs[..., 1] = torch.tensor(data['dof_velocity'], dtype=torch.float32)
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(roots))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dofs))
    targets = dofs[..., 0].contiguous().clone()
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(targets))
    initial_root, initial_boxes = roots[:, :2].clone(), roots[:, 2:].clone()
    geo = geometry(data)
    valid = torch.tensor(np.maximum.reduce([v for k, v in geo.items() if k != 'worst_nonhand_shape']) <= .02)
    support_geometry_accepted = count
    if support_size is not None:
        support_data = dict(data)
        support_data['box_position'] = np.column_stack((data['terminal_box_xy'], np.full(count, support_size[2]/2)))
        support_data['box_rotation_xyzw'] = np.tile([0., 0., 0., 1.], (count, 1))
        support_data['box_size'] = np.asarray(support_size)
        support_geo = geometry(support_data)
        support_valid = np.maximum.reduce([v for k,v in support_geo.items() if k != 'worst_nonhand_shape']) <= .02
        valid &= torch.tensor(support_valid)
        support_geometry_accepted = int(support_valid.sum())
    allowed = data.get('source_rsi_allowed', np.ones(count, dtype=bool))
    valid &= torch.tensor(allowed.astype(bool))
    lower = torch.tensor(properties['lower'])
    upper = torch.tensor(properties['upper'])
    limit_error = torch.maximum((lower-dofs[..., 0]).clamp_min(0), (dofs[..., 0]-upper).clamp_min(0)).flatten(1).amax(-1)
    valid &= limit_error <= 1e-5
    stiffness_range = [float(properties['stiffness'].min()), float(properties['stiffness'].max())]
    contact_steps = torch.zeros(count, 2)
    simultaneous_steps = torch.zeros(count)
    peak_hand_force = 0.
    nearest_hand = float("inf")
    for _ in range(6):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        for person in range(2):
            valid &= screen_states(roots[:, person], roots[:, 2:], initial_root[:, person], initial_boxes)
        valid &= torch.isfinite(dofs).flatten(1).all(-1)
        hands = bodies[:, [5, 8, 20, 23], :3].reshape(count, 2, 2, 3)
        force = contacts[:, [5, 8, 20, 23]].norm(dim=-1).reshape(count, 2, 2)
        local = hands-roots[:, 2, None, None, :3]
        q = roots[:, 2, None, None, 3:7].expand(-1, 2, 2, -1)
        cross = 2*torch.cross(q[..., :3], local, dim=-1)
        local = local-q[..., 3:]*cross+torch.cross(q[..., :3], cross, dim=-1)
        distance = (local.abs()-torch.tensor(data['box_size'])[None, None, None]/2).clamp_min(0).norm(dim=-1)
        peak_hand_force = max(peak_hand_force, float(force.max()))
        nearest_hand = min(nearest_hand, float(distance.min()))
        touching = ((force > .1) & (distance <= .065)).any(-1)
        contact_steps += touching
        simultaneous_steps += touching.all(-1)
    carrying = torch.tensor(data.get('carry_mask', np.full(count, str(data['skill']) == 'carryWith')))
    valid &= ~carrying | (simultaneous_steps >= 3)
    box_displacement = float((roots[:,2,:3]-initial_boxes[:,0,:3]).norm(dim=-1).max())
    return valid.numpy(), dict(frames=count, accepted=int(valid.sum()),
                              geometry_mask=(np.maximum.reduce([v for k,v in geo.items() if k != 'worst_nonhand_shape']) <= .02).tolist(),
                              contact_fraction_by_frame=(contact_steps/6).tolist(),
                              simultaneous_contact_fraction_by_frame=(simultaneous_steps/6).tolist(),
                              support_geometry_accepted=support_geometry_accepted,
                              geometry_accepted=int((np.maximum.reduce([v for k,v in geo.items() if k != 'worst_nonhand_shape']) <= .02).sum()),
                              mean_contact_fraction=(contact_steps/6).mean(0).tolist(), peak_hand_force=peak_hand_force, nearest_hand=nearest_hand,
                              stiffness_range=stiffness_range, maximum_joint_limit_violation_rad=float(limit_error.max()),
                              box_displacement=box_displacement, device='selected CUDA GPU')
