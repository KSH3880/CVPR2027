#!/usr/bin/env python3
"""No-policy, no-human native asset rest/reset check on the selected GPU."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from isaacgym import gymapi, gymtorch
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--case', default='carry_table')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asset = json.loads(args.manifest.read_text())['cases'][args.case]['assets'][0]
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.num_threads = 1
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 0
    params.physx.contact_offset = .02
    params.physx.rest_offset = 0.
    params.physx.bounce_threshold_velocity = .2
    params.physx.max_depenetration_velocity = 10.
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    opts = gymapi.AssetOptions()
    opts.use_mesh_materials = True
    opts.vhacd_enabled = True
    opts.vhacd_params.resolution = 100000
    opts.vhacd_params.max_convex_hulls = 128
    opts.vhacd_params.max_num_vertices_per_ch = 64
    opts.override_inertia = True
    handle = gym.load_asset(sim, asset['local'], 'asset.urdf', opts)
    envs, actors = [], []
    for i in range(4):
        env = gym.create_env(sim, gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 3), 2)
        pose = gymapi.Transform()
        pose.p.z = asset['bbox'][2] / 2
        envs.append(env)
        actors.append(gym.create_actor(env, handle, pose, 'native_object', i, 0))
    gym.prepare_sim(sim)
    states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    initial = states.clone()
    bodies = gym.get_actor_rigid_body_properties(envs[0], actors[0])
    report = dict(mass=bodies[0].mass, com=[bodies[0].com.x, bodies[0].com.y, bodies[0].com.z], stages={})
    for stage in ('creation', 'teleport_yaw'):
        if stage == 'teleport_yaw':
            states.copy_(initial)
            angles = torch.tensor([2.36, 5.46, 1.82, 3.94], device=states.device)
            states[:, 5] = (angles / 2).sin()
            states[:, 6] = (angles / 2).cos()
            states[:, 7:] = 0
            states[:, 0] += 2
            ids = torch.arange(4, dtype=torch.int32, device=states.device)
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(states), gymtorch.unwrap_tensor(ids), 4)
        trace = []
        for step in range(60):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            if step < 3 or step == 59:
                trace.append(dict(step=step+1, states=states.cpu().tolist()))
        report['stages'][stage] = trace
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
