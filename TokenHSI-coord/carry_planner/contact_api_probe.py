"""Controlled positive/negative check for the installed PhysX contact API."""
import json
from isaacgym import gymapi
from carry_planner.physical_contact import classify_contacts


def main():
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    sim = gym.create_sim(-1, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("CPU PhysX creation failed")
    try:
        fixed = gymapi.AssetOptions(); fixed.fix_base_link = True
        static = gym.create_box(sim, .5, .5, .5, fixed)
        sphere = gym.create_sphere(sim, .2, gymapi.AssetOptions())
        envs, maps = [], []
        for i, height in enumerate((3., .45)):
            env = gym.create_env(sim, gymapi.Vec3(-2,-2,0), gymapi.Vec3(2,2,4), 2)
            pose = gymapi.Transform(); pose.p = gymapi.Vec3(0,0,.25)
            a = gym.create_actor(env, static, pose, "agent0", i, 0)
            pose.p = gymapi.Vec3(0,0,height)
            b = gym.create_actor(env, sphere, pose, "agent1", i, 0)
            maps.append({gym.get_actor_rigid_body_index(env,a,0,gymapi.DOMAIN_ENV):('agent',0),
                         gym.get_actor_rigid_body_index(env,b,0,gymapi.DOMAIN_ENV):('agent',1)})
            envs.append(env)
        gym.prepare_sim(sim)
        hits=[False,False]; contact_counts=[0,0]; fields=None
        for _ in range(20):
            gym.simulate(sim); gym.fetch_results(sim,True)
            for i,env in enumerate(envs):
                contacts=gym.get_env_rigid_contacts(env)
                fields=contacts.dtype.names
                contact_counts[i]+=len(contacts)
                hits[i] |= classify_contacts(contacts,maps[i])['agent_agent']
        assert hits == [False,True], (hits,contact_counts,fields)
        print('PHYSICAL_CONTACT_API_PROBE '+json.dumps(dict(hits=hits,contact_counts=contact_counts,fields=fields)))
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__': main()
