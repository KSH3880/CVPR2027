"""Live BEFORE alias, solo PUSH physics, path and partial-reset checks."""
from check_before_object_alias import inspect_alias
from check_before_pickup_carry_rsi import main
import torch
from utils.before_push_layout import paths_valid
from utils.jointpush_scene import goal_angles_valid


def inspect(task):
    result=inspect_alias(task)
    ids=torch.arange(task.num_envs,device=task.device)
    push=task._templates==5
    paired=task._before_predecessor>=0
    r,f=paired.nonzero(as_tuple=True);paired[r,task._before_predecessor[r,f]]=True
    assert not (push&paired).any()
    assert task.num_physical_objects==5*task.num_agents
    primary=task._primary_objects(task._templates)
    r,a=push.nonzero(as_tuple=True)
    assert (primary[r,a]==4*task.num_agents+a).all()
    actual=task._scenario_physical_object(r,a)
    assert torch.equal(actual,primary[r,a])
    assert (task._box_size[r,actual,2]>=1.45).all()
    density=task._box_density.reshape(task.num_envs,-1)
    assert (density[:,16:]==3).all() and (density[:,:16]==100).all()
    for e,o in zip(r.cpu().tolist(),actual.cpu().tolist()):
        props=task.gym.get_actor_rigid_body_properties(task.envs[e],task._box_handles[e*task.num_physical_objects+o])
        mass=float(task._box_size[e,o].prod()*3)
        assert abs(props[0].mass-mass)<1e-4
        shapes=task.gym.get_actor_rigid_shape_properties(task.envs[e],task._box_handles[e*task.num_physical_objects+o])
        assert abs(shapes[0].friction-.4)<1e-6
    boxes,sizes,goals=task._box_states,task._box_size,task._tar_pos
    active=task._active_physical(ids)
    spec=task._joint['scene']
    assert paths_valid(boxes,sizes,goals,task._humanoid_root_states[...,:3],push,primary,active,spec).all()
    source=boxes[ids[:,None],primary]
    assert goal_angles_valid(source,goals,push,spec).all()
    for distance in ((source[...,:2]-task._humanoid_root_states[...,:2]).norm(dim=-1),(source[...,:2]-goals[...,:2]).norm(dim=-1)):
        assert ((distance[push]>=2-3e-5)&(distance[push]<=10+3e-5)).all()
    family=task._amp_family().reshape(task.num_envs,task.num_agents)
    assert (family[push]==3).all() and not (family[~push]==3).any()
    assert not task._before_rsi_skill.any()
    for e,a in zip(*[v.cpu().tolist() for v in (task._reset_ref_slots['loco'])]):
        assert task._templates[e,a]>=1
    result.update(push_people=int(push.sum()),push_pairs=0,physical_banks=task.num_physical_objects,
        push_mass_friction=True,push_paths=True,amp_families=4,loco_only=True)
    return result


if __name__=='__main__':
    import sys
    preset=sys.argv[sys.argv.index('--task_graph')+1] if '--task_graph' in sys.argv else 'random_scenario'
    cases=(preset,) if preset in ('before_sit','before_climb') else ('random_scenario','all_solo','push','before_stack','one_pair_stack')
    main(inspector=inspect,presets=[(p,'loco') for p in cases])
