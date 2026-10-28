"""Keep nuisance boxes away from human approaches and physical PUSH sweeps."""
import math
import torch
from utils.jointpush_scene import segment_distance
from utils.stage2_jointpush_rsi import hand_targets
from utils import stage2_jointpush_scene as scene

MODE = 'far_clear_paths_v1'


def clear_approaches(points, half_sizes, roots, targets, clearance):
    # Circumcircle bounds the entire nuisance-box footprint for every yaw.
    radius = half_sizes.norm(dim=-1)
    distance = segment_distance(roots, targets, points[:,None], points[:,None])
    return (distance >= radius[:,None]+clearance).all(-1)


def approach_paths(task, ids):
    boxes = task._assigned_box_values(task._box_states, ids)
    sizes = task._assigned_box_values(task._box_size, ids)
    target = hand_targets(boxes, sizes, task._tar_pos[ids], task._members[ids],
        task._regions[ids], task._joint['contact']).mean(-2)[...,:2]
    direction = task._tar_pos[ids,:,:2]-boxes[...,:2]
    direction /= direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    target -= direction  # Human approach ends 1m outside the selected pushing face.
    roots = task._humanoid_root_states[ids,:,:2]
    return roots, target


def install(env):
    spec = env.get('stage2JointPush', {}).get('distractor_placement')
    if spec is None:
        return
    if (spec['mode'] != MODE or spec['radius'] != [8.,12.]
            or spec['agent_path_clearance'] != 1.5 or spec['push_path_clearance'] != 1.5):
        raise ValueError('Unknown nuisance-box placement contract')
    original_infeasible = scene.scene_infeasible

    def place(task, ids):
        if not hasattr(task, '_distractor_placement_failed'):
            task._distractor_placement_failed = torch.zeros(task.num_envs,device=task.device,dtype=torch.bool)
        task._distractor_placement_failed[ids] = False
        for owner in range(4):
            pending = ids.clone()
            slot = 4+owner
            for _ in range(256):
                if not len(pending):
                    break
                draws = torch.rand(len(pending),2,device=task.device)
                angle = draws[:,0]*2*math.pi
                low,high=spec['radius']
                radius = (low*low+draws[:,1]*(high*high-low*low)).sqrt()
                point = task._env_origins[pending,:2]+radius[:,None]*torch.stack((angle.cos(),angle.sin()),-1)
                task._box_states[pending,12+owner,:2] = point
                starts,ends,yaw,half = scene.paths(task,pending)
                hits = scene.swept_overlap(point[:,None],point[:,None],yaw[:,slot,None],half[:,slot,None],
                    starts,ends,yaw,half,spec['push_path_clearance'])
                hits[:,slot] = False
                roots,targets=approach_paths(task,pending)
                valid = ~hits.any(-1) & clear_approaches(point,half[:,slot],roots,targets,spec['agent_path_clearance'])
                pending = pending[~valid]
            task._distractor_placement_failed[pending] = True

    def infeasible(task, ids, rsi=False):
        result = original_infeasible(task,ids,rsi=rsi)
        if hasattr(task, '_distractor_placement_failed'):
            result |= task._distractor_placement_failed[ids]
        return result

    scene.place_distractors = place
    scene.scene_infeasible = infeasible
    print('KSH_DISTRACTOR_CLEARANCE',spec,flush=True)
