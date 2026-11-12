"""Read-only near/far/facing selection from unmodified random reset snapshots."""
import argparse
import json
from pathlib import Path
import isaacgym
import torch


def select(initial):
    fixed=initial.get('fixed_layout')
    root = initial['roots'].double()
    n = root.shape[0]
    ids = torch.arange(n)
    physical = initial['physical_indices']
    box = initial['boxes'][ids, physical].double()
    size = initial['sizes'][ids, physical].double()
    q = box[:, 3:7]
    yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]), 1-2*(q[:,1]**2+q[:,2]**2))
    c, s = yaw.cos(), yaw.sin()
    delta = root[..., :2]-box[:, None, :2]
    local = torch.stack((delta[...,0]*c[:,None]+delta[...,1]*s[:,None],
                         -delta[...,0]*s[:,None]+delta[...,1]*c[:,None]), -1)
    region = initial['regions'].double()
    target = torch.stack(((region-.5)*size[:,None,0]/2,
                          -size[:,None,1].expand(-1,2)/2), -1)
    aim = target-local
    distance = aim.norm(dim=-1)
    rq = root[...,3:7]
    forward = torch.stack((1-2*(rq[...,1]**2+rq[...,2]**2),
                           2*(rq[...,0]*rq[...,1]+rq[...,3]*rq[...,2])), -1)
    local_forward = torch.stack((forward[...,0]*c[:,None]+forward[...,1]*s[:,None],
                                -forward[...,0]*s[:,None]+forward[...,1]*c[:,None]), -1)
    dot = (local_forward*aim).sum(-1)/(local_forward.norm(dim=-1)*distance).clamp_min(1e-9)
    angle = torch.acos(dot.clamp(-1,1))*180/torch.pi
    normal_gap = -local[...,1]-size[:,None,1]/2
    rows=[]
    for env in range(n):
        for a,b in ((0,1),(1,0)):
            if (fixed and a==fixed['A']) or (not fixed and .8 <= distance[env,a] <= 2. and 4. <= distance[env,b] <= 8.
                    and distance[env,b] >= 2.5*distance[env,a] and angle[env,a] <= 30.
                    and normal_gap[env,a] >= .35 and normal_gap[env,b] >= 2.
                    and local[env,a,0].abs() <= size[env,0]/2+.35):
                rows.append(dict(seed=int(initial['seed']),env_id=env,A=a,B=b,
                    a_face_target_distance_m=float(distance[env,a]), b_face_target_distance_m=float(distance[env,b]),
                    a_heading_error_degrees=float(angle[env,a]), a_face_normal_gap_m=float(normal_gap[env,a]),
                    b_face_normal_gap_m=float(normal_gap[env,b])))
    return dict(seed=int(initial['seed']),num_envs=n,env_ids=[r['env_id'] for r in rows],cases=rows,
        criteria=(dict(profile='user_authorized_fixed_positions',a_face_target_distance_m=[fixed['a_surface_normal_gap_m']]*2,
            b_face_target_distance_m=[fixed['b_surface_normal_gap_m']]*2,
            a_heading_offsets_degrees=fixed['a_heading_offsets_degrees'],b_heading_offsets_degrees=fixed['b_heading_offsets_degrees'],
            all_pose_heading_variants_retained=True) if fixed else dict(a_face_target_distance_m=[.8,2.],b_face_target_distance_m=[4.,8.],
            minimum_far_near_ratio=2.5,maximum_a_heading_error_degrees=30.,
            distance_definition='pelvis XY to assigned rear pushing face slot centre; actual force-stage B hand gap checked later')),
        initial_state_modified=bool(fixed),fixed_layout=fixed,selection_inputs_mutated=False,policy_actions_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('initial',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    initial=torch.load(args.initial,map_location='cpu',weights_only=False)
    result=select(initial)
    result['source_initial']=str(args.initial.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('seed',result['seed'],'initial scenes',result['num_envs'],'distance/facing matches',len(result['env_ids']))


if __name__=='__main__':
    main()
