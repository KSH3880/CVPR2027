import concurrent.futures
import subprocess
import sys
import time

import torch

from ind_mechanisms_common import ROOT, OURS, SCT, command, digest, write
from ind_mechanisms_scenes import finish, rotate_state, rotate_xy
from run_ind_mechanisms import memory

OUT = ROOT/'output_etc/check/sit_crossing_v4_2000_env64_seed1'


def scene(source, separated):
    n = len(source['roots'])
    rng = torch.Generator().manual_seed(10001)
    distance = 3+2*torch.rand(n, generator=rng)
    yaw = 2*torch.pi*torch.rand(n, generator=rng)
    side = torch.where(torch.rand(n, generator=rng) < .5, -1., 1.)
    headings = torch.stack((yaw, yaw+(0 if separated else torch.pi/2)*side), -1)
    centers = torch.zeros(n, 2, 3)
    if separated:
        centers[:, 0, 1], centers[:, 1, 1] = -4., 4.
        centers = rotate_xy(centers, yaw)
    roots = source['roots']
    q = roots[..., 3:7]
    old_heading = torch.atan2(2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1]),
                             1-2*(q[..., 1]**2+q[..., 2]**2))
    local = torch.zeros(n, 2, 3)
    local[..., 0] = -distance[:, None]/2
    local[..., 2] = roots[..., 2]-source['origins'][:, None, 2]
    center_world = centers+source['origins'][:, None]
    target = rotate_xy(local, headings)+center_world
    rotation = headings-old_heading
    translation = target-rotate_xy(roots[..., :3], rotation)
    sizes = source['physical_sizes'][:, [2, 3, 6, 7]].clone()
    boxes = torch.zeros(n, 4, 13)
    boxes[..., 6] = 1
    boxes[..., 2] = sizes[..., 2]/2
    boxes[:, :2, 0] = distance[:, None]/2
    box_yaw = 2*torch.pi*torch.rand(n, 2, generator=rng)
    boxes[:, :2, 5], boxes[:, :2, 6] = (box_yaw/2).sin(), (box_yaw/2).cos()
    boxes[:, 2:, 1] = 5
    boxes[:, :2] = rotate_state(boxes[:, :2], headings, center_world)
    boxes[:, 2:] = rotate_state(boxes[:, 2:], headings, center_world)
    return finish(dict(roots=rotate_state(roots, rotation, translation),
        kinematic=rotate_state(source['kinematic'], rotation, translation),
        dof_pos=source['dof_pos'].clone(), dof_vel=source['dof_vel'].clone(),
        templates=torch.ones(n, 2, dtype=torch.long), origins=source['origins'].clone(),
        sizes=sizes, density=source['density'][:, [2, 3, 6, 7]].clone(), boxes=boxes,
        goals=center_world.clone(), crossing_centers=center_world, headings=headings,
        route_length=distance, condition='separated' if separated else 'cross'))


def prepare():
    source = torch.load(OUT/'capture/extended.pt', map_location='cpu', weights_only=False)
    if len(source['roots']) != 64:
        raise ValueError('Expected 64 captured environments')
    scenes = {c:scene(source, c == 'separated') for c in ('separated', 'cross')}
    for condition, s in scenes.items():
        start, end, center = s['roots'][..., :2], s['boxes'][:, :2, :2], s['crossing_centers'][..., :2]
        expected = s['route_length'][:, None].expand(-1, 2)
        if not torch.allclose((end-start).norm(dim=-1), expected, atol=5e-5):
            raise ValueError('Start-to-chair distance mismatch')
        if not torch.allclose((start+end)/2, center, atol=5e-5):
            raise ValueError('Crossing point is not the midpoint')
        if (s['packet'][..., 1] != 0).any():
            raise ValueError('Non-SIT task')
        radii = s['sizes'][:, :2, :2].norm(dim=-1)/2
        if ((end[:, 0]-end[:, 1]).norm(dim=-1) <= radii.sum(-1)+.1).any():
            raise ValueError('Chairs overlap')
        if ((start[:, :, None]-end[:, None]).norm(dim=-1) <= radii[:, None]+.4).any():
            raise ValueError('Initial human/chair overlap')
        if ((center-end).norm(dim=-1)-radii < .8).any():
            raise ValueError('Chair intrudes into crossing zone')
        path = OUT/f'{condition}_seed1.pt'
        if path.exists():
            raise ValueError('Refusing to overwrite scene: '+str(path))
        torch.save(s, path)
    a, b = scenes.values()
    for key in ('sizes', 'density', 'dof_pos', 'dof_vel', 'route_length'):
        if not torch.equal(a[key], b[key]):
            raise ValueError('Paired scenes differ: '+key)
    for key in ('roots', 'boxes', 'kinematic'):
        def local(s):
            if key == 'boxes':
                return torch.cat([rotate_xy(s[key][:, i:i+2, :3]-s['crossing_centers'], -s['headings']) for i in (0, 2)], 1)
            centers = s['crossing_centers'] if key == 'roots' else s['crossing_centers'][:, :, None]
            return rotate_xy(s[key][..., :3]-centers, -s['headings'])
        if not torch.allclose(local(a), local(b), atol=5e-5):
            raise ValueError('Individual geometry differs: '+key)
    write(OUT/'manifest.json', dict(envs=64, seed=1, agents=2, steps=900,
        task='SIT', distance_uniform_m=[3,5], midpoint_equal_distance=True,
        separated_gap_m=8, cross_angle_deg=90, zone_radius_m=.8,
        unused_goal_tokens='group midpoint; translated/rotated with each group',
        metric='geometric proximity, not attributed physical contact',
        source_sha256=digest(OUT/'capture/extended.pt'),
        checkpoints={str(p):digest(p) for p in (SCT, OURS)},
        code_sha256={str(ROOT/'eval'/name):digest(ROOT/'eval'/name) for name in
            ('run_sit_crossing_pilot.py','summarize_sit_crossing_pilot.py','ind_mechanisms_eval.py')},
        scenes={c:digest(OUT/f'{c}_seed1.pt') for c in scenes}))


def run_gpu(gpu, model):
    for condition in ('separated', 'cross'):
        while memory()[gpu]+6144 > 70000:
            time.sleep(5)
        path = OUT/'runs'/condition/model
        path.mkdir(parents=True, exist_ok=False)
        cmd, env = command('ind_mechanisms_eval.py', path, 1, 2, gpu,
            dict(IND_SCENE=str(OUT/f'{condition}_seed1.pt'), IND_MODEL=model))
        for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY'):
            env.pop(key, None)
        cmd[cmd.index('--num_envs')+1] = '64'
        started = time.monotonic()
        with (path/'run.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(path/'execution.json', dict(pid=proc.pid, gpu=gpu, finished=False))
            code = proc.wait()
        write(path/'execution.json', dict(pid=proc.pid, gpu=gpu, finished=True,
            returncode=code, elapsed_seconds=time.monotonic()-started))
        if code:
            raise RuntimeError('Evaluation failed: '+str(path))
        print(condition, model, 'finished', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    prepare()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_gpu, gpu, model) for gpu, model in ((6,'sct'),(7,'ours'))]
        for job in jobs:
            job.result()
    subprocess.run([sys.executable, str(ROOT/'eval/summarize_sit_crossing_pilot.py')], cwd=ROOT, check=True)
