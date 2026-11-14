from pathlib import Path
import sys

import torch

from ind_mechanisms_common import OUT, ROOT, digest, write
sys.path.insert(0, str(ROOT/'tokenhsi'))
from utils import stage2_spec
from utils.task_role_spec import task_graph_packet


def finish(scene):
    n, m = scene['templates'].shape
    groups = torch.arange(m).expand(n, -1).clone()
    graph = stage2_spec.compose_graph(scene['templates'], groups, torch.full_like(groups, -1), 2*m, False)
    mapping, primary, used = stage2_spec.physical_mapping(scene['templates'], groups, torch.full_like(groups, -1), graph)
    scene.update(packet=task_graph_packet(graph, n).reshape(n, m, 5).long(),
                 mapping=mapping, primary=primary, used=used, groups=groups)
    return scene


def solo(source, slot):
    objects = [slot, 2+slot]
    scene = {key:source[key][:, slot:slot+1].clone()
             for key in ('roots', 'kinematic', 'dof_pos', 'dof_vel', 'goals')}
    scene.update(templates=source['packet'][:, slot:slot+1, 1]+1,
        boxes=source['boxes'][:, objects].clone(), sizes=source['sizes'][:, objects].clone(),
        density=source['density'].gather(1, source['logical_box_order'][:, objects]),
        origins=source['origins'].clone(), source_slot=slot)
    finish(scene)
    obs = source['observation']
    poses = obs[:, 568:624].reshape(-1, 8, 7)
    expected = torch.cat((obs[:, slot*223:(slot+1)*223],
        obs[:, 446:566].reshape(-1, 4, 30)[:, objects].flatten(1), obs[:, 566+slot:567+slot],
        poses[:, [slot, 2+slot, 4+slot, 6+slot]].flatten(1),
        scene['packet'].flatten(1).float(), scene['groups'].float()), -1)
    scene['expected_observation'] = expected
    return scene


def rotate_xy(values, angle):
    c, s = angle.cos(), angle.sin()
    while c.ndim < values.ndim-1:
        c, s = c[..., None], s[..., None]
    result = values.clone()
    result[..., 0] = c*values[..., 0]-s*values[..., 1]
    result[..., 1] = s*values[..., 0]+c*values[..., 1]
    return result


def rotate_state(states, angle, translation):
    result = states.clone()
    delta = translation
    while delta.ndim < states.ndim:
        delta = delta.unsqueeze(-2)
    result[..., :3] = rotate_xy(states[..., :3], angle)+delta
    result[..., 7:10] = rotate_xy(states[..., 7:10], angle)
    result[..., 10:13] = rotate_xy(states[..., 10:13], angle)
    c, s = (angle/2).cos(), (angle/2).sin()
    while c.ndim < states.ndim-1:
        c, s = c[..., None], s[..., None]
    x, y, z, w = states[..., 3:7].unbind(-1)
    result[..., 3:7] = torch.stack((c*x-s*y, s*x+c*y, c*z+s*w, c*w-s*z), -1)
    return result


def crossing(source, seed, separated):
    n = len(source['roots'])
    rng = torch.Generator().manual_seed(10000+seed)
    distance = 3.+2*torch.rand(n, generator=rng)
    approach = 1.+4*torch.rand(n, generator=rng)
    yaw = 2*torch.pi*torch.rand(n, generator=rng)
    flip = torch.where(torch.rand(n, generator=rng) < .5, -1., 1.)
    headings = torch.stack((yaw, yaw+(0 if separated else torch.pi/2)*flip), -1)
    centers = torch.zeros(n, 2, 3)
    if separated:
        centers[:, 0, 1], centers[:, 1, 1] = -4., 4.
        centers = rotate_xy(centers, yaw)
    roots = source['roots']
    q = roots[..., 3:7]
    old_heading = torch.atan2(2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1]),
                             1-2*(q[..., 1]**2+q[..., 2]**2))
    root_local = torch.zeros(n, 2, 3)
    root_local[..., 0] = (-distance/2-approach)[:, None]
    root_local[..., 2] = roots[..., 2]-source['origins'][:, None, 2]
    target_root = rotate_xy(root_local, headings)+centers+source['origins'][:, None]
    rotation = headings-old_heading
    translation = target_root-rotate_xy(roots[..., :3], rotation)
    scene = dict(roots=rotate_state(roots, rotation, translation),
        kinematic=rotate_state(source['kinematic'], rotation, translation),
        dof_pos=source['dof_pos'].clone(), dof_vel=source['dof_vel'].clone(),
        templates=torch.full((n, 2), 3), origins=source['origins'].clone(),
        sizes=source['physical_sizes'][:, [0, 1, 6, 7]].clone(),
        density=source['density'][:, [0, 1, 6, 7]].clone(),
        route_length=distance, approach_length=approach, headings=headings)
    boxes = torch.zeros(n, 4, 13)
    boxes[..., 6] = 1.
    boxes[:, :2, 0] = -distance[:, None]/2
    boxes[:, :2, 2] = scene['sizes'][:, :2, 2]/2
    box_yaw = 2*torch.pi*torch.rand(n, 2, generator=rng)
    boxes[:, :2, 5] = (box_yaw/2).sin()
    boxes[:, :2, 6] = (box_yaw/2).cos()
    delta = centers+source['origins'][:, None]
    boxes[:, :2] = rotate_state(boxes[:, :2], headings, delta)
    boxes[:, 2:, :3] = source['origins'][:, None]+torch.tensor([[[6., -6., 0.], [-6., 6., 0.]]])
    boxes[:, 2:, 2] = source['origins'][:, None, 2]+scene['sizes'][:, 2:, 2]/2
    goals = torch.zeros(n, 2, 3)
    goals[..., 0] = distance[:, None]/2
    goals[..., 2] = scene['sizes'][:, :2, 2]/2
    scene.update(boxes=boxes, goals=rotate_xy(goals, headings)+delta)
    return finish(scene)


def validate_pair(cross, separated):
    for scene in (cross, separated):
        roots, boxes, goals = scene['roots'][..., :2], scene['boxes'][:, :2, :2], scene['goals'][..., :2]
        if not torch.allclose((roots-boxes).norm(dim=-1), scene['approach_length'][:, None].expand(-1, 2), atol=1e-4):
            raise ValueError('Approach distance changed')
        if not torch.allclose((goals-boxes).norm(dim=-1), scene['route_length'][:, None].expand(-1, 2), atol=1e-4):
            raise ValueError('Carry distance changed')
        if ((roots[:, 0]-roots[:, 1]).norm(dim=-1) < .7).any():
            raise ValueError('Initial people overlap')
        radii = scene['sizes'][:, :2, :2].norm(dim=-1)/2
        if ((boxes[:, 0]-boxes[:, 1]).norm(dim=-1) <= radii.sum(-1)+.1).any():
            raise ValueError('Initial boxes overlap')
        gaps = (roots[:, :, None]-boxes[:, None]).norm(dim=-1)
        if (gaps <= radii[:, None]+.4).any():
            raise ValueError('Initial person-box overlap')
    for key in ('sizes', 'density', 'dof_pos', 'dof_vel', 'route_length', 'approach_length'):
        if not torch.equal(cross[key], separated[key]):
            raise ValueError('Pair changed intrinsic task: '+key)
    for scene in (cross, separated):
        local = rotate_xy(scene['roots'][..., :3]-scene['boxes'][:, :2, :3], -scene['headings'])
        scene['local_root_offset'] = local
    if not torch.allclose(cross['local_root_offset'], separated['local_root_offset'], atol=1e-4):
        raise ValueError('Pair changed local pose')


def prepare(seed):
    path = OUT/f'capture/seed{seed}/extended.pt'
    source = torch.load(path, map_location='cpu', weights_only=False)
    if not torch.equal(source['packet'][..., 2], torch.arange(2).expand(512, -1)):
        raise ValueError('Source packet owner order changed')
    scenes = {'solo_a':solo(source, 0), 'solo_b':solo(source, 1),
              'cross':crossing(source, seed, False), 'separated':crossing(source, seed, True)}
    validate_pair(scenes['cross'], scenes['separated'])
    for name, scene in scenes.items():
        destination = OUT/f'scenes/{name}_seed{seed}.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError('Scene already exists: '+str(destination))
        torch.save(scene, destination)
        write(destination.with_suffix('.json'), dict(source=str(path), source_sha256=digest(path),
            scene_sha256=digest(destination), condition=name, seed=seed, envs=512,
            agents=scene['templates'].shape[1], paired_geometry_verified=True,
            collision_metric='body-point surface proximity; not contact-pair collision'))


if __name__ == '__main__':
    for value in sys.argv[1:]:
        prepare(int(value))
