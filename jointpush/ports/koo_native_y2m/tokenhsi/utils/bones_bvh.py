"""Read BONES SOMA BVH without FBX, preserving its absolute Hips translation."""
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation


def read_bvh(path):
    header, motion = Path(path).read_text().split('MOTION', 1)
    tokens = re.findall(r'[{}]|[^\s{}]+', header)
    cursor = 0
    nodes = []

    def take():
        nonlocal cursor
        result = tokens[cursor]
        cursor += 1
        return result

    def read_node(parent):
        nonlocal cursor
        kind = take()
        if kind not in ('ROOT', 'JOINT'):
            raise ValueError('Invalid BVH node')
        node = dict(name=take(), parent=parent, channels=[])
        index = len(nodes)
        nodes.append(node)
        assert take() == '{'
        while tokens[cursor] != '}':
            field = take()
            if field == 'OFFSET':
                node['offset'] = np.array([float(take()) for _ in range(3)])
            elif field == 'CHANNELS':
                node['channels'] = [take() for _ in range(int(take()))]
            elif field == 'JOINT':
                cursor -= 1
                read_node(index)
            elif field == 'End':
                assert take() == 'Site' and take() == '{'
                depth = 1
                while depth:
                    value = take()
                    depth += (value == '{') - (value == '}')
            else:
                raise ValueError('Unsupported BVH field: ' + field)
        take()

    assert take() == 'HIERARCHY'
    read_node(-1)
    lines = motion.strip().splitlines()
    frames = int(lines[0].split(':')[1])
    dt = float(lines[1].split(':')[1])
    channel_count = sum(len(n['channels']) for n in nodes)
    values = np.fromstring(' '.join(lines[2:]), sep=' ').reshape(frames, channel_count)
    if not np.isfinite(values).all() or dt <= 0:
        raise ValueError('Nonfinite BVH or invalid frame time')
    if nodes[0]['name'] != 'Root' or not np.all(values[:, :6] == 0):
        raise ValueError('Expected static BONES Root wrapper')
    positions, rotations = [], []
    cursor = 0
    for node in nodes:
        channels = node['channels']
        data = values[:, cursor:cursor + len(channels)]
        cursor += len(channels)
        rotation_ids = [i for i, c in enumerate(channels) if c.endswith('rotation')]
        order = ''.join(channels[i][0] for i in rotation_ids)
        local_rot = Rotation.from_euler(order, data[:, rotation_ids], degrees=True).as_matrix()
        translation_ids = [i for i, c in enumerate(channels) if c.endswith('position')]
        # SOMA Hips channels are absolute, not Hips OFFSET + translation.
        local_pos = data[:, translation_ids] if translation_ids else np.broadcast_to(node['offset'], (frames, 3))
        parent = node['parent']
        if parent < 0:
            pos, rot = local_pos, local_rot
        else:
            pos = positions[parent] + np.einsum('nij,nj->ni', rotations[parent], local_pos)
            rot = rotations[parent] @ local_rot
        positions.append(pos)
        rotations.append(rot)
    return nodes, np.stack(positions, axis=1) / 100, np.stack(rotations, axis=1), dt
