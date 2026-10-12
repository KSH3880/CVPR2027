import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(root):
    labels = ['baseline', 'amp05', 'amp0']
    data = {name: np.load(root / name / 'traces.npz') for name in labels}
    metrics = {name: json.loads((root / name / 'evaluation.json').read_text()) for name in labels}
    hashes = [json.loads((root / name / 'initial_check.json').read_text())['initial_state_sha256'] for name in labels]
    if len(set(hashes)) != 1:
        raise RuntimeError('Evaluation initial states differ')
    for name, values in data.items():
        if not all(np.isfinite(values[key]).all() for key in ('body', 'box', 'distance')):
            raise RuntimeError('Nonfinite evaluation: ' + name)
    (root / 'summary.json').write_text(json.dumps(dict(initial_states_identical=True, evaluations=metrics), indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, sign, title in zip(axes, (1, -1), ('A forward / B backward', 'A backward / B forward')):
        for name, values in data.items():
            mask = values['sign'] == sign
            distance = values['distance'][:, mask].copy()
            live = values['alive'][:, mask]
            for frame in range(1, len(distance)):
                distance[frame, ~live[frame]] = distance[frame-1, ~live[frame]]
            axis.plot(np.arange(len(distance))/30, distance.mean(1), label=name)
        axis.set(title=title, xlabel='Seconds', ylabel='Box-goal XY distance (m)')
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(root / 'distance.png', dpi=160)
    plt.close(fig)

    edges = [(0,1),(1,2),(1,3),(3,4),(4,5),(1,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(0,12),(12,13),(13,14)]
    colors = [(220,110,40), (70,170,60)]
    for env in (0, 1):
        baseline = data['baseline']
        center = baseline['box'][0, env].copy()
        center[2] = 0
        forward = baseline['target'][env, :2] - center[:2]
        forward /= np.linalg.norm(forward)
        side = np.array([-forward[1], forward[0]])

        def project(points, column):
            delta = points - center
            x = delta[..., :2] @ forward
            depth = delta[..., :2] @ side
            return np.stack((column*480+210+60*(x+.4*depth),
                             310-95*delta[..., 2]+20*depth), -1).astype(int)

        path = root / ('scene_' + str(env) + '.mp4')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1440, 420))
        for frame in range(len(baseline['body'])):
            canvas = np.full((420, 1440, 3), 248, np.uint8)
            for column, name in enumerate(labels):
                values = data[name]
                valid = np.flatnonzero(values['alive'][:, env])
                step = min(frame, int(valid[-1]))
                points = project(values['body'][step, env], column)
                goal = project(values['target'][env], column)
                box = project(values['box'][step, env], column)
                cv2.line(canvas, (column*480, 310), ((column+1)*480-1, 310), (210,210,210), 1)
                cv2.drawMarker(canvas, tuple(goal), (20,150,200), cv2.MARKER_STAR, 22, 2)
                cv2.circle(canvas, tuple(box), 8, (20,185,225), -1)
                for agent in range(2):
                    for a,b in edges:
                        cv2.line(canvas, tuple(points[agent,a]), tuple(points[agent,b]), colors[agent], 3)
                    for point in points[agent]:
                        cv2.circle(canvas, tuple(point), 3, colors[agent], -1)
                distance = values['distance'][step, env]
                held = values['holding'][step, env]
                cv2.putText(canvas, name+' | distance %.2fm'%distance, (column*480+12,32), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,30,30), 1)
                cv2.putText(canvas, 'joint holding: '+str(bool(held)), (column*480+12,58), cv2.FONT_HERSHEY_SIMPLEX, .5, (30,30,30), 1)
                cv2.putText(canvas, 'A blue / B green | box yellow / goal star', (column*480+12,385), cv2.FONT_HERSHEY_SIMPLEX, .43, (60,60,60), 1)
            cv2.putText(canvas, 'simulation %.2fs | fixed scene %d'%(frame/30,env), (12,413), cv2.FONT_HERSHEY_SIMPLEX, .5, (30,30,30), 1)
            writer.write(canvas)
        writer.release()
        reader = cv2.VideoCapture(str(path))
        count = 0
        while reader.read()[0]:
            count += 1
        reader.release()
        if count != len(baseline['body']):
            raise RuntimeError('Incomplete comparison video')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    report(parser.parse_args().directory)
