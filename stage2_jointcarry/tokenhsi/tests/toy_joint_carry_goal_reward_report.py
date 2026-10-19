import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(root):
    labels = ['original', 'along_motion', 'action_replay']
    data = {name: np.load(root / name / 'reward_traces.npz') for name in labels}
    if len({str(v['physical_hash']) for v in data.values()}) != 1:
        raise RuntimeError('Initial physical states differ')
    summary = dict(initial_physical_states_identical=True, conditions={})
    for name, p in data.items():
        d, held, alive = p['distance'], p['holding'], p['alive']
        end = alive.sum(0)-1
        valid = alive & held
        predicted = .2 / (1 + np.maximum(d-.1, 0)) * held
        error = np.abs(predicted[..., None] - p['at_progress_component']).max()
        near = valid[1:] & valid[:-1] & (np.diff(d, axis=0) < -1e-5)
        change = np.diff(p['at_progress_component'].mean(-1), axis=0)
        result = dict(mean_final_distance=float(d[end, np.arange(len(end))].mean()),
            held_approach_30cm_fraction=float((np.where(valid, d, np.inf).min(0) <= 1.2).mean()),
            holding_fraction=float(valid.sum()/alive.sum()),
            at_progress_formula_max_error=float(error),
            held_approaching_steps=int(near.sum()),
            at_progress_decreases_while_approaching=int((near & (change < -1e-6)).sum()),
            scenes={})
        for env in (0, 1):
            closest = int(np.argmin(np.where(valid[:, env], d[:, env], np.inf)))
            samples = {}
            for tag, start, stop in [('first_half_second', 0, 15),
                                      ('near_closest', max(0, closest-4), min(int(end[env])+1, closest+5))]:
                window = slice(start, stop)
                samples[tag] = dict(distance=float(d[window, env].mean()),
                    holding=float(held[window, env].mean()),
                    at_reward=p['at_reward'][window, env].mean(0).tolist(),
                    holding_reward=p['holding_reward'][window, env].mean(0).tolist(),
                    alignment=float(p['alignment'][window, env].mean()),
                    task=p['reward_terms'][window, env, :, -1].mean(0).tolist(),
                    amp=p['amp'][window, env].mean(0).tolist(),
                    combined=p['combined'][window, env].mean(0).tolist())
            displacement = p['body'][end[env], env, :, 0, :2] - p['body'][0, env, :, 0, :2]
            result['scenes'][str(env)] = dict(closest_held_distance=float(d[closest, env]),
                closest_time=(closest+1)/30, final_distance=float(d[end[env], env]),
                displacement_along_initial_facing=(displacement*p['initial_facing'][env]).sum(-1).tolist(),
                samples=samples)
        summary['conditions'][name] = result
    a, b, replay = (data[k] for k in labels)
    summary['goal_change_initial_action_mean_abs_difference'] = float(np.abs(a['action'][0]-b['action'][0]).mean())
    summary['replay_action_max_error'] = float(np.abs(a['action']-replay['action']).max())
    summary['replay_body_max_error'] = float(np.abs(a['body']-replay['body']).max())
    summary['replay_box_max_error'] = float(np.abs(a['box']-replay['box']).max())
    summary['replay_amp_max_error'] = float(np.abs(a['amp']-replay['amp']).max())
    summary['replay_holding_reward_max_error'] = float(np.abs(a['holding_reward']-replay['holding_reward']).max())
    closest = int(np.argmin(np.where(replay['alive'][:, 1] & replay['holding'][:, 1],
                                    replay['distance'][:, 1], np.inf)))
    window = slice(max(0, closest-4), min(180, closest+5))
    summary['scene1_same_motion_goal_only_comparison'] = {}
    for name, p in [('original_goal', a), ('goal_along_motion', replay)]:
        summary['scene1_same_motion_goal_only_comparison'][name] = dict(
            distance=float(p['distance'][window, 1].mean()),
            at_reward=p['at_reward'][window, 1].mean(0).tolist(),
            task=p['reward_terms'][window, 1, :, -1].mean(0).tolist(),
            combined=p['combined'][window, 1].mean(0).tolist())
    if any(r['at_progress_formula_max_error'] > 1e-5 or r['at_progress_decreases_while_approaching'] for r in summary['conditions'].values()):
        raise RuntimeError('AT distance reward failed monotonicity check')
    (root/'summary.json').write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for env in (0, 1):
        for name, p in data.items():
            last = int(p['alive'][:, env].sum())
            t = np.arange(1, last+1)/30
            for col, values in enumerate((p['distance'][:last, env],
                    p['at_reward'][:last, env].mean(-1), p['combined'][:last, env].mean(-1))):
                axes[env, col].plot(t, values, label=name)
        for col, title in enumerate(('Box-goal distance (m)', 'AT reward per human', '0.5 task + 0.5 AMP per human')):
            axes[env, col].set(title='Scene %d: '%env+title, xlabel='Seconds')
            axes[env, col].grid(alpha=.2)
            axes[env, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'reward_comparison.png', dpi=140)
    plt.close(fig)

    edges = [(0,1),(1,2),(1,3),(3,4),(4,5),(1,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(0,12),(12,13),(13,14)]
    colors = [(220,110,40), (70,170,60)]
    titles = ['Original goal / policy', 'Goal along motion / policy', 'Goal along motion / action replay']
    for env in (0, 1):
        center = a['initial_box'][env].copy(); center[2] = 0
        forward = b['target'][env, :2] - center[:2]
        forward /= np.linalg.norm(forward)
        side = np.array([-forward[1], forward[0]])

        def project(points):
            delta = points-center
            x, depth = delta[..., :2]@forward, delta[..., :2]@side
            return np.stack((220+48*(x+.4*depth), 300-90*delta[...,2]+16*depth), -1).astype(int)

        path = root/('scene_'+str(env)+'.mp4')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1440,420))
        for frame in range(180):
            columns = []
            for name, title in zip(labels, titles):
                p = data[name]; step = min(frame, int(p['alive'][:,env].sum())-1)
                canvas = np.full((420,480,3), 248, np.uint8)
                points = project(p['body'][step,env])
                cv2.line(canvas,(0,300),(479,300),(210,210,210),1)
                cv2.drawMarker(canvas,tuple(project(p['target'][env])),(20,150,200),cv2.MARKER_STAR,22,2)
                cv2.circle(canvas,tuple(project(p['box'][step,env])),8,(20,185,225),-1)
                for agent in range(2):
                    for x,y in edges:
                        cv2.line(canvas,tuple(points[agent,x]),tuple(points[agent,y]),colors[agent],3)
                lines = [title, 'distance %.2fm | holding %s'%(p['distance'][step,env],p['holding'][step,env]),
                         'AT %.3f | task %.3f | mixed %.3f'%(p['at_reward'][step,env].mean(),p['reward_terms'][step,env,:,-1].mean(),p['combined'][step,env].mean())]
                for row,text in enumerate(lines):
                    cv2.putText(canvas,text,(10,25+row*24),cv2.FONT_HERSHEY_SIMPLEX,.47,(30,30,30),1)
                cv2.putText(canvas,'A blue / B green | box dot / goal star',(10,380),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
                cv2.putText(canvas,'%.2fs | per-human mean reward'%((frame+1)/30),(10,405),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
                columns.append(canvas)
            writer.write(np.concatenate(columns,axis=1))
        writer.release()
        capture=cv2.VideoCapture(str(path)); count=0
        while capture.read()[0]: count+=1
        capture.release()
        if count != 180: raise RuntimeError('Video decode failed')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    report(parser.parse_args().directory)
