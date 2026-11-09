"""Render a selected recorded rollout. No policy, physics stepping, or hidden actors."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from isaacgym import gymapi
import numpy as np


def rigid_states(values, origin):
    states = np.zeros(len(values), dtype=gymapi.RigidBodyState.dtype)
    for axis, index in zip('xyz', range(3)):
        states['pose']['p'][axis] = values[:, index] - origin[index]
        states['vel']['linear'][axis] = values[:, 7+index]
        states['vel']['angular'][axis] = values[:, 10+index]
    for axis, index in zip('xyzw', range(4)):
        states['pose']['r'][axis] = values[:, 3+index]
    return states


class Recorder:
    def __init__(self, case, fps):
        self.root = case / 'recordings'
        self.fps = fps
        self.directory = None
        self.count = 0

    def start(self):
        self.directory = self.root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory.mkdir(parents=True)
        self.count = 0
        print('Recording started:', self.directory, flush=True)

    def capture(self, gym, viewer):
        if self.directory is not None:
            gym.write_viewer_image_to_file(viewer, str(self.directory / f'frame_{self.count:06d}.png'))
            self.count += 1

    def stop(self):
        directory, self.directory = self.directory, None
        if directory is None or self.count == 0:
            return
        (directory / 'recording.json').write_text(json.dumps({'fps': self.fps, 'frames': self.count}, indent=2))
        print('PNG frames saved:', directory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--record', action='store_true', help='Record immediately; save individual PNG frames.')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--frames', type=int, default=0, help='Exit after this many displayed frames (verification).')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--start-frame', type=int, default=0)
    args = parser.parse_args()
    if args.speed <= 0: parser.error('--speed must be positive')
    meta = json.loads((args.case/'metadata.json').read_text())
    data = np.load(args.case/'trajectory.npz')
    people = meta['people'] if 'people' in meta else [meta['leader'], meta['follower']]
    packet, order = data['packet'], data['logical_box_order']
    m = meta['num_agents']
    # Show only the prerequisite support box and the follower's payload.
    boxes = meta.get('replay_box_ids', [int(order[int(packet[a, 3])-m]) for a in people])
    boxes = list(dict.fromkeys(boxes))
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0,0,0)
    params.dt = meta['dt']
    params.physx.use_gpu = True
    sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,params)
    if sim is None: raise RuntimeError('Cannot create replay renderer')
    viewer = None
    recorder = Recorder(args.case, args.speed / meta['dt'])
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0,0,1)
        gym.add_ground(sim,plane)
        env = gym.create_env(sim,gymapi.Vec3(-20,-20,0),gymapi.Vec3(20,20,20),1)
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        root = Path(__file__).resolve().parent/'data/assets/mjcf'
        asset = gym.load_asset(sim,str(root),'phys_humanoid_v3.xml',options)
        assert gym.get_asset_rigid_body_count(asset) == data['bodies'].shape[2]
        assert gym.get_asset_dof_count(asset) == data['dofs'].shape[2]
        humans = []
        colors = [(0.24,0.52,0.86),(0.94,0.47,0.18),(.3,.75,.4),(.75,.35,.8),
                  (.9,.75,.2),(.2,.75,.8),(.85,.3,.4),(.65,.65,.65),(.55,.65,.95),(.95,.65,.45),
                  (.55,.9,.65),(.85,.6,.95),(.8,.85,.5),(.45,.9,.9),(.95,.55,.7),(.8,.75,.65)]
        for index,a in enumerate(people):
            actor = gym.create_actor(env,asset,gymapi.Transform(),f'agent_{a}',0,0)
            for body in range(gym.get_actor_rigid_body_count(env,actor)):
                gym.set_rigid_body_color(env,actor,body,gymapi.MESH_VISUAL,gymapi.Vec3(*colors[index]))
            humans.append(actor)
        objects = []
        for index,b in enumerate(boxes):
            size = data['sizes_physical'][b]
            obj = gym.create_box(sim,*map(float,size),options)
            actor = gym.create_actor(env,obj,gymapi.Transform(),f'box_{b}',0,0)
            gym.set_rigid_body_color(env,actor,0,gymapi.MESH_VISUAL,gymapi.Vec3(*colors[meta['box_color_agents'][index] if 'box_color_agents' in meta else index]))
            objects.append(actor)
        gym.prepare_sim(sim)
        # Initialize PhysX/graphics once before applying any recorded frame.
        # Subsequent frames only overwrite recorded states; no simulation steps.
        gym.simulate(sim)
        gym.fetch_results(sim,True)
        camera = gymapi.CameraProperties()
        camera.width, camera.height = 1600,900
        viewer = gym.create_viewer(sim,camera)
        if viewer is None: raise RuntimeError('Cannot open replay viewer')
        for key,name in [(gymapi.KEY_SPACE,'pause'),(gymapi.KEY_R,'restart'),
                         (gymapi.KEY_L,'record'),(gymapi.KEY_RIGHT,'next'),(gymapi.KEY_LEFT,'previous'),(gymapi.KEY_ESCAPE,'quit')]:
            gym.subscribe_viewer_keyboard_event(viewer,key,name)
        origin = data['origin']
        path = data['bodies'][:,people,0,:3]-origin
        low,high = path.reshape(-1,3).min(0),path.reshape(-1,3).max(0)
        center = (low+high)/2
        radius = max(float(np.linalg.norm((high-low)[:2]))*.75,4.)
        target = gymapi.Vec3(float(center[0]),float(center[1]),.85)
        eye = gymapi.Vec3(float(center[0]+radius*.8),float(center[1]-radius),radius*.60)
        gym.viewer_camera_look_at(viewer,env,eye,target)
        print('camera',eye,target,'env origin',gym.get_env_origin(env),flush=True)
        goals = data['goals'] if meta.get('preset') == 'random_scenario' else data['pair_goal'][None]
        angles = np.linspace(0,2*np.pi,49)
        rings=[]
        for goal in goals-origin:
            ring=np.stack((goal[0]+.25*np.cos(angles),goal[1]+.25*np.sin(angles),
                           np.full_like(angles,.015)),axis=-1).astype(np.float32)
            rings.append(np.stack((ring[:-1],ring[1:]),axis=1).reshape(-1,6))
        lines=np.concatenate(rings)
        line_colors=np.tile(np.array([.25,.7,.3],np.float32),(len(lines),1))
        print('Recorded '+meta.get('task_name','ON_TOP')+' replay. SPACE pause, R restart, L PNG record/stop, arrows step, ESC close.',flush=True)
        print(json.dumps(meta,indent=2),flush=True)
        index,shown,paused = args.start_frame,0,False
        length = len(data['bodies'])
        if not 0 <= index < length: raise ValueError('start-frame outside saved trajectory')
        if args.record: recorder.start()
        while not gym.query_viewer_has_closed(viewer):
            tick = time.monotonic()
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0: continue
                if event.action == 'quit': return
                if event.action == 'record':
                    if recorder.directory is None: recorder.start()
                    else: recorder.stop()
                if event.action == 'pause': paused = not paused
                if event.action == 'restart': index = 0
                if event.action in ('next','previous'):
                    paused = True
                    index = min(length-1,max(0,index+(1 if event.action == 'next' else -1)))
            states = np.concatenate([data['bodies'][index,a] for a in people] +
                                    [data['roots'][index,m+b:m+b+1] for b in boxes])
            gym.set_env_rigid_body_states(env,rigid_states(states,origin),gymapi.STATE_ALL)
            for a,actor in zip(people,humans):
                dofs = np.zeros(len(data['dofs'][index,a]),dtype=gymapi.DofState.dtype)
                dofs['pos'],dofs['vel'] = data['dofs'][index,a].T
                gym.set_actor_dof_states(env,actor,dofs,gymapi.STATE_ALL)
            gym.step_graphics(sim)
            gym.clear_lines(viewer)
            gym.add_lines(viewer,env,len(lines),lines,line_colors)
            gym.draw_viewer(viewer,sim,True)
            if args.screenshot and shown == min(5,args.frames-1 if args.frames else 5):
                gym.write_viewer_image_to_file(viewer,str(args.screenshot))
            recorder.capture(gym,viewer)
            shown += 1
            if args.frames and shown >= args.frames: break
            if not paused:
                index += 1
                if index == length:
                    index = 0
                    print('Replay loop restarted',flush=True)
            time.sleep(max(0.,meta['dt']/args.speed-(time.monotonic()-tick)))
    finally:
        recorder.stop()
        if viewer is not None: gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
