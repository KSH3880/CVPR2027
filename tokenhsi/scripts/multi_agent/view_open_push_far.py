"""Evaluation-only far PUSH starts, hidden doors and optional continuous recording."""
from isaacgym import gymapi  # Must precede torch.
import runpy
import torch
from utils import open_push_scene
from utils.door_reference import rotate

original = open_push_scene.sample_loco_positions

def far_start(boxes, relative_body, root_height, size, distance_range, **kwargs):
    target = original(
        boxes, relative_body, root_height, size, (4.0, 4.0), **kwargs
    )
    forward = torch.zeros_like(boxes[:, :3])
    forward[:, 1] = 1.
    forward = rotate(boxes[:, 3:7], forward)
    offset = target[:, :2] - boxes[:, :2]
    flip = (offset * forward[:, :2]).sum(-1) < 0
    target[flip, :2] = boxes[flip, :2] - offset[flip]
    return target

open_push_scene.sample_loco_positions = far_start
# Hide door visuals while preserving articulation, collision and observations.
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

_door_dir = tempfile.TemporaryDirectory(prefix="open_push_invisible_door_")
_door_xml = ET.parse("tokenhsi/assets/door/door_hold.urdf")
for _link in _door_xml.getroot().findall("link"):
    for _visual in list(_link.findall("visual")):
        _link.remove(_visual)
    # Keep an explicit visual: Gym otherwise renders collision geometry.
    _visual = ET.SubElement(_link, "visual")
    ET.SubElement(_visual, "origin", xyz="0 0 -1000")
    _geometry = ET.SubElement(_visual, "geometry")
    ET.SubElement(_geometry, "box", size="0.001 0.001 0.001")
_door_xml.write(str(Path(_door_dir.name) / "door_hold.urdf"))
_original_load_box_asset = HumanoidMAOpenPush._load_box_asset

class _DoorVisualProxy:
    def __init__(self, gym):
        self._gym = gym
    def __getattr__(self, name):
        return getattr(self._gym, name)
    def load_asset(self, sim, root, filename, options):
        if filename == "door_hold.urdf":
            root = _door_dir.name
        return self._gym.load_asset(sim, root, filename, options)

def _load_without_door_visuals(self):
    gym = self.gym
    self.gym = _DoorVisualProxy(gym)
    try:
        return _original_load_box_asset(self)
    finally:
        self.gym = gym

HumanoidMAOpenPush._load_box_asset = _load_without_door_visuals


def enable_continuous_recording():
    import datetime
    import json
    import os
    import signal
    import sys
    import yaml
    from learning.multi_agent.ma_players import MAPlayerContinuous
    from utils.segmented_video import SegmentedVideoWriter

    camera_mode = os.environ.get('VIDEO_CAMERA', 'follow')
    if camera_mode not in ('follow', 'top'):
        raise ValueError('VIDEO_CAMERA must be follow or top')

    # Only presentation settings change; checkpoint physics/reference checks stay intact.
    config_index = sys.argv.index('--cfg_env') + 1
    with open(sys.argv[config_index]) as stream:
        cfg = yaml.safe_load(stream)
    cfg['env']['video'].update(
        enable=True, focus='agent0', numFrames=1,
        width=int(os.environ.get('VIDEO_WIDTH', '1280')),
        height=int(os.environ.get('VIDEO_HEIGHT', '720')),
        camDistance=3.0, camHeight=1.8,
    )
    config_path = Path(_door_dir.name) / 'record.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sys.argv[config_index] = str(config_path)
    original_eval = MAPlayerContinuous.run_eval

    def top_camera_pose(self):
        import math
        # Include both people, movable boxes and goals; exclude the hidden doors.
        points = torch.cat((self._humanoid_root_states[0, :, :3],
                            self._box_states[0, 2:, :3], self._tar_pos[0]), 0)
        xy = points[:, :2]
        low, high = xy.amin(0), xy.amax(0)
        center = (low + high) / 2 - self._env_origins[0, :2]
        half = (high - low) / 2 + 1.0
        tan_h = math.tan(math.radians(self._video_fov) / 2)
        tan_v = tan_h * self._video_h / self._video_w
        height = max(10.0, float(half[0]) / tan_h,
                     float(half[1]) / tan_v) * 1.15 + 2.0
        x, y = center.cpu().tolist()
        # Tiny Y offset avoids a look-at singularity with the world's Z up axis.
        position = gymapi.Vec3(x, y - 0.01, height)
        target = gymapi.Vec3(x, y, 0.0)
        return position, target

    def update_top_viewer(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        position, target = top_camera_pose(self)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], position, target)

    def capture_top_frame(self):
        position, target = top_camera_pose(self)
        self.gym.set_camera_location(self._video_cam, self.envs[0], position, target)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        image = self.gym.get_camera_image(
            self.sim, self.envs[0], self._video_cam, gymapi.IMAGE_COLOR)
        frame = image.reshape(self._video_h, self._video_w, 4)[:, :, :3]
        self._continuous_video.write(frame)

    def write_frames(self):
        frames, self._video_frames = self._video_frames, []
        for frame in frames:
            self._continuous_video.write(frame)
        # HumanoidMA calls this after each frame; keep capturing across resets.
        self._recording = True

    def run_recorded_eval(player):
        task = player.env.task
        if not (task._is_eval and task.cfg['args'].task_graph == 'push'):
            raise ValueError('Continuous far recording requires PUSH evaluation')
        directory = Path(task.cfg['args'].output_path) / 'videos'
        suffix = '_top' if camera_mode == 'top' else ''
        directory /= datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + suffix
        seconds = float(os.environ.get('VIDEO_SEGMENT_SECONDS', '60'))
        task._continuous_video = SegmentedVideoWriter(
            directory, fps=max(1, round(1.0 / task.dt)),
            segment_frames=max(1, round(seconds / task.dt)),
        )
        metadata = dict(
            seed=int(torch.initial_seed()), requested_seed=task.cfg['args'].seed,
            checkpoint=task.cfg['args'].checkpoint, camera=camera_mode,
            episode_length=task.max_episode_length, start_distance_m=4.0,
            width=task._video_w, height=task._video_h,
        )
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print('[record] seed:', metadata['seed'], flush=True)
        task._video_num_frames = 1
        task.request_video(str(directory / 'continuous.mp4'))
        previous_handler = signal.getsignal(signal.SIGTERM)

        def stop(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop)
        print('[record] continuously saving clean camera frames to', directory, flush=True)
        try:
            return original_eval(player)
        finally:
            task._recording = False
            task._continuous_video.close()
            signal.signal(signal.SIGTERM, previous_handler)

    HumanoidMAOpenPush._write_video = write_frames
    if camera_mode == 'top':
        HumanoidMAOpenPush._capture_video_frame = capture_top_frame
        # The interactive viewer has a separate camera from the recorded sensor.
        HumanoidMAOpenPush._update_camera = update_top_viewer
    MAPlayerContinuous.run_eval = run_recorded_eval


if __name__ == '__main__':
    import os
    if os.environ.get('CONTINUOUS_VIDEO', '0') == '1':
        enable_continuous_recording()
    runpy.run_path('tokenhsi/run.py', run_name='__main__')
