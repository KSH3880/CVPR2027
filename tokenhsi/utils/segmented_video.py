"""Stream RGB frames to numbered MP4 files without buffering a full clip."""
from pathlib import Path


class SegmentedVideoWriter:
    def __init__(self, directory, fps, segment_frames):
        if fps <= 0 or segment_frames < 1:
            raise ValueError('fps and segment_frames must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.segment_frames = segment_frames
        self.writer = None
        self.index = 0
        self.frames = 0
        self.path = None
        self.shape = None

    def write(self, frame):
        import cv2
        if self.writer is None:
            self.index += 1
            self.path = self.directory / ('part_%06d.mp4' % self.index)
            height, width = frame.shape[:2]
            self.shape = frame.shape
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps, (width, height),
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError('Cannot open MP4 writer: ' + str(self.path))
            self.frames = 0
        if frame.shape != self.shape:
            raise ValueError('Video frame dimensions changed during recording')
        self.writer.write(frame[:, :, ::-1])  # RGB -> BGR
        self.frames += 1
        if self.frames >= self.segment_frames:
            self.close()

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print('[record] saved {} ({} frames)'.format(self.path, self.frames), flush=True)
