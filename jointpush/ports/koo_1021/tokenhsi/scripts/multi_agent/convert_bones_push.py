"""Convert the two selected BONES Uniform push references for joint OPEN/PUSH AMP."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from isaacgym import gymapi  # Must precede torch/Poselib.
from tokenhsi.scripts.multi_agent.convert_bones_door import ROOT, convert

CLIPS = {'push_obstacle_180_101__A341': 12.2,
         'push_obstacle_180_003__A342': 11.733333}

if __name__ == '__main__':
    for name, end in CLIPS.items():
        paths = list((ROOT/'bones_seed/soma_uniform/bvh').rglob(name+'.bvh'))
        if len(paths) != 1:
            raise RuntimeError('Extract the selected original BVH first: '+name)
        output = ROOT/'tokenhsi/data/dataset_bones_push/motions'/name/'phys_humanoid_v3'
        convert(paths[0], output, start=0., end=end, profile='push')
