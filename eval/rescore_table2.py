import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tokenhsi'))
import stage2_xy100
from table2_metrics import replay


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--distance-cm',nargs='+',type=float,default=[6,8,12])
    parser.add_argument('--start-displacement-cm',type=float,default=20)
    parser.add_argument('--start-hand-distance-cm',type=float)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if (args.start_displacement_cm <= 0 or any(value <= 0 for value in args.distance_cm)
            or (args.start_hand_distance_cm is not None and args.start_hand_distance_cm <= 0)):
        parser.error('Distances must be positive')
    stage2_xy100.install()
    start_hand = None if args.start_hand_distance_cm is None else args.start_hand_distance_cm/100
    result = replay(args.run.resolve(),tuple(value/100 for value in args.distance_cm),args.start_displacement_cm/100,start_hand)
    text = json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
