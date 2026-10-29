"""Checkpoint selection and compact output for the Chair evaluation command."""
import argparse
import json
from pathlib import Path
import pickle
import re
import time


def select_latest(active_path, attempts=10, delay=.5):
    import torch
    torch.set_num_threads(1)
    active = json.loads(Path(active_path).read_text())
    train = Path(active['run']) / 'train'
    pattern = re.compile(r'^ChairAdaptResidual_(\d+)\.pth$')
    for attempt in range(attempts):
        candidates = [(int(pattern.fullmatch(p.name).group(1)), p)
                      for p in train.glob('ChairAdaptResidual_*/nn/*.pth')
                      if pattern.fullmatch(p.name)]
        if not candidates:
            raise FileNotFoundError('No numbered Chair residual checkpoint under ' + str(train))
        epoch, path = max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime_ns))
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except (EOFError, RuntimeError, pickle.UnpicklingError, OSError):
            if attempt + 1 == attempts:
                raise RuntimeError('Latest checkpoint is still incomplete; retry shortly: ' + str(path))
            time.sleep(delay)
            continue
        config = checkpoint['relation_experiment_config']
        if (int(checkpoint['epoch']) != epoch
                or config['env']['holdAtFT']['category'] != 'ArmChair_Normal'
                or config['train']['params']['network']['table_adaptation']['mode'] != 'residual'):
            raise ValueError('Expected a numbered native Chair residual checkpoint: ' + str(path))
        return path.resolve()


def format_summary(summary, output):
    if summary['smoke']:
        return 'SMOKE ONLY: not a performance evaluation.\nResults: ' + str(output)
    stats = summary['statistics']
    text = 'Mean ± Std: {:.2f} ± {:.2f}%'.format(stats['mean_percent'], stats['sample_std_percentage_points'])
    if 'tokenhsi_criteria_statistics' in summary:
        secondary = summary['tokenhsi_criteria_statistics']
        text = 'Ours (30s AT) ' + text
        text += ('\nTokenHSI criteria (20s AT + upright, same episode) Mean ± Std: '
                 '{:.2f} ± {:.2f}%').format(secondary['mean_percent'], secondary['sample_std_percentage_points'])
    return text + ('\n(sample std across {} independent evaluation seeds; ddof=1)\n'
                   'Results: {}').format(len(summary['records']), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the latest complete numbered Chair residual checkpoint.')
    parser.add_argument('active_metadata', type=Path)
    print(select_latest(parser.parse_args().active_metadata))
