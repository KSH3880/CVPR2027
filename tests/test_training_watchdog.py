import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from watch_training import eligible, candidates


def test_only_cuda_timeout_restarts():
    old = dict(pid=42, output_path='/run', status='running')
    timeout = 'RuntimeError: CUDA error: the launch timed out and was terminated'
    assert eligible(old, old, timeout, False)
    assert not eligible(old, old, timeout, True)
    assert not eligible(old, dict(old, status='stopped_by_user'), timeout, False)
    assert not eligible(old, dict(old, pid=43), timeout, False)
    assert not eligible(old, old, 'KeyboardInterrupt', False)
    assert not eligible(old, old, 'CUDA out of memory', False)
    assert not eligible(old, old, timeout+' MAX EPOCHS NUM!', False)


def test_latest_checkpoint_and_resume_fallback(tmp_path):
    nn = tmp_path / 'run' / 'experiment' / 'nn'
    nn.mkdir(parents=True)
    fallback = tmp_path / 'TokenHSIPushKD_00018500.pth'
    fallback.touch()
    active = dict(output_path=str(tmp_path / 'run'), resume_checkpoint=str(fallback))
    assert candidates(active) == [fallback]
    newer = nn / 'TokenHSIPushKD_00019000.pth'
    newer.touch()
    (nn / 'TokenHSIPushKD.pth').touch()
    assert candidates(active) == [newer, fallback]
