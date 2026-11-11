"""Local training entry point; safe for the spawned native simulator worker."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]

if __name__ == '__main__':
    from scripts.local_cuda_compat import install
    install()
    runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')
