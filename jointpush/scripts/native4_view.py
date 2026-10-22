"""Original player/viewer loop with the validated evaluation-only four-person task."""
from pathlib import Path
import os,sys,runpy
ROOT=Path(__file__).resolve().parents[1]
PORT=ROOT/'ports/koo_native_y2m'
sys.path.insert(0,str(PORT/'tokenhsi'))
import isaacgym
from native4_y2m import install
install()
if os.environ.get('JOINTPUSH_VIEW_BIAS')=='1':
 from bias_view import install as install_view_bias
 install_view_bias()
runpy.run_path(str(PORT/'tokenhsi/run.py'),run_name='__main__')
