from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import isaacgym
import depbias_ablation
from depbias_two_person import TwoPersonShuffle

depbias_ablation.DependencyBiasAblation = TwoPersonShuffle
runpy.run_path(str(ROOT/'eval/depbias_ablation_eval.py'), run_name='__main__')
