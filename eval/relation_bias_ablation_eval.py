from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import isaacgym
import depbias_ablation
from relation_bias_ablation import RelationBiasAblation

depbias_ablation.DependencyBiasAblation = RelationBiasAblation
runpy.run_path(str(ROOT/'eval/depbias_ablation_eval.py'), run_name='__main__')
