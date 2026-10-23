"""Register an evaluation-only task/player; leave running training code untouched."""
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from evaluation.finetuned.backend import EvaluationTask,EvaluationPlayer
import utils.parse_task as parser
parser.EvaluationTask=EvaluationTask
spec=importlib.util.spec_from_file_location('evaluation_run',ROOT/'tokenhsi/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
original=run.build_alg_runner
def build(observer):
    runner=original(observer);runner.player_factory.register_builder('chair_ft',lambda **kwargs:EvaluationPlayer(**kwargs));return runner
run.build_alg_runner=build
run.main()
