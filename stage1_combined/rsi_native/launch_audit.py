from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from rsi_native.audit_backend import NativeRsiAuditTask,NativeRsiAuditPlayer
import utils.parse_task as parser
parser.NativeRsiAuditTask=NativeRsiAuditTask
spec=importlib.util.spec_from_file_location('native_rsi_run',ROOT/'tokenhsi/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
original=run.build_alg_runner
def build(observer):
 runner=original(observer);runner.player_factory.register_builder('chair_ft',lambda **kw:NativeRsiAuditPlayer(**kw));return runner
run.build_alg_runner=build;run.main()
