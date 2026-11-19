import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output_etc/check/dct_delay_env64_seed1_20261115'
OURS = ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth'
SCT = ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
COOP_ROOT = ROOT/'table89_assets/sct_coopft'
COOP = COOP_ROOT/'output/sct_coopft/SCTCoopFT_10-22-24-41/nn/SCTCoopFT_00007000.pth'
MODELS = dict(sct=SCT, fsm=SCT, coopft=COOP, ours=OURS)
SETTINGS = dict(enable_sit=2, coupled_3=3)
DELAYS = (0, 2, 4)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    pending.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()
