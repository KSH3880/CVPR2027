import os
from pathlib import Path

from run_group_spacing_formal import Formal, ROOT


class FourMeter16H(Formal):
    people = (16,)
    gap = 4.
    output_path = ROOT/'output_etc/check/table1_4m_16h_env512_seeds167_baselines_20261111'
    runner_path = Path(__file__)


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    FourMeter16H().run()
