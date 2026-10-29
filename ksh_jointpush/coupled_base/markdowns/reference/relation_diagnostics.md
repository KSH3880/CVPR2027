# 관계·샘플링 지표

`edge_context_task.py`, `edge_ontop_task.py`가 state/progress/success·CPA·상자 속도·물리 배치 재시도를 집계하고 `relation_diagnostics.py`가 TensorBoard 태그를 정리한다.

BEFORE는 gate/raw_success/success와 과제별 값을 기록한다. mix226은 original/medium/short 비율, near50은 가까운 쌍 비율, RSI433는 스킬별 requested/executed/fallback을 기록한다.

`viewer_diagnostics.py`는 viewer의 학습 보상 기준 최종/누적 진단값이며 정식 평가표가 아니다. 정식 점/영역 및 BEFORE 지표는 [평가 규약](evaluation.md)의 `formal_eval.py`를 따른다. [structure](../structure.md), [config](../config.md)를 따른다.
