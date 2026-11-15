# 관계·샘플링 지표

`edge_context_task.py`, `edge_ontop_task.py`가 state/progress/success·CPA·상자 속도·물리 배치 재시도를 집계하고 `relation_diagnostics.py`가 TensorBoard 태그를 정리한다.

PUSH 포함 설정은 `relation/edge/contact/*`(PUSH 접촉 관계)와 `relation/edge/push_at/*`(PUSH 소유자의 AT 관계)를 분리한다. `relation/edge/at/*`에는 운반 AT만 집계한다. 세 그룹 모두 `own_success`, `phi_raw`, `progress_raw`, `reward_saturated`, 보상 성분·거리 등을 동일하게 기록하며, 유효한 해당 관계가 없는 집계 구간은 태그를 쓰지 않는다. PUSH가 없는 설정의 기존 태그는 유지한다.

`own_success`는 해당 구간의 유효 edge/frame 기준 현재 성공 비율이며 episode 누적 성공률이 아니다. `contact/own_success`는 양손 위치·접촉 힘 조건이고, `push_at/own_success`는 학습 보상의 목표 도달 조건(중심 거리20cm·Z 오차1mm)이다. 목표 도달 시 접촉 보상이 포화될 수 있으므로 `contact/reward_saturated`와 실제 접촉 `own_success`는 다를 수 있다. 정식 평가의 PUSH 성공은 아래 평가 규약을 따른다.

이 분리는 진단 마스크·집계만 바꾸며 보상·정책 입력·loss·성공 판정은 바꾸지 않는다. 집계와 기록에는 실행 시간 비용이 있으며 속도 영향은 측정하지 않았다. 이미 실행 중인 프로세스에는 자동 반영되지 않고 다음 시작·재개부터 적용된다. 변경 전 PUSH 실행의 `edge/at/*`는 운반 AT와 PUSH AT가 혼합된 값이므로 변경 후 운반 전용 값과 직접 비교하지 않는다.

BEFORE는 gate/raw_success/success와 과제별 값을 기록한다. mix226은 original/medium/short 비율, near50은 가까운 쌍 비율, RSI433는 스킬별 requested/executed/fallback을 기록한다.

`viewer_diagnostics.py`는 viewer의 학습 보상 기준 최종/누적 진단값이며 정식 평가표가 아니다. 정식 점/영역 및 BEFORE 지표는 [평가 규약](evaluation.md)의 `formal_eval.py`를 따른다. [structure](../structure.md), [config](../config.md)를 따른다.
