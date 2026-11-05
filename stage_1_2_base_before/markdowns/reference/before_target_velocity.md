# relfusion + 목표 수평 속도 L2

현재 `_velocity_l2` YAML에는 `after_release: disable`이 추가되어 선행 최초 성공 후 속도비용을 완전히 해제한다. 아래 완료 후 추종/fade 설명은 이 옵션이 없는 이전 저장 설정의 동작이다. 사용자 후속 요청으로 현재 제안은 [속도 상한 방식](before_speed_cap.md)으로 분리했고, 학습은 모두 중단한 상태에서 추가 시작하지 않았다.

실험: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_velocity_l2`.

원본 relfusion 모델, 가상박스, progress/state/success, 낙상/시간 종료는 그대로다. 새 waiting_penalty.target_velocity 옵션에서 SIT/CLIMB는 사람 root, ON_TOP은 운반 박스의 XY 위치·속도를 사용한다. ON_TOP은 최초 HOLDING 성공 전 속도비용을0으로 두며, 성공 후에는 해당 에피소드 내 활성 상태를 유지한다. 접근 early term, asym, state 정규화는 추가하지 않는다.

## 계산

선행 최초 AT성공 전: target=선행 goal XY, R=지지박스 XY반대각선+3m.
성공 후: target=실제 지지박스 XY, R=XY반대각선+스킬 buffer(SIT/CLIMB0.3m, ON_TOP0.1m).

- d=target과 제어 대상(root 또는 운반 박스)의 수평 거리, e=max(d−R,0).
- 목표 속도 크기=min(e/2초,0.5m/s), 방향=(target−제어 대상)/d. d=0에서도 finite.
- 선행 완료 전 weight=1. 완료 후 weight=clip(e/1m,0,1).
- 벌점=−1.0×weight×L2(실제 제어 대상 XY속도−목표 XY속도). norm은 제곱하지 않는다.

경계1m밖까지 0.5m/s, 마지막1m는 선형 감속, 안쪽은 목표0. 밖에서 정지하면 목표속도만큼 비용이 생기고, 올바른 접근이면 비용0. 경계 안에서 원주 운동도 비용을 받는다. 경계 재이탈 시 현재 위치로 접근 목표를 재계산한다. 수직 속도는 이 비용에서 제외한다. ON_TOP의 집기·들기·내려놓기 수직 운동 자체를 속도비용으로 벌주지 않는다.

최초 선행 성공 후 release latch를 유지한다. 실제 박스 근처에서는 속도비용이0으로 줄어 기존 앉기/오르기를 강제 수평속도로 방해하지 않는다. goal 위치가 아니라 실제 박스가 이동하면 목표도 따라간다. 선행 성공 전후 target/weight 전환은 시간 블렌딩 없이 이산 전환이므로 보상 점프가 남을 수 있다. 이 옵션은 그 전환의 연속성을 보장하는 실험이 아니다. 대기 경계 안쪽 정지 위치를 구분하거나 깊은 침범 자체를 막는 항도 없다.

## 실행과 관측

전용 train/test/vnc/eval wrapper와 [실행 명령](../config.md)을 사용한다. 기존 실행은 자동 변경하지 않는다. 기본 학습 초기화는 Stage1 PUSH26k이며 원본 relfusion3500에서 finetune/resume를 수행하지 않았다. 모델 구조는 동일하지만 보상 계약은 다르므로 기존 checkpoint의 저장 설정을 새 보상으로 오인하지 않는다.

기존 TensorBoard `wait_penalty` 항에 새 L2 비용이 포함된다. 런타임 `_before_wait.target_velocity`와 `velocity_weight`에 agent별 목표 XY속도/가중치를 보관한다(단독 및 집기 전 ON_TOP은0). `_before_wait.picked_up`는 edge별 최초 HOLDING 성공 latch다. CPU 회귀와 짧은 headless 물리 검증으로 연결을 확인하며, 학습 성능 개선 여부는 별도 학습/평가가 필요하다.

## ON_TOP 활성 조건

ON_TOP source(payload)와 HOLDING target이 같고, HOLDING source가 후행 owner인 유효 edge를 정확히 하나 찾는다. 그래프 edge 순서와 무관하게 매칭하며, `raw_own_success`로 최초 성공을 포착한다. 선행의 AT 성공만으로는 활성화하지 않는다. 부분 리셋은 해당 환경의 latch만 지운다. 이미 집은 초기 상태는 reset의 HOLDING 성공을 반영한다.

집기 전에는 기존 hard root-speed 벌점도 대체해 완전히0이다. 집기 후 선행 미완료이면 goal 주위 대기 경계로 운반하고, 선행 완료 이후에는 실제 지지박스로 운반한다. 쌓기 목표 접근 구간에서 비용을 줄인다. 박스를 놓거나 떨어뜨려도 latch는 유지되므로 집기 상태를 끊어 비용을 회피할 수 없지만, 낙하 후 재집기 과정에도 비용이 남을 수 있다. HOLDING 최초 성공 순간 목표속도 비용이 켜지는 전환도 별도 smoothing 없이 이산적이다.

`ontop` 옵션이 없는 이전 SIT/CLIMB 전용 설정은 ON_TOP legacy 동작을 유지한다. 현재 `_velocity_l2` 설정에는 `ontop.activation: first_holding_success`, `goal_buffer: 0.1`을 추가했으며 checkpoint 보상 계약에도 포함된다.
