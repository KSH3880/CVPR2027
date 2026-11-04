# 선행 배치 후 2초 이내 최초 접근 확인

원본 `virtual_asym_goal5_goalroute` 및 일반 `virtual_goal5_goalroute`에 종료 옵션을 추가한 두 별도 실험이다. 두 설정은 asym 유무 외에는 같은 최초 접근 조건을 사용한다. state 정규화, 보상 추가, 조기 접근 종료는 포함하지 않는다. 기존 원본 설정의 동작은 바꾸지 않는다.

- 선행 배치 판정은 기존 prerequisite gate(선행 AT current success)를 재사용한다. 손을 놓았는지를 별도로 감지하지 않는다.
- 각 BEFORE 후행 edge의 gate가 처음 열리면 후행 agent root XY와 선행 goal XY 사이 거리를 기록하고 타이머를 시작한다. SIT/CLIMB/ON_TOP 후행 모두 같은 agent 거리 정의를 쓴다(ON_TOP의 기존 progress는 payload 거리이지만 이 타이머는 agent 거리).
- 최초 gate 개방 시 거리보다 누적 1cm 이상 가까워지면 접근 시작으로 인정하고 해당 edge 감시를 영구 해제한다(다음 환경 reset 전까지). 이후 정지·후퇴·gate 재개폐로 재무장하지 않는다.
- 최초 접근 없이 2초가 지나면 해당 환경 전체를 reset하고 모든 agent의 terminate를1로 만든다. 현재30Hz에서는 최초 gate 프레임을0으로 두고60 control step 후 종료한다. 별도 유예 시간은 없다.
- 최초 선행 성공 이후 gate가 닫혀도 타이머는 계속 흐른다. 후행이 한 번 성공하면 해당 edge 감시를 해제한다. 환경 reset 시 해당 환경의 타이머만 초기화한다.
- 이미 가까이 있어도 최초 선행 성공 이후 2초 내에 1cm 수평 접근 또는 후행 성공이 필요하다. 한 번 접근하면 이후 등반에 오래 걸려도 이 조건으로 종료하지 않는다. 높이만 변하고 수평 접근도 성공도 없으면 최초 2초 마감에 종료될 수 있다.
- 일반 학습/일반 viewer에는 적용한다. `eval_protocol` 정식 평가에서는 비활성화하여 기존 full-horizon 평가와 비교한다. 정식 평가 종료 조건을 바꾼 성능으로 비교하지 않는다.
- `before.approach_timeout`: `mode: first_approach_v1`, `seconds: 2.0`, `min_improvement_m: 0.01`. checkpoint 계약에 포함되므로 기존 Stage2 checkpoint를 동일 설정 resume로 취급하지 않는다. 기본 전이는 Stage1 PUSH26k다.
- 진단: `termination/approach_timeout_count`, `termination/approach_timeout_per_scene_step`. 후자는 성공률이 아니라 환경-step당 timeout 발생률이다.

구현은 `tokenhsi/env/tasks/multi_agent/before_approach_timeout.py`, 환경 연결은 `humanoid_ma_before_push.py`다. 실행 명령은 [config](../config.md)를 따른다.

## 종료 원인 계측

원인별 count/fraction: `termination/fall_only`, `termination/approach_timeout_only`, `termination/fall_and_approach_timeout`, `termination/horizon_only`와 전체 `termination/all_count`. 분모는 같은 rollout 구간에서 종료된 환경 수다. 여러 agent가 넘어져도 환경당1건, 낙상·timeout 동시 발생은 별도1건이다. 시간 제한과 실패가 동시에 발생하면 실패로 분류한다. 기존 `approach_timeout_count`는 동시 발생을 포함한다.

2026-11-03 16:13 시작된 실행에는 세부 계측이 로드되지 않았다. 과거 낙상/시간제한/동시발생은 복원할 수 없으며 다음 실행부터 적용된다. 현재 학습은 재시작하지 않았다.

## Asym 3m 대기 경계

`stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_radius3m`은 원본 virtual_asym 설정에 `before.asymmetric_progress.wait_radius_m: 3.0`만 추가한다. 선행 current AT success가 false일 때 progress의 경계 R을 고정3m로 사용한다. goal 중심 수평거리 d에 대해 오차는 `max(d-3,0) + 10*max(3-d,0)`이며 progress curve는 기존과 같다. 선행 완료 시 기존 박스 크기/버퍼의 R과 내부 페널티 없는 접근 보상으로 복귀한다.

SIT/CLIMB의 d는 사람 rootXY, ON_TOP은 옮기는 물체 XY 기준이다. state/success는 변경하지 않으며 추가 접근2초 종료는 사용하지 않는다. 기본 낙상·episode horizon 종료는 유지한다. 대기 경계 변경은 정지 속도를 강제하지 않는다. 원본 설정과 체크포인트 계약을 분리했으며 실행 중 학습에는 소급 반영되지 않는다.
