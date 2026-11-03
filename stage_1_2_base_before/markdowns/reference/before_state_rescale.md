# Virtual asym 기반 추가

현재 추가 학습 설정은 `stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_state_rescale`이다.
원본 virtual_asym_goal5_goalroute에서 state 구간 정규화만 추가한다. 가상박스 GTA 위치와
`asymmetric_progress.lambda_inner=10`을 유지하고 progress/success/모델/샘플링은 변경하지 않는다.
wait_clear는 포함하지 않는다. Stage1 PUSH26k에서 새 Stage2를 시작하며 기존 비가상 학습은 유지한다.
아래 정규화 공식·예외·기준값 수명은 세 기반에 공통이다.

# BEFORE state 보상의 개방 구간 정규화

앞서 선택한 비가상 실험: `stage2_before_push_near50_pair50_source5_goal5_goalroute_state_rescale`.
비가상 `stage2_before_push_near50_pair50_source5_goal5_goalroute`에서 state 정규화만 추가한다.
가상박스 GTA 이동은 사용하지 않는다. Stage1 PUSH26k에서 새 Stage2를 학습하는 설정이다.

비교용으로 앞서 만든 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_state_rescale`도 보존한다.
두 설정 모두 각 원본 goalroute에서 `before.state_rescale: first_open_affine_v1`만 추가한다.
wait_clear/asym은 포함하지 않는다. 기존 실험 및 체크포인트의 보상은 바뀌지 않는다.

기존 기하 state 함수 `phi=f(x)`와 성공 시 state=1로 포화하는 규칙은 유지한다.
그 결과를 `s`라고 할 때 후행의 첫 유효 gate 개방 reward step에서 `s0`를 저장한다.
이후 해당 후행 edge의 state 보상만 다음처럼 계산한다.

```
state_component = state_reward_weight * gate * clip((s - s0)/(1 - s0), 0, 1)
```

기존 비선형 함수를 새 함수로 바꾸지 않고 출력 구간만 affine 변환한다.
따라서 활성 구간의 곡선 형태는 유지되지만 기울기/곡률의 크기는 `1/(1-s0)`배다.
`s < s0` 구간은 0으로 잘린다. 시간 ramp 및 프레임별 보상 차분이 아니며,
같은 상태를 유지하면 같은 보상을 받는다. 상태 개선은 다른 agent의 물체 이동으로도 발생할 수 있다.
목표/성공 포화 `s=1`에서 기존 state 최대값(현재 0.2)에 도달한다.

- 기준값은 환경·edge별로 저장하며 첫 개방 뒤 고정한다. gate 재잠금 시 보상은 0,
  재개방 시 기존 기준값을 재사용한다. 새로운 개방마다 0으로 재시작하지 않는다.
- 에피소드 reset 시 해당 환경의 모든 기준값/개방 표시를 지운다. 새 그래프의 edge 순서와
  이전 에피소드 기준값이 섞이지 않는다. reset 상태 진단과 components 조회는 기준을 기록하지 않는다.
- 이미 `s0=1`인 경우 남은 구간이 없으므로 기존 state 값을 사용한다. 이 경우 개방 보상은
  0이 아니라 기존 최대값이다. 시간만 지나서 최대값을 지급하는 예외를 만들지 않는다.
- `s0<1`이면 매우 작은 양의 분모도 그대로 쓴다. epsilon으로 제한하면 최대 보상이 낮아지므로
  적용하지 않는다. 목표에 아주 가까운 개방은 작은 상태 변화에도 보상이 민감할 수 있다.
- progress, success bonus, 성공 판정, 선행/단독/HOLDING/PUSH 보상은 유지한다.
  success bonus가 즉시 켜지는 현상까지 없애는 변경은 아니다.
- 체크포인트 계약에 옵션을 저장한다. 기존 Stage2 체크포인트를 이 실험으로 resume하지 않는다.
  기본 Stage1 PUSH 26k 전이는 기존 virtual과 동일하다. 실행 명령은 [config](../config.md).

검증: `tokenhsi/tests/test_before_state_rescale.py`는 개방·정지·개선·퇴행·최대값·재잠금·
부분 reset·퇴화구간·기존 항 보존·계약 분리를 검사한다.
`check_before_state_rescale.py`는 정책을 로드하지 않고 zero action으로 실제 환경 연결을 검사한다.
성능 개선 여부는 별도 학습·평가가 필요하다.
