# wait_penalty_v2 soft entry 1m

실험: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_soft_entry1m`.
원본 v2와 다른 설정은 `before.waiting_penalty.soft_entry_width: 1.0`뿐이다. ksh 원본은 수정하지 않는다.

경계 R은 기존처럼 지지 박스 XY 반대각선+3m다. 경계 바깥1m에서 시작하여 경계에 도달할 때까지 속도벌점 강도를 smoothstep으로 올린다.

`u=clip((R+1-d)/1,0,1)`, `w=3*u²-2*u³`.

| 현재 거리(최초 진입 중) | 벌점 강도 |
|---|---:|
| R+1m 이상 | 0 |
| R+0.75m | 0.15625 |
| R+0.5m | 0.5 |
| R+0.25m | 0.84375 |
| R 이하 | 1 |

실제 벌점은 `coefficient × max_episode_so_far(w) × 후행 root XYZ 속력`이다. 이미 경계 안에서 시작하면 기존과 동일하게 즉시100%다. 시간에 따라 강도를 올리지 않는다. 같은 위치에 머물면 강도가 유지되며, 물러나도 이미 적용된 강도를 낮추지 않아 기존의 진입 latch 의미를 보존한다. 이전에 들어갔다 나온 경우에는 위 표의 현재 거리보다 강도가 높을 수 있다.

SIT/CLIMB 진입거리는 사람 rootXY, ON_TOP은 payloadXY다. 선행 AT 최초 성공 시 기존처럼 즉시 벌점을 해제하고 reset까지 해제 상태를 유지한다. 이번 변경은 **진입 경계의 불연속만** 완화하며, 선행 성공 순간 해제의 불연속은 유지한다. state/progress/success, 높이 허용0.2, AMP, 기본 낙상 종료는 동일하고 추가 early termination은 없다. 이 변경만으로 학습된 정책이 더 잘 멈춘다는 보장은 없으며 학습 후 검증해야 한다.

코드 `before_wait.py`, CPU `test_before_soft_wait_entry.py`, 물리 연결 검사 `check_before_soft_wait_entry.py`. 옵션이 없는 원본 v2는 기존 hard entry 경로를 그대로 사용한다. 실행 중인 기존 viewer에는 적용되지 않는다.
