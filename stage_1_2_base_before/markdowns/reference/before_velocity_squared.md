# relfusion + 제곱 속도 오차 + 3m 감속

설정: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_velocity_l2_squared`.

보상은 −k[(vx−vx_target)^2+(vy−vy_target)^2], k=1. 방향도 포함하는 속도 추종이며 속도 상한 방식이 아니다. 기존 L2 norm(루트 포함)과 구별하기 위해 mode=sit_climb_l2_squared_v1로 checkpoint 계약을 분리한다.

R=지지박스 XY반대각선+3m, d=선행 goal까지 수평거리. 목표속력=clip((d−R)/6초,0,0.5m/s). 최대속력0.5×감속시간6=감속구간3m. R+3m 바깥에서는0.5 유지, R+3m→R에서0.5→0으로 선형감속. 시간에 따라 강제로 속력을 감소시키는 것이 아니라 현재 거리에 따른 목표값이다.

SIT/CLIMB는 root XY, ON_TOP은 최초 자기 HOLDING 성공 후 운반 박스 XY 위치·속도로 계산한다. ON_TOP은 HOLDING 상실에도 latch 유지. 선행 최초 AT성공 후 after_release=disable로 속도항 완전히0, 재닫힘에도 해제 유지. progress/state/success 및 모델구조는 기존 relfusion과 동일하다. 별도 접근early term이나 asym 없음.

기존 선형 오차 모델과 비교해 계수는1로 유지했으므로0.5m/s오차 비용은0.5→0.25로 줄었다. 작은 잔여 움직임 억제도 약해질 수 있다. 이전 학습·로그는 보존하며 새 Stage1 PUSH26k 전이 학습이다. CPU102 passed(3m 감속, 제곱 오차, 집기 활성, 완료해제, 기존 회귀). 실행 metadata는 해당 output/launch.json.
