# 먼 거리 접근 하한 → 제곱 속도 추종 전환

실험: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_velocity_l2_squared_floorblend`.

기존 relfusion 제곱 속도오차 +3m감속을 유지하되, 먼 구간은 정확한 속도 추종 대신 goal 방향 접근속도 하한만 적용한다. k1, 하한0.5m/s, 대기반경R=지지박스XY반대각선+3m.

s=max(goal까지 수평거리−R,0), 방향u=(goal−위치)/거리, 접근속도vr=vXY·u.

- s≥4m: E_floor=max(0.5−vr,0)^2. vr≥0.5면 더 빨라도 비용0. 옆으로만 움직이면 vr0이라 비용0.25.
- 4→3m: x=clip((4−s)/1,0,1), w=x²(3−2x). 비용=(1−w)E_floor+w E_track. 추종 목표는 이 구간에서0.5m/s.
- 3→0m: w1, 목표속도=(s/6초)u로0.5→0 선형감속, E_track=norm(vXY−vTarget)^2.
- 경계 안: 목표0, 수평속력 제곱 비용. 방향별 강제 대기점은 없음.
- 속도 보상은 비용에−k를 곱한다. 동일 속도에서4m와3m 전환은 연속. 3m에서 감속식의 기울기 전환은 남아 있다.

SIT/CLIMB root XY, ON_TOP은 최초 자기 HOLDING성공 후 운반 박스 XY. ON_TOP 집기 전0, 활성은 episode latch. 선행 최초 AT성공 후 after_release=disable로 모든 속도비용0. 기존 모델/progress/state/success/종료조건 유지.

먼 구간에서는 하한을 만족하는 한 횡방향 성분도 별도로 벌주지 않는다. 상한이 없으므로 매우 빠르게 접근해도 먼 구간에서 비용이 없으며, 4→3m 전환에서 감속을 유도한다. 이 전환이 실제 보행에 충분한지는 학습 후 평가 대상이다.

새 `far_approach_blend_width: 1.0` 옵션을 제곱오차/release-off 조합에만 허용한다. 기존 저장 config에 옵션이 없으면 이전 제곱 추종 유지. 체크포인트 계약 분리. 학습 초기화는 기존과 같은 Stage1 PUSH26k fresh transfer.
