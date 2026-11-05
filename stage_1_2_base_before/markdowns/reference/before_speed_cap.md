# relfusion + 거리별 수평 속도 상한

실험: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_speed_cap`.

사용자 요청: 목표 속도를 따라가게 하지 않고, 상한을 넘는 속력에만 비례 벌점. 선행 완료 후에는 규제 해제. 이후 사용자 승인으로 GPU2 학습을 시작했다.

## 비용

- R=지지박스 XY반대각선+3m, d=선행 goal까지 수평 거리.
- speed_limit=clip((d−R)/2초, 0, 0.5m/s).
- 속도 보상=−1×max(실제 XY속력−speed_limit, 0).
- 상한 이하이면 정지/저속/역방향/옆방향도 이 항은0. 방향 보상 없음. 접근 유인은 기존 progress가 담당한다.
- 경계1m이상 밖은 상한0.5, 마지막1m에서 선형으로0까지 감소, 경계 안에서는 상한0.
- 선행 최초 AT성공 후 비용을 완전히 해제하며, success가 다시 false가 되어도 release latch 유지. 목표속도/최종스킬 접근 비용 없음.

SIT/CLIMB는 사람 root XY 위치·속도, ON_TOP은 운반 박스 XY 위치·속도 사용. ON_TOP은 최초 자기 HOLDING 성공 전 비용0, 성공 후 활성 latch, 놓쳐도 유지. 부분 reset은 해당 환경만 해제. 수직속도는 벌점에 포함하지 않는다. 모델/가상박스/progress/state/success는 유지.

`mode: sit_climb_speed_cap_v1`, `after_release: disable`로 기존 목표속도 L2 방식과 계약을 구분한다. helper의 방향벡터 반환값은 API 호환/상한 크기 진단용이며, 이 모드 비용은 방향을 전혀 사용하지 않는다. `goal_buffer`/`arrival_fade_distance`는 기존 옵션 스키마와의 호환을 위해 남아 있지만 release 이후 비용이0이므로 새 비용에는 영향을 주지 않는다.

## 검증과 상태

CPU 테스트는 상한 아래 비용0, 초과량 선형, 방향 독립성, ON_TOP 집기 전0, 선행 완료 후0 및 기존 L2/hard/soft/fusion 회귀를 검사한다. `output_etc/check/relfusion_speed_cap/cpu.log` 참조. 새 speed-cap 학습은 사용자 승인 후 GPU2에서 시작했다(4명/1024환경/seed42). 실행 기록은 해당 output/launch.json 참고. 이전 추종 학습4122617, 수정 중간 학습4168774는 MPS terminate_client로 종료했다. 기존 로그/체크포인트는 보존한다.
