# BEFORE virtual wait_penalty_v2 base

사용자 지정 원본: `/home/hwanhee/ksh/stage_1_2_base_1031/output/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2`.
ksh는 읽기만 했으며 코드·YAML·6500 checkpoint를 로컬로 복사했다. 출처와 SHA256은 `output_etc/check/import_wait_penalty_v2/provenance.json`에 보존한다.

새 설정은 원본과 같은 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2`다. virtual box 유지, asym/연속progress/state정규화/접근timeout 없음. Stage1 PUSH26k로부터4명·1024환경·8객체·seed42로 학습한다. 가져온 Stage2 6500은 호환성 검사와 참조용이며 본학습 재개 지점이 아니다.

- `waiting_penalty.radius_buffer=3.0`, `coefficient=1.0`.
- 대기 범위는 goal 중심 **박스 XY 반대각선 +3m**이며 고정 반경3m가 아니다.
- SIT/CLIMB는 사람, ON_TOP은 payload의 수평거리로 진입을 판정한다.
- 범위에 한 번 진입하면 선행 완료 전까지 armed 상태를 유지한다. 다시 바깥으로 나가도 해제되지 않는다.
- 벌점은 후행 사람 root의 **XYZ 속도 norm ×1.0**. 정지하면0이다.
- 선행 AT가 한 번 성공하면 reset 전까지 벌점을 해제한다. 별도 waiting 조기 종료는 없다.
- AT 성공 높이 허용치가 기존0.001m에서 **0.2m**로 바뀐다. state/success gate는 current AT success이며 벌점 해제만 latch된다.

원본 단일 `BeforeWait` 코드에는 termination 기능도 포함되지만 이 설정은 penalty 경로만 연결하며 해당 termination 기능을 호출하지 않는다. 이전 strict-height 모델과 보상/성공값을 비교할 때 이 설정 차이를 명시한다.
