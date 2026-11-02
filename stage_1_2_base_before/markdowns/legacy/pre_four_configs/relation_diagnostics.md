# Stage1 지표와 trace

학습의 `relation/edge/{holding,at,ontop,sit,climb}/`은 primitive edge의 현재 기하·보상·성공이다. Carry 완수는 HOLDING과 placement의 관계를 포함하므로 task 완수율과 primitive 성공을 구분한다.

- 보상: `state_component`·`progress_component`·`success_component`를 자기 edge별로 합산한다. 동료 보상은0이다. placement 성공 시 연결된 HOLDING도 보상 포화된다.
- 패널티: 사람 root XY 상대 위치/속도의 CPA 위험, graph 담당 상자의 속도, power를 별도로 기록한다. AMP는 학습 reward 조합 단계에서 더해진다.
- `sampling/`: 실제 reset task·skill·물리 재추첨·RSI fallback과 reference/첫 step 성공. 설정의 명목 비율과 실제 비율을 구분한다.
- `ontop/`, `sit/`, `climb/`: 상판 영역·높이·접촉·유지 진단이다. 보상을 추가하는 항은 아니다.
- `diagnostics/edge_context_steps.csv`: 제한된 환경의 step·episode·edge·owner·논리/물리 binding·기하·보상 trace다.

평가 JSON의 `ever_agent_goal_success`는 에피소드 중 한 번이라도 해당 사람의 required goal을 달성한 비율이고 `final_agent_goal_success`는 종료 시 성공이다. scene 지표는 같은 시점에 장면의 모든 required goal을 만족하는지를 검사한다. 낙상·timeout으로 종료되며, 성공만으로 조기 종료하지 않는다. `stage1_sampling_mode`를 확인해 구 근거리 snapshot과 TokenHSI 거리 분포를 구분한다.

실행법은 [config.md](config.md)를 따른다.
