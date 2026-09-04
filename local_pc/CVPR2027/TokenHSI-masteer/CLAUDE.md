# TokenHSI-masteer — Claude 진입점

작업 전에 [공통 운영 규칙](../docs/OPERATIONS.md)을 읽는다.

1. 현재 계획·큐: [PLAN_masteer](../plan/PLAN_masteer.md)
2. 자동 수치: [AUTO_RESULTS](../experiments/AUTO_RESULTS_masteer.md)
3. 검토된 결론: [EXPERIMENTS](../experiments/EXPERIMENTS_masteer.md)
4. 코드 변경 이유: [CHANGELOG](CHANGELOG.md)
5. A=2 이식 함정: [../docs/MULTIAGENT_PORT.md](../docs/MULTIAGENT_PORT.md)

**ma + steer 통합 레포.** `HumanoidMASteerCarry < HumanoidMACarry < HumanoidTrajSitCarryClimb`.
속도 명령은 창의 길이 `M` 이 담는다 — 자세한 것은 `../docs/HANDOFF_steer_time.md`.

전체 연구 목적과 저장소 관계는 `../METHOD_V2.md`, `../REPOS.md`가 기준이다.
