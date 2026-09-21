# Carry collision-avoidance planner

`stack_planner`의 Transformer, 33-point joint path/speed head, decision
history, PPO action distribution, checkpoint format을 그대로 사용한다. 실행 task만
sequential stack이 아닌 기존 simultaneous `HumanoidMASteerCarry` 계열이다.

- 두 agent 모두 `root -> own box -> own goal`을 동시에 수행한다.
- sequential phase, A1 retreat, A2 handoff, stacking reward는 없다.
- pickup(index 16)과 placement(index 32)는 정확한 hard anchor다.
- reward는 frozen ms18의 원래 Carry reward와 task progress를 유지하면서
  agent-agent, agent-other-box, box-box proximity cost를 감점한다.
- 기본 stress distribution은 `MS_SCEN=cross`다.
- reset의 기본 75%는 두 goal이 공통 중심 주위로 수렴하는 hard case, 25%는 일반
  Cross다. goal 간격은 두 box의 circumscribed radius 합 + 0.25 m라 최종 배치는
  가능하지만, 먼저 도착해 정지한 agent/box를 후발 agent가 공간적으로 돌아야 한다.
  비율과 여유는 `CARRY_PLANNER_CONVERGE_PROB`, `CARRY_PLANNER_GOAL_MARGIN`으로 바꾼다.

Smoke 예시:

```bash
CARRY_PLANNER_ENVS=8 CARRY_PLANNER_ITERS=1 \
CARRY_PLANNER_HORIZON=4 CARRY_PLANNER_LOW_STEPS=6 \
MA_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi118 \
  bash TokenHSI-coord/carry_planner/train.sh carry_smoke \
  /path/to/ms18/Humanoid.pth
```

출력은 `runs/carry_planner/<tag>/`에 저장한다. 동일 tag는 덮어쓰지 않는다.

Viewer:

```bash
MA_GPU=0 bash TokenHSI-coord/carry_planner/view.sh \
  runs/carry_planner/<tag>/planner_000200.pth \
  TokenHSI-masteer/output/ms18_maskteam_origscale_c06_s0_00009000.pth
```

기본은 `MS_SCEN=cross`와 viewer-only timed-cross 배치다. 두 agent-box 묶음의
직선·정속 기준 교차점 도착 시각을 맞추고, 각 goal을 같은 교차점 너머에 두므로
회피하지 않으면 실제로 동시에 부딪치는 stress scene이 된다. 로그의
`[coord-view timed-cross]`에서 예상 도착시간 차이를 확인할 수 있다.

6 action-step마다 deterministic mean path를 다시 계획한다. 일반 Cross로 되돌리려면
`MS_VIEW_TIMED_CROSS=0`, free 배치는 `MS_SCEN=free`를 사용한다. 충돌 지점 전후 길이는
`MS_VIEW_TIMED_CROSS_PRE`(기본 2 m), `MS_VIEW_TIMED_CROSS_POST`(기본 4 m)로 조절한다.
`CARRY_PLANNER_REPLAN_STEPS=<N>`으로 replan 주기도 바꿀 수 있다.
