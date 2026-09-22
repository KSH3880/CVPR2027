# Carry collision-avoidance planner

`stack_planner`의 Transformer, decision history, PPO action distribution과 33-point
실행 ABI를 사용한다. 모델은 agent별 고정 `start/box/goal`과 학습된 중간점 4개를
두 cubic Hermite spline으로 펼치고, speed knot 7개도 monotone spline으로 펼친다.
따라서 실행 task만
sequential stack이 아닌 기존 simultaneous `HumanoidMASteerCarry` 계열이다.

- 두 agent 모두 `root -> own box -> own goal`을 동시에 수행한다.
- sequential phase, A1 retreat, A2 handoff, stacking reward는 없다.
- pickup(index 16)과 placement(index 32)는 정확한 hard anchor다.
- 위치 action은 agent당 중간점 4개의 XY(8D), speed action은 7개 knot이며 최종
  path/speed는 기존과 같은 33 point다. 중간점은 직선의 dense residual이 아니고
  `CARRY_PLANNER_CONTROL_SCALE`(기본 4m) 범위에서 독립적으로 움직인다.
- reward는 frozen ms18의 원래 Carry reward와 task progress를 유지하면서
  agent-agent, agent-other-box, box-box proximity cost를 감점한다.
- 기본 stress distribution은 `MS_SCEN=cross`다.
- reset의 기본 75%는 각 `box -> goal` 선분이 동일한 crossing을 지나도록 goal을
  crossing 너머에 두는 hard case, 25%는 일반 Cross다. goal 간격은 두 box의
  circumscribed radius 합 + 0.25 m라 최종 배치는 가능하지만, 먼저 도착해 정지한
  agent/box를 후발 agent가 공간적으로 돌아야 한다.
  비율과 여유는 `CARRY_PLANNER_CONVERGE_PROB`, `CARRY_PLANNER_GOAL_MARGIN`으로 바꾼다.

학습 기본 sparse-point exploration은 `CARRY_PLANNER_DELTA_STD=0.25`다. replan에서는
이미 실행한 prefix만 보존하고 아직 실행하지 않은 suffix는 새 spline으로 교체한다. 로그의
`sample_path_deviation`은 실제 실행용 stochastic path가 직선 base에서 벗어난 평균
거리(m), `mean_path_deviation`은 viewer/checkpoint에서 보이는 deterministic path의
평균 거리다. `plan_valid_fraction`이 낮으면 곡률 검사에서 proposal이 거부되어 이전
plan 또는 analytic fallback이 실행되고 있다는 뜻이다.

`CARRY_PLANNER_SMOOTHNESS_COEF`(기본 10)는 직선 이탈이나 일정한 곡률을 벌점으로
주지 않고, pickup에서 분리한 두 spline 각각의 **curvature 변화량**을 줄인다. 따라서
넓게 한 방향으로 우회하는 path는 유지하면서 좌우로 흔들리는 S-curve/noise를 억제한다.
46도 초과 급회전은 아래 analytic curvature term이 별도로 처리한다.

학습은 physical PPO에 더해 episode의 첫 full-plan을 96개 미래 시점으로 펼치고,
충돌 위험이 큰 top-8 시점의 agent-agent, agent-box, box-box overlap을 직접
최소화한다. 이 auxiliary loss의 speed는 timing 계산에만 사용하고 detach하므로
gradient는 path에만 간다. 계수와 focus 수는
`CARRY_PLANNER_ANALYTIC_COLLISION_COEF`(기본 1),
`CARRY_PLANNER_ANALYTIC_FOCUS_STEPS`(기본 8)로 조절한다.
46도 실행 곡률 제한은 `CARRY_PLANNER_ANALYTIC_CURVATURE_COEF`(기본 20)의
cosine-space penalty로 함께 학습한다.

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

학습 없이 sparse spline과 frozen executor 연결만 확인하려면 제공된 V16 sanity
checkpoint를 사용한다. A1은 +Y, A2는 -X로 약 1m 우회하고 pickup/crossing 근처에서
서로 다른 speed dip을 사용하므로 path ribbon과 speed color를 함께 확인할 수 있다.

```bash
MA_GPU=0 bash TokenHSI-coord/carry_planner/view_sanity.sh
```

checkpoint를 다시 만들 때는 기존 파일을 자동으로 덮어쓰지 않는다.

```bash
PYTHONPATH=TokenHSI-coord python \
  TokenHSI-coord/carry_planner/make_sanity_checkpoint.py \
  runs/carry_planner/sanity_sparse_v16/planner_detour_slow.pth
```

기본 viewer는 평가 6회 후 종료하지 않고 창을 닫을 때까지 계속 실행하며, reset마다
학습과 같은 randomized `mixed` 분포(기본 75% close-goal, 25% 일반 Cross)를 새로
뽑는다. 실행 episode 상한은 사실상 무한대인 `CARRY_PLANNER_VIEW_GAMES=1000000000`이며
필요하면 덮어쓸 수 있다.

고정된 collision stress test가 필요할 때만 `MS_VIEW_TIMED_CROSS=1`을 지정한다. 이
모드에서는 두 agent-box 묶음의 직선·정속 기준 교차점 도착 시각을 맞추며, 로그의
`[coord-view timed-cross]`에서 예상 도착시간 차이를 확인할 수 있다.

6 action-step마다 deterministic mean path를 다시 계획한다. free 배치는
`MS_SCEN=free CARRY_PLANNER_CONVERGE_PROB=0`을 사용한다. 충돌 지점 전후 길이는
`MS_VIEW_TIMED_CROSS_PRE`(기본 2 m), `MS_VIEW_TIMED_CROSS_POST`(기본 4 m)로 조절한다.
`CARRY_PLANNER_REPLAN_STEPS=<N>`으로 replan 주기도 바꿀 수 있다.

다양한 episode 정량 평가:

```bash
MA_GPU=1 bash TokenHSI-coord/carry_planner/eval_suite.sh \
  runs/carry_planner/<tag>/planner_000200.pth \
  TokenHSI-masteer/output/masteer/<executor>.pth 64 0 1 2
```

각 seed에서 학습 분포 `mixed`(75% close-goal), `converge` 100%, 일반 `cross`,
비정형 `free`를 각각 기존 evaluator의 3회 반복으로 측정한다. 원시 log/npy와 JSON은
`runs/results/carry_planner/<tag>/`에 저장되며 `both_place`, pair collision,
minimum distance, command-speed 사용률, invalid/curve rate를 함께 보고한다. 한 분포만
보려면 `eval_one.sh <planner> <executor> <profile> [envs] [seed]`를 사용한다.
