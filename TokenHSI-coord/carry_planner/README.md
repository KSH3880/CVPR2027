# Carry collision-avoidance planner

stack_planner의 Transformer, decision history, PPO action distribution과 33-point
실행 ABI를 사용한다. 새 Carry decoder는 episode-fixed 전체 경로를 다시 만들지 않고
매 decision마다 현재 측정 위치부터 남은 task suffix를 생성한다.

- 두 agent 모두 root → own box → own goal을 동시에 수행한다.
- pickup 전에는 current→box와 box→goal spline을 합치고, pickup 후에는
  current→goal spline만 생성한다.
- box는 고정 index가 아니다. 두 leg의 실제 arc length로 정한 dynamic index에 정확히
  삽입하며 goal은 항상 마지막 point다.
- 기본 implicit decoder는 agent당 위치 latent 8D와 속도 latent 7D를 출력한다.
  공유 네트워크가 진행률에 따른 연속 XY/속도 곡선을 생성하고, 접근·운반 구간을
  실제 arc length로 재샘플링해 기존 33-point executor ABI에 맞춘다.
  root/box/goal은 정확한 anchor이며 학습 제어점의 1:3 배분은 없다.
- 연속 곡선의 각 XY축 변위 상한은 CONTROL_SCALE이고 남은 직선거리로 축소되지 않는다.
  `CARRY_PLANNER_IMPLICIT_CURVE=0`은 기존 4-point sparse decoder를 재현한다.
  두 decoder의 checkpoint는 config가 구별하며 서로 resume할 수 없다.
- accepted replan은 executor cursor를 새 current-root path의 시작으로 reset하며,
  지나온 prefix를 새 spline에 접합하지 않는다.
- sequential phase, A1 retreat, A2 handoff, stacking reward는 없다.
- reward는 frozen ms18 실행에 대한 환경 Carry reward와 task progress를 유지하면서
  agent-agent, agent-other-box, box-box proximity cost를 감점한다.
  progress는 12개 low-level step 전후의 남은 task 거리 감소량이다. pickup 전에는
  root→box + box→goal, pickup 후에는 box→goal, 목표에 내려놓으면 0으로 계산해
  pickup/putdown 전환에서 거리가 인위적으로 뛰지 않게 한다.
  `CARRY_PLANNER_PROGRESS_COEF`(기본 2.0)으로 progress reward의 배율만
  바꿀 수 있다. 작은 우회가 직선 진척을 일시적으로 줄이는지 확인할 때 사용한다.
- 기본 stress distribution은 `MS_SCEN=cross`다.
- reset의 기본 75%는 각 `box -> goal` 선분이 동일한 crossing을 지나도록 goal을
  crossing 너머에 두는 hard case, 25%는 일반 Cross다. goal 간격은 두 box의
  circumscribed radius 합 + 0.25 m라 최종 배치는 가능하지만, 먼저 도착해 정지한
  agent/box를 후발 agent가 공간적으로 돌아야 한다.
  비율과 여유는 `CARRY_PLANNER_CONVERGE_PROB`, `CARRY_PLANNER_GOAL_MARGIN`으로 바꾼다.
- 학습 시작 시 모든 env가 같은 age=0에서 시작해 600-step timeout이 주기적으로
  몰리지 않도록 첫 에피소드의 종료 시점만 env별 3~600 step에 균등하게 분산한다.
  실제 `progress_buf`는 조작하지 않으며, 그 다음 에피소드부터는 원래 600-step
  제한을 그대로 쓴다. 초기 중복 reset은 첫 deadline을 다시 뽑지 않는다.

학습 기본 latent exploration은 CARRY_PLANNER_DELTA_STD=0.10이다.
매 proposal은 current-root anchored remaining suffix이며, 이전 path는 Transformer
context로만 입력된다. sample_path_deviation과 mean_path_deviation은 각각 stochastic
path와 deterministic mean path가 현재의 직선 two-leg/direct reference에서 벗어난
평균 거리다. proposal이 hard validator에서 거절되면 이전 valid path 또는 analytic
fallback이 계속 실행된다.

`CARRY_PLANNER_SMOOTHNESS_COEF`(기본 10)는 직선 이탈이나 일정한 곡률을 벌점으로
주지 않고, pickup에서 분리한 두 spline 각각의 **curvature 변화량**을 줄인다. 따라서
넓게 한 방향으로 우회하는 path는 유지하면서 좌우로 흔들리는 S-curve/noise를 억제한다.
46도 초과 급회전은 아래 analytic curvature term이 별도로 처리한다.

hard gate에서 거절된 sampled proposal에는 거절 원인이나 turn 크기와 무관하게
`CARRY_PLANNER_INVALID_PLAN_COEF`(기본 0.25)를 macro reward에서 직접 감점한다.
최초 거절 시 analytic fallback, 이후 거절 시 이전 valid path가 실행되더라도 거절 action이
fallback 실행 reward를 그대로 받지 않도록 하는 PPO credit penalty다. curve 초과량에
따라 penalty를 키우면 직선 선호가 강해지므로 hard gate와 학습 신호의 역할을 분리한다.
로그의 `invalid_plan_penalty`는 전체 proposal당 실제 평균 감점값이다.

거절 원인 디버깅을 위해 전체 metric에는 finite/buffer/speed/curve predicate 통과율,
평균 최대 turn과 path 길이를 기록한다. 콘솔의 `plan_curve_fraction`은 실제 46도 gate
통과율이다. `plan_zero_turn_false_reject_fraction`은 길이 0인 인접 segment를 공통
validator처럼 turn 검사에서 제외했을 때만 살아나는 proposal 비율이며, 0보다 크면
Carry validator의 degenerate-segment 오거절 가능성을 뜻한다. 콘솔은 sampled proposal과
noise-free mean path의 curve 통과율·평균 최대 turn을 따로 출력하므로 exploration noise와
mean decoder 문제를 구분할 수 있다. 진단 단계에서는 실제 수락/거절 동작을 바꾸지 않는다.

학습은 physical PPO에 더해 current-root anchored suffix를 96개 미래 시점으로
펼치고, 충돌 위험이 큰 top-8 시점의 agent-agent, agent-box, box-box overlap을 직접
최소화한다. suffix에는 지나온 prefix가 존재하지 않으며 dynamic box index의 도착시간을
pickup 시점으로 사용한다. executor 도착시간 오차에도 우회하도록 기본 ±1.5초 구간을 7개
상대 timing offset으로 검사한다. forward loss 값은 기존처럼 offset 중 worst case를
유지하지만, backward는 detached-softmax weight로 모든 offset의 gradient를 합친다. 따라서
정확한 overlap 한 점의 무방향 gradient에 갇히지 않으면서 좌/우 회피 방향은 지정하지
않는다. 이는 96×96 모든 시간쌍을 만들지 않아 minibatch 메모리는 미래 시점 수에 선형이다.
auxiliary loss의 speed는 timing 계산에만 사용하고
detach하므로 gradient는 path에만 간다. 계수와 관련 손잡이는
`CARRY_PLANNER_ANALYTIC_COLLISION_COEF`(기본 1),
`CARRY_PLANNER_ANALYTIC_FOCUS_STEPS`(기본 8),
`CARRY_PLANNER_ANALYTIC_TIME_UNCERTAINTY`(기본 1.5초),
`CARRY_PLANNER_ANALYTIC_TIME_SAMPLES`(기본 7)로 조절한다.
46도 실행 곡률 제한은 `CARRY_PLANNER_ANALYTIC_CURVATURE_COEF`(기본 20)의
cosine-space penalty로 함께 학습한다.

suffix path는 replan마다 시작점 자체가 달라 point index가 서로 대응하지 않으므로,
legacy dense-point consistency loss는 기본 0으로 비활성화한다. 대신 시작점은 현재
measured root로 hard anchor되고 진행 방향 constraint가 첫 suffix tangent를 제한한다.
CARRY_PLANNER_EXCESS_LENGTH_COEF(기본 0.05)는 현재 suffix 길이가 직접
root→box→goal(held 이후 root→goal) 거리의 1.15배에 0.25m를 더한 허용 길이를 넘을
때만 적용한다. 초과거리는 0.5m Huber transition 뒤에도 선형 gradient를 유지한다.

replan 순간 실제 이동 방향과 새 path 접선이 불연속이 되지 않도록
`CARRY_PLANNER_DIRECTION_COEF`(기본 0.10)를 별도로 적용한다. 현재 root에서 새 future
path를 arc-length 0.5m 진행한 방향과 실제 `root_vel_xy` 방향을 비교하며 15도까지는
무료다. 0.2m/s 이하에서는 꺼지고 0.8m/s까지 선형으로 강해진다. 이는 물리적 실행
가능성 제약이라 collision-risk gate로 끄지 않지만, 다른 geometry term과 같이 첫 5
iteration에 warm-up한다. lookahead, free angle, 속도 범위는 각각
`CARRY_PLANNER_DIRECTION_LOOKAHEAD`, `CARRY_PLANNER_DIRECTION_FREE_ANGLE_DEG`,
`CARRY_PLANNER_DIRECTION_MIN_SPEED`, `CARRY_PLANNER_DIRECTION_FULL_SPEED`로
조절한다. pickup 직후처럼 `held=1`인데 실제 속도가 낮으면 직전 committed path의
carry 방향을 fallback reference로 사용하고, 가속하면서 실제 velocity 방향으로 부드럽게
전환한다. fallback은 goal까지 lookahead 이상 남았을 때만 켜며 기본 상대 강도 0.5는
`CARRY_PLANNER_PICKUP_DIRECTION_FALLBACK_WEIGHT`로 조절한다. 로그의
`direction_loss`, `weighted_direction_loss`, `mean_direction_error_deg`,
`direction_active_fraction`, `direction_fallback_fraction`으로 동작을 확인한다.
시간/makespan loss는 speed 최대화와 collision timing 악용을 피하기 위해 넣지 않는다.

Smoke 예시:

```bash
CARRY_PLANNER_ENVS=8 CARRY_PLANNER_ITERS=1 \
CARRY_PLANNER_HORIZON=4 CARRY_PLANNER_LOW_STEPS=12 \
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

학습 없이 suffix spline과 frozen executor 연결만 확인하려면 제공된 sanity
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

기본 12 action-step마다 deterministic mean suffix를 다시 계획한다. free 배치는
`MS_SCEN=free CARRY_PLANNER_CONVERGE_PROB=0`을 사용한다. 충돌 지점 전후 길이는
`MS_VIEW_TIMED_CROSS_PRE`(기본 2 m), `MS_VIEW_TIMED_CROSS_POST`(기본 4 m)로 조절한다.
`CARRY_PLANNER_REPLAN_STEPS=<N>`으로 replan 주기도 바꿀 수 있다.

학습 중 hard rejection을 그대로 재현하려면 stochastic proposal을 켠다. 거절된
raw proposal은 다음 replan까지 A1 빨강/A2 주황 선으로 표시되고, 기존 밝은 ribbon은
실제로 설치되어 executor가 따라가는 path로 남는다. checkpoint에 저장된
action_log_std를 우선 사용하므로 학습 exploration 분포를 그대로 확인할 수 있다.

~~~bash
CARRY_PLANNER_VIEW_STOCHASTIC=1 CARRY_PLANNER_VIEW_SEED=0 \
MA_GPU=0 bash TokenHSI-coord/carry_planner/view.sh \
  runs/carry_planner/<tag>/planner_000200.pth \
  TokenHSI-masteer/output/masteer/<executor>.pth
~~~

CARRY_PLANNER_DRAW_REJECTED=0으로 overlay만 끌 수 있다. 콘솔에는 각 replan의
valid, reason, max_turn_deg가 같이 출력된다.


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
