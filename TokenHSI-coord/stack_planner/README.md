# Stack trajectory planner V1

Stack task 전용 Transformer planner다. 기존 `coordinator/`와
`trajectory_predictor/`의 코드 및 checkpoint namespace를 건드리지 않도록 별도 패키지로
분리했다.

## 현재 계약

- 입력: 기존 `coordinator.schema.CoordinatorState`와 완전히 동일한 10개 state tensor
- backbone: 6개 `(root, box, goal) x 2 agents` entity token Transformer encoder
- candidate 생성: learned query를 쓰는 Transformer decoder
- 출력: 기존 `JointCoordinator`와 동일한 joint XY path, speed, acceleration,
  pickup dwell, value/risk diagnostics + A1의 물리적 `retreat_path/speed`
- hard anchor: path의 `P0=root`, `P16=box`, `P32=goal`
- checkpoint schema: `tokenhsi-stack-planner-v1`

planner 입력에는 virtual box를 넣지 않는다. 현재 Carry executor에만 필요한 virtual box는
`env_adapter.py`가 learned retreat endpoint에서 만들어 관측 직전에 변환한다. 이후 steering과
Carry가 분리되면 이 adapter만 제거하고 planner의 물리 경로 출력은 유지할 수 있다.

```python
from stack_planner.model import StackTrajectoryPlanner

model = StackTrajectoryPlanner()
output = model(coordinator_state)
path = output["path_world"]   # [B, K, 2, 33, 2]
speed = output["speed"]      # [B, K, 2, 33]
retreat = output["retreat_path_world"]  # [B, K, 2, 33, 2]
```

검증:

```bash
python -m unittest discover -s stack_planner/tests -v
```

## Planner macro reward

`reward.py`는 기존 agent/task reward와 분리된 planner 전용 reward다. 한 planner
trajectory를 frozen executor가 일정 low-level step 동안 실행한 뒤, 실행 전후의
`StackPhysicalState`와 구간 누적 `StackIntervalCosts`를 넘긴다.

```python
from stack_planner.reward import compute_stack_planner_reward

terms = compute_stack_planner_reward(before, after, interval)
planner_reward = terms["total"]
```

potential은 `bottom quality -> bottom quality * clearance -> bottom quality *
clearance * top quality` 순서로 dependency를 구성한다. 고정 retreat 방향, GT path,
virtual box 및 legacy Carry observation은 reward 입력에 포함하지 않는다. `collision`과
`bottom_disturbance`는 짧은 접촉을 놓치지 않도록 planner macro interval 동안 simulator에서
누적해 전달한다. `humanoid_fall`은 reset 전에 두 agent 중 하나라도 넘어졌는지를 latch한
0/1 값이며, frame마다 반복하지 않고 macro transition에 기본 `-3.0`을 한 번 적용한다.

초기 접근 구간도 학습되도록 실제 root-box 거리의 bounded rational quality를 쓰며, 이 항은
placement quality가 올라갈수록 자동으로 0에 가까워져 배치 뒤 retreat를 방해하지 않는다.
top 목표는 stage flag가 아니라 현재 bottom box의 물리 pose/크기에서 직접 계산한다.

## Closed-loop PPO

`train_closed_loop.py`는 sequential-stack agent checkpoint를 inference-only로 고정하고 planner
parameter와 action log-std만 최적화한다. 한 PPO transition은 기본 6 low-level step이다.

```bash
STACK_PLANNER_ENVS=64 STACK_PLANNER_ITERS=200 \
  bash TokenHSI-coord/stack_planner/train.sh stack_planner_v0_s0
```

metric, sidecar, planner checkpoint는 `runs/stack_planner/<tag>/`에만 저장된다.

## Checkpoint viewer

학습된 planner의 mean action을 deterministic하게 실행하는 별도 viewer task다. frozen agent
checkpoint도 학습 때 쓴 파일을 같이 지정해야 한다. planner는 기본 6 low-level step마다,
그리고 stack phase가 바뀌는 즉시 다시 계획한다. 학습용 task나 기존 coordinator viewer를
수정하지 않는다. 바닥 띠는 두 agent의 전체 planner path와 구간별 속도를, 허리 높이 띠는
현재 frozen agent에 들어가는 steering window를 표시한다.

```bash
bash TokenHSI-coord/stack_planner/view.sh \
  runs/stack_planner/<tag>/planner_000010.pth \
  TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00011000.pth
```

로컬 desktop에 Isaac Gym 창을 직접 띄우며 noVNC를 사용하지 않는다. 기본 물리 GPU는 0이고
`MA_GPU=1`처럼 바꿀 수 있다. 현재 `DISPLAY`를 사용하고 값이 없으면 `:0`을 쓴다. 여러
scene을 보고 싶으면 세 번째 인자에 env 수를 주고, 재계획 주기는
`STACK_PLANNER_REPLAN_STEPS`로 바꿀 수 있다.
