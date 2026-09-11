# Stack path planner V5

Stack task 전용 Transformer planner다. 기존 `coordinator/`와
`trajectory_predictor/`의 코드 및 checkpoint namespace를 건드리지 않도록 별도 패키지로
분리했다.

## 현재 계약

- 입력: 기존 `coordinator.schema.CoordinatorState`와 완전히 동일한 10개 state tensor
- backbone: 6개 `(root, box, goal) x 2 agents` entity token Transformer encoder
- candidate 생성: learned query를 쓰는 Transformer decoder
- 모델 출력: 두 agent의 단일 end-to-end joint XY path만
- hard anchor: 각 path의 `P0=root`만 유지
- route constraint: 연속 선분 투영 거리로 `box → stack goal` ordered visit를 학습
- checkpoint schema: `tokenhsi-stack-planner-v5`

planner 입력에는 virtual box를 넣지 않는다. 현재 Carry executor에만 필요한 virtual box는
`env_adapter.py`가 learned retreat endpoint에서 만들어 관측 직전에 변환한다. 이후 steering과
Carry가 분리되면 이 adapter만 제거하고 planner의 물리 경로 출력은 유지할 수 있다.

```python
from stack_planner.model import StackTrajectoryPlanner

model = StackTrajectoryPlanner()
output = model(coordinator_state)
path = output["path_world"]  # [B, K, 2, 33, 2]
```

별도 `retreat_path`, retreat goal, speed, switch output이나 사후 path concatenation은 없다.
각 경로의 첫 점만 현재 root에 hard-anchor되고 나머지 32점은 planner가 한 번에 정한다.
box와 stack goal은 특정 index에 묶지 않고 모든 path segment에 수선의 발을 내려 최소 거리를
구한다. 같은 segment이면 projection fraction, 다른 segment이면 index를 이용해 box가 stack
goal보다 먼저 방문되도록 제한한다. 허용 반경은 기본 0.15 m이고 초과 거리의 제곱을
`STACK_PLANNER_VISIT_PENALTY`로 reward에서 감점한다. frozen Carry의 box task goal 자체는
계속 stack goal이며 retreat endpoint로 바뀌지 않는다.

### Frozen Carry execution bridge

학습 모델과 frozen agent 사이에서만 `execution.py`가 같은 path의 실행 구간을 만든다.

- placement 전: ordered `box → carry goal` projection에서 goal의 arc 위치를 찾고,
  현재 root부터 그 위치까지를 33점으로 보간한다. 마지막 점은 실제 carry goal과 정확히 맞춘다.
- placement 후: 같은 path의 goal 이후 suffix만 33점으로 보간하고, 시작점이 실제 현재 root가
  되도록 suffix 전체를 평행이동한다. 그 끝점을 기존 Carry 호환용 virtual box로 사용한다.
- 두 구간 모두 이후 기존 0.1 m/320-point steering ABI로 다시 resample한다. 실행 속도는
  adapter가 고정 `MAX_SPEED`로 부여하며 모델은 속도를 예측하지 않는다.

즉 33점은 항상 모델의 한 path다. carry용 33점과 retreat용 33점을 모델이 따로 출력하는
구조가 아니다. 어느 구간을 frozen agent에 설치할지는 simulator의 실제 placement phase를
사용하는 execution 문제이며, planner action이나 학습 head에는 switch가 없다.

검증:

```bash
python -m unittest discover -s stack_planner/tests -v
```

## Planner macro reward

`reward.py`는 기존 agent/task reward와 분리된 planner 전용 reward다. 한 planner
path를 frozen executor가 일정 low-level step 동안 실행한 뒤, 실행 전후의
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
parameter와 action log-std만 최적화한다. 한 PPO transition은 기본 30 action step(30 Hz에서
약 1초) 동안 같은 plan을 유지한다. stack phase 전환은 이 hold를 기다리지 않는다.
다음 replan의 mean path에는 직전 mean plan을 elapsed time만큼 전진시킨
remainder와 맞추는 temporal consistency loss를 기본 적용한다. 단일 33-point path 전체를
비교하며, episode reset과 물리 상태 전환은 mask한다. `STACK_PLANNER_CONSISTENCY_COEF`로
강도를 조절하고 `0`이면 끌 수 있다.
각 episode는 기본적으로 `loco_carry=1.0`으로 두 agent와 두 box를 새로 배치하고 접근
단계부터 시작한다. 기존 carry의 `carryWith`/`putDown` 중간 RSI는 full-stack planner
reset에 섞지 않는다.

```bash
STACK_PLANNER_ENVS=64 STACK_PLANNER_ITERS=200 \
  bash TokenHSI-coord/stack_planner/train.sh stack_planner_v0_s0
```

metric, sidecar, planner checkpoint는 `runs/stack_planner/<tag>/`에만 저장된다.
같은 tag의 디렉터리가 이미 있으면 덮어쓰지 않고 종료한다. 현재 v5 설계의 첫 학습은 다음처럼
실행할 수 있다.

```bash
MA_GPU=7 STACK_PLANNER_SEED=0 \
  bash TokenHSI-coord/stack_planner/train.sh stack_path_v5_s0
```

기본값은 64 env, 200 iteration, planner action당 frozen executor 30 step이며 10 iteration마다
checkpoint를 저장한다. 중간 checkpoint에서 이어갈 때도 기존 run을 덮어쓰지 않고 새 tag를 쓴다.
`STACK_PLANNER_ITERS`는 최종 iteration 번호가 아니라 추가로 실행할 iteration 수다.

```bash
STACK_PLANNER_INIT=runs/stack_planner/stack_path_v5_s0/planner_000200.pth \
STACK_PLANNER_ITERS=200 MA_GPU=7 \
  bash TokenHSI-coord/stack_planner/train.sh stack_path_v5_s0_resume1
```

## Checkpoint viewer

학습된 planner의 mean action을 deterministic하게 실행하는 별도 viewer task다. frozen agent
checkpoint도 학습 때 쓴 파일을 같이 지정해야 한다. planner는 기본 30 action step마다,
그리고 stack phase가 바뀌는 즉시 다시 계획한다. 학습용 task나 기존 coordinator viewer를
수정하지 않는다. 바닥 띠는 두 agent의 전체 planner path를, 허리 높이 띠는
현재 frozen agent에 들어가는 steering window를 표시한다.

```bash
bash TokenHSI-coord/stack_planner/view.sh \
  runs/stack_planner/<tag>/planner_000010.pth \
  TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00011000.pth
```

로컬 desktop에 Isaac Gym 창을 직접 띄우며 noVNC를 사용하지 않는다. 기본 물리 GPU는 0이고
`MA_GPU=1`처럼 바꿀 수 있다. 현재 `DISPLAY`를 사용하고 값이 없으면 `:0`을 쓴다. 여러
scene을 보고 싶으면 세 번째 인자에 env 수를 주고, 재계획 주기는
`STACK_PLANNER_REPLAN_STEPS`로 바꿀 수 있다. 실행 속도 command는 replan 경계에서 즉시
점프하지 않고 기본 `0.75m/s²` 제한으로 고정 목표 속도까지 이동한다.

viewer의 path renderer는 carry-only scene에 존재하지 않는 generic sit/climb marker actor를
갱신하지 않는다. box와 task target은 reset/physics가 관리하고 planner path line만 추가로
그리므로, 이 생략은 시뮬레이션 state나 planner 입력을 바꾸지 않는다.
