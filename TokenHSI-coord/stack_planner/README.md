# Stack path-and-speed planner V13

Stack task 전용 Transformer planner다. 기존 `coordinator/`와
`trajectory_predictor/`의 코드 및 checkpoint namespace를 건드리지 않도록 별도 패키지로
분리했다.

## 현재 계약

- 물리 입력: 기존 `coordinator.schema.CoordinatorState`와 동일한 10개 state tensor
- planner memory: 최근 decision history, 고정-origin 33-point world trajectory,
  agent별 monotonic point progress
- backbone: root/box/goal 전용 tokenizer와 learnable `[SCENE]` token을 쓰는 Transformer encoder
- candidate 생성: 하나의 scene feature를 받는 독립 path-and-speed head 4개
- candidate 평가: `(scene feature, detached path/speed proposal)` 공유 evaluator
- 실행 출력: 선택된 하나의 두-agent end-to-end joint XY path와 pointwise speed profile
- hard anchor: 첫 decision의 `P0=departure root`; 이후 실행한 prefix는 고정
- route constraint: 연속 선분 투영 거리로 `box → stack goal` ordered visit를 학습
- checkpoint schema: `tokenhsi-stack-planner-v13`

planner 입력에는 virtual box를 넣지 않는다. 현재 Carry executor에만 필요한 virtual box는
`env_adapter.py`가 learned retreat endpoint에서 만들어 관측 직전에 변환한다. 이후 steering과
Carry가 분리되면 이 adapter만 제거하고 planner의 물리 경로 출력은 유지할 수 있다.
placement 전 마지막으로 유효했던 unified A1 endpoint를 보관하며, `A1_RETREAT` 진입 순간
virtual box를 그 endpoint로 즉시 이동한다. 따라서 phase 전환 후 다음 planner period까지
현재 위치에 virtual box가 남는 한 tick 지연이 없다.

```python
from stack_planner.model import StackTrajectoryPlanner

model = StackTrajectoryPlanner()
output = model(coordinator_state)
path = output["path_world"]  # selected path: [B, 1, 2, 33, 2]
speed = output["speed"]      # selected speed: [B, 1, 2, 33], m/s
candidate = output["selected_candidate"]  # [B]
```

한 decision에서 각 독립 head는 A1 carry부터 retreat까지와 A2 carry를 포함한 직전 33-point
trajectory에 적용할 pointwise XY correction을 제안한다. segment별 head가 아니다. 최초/reset
decision은 현재 `root → box → goal` geometric trajectory를 reference로 쓴다. 이후에는 이
33개 point의 출발점과 index를 그대로 유지한다. 현재 root를 직전 trajectory에 투영해 agent별
monotonic point progress를 구하고, progress 이전 correction은 0으로 고정하며 경계 두 point에
걸쳐 correction 자유도를 부드럽게 연다. 따라서 목적지에 가까워져도 짧아진 잔여 경로를 다시
33점으로 늘리지 않고 이미 지나온 prefix도 재계획하지 않는다. progress는 plan embedding에도
명시적으로 들어간다. head는
root를 제외한 `2 agents × 32 points × XY = 128D` correction과
`2 agents × 33 points = 66D` speed profile을 함께 출력한다. path correction은 두 번 low-pass한
`0.5*tanh(delta)`를 reference path에 직접 더한다. latent path parameter는 저장하거나 누적하지
않는다. speed는 sigmoid로 `[0.375, 1.5] m/s`에 제한하고 두 번 low-pass한다. 직전 trajectory
자체를 shared frame으로 바꿔 plan token으로 Transformer 입력에도
포함한다. evaluator가 네 proposal을
점수화한다. 학습에서는 같은 simulator state를 snapshot한 뒤 네 proposal을 모두 각각 30
low-level step 실행해 return을 직접 비교한다. evaluator는 최고 return 후보 index를 supervised
target으로 배우고, 다음 decision은 env별 최고 후보의 실제 종료 state에서 이어진다. PPO의
log-prob과 entropy에서도 progress 이전 action dimension을 mask하므로 이미 실행한 point에는
collision reward의 credit이 돌아가지 않는다. 나머지 continuous action credit은 각 후보가 직접
만든 return에만 연결된다. deterministic view/eval은
evaluator argmax 하나를 실행한다. 후보 붕괴 방지를 위해 A1 full-path 간 기본 0.25m margin의
diversity loss를 적용한다. diversity는 raw point가 아니라 네 차례 low-pass하고 4점 간격으로
고른 coarse route에서 계산한다. 각 mean correction의 second finite difference에도
`STACK_PLANNER_SMOOTHNESS_COEF`(기본 10.0)를 적용해 좌우 교대 zigzag가 후보 차이로 인정되지
않게 한다.

별도 `retreat_path`, retreat goal, switch output이나 사후 path concatenation은 없다.
각 경로의 첫 점은 departure root로 고정되고 아직 실행하지 않은 point만 planner가 계속 수정한다.
box와 stack goal은 특정 index에 묶지 않고 모든 path segment에 수선의 발을 내려 최소 거리를
구한다. 같은 segment이면 projection fraction, 다른 segment이면 index를 이용해 box가 stack
goal보다 먼저 방문되도록 제한한다. 허용 반경은 기본 0.15 m이고 초과 거리의 제곱을
`STACK_PLANNER_VISIT_PENALTY`로 reward에서 감점한다. frozen Carry의 box task goal 자체는
계속 stack goal이며 retreat endpoint로 바뀌지 않는다.

A2도 최초 joint decision에서 `departure → Box2 → eventual stack top` 전체 path를 A1과 동시에
출력하고 설치한다. 단, frozen Carry가 보는 실제 A2 placement goal은 handoff 전까지 Box2의
초기 위치로 유지되어 Box2 근처에서 멈춘다. bottom 안정화와 1초 delay 신호가 열리면 path를
재생성하지 않고 Carry goal만 실제 Box1 top으로 바꾼다. A2 executor는 이미 설치된 같은 path의
현재 투영 arc부터 이동을 이어간다.

### Frozen Carry execution bridge

retreat endpoint는 latch하지 않는다. `STACK_PLANNER_RETREAT_ENDPOINT_PENALTY`(default 1.0)는
직전 retreat world endpoint에서 tolerance(default 0.10m)를 초과한 변화의 제곱에 적용되며
최대 cost는 4로 제한한다. 최초 retreat/reset에는 적용하지 않는다. 경로 전체 consistency는
`STACK_PLANNER_RETREAT_PATH_CONSISTENCY_SCALE`(default 0.05)로 약화해 우회 경로 변경을 허용한다.
pointwise correction exploration std는 `STACK_PLANNER_DELTA_STD=0.12`, 마지막 path point는
`STACK_PLANNER_ENDPOINT_STD=0.20`, 초기 geometric box/goal 인근 point는
`STACK_PLANNER_ANCHOR_STD=0.03`이다. correction을 path 축으로 low-pass하고 second-difference
loss를 적용하므로 완만한 우회는 허용하면서 고주파 지그재그를 억제한다. 별도의 Bézier latent
parameter는 없다.
충돌 cost에는 root/box proxy 외에도 다른 agent의 held box와 비손 rigid-body의 3D proximity가
포함된다. endpoint 유지보다 회피가 유리해질 수 있으며 위험 접근 event나 수동 switch는 없다.

학습 모델과 frozen agent 사이에서만 `execution.py`가 같은 path의 실행 구간을 만든다.

- placement 전: ordered `box → carry goal` projection에서 goal의 arc 위치를 찾고,
  고정 departure부터 그 위치까지를 33점으로 보간한다. 설치할 때 현재 root를 실행 경로에
  투영하고 해당 arc부터 steering을 재개한다. 마지막 점은 실제 carry goal과 정확히 맞춘다.
- placement 후: 같은 path에서 `max(goal arc, 현재 root의 suffix 투영 arc)` 이후만 33점으로
  보간한다. 이미 지나간 retreat 구간은 executor와 box-clearance reward 양쪽에서 제외되고,
  planner 원본의 departure와 endpoint는 이동시키지 않는다.
  A1은 phase 2뿐 아니라 A2가 접근하고 stacking하는 phase 3/4에도 active row로 남는다.
  매 planner tick의 유효 suffix를 다시 설치하며
  path·arc·virtual box도 함께 갱신한다. execution latch는 없으며 endpoint가 이동 중 변할 수 있다.
  planner 전용 task는 phase 2 진입 시 manual retreat 대신 현재 위치의 stationary hold를
  설치한다. invalid 후보는 analytic collision/clearance penalty를 받고 다음 decision에서
  재평가한다. invalid raw 후보는 실행 path와 가상 box 어느 쪽도 변경하지 않는다. 유효 후보가
  설치될 때만 같은 endpoint를 path와 가상 box에 함께 적용하므로 steering token과 carry token이
  서로 다른 목표를 가리키지 않는다. 기존 planner 없는 task는 그대로다.
- 두 구간 모두 이후 기존 0.1 m/320-point steering ABI로 다시 resample한다. 모델이 예측한
  pointwise speed도 path와 같은 arc로 보간한다. 독립 replan 사이의 명령 변화는
  `STACK_PLANNER_COMMAND_ACCEL`(기본 `0.75 m/s²`)로 rate-limit한다. 학습 중 speed profile의
  pointwise 요동은 `STACK_PLANNER_SPEED_SMOOTHNESS_COEF`(기본 1.0)로 감점한다.

즉 33점은 항상 모델의 한 path다. carry용 33점과 retreat용 33점을 모델이 따로 출력하는
구조가 아니다. 어느 구간을 frozen agent에 설치할지는 simulator의 실제 placement phase를
사용하는 execution 문제이며, planner action이나 학습 head에는 switch가 없다.
A2 출발은 learned endpoint 도달이나 A1 이동 거리로 판정하지 않는다. bottom box가 물리적으로
안정화되면 A1 retreat를 열고, 그 안정 상태가 추가로 1초 유지되는 즉시 A2 goal을 활성화한다.
대기 시간은 `STACK_PLANNER_A2_STABLE_DELAY`로 설정한다.
planner task의 성공 `DONE`은 기본적으로 `STACK_PLANNER_RELAXED_DONE=1`을 사용한다. top box
중심의 XY가 bottom box 중심에서 footprint 반대각선
`0.5 * sqrt(size_x^2 + size_y^2)` 이내이고 목표 높이 오차가 tolerance 안이면 성공이다.
속도 및 연속 안정화 frame 조건은 요구하지 않는다. `0`으로 설정하면 parent sequential task의
strict stable-DONE을 사용하며, planner 없는 기존 환경의 판정은 변경하지 않는다.
A1이 active인 phase 2~4의 매 설치 execution suffix에는 놓인 Box1의 yaw/XY size를 반영한 oriented footprint
clearance penalty도 적용한다. footprint는 agent root 반경 0.35 m와 safety margin 0.10 m만큼
팽창하며, release 직후 가까이 서 있는 것 자체보다 이후 point가 box 안쪽으로 파고들거나
끝까지 안전 영역을 빠져나오지 않는 경로를 감점한다. 기본 계수는
`STACK_PLANNER_RETREAT_BOX_PENALTY=10.0`이다. 실제 rollout에서 bottom box가 밀린 누적량도
`STACK_PLANNER_BOTTOM_DISTURBANCE_WEIGHT=2.0`으로 감점한다. 따라서 기본 agent-agent collision
weight 1.0보다 box 경로 침범과 배치된 box 교란을 우선 회피하되, analytic footprint shaping이
전체 reward를 지배하지 않도록 한다.

### Decision history 입력

새 학습은 기본적으로 최근 4번의 planner decision state token을 함께 입력한다.
`STACK_PLANNER_HISTORY_STEPS`로 길이를 바꿀 수 있다. 각 시점은 기존과 동일한 6개
root/box/goal entity token이며 temporal embedding과 valid mask를 추가한다. episode reset 시 해당
env의 history만 비우고 현재 state 한 칸부터 다시 시작한다. recurrent hidden state가 아니므로 PPO가
transition을 섞어 minibatch로 학습해도 당시 observation을 정확히 재현한다. checkpoint의
`model_config.history_steps`에 길이를 저장하며 train/view/eval 모두 이를 사용한다.

V13 이전 planner checkpoint는 speed action/output 및 executor 계약이 달라 load하지 않는다. V13 checkpoint도
history 길이 또는 candidate 수가 다른 설정으로 resume하는 것은 거부한다.

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
clearance * top quality` 순서로 dependency를 구성하지만, `A1_RETREAT`에서 선택한 planner
transition의 total reward에서는 `potential_delta`를 제거한다. 따라서 retreat는 실제 collision,
box disturbance, fall/time과 analytic path collision penalty로 학습하며 A1-box 거리 증가 자체를
보상하지 않는다. placement와 A2 stacking 구간의 potential 학습 신호는 유지한다. 고정 retreat 방향, GT path,
virtual box 및 legacy Carry observation은 reward 입력에 포함하지 않는다. `collision`과
`bottom_disturbance`는 짧은 접촉을 놓치지 않도록 planner macro interval 동안 simulator에서
누적해 전달한다. `humanoid_fall`은 reset 전에 두 agent 중 하나라도 넘어졌는지를 latch한
0/1 값이며, frame마다 반복하지 않고 macro transition에 기본 `-3.0`을 한 번 적용한다.

초기 접근 구간도 학습되도록 실제 root-box 거리의 bounded rational quality를 쓰며, 이 항은
placement quality가 올라갈수록 자동으로 0에 가까워져 배치 뒤 retreat를 방해하지 않는다.
top 목표는 stage flag가 아니라 현재 bottom box의 물리 pose/크기에서 직접 계산한다.

## Closed-loop PPO

`train_closed_loop.py`는 sequential-stack agent checkpoint를 inference-only로 고정하고 planner
network parameter와 candidate별 action log-std만 최적화한다. 한 PPO transition은 기본 30 action step(30 Hz에서
약 1초) 동안 같은 plan을 유지한다. stack phase 전환은 이 hold를 기다리지 않는다.
한 scene마다 후보 4개를 동일한 시작 snapshot에서 순차 실행하므로 simulator workload는
단일 후보 방식의 약 4배다. evaluator는 rollout return argmax를 직접 분류하고,
`candidate_evaluator_loss/accuracy`, `candidate_usage_*`를 기록한다.
다음 replan의 mean path에는 직전 mean plan의 같은 고정 point index와 맞추는 temporal
consistency loss를 기본 적용한다. 실행한 prefix는 correction/action 양쪽에서 이미 freeze되므로
미실행 future의 불필요한 흔들림만 억제한다. episode reset과 물리 상태 전환은 mask하며,
`STACK_PLANNER_CONSISTENCY_COEF`로 강도를 조절하고 `0`이면 끌 수 있다.
각 episode는 기본적으로 `loco_carry=1.0`으로 두 agent와 두 box를 새로 배치하고 접근
단계부터 시작한다. 기존 carry의 `carryWith`/`putDown` 중간 RSI는 full-stack planner
reset에 섞지 않는다.

```bash
MA_GPU=7 STACK_PLANNER_ENVS=512 STACK_PLANNER_ITERS=200 \
STACK_PLANNER_SEED=0 STACK_PLANNER_CANDIDATES=4 \
  bash TokenHSI-coord/stack_planner/train.sh stack_path_speed_v13_s0
```

로컬 physical GPU 1에서는 로컬 checkpoint 배치와 `tokenhsi118` 환경을 사용하는
전용 wrapper를 실행한다. 기본 env 수는 24-GB GPU에서 실제 PPO backward까지 검증한
1024이다.

```bash
bash TokenHSI-coord/stack_planner/train_local_gpu1.sh local_stack_s0
```

짧은 smoke는 다음처럼 실행한다.

```bash
STACK_PLANNER_ENVS=8 STACK_PLANNER_ITERS=1 \
STACK_PLANNER_HORIZON=8 STACK_PLANNER_LOW_STEPS=6 \
  bash TokenHSI-coord/stack_planner/train_local_gpu1.sh local_stack_smoke
```

metric, sidecar, planner checkpoint는 `runs/stack_planner/<tag>/`에만 저장된다.
`metrics.jsonl`에는 reward 항 외에 planner/executor 품질을 직접 읽을 수 있는 다음 metric을
iteration마다 기록한다.

- `path_mae`: active agent root와 설치된 dense planner path 사이 평균 횡오차(m)
- `fall_ratio`: planner macro transition 중 한 agent라도 넘어진 비율
- `collision_ratio`: 실행 low-level step 중 기존 collision proxy가 양수인 비율
- `collision_cost`: macro별 연속 collision cost 평균
- `collision_agent_agent_{ratio,cost}`: 두 humanoid root proximity
- `collision_box_box_{ratio,cost}`: 두 box footprint proximity
- `collision_agent_box_{ratio,cost}`: 각 agent root와 상대 box proximity
- `collision_held_box_body_{ratio,cost}`: 들고 있는 box와 상대 humanoid rigid-body의 3-D proximity
- `path_smoothness_loss`: mean correction의 second-difference 제곱 평균
- `speed_smoothness_loss`: mean speed profile의 adjacent-point difference 제곱 평균
- `bottom_postplace_linear_speed`: bottom placement 이후 평균 선속도(m/s)
- `bottom_postplace_angular_speed`: bottom placement 이후 평균 각속도(rad/s)
- `bottom_postplace_motion_per_interval`: bottom placement가 관측된 macro당 누적 이동량(m)
- `bottom_postplace_exposure`: 전체 실행 step 중 bottom placement 이후 상태가 차지한 비율

마지막 exposure를 함께 봐야 흔들림 metric의 0이 안정적인 box인지, 아직 placement phase에
도달하지 못한 것인지 구분할 수 있다.
같은 tag의 디렉터리가 이미 있으면 덮어쓰지 않고 종료한다. 현재 v13 설계의 서버 학습은 다음처럼
실행할 수 있다.

```bash
MA_GPU=7 STACK_PLANNER_ENVS=512 STACK_PLANNER_ITERS=200 \
STACK_PLANNER_SEED=0 STACK_PLANNER_CANDIDATES=4 \
STACK_PLANNER_DELTA_SCALE=0.5 \
STACK_PLANNER_SMOOTHNESS_COEF=10.0 \
STACK_PLANNER_SPEED_STD=0.20 \
STACK_PLANNER_SPEED_SMOOTHNESS_COEF=1.0 \
  bash TokenHSI-coord/stack_planner/train.sh stack_path_speed_v13_s0
```

launcher 기본값은 2,048 env지만 full candidate rollout은 후보 4개를 모두 물리 실행하고 branch
snapshot도 보존하므로 첫 run은 512 env를 권장한다. 200 iteration, 후보당 frozen executor 30
step이며 5 iteration마다 checkpoint를 저장한다. PPO minibatch 기본값은 env와 horizon에
비례하며 candidate 4개 때문에 epoch당 16개 minibatch가 된다. 중간 checkpoint에서
이어갈 때도 기존 run을 덮어쓰지 않고 새 tag를 쓴다.
`STACK_PLANNER_ITERS`는 최종 iteration 번호가 아니라 추가로 실행할 iteration 수다.

```bash
STACK_PLANNER_INIT=runs/stack_planner/stack_path_speed_v13_s0/planner_000200.pth \
STACK_PLANNER_ITERS=200 MA_GPU=7 \
  bash TokenHSI-coord/stack_planner/train.sh stack_path_speed_v13_s0_resume1
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
`MA_GPU=1`처럼 바꿀 수 있다. graphics Vulkan ID도 기본적으로 같은 번호를 사용하며,
Vulkan 순서가 CUDA physical 순서와 다르면 `TOKENHSI_GRAPHICS_DEVICE_ID`로 override한다.
흰 선은 최신 planner 원본, 파란 wireframe box는 retreat observation 속 가상 box다.
기존 실행 path와 steering window 색상 overlay는 표시하지 않는다. invalid/slicing 상태 때문에
원본과 실행 path가 다를 수 있으며 콘솔의 `valid/installed/retreat_ready`로 구분한다.
현재 `DISPLAY`를 사용하고 값이 없으면 `:0`을 쓴다. 여러
scene을 보고 싶으면 세 번째 인자에 env 수를 주고, 재계획 주기는
`STACK_PLANNER_REPLAN_STEPS`로 바꿀 수 있다. 실행 속도 command는 replan 경계에서 즉시
점프하지 않고 기본 `0.75m/s²` 제한으로 고정 목표 속도까지 이동한다.

viewer의 path renderer는 carry-only scene에 존재하지 않는 generic sit/climb marker actor를
갱신하지 않는다. box와 task target은 reset/physics가 관리하고 planner path line만 추가로
그리므로, 이 생략은 시뮬레이션 state나 planner 입력을 바꾸지 않는다.

### Path 중간 release 진단 view

Planner나 sequential-stack coordinator 없이 frozen `HumanoidMASteerCarry` agent가 path 중간의
carry goal에서 box를 내려놓은 뒤에도 steering을 계속할 수 있는지 확인한다. reset마다 A1의
고정 steering path를 `현재 root → 실제 box → carry goal → goal 너머`로 만든다.
경로를 만들 때만 임시 endpoint를 사용하고, policy observation과 reward가 보는 실제 box carry
target은 원래 goal로 즉시 복원한다. phase 전환이나 release/retreat signal은 전혀 주지 않는다.
또한 box→endpoint 구간을 직선으로 만들어 carry goal을 정확히 지나도록 한다. 학습에는 쓰지 않는다.

```bash
MA_GPU=1 bash TokenHSI-coord/stack_planner/view_midpath_release.sh \
  <frozen-agent.pth> 1
```

이 진단 스크립트는 CUDA remap을 사용하지 않고 compute/rl/graphics를 모두 physical GPU 1로
명시한다. Python 진입 직후에도 `torch.cuda.set_device(1)`을 호출한다. `MA_GPU`로 다른 physical
GPU를 선택할 수 있다.
학습/view/eval launcher는 오래된 PhysX가 요구하는 unversioned `libcuda.so` 호환 링크를
`$ROOT/.runtime/physx-lib`에 자동으로 만들고 `LD_LIBRARY_PATH`에 넣는다. `/tmp` 아래 수동
링크에는 의존하지 않는다.
환경 생성 reset의 초기 observation을 checkpoint 로드 뒤 재사용한다. 또한 carry-only scene에
존재하지 않는 generic sit/climb marker actor 갱신은 생략한다. 두 처리는 각각 이 빌드의
중복 full-reset 오류와 첫 render CUDA 오류를 막는다.
표준 rl_games player는 1-env episode가 끝나면 다음 game 시작 시 `reset(None)`을 호출한다.
이 진단은 그 경로를 사용하지 않고 계속 step하다가 done인 env의 agent row만 명시적으로
reset한다. 따라서 프로세스를 다시 띄우지 않고 새 시나리오를 반복하며, 이는 planner 학습과
독립 평가에서 사용하는 episode reset 방식과 같다.
기본 goal 너머 거리는 2m이며 `STACK_MIDPATH_EXTENSION`으로 바꿀 수 있다. 콘솔의
`carry_goal`, `path_end`, `installed_error`로 실제 carry target과 별도 endpoint 및 설치 오차를
확인할 수 있다. cyan 띠가 policy 입력 path, 노란 십자가 carry goal, 빨간 십자가 endpoint다.

## Retreat checkpoint 평가

학습 프로세스를 수정하거나 종료하지 않는 별도 deterministic 평가다. 학습 run의
`run.env`에서 executor/stage1/env config와 환경변수를 재사용하며 sidecar가 없으면 거부한다.
GPU는 학습과 겹치지 않게 지정한다. 서버 예시:

```bash
MA_GPU=7 \
STACK_PLANNER_EVAL_ENVS=64 STACK_PLANNER_EVAL_SEED=0 \
bash TokenHSI-coord/stack_planner/eval_retreat.sh \
  <training-tag> <planner-checkpoint.pth> <unique-eval-tag>
```

기본 3600 physics steps, replan 간격은 학습 `STACK_PLANNER_LOW_STEPS`를 사용한다.
`STACK_PLANNER_EVAL_STEPS/STACK_PLANNER_EVAL_REPLAN`으로 override할 수 있다.
로컬은 `TOKENHSI_CONDA_ENV=tokenhsi118 CONDA_BASE=/home/injesus1010/anaconda3 MA_GPU=1`을 추가한다.
checkpoint 비교 시 seed/env 수/steps/replan과 학습 환경 설정을 같게 유지한다.

기본 평가는 `STACK_EVAL_BOX_GRID=1`로 small/medium/large bottom/top box의 3x3 조합을
환경 벡터 전체에 고르게 배정한다. 박스 asset 크기는 환경 생성 때 정해지므로 같은 env의
episode reset에서는 바뀌지 않지만, 64-env 평가는 모든 조합을 포함한다. 배치는 기본 `free`로
episode마다 다시 샘플링한다. 하나의 checkpoint를 free/cross/parallel/solo 네 배치에서 연속
평가하려면 다음 suite를 사용한다.

```bash
MA_GPU=7 STACK_PLANNER_EVAL_ENVS=64 \
bash TokenHSI-coord/stack_planner/eval_retreat_suite.sh \
  <training-tag> <planner-checkpoint.pth> <unique-eval-prefix>
```

단일 배치만 평가할 때는 `STACK_PLANNER_EVAL_SCEN=free|cross|parallel|solo`를
`eval_retreat.sh`에 넘긴다. summary에는 실제 unique bottom/top size 수와 size-pair 수가
기록되어 size grid가 적용됐는지 바로 확인할 수 있다.

결과는 `runs/stack_planner_eval/<eval-tag>/summary.json`과 `episodes.jsonl`이다.
도달은 현재 endpoint 0.25m 이내, 정지는 그 범위에서 XY 속도 0.15m/s 이하를 한 번이라도
만족한 비율(지속 정지 보장은 아님)이다. 완결되고 retreat에 진입한 episode만 비율의
분모에 넣고, 평가 끝에서 잘린 episode는 censored로 별도 기록한다. retreat 진입 수와
완결 수가 작거나 0이면 평가 길이를 늘려야 하며 0 표본 rate는 null이다.
path MAE는 실제 A1 lateral deviation, endpoint drift는 retreat replan 사이 world 이동량이다.
collision은 contact force가 아니라 학습과 같은 proximity proxy다. 최소 gap은 XY 원형
근사의 body/box 표면 간격이고 episode 최소값의 평균이며 음수는 근사상 침범이다.
프레임 진단은 censored 구간도 포함한다.

Agent2 접근(radial speed >0.1m/s)/정지(XY speed <=0.1m/s) 자연 발생 여부와 조건별 도달·충돌
통계를 기록한다. 두 subset은 겹칠 수 있으며 강제 접근/정지 시나리오나 인과적 비교가 아니다.
동일 seed는 초기 난수 조건을 맞추지만 GPU physics의 bit-exact 재현을 보장하지 않는다.
