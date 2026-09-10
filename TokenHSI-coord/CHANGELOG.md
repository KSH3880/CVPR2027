# TokenHSI-coord — 변경 기록

> 파일 변경은 hook이 자동 기록. 무엇을/왜 바꿨는지는 Claude가 `###` 항목으로 덧붙인다.

## 2026-09-10

### Stack 전용 Transformer planner 패키지 분리

- 기존 coordinator 입력 10개 tensor와 joint path/speed 출력 계약을 유지하는
  `stack_planner/`를 별도 추가했다.
- 6-entity Transformer encoder와 learned-query Transformer decoder 뒤에 path, speed,
  dwell, value/risk head를 분리해 이후 stack event/3-D placement 출력을 확장할 경계를 만들었다.
- 기존 C1/C2 모델, checkpoint loader, simulator runtime은 수정하지 않았고 stack checkpoint는
  `tokenhsi-stack-planner-v1` schema로 격리했다.

### Stack planner 전용 outcome reward 분리

- `stack_planner/reward.py`에 frozen executor의 macro-step 실행 전후 physical state로 계산하는
  team potential reward를 추가했다.
- bottom placement, direction-free clearance, top placement를 곱으로 연결해 dependency를
  표현하고 실제 collision, bottom disturbance, 시간, invalid/unsafe cost를 별도 항으로 뒀다.
- GT path, 고정 retreat 방향, virtual box와 기존 Carry reward는 입력 계약에서 제외했다.
- planner macro interval에서 any-agent fall을 reset 전에 latch해 한 번만 적용하는 기본 `-3.0`
  team penalty를 추가했다.

### Stack planner closed-loop PPO와 실제 simulator smoke

- 동일 Transformer decoder에 물리적 A1 retreat path/speed head를 추가했다. planner는 실제
  box만 입력받고, frozen Carry agent용 virtual box 변환은 새 `stack_planner/env_adapter.py`
  경계 안에서만 수행한다.
- sequential-stack agent를 완전히 freeze한 macro-step PPO trainer와 독립 launcher/sidecar/
  checkpoint를 `stack_planner/`에 추가했다. 기존 coordinator 및 masteer 소스는 수정하지 않았다.
- reward의 먼 거리 `exp(-distance^2)` 포화를 확인해 bounded rational quality로 바꾸고,
  실제 root-box 접근 진행을 placement가 낮을 때만 보조하도록 추가했다. top target은 stage
  command가 아닌 현재 bottom box의 물리 geometry에서 계산한다.
- 누락된 비활성 sit/climb URDF actor까지 commit하던 generic reset을 stack 전용 adapter에서
  제외했다. 8-env, horizon 8, low-step 6 smoke가 terminal reset, PPO update, checkpoint 저장까지
  통과했고 potential delta가 `0`에서 `2.38e-4`로 살아났다. pure-PyTorch 단위검사 10개도 통과했다.
- 본 학습 `stack_planner_v0_s0a`를 frozen `anti_feat_top_s2/Humanoid_00011000.pth`,
  64 env, horizon 32, low-step 6, 200 iteration으로 GPU0에 시작했다. iteration 10에서 첫
  12 MB checkpoint를 저장했다. 이후 서버에서 본 학습하기로 해 사용자 요청에 따라 로컬
  process group만 정상 종료했으며 산출물은 보존했다.

### Stack planner checkpoint viewer 분리

- 학습 checkpoint를 strict schema 검사 후 로드하고 candidate 0의 mean trajectory를
  deterministic하게 실행하는 `HumanoidMAStackPlannerView`와 전용 launcher를 추가했다.
- 기본 6 low-level step 및 stack phase 전환마다 실제 simulator state에서 재계획하고,
  planner의 path/speed와 learned retreat endpoint를 frozen sequential-stack agent에 적용한다.
- 기존 coordinator/viewer와 파일을 공유하지 않는다. launcher는 noVNC 없이 로컬 Isaac Gym
  창을 직접 띄우고 기본 GPU 0을 사용하며 `MA_GPU`로 로컬 GPU를 선택할 수 있다.
- retreat head는 이미 있으나 중복 contract field 두 개가 없던 초기 v1 weight도 해당 값을
  model config에서 복구한 뒤 나머지 schema와 state dict를 동일하게 strict 검사한다.

## 2026-09-09

### Sequential stack의 가상 retreat를 coordinator에 연결

- `juan`에서 누락돼 import가 깨졌던 sequential execution bridge를 복원했다.
- A1의 place/verify뿐 아니라 retreat phase도 planner 소유로 두고, 이 구간의 planner
  snapshot에는 executor carry token과 같은 retreat endpoint의 가상 box/goal을 넣는다.
- 가상 상태는 clone에만 기록해 Isaac Gym의 실제 배치 완료 box를 건드리지 않으며,
  path의 box arc와 invalid fallback도 동일한 가상 goal을 사용하도록 맞췄다.
- CPU 회귀검사에 phase ownership과 box=goal인 degenerate retreat leg를 추가했다.

## 2026-09-04

### C11 learned episode-persistent priority

- C2 MLP에 선택적 2-class `priority_head`를 추가했다.
- priority는 첫 plan의 categorical PPO action으로만 샘플링하고 episode 동안 고정한다.
- 선택된 agent는 직선·1.5m/s, 반대 agent만 learned path/speed를 실행한다.
- checkpoint가 learned-priority 계약을 저장하며 viewer/eval은 이를 자동 복원한다.
- C2 unit test 58개와 2-iteration Isaac Gym smoke를 통과했다.

### Viewer-only time-synchronized Cross

- `scripts/coord/view_local.sh`에 opt-in `MS_VIEW_TIMED_CROSS=1`을 추가했다.
- 기존 직각 Cross 배치 뒤, 사람·박스·받침대의 상대 자세는 보존한 채 먼저 도착하는
  agent 묶음을 교차점 반대 방향으로 이동시킨다. 시간은 측정된 ms18 평속
  approach/carry 속도와 phase별 pickup dwell로 계산한다.
- 두 box→goal 구간은 같은 교차점을 통과한다. coordinator 입력에는 경로·교차점·시간을
  추가하지 않고 기존 state-only 계약을 유지한다.
- `COORD_VIEWER=1`이 아니면 fail-closed하므로 학습 및 batch 평가 분포는 변하지 않는다.
  1-env headless smoke에서 예상 도착시간 차이는 반복 reset 모두 `0.000–0.001초`로,
  기본 허용치 `0.25초` 안이었다.

### C4 smooth future proximity

- C2 자유 감속창에 scenario-agnostic proximity collision 모드를 추가했다. time-aligned
  root 거리와 box 크기/carry 상태로 동적 envelope를 만들며, 접근 중인 미래 위치는
  closing-rate로 위험도를 더 키운다.
- 기존 hinge 충돌 로스는 기본값으로 그대로 보존했다. `COORD_C2_PROXIMITY_COLLISION=1`일
  때만 새 로스를 쓰며 raw proximity와 설정값을 metrics/sidecar에 기록한다.
- `c4_compare.sh`로 detour-delay 3/10을 제외한 모든 조건을 동일하게 비교할 수 있다.

### Smooth slowdown depth option

- 기존 slowdown/conflict window의 zero-centred clamp는 raw depth가 음수가 되면 gradient도
  0이 되는 dead zone이었다. 기존 체크포인트 동작은 유지하면서, 새
  `COORD_C2_SMOOTH_DEPTH=1` 옵션에서 `1.125*sigmoid(raw-3)` decoder를 사용하도록 했다.
- model forward와 stochastic action decode가 같은 config 값을 사용하며 checkpoint와
  sidecar에도 저장한다. 두 window 모드에서 음수 raw의 nonzero depth/gradient를 회귀
  검사했고 전체 pure-PyTorch 67개 테스트가 통과했다.

### Configurable proximity safety margin

- C4의 각 root/carry envelope에 더하는 `COORD_C2_PROXIMITY_MARGIN`을 추가했다.
  0이 기존 동작이며 음수는 거부한다. margin 증가가 동일 trajectory의 proximity risk를
  단조 증가시키는 회귀 검사를 추가했다.
- C3/C4 비교 스크립트의 collision, delay 및 clearance 설정을 외부에서 명시적으로
  override할 수 있게 해 conservative-risk r1/r2 비교를 분리했다.

### C3 conflict-attached slowdown window

- C2의 자유 `(center,width,depth)` 대신 두 joint predicted path의 closest crossing에서 끝나는
  `(length,depth)` 감속 decoder를 추가했다. 교차점 이후는 구조적으로 1.5m/s로 복귀한다.
- MLP가 waypoint residual과 두 속도 파라미터를 한 번에 출력하는 state-only 계약, frozen
  ms18 bridge, random priority 및 timebias3 loss는 유지했다.
- 새 config/checkpoint/action 계약과 학습 sidecar 손잡이 `COORD_C2_CONFLICT_WINDOW`를 추가하고
  `c3_crosswin_s0` 실행 스크립트를 만들었다.
- pure-PyTorch 65개 테스트에서 hard anchor, 감속구간 경계, nominal suffix, 124-D action 및
  checkpoint bit-exact round-trip을 포함해 모두 통과했다.

## 2026-09-03

### compact slowdown-window 두 트랙

- C2 MLP에 agent별 `center/width/depth` 3-parameter slowdown window 출력을 추가했다.
  거리축 cosine decoder가 이를 기존 33점 speed profile로 변환하며 priority agent는 계속
  직선·1.5m/s로 강제된다.
- 양보 agent의 동일 경로 full-speed 대비 추가시간 loss를 추가했다. 기존 C2 기본값은
  바꾸지 않았고 새 실험 스크립트에서만 기존 speed regularizer를 끄고 이 항을 사용한다.
- 최대 감속 폭 3m와 full-path 허용을 250 iter 병렬 비교하는 실행 스크립트를 추가했다.
- 새 출력의 shape, 폭 제한, gradient, strict checkpoint bit-exact round-trip을 포함해
  coordinator 테스트 62개가 통과한다.

### 정답 없는 explicit-yield loss 대조군

- full-speed straight counterfactual로 찾은 교차점에서 episode-assigned yielder가 priority보다
  1초 이상 늦게 도착하도록 하는 gap hinge를 추가했다. learned path가 우회해도 이 속도
  신호는 사라지지 않는다.
- slowdown window의 교차점 방향 경계를 박스 호좌표에 붙이는 anchor loss와, 교차점을
  1m 지난 뒤 1.5m/s로 복원하는 post-cross loss를 추가했다. 감속 위치·폭·세기 label이나
  oracle trajectory는 사용하지 않는다.
- reset 직후 high-level 계획 표본에 위 명시적 항을 3배 가중할 수 있게 했다. 모든 옵션은
  기본 off라 기존 체크포인트와 실행에는 영향이 없다.
- pure-PyTorch test 63개와 2-env frozen-ms18 smoke를 통과했다. 정식 실행 태그는
  `c2_k1_slowwin_explicit_s0`이다.

### 감속 대 우회를 같은 시간 단위로 비교

- 같은 predicted path의 full-speed 대비 추가시간을 slowdown cost로, straight
  root→box→goal의 full-speed 대비 predicted path의 추가시간을 detour cost로 분리했다.
  실제 collision 계산은 predicted path/speed만 사용한다.
- 기본-off `COORD_C2_DETOUR_DELAY_COEF`를 추가했다. time-bias 비교에서는 slowdown `1`,
  detour `3`으로 두어 동일 완료시간이면 감속을 선호하지만 효율적인 우회는 남긴다.
- in-place strict-checkpoint resume와 3000-step/200-checkpoint 후속 lane 스크립트를 추가했다.
  테스트 64개와 2-env frozen-ms18 smoke를 통과했다.

### 랜덤 fixed-priority와 재계획 consistency 실험

- C2에 기본-off `COORD_C2_RANDOM_PRIORITY`를 추가했다. 에피소드마다 agent 0/1 중
  하나를 무작위 priority로 고정하고, 그 agent의 계획은 `root→box→goal` 직선과 전 구간
  `1.5m/s`로 강제한다. 반대 agent의 learned `(x,y,v)`만 양보 또는 우회에 사용한다.
  별도 role/scenario 입력은 추가하지 않았으며 checkpoint가 이 runtime 계약을 보존한다.
- 연속 두 replan을 같은 절대 미래시각에서 비교하는 기본-off
  `COORD_C2_CONSISTENCY_COEF`를 추가했다. position Smooth-L1과 speed MSE만 쓰며 reset과
  phase 전환은 mask한다. collision loss는 그대로라 위험 변화가 생기면 새 계획으로
  수정할 수 있다.
- 59개 pure-PyTorch test와 2-env frozen-ms18 smoke를 통과했다. 정식 태그
  `c2_k1_jointpoint_future8_meastime_rprio_cons1_s0`은 64 env/500 iter로 실행하며
  20/50/100/250 milestone을 평가한다.

### frozen ms18 시간 실측 및 measured-timing C2

- coordinator 학습과 분리된 `measure_executor`를 추가했다. Free/Cross에서 고정
  `0.375/0.75/1.125/1.5m/s`와 carry 중 step 감속을 frozen ms18에 보내고, 실제 root
  변위속도·pickup dwell·교차점 도착시각·rigid-body 최소거리를 raw NPZ로 저장한다.
- Cross seed0 48 env와 seed1 96 env를 합쳐 명령→실제 속도를 실측했다. approach는
  `[0.812, 0.898, 1.170, 1.401]m/s`, carry는
  `[0.658, 0.838, 1.124, 1.310]m/s`였고 pickup dwell 중앙값은 `0.90s`였다.
  따라서 기존의 `v_actual=v_cmd`, pickup `1.5s` 가정은 실제 executor와 맞지 않는다.
- 위 4점 piecewise lookup과 phase0/1 dwell `0.90/0.53s`를 future-collision rollout에만
  적용하는 `COORD_C2_MEASURED_EXECUTOR_TIMING`을 추가했다. 기본값은 0이라 기존
  checkpoint/runtime/실행 중인 실험은 변하지 않는다.
- 보정 전후 교차점 예측 오차 중앙값은 `1.219→1.038s`, 두 agent 시간차 오차는
  `1.006→0.767s`로 감소했다. 별도 태그 `c2_k1_jointpoint_future8_meastime_s0`에서
  동일 MLP·joint `(x,y,v)`·loss 계수를 두고 이 옵션 하나만 켜 학습과 milestone 평가를
  시작했다. pure-PyTorch 55 tests와 shell/compile 검사를 통과했다.
- 학습/자동평가 완료 후 250-iter Cross가 success/collision `0.7109/0.1043`으로 가장
  낮은 충돌률을 냈고, 500-iter는 Cross `0.7201/0.1131`, Free `0.9245/0.2978`이었다.
  seed0에서는 analytic 대비 task를 보존하며 충돌을 줄였지만 후반 표류가 있어 250 PTH를
  후보로 두고 다른 seed 재현성 평가 전에는 최종 채택하지 않는다.
- measured-timing 20-iter seed0 Cross는 success/collision/invalid
  `0.7500/0.1210/0.0018`이다. task는 보존됐지만 collision은 analytic `0.1191`보다 아직
  높으므로 채택하지 않고 50/100/250 milestone을 계속 평가한다.

### C2 실제 시간 오차를 위한 robust future-collision 비교

- 명시적 감속 target 없이 joint `(x,y,v)` 출력의 상위 8개 미래 충돌 시점만 학습하는
  설정을 추가했다. seed0 50-iter에서는 collision이 `0.1191→0.0909`로 줄었지만 다른
  scene seed에서 재현되지 않았고, 실제 충돌이 carry phase에 집중됨을 확인했다.
- frozen ms18의 집기·보행 시간 오차를 반영하도록 ±0.75초 위치 조합을 비교하는 robust
  collision loss를 추가했다. 계수 50은 invalid와 task 저하를 일으켜, 동일 구조에서 계수만
  10으로 낮추는 새 독립 태그를 실행한다.
- `COORD_C2_FUTURE_COLLISION_COEF`를 trainer와 실행 sidecar에 연결했다. 기존 기본값 50은
  유지되며 50개 pure-PyTorch test, py_compile, shell syntax 및 diff check를 통과했다.

### C2 K=1 collision-first residual track 추가

- C1/B 체크포인트와 동작을 보존하면서 `COORD_MODEL=c2`를 추가했다. C2는 6-entity
  Transformer와 residual trajectory/smooth speed를 유지하되 후보를 K=1로 줄이고 pickup
  dwell을 1.5초로 고정한다.
- best-of-K/diversity/risk auxiliary를 제거하고 time-aligned future collision, speed
  smoothness, full-speed counterfactual 기반 unnecessary slowdown 세 항만 사용한다.
- C2 checkpoint를 `tokenhsi-coord-c2-v1`/`coord_c2_*`로 분리하고 trainer, runtime wrapper,
  eval/view loader를 연결했다. raw command speed bin과 비대칭 감속 episode 지표도 평가
  JSON에 추가했다.
- 기존 C1/B 13개와 C2 전용 4개를 합한 pure-PyTorch 17 tests, shell syntax, 2-env 실제
  frozen-ms18 PPO smoke를 통과했다. smoke에서 future collision이 path/speed 양쪽 head에
  gradient를 주고 invalid/fallback 없이 C2 PTH가 저장되는 것을 확인했다.

### analytic prior로 C1 task 보존 및 독립 B0 minimal track 추가

- unrestricted learned path가 analytic 기준선보다 carry/place를 크게 무너뜨리는 것을 확인해,
  C1 pickup leg를 `root→box` 직선으로 고정하고 carry residual을 최대 1m로 제한했다.
  path/speed head는 analytic path와 1.5m/s를 정확히 내도록 zero-init한다.
- `c1_analyticprior_s0` 고정 128-env 평가에서 Cross success/collision
  `0.724/0.104`, Free `0.901/0.290`을 기록했다. analytic 기준선의 Cross
  `0.698/0.119`, Free `0.921/0.300`과 비교해 toy coordinator 가설을 처음 통과했다.
- 복잡한 C1과 별개로 `simple_model.py`, `simple_policy.py`, `simple_checkpoint.py`에 B0를
  추가했다. B0는 2-layer Tanh MLP가 A/B lateral과 A/B slowdown 4 action만 내고,
  Transformer/candidate/risk head/auxiliary loss를 전혀 사용하지 않는다.
- B0 checkpoint는 `tokenhsi-coord-b0-v1` schema와 `coord_b0_*` prefix를 사용한다.
  C1 checkpoint와 초기화·loader·태그를 분리했으며, 두 트랙은 frozen ms18과 동일 평가만
  공유한다.
- B0 pure-PyTorch 검사를 포함한 전체 12 tests와 4-env/1-iteration Isaac smoke를 통과했다.
  smoke에서 frozen ms18 rollout, PPO update, aux=0, B0 PTH 저장을 확인했다.
- 현재 C1의 실제 collision 계수만 `2→50`으로 바꾼 `c1_analyticprior_col50_s0`와 B0의
  첫 정식 실험 `b0_mlp_col50_s0`를 동시에 실행한다. 둘 다 500 iter 후 Cross/Free를 자동
  평가하며 한 트랙의 결과나 변경을 다른 트랙에 섞지 않는다.
- B0 100-iter 중간평가는 seed0에서 Cross success/collision `0.755/0.112`, Free
  `0.928/0.237`로 analytic `0.698/0.119`, `0.921/0.300`을 보존·개선했다. 다만 plan
  invalid가 `12~16%`라 fallback을 이용하는 허점이 확인됐다. 모델과 auxiliary=0은 그대로
  두고 invalid reward만 `0→-1`로 켠 `b1_mlp_invalid1_col50_s0`를 별도 태그로 시작했다.
- C1 collision 50은 최종 Cross `0.706/0.127`로 기존 C1 `0.724/0.104`보다 악화돼
  폐기했다. 기존 C1은 held-out seed1에서도 Cross `0.729/0.108`, Free `0.926/0.216`으로
  유지돼 Track A 최종 PTH로 확정했다.
- B1은 invalid를 줄였지만 최종 Cross `0.736/0.137`, Free `0.904/0.302`로 B0보다
  나빴다. 벌점 대신 경로를 유효하게 만드는 B2 safe-bow를 추가했다.
- B2는 carry lateral을 box→goal 거리 이하로 제한하는 단일 구조 변경이다. 13개 CPU test와
  500-iter 학습을 통과했고 전체 학습 invalid는 `0.27%`였다. B2-100은 seed0 Cross/Free
  success `0.715/0.923`, collision `0.119/0.288`, invalid `0.19%/0.14%`를 기록했다.
- 최종적으로 Track B의 점수 upper bound는 B0-100, fail-closed bridge 추천은 B2-100이다.
  두 모델 모두 200회 이후 성능이 좋아지지 않아 100-iter early stopping을 채택했다.

## 2026-09-02

### C1 state-only joint coordinator 저장소 분리

- `TokenHSI-masteer`의 코드만 복사하고 24GB `output/`은 복사하지 않아 ms18 실행
  계약을 보존하면서 실험 산출물은 분리했다.
- `coordinator/`에 shared-state schema, 6-entity Transformer, 이름 없는 K=4 query,
  hard-anchor Bézier joint path, 8개 acceleration knot 기반 smooth speed, pickup dwell,
  time-aligned 사람/상자 충돌 평가, safety-first selector와 strict checkpoint를 추가했다.
- `HumanoidMACoordCarry`는 selected path/speed를 기존 `[320,2]`/`_mscale` 버퍼에
  설치한다. 매 6 action step과 phase 변화에 actual state로 replan하고, 새 plan 사이의
  속도 명령에는 acceleration rate limit을 적용한다.
- pure-PyTorch 단위검사 8개로 hard anchor, speed dynamics, K selection, held dwell,
  dense bridge, checkpoint bit-exact round-trip과 backward gradient를 확인했다.
- closed-loop runner를 2 env/1 iteration으로 실제 구동해 frozen ms18 rollout,
  coordinator PPO update, numbered/latest PTH 저장을 확인했다. 그 PTH를 다시 불러온
  headless viewer도 599-step episode 2회를 연속 완주했다.
- coordinator가 곧 폐기할 부모 GT 경로에 기존 cross 직선 self-check를 적용하던 충돌을
  제거하고, 실제 선택 C1 plan은 anchor/길이/곡률/속도 범위로 별도 fail-closed 검사한다.
- 첫 `c1_cross_s0` 학습에서 all-invalid sample의 선택 금지 sentinel `1e9`가 auxiliary
  best-of-K에도 들어가 aux가 `3~7e8`로 오염되는 것을 확인했다. 선택용 hard ban과
  학습용 연속 곡률/버퍼 penalty를 분리했으며, 안정적인 cosine-space 곡률 gradient와
  pure-PyTorch 8 tests를 통과했다. 기존 실행은 비교 기준으로 보존한다.

### 복사 시점의 masteer 기록

### ms18 호환 Joint Trajectory Predictor V1

- `trajectory_predictor/`에 Isaac Gym 비의존 state schema, Free/Cross CPU
  sampler, 9-candidate joint oracle, dataset I/O, anchored 33-point Transformer,
  고정 loss, strict checkpoint, train/open-loop eval CLI와 단위검사를 추가했다.
- `HumanoidMAPlannerCarry`는 기존 `HumanoidMASteerCarry`를 상속하고 경로·속도
  버퍼 공급자만 바꾼다. ms18 policy 관측/보상/network/PTH에는 손대지 않으며 reset,
  lift phase 전환, 매 6 action step에 refreshed state로 joint replan한다.
- predictor 출력은 기존 `steer_path.resample()`의 `[320,2]`와 4-class 속도
  profile로 설치되어 ms18의 6점/12-D root-local steer token 계약을 유지한다.
- checkpoint는 schema/model/normalizer/DS/V/속도/dataset hash를 fail-closed로
  검사한다. 첫 invalid는 analytic fallback, 이후 invalid는 직전 valid plan 유지이며
  종료 시 invalid/fallback 비율을 출력한다.
- `gt/oracle/analytic/learned` provider와 closed-loop state 수집 및 episode 단위
  DAgger relabel/fine-tune 경로를 함께 추가했다.
- predictor 학습은 기본 5 epoch마다 atomic numbered PTH를 보존한다. 별도
  `watch_videos` 프로세스가 같은 고정 validation Free/Cross 장면의 oracle/예측
  경로와 예측 속도 이동을 MP4로 만들므로 인코딩 중에도 GPU 학습은 다음 epoch를
  계속한다. 이 과정은 CPU 2D 렌더라 Isaac Gym이나 별도 GPU context를 사용하지 않는다.
- CPU 단위검사 8개와 tiny-data one-epoch end-to-end smoke를 통과했다. 실제
  `tokenhsi` 환경에서도 GT provider와 별도 smoke predictor provider가 기존 ms18 PTH를
  변환 없이 로드해 여러 599-step episode를 실행했다. 전체 200k/20k/20k pretrain과
  정식 closed-loop 합격 판정은 별도 실행 단계다.

## 2026-08-13

- 16:00  EXPERIMENTS.md
- 16:00  CLAUDE.md

## 2026-08-16

- 17:39  tokenhsi/env/tasks/base_task.py
- 17:39  tokenhsi/env/tasks/humanoid.py
- 17:39  tokenhsi/env/tasks/vec_task.py

### P0-1 step 1 — 에피소드를 env 단위로, 종료를 agent 단위로 분리

**결정**: `reset`/`progress` 는 per-env, 넘어짐(`terminate`)은 per-agent 로 계산 후 env 로 축약.
per-agent reset 을 버린 이유는 셋이다 — (1) 리스폰이 아직 작업 중인 상대 위로 떨어질 수 있고
(2) M2 가 넣을 teammate 관측에 물리적으로 불가능한 순간이동이 섞이며 (3) makespan("늦게 끝난
쪽")·deadlock("둘 다 정지")·`success_both` 가 전부 같은 에피소드를 전제해서 경계가 다르면
정의가 성립하지 않는다. 대가는 먼저 끝난 에이전트의 잉여 프레임인데, **단일 baseline 이 이미
성공 후에도 600 프레임을 다 돌기 때문에** 회귀가 아니다.

`TeamHOI/teamhoi/env/tasks/humanoid_multi.py` 가 같은 선택을 하고 있었다 (`_terminate_buf`,
`reset_buf`, `progress_buf` 전부 `num_envs`). 독립적으로 같은 결론이 나온 것.

**코드 쓰기 전에 전제부터 쟀다.** IsaacGym 없이 같은 메모리 배치를 torch 로 만들어
6 가지를 확인 — `(E,A,D)` 슬라이스는 창, `.reshape(E*A,D)` 는 복사본, **env 단위 쓰기
`h[env_ids] = …` 도 창으로 남는다**, per-agent box 도 같은 방식으로 잡힌다, 평탄 복사본의
값 정합성 OK. 리셋 경로가 창으로 남는다는 게 이 설계의 핵심 전제였고 그게 확인됐다.

**변경**
- `base_task.py` — `reset`/`progress`/`randomize` 를 `_rows`(E×A) 에서 `num_envs` 로 내렸다.
  `obs`/`states`/`rew` 만 `_rows` 로 남는다. 이 분리가 멀티에이전트 설계 자체다
- `humanoid.py` — `_humanoid_actor_ids_per_env` (E,A) 추가. 서브클래스 20여 곳이 평탄
  `_humanoid_actor_ids + idx` 로 물체 id 를 만들고 있어서 **기존 평탄 테이블은 그대로 두고**
  리셋 경로만 새 축을 쓴다. `_reset_env_tensors` 가 `[env_ids].flatten()`
- `humanoid.py` — `_compute_reset` 을 per-agent 종료 → env 축약으로. jit 함수는 안 건드렸다
- `vec_task.py` — `dones` 를 `repeat_interleave(A)` 로 row 수에 맞춤. `reset()` 액션 shape 도 row 기준

**A=1 회귀 (128 trials × 3, `--eval_task carry --seed 0`)**: 0.9844 / 0.9844 / 0.9688,
평균 **0.9792**. 원본 baseline 0.964, 기존 3 단계 0.958/0.971/0.958, 노이즈 ±0.02 → 구분 안 됨.

**단, 이 회귀가 검증하지 못한 것 두 가지를 명시한다.**
1. `humanoid._compute_reset` 은 **실행되지 않았다** — `HumanoidTrajSitCarryClimb` 이 override 한다
   (line 1969). 그 버전의 agent-aware 화는 step 2 항목이다
2. A=1 에서는 `num_envs == _rows` 라 base_task 변경이 **구조적으로 no-op** 이다. 이 회귀는
   "안 깨졌다"만 증명하고 A≥2 의미론은 증명하지 못한다 (PITFALLS #11 과 같은 함정)

**다음 (step 2)**: 시뮬 텐서 평탄화 제거 → `(E,A,…)`. 읽기용 평탄 뷰는 `_refresh_sim_tensors()`
직후 **스텝당 한 번만** 만든다 (freshness 계약을 한 곳에 묶고, A=1 에서는 reshape 이 view 라
복사 자체가 없다). 그리고 multi_task 의 `_compute_reset` agent-aware 화.
- 17:50  tokenhsi/env/tasks/humanoid.py

### P0-1 step 2 — 평탄화 제거, agent 축 복원 (base 레이어 완료)

평탄화를 걷어내고 `(E, A, …)` 창으로 되돌렸다. **핵심은 `A == 1` 일 때 `squeeze(1)` 로
agent 축을 없앤다는 것** — 그러면 단일 에이전트 코드가 보던 모양과 한 톨도 안 다르고,
agent 축은 A≥2 에서만 생긴다. 서브클래스 11개가 이 텐서들을 232곳에서 읽고 있어서
전역으로 모양을 바꾸면 전부 깨지는데, 이 방식이면 손댈 범위가 multi_task 하나로 국한된다.

읽기용 평탄 복사를 스텝마다 만들려던 앞선 계획은 **폐기했다.** 필요가 없다 —
`_reset_actors` 의 `self._humanoid_root_states[env_ids] = …` 가 env 인덱스라
`(E, A, …)` 창에 그대로 써지고 시뮬에 도달한다 (view_check 테스트 3에서 확인).
복사본을 만들 이유가 애초에 없었다.

**A=1 회귀**: 0.9844 / 0.9844 / 0.9688 — step 1 과 **시행 수까지 완전히 동일**
(126/126/124). squeeze 가 진짜 no-op 이라는 증거.

### A=2 스모크 — base 통과, task 레이어에서 멈춤

`numAgents: 2`, 64 env. env 생성·텐서 획득·물리 초기화까지 **전부 통과**했고
첫 task 소비자에서 죽었다:

```
_build_traj_generator:790   root_pos = self._humanoid_root_states[:, 0:3]
→ (E, A, 13) 에서 [:, 0:3] 은 agent 축을 자른다 → traj_gen 이 (E,2,13) 을 받음
```

기계적인 부분은 `[:, X:Y]` → `[..., X:Y]` 패턴이지만, **진짜 문제는 그게 아니다.**
traj/sit/carry/climb 의 태스크 상태가 전부 per-env 로 잡혀 있다 — traj_gen 은 env 당
궤적 하나, box·target 도 env 당 하나. A=2 면 각 에이전트가 자기 것을 가져야 한다.

### 다음 사이클의 결정 지점 — 여기서 갈린다

| | 방법 | 대가 |
|---|---|---|
| A | `multi_task` 2000줄을 통째로 A-aware 화 | 4개 스킬이 얽혀 있고, A=1 회귀의 기준 태스크 자체를 건드리게 된다 |
| B | `HumanoidTrajSitCarryClimb` 를 상속한 **per-agent carry 전용** 서브클래스 | 새 파일. 기준 태스크는 안 건드림. obs 레이아웃은 상속해서 stage1 체크포인트 호환 유지 |

**B 로 기운다.** 이유: (1) PLAN 이 "Carry skill 만, 다른 skill 은 범위 밖"이라고 못 박았다
(2) A=1 회귀가 계속 성립하려면 기준 태스크가 안 변해야 한다 (3) M0~M7 이 필요한 건
carry 하나뿐인데 sit/climb 까지 per-agent 로 만드는 건 순수 낭비다.
단 obs 레이아웃은 stage1 체크포인트가 기대하는 multi_task 형태를 유지해야 하므로
**상속**이지 새로 쓰는 게 아니다.
- 17:58  tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py

### A/B 결정 정정 — 부모를 좁게 고치고, per-agent box 만 서브클래스로

앞 항목에서 "B(서브클래스)로 기운다"고 썼는데 실제로는 **둘을 갈랐다.** 근거는
태스크 상태가 두 종류라는 것이다:

| | 무엇 | 어디서 |
|---|---|---|
| **가상** | 궤적·타겟·태스크 지시자 — 그냥 숫자 | **부모**를 per-row 로. A=1 에서 `_rows == num_envs` 라 no-op |
| **물리** | box·의자 — 실제 actor | **서브클래스**에서 A 개 생성 (P0-2) |

가상 상태까지 서브클래스로 미루면 부모의 per-env 가정이 그대로 남아 어차피 부딪힌다.
그리고 부모 쪽 변경은 A=1 회귀로 매번 검증되므로 안전하다. 물리 쪽만 B 안으로 간다.

**변경**
- `humanoid_rows(t)` 헬퍼 — 시뮬 텐서의 agent 축을 row 로 접는다 (A=1 이면 그대로).
  읽기 전용이라는 걸 docstring 에 못 박았다
- `_build_traj_generator` — `num_envs` → `_rows`. 이제 에이전트마다 자기 궤적을 갖는다
- `_set_env_state` — **쓰기 경로**라 반드시 `(env, agent)` 창에 써야 한다.
  `[env_ids, 0:3]` 이 A≥2 에서 agent 축을 자르고 있었다 → `[env_ids, ..., 0:3]` +
  값을 `(len(env_ids), A, …)` 로 접어서 대입

### A=2 진행 상황 — 벽이 한 칸씩 뒤로 밀린다

```
1차  _build_traj_generator   root_pos 축 오류        → 고침
2차  _set_env_state          리셋 쓰기 축 오류        → 고침
3차  _reset_ref_state_init   (다음) 모션 샘플을 env 당 1개만 뽑는다. A개 필요
```

**다음 사이클**: `_reset_ref_state_init` (1804) 이 `len(env_ids)` 개가 아니라
`len(env_ids) * A` 개 모션 프레임을 뽑게. 그 뒤 A=1 회귀 재확인 (부모를 건드렸으므로).

### 밤 상황 정정 — GPU 는 2시에 안 빈다

steer F11 훈련 8개가 18시경 정상 종료(로그 끝 `saving checkpoint`)하고 평가로 넘어갔다.
크래시 아님. 그런데 **PLAN_steer 큐에 9줄이 남아 있다** — `f11_C_a~d` + `f13_*` 5개.
`f11_A` 가 6000 iter 에 11.5 시간 걸렸으므로 18:30 에 시작한 `f11_C_a` 는 내일 06시경 종료.
autofill 은 정상 가동 중(1일 20시간).

따라서 **오늘 밤 ma 의 GPU 창은 없다.** 5분짜리 A=1 회귀만 한 칸 얹어 쓴다(2잡 허락받음).
밤새 할 일은 코드이고, 그게 어차피 병목이었다. GPU 감시는 autofill 로그의
`queue empty` 를 보도록 다시 걸었다 — steer 큐가 진짜 소진되는 순간의 신호다.
- 21:36  tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py

### 리셋 경로를 row 공간으로 — 벽 3개 더 밀어냄

**A=1 회귀 3연속 통과.** 126/126/124 로 step1·step2 와 시행 수는 물론
`success_precision` 소수점까지 동일하다. 부모 클래스를 계속 건드리는데도 A=1 경로가
비트 단위로 안 변한다는 뜻이라, 이 회귀가 계속 유효한 검증으로 쓰인다.

**설계 확정 — 쓰기는 두 인덱스로 창에 꽂는다.** `_set_env_state` 를 row 공간으로 바꾸면서
`h[row]` 가 아니라 `h[row // A, row % A]` 로 쓴다. 평탄화하면 복사본이 되어 조용히
사라지지만, 인덱스 쌍으로 주면 advanced indexing 이 창에 그대로 써준다.
`agent_axis()` 가 A==1 에서 크기 1 축을 끼워주므로 (여전히 view) 분기 없이 한 줄로 돈다.

**per-row 로 올린 것**: `_task_indicator`, `_task_mask` (obs 가 row 당 하나라 필수),
`_every_env_init_dof_pos`, `_kinematic_humanoid_rigid_body_states`, traj generator.
헬퍼 4개 추가 — `agent_rows()` (env→row), `agent_axis()` (쓰기용 창),
`humanoid_rows()` (읽기용 평탄화), `progress_rows()` (env 시계를 row 로 확장).

```
1차  _build_traj_generator   궤적 축          → 고침
2차  _set_env_state          리셋 쓰기 축      → 고침
3차  _reset_task_indicator   태스크 배정 축     → 고침 (에이전트마다 자기 모션)
4차  _reset_task_traj        궤적 리셋 축      → 고침
5차  _reset_task_sit         ← 지금. CUDA assert (인덱스 범위 초과)
```

### 5차 벽이 드러낸 것 — 가상/물리 경계가 여기다

`_reset_ref_env_ids` 가 이제 row id 인데 `_sit_object_states` 는 **env 당 물체 하나**라
env 단위다. row(=2×env) 로 인덱싱하니 범위를 넘었다. 예상했던 경계가 실제로 여기서 터진 것.

**규칙을 명시한다.**

| 상태 | 인덱스 | 예 |
|---|---|---|
| 가상 (숫자) | row | 궤적·타겟·태스크 지시자·모션 샘플 |
| 물리 (actor) | `row // A` | sit object, climb object, **box** |

**다음 사이클**: `_reset_task_sit` / `_reset_task_carry` / `_reset_task_climb` 에서
물리 텐서를 `row // A` 로 인덱싱하게. 이러면 A=2 에서 두 에이전트가 같은 물체를 가리키는데,
sit/climb 은 범위 밖이라 무해하고 **carry 의 box 가 바로 P0-2 가 고칠 지점**이다.
즉 이 단계를 지나면 "크래시 없이 돌지만 상자를 공유하는" 상태가 되고, 그게 P0-2 의 출발점이다.

### GPU 상황

22시 기준 steer 큐 9줄 전부 투입 완료. `f11_C_a~d`(6000 iter)는 내일 06시경,
`f13_*`(3000 iter)는 새벽에 끝난다. 중간중간 2~3장이 비어서 ma 회귀·스모크는
그 빈 칸에서 돌리고 있다 (steer 런에 얹지 않음). autofill 은 `queue empty` 를 매분 찍어서
모니터가 시끄러웠고, 장애 신호만 보도록 다시 걸었다 — GPU 여유는 사이클마다 직접 확인한다.

### P0-1 — 리셋 경로 완주, 관측 경로 진행 중

오늘 벽 8개를 뚫었다. 순서대로: 궤적 축 → 리셋 쓰기 축 → 태스크 배정 축 → 궤적 리셋 축
→ 물체 인덱싱 → per-row 버퍼 → 물체 포즈(모션 A배) → 관측 row 공간.
**리셋 경로는 전부 통과**했고 지금은 `_compute_amp_observations` 가 아직 env 단위다.

**구조 결정을 도중에 한 번 뒤집었다.** 처음엔 "가상=row, 물리=env" 로 잡고 `env_of()` 로
23곳을 감쌌는데, 그러면 **한 env 의 두 에이전트가 서로 다른 태스크를 받았을 때 물체 배치가
모순**된다 — 한 명은 의자를 쓰겠다는데 다른 한 명이 그 의자를 땅속으로 치운다. 되돌리고
이렇게 다시 잡았다:

| | 단위 | 이유 |
|---|---|---|
| 태스크 · 스킬 | **env** | 물체가 env 당 하나다. 에이전트별로 다르면 배치가 모순 |
| 모션 시작 프레임 | **agent** | 그래야 두 에이전트가 다른 자세·다른 위치에서 시작한다 |
| 관측 · 보상 | **row** | row 당 한 줄 |

이 구조에서는 `env_of()` 가 통째로 필요 없어진다. 헬퍼는 `agent_rows`(env→row),
`agent_axis`(쓰기용 창), `humanoid_rows`(읽기용 평탄화), `per_env_rows`(env 값을 row 로 복제),
`agent0`(물체용 대표 선택), `task_of`(env 의 태스크) 여섯이고 앞 넷은 `Humanoid` 로 올렸다.

**P0-2 가 왜 필요한지가 코드로 드러났다.** 물체는 env 당 하나인데 모션은 에이전트마다
다르다. 물체 포즈를 정하려면 **둘 중 하나를 골라야 해서** 지금은 agent 0 기준으로 놓고
agent 1 은 물체와 안 맞는 자세로 시작한다 (`agent0()` 헬퍼). 버그가 아니라 per-agent
물체가 없는 동안의 필연이고, docstring 에 그렇게 적어뒀다.

**아직 A=1 회귀를 안 돌렸다.** 오늘 `humanoid.py` 와 multi_task 를 크게 건드려서, 3연속
통과했던 그 검증이 유효한지 확인이 안 됐다. **A=2 가 끝까지 돌기 전에 A=1 부터 다시 봐야 한다.**

### ma 환경 설정 — 확정

접촉쌍은 ma 에서 제약이 아니다. A=1 격자 12점이 전부 warn=0 이고, **양성 대조군으로 넣은
1 M 조차 안 넘친다**(4096 env, 60 iteration). `envSpacing` 을 0 으로 해도 안 넘친다.

속도(동거 상태 상대 비교, minibatch 를 envs×4 로 고정해 학습 등가):

| num_envs | minibatch | 정상상태 fps | 배율 |
|---|---|---|---|
| 1,024 | 4,096 | 3,937 | — |
| 2,048 | 8,192 | 6,996 | 1.78× |
| 4,096 | 16,384 | 11,600 | 1.66× |
| 8,192 | 32,768 | 17,791 | 1.53× |
| 16,384 | 65,536 | **25,849** | 1.45× |

**16,384 까지 계속 오른다.** steer 는 8192 에서 128 M 버퍼가 강제라 오히려 느려졌는데
ma 는 그 벽이 없다. 다만 이 수치는 카드를 steer 와 나눠 쓰며 잰 것이라 **절대값은 못 쓰고
상대 비교만 유효**하다. 확정은 P0-2 뒤에 A_max 기준·단독 카드에서.

### ✅ P0-1 완료 — A=2 가 끝까지 돈다

`numAgents: 2`, 64 env 로 크래시 없이 에피소드를 완주한다 (`reward: … steps: 114/203/230`).
**A=1 회귀 5연속 통과** — 126/126/124 로 시행 수가 처음과 동일하다. 오늘 base 클래스와
multi_task 를 100 곳 넘게 고쳤는데도 단일 에이전트 경로가 비트 단위로 안 변한다.

**중간에 방식을 바꾼 것이 결정적이었다.** 크래시를 하나씩 잡는 방식으로 22 번을 돌렸는데
남은 개수를 모른 채였다. 대신 **초기화 직후 모든 텐서를 축별로 찍는 덤프**를 넣었다
(`Humanoid.dump_agent_layout`, `MA_DUMP=1`). 한 번에 전체 지도가 나왔고:

```
env (41)       물체(박스·의자·climb·플랫폼) · 에피소드 시계
(E,A,·) (13)   시뮬 창 -- 전부 정상
row (22)       관측 · 보상 · AMP · actor id
기타 (17)      상수 테이블
```

**그 표가 즉시 숨은 버그를 잡았다.** `_box_actor_ids` 등 물체 actor id 5 개가 row(128)
크기였다. `_humanoid_actor_ids + idx` 로 만들어서 사람 수를 따라간 것인데, 물체는 env 당
하나다. 개수뿐 아니라 **값도 틀렸다** — agent 1 쪽이 `e*num_actors + 1 + idx` 가 되어
옆 actor 를 가리킨다. 이걸로 `set_actor_root_state_tensor_indexed` 를 부르면 엉뚱한 물체를
옮기는데 **크래시가 안 난다.** 표를 안 뽑았으면 "왜 박스가 이상하지" 로 며칠 태웠을 것이다.

**남은 수정**
- 물체 actor id 8 곳 → `_humanoid_actor_ids_per_env[:, 0] + idx` (env 단위)
- `multi_task._compute_reset` → row 로 판정 후 `amax(dim=1)` 로 env 축약 (base 와 동일 규칙)
- `Humanoid.reset()` → 학습 루프가 넘기는 row 인덱스를 env 로 접는다. dones 를 row 로
  broadcast 했으므로 되돌아오는 인덱스도 row 다
- AMP 버퍼 2 곳 (base + multi_task 가 각자 잡고 있었다)
- IET 버퍼 → row

**A>=2 는 carry 전용으로 고정했다.** traj/sit/climb 의 리셋·보상은 "env 당 물체 하나 =
사람 하나" 를 전제하는데, M0~M7 은 carry 만 쓴다. 안 쓰는 스킬을 A-aware 하게 만들면
**돌려볼 방법이 없는 코드**가 생기고 검증 못 하는 코드는 조용히 틀린다. `_task_init_prob`
을 carry 로 고정해 나머지 분기가 아예 실행되지 않게 했다. 이 결정으로 남은 작업이 절반
이하로 줄었다.

**다음 (P0-2)**: 덤프의 `env` 칸에서 박스·플랫폼 관련 12 개를 row 로 옮긴다.
`_box_states / _box_tar_pos / _initial_box_states / _prev_box_pos / _box_lib._box_{bps,scale,size} /
_platform_{states,pos,default_pos} / _tar_platform_{states,pos,default_pos}`.
의자·climb 21 개는 carry 범위 밖이라 그대로 둔다. 그러면 `agent0()` 헬퍼가 사라진다 --
물체 포즈를 정할 때 에이전트 하나를 골라야 했던 이유가 없어지기 때문이다.

## 2026-08-18

- 00:08  tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py
- 00:08  tokenhsi/env/tasks/humanoid.py
- 00:21  tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py
- 00:54  tokenhsi/run.py
- 00:56  tokenhsi/learning/amp_agent.py

### M0 통과 — A=2 성공률 0.61 의 진범은 `loco_carry` 리셋의 env/row 혼동

`A=1 0.96` vs `A=2 0.61` 의 격차를 끝까지 파서 닫았다. **결과: 0.9648 vs 0.9668 / 0.9609.**

진범은 `_reset_task_carry` 의 `loco_carry` 분기였다. `_reset_ref_env_ids` 는 **env id** 를
담는데(스킬은 env 단위로 뽑고 모션만 env 당 A 개 뽑는다) 그 분기만 `ids // num_agents` 로
**row 취급**했다. 바로 위 `pickUp` 분기는 같은 딕셔너리를 env 로 읽는다 — 한 파일 안에서
같은 값을 두 가지로 해석하고 있었다. env id 5 를 row 로 읽으면 `env 2, agent 1` 에 박스를
놓고 나머지 에이전트 박스는 아예 안 놓는다. **`//1` 이 항등이라 A=1 에선 완전히 숨는다.**

가는 길에 고친 것:
- `apply_layout` 이 받침대(platform·tar_platform)를 안 옮겼다. `carryResetRandomHeight` 는
  박스를 받침 위에 올려두고 시작하는데 박스만 옮기니 **공중에서 떨어져 굴러갔다.**
  리셋 직후 박스 속도가 0 이 아닌 것으로 관측에서 잡혔다 (carry 블록 281-286).
- `_reset_env_tensors` 의 `_box_actor_ids[env_ids]` → `[rows]`. row 크기 표를 env 로 인덱싱.
- `_compute_reset` 이 성공(IET)과 실패를 함께 `amax` 로 접었다. 먼저 끝낸 에이전트가 env 를
  끝내 파트너를 실패로 만든다. 성공/실패를 갈라 접게 고쳤다 (이 config 는 IET 가 꺼져 있어
  측정엔 영향 없었지만 켜지면 바로 터진다).

**측정 방법론에서 배운 것 — 대조군이 무효였다.** "A=1 + 내 배치 = 0.98" 로 배치를 무죄
판정했는데, `apply_layout` 이 `num_agents < 2` 에서 즉시 return 하고 있었다. **A=1 대조군은
배치를 한 번도 적용한 적이 없다.** 좌표를 찍어보고서야 알았다. 실제로 배치를 A=1 에 적용하면
0.96 → 0.69 로 떨어진다 — `pickUp`/`carryWith` 는 참조 모션에서 이미 박스를 든 자세로
시작하는데 배치가 박스를 사람 앞 0.8 m 로 재배치해 손-박스 관계를 끊기 때문이다.
**대조군은 "돌렸다"가 아니라 "실제로 달랐다"를 확인해야 한다.**

그래서 M1 축을 `MA_LAYOUT` → `MA_SEP` 으로 교체했다. `MA_SEP` 은 사람·박스·받침·목표를
**같은 벡터만큼** 평행이동만 해서 상대 기하가 안 변한다 → 분포 안에 남는다.
`MA_LAYOUT` 은 정책이 학습된 뒤의 M5·M6 용으로 남긴다.

### A=2 학습 경로 개통 + 팀 보상 구현

`--test` 만 되고 학습은 한 번도 안 돌려봤던 게 드러났다. 세 곳이 막혀 있었다:
- `run.py:get_env_info()` 가 `agents` 를 안 넘겨 rl_games 가 `num_agents=1` 로 batch 를
  계산 → minibatch 와 안 맞아 assert. **평가 경로는 에이전트 수를 다른 데서 읽어 안 걸렸다.**
- `_build_rand_action_probs` 가 env 크기로 만들어져 row 크기 액션과 안 맞음. row 로 편다.
- (batch 규칙) `num_actors` 는 cfg 의 `numEnvs` 에서 온다. `--num_envs` 로 덮어도 안 따라온다.

팀 보상 `_apply_team_reward` 를 넣었다. `MA_C` `MA_BETA` `MA_K`, τ 는 지표의 `MA_TAU` 를
그대로 재사용한다 (재는 정의와 벌주는 정의가 같아야 어긋나지 않는다).
c=β=0 이면 즉시 return 하므로 `m3_zero` 대조군은 원본과 비트 단위로 같다.
수치 확인: `d ≥ τ` 에서 벌점 정확히 0, 공유가 env 내 합을 보존(천장 불변).
학습 스모크 4칸(c0b0 / c05 / b05 / c05b25) 전부 완주, fps ~3,000 @ 256env·A=2.

### 밤새 돌린 M3-0 4시드는 전부 무효 — `--resume` 없이는 조용히 랜덤 초기화로 학습한다

4시드 전부 `rc=0 warn=0` 으로 완주했고 fps 도 정상이었는데 **평가가 0.0** 이었다.
원인 두 개가 겹쳤다:

1. **체크포인트가 아예 안 올라왔다.** `--checkpoint` 만 주면 `load_checkpoint` 설정이
   꺼져 있어 rl_games 가 복원을 **조용히 건너뛴다** (`torch_runner.py:123`). stage1 에서
   이어 학습하려면 `--resume 1` 과 `--checkpoint` 가 **둘 다** 필요하다
   (`utils/config.py:125-129`). `--hrl_checkpoint` 는 stage2 합성 태스크가 stage1 을
   별도 low-level 정책으로 쓸 때의 플래그라 여기선 틀리다 — 원본 스크립트를 보고
   그걸로 갈아탔다가 한 번 더 헛돌았다.
2. **1000 iter 는 너무 짧다.** steer 관례가 3000~6000 인데 "밤새 도는 시간"에 맞춰
   임의로 1000 을 잡았다. 근거 없는 숫자였다.

**`rc=0` 도 `warn=0` 도 "학습이 됐다"를 뜻하지 않는다.** 로그에 `=> loading checkpoint`
가 찍혔는지가 실제 통과 조건이다. 검증: 고친 뒤 **20 iter** 만 돌려 평가했더니 0.8789 로
학습 전 0.8767 과 일치 — 파이프라인이 stage1 에서 이어간다는 확인이다. **6000 iter 짜리를
다시 넣기 전에 20 iter 로 먼저 확인하는 게 맞다** (이번에 4시드 × 1.7시간을 태웠다).

`train.sh` 에 `MA_MODE=train` 을 추가하면서 같이 뚫은 것들 — A=2 학습 경로는 오늘까지
한 번도 안 돌려본 상태였다:
- `run.py:get_env_info()` 가 `agents` 를 안 넘겨 rl_games 가 batch 를 절반으로 계산
- `_build_rand_action_probs` 가 env 크기라 row 크기 액션과 안 맞음
- `num_actors` 는 cfg 의 `numEnvs` 에서 온다 — `--num_envs` 로 덮어도 안 따라온다
- 10:21  tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py

### 뷰어 경로 A=2 대응 — 그리기·카메라·마커 세 곳

`--headless` 를 빼면 A=2 에서 즉시 죽는 상태였다. headless 로만 돌려와서 안 드러났다.
가상 디스플레이(`xvfb-run`)로 실제로 태워서 하나씩 잡았다 — **정적 분석으로는 두 번째
세 번째가 안 보였다** (첫 크래시가 뒤를 가린다).

- `_draw_task`: `self._box_states[:, 0:3]` 이 A=2 에서 상태가 아니라 **에이전트 축**을
  자른다 (rot 은 빈 텐서). row 로 펴고 reshape 을 `(-1, 8, 3)` 으로. 그리기 루프는
  env 당 A 쌍(박스+목표)을 그리게 했다.
- `_init_camera` / `_update_camera`: `_humanoid_root_states[0, 0:3]` 이 xyz 가 아니라
  **앞 세 에이전트**를 고른다. `humanoid_rows(...)[0, 0:3]` 로.
- `_update_marker`: 마커는 env 당 하나인데 궤적 표본·`_task_indicator` 는 row 당 하나.
  A>=2 는 carry 전용이라 traj 마커는 시각용일 뿐이므로 agent 0 것만 그린다.

A=2·A=1 둘 다 뷰어에서 크래시 없이 뜬다. 뷰어는 `--num_envs` 16 이하로 쓴다.

### `pkill -f` 로 또 자기 셸을 죽였다 (exit 144, 이 세션 두 번째)

`pkill -f "xvfb-run"` 이 그 명령을 실행 중인 셸 자신의 커맨드라인에 매칭됐다.
PITFALLS 운영사고 #1 과 같은 것이고, **이 세션에서 이미 한 번 내고 기록까지 해놓고
또 냈다.** 이번엔 데몬 3개와 학습 4개가 전부 살아남았지만 (분리 실행이라) 운이었다.
`pgrep -f -- "--track $t --loop"` 처럼 **자기 명령줄에 안 나타나는 패턴**으로만 찾거나,
PID 를 먼저 확보해 `kill <pid>` 로 죽인다.
- 11:03  tokenhsi/env/tasks/adapt_interaction_skills/humanoid_adapt_carry_ground2terrain.py
- 11:12  tokenhsi/env/tasks/adapt_interaction_skills/humanoid_ma_carry.py

### 큐·GPU 운영 규칙을 코드 게이트로 전환

공통 규칙을 `docs/OPERATIONS.md`로 모으고 ma 큐의 허용 환경변수와 동일 설정당 시드
2개 상한을 track JSON에 선언했다. GPU 지도는 이제 `t1_*` 같은 태그도 `output/ma/`
경로로 정확히 분류하며, `check_docs.py`가 잘못된 큐·마커·경로면 데몬 시작을 막는다.
현재 잡과 데몬은 종료하거나 재시작하지 않았다.

### 큐 생명주기와 자동 종료 원장 분리

사용자가 PLAN 큐를 만들고 자동화가 상태/GPU 열과 완료 행 소비를 맡도록 공통 관리 코드를
분리했다. MA는 로그의 `MA_SUMMARY`와 종료 코드를 상태 JSON 및
`experiments/AUTO_RESULTS_ma.md`에 먼저 보존한 뒤 활성 큐 행을 제거한다.
기존 학습은 유지하고 큐 데몬만 검증 후 재시작해 새 동작을 활성화했다.

### 활성 문서 경로 전환

ma 큐와 검토 결과의 활성 경로를 `plan/PLAN_ma.md`와
`experiments/EXPERIMENTS_ma.md`로 분리했다. 레포 내부 결과 원장은 읽기 전용 보존본으로
남겼고, 경로 검사와 큐 데몬 재시작 뒤 기존 학습이 유지되는 것을 확인했다.

### adapt 학습의 최종 평가 자동화

`MA_MODE=train/adapt` 행은 학습 종료만으로 소비하지 않고, 학습 sidecar·생성 cfg·최종
체크포인트를 512 env로 재구성한 평가가 `MA_EVAL_SUMMARY rc=0`을 남긴 뒤에만 완료한다.
기본 실행 프로필은 2048 env × 2 agents, `envSpacing=5`, contact pairs 4 M으로 고정했다.
실패 평가는 큐를 보존하고 30분 뒤 재시도한다. 현재 실행 중인 네 학습의 설정은 바꾸지
않았고, 큐 데몬만 재시작했다.

### 활성 큐 표 간소화

큐 열을 `tag | 질문 | 상태/GPU | 환경변수`로 바꾸고 adapt Carry의 공통 설정을 track
`base_env`로 올렸다. 행에는 토큰·시드·`c`·`β` 같은 변경축만 남긴다. 새 기본값 적용 전
시작된 `t1_*`의 contact 8M은 실제 설정을 잃지 않도록 override로 유지했다.

### 팀 보상 ①② 구현 — 축 B 를 갈아끼울 수 있게 했다

보상은 축이 둘이다. 축 A(충돌)는 `c·(pin(d_min)−1)` 하나로 확정됐고, 축 B(팀 결합)는
형태가 여럿이라 `MA_TEAM` 으로 고르게 만들었다.

```
share  mean_j(r_j)                     ③ 가중평균 (기존). 게으름이 **가능**하다
minp   Δmin_i(progress) × 에피소드길이   ② 뒤처진 쪽 기준. 게으름이 **불가능**하다
mkspn  종단 success_all·(T_ref/T_last)   ① sparse
both   minp + mkspn
```

**②가 게으름 대책 그 자체다.** ③은 한쪽이 놀아도 `β·r_j` 를 받아먹을 수 있지만,
②는 내가 놀면 내가 곧 `min` 이 되어 팀 항 자체가 0 이 된다. 그래서 "게으름 수정"과
"①② 투입"이 별개 단계가 아니라 하나다.

구현에서 조심한 것:
- **progress 는 수준이라 그대로 더하면 폭주한다.** 증분을 쓰고, 한 에피소드 합이 1 이
  되므로 `max_episode_length` 를 곱해 원본 carry 보상(스텝당 ~1)과 눈금을 맞췄다.
  안 맞추면 β 를 걸어도 팀 항이 300 분의 1 이라 아무 일도 안 일어난다.
- **후퇴는 벌하지 않는다** (`clamp(min=0)`). 박스를 놓쳐 뒤로 밀리는 건 원본 carry
  보상이 이미 처리하므로 여기서 또 빼면 같은 사건을 두 번 벌한다.
- **종단 보상은 `_compute_reset` 에서 `reset_buf` 가 정해진 뒤에** 준다. `_compute_reward`
  시점에는 이 스텝이 마지막인지 알 수 없다 (reset_buf 가 아직 이전 값이다).
  `update_metrics()` 가 `_compute_reset` 첫 줄이라 `_ep_finish` 는 그 스텝까지 반영돼 있다.
- **진척률의 분모는 `_refresh_sim_tensors` 뒤에** 다시 잡는다. 그 전에는 박스가 아직
  리셋 위치로 안 옮겨져서 직전 에피소드의 거리를 분모로 쓰게 된다 (조용히 틀린다).

`T_ref=294` 는 `b_solo` 의 단독 하한(t50) 실측값이다. 완전 병렬 1.0 / 순차 0.5 로
makespan 정의와 정확히 맞는다. 수치 확인 완료, 4 모드 전부 학습 스모크 통과.

### 1·2단계 판독 — 토큰은 무시됐고, c 는 애초에 크기가 안 됐다

8행 3000 iter 완주. 결론 둘.

**1단계**: `zero` 0.8723 vs `live` 0.8767, Δ=+0.004. **조건 내 시드 편차(0.019/0.014)가
조건 간 차이보다 크다** — 신호 없음. 보상에 이유가 없으면 정책은 토큰을 안 쓴다는
PLAN 의 전제가 실측으로 확인됐다.

**2단계**: 승자 없음(최고 Δ=+0.017, 문턱 0.05). 원인을 `_ep_task_r` 로 확정했다 —
**c=0.5 의 벌점 총량이 에피소드 보상의 0.08 %** (평균 근접으로 잡아도 1.8 %).
`근접/이동` 이 전 조건에서 1.62~1.93 으로 평평하다(학습 전 1.75 포함). c 는 축을
탐색한 게 아니라 **0 근처만 두 번 찍었다.** c=2/5 로 다시 간다.

**이번에 `_ep_task_r` 누적을 고쳐둔 게 값을 했다.** 그게 0 이었으면 "c 가 효과 없음"
까지만 알고 *왜* 인지는 못 짚었을 것이다. 지표가 죽어 있으면 원인 규명이 통째로 막힌다.

지표를 이동거리로 정규화해서 봐야 한다는 것도 이번에 드러났다. `근접에피`만 보면
25.9 %→40 % 라 "벌점이 역효과"로 읽히는데, 이동이 9.07→12.9 로 늘어난 부산물이었다.

## 2026-08-20

### GT 시나리오 — 배치·M 프로파일·지표

**결정**: `MS_SCEN` 하나가 배치 환경변수를 일관되게 정하고, 감속은 **교차점 앞 3 m
구간에서만** 건다.

배치는 `apply_layout` 의 `side`/`parallel`/`far` 를 재사용한다. 시나리오별로 변수를
따로 주게 두지 않은 이유는 조합이 조용히 어긋나기 때문이다 — `MA_SPAWN_GAP`(기본 1.0 m)
은 `apply_layout` **뒤에** 돌면서 `parallel 0.5` 를 통째로 다시 뽑아버린다.

**시나리오는 직선 경로를 쓴다** (`MS_LAT_MAX=0`). 실측:

    곡선(2.2)  교차점까지 호길이 4.299 m 대 4.597 m, 경로가 원점을 1.86 m 빗나감
    직선(0.0)  양쪽 4.500 m, 빗나감 0.000 m

곡선이면 `dt=0` 을 만들려고 한쪽 속도를 억지로 바꿔야 해서 **대조군이 대조군이 아니게
된다.** 직선이면 둘 다 평속으로 두는 것만으로 동시 도착이다.

감속을 전 구간이 아니라 국소로 한 이유: 전 구간 감속은 `MS_MRAND` 가 이미 하는
"느리게 걷기"라 새로울 게 없다. 국소 감속은 **속도를 바꿨다 되돌리는 전이**이고,
총 소요시간은 전 구간 감속과 같다 (둘 다 `경로/1.5 + dt`). 공짜로 더 나은 것을 얻는다.

    v_slow = W/(W/1.5 + dt)     구간 통과 시간이 정확히 dt 만큼 는다
    dt=1 -> M 배수 .67          dt=2 -> .49

감속 창은 **미터가 아니라 셀 인덱스로** 잡는다. `_m_at` 이 0.1 m 계단이라 미터로
자르면 실효 W 가 격자만큼 어긋나 오프셋이 3% 빗나간다 (프로토타입에서 +0.97s / +1.93s).

`MS_DT_RAND=1` 로 지연을 에피소드마다 뽑는다. **고정 dt 로 학습하면 안 된다** —
M 프로파일이 매번 같아서 정책이 "호길이 1.5~4.5 에서 느려져라"를 창을 읽지 않고
외운다. 그러면 창을 0 으로 해도 같은 값이 나와 "타이밍이 된다"가 공허해진다.

### apply_layout 이 env 원점 대신 각자의 생성 위치를 썼다

사람은 env 원점 둘레 반지름 1.5 m 원 위에 생성된다 (`humanoid.py`: `1.5*cos/sin`).
`apply_layout` 이 그 위치를 각자의 원점으로 써서 A=2 에서 두 배치가 3 m 어긋났다.

    side      교차하긴 하지만 교차점까지 호길이가 1.5 m 대 4.5 m
    parallel  나란한 게 아니라 한쪽이 3 m 앞선 대각선

생성 오프셋은 원 위에 대칭이라 **에이전트 평균이 곧 env 원점**이다. 그걸로 고쳤다.
ma 트랙의 `MA_LAYOUT` 런들은 고치기 전 코드로 돌았다.

### 지표 확장 훅과 시나리오 열

부모에 `_metric_extra_cols` / `_metric_reset_extra` 를 두어 하위 클래스가 열을 뒤에
덧붙이게 했다. 시나리오 열은 13~19 다.

    13 xtime  교차점에 가장 가까웠던 스텝 (정확히 통과 안 해도 정의된다)
    14 xdist  그때의 거리. <1.5 m 인 행만 도달로 친다
    15 encd   **조우 중** 상대와의 최소거리 (교차점 MS_ENC_R 이내일 때만)
    16 wn / 17 wv / 18 wc  감속 구간 체류 스텝과 실제·명령 속도 합
    19 dtcmd  그 에피소드에 명령한 지연. 반응 곡선의 축

`encd` 를 따로 둔 이유: 에피소드 전체 최소거리는 경로 대부분이 서로 멀어 **평평해진다.**
ma 에서 17 조건 전부 1.62~1.93 이 나온 것이 그 때문이다. 교차점 근처로 자르면 희석이 없다.

실제 속도는 **호길이 증가율**로 잰다. 경로상 속도의 정의 그 자체이고 명령 `M/1.6` 과
직접 비교된다. xy 속도 크기를 쓰면 옆으로 비키는 성분이 섞인다.

검증 (64 env, 45 iter, 미학습 스캐폴드):

    배치      양쪽 호길이 4.500 m, 빗나감 0.000, 곡률 0.000, a1 만 M 배수 0.500
    명령      dbg_sync  a0 1.500 / a1 1.500
              dbg_cross a0 1.500 / a1 **0.750**
    실제      둘 다 0.56 -- 미학습이라 아무것도 못 따른다 (귀무 상태가 구분된다)
    조건차    parallel 1.0 은 colEp 0.566 dmin 0.28, cross 는 0.021 / 3.10

### scripts/masteer 의 지표 이름이 틀렸다

`train.sh` 와 `eval_one.sh` 가 `MS_METRICS` 만 export 했는데 env 코드가 읽는 이름은
`MA_METRICS` 다. **masteer 지표가 한 번도 안 쓰였다** (`runs/results/masteer/` 가 계속
비어 있었다). 둘 다 내보내도록 고치고, `eval_one.sh` 가 masteer 결과를
`runs/results/ma/` 에 쓰던 것도 `runs/results/masteer/` 로 바로잡았다.

### 시나리오 설계 교정 — 회복 구간·행 인덱싱·단일 출처

**교차 속도 교란.** 감속 창이 교차점에서 끝나면 감속한 쪽이 **느린 채로 교차점을 지난다.**
교차 구역 체류시간이 0.67 s → 1.33 s 로 두 배가 되므로 "늦게 왔다" 와 "느리게 지났다" 가
같은 조작이 되고, 여유거리 차이가 순전히 운동학적 이유로 생긴다. `MS_RECOV=1.5` 로
창과 교차점 사이에 **평속 회복 구간**을 둬서 모든 조건이 교차점을 1.5 m/s 로 지나게 했다.
자리를 만들려면 경로가 길어야 해서 `MS_L` 을 9 → 12 로 올렸다 (9 면 회복 구간이
접근 다리를 통째로 삼킨다).

**`_m_at` 이 부분 rows 를 arange 로 인덱싱했다.** `_steer_obs` 가 리셋 때
`_arc_root[rows]` 를 넘기는데 `_m_at` 은 `_mscale[arange(n), j]` 를 읽어 **남의 M
프로파일**을 돌려줬다. 시나리오는 프로파일이 패리티에만 의존하고 `agent_rows` 가
패리티 순서를 보존해 우연히 무해하지만, `MS_MRAND>0` 은 행마다 뽑으므로 실제로 틀린
값이 첫 관측에 들어간다. **`ms1_m4_s0/s1`, `ms1_m8_s0` 이 이 코드로 돌고 있다.**
(`ms1_reg` 는 `_mscale≡1`, `ms1_zero` 는 창이 0 이라 무관하다.)

**`MA_LAYOUT` 을 `setdefault` 로 뒀다.** 기하를 정하는 유일한 변수인데, 평가는
사이드카를 `source` 한 뒤 python 을 띄우므로 낡은 값을 물려받는 게 정상 경로다.
그러면 기하가 바뀌는데 `_scen_xp`(env 원점)와 `_win_hi`(L/2)는 없어진 교차점 기준으로
계속 계산되고 **아무것도 죽지 않은 채 모든 숫자가 그럴듯하게 나온다.** hard-set 으로
바꾸고 충돌하면 예외를 던진다.

**`MS_L` 기본값이 두 곳에 있었다.** `_scen_env` 는 9.0, 창 계산은 12.0 을 써서 실제
경로 9 m 에 교차셀 60(=6.0 m) 이 잡혔다 — 회복 구간이 통째로 사라진 상태.
`MS_L_DEFAULT` 상수 하나로 통일하고, 자기검사가 `호길이 != MS_L/2` 면 예외를 던진다.

검증:

    MS_L=12 MS_RECOV=1.5   총호 12.00m, 교차점까지 6.000m, 창 cells[15:45], 교차셀 60
    MS_DECEL=both          a0·a1 둘 다 M배수 0.500, scen_id=12
    MS_PLACEBO=1           창 cells[75:105] (교차점 뒤), scen_id=22
    MA_LAYOUT=far 충돌     ValueError

**지표 열 20~25 추가**: `wdone` `wn_box` `latw` `encn` `arc_end` `scen_id`.
주 지표를 env 당 쌍 `ΔT_w` 로 바꿨다. 미학습 귀무값 **−0.033 s** 실측.
`vcmd` 는 창 안에서 `wc ≡ wn × 상수`라 정보량이 0 이므로 보고에서 뺐다.

### extra 토크나이저가 한 번도 학습된 적이 없다 (ma·masteer 공통)

`MA_TOKENIZER_ZERO=1` 은 extra 토크나이저의 **마지막 Linear 가중치를 0** 으로 둔다.
의도는 "시작 시점 출력이 정확히 0 이라 zero 대조군과 동일하되, 마지막 층 gradient 는
이전 층 활성값이라 0 이 아니므로 학습은 정상" 이었다. **그 논리가 틀렸다.**

`_build_mlp` 는 `units` 의 **모든** 항목 뒤에 활성을 붙이는데, 여기서는 units 의 마지막이
곧 토큰 차원(64)이라 출력 뒤에 ReLU 가 하나 더 남는다:

    enc0: ... (6): Linear(512, 64)  (7): ReLU()      <- 이것
    enc2: ... (4): Linear(128, 64)  (5): ReLU()      <- 마지막 가중치가 0 이 아니라 무해

마지막 Linear 가 0 이면 활성 전 값이 정확히 0 이고 PyTorch 의 `ReLU'(0) = 0` 이다.
backward 가 0 을 곱하니 **토크나이저 전체 gradient 가 영원히 정확히 0** 이 된다.

실측 (`MS_GRADCHK=1`, 첫 backward 직후):

    enc0 (teammate) requires_grad 8/8  grad텐서 8/8  |grad|합 = 0.000000e+00
    enc1 (steer)    requires_grad 8/8  grad텐서 8/8  |grad|합 = 0.000000e+00
    enc2 (new_carry)                                 |grad|합 = 3.148003e+02
    adapt_mlp                                        |grad|합 = 1.890226e+03

체크포인트에서도 확인된다. 3001 epoch 뒤 `task_encoder.0/1` 의 모든 bias 가 정확히 0,
마지막 층 가중치가 정확히 0, 나머지 층은 **초기화 상한과 정확히 일치**한다
(`Linear(12,2048)` 의 max 가 2.8867e-01 = 1/sqrt(12)). weight decay 조차 안 걸렸다.

**무효가 되는 결과:**

    masteer  ms1_zero 0.4019 / ms1_reg 0.4526 / ms1_m4 0.4277  -- 셋이 같은 이유가 이것이다.
             steer 창 토큰이 늘 0 이라 세 조건이 **같은 네트워크**였다.
    ma       t1_live vs t1_zero (Δ=+0.004), t4_c5_zero, g1_live_s1, t6_nf_c0(동결해제 포함)
             전부 teammate 토크나이저가 초기값 그대로다. "토큰 효과 없음" 은
             **정책이 무시한 것이 아니라 토큰이 두 조건 모두에서 0 이었던 것이다.**
             입력축 결론 전체를 다시 재야 한다.

**수정**: extra 토크나이저의 마지막 활성을 `nn.Identity()` 로 바꾼다. 토큰 임베딩에
ReLU 를 두면 음수 성분을 못 갖는 제약도 생기는데 `weight_token`·`self_token` 에는
그런 제약이 없으므로, 걷어내는 것이 구조적으로도 맞다. 파라미터가 없는 모듈이라
체크포인트 키는 그대로다.

검증 (같은 명령, 수정 전후):

    enc0  0.000000e+00 -> 8.568004e+01
    enc1  0.000000e+00 -> 7.888811e+01
    enc2  3.148003e+02 -> 3.148003e+02   (비트 단위 동일 -- 다른 곳을 안 건드렸다)

`MS_GRADCHK=1` 을 `trans_agent.py` 에 남겼다. 첫 backward 뒤 토크나이저별
requires_grad·grad 텐서 유무·|grad| 합과 모듈 구조를 한 번 찍는다. **새 토큰을 붙이면
반드시 이걸 먼저 돌린다** -- 이 버그는 학습이 끝날 때까지 아무 신호도 내지 않았다.

### 경로 이탈·속도 추종을 늘 재도록 (열 26~30)

`update_metrics` 가 `if self.scen == "free": return` 로 조기 반환해서 **기준선 4행
(reg/zero/m4/m8)이 시나리오 지표를 하나도 안 쌓고 있었다.** PLAN 에는 `m4` 의 판정
기준을 "`spd_err` 이 내려가야 한다" 로 적어놨는데 그 값을 만드는 코드가 없었다.

성공률만으로는 **"경로를 따라간 것" 과 "목표로 직진한 것" 을 구분할 수 없다.**
steer 의 핵심 주장이 정확히 그 구분인데 지표가 없어서, 성공률만 보고
"A=2 에서 steer 가 된다" 고 말할 뻔했다.

    26 latr   |lat_root| 합    경로 이탈
    27 latb   |lat_box| 합     박스 이탈
    28 spd    |v_real - v_cmd| 합
    29 vr     v_real 합
    30 steps  분모

전부 시나리오와 무관하게 매 스텝 쌓는다. 평가 요약에 `MS_TRACK_SUMMARY` 로 나오고
에이전트별(`a0lat`, `a1lat`)로도 가른다 -- A=2 에서 한쪽만 경로를 지키는 상황을 잡는다.

**재학습은 필요 없다.** 최종 숫자는 평가에서 나오고 평가는 새 프로세스라 이 코드를 쓴다.
학습 중 npy 는 26 열로 남고 평가 npy 가 31 열이 되는데, 파서가 열 수로 분기한다.

미학습 스캐폴드(50 iter) 귀무값:

    lat_root 0.271  lat_box 0.176  spd_err 0.835  v_real 0.357
    이탈>0.5m 20.8%   >1.0m 5.6%

`spd_err 0.835` 에 `v_real 0.357` -- 명령이 평균 1.0 근처인데 전혀 못 따라간다.

## 2026-08-21 · 시각화 인자 정리

### MS_VIZ — 시나리오를 인자 하나로

`scripts/masteer/viz_env.sh` 를 새로 만들고 `view.sh`·`record.sh` 가 **둘 다 이것을
source** 한다. 매핑이 두 곳에 있으면 뷰어에서 확인한 것과 영상이 어긋난다.
`MS_VIZ` 값이 곧 영상 파일명이라 영상 이름이 그대로 재현 명령이 된다.

12 값 = 6 시나리오 x {straight, curve}. 표는 `plan/PLAN_masteer.md` 의 `## 시나리오`.

**`*_curve` 는 `MS_SCEN_CURVE=1` 을 같이 켠다.** `_scen_env` 가 시나리오일 때
`MS_LAT_MAX` 를 0 으로 덮어쓰므로, 이게 없으면 3·4·6 의 `_curve` 가 **조용히
직선이 된다** -- 화면은 멀쩡하고 파일명만 curve 인 상태가 되어 제일 나쁘다.

`viz_expand` 는 `[ .. ] && return` 대신 `if` 를 쓴다. 호출부가 `set -e` 라
`&&` 예외 규칙에 기대게 되는데 그건 bash 판이 바뀌면 조용히 깨진다.

### env 12 개를 상수로 강등

65 개 학습 행의 `.env` 를 전부 뒤져 **어느 것도 설정한 적이 없음**을 확인하고 뺐다.

    MS_K(6) MS_M_NOM(2.4) MS_BACK(0.5) MS_ENC_R(2.5) MS_LAT_FRAC(0.25)
    MS_TURN_MAX(120) MS_SKEW(0.8) MS_HUMP2(0.1) MS_SPREAD_MIN(0.85)
    MS_SPREAD_MAX(1.8) MS_GT_WIDTH(0.30) MS_WIN_WIDTH(0.16)

**`MS_K` 가 제일 중요하다** -- 관측 크기가 `2K` 라 env 로 두면 언젠가 네트워크를
깨뜨린다. `MS_LAT_MAX`(`_scen_env` 가 쓴다)·`MS_M_LO`(5번 시나리오)·
`MS_CAM_H`/`MS_CAM_B`(영상 줌)는 남겼다.

값은 전부 옛 기본값 그대로다. 같은 체크포인트(`ms6_m8_long_s0` ep6100)를
정리 전후로 평가해 지표가 같은지 확인했다.

### MS_CLIP 기본값 0 -> 1, ENVS 통일

`MS_CLIP` 은 창·조준점·래칫을 바꾸는 **관측** 노브인데 `view.sh` 기본이 0 이라
`MS_CLIP=1` 로 학습한 ms11 이후 태그를 **학습과 다른 입력으로** 띄우고 있었다.
옛 태그(`ms1`~`ms10`, 영상용 `ms4_m4_s0`)는 `MS_CLIP=0` 을 명시해야 한다.

`view.sh` 는 env 수를 위치인자 `$2` 로만 받아 `ENVS=1 view.sh tag 3` 이 조용히
3 으로 떴다. `record.sh` 와 맞춰 `ENVS` 를 우선하게 했다 (위치인자는 뒤로 남김).

## 2026-08-31

### 로컬 데스크톱용 masteer 뷰어

`scripts/masteer/view_local.sh` 를 추가했다. 현재 데스크톱의 `DISPLAY`에 Isaac Gym
뷰어를 직접 띄우므로 서버용 `view.sh`의 Xvfb·x11vnc·noVNC·PORT가 필요 없다.
태그와 `.pth` 경로를 모두 받고, 최신 `Humanoid.pth` 선택·stage1 자동 탐색·1-env cfg
생성·체크포인트 스냅샷까지 한 스크립트에서 처리한다. 단일 GPU 로컬 기본값은 시스템의
`cuda:0`이며 `MA_GPU`를 요구하지 않는다.

Git에서 제외되는 대용량 모션·오브젝트 데이터가 masteer 복사본에 없으면, 이 PC에 이미
있는 원본 TokenHSI의 `tokenhsi/data`를 자동 탐색해 필요한 여섯 디렉터리만 심볼릭 링크로
공유한다. 원본 데이터는 복사하거나 수정하지 않으며, 다른 배치에서는
`TOKENHSI_DATA_ROOT`로 명시할 수 있다.

masteer와 무관한 long-horizon 태스크의 `pytorch3d` import가 `parse_task.py` 최상단에서
실행되어 모든 태스크를 막고 있었다. `HumanoidLongTerm4BasicSkills`를 선택했을 때만
지연 import하도록 바꿔, `pytorch3d`가 없는 환경에서도 `HumanoidMASteerCarry`를 실행할
수 있게 했다. long-horizon 태스크 자체의 의존성 요구는 그대로다.

### carry 학습 의미론 교정 — height gate·독립 시작 skill·정확한 clip 끝점

- masteer의 carry 속도항에 원본 TokenHSI와 같은 height gate를 복원했다. 박스 중심이
  `box_height/2 + 0.2m` 이하이면 속도 보상은 0이고, 원본 순서대로 목표 0.5m 안의 pin만
  그 뒤에 1로 만든다. `MS_VEL_W`, `MS_POS_C` 등 현재 가중치는 바꾸지 않았다.
- 태스크는 공유 scene 때문에 env 단위로 유지하되, carry 시작 skill과 reference motion은
  `(env, agent)` row마다 독립 표본을 뽑는다. 사람 state, AMP history, 각자의 박스·platform·
  목표 reset을 모두 같은 row 계약으로 바꿨다. A=1은 `row==env`라 동작이 같다.
- masteer에서 실제로 지원하는 보상 형태는 `onlyVelReward=True`,
  `onlyHeightHandHeldReward=False`로 고정했다. 상속 부모가 요구해 YAML key는 남기지만,
  다른 값을 주면 조용히 무시하지 않고 즉시 `ValueError`를 낸다.
- `MS_CLIP=1`의 끝점이 목표 0.1~0.2m 앞에 생기던 floor 셀 오류를 고쳤다. 마지막 waypoint를
  ceil 셀에 보존하고 그 셀 호좌표를 반환한다. 10,000개 랜덤 경로에서 목표점 최대 오차는
  `4.5e-6m`였다.
- 원본 carry의 실행 모드를 다시 대조했다. 학습과 일반 `--test`는 혼합 시작 skill이고,
  final `--test --eval`만 eval YAML의 `loco=1.0`을 쓴다. masteer는 어느 모드든 선택된
  분포에서 row별로 독립 샘플링한다. final eval에서는 두 skill 이름이 loco로 같아도
  reference motion/time과 각자의 박스·목표 상태는 독립이다.
- A=2 final evaluator가 row별 성공을 세면서 종료 목표를 `num_envs`로 두어 전체 agent의
  절반 분량만 집계하던 것을 `num_envs * num_agents`로 고쳤다. A=1 trial 수는 그대로다.
- `view.sh`와 `view_local.sh`는 원본 viewer처럼 기본적으로 일반 test(혼합 시작)를 쓰고,
  `MS_EVAL=1`일 때 final-eval(loco 시작)을 쓴다. `record.sh`는 final-eval 전용이며,
  ms11 이후 정책과 맞지 않던 `MS_CLIP` 기본값 0도 1로 맞췄다.

### 집기→운반→놓기 생애주기 지표 (열 40~49)

`hand_near`(손 근접 proxy), 원본 height gate가 열린 스텝, 손 근접 중 박스 이동거리,
시작 대비 최대 상승, 최초 근접 시각, delivery/엄격 putdown을 분리해 누적한다. 최종
`carry`/`place`는 손 근접 중 0.5m 이상 실제로 옮긴 뒤 목표에 닿은 경우만 세며,
`shortcut`은 목표에는 닿았지만 그 운반 증거가 없는 경우다. `eval_one.sh`의 한 줄 요약과
`analyze.py` 출력에도 모두 붙였다.

검증은 `py_compile`, `bash -n`, `git diff --check`, 10,000개 CPU 경로 끝점 검사와
2-agent·혼합 시작 skill Isaac Gym 스모크로 했다. 스모크는 90초 동안 23개 에피소드를
재시작했고 인덱스/CUDA/runtime 오류 없이 계속 실행됐다(마지막 `KeyboardInterrupt`는
의도한 timeout 종료).

## 2026-09-01

### ms18 — teammate 토큰 exact attention mask

`MA_TOKEN=mask`를 추가했다. teammate 21-D 관측과 네트워크 크기는 `live`와 동일하게
유지하지만, adapt Transformer의 첫 extra 위치를 `src_key_padding_mask=True`로 두어
모든 self-attention 층의 key/value에서 제외한다. `MA_TOKEN=zero`는 0 벡터도 attention
softmax 분모에 남으므로 토큰 부재 대조군이 아니며, `mask`가 그 차이를 분리한다.
따라서 ms18의 grad-check에서 teammate encoder grad=0은 의도된 정상값이고,
steer·new-carry tokenizer와 internal adapter는 계속 nonzero gradient여야 한다.

`ms18_maskteam_origscale_c06_s0`은 `ms17_origscale_c06_s0`에서 이 mask만 바꾼다.
활성 actor 토큰은 `weight+self+steer+new_carry`이며, 현재 실행 중인 ms17은 건드리지 않고
GPU0 다음 대기 행으로 추가했다. 기존 sidecar가 이미 `MA_TOKEN`을 저장하므로 별도 노브나
학습/평가 스크립트 분기는 만들지 않았다. 로컬·서버 viewer와 record도 사용자가
`MA_TOKEN`을 직접 덮지 않았을 때 태그 sidecar의 값만 복원해, mask 정책을 기본 live로
잘못 여는 관측 불일치를 막는다.

### masteer adapt 중간 체크포인트 기본 주기 1000

로컬 `train_local.sh`가 기본으로 쓰는 adapt 학습 설정의 `save_frequency`를 500에서
1000으로 바꿨다. 새 학습은 최신 `Humanoid.pth`와 별도로
`Humanoid_00001000.pth`, `Humanoid_00002000.pth`, ...를 보존한다. 이미 설정을 읽고
실행 중인 학습 프로세스에는 영향을 주지 않는다.

## 2026-09-04

### C3b — 교차점 최저속도와 물리적 post 복귀

기존 C3의 대칭 pre-conflict cosine을 체크포인트 호환 옵션으로 분리했다.
`conflict_min_at_crossing=True`에서는 예측한 pre 길이 동안 smoothstep으로 감속하여
기하학적 conflict point에서 최저속도를 찍고, 이후 executor의 `0.75m/s^2` 가속 한계로
계산한 거리 동안 1.5m/s로 복귀한다. 모델 출력 `(pre_length, depth)`는 늘리지 않았고
기존 C3 PTH는 옵션 기본값 false로 bit-identical한 옛 decoder를 계속 사용한다.

환경변수 `COORD_C2_CONFLICT_MIN_AT_CROSSING`을 학습 sidecar에 기록하도록 연결했다.
legacy/new profile, 최저속도 위치, 복귀 완료, checkpoint round-trip을 포함한 pure-PyTorch
단위검사 68개를 통과했다. 새 비교는 `c3b_r1_s0`, `c3b_r2_s0` 두 fresh 런이며 각각
2000 iter, 100마다 Cross 평가한다.
## 2026-09-04 — C5 general pointwise speed caps

- Added optional `physical_speed_caps` decoding for joint waypoint-speed MLPs.
  Learned per-point caps are converted to reachable pre-braking and post-risk
  recovery profiles without a hand-authored crossing/window representation.
- Added scenario-agnostic, time-aligned per-path-point proximity risk. In this
  mode `unnecessary_slow` regularizes the raw learned requests only at nominally
  safe points; collision loss remains responsible for activating slowdown.
- Added strict checkpoint/config/env wiring and `scripts/coord/c5_pointcaps.sh`.
  Two localization-weight variants are running as `c5_r1_s0` and `c5_r2_s0`.

## 2026-09-04 — C6 crossing plateau

- Added an opt-in `conflict_plateau` decoder. The learned interval is now a
  constant minimum-speed plateau ending at the geometric conflict point;
  braking and recovery ramps are derived at `0.75m/s²`.
- Kept the C3/C3b checkpoint meanings unchanged. C6 uses the same action shape
  but a distinct checkpoint config flag and fresh experiment tags.
- Added checkpoint/profile regression coverage and `scripts/coord/c6_plateau.sh`.

## 2026-09-04 — C7 joint-both general risk

- Reused C5's state-only joint waypoint/speed MLP and physical speed envelopes,
  but disabled fixed-priority projection so both agents' outputs control ms18.
- Kept only general time-aligned proximity safety plus a weak, symmetric total
  delay objective. Equal speed-delay and detour-delay weights avoid prescribing
  slowdown versus spatial avoidance.
- Added `scripts/coord/c7_jointboth.sh`; `c7_jointboth_s0` runs for 2000
  iterations and saves/evaluates every 100 while the existing C5 runs continue.
- C7 was stopped after preserving its 600 checkpoint: weak path residual
  regularization caused roughly 50% invalid curves. Added the single-variable
  C8 diagnostic with the stable path residual coefficient restored to 1.0.
- C8 showed that residual weight alone was insufficient; C9 restored the full
  C5-r1 stabilizers and changed only fixed priority to both-agent control.
- Added optional `dual_delay`: the product of both agents' speed-induced plan
  delays. It encourages the joint model to concentrate yielding on one agent
  selected from state, without a yielder label or Cross-specific rule.
