# TokenHSI-coord — 변경 기록

> 파일 변경은 hook이 자동 기록. 무엇을/왜 바꿨는지는 Claude가 `###` 항목으로 덧붙인다.

## 2026-09-23

### carry planner recurrent robust collision supervision

- episode 첫 plan에만 걸리던 analytic collision loss를 매 replan의 실제 미래 suffix에
  적용한다. 고정된 과거 prefix는 현재 measured root로 접고, held agent는 pickup anchor를
  이미 지난 것으로 처리해 지나온 길을 미래 motion으로 다시 계산하지 않는다.
- frozen executor의 tracking/timing 오차 때문에 exact-time rollout은 안전하지만 실제로
  충돌하던 불일치를 줄이기 위해 기본 ±1.5초의 상대 timing offset 7개에서 worst collision을
  사용한다. 96×96 pairwise 행렬 대신 offset 수에 선형인 계산으로 큰 PPO minibatch의
  메모리 사용을 제한한다.
- 과거 prefix에는 collision/curvature gradient가 생기지 않는지, replan suffix에는
  gradient가 도달하는지, robust timing loss가 exact-time loss 이상인지 단위 테스트했다.

## 2026-09-22

### Plain Carry sparse waypoint/speed spline head (V16)

- plain Carry의 agent별 33-point 위치 residual과 33-point speed 직접 출력을
  `start / learned approach / box / learned carry x3 / goal` 7-knot 표현으로 교체했다.
  위치는 box에서 분리한 cubic Hermite spline, 속도는 같은 knot의 monotone cubic
  spline으로 만든 뒤 arc-length 기준 33점으로 재샘플링한다.
- 위치 action을 128D에서 16D, speed action을 66D에서 14D로 줄이면서 frozen executor,
  collision evaluator와 analytic loss가 받는 dense ABI는 유지했다. start, pickup(index
  16), placement(index 32)는 exact anchor이고 실행된 prefix만 기존 path에서 보존한다.
- sparse control point는 직선에 작은 dense residual을 더하지 않고 기본 4m workspace에서
  독립적으로 움직인다. checkpoint/action 계약 변경을 명시하기 위해 schema를 V16으로
  올렸다.
- 기존 dense residual second-difference smoothness는 곡선 자체를 직선으로 당기는 문제가
  있어 plain Carry에서 제거했다. pickup을 경계로 두 spline을 나누고 단위 접선의 변화량
  차이(curvature variation)를 penalize해 일정한 곡률의 우회는 허용하면서 S-curve noise만
  억제한다. 기존 46도 초과 curvature penalty는 급회전 제한으로 함께 유지한다.
- 학습 없이 실제 frozen ms18 bridge를 확인할 수 있도록 고정 opposite-detour와 speed-dip
  bias를 넣는 `make_sanity_checkpoint.py`, 생성된 V16 checkpoint, one-command
  `view_sanity.sh`를 추가했다. 생성 checkpoint의 33점 shape, speed 범위와 세 hard anchor를
  viewer 실행 전 CPU에서 검증했다.
- 거절된 sampled plan이 fallback/이전 path의 물리 reward를 그대로 받아 PPO credit이
  뒤집히던 문제를 막기 위해 `CARRY_PLANNER_INVALID_PLAN_COEF`(기본 0.25)를 직접
  macro reward에서 차감한다. 평균 실제 감점은 `invalid_plan_penalty`로 기록하고 checkpoint
  extras에도 계수를 저장한다.

### C2/simple MLP convergence-task viewer

- 기존 C2/C13 MLP coordinator checkpoint를 frozen ms18 위에서 직접 로드하고,
  현재 Carry planner와 같은 box→공통 crossing→close-goal reset을 적용하는 전용
  launcher를 추가했다. checkpoint metadata에서 C2/simple/C1 loader도 자동 선택한다.
- viewer 전용 환경변수로 일반 Cross 혼합 비율과 goal margin을 조절하며 기존 학습과
  batch evaluation의 reset geometry에는 영향을 주지 않는다.
- 공용 coordinator viewer가 `MA_GPU`를 읽기만 하고 실제 CUDA/PhysX 인자로 전달하지
  않던 문제를 고쳤다. 추가로 Isaac Gym native PhysX가 `CUDA_VISIBLE_DEVICES` remap과
  달리 `compute_device_id=0`을 물리 GPU 0으로 잡아 Torch는 GPU 1, PhysX는 GPU 0에
  갈라지던 것을 확인했다. viewer에서는 remap을 제거하고 Torch/PhysX/Vulkan 모두
  동일한 명시적 물리 ordinal(`cuda:$MA_GPU`)을 사용한다.

### close-goal Carry layout의 carry segment 교차 보장

- 기존 hard layout은 goal 두 개를 공통 중심 가까이에 두기만 해 `box->goal` 선분이
  실제로 만나지 않는 배치가 포함됐다. 각 현재 box에서 공통 crossing으로 향하는 ray를
  그대로 연장한 지점에 goal을 두어 두 carry segment가 반드시 같은 점을 지나게 했다.
- 두 ray의 동일한 post-crossing 거리로 goal 간격을 box circumscribed radius 합+margin에
  정확히 맞춘다. 거의 평행하거나 crossing과 box가 겹친 퇴화 샘플은 원래 Cross를
  유지하며 hard-case로 집계하지 않는다.

### carry planner viewer 연속 randomized episode

- interactive `view.sh`가 `--eval`로 들어가 1 env × 2 agents × 3 repeats, 총 6 trial
  뒤 종료되던 것을 일반 deterministic player로 바꿨다. 기본 games 상한은 10억이라
  사용자가 창을 닫을 때까지 reset마다 계속 실행한다.
- 기본 timed-cross 강제를 제거하고 학습과 같은 75% close-goal/25% 일반 Cross를
  episode마다 다시 뽑는다. timed-cross는 `MS_VIEW_TIMED_CROSS=1`인 명시적 stress
  viewer에서만 사용한다.

### Sequential stack 대신 plain Carry collision-avoidance planner 추가

- `carry_planner/`에 기존 simultaneous `HumanoidMACoordCarry` 실행 경로를 재사용하는
  `HumanoidMACarryPlannerTrain` adapter와 전용 PPO/train wrapper를 추가했다.
- 신경망은 최신 `stack_planner`의 scene-token Transformer, 33-point joint path/speed head,
  decision history, actor-critic 및 checkpoint 형식을 그대로 사용한다. task만 두 agent가
  각자 box를 동시에 운반하는 plain Carry이며 sequential phase, stacking, retreat 보상은 없다.
- plain Carry reference는 17-point `root->box`와 17-point `box->goal`을 합친 33점이다.
  pickup(index 16)과 goal(index 32)을 hard anchor 및 action mask로 고정해 frozen Carry가
  실제 자기 box/goal을 잃지 않게 했다.
- macro reward는 원래 Carry reward와 remaining-distance progress를 보존하고,
  agent-agent / agent-other-box / box-box proximity cost를 감점한다. 기본 stress scene은
  `MS_SCEN=cross`, 출력은 `runs/carry_planner/<tag>`에 분리한다.
- 검증: `tokenhsi118`에서 stack planner 단위 테스트 45개 통과. GPU0에서
  `4 env x 2 agent`, 1 iteration smoke를 실행해 reset, path 설치, frozen ms18 rollout,
  PPO backward, checkpoint 저장까지 통과했다. `invalid=0`, fallback=0.
- 2,048 env 첫 episode reset에서 inherited multi-task reset이 indexed root setter를 여러 번
  제출해 PhysX GPU pipeline이 illegal memory access로 무너지는 것을 확인했다. plain Carry
  adapter에도 carry-only single-commit reset transaction을 적용해 모든 tensor mutation을 먼저
  staging하고 root/DOF state를 각각 한 번만 제출하도록 수정했다.
- 기존 coordinator의 매-replan current-root/current-box anchor equality 검사는 실행 prefix를
  고정하는 stack planner 계약과 맞지 않아 두 번째 decision부터 정상 경로를 invalid 처리했다.
  plain Carry adapter에서는 이 동적 검사만 제외하고 최초 pickup/goal hard anchor와
  finite/buffer/speed/curvature 검사를 유지했다.
- `HumanoidMACarryPlannerView`와 `carry_planner/view.sh`를 추가했다. plain-carry checkpoint를
  deterministic mean으로 로드하고 학습과 동일한 decision history/고정-origin EMA path를
  6 action-step마다 갱신하며, 기존 steer Carry의 installed-path/speed renderer로 표시한다.
- recurrent fixed-origin plan을 다시 설치할 때 기존 Carry bridge가 `_arc_root/_arc_box`를 0으로
  초기화해 pickup 직후 출발 경로를 역주행시키던 문제를 수정했다. 첫 plan만 0에서 시작하고,
  이후 replan은 실행 cursor와 metric cursor를 보존하며 새 path 끝 범위로만 clamp한다.
- plain Carry planner viewer의 기본 Cross를 viewer-only timed-cross 배치로 강화했다. 두
  agent-box 패키지의 정속 기준 교차 도착 시각을 맞추고 goal을 공통 교차점 너머에 배치해,
  회피가 없으면 실제 동시 충돌이 나도록 한다. 학습·평가 분포에는 적용되지 않으며
  `MS_VIEW_TIMED_CROSS=0`으로 기존 무작위 Cross를 볼 수 있다.
- Carry planner viewer에서 generic `enableDebugVis`를 끈다. 해당 hook은 매 action step의
  post-physics에서 모든 debug line을 지우지만 planner ribbon은 render 뒤에 다시 그려져,
  action step마다 한 프레임씩 경로가 사라지는 깜박임을 만들었다. 전용 path/speed ribbon은
  `render()`의 `_draw_task()`가 독립적으로 그리므로 그대로 유지된다.
- Carry planner 학습 reset을 일반 Cross 25%와 close-goal convergence 75%로 섞었다.
  hard case는 두 goal을 공통 중심 주위의 임의 축에 배치하되 box circumscribed radius 합과
  0.25 m margin을 보장해 불가능한 최종 중첩은 만들지 않는다. 선행 agent/box가 goal에서
  정지한 뒤 후발 agent가 돌아가야 하므로 속도 조절만으로 끝나는 해를 줄인다. fresh run의
  기본 path correction 폭은 이 우회를 허용하도록 0.5 m에서 1.0 m로 넓혔다.
- `carry_planner/train.sh`가 실수로 motion dataset YAML을 `--cfg_env`에 넘겨
  `load_cfg()`에서 `KeyError: 'env'`로 중단되던 경로를 정식 multi-task environment
  config로 복구했다. 생성 전 source 존재 여부와 생성 후 최상위 `env:`를 검사해 같은
  오배선을 Isaac Gym 초기화 전에 명확한 오류로 차단한다.

### Full stack과 분리한 post-place A1 retreat 검증 task (V15)

- Box1을 목표에 놓고 A1을 release 이후 0.45m 거리에서 시작시키는
  `HumanoidMAStackPlannerRetreatTrain/View`를 추가했다. A2와 Box2는 3m 밖 hold 상태로 두고
  planner 실행 및 PPO action gradient에서 완전히 mask한다.
- retreat-only planner의 고정 base는 A1 departure root를 33점에 복제한 path다. carry-prefix
  projection과 `box → goal` visit penalty를 건너뛰고 A1의 전체 32개 future point를 최대
  2m/component bounded offset으로 직접 실행한다.
- 방향을 지정하는 GT retreat target 없이 1.5m에서 포화되는 물리 root-box clearance potential,
  box collision/disturbance, fall, time과 path geometry만 채점한다. 성공 시 episode를 종료한다.
- 전용 `train_retreat_only.sh`, `view_retreat_only.sh`와 `retreat_success_rate`,
  `retreat_clearance` metric을 추가하고 checkpoint schema를 V15로 올렸다.
- GPU 실행 없이 CPU unit test 42개와 Python/shell 정적 검사를 수행했다.

## 2026-09-18

### Bounded absolute path target과 mean planning reference 분리 (V14)

- episode 최초 geometric path를 고정 base로 저장하고 head 출력을 누적 increment가 아닌 base
  기준 bounded absolute target으로 재정의했다. 일반 구간은 0.5m/component, A1 post-goal retreat
  suffix는 2.0m/component 범위이며 직전 mean에서 target으로 기본 0.25 EMA update한다.
- PPO rollout은 noise가 포함된 sampled target update를 frozen executor에 설치하지만 다음 decision의
  `previous_path`, bootstrap observation과 temporal consistency에는 noise-free mean path만 commit한다.
  mean delta와 exploration noise가 horizon마다 누적되던 두 random-walk 경로를 모두 차단했다.
- base path와 previous mean path를 history/checkpoint 계약에서 분리하고 schema를 V14로 올렸다.
  deterministic view/eval도 동일 EMA contract를 사용한다.

### Single-head baseline 및 정상 stack 접촉 제거

- planner launcher와 model config의 기본 candidate 수를 4에서 1로 바꿨다. full candidate
  rollout 구현과 checkpoint별 candidate 복원은 유지하므로 이후 명시적인
  `STACK_PLANNER_CANDIDATES=4` 확장은 가능하지만, 우선 evaluator 없이 단일 path head의
  학습 가능성을 검증한다.
- 정상 stack에서도 XY footprint가 필연적으로 겹치는 box-box proximity를 collision reward,
  aggregate collision cost와 학습 metric에서 제거했다. agent-agent, agent-box 및 held-box/body
  proximity는 유지한다.

### Learned pointwise speed profile (V13)

- 각 후보 action을 128D path correction과 66D pointwise speed profile로 확장했다. speed는
  `[0.375, 1.5] m/s`로 bounded decode하고 path와 같은 arc에서 frozen steering ABI로 보간한다.
- full candidate rollout, PPO log-prob 및 evaluator가 path와 speed를 함께 평가한다. 인접 speed의
  제곱 차이에 별도 smoothness loss를 적용하고 실행 시 기존 `0.75 m/s²` command limiter를 유지한다.
- 고정 `MAX_SPEED` adapter를 제거하고 schema를 V13으로 올려 path-only V12 checkpoint resume을
  명시적으로 거부한다. CPU 단위 테스트 40개를 통과했다.
- planner 전용 `DONE`을 기본 box-radius cutoff로 맞췄다. top-box 중심 XY가 bottom footprint의
  반대각선 안이고 높이 tolerance를 만족하면 종료하며, parent sequential 환경의 strict 판정은
  유지한다. `STACK_PLANNER_RELAXED_DONE=0`으로 strict 판정을 다시 사용할 수 있다.
- planner checkpoint 기본 저장 주기를 10 iteration에서 5 iteration으로 단축했다.
- aggregate collision metric을 agent-agent, box-box, agent-box, held-box/body 네 종류의
  ratio와 cost로 분해했다. 기존 total collision reward는 네 cost의 합으로 유지한다.
- placement 전에 출력된 마지막 유효 A1 unified-path endpoint를 보관하고 `A1_RETREAT` phase
  진입 즉시 virtual box에 적용한다. 종전처럼 phase 전환 뒤 한 planner period 동안 virtual
  box가 이전/current-root 위치에 남는 지연을 제거했다.
- 캐시 복사에만 의존하지 않고 모든 유효 plan 설치 시 virtual box XY를 raw A1 path 마지막
  point와 직접 동기화한다. retreat 설치 후 두 좌표가 다르면 즉시 실패하는 invariant도 추가했다.
- viewer의 invalid plan을 finite/root-reachable/32m-buffer/speed/46도-turn gate로 분해해 출력한다.
  virtual box가 갱신되지 않을 때 endpoint 연결 문제와 hard validity 거부를 즉시 구분할 수 있다.
- 46도 turn 제한이 모든 curved proposal을 hard-invalid로 만들어 physical rollout과 virtual box
  갱신을 차단하던 문제를 수정했다. 46도 초과는 differentiable-shaped soft reward로 옮기고,
  175도 이상의 사실상 역주행만 emergency invalid로 유지했다.
- A2 launch가 `stable_steps + delay`의 연속 안정성을 요구해 A1이 bottom box를 한 번 차기만 해도
  영원히 취소되던 문제를 수정했다. 최초 안정화 사건을 latch한 뒤 1초 countdown을 monotonic하게
  진행하며, 이후 box 교란은 launch gate가 아니라 물리 penalty로 처리한다.
- 실제 placed bottom-box 이동 누적량의 기본 weight를 2에서 10으로 높이고
  `bottom_disturbance_penalty`를 iteration 콘솔 로그에 노출했다. A2 countdown은 box 교란과
  무관하게 계속 진행한다.
- `STACK_TOP_FOLLOWS_BOTTOM=1`에서 moving target이 bottom position error를 상쇄하던 측정 허점을
  제거했다. 최초 안정화 이후 실제 bottom-box world position의 frame 간 이동 거리를 직접
  누적하므로 A1이 support box를 차거나 미는 동작이 그대로 disturbance penalty에 들어간다.

### Joint A2 preplan with deferred Carry goal (V12)

- A2 path를 handoff 때 새로 만드는 대신 최초 joint planner decision부터 A1과 함께
  `departure → Box2 → eventual stack top`으로 출력하고 A2 executor에도 미리 설치한다.
- planner observation의 A2 goal은 handoff 전에도 예정된 stack top XY를 사용하지만 frozen Carry의
  실제 `_box_tar_pos`는 Box2 초기 위치에 유지한다. 안정화 신호가 열리면 steering path를
  `_reset_steer_to()`로 덮어쓰지 않고 Carry goal만 실제 Box1 top으로 변경한다.
- `STACK_TOP_FOLLOWS_BOTTOM=1`에서도 inherited 코드처럼 `_gt_path`를 매 frame 평행이동하지 않고,
  물리 Carry goal만 support를 따라가게 한다. path 수정은 다음 planner replan이 담당한다.
- planner input/execution 의미가 달라 schema를 V12로 올리고 V11 checkpoint resume을 거부한다.
- fixed-index consistency valid mask의 `expand()` 결과를 그대로 반환해 decision mask의 in-place
  intersection에서 stride-0 alias 오류가 나던 것을 materialized clone으로 수정했다.

### Fixed-origin future-masked trajectory planner (V11)

- 잔여 경로를 매 decision마다 33점으로 재보간하던 V10 reference를 제거했다. 최초 departure에서
  만든 33-point index와 P0를 유지하고, 현재 root의 monotonic point progress 이전 correction을
  freeze한다. 경계는 두 point ramp로 연결해 현재 위치에서 꺾임이 생기지 않게 했다.
- progress를 plan embedding에 추가하고 이미 실행한 action dimension은 PPO log-prob과 entropy에서
  제외한다. temporal consistency도 같은 고정 point index끼리 비교하도록 변경했다.
- frozen executor는 새 plan 설치 때 arc 0으로 돌아가지 않고 현재 root를 실행 path에 투영한 arc에서
  재개한다. retreat 실행과 box-clearance reward는 현재 root 이후 suffix만 사용해 과거 carry/retreat
  구간을 미래 충돌로 재채점하지 않는다.
- 모델 입력, action credit, 실행 계약 변경으로 schema를 V11로 올리고 V10 resume을 거부한다.
  CPU 단위 테스트 40개를 통과했다.

### Progress-advanced smooth trajectory correction planner (V10)

- 직전 채택 path의 point 0만 현재 root로 옮기던 V9 reference 갱신을 제거했다. 현재 root를
  직전 polyline에 투영하고 이미 실행한 prefix를 버린 뒤 남은 point sequence를 33점으로
  재보간한다. endpoint는 보존하며 stationary replan은 기존 point를 그대로 유지한다.
- 후보 diversity를 raw 33 points 대신 4회 low-pass한 coarse route에서 계산한다. mean
  pointwise correction의 second finite difference 제곱을 `STACK_PLANNER_SMOOTHNESS_COEF`
  (기본 10.0)로 penalize해 좌우 zigzag가 diversity나 path correction으로 남지 않게 했다.
- `path_smoothness_loss`와 기존에 파일에만 있던 `collision_cost`를 콘솔에도 출력한다.
  reference/checkpoint 의미가 바뀌므로 schema를 V10으로 올리고 V9 checkpoint resume을 거부한다.

## 2026-09-17

### E2E DAgger planner 전용 viewer

- 저장소 루트에 `visualize_e2e_planner.sh`와 `e2e_viewer/run_view.py`를 추가했다.
  별도 e2e checkout을 읽기 전용 source로 사용해 `m4_bc_r5/last.pth`와 frozen executor를
  native Isaac Gym 창에서 실행한다.
- 발표 수치 0.249의 실행 조건인 `E2E_CARRY_END_OFFSET=0.15`를 기본으로 재현한다.
  화면에는 agent별 설치 전체 경로, 실제 steering window, aim point를 구분해 표시한다.
- viewer subclass는 dataset URDF 절대경로를 parent directory와 basename으로 분리해 Isaac Gym에
  전달한다. 원본의 `./` + 절대경로 결합으로 asset 전체가 parse 실패하고 PhysX CUDA illegal
  access로 이어지던 문제를 viewer 범위에서 수정했다.
- 연속 speed profile에서는 동일 속도 run이 없어 ribbon이 생략될 수 있으므로 전체 path band를
  기본으로 그리고, agent별 planner route 중심선을 바닥 위에 항상 추가로 표시한다.
- checkpoint 계약 사전 검사와 source/checkpoint/GPU/Vulkan/env/seed override를 지원한다.

### Stack planner viewer 루트 진입점

- 저장소 루트에 `visualize_planner.sh`를 추가했다. 현재 planner schema와 호환되는 최신
  checkpoint 자동 선택, `--list` metadata 확인, `--check` 사전 검증, GPU/Vulkan device와
  env 수 선택을 지원하고 기존 deterministic Isaac Gym viewer를 호출한다.
- 구 checkpoint를 현재 모델로 암묵 변환하지 않으며 schema가 다르면 창을 띄우기 전에
  명시적으로 거부한다.

### Stack planner 평가 box/layout coverage

- deterministic retreat 평가가 기본적으로 small/medium/large bottom/top box의 3x3 size
  grid를 환경 벡터에 배정하도록 했다. 박스 asset은 env 생성 시 고정되므로 episode별 resize를
  시도하지 않고 env 간 다양성으로 평가한다.
- `STACK_PLANNER_EVAL_SCEN=free|cross|parallel|solo`를 추가하고, 네 배치를 순차 평가하는
  `eval_retreat_suite.sh`를 추가했다. 학습 sidecar를 읽은 뒤 평가 scenario를 명시적으로
  덮어써 외부 shell의 낡은 `MA_LAYOUT`이 평가를 오염시키지 않게 했다.
- summary에 scenario, box-grid 활성 여부, unique bottom/top/pair 크기 수를 기록한다.
  학습 분포는 변경하지 않았다. 기존 학습은 env 생성 시 box size를 env별로 배정하고,
  free 배치에서 box/goal 위치를 episode reset마다 다시 샘플링한다.

### Previous-trajectory correction planner (V9)

- V8의 latent 30D parameter 누적을 제거했다. planner memory에는 직전 채택된 실제
  `[2 agents, 33 points, XY]` world trajectory를 저장하고 shared-frame plan token으로 입력한다.
- 최초/reset에는 현재 root→box→goal geometric path를 reference로 사용한다. 이후 reference는
  직전 trajectory의 시작점만 현재 root에 맞게 endpoint 방향으로 선형 감쇠 re-anchor한다.
  각 head는 root를 제외한 128D pointwise XY correction을 직접 출력하고, 두 번 low-pass한
  `delta_scale*tanh(delta)`를 reference trajectory에 더한다. latent parameter는 누적하지 않는다.
- full counterfactual rollout의 유효한 최고 후보 trajectory만 다음 reference로 commit한다.
  executor 출력은 계속 완성된 33-point path이며 action/encoder 계약 변경으로 schema는 V9다.

### Previous-plan latent delta planner (V8 precursor, superseded)

- 직전 실제 채택 path parameter 30D를 planner memory와 Transformer plan token으로 추가했다.
  각 full-path head는 absolute parameter 대신 bounded delta를 출력하며, 새 proposal은
  `previous + delta_scale*tanh(delta)`로 만든 뒤 현재 root/box/goal 기준으로 다시 decode한다.
- full counterfactual rollout에서 최고 후보의 유효하게 실행된 proposal만 다음 decision base로
  commit한다. invalid/inactive 후보는 이전 base를 보존하고 episode reset은 zero geometric prior로
  되돌린다. 후보 branch의 bootstrap value도 해당 후보가 commit된 다음 observation으로 계산한다.
- frozen executor에는 이전과 동일한 완성된 33-point world trajectory만 전달한다. PPO action
  의미와 encoder 입력이 바뀌므로 checkpoint schema를 호환되지 않는 V8로 올렸다.

### Full counterfactual multi-head planner (V7)

- 같은 scene의 독립 full-path 후보 4개를 동일 simulator/task snapshot에서 전부 물리 rollout한다.
  후보별 macro return을 continuous PPO credit으로 사용하고, env별 최고-return branch의 종료
  state만 다음 decision으로 이어간다.
- 공유 evaluator는 sampled categorical PPO 대신 실제 후보 return의 argmax를 supervised target으로
  학습한다. 비선택 후보도 직접 실행 결과를 받으므로 evaluator와 path head의 credit assignment를
  분리했다.
- counterfactual preview가 decision history를 변경하지 않도록 non-commit reset을 분리하고,
  checkpoint schema를 호환되지 않는 V7로 올렸다.
- 첫 low-level step에서 task에 lazy cache tensor가 추가되어 pre/post snapshot key 집합이 달라지는
  경우, 시작 snapshot을 authoritative state layout으로 사용한다. 시작 state tensor의 누락이나
  shape 변경은 계속 fail-closed하고 rollout 중 새로 생긴 derived/cache tensor만 병합에서 제외한다.

### Scene-token + independent full-path multi-head planner (V6 precursor)

- root/box/goal 전용 tokenizer 뒤에 learnable `[SCENE]` token을 붙여 Transformer encoder가
  단일 scene representation을 만들도록 했다. candidate query와 Transformer decoder는 제거했다.
- 독립 head 4개가 각각 A1 carry→retreat와 A2 carry를 포함한 완전한 30D joint path를 제안한다.
  공유 evaluator는 scene과 detached proposal을 함께 보고 categorical candidate를 선택한다.
- PPO action을 `선택 index + 선택된 30D path`로 바꿔 실제 실행된 head에만 continuous
  rollout credit이 가도록 했다. deterministic view/eval은 evaluator argmax를 사용한다.
- 후보 간 A1 full-path 거리에 diversity margin을 적용하고 candidate usage/distance metric을
  추가했다. checkpoint schema는 호환되지 않는 V6로 올렸다.

### A2 stacking 중 A1 동적 회피 유지

- planner task에서 A1을 `A1_RETREAT`뿐 아니라 `A2_RESUME`와 `VERIFY_STACK`에서도 active row로
  유지한다. A2가 접근하고 box를 올리는 동안 A1 goal 이후 suffix를 계속 재계획·설치하므로,
  같은 macro action의 collision outcome이 실제 실행된 A1 회피 path에 귀속된다.
- post-placement phase에서는 A1 suffix의 Box1 footprint penalty와 낮춘 consistency scale을
  계속 적용한다. A2는 기존 placement row ownership과 stacking goal을 유지한다.
- invalid raw 후보가 가상 box만 움직이고 `_gt_path`는 이전 값에 남던 command 불일치를 제거했다.
  이제 유효 A1 plan이 설치될 때만 가상 box와 path endpoint를 함께 갱신한다.
- 서버의 긴 `ldconfig` 출력에서 `awk` 조기 종료가 SIGPIPE를 발생시키고 `set -eo pipefail`이
  학습 launcher를 아무 로그 없이 종료하던 문제를 수정했다. CUDA driver 탐색은 입력 전체를
  소비하면서 첫 `libcuda.so.1`만 선택한다.

### 과거 decision state 기반 temporal planner 입력

- planner observation에 최근 decision의 6개 entity token을 FIFO로 저장하고 temporal embedding과
  padding mask를 적용했다. 기본 길이는 4이며 `STACK_PLANNER_HISTORY_STEPS`로 설정한다.
- recurrent hidden state 대신 각 PPO transition에 history token/mask를 함께 저장하므로 shuffled
  minibatch에서도 action 당시 입력을 그대로 재현한다. done env만 history를 초기화한다.
- train, deterministic viewer, retreat eval이 checkpoint의 동일 history 길이를 사용한다. 기존
  state-only checkpoint는 `history_steps=1`로 호환하며 길이가 다른 checkpoint resume은 거부한다.

### A1 retreat endpoint soft stabilization 강화

- endpoint 변경 penalty 기본 계수를 `0.10→1.0`으로 높이고 0.10m deadband는 유지했다.
  hard latch나 실행 clamp는 추가하지 않아 큰 위험에서는 endpoint 변경이 가능하다.
- 전체 path consistency는 `0.05`로 유지한다. 따라서 endpoint 왕복만 억제하고 Bézier control
  point를 통한 동적 곡선 회피 자유도는 제한하지 않는다.
- analytic Box1 footprint penalty는 `25→10`으로 낮췄다. 근접 경로 shaping이 PPO reward를
  지배하는 현상을 줄이되, 실제 rollout에서 box가 밀린 양의 penalty weight `2.0`은 유지한다.

## 2026-09-16

### 고정 거리 대신 retreat 충돌 회피 shaping 강화

- 직선 prior에서 곡선 우회를 거의 탐색하지 못하던 Bézier control std를 `0.03→0.12`로
  높였다. box/goal anchor std는 `0.03`, A1 endpoint std는 `0.20`으로 유지하고 retreat path
  consistency scale은 `0.10→0.05`로 낮췄다. 46도 invalid turn 제한은 유지한다.
- Agent2를 피하다가 놓인 Box1을 치는 해를 억제하기 위해 retreat footprint penalty 기본값을
  `10→25`, 실제 bottom-box displacement penalty weight를 `0.5→2.0`으로 높였다. 각각
  `STACK_PLANNER_RETREAT_BOX_PENALTY`, `STACK_PLANNER_BOTTOM_DISTURBANCE_WEIGHT`로 조절한다.
- `A1_RETREAT`에서 선택된 planner transition은 total reward에서 `potential_delta`를 제거했다.
  placement/A2 stack potential은 유지하되 retreat 중 A1-box 거리 증가 자체는 보상하지 않고,
  collision·box disturbance·fall/time 및 analytic path collision penalty로 학습한다.
- planner task의 A2 handoff를 A1 이동 거리/endpoint 완료와 분리했다. bottom box 안정 판정 즉시
  retreat를 열고, 안정 상태가 추가로 1초(`STACK_PLANNER_A2_STABLE_DELAY`) 지속되면 A2를 바로
  출발시킨다. planner 없는 sequential-stack task에는 영향을 주지 않는다.
- planner task에서는 `STACK_RETREAT_DIST=1.5m`와 learned endpoint 도달을 A2 phase 전환에
  사용하지 않으며, predicted endpoint에도 고정 거리 reward를 추가하지 않았다.
- 놓인 box의 확장 footprint 안에서 retreat path가 끝나면 endpoint overlap penalty를 직접
  부과한다. footprint 밖으로 나온 뒤에는 더 멀리 갈수록 추가 reward를 주지 않는다.
- valid plan으로 채택된 경로만 보던 기존 계산을 바꿔 모든 finite retreat 후보의 path와
  endpoint clearance를 채점한다. 초기 invalid 후보도 방향별 충돌 위험을 구분해 PPO credit을
  받을 수 있다.
- 아직 plan이 채택되지 않은 retreat의 invalid action에서 clearance/route/unsafe shaping을
  버리고 binary invalid penalty만 남기던 override를 제거했다.
- `retreat_box_endpoint_clearance` metric을 checkpoint/log에 추가했다.
- 검증: unit test 26개, Python compile 및 diff check 통과.

### PhysX CUDA driver compatibility 경로 영속화

- stack planner의 train/view/eval launcher가 사라질 수 있는
  `/tmp/hwanhee-physx-lib`를 선택적으로 사용하는 대신, 시스템 `libcuda.so.1`을 찾아
  저장소 `.runtime/physx-lib/libcuda.so` 호환 링크를 자동 생성하고 항상 로드한다.
- local GPU 1 학습 wrapper가 로그와 달리 `MA_GPU=0`을 export하던 오류를 `MA_GPU=1`로
  수정했다.
- 검증: 생성된 링크에서 `ctypes.CDLL("libcuda.so")` 로드, 관련 launcher bash 문법 및
  diff check 통과. GPU simulator 재실행은 수행하지 않았다.

### Planner 환경 reset의 Isaac Gym state setter 단일화

- episode 전환 로그를 CUDA 동기화해 확인한 결과 `env.step`은 정상 완료되고 reset 내부
  state commit 직후 GPU pipeline이 깨지는 것으로 범위를 확정했다.
- 상속된 multi-task/F22/sequential reset은 한 simulation step 안에서 indexed root-state
  setter를 humanoid, object, platform, 최종 humanoid 순으로 여러 번 호출하고 있었다.
- planner train/view 전용 mixin은 상속 reset의 텐서·bookkeeping 변경은 그대로 실행하되
  중간 Gym setter/refresh를 보류하고, 마지막에 해당 env의 전체 actor root를 한 번과 humanoid
  DOF를 한 번만 제출한다. 이후 simulator tensor, observation, AMP history를 다시 동기화한다.
  `TokenHSI-masteer` 및 planner 없는 환경은 변경하지 않았다.
- 검증: Python compile, unit test 26개 및 diff check 통과. 실제 GPU episode reset 재검증은
  사용자가 실행할 단계로 남겼다.

### Frozen agent의 path 중간 release 진단 view

- 기존 진단이 의도와 다르게 sequential-stack의 `putdown_retreat` phase 전환을 사용하던 것을
  수정했다. 이제 전용 task는 평범한 `HumanoidMASteerCarry`를 직접 상속하며 stack coordinator,
  phase, release/retreat signal을 전혀 사용하지 않는다.
- 실제 carry goal은 그대로 둔 채 경로 생성 순간에만 goal 너머 endpoint를 임시 target으로
  사용한다. 생성 직후 carry target을 복원하므로 reward/observation은 중간 goal에 놓기를 요구하고,
  steering path는 그 뒤까지 계속된다. box→endpoint leg는 직선으로 고정해 goal을 정확히 지난다.
- planner 없이 frozen agent만 사용하는 `view_midpath_release.sh`와 전용 task를 추가했다.
  이 진단만 변경하며 기존 task와 학습 실행은 변하지 않는다.
- 전용 local view는 CUDA remap을 제거하고 sim/rl/graphics/default torch device를 모두 같은
  physical `MA_GPU`(기본 1)로 명시한다. graphics logical 0이 physical GPU 1이라는 기존 판단은
  잘못이어서 철회했다.
- 생성 직후 observation을 그대로 쓰고, carry-only scene에 없는 generic sit/climb marker
  actor 갱신도 생략한다. 실제 물리 state 갱신은 유지한다.
- 표준 player의 episode 경계 `reset(None)` 대신 계속 실행되는 전용 loop에서 done env의
  agent row만 명시적으로 reset한다. 한 episode로 종료하는 우회가 아니라 같은 프로세스에서
  시나리오를 반복하며 planner 학습/평가의 reset 호출 규약과 맞췄다.
- 전용 view는 기본적으로 각 physics step과 done-env reset 뒤 CUDA를 동기화한다. 비동기로
  늦게 보고되는 illegal access가 `env.step`에서 시작됐는지 reset에서 시작됐는지 traceback에
  최초 실패 구간을 표시한다. 필요하면 `STACK_CUDA_SYNC_DEBUG=0`으로 끌 수 있다.
- generic marker actor 갱신을 끈 뒤 누락됐던 path 표시를 actor setter가 필요 없는 line
  renderer로 복구했다. cyan 띠는 A1 정책에 실제 입력되는 `_gt_path`, 노란 십자는 stack goal,
  빨간 십자는 path endpoint다. reset 로그에 생성 endpoint와 설치된 path 끝의 오차도 출력한다.
- 단일 선이 checkerboard 바닥에 묻히므로 A1 path를 바닥에서 14cm 띄운 36cm 폭의 밝은
  cyan 평행선 ribbon으로 변경하고 goal/endpoint 십자 크기도 키웠다.
- framework construction reset의 hook 순서에 path 설치를 의존하지 않는다. checkpoint/player
  생성 직후와 각 done-env reset 직후 `install_midpath_paths()`를 명시적으로 호출하고 policy
  observation을 다시 계산한다. renderer는 endpoint 상태가 아직 없으면 안전하게 건너뛴다.
- 검증: unit test 26개, Python compile, bash syntax 및 diff check 통과. 전용 반복 loop의 실제 GUI
  episode 전환 검증은 사용자가 실행할 단계로 남겼다.

## 2026-09-15

### Retreat checkpoint 독립 평가 하네스

- eval_retreat.sh/py로 학습 sidecar를 재사용하는 deterministic 평가를 추가했다. 학습과
  baseline은 수정하지 않으며 고유 eval 출력 디렉터리로 기존 결과 덮어쓰기를 거부한다.
- episode raw 기록과 endpoint drift/path MAE, reach/stop/fall, collision proxy 및 minimum
  gap 요약을 저장한다. 미완결 episode는 rate 분모에서 제외하고 censored 수를 보고한다.
- Agent2 접근/정지 자연 발생 subset을 기록한다. 강제 개입 scenario는 아직 구현하지 않았다.
- 검증: unit test 26개 통과, Python compile/bash syntax/diff check 통과. 실제 simulator
  평가와 동적 회피 성능 검증은 아직 실행하지 않았다.

### Planner view 기존 경로 overlay 제거

- 실행 path(분홍/주황)와 steering window(cyan/초록) 표시를 제거하고 흰 planner 원본과
  파란 가상 box만 남겼다. 실행·학습 로직과 non-planner viewer는 변경하지 않았다.
- 검증: view_env Python compile 및 diff check 통과. GUI 육안 검증은 수행하지 않았다.

### Retreat goal 유지와 동적 body/held-box 회피 학습

- 매 retreat action에서 직전 world endpoint 대비 soft drift cost를 PPO reward에 추가했다.
  default tolerance 0.10m, coefficient 0.10, bounded cost max 4이며 reset/최초 진입은 mask한다.
  endpoint latch/수동 위험 event는 없고 현재 planner endpoint가 계속 실행/virtual box에 적용된다.
- retreat A1의 전체 path consistency는 default scale 0.10으로 약화해 같은 goal을 유지하면서
  우회하기 쉬워지게 했다. 기존 position-valid denominator를 유지하고 error contribution만
  줄여 scale이 정규화에서 상쇄되지 않도록 했다. 실제 설정은 metrics/checkpoint extras에 남긴다.
- 기존 root/box 거리 proxy에 real held box의 yaw/XYZ extent와 상대 비손 rigid-body 위치로
  계산한 3D proximity cost(max-body penetration, weight 10)를 더했다. 자신의 box는 제외한다.
  planner 전용 reward만 변경하므로 기존 non-planner task에는 영향이 없다.
- 검증: unit test 24개, Python compile 및 diff check 통과. 로컬 GPU 1에서
  `smoke_20260915_retreat_adapt`(8 env, horizon 4, low steps 6, 1 iteration)의
  rollout/PPO/checkpoint 저장이 exit 0으로 완료됐다. 짧은 smoke이므로 실제 retreat 중
  동적 회피 성능을 검증한 결과는 아니며 변경 reward로 추가 학습이 필요하다.

### Planner virtual box Z를 바닥 기준으로 고정

- virtual center Z를 STACK_GROUND_Z(default 0) + box_height/2로 설정한다. planner adapter에서
  inherited Carry observation의 center/BPS/goal Z를 모두 같은 높이로 보정하며 실제 box 상태는
  쓰거나 변경하지 않는다. 실제 box Z를 복사하던 viewer 코드도 제거해 동일 buffer Z를 그린다.
  planner 없는 inherited task 코드는 변경하지 않았다.

### Virtual box를 최신 planner 끝점에 무조건 배치

- 사용자 요청대로 valid/installed/retreat_ready 조건을 virtual box 배치에서 제거했다.
  매 output의 candidate 0, A1 마지막 world XY를 즉시 사용하며 view에서도 숨기지 않는다.
  대기 중 실제 Carry token으로 되돌리던 override도 제거했다. 실행 path validity와 A2 gate는
  별도 실행 정책으로 남지만 가상 box 생성/표시에는 관여하지 않는다. 기존 non-planner task는
  변경하지 않았다.

### Virtual retreat box를 planner world endpoint에 정렬

- execution_view의 suffix 전체 평행이동을 제거했다. 첫 점만 현재 root에 연결하고 남은
  suffix 샘플은 원래 world 좌표를 유지하며 마지막 점을 원본 planner endpoint와 정확히 맞춘다.
  따라서 짧거나 zero-length suffix를 root로 옮겨 가상 box가 agent에 생기던 변환을 제거했다.
- 미채택 대기 중에는 root 위치의 가상 box를 만들거나 표시하지 않는다. planner 전용
  observation bridge는 실제 Carry token을 유지하고 zero steering으로 대기하며, 채택 이후에만
  virtual token/wireframe을 사용한다. readiness는 A2 시작 뒤에도 유지하고 reset/새 retreat에서
  해제한다. planner 없는 기존 task는 변경하지 않았다.
- 검증: Python compile/기존 회귀와 endpoint 보존·zero-length suffix 회귀를 포함한 21개 검사
  통과. 실제 native view 및 물리 대기 자세는 아직 검증하지 않았다.

### 사용자 요청으로 retreat execution latch 제거

- 매 active planner tick의 valid retreat suffix/path/virtual endpoint를 재설치한다.
  retreat_ready는 최초 유효 후보 이전의 대기/A2 gate에만 쓰며 replan을 막지 않는다.
  manual fallback 제거는 유지했고 planner 없는 task는 수정하지 않았다.
- 매 retreat tick을 PPO decision으로 처리하며 clearance shaping도 매 설치마다 계산한다.
  inherited 진행도 origin은 phase 진입 root를 유지해 replan마다 progress를 초기화하지 않는다.
  moving-target 방지는 제거되므로 endpoint가 이동 중 변할 수 있다.

### Planner 전용 retreat에서 manual fallback 제거

- stack_planner/env_adapter.py subclass에서만 inherited manual retreat goal/path 생성 hook을
  stationary hold로 override했다. invalid 후보이면 출발하거나 fallback을 latch하지 않고,
  hold-position virtual box와 zero steering command를 유지하며 다음 planner decision을 기다린다.
  유효 후보가 채택된 순간 learned suffix/path/endpoint와 virtual box를 함께 latch한다.
- 미채택 hold target의 도달 판정이 A2를 조기 활성화하지 않도록 A2 goal activation과 phase
  전환을 모두 latch 여부로 gate했다. invalid 대기 action은 invalid penalty만 기록하며 GAE
  경계를 끊고, 채택 이후 continuation은 기존 actor mask/GAE 동작을 유지한다.
- TokenHSI-masteer의 기존 sequential-stack/steer/base task는 수정하지 않았다. 따라서 planner
  없는 기존 task에는 적용되지 않는다. Python compile 및 기존 단위검사 20개 통과.
  실제 subclass를 mock state로 호출해 stationary hold/virtual-box 좌표와 미채택 env의 A2
  activation/phase side-effect 차단도 통과했다. 실제 물리 rollout에서 대기 자세는 미검증이다.

### Planner view의 GPU 선택과 원본/실행 path 시각화 분리

- 단일 add_lines가 한 픽셀 실처럼 보여 경로를 구분하기 어려운 문제를 보완했다. 원본은
  0.10m 흰 띠, 실행 path는 0.30m 색 띠, steering은 0.16m 띠로 표시하며 가상 box edge도
  0.05m로 보강했다. Isaac Gym의 line-width 미지원은 촘촘한 평행 선으로 처리한다.

- view launcher가 모든 Python 실행 전에 CUDA_VISIBLE_DEVICES를 설정하고,
  TOKENHSI_GRAPHICS_DEVICE_ID와 graphics CLI ID를 명시한다. 기본 Vulkan ordinal은 MA_GPU와
  같으며 Vulkan 장치 순서가 다르면 override할 수 있다. compute GPU 1과 graphics GPU 0이
  함께 사용될 수 있던 launcher 설정을 제거했다.
- 최신 candidate-0 원본 path는 흰 선, 실제 _gt_path는 분홍/주황, 정책 steering window는
  cyan/green, observation-space virtual retreat box는 파란 wireframe으로 그린다. 설치되지
  않거나 latch 때문에 무시된 원본도 실제 실행 path와 비교 가능하다.
- inherited speed-run/decimation renderer 대신 dense 실행 구간과 끝점을 그린다.
  phase/valid/installed/retreat_latched 변화는 콘솔에 출력하며 물리 actor는 변경하지 않는다.
- 검증: bash syntax/Python compile 및 기존 순수 PyTorch 단위검사 20개 통과.
  실제 native window와 Vulkan physical-device mapping은 아직 검증하지 않았다.

## 2026-09-14

### A1 retreat를 단일 PPO option으로 고정

- retreat 진입 첫 decision을 validity와 무관하게 latch한다. 유효한 learned suffix는 그대로
  설치하고, invalid이면 phase 진입 때 만들어진 inherited outward fallback과 그 endpoint의
  virtual box를 함께 고정한다. 따라서 invalid 뒤 후속 valid replan이 path와 virtual box를
  이동 중에 갑자기 교체하지 못한다.
- latch된 retreat의 후속 planner tick은 PPO actor decision에서 제외했다. 그동안의 physical
  reward는 GAE를 통해 최초 retreat action으로 전달되며, 실행되지 않은 후속 샘플에는 붙지
  않는다. route/validity shaping과 consistency도 실제 decision tick에만 적용한다.
- invalid action은 fallback의 물리 성과를 자기 보상으로 가져가지 않도록 analytic penalty만
  기록하고 GAE 경계를 끊는다. `planner_decision_rate`를 추가해 option continuation 비율을
  확인할 수 있게 했다.
- 로컬 physical GPU 1, `tokenhsi118`, 로컬 executor/stage1 checkpoint 배치를 기본으로 쓰는
  `stack_planner/train_local_gpu1.sh`를 추가했다. 24-GB GPU에서 1024 env와 자동 기본값과 같은
  8192 PPO minibatch의 backward를 smoke 검증해 기본 env 수를 1024로 두었으며, 기존 서버용
  launcher의 GPU 7 기본값은 변경하지 않았다.

### A1 retreat moving-target 제거

- phase 2의 매 replan마다 model suffix를 현재 root에 다시 붙이며 virtual retreat endpoint가
  함께 달아나던 오류를 수정했다. phase 2 진입 뒤 처음 설치에 성공한 A1 suffix/path/endpoint를
  latch하고, phase가 끝날 때까지 후속 replan이 `_gt_path`, arc progress와 virtual box를
  덮어쓰지 않는다.
- inherited FSM의 phase 2 완료 기준도 legacy manual retreat goal/direction 대신 같은 latched
  learned endpoint/start/direction을 보도록 맞췄다. planner는 계속 단일 full path만 출력하며
  latch는 frozen Carry compatibility execution에만 존재한다.
- A1 retreat execution path가 놓인 Box1을 관통하는 문제에 대해 box yaw/size와 agent 반경을
  포함한 oriented-footprint clearance penalty를 추가했다. phase 2 최초 latch에만 적용해 운반 중
  필요한 자기 box 접촉이나 실행되지 않는 후속 action을 벌하지 않으며, box 쪽으로 더 깊이
  들어가는 경로와 끝까지 footprint를 벗어나지 않는 경로를 각각 감점한다.
- clearance adapter가 `CoordinatorState`의 실제 yaw 필드명 `box_heading` 대신 존재하지 않는
  `box_yaw`를 참조해 첫 retreat sample에서 죽던 오류를 바로잡았다.

### Stack planner PPO env scaling 수정

- 2,048 env rollout에서 64-env용 고정 minibatch 512를 그대로 사용해 iteration당 optimizer
  step이 12회에서 384회로 32배 증가하고, route visit penalty·value loss·path length가 함께
  발산하는 설정 오류를 수정했다.
- launcher의 기본 minibatch를 `envs × horizon / 4`로 계산해 PPO epoch당 4 minibatch를
  유지한다. 기본 horizon 32에서 64 env는 512, 2,048 env는 16,384이며 학습 시작 로그에
  실제 minibatch와 iteration당 optimizer step 수를 출력한다.

### Planner execution metric 추가

- iteration별 `path_mae`, `fall_ratio`, `collision_ratio/cost`를 추가했다. path MAE는 현재
  phase에서 active인 agent root와 실제 설치된 dense path 사이의 평균 횡오차이며, collision은
  발생 low-level step 비율과 기존 연속 proxy cost를 분리해 기록한다.
- bottom box는 placement 이후 phase에서만 선속도, 각속도와 누적 이동량을 집계한다.
  `bottom_postplace_exposure`를 함께 기록해 0이 안정성을 뜻하는지 placement 미도달을 뜻하는지
  구분할 수 있게 했다.

## 2026-09-11

### 단일 path 모델과 Carry execution slicing 분리

- planner 공개 출력을 `path_local/path_world [B,K,2,33,2]`로 한정했다. 별도 retreat goal/path,
  speed, acceleration, switch, risk 출력은 없으며 PPO value는 policy 내부 critic으로만 사용한다.
- 모델은 Agent 1의 `root → box → carry goal → final endpoint`를 한 path로 생성한다. frozen Carry
  실행 시에만 ordered projection으로 carry goal까지의 prefix를 33점 보간하고, placement 뒤에는
  같은 path의 goal 이후 suffix를 현재 실제 root로 평행이동해 33점 보간한다.
- pre-placement 실행 path의 끝은 실제 carry goal에 정확히 고정하고, post-placement 실행 path의
  끝은 기존 low-level policy 호환용 virtual box로 사용한다. 속도는 execution adapter가 고정값으로
  부여한다. 계약 변경을 분리하기 위해 checkpoint schema를 v5로 올렸다.
- execution slicing을 Isaac Gym과 분리한 순수 PyTorch 모듈로 만들고 prefix endpoint, suffix translation,
  혼합 agent 동작을 포함해 stack planner 단위검사 19개가 통과했다. 이어 8-env, horizon 8,
  low-step 6 simulator smoke가 PPO update와 v5 path-only checkpoint 저장까지 통과했다.
- 실제 학습 재현을 위해 launcher sidecar에 init checkpoint, seed, PPO gamma/GAE/clip/value/entropy와
  stack episode tolerance를 모두 기록한다. resume는 v5 checkpoint와 새 tag를 요구하며 iteration 수는
  checkpoint 이후 추가 실행량으로 명시했다.
- 장기 학습은 서버의 physical GPU 7 할당을 사용하므로 stack planner train launcher의 기본
  `MA_GPU`를 7로 설정했다. 프로세스 내부 CUDA/Isaac Gym device는 `CUDA_VISIBLE_DEVICES=7`에 의해
  기존과 동일하게 logical `cuda:0`으로 유지된다.

### Retreat 전환을 규칙이 아닌 reward 학습 대상으로 변경

- 직전 v3의 65-point route/suffix와 physical threshold 기반 route collapse를 제거했다. planner는
  다시 32개 segment의 단일 33-point path만 출력하며 별도 retreat goal/head/switch는 없다.
- 초기 zero prior는 기존 Carry와 같이 endpoint가 box goal인 `root → box → goal`이다. PPO는
  경로를 직접 바꿀 수 있고, 실제 placement quality가 낮을수록 endpoint-goal 오차와 ordered
  visit 오차를 크게 받는다. placement quality가 연속적으로 올라가면 이 shaping이 약해지고
  기존 direction-free clearance outcome reward가 retreat path를 선택하게 한다.
- decoder/runtime가 `retreat_ready`를 판정해 trajectory를 교체하지 않으므로 언제 path가
  placement형에서 retreat형으로 변할지는 state-conditioned planner가 reward로 학습한다.
  action/trajectory 의미 변경을 분리하기 위해 checkpoint schema를 v4로 올렸다.
- 순수 PyTorch 단위검사 18개와 8-env, horizon 3, low-step 6 simulator smoke가 PPO update와
  v4 checkpoint 저장까지 통과했다. 보수적인 endpoint exploration std `0.08`에서 초기
  endpoint/visit shaping reward는 각각 `-0.197/-0.227`로 유한하게 계측됐다.

### Stack planner를 단일 end-to-end trajectory로 통합

- 별도 interaction/retreat head와 사후 `agent1_full_*` concatenation을 제거하고, planner의
  공개 출력을 joint 65-point `path_world/speed/trajectory` 하나로 통일했다. Agent 1은 한
  path 안에서 `root → box → stack goal → learned endpoint`를 표현하고 Agent 2는 goal 이후
  같은 위치를 유지한다. 사용하지 않던 pickup dwell 출력도 함께 제거했다.
- 매 replan에서 실제 `held`, box-goal 거리, box 선·각속도로 placement 완료를 판정한다.
  완료 뒤에는 같은 Agent 1 path의 지난 route가 current root로 접혀 learned endpoint까지의
  남은 retreat만 실행되며, 별도 retreat path나 planner 입력 phase signal로 head를 고르지 않는다.
- ordered projection penalty는 아직 방문이 필요한 agent에만 적용하고 temporal consistency,
  executor resampling, validity와 PPO action contract를 65-point 단일 path에 맞췄다. incompatible
  checkpoint가 조용히 로드되지 않도록 schema를 `tokenhsi-stack-planner-v3`로 올렸다.
- 순수 PyTorch 단위검사 18개와 8-env, horizon 3, low-step 6 실제 simulator smoke가
  PPO update 및 v3 checkpoint 저장까지 통과했다. smoke의 평균 box/goal projection 거리는
  `0.0959/0.0727 m`, visit penalty reward는 `-0.00741`이었다.

### Stack planner temporal plan consistency

- 기본 30 action-step plan hold 뒤의 새 replan이 매번 독립적으로 흔들리지 않도록, 직전
  mean plan의 아직 실행되지 않은 future를 elapsed time만큼 정렬해 현재 mean plan과 비교하는
  temporal consistency loss를 추가했다.
- 일반 root→box→goal path/speed와 실제 execution bridge가 소비하는 Agent 1 retreat
  path/speed를 함께 학습하며, episode reset과 agent별 물리 상태 phase 변화는 mask한다.
- `STACK_PLANNER_CONSISTENCY_COEF`를 기본 `1.0`으로 launcher·sidecar·checkpoint·metrics에
  기록한다. time alignment, reset/phase mask, 두 path head gradient를 포함한 stack planner
  단위검사 14개가 통과했다. 8-env, horizon 3, low-step 6 실제 simulator smoke도 PPO update와
  checkpoint 저장까지 통과했고 consistency가 `0.00548`, valid fraction이 `0.3333`으로 기록됐다.

### Agent 1 placement-to-retreat full trajectory

- 33-point interaction route와 learned retreat suffix를 연결한 65-point
  `agent1_full_path_world/speed/trajectory` 출력을 추가했다. interaction route의 실제 learned
  endpoint에서 suffix가 연속적으로 시작한다.
- frozen Carry용 retreat path는 기존처럼 현재 actual root에 re-anchor해 실행하되, full trajectory는
  같은 learned displacement를 route 끝에 붙인다. 따라서 agent의 box task goal을 retreat endpoint로
  바꾸지 않고 planner의 최종 retreat objective를 별도로 표현한다.

### Hard waypoint를 ordered projection constraint로 교체

- ordinary path의 `P16=box`, `P32=goal` hard anchor를 제거하고 `P0=current root`만 고정했다.
  zero-initialized head는 기존 root→box→goal 경로를 prior로 유지하지만 junction과 endpoint를
  포함한 나머지 경로는 모두 학습 가능하다. action/head shape가 달라 checkpoint schema를
  `tokenhsi-stack-planner-v2`로 올려 구 weight의 의미가 조용히 바뀌지 않게 했다.
- 각 target을 모든 유한 선분에 투영하고, 서로 다른 선분 index 또는 같은 선분의 projection
  fraction으로 `box → stack goal` 방문 순서를 강제하는 reward penalty를 추가했다. 기본 허용
  반경은 0.15 m, 계수는 10이며 launcher sidecar/checkpoint/metrics에 함께 기록한다.
- stack adapter의 validity는 root anchor, finite, buffer, speed, curvature만 검사한다. box/goal
  방문 여부는 hard invalidation이 아니라 위 연속 penalty가 담당한다.
- 순수 PyTorch 단위검사 17개와 8-env, horizon 3, low-step 6 실제 simulator smoke가 통과했다.
  smoke는 v2 checkpoint 저장까지 완료했고 평균 box/goal projection 거리는 각각
  `0.0896/0.1029 m`, visit penalty reward는 `-0.0168`이었다.

## 2026-09-10

### Deterministic executor path-distortion MLP

- `execution_bias/`에 joint 33-point `(x,y,v)` 계획 전체를 받아 각 planned timestamp의
  world-coordinate `(dx,dy)` 33개를 직접 출력하는 deterministic residual MLP를 분리했다.
- shared SE(2) canonicalization으로 전역 이동·회전에 equivariant하면서 두 agent의 상대
  geometry와 먼 goal 문맥은 보존한다. zero-init 모델은 정확히 identity execution이고,
  `planned_xy + predicted_error`를 planner collision cost에 넣을 때 gradient가 plan까지 흐른다.
- nominal-time simulator trace 정렬, curve-weighted supervised loss, strict checkpoint, NPZ
  dataset/training CLI와 planner candidate adapter를 추가했다. 새 전용 검사 6개와 기존
  stack-planner 11개, coordinator 78개 검사가 모두 통과했다.

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
- 기본 30 action step 및 stack phase 전환마다 실제 simulator state에서 재계획하고,
  planner의 path/speed와 learned retreat endpoint를 frozen sequential-stack agent에 적용한다.
- 기존 coordinator/viewer와 파일을 공유하지 않는다. launcher는 noVNC 없이 로컬 Isaac Gym
  창을 직접 띄우고 기본 GPU 0을 사용하며 `MA_GPU`로 로컬 GPU를 선택할 수 있다.
- retreat head는 이미 있으나 중복 contract field 두 개가 없던 초기 v1 weight도 해당 값을
  model config에서 복구한 뒤 나머지 schema와 state dict를 동일하게 strict 검사한다.
- 로컬 viewer의 첫 render에서 generic marker updater가 carry-only scene에 없는 sit/climb
  actor ID를 함께 commit해 PhysX illegal memory access가 나던 것을 막았다. planner path
  overlay는 유지하고 시뮬레이션 state를 쓰지 않는 viewer 전용 marker 갱신만 생략한다.

### Stack planner plan hold와 replan 경계 속도 연속성

- 학습 macro action과 viewer의 기본 plan hold를 6 action step(약 0.2초)에서 30 step(30 Hz에서
  약 1초)으로 함께 늘렸다. 일반 구간은 그동안 accepted path를 유지하고 stack phase 전환은
  즉시 replan한다.
- 각 path 내부 가속도 제한과 별도로, accepted plan이 바뀔 때 executor에 전달되는 속도
  command도 이전 command에서 기본 `0.75m/s²` 이하로 변화시킨다. 새 knob는
  `STACK_PLANNER_COMMAND_ACCEL`이며 학습 sidecar에 기록한다.
- invalid replan은 기존처럼 마지막 valid path를 덮어쓰지 않는다. 이 변경은 GT trajectory나
  retreat 방향을 추가하지 않고 plan의 시간적 유지와 물리적 command 연속성만 보장한다.

### Stack planner episode full reset

- stack 전용 reset이 humanoid root와 box/platform root를 두 번의 indexed setter로 나눠
  제출하던 것을 한 번의 actor-ID commit으로 합쳤다. Isaac Gym GPU pipeline에서 마지막
  setter만 남아 이전 episode의 agent 또는 box pose가 물리에 유지될 수 있던 문제를 막는다.
- reset 대상은 해당 env의 agent 2명, box 2개와 활성 platform 전부이며 DOF, root/object
  linear·angular velocity, stack phase, planner path/speed 상태도 새 episode 값으로 초기화한다.
- sequential 후처리가 target platform을 숨기며 추가 root setter를 호출하는 경우에도, 모든
  후처리 뒤 agent·box·platform 전체 actor 집합을 마지막 authoritative reset으로 다시 제출한다.
- full-stack planner의 episode reset은 기본 `STACK_PLANNER_FRESH_START=1`로 carry skill
  분포를 `loco_carry=1.0`에 고정한다. 기존 mixed reset의 `carryWith`/`putDown` RSI 때문에
  정상 reset도 이전 episode의 마지막 자세처럼 보이고 전체 순서를 건너뛰던 문제를 제거한다.

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

## 2026-09-21

### carry planner spatial exploration 및 경로 학습 진단

- 기존 기본값 `delta_std=0.12`, `path_update_alpha=0.25`에서는 smoothing과 point
  ramp까지 거친 stochastic path가 직선 base에서 수 cm만 벗어나 collision reward가
  사실상 속도 쪽으로만 흐를 수 있었다. fresh-run 기본값을 각각 0.25, 0.5로 올렸다.
- 매 iteration에 sampled/deterministic path의 base 대비 평균·최대 이탈 거리(m),
  proposal 유효 비율, 현재 path action std를 기록한다. 이제 "직선처럼 보임"이 작은
  exploration인지, invalid proposal fallback인지, 아직 mean이 학습되지 않은 것인지
  학습 로그에서 분리할 수 있다.

### carry planner 96-step analytic path collision supervision

- 33-point 전체 action에 이후 6 physical step의 scalar PPO reward만 주던 긴-horizon
  credit 희석을 보완했다. episode 첫 full-plan을 measured ms18 timing으로 96개 미래
  시점에 펼치고, 충돌이 큰 top-8의 agent-agent/agent-box/box-box overlap을 직접
  path head에 역전파한다.
- 보조항의 speed profile은 도착 시간 계산에는 쓰되 detach하여 감속만으로 loss를
  피하지 못하게 했다. 이미 지나간 fixed prefix를 미래로 오인하지 않도록 recurrent
  replan에는 analytic loss를 적용하지 않고 physical PPO만 유지한다.
- `analytic_collision_loss`, 적용 batch 비율, 예측 최소 agent 거리와 box margin을
  iteration 로그 및 checkpoint metadata에 기록한다.
- invalid proposal이 commit되지 않은 것을 새 episode로 오인해 analytic loss를 반복
  적용하던 feedback을 제거했다. 적용 mask는 이제 실제 reset transition에서만 만들며,
  기본 collision 계수는 10에서 1로 낮추고 46도 초과 곡률에 직접 cosine penalty를
  추가해 공간 우회와 실행 가능한 곡선을 함께 학습한다.

### carry planner multi-distribution evaluation

- single-env timed-cross viewer와 별도로 deterministic multi-env evaluator를 추가했다.
  `mixed`(학습과 같은 75% close-goal), `converge`, 일반 `cross`, `free`를 seed별로
  평가하고 기존 MA episode metrics와 coord summarizer를 그대로 사용한다.
- suite 기본은 64 env × evaluator 3회 × 4 profiles × seeds 0/1/2이며, raw npy/log와
  JSON summary를 `runs/results/carry_planner/<run>/`에 분리 보존한다.
- multi-env 평가의 비동기 reset에서 한 env의 due event가 전체 batch history를
  shift/replan하던 viewer 전제를 제거했다. 선택된 env의 history/path/tick만 갱신해
  학습과 동일한 env별 6-step replan 주기를 유지한다.

### carry planner 물리 GPU index 고정

- `MA_GPU` 숫자를 `CUDA_DEVICE_ORDER=PCI_BUS_ID`의 CUDA ordinal로 직접 넘기면 서버의
  CUDA/NVML 열거 순서 차이 때문에 nvitop에서 다른 물리 GPU를 잡을 수 있었다.
- train/view launcher가 `nvidia-smi -i <MA_GPU>`로 NVML index의 UUID를 해석한 뒤
  `CUDA_VISIBLE_DEVICES`에 UUID를 넘긴다. 프로세스 내부 device는 계속 logical
  `cuda:0`이지만 nvitop의 물리 GPU는 `MA_GPU`와 정확히 일치한다.
- UUID 하나만 노출한 CUDA compute(Torch/PhysX)는 logical `cuda:0`을 유지한다. 반면
  viewer의 Vulkan graphics index는 CUDA visibility와 무관한 물리 ordinal이므로,
  non-headless carry viewer에서만 graphics id를 `MA_GPU`의 NVML index로 맞춘다.

### stack planner V13 single-head viewer compatibility

- `view.sh`와 interactive viewer에서만 V13 checkpoint의 당시
  `previous_path + bounded delta` decoder를 복원한다. V15 기본값을 억지로 채워 path 의미가
  달라지는 migration은 하지 않는다.
- 학습 재개용 strict loader는 계속 V13을 거부한다. legacy contract와 state dict가 정확히
  맞을 때만 read-only viewer loader가 허용한다.

### retreat-only 학습 task subclass 허용

- `train_retreat_only.sh`가 등록하는 `HumanoidMAStackPlannerRetreatTrain`은 정식
  `HumanoidMAStackPlannerTrain` subclass인데도, closed-loop player가 클래스 이름의 완전
  일치만 허용해 초기화 직후 중단됐다. 검사를 `isinstance`로 바꿔 retreat-only subclass를
  허용하고, 이후의 environment mode와 checkpoint mode 일치 검증은 유지한다.
