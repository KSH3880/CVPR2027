# TokenHSI-ma — 변경 기록

> 파일 변경은 hook이 자동 기록. 무엇을/왜 바꿨는지는 Claude가 `###` 항목으로 덧붙인다.

## 2026-09-21
### Executor-aware world-model MPPI planner V2 스캐폴드

- `coordinator_v2/`에 actual-state history GRU, 저차원 joint path/speed proposal,
  recurrent ensemble world model, uncertainty-aware MPPI와 BOOM식 value-weighted
  alignment loss를 추가했다. TokenHSI action/observation과 MS18 checkpoint는 수정하지 않는다.
- 기존 33점 root/box/goal 계약과 deterministic collision/curvature selector를 재사용하고,
  `HumanoidMACoordCarry`에서 `COORD_MODEL=v2`일 때만 새 planner를 불러와 기존
  320점 `_gt_path`/`_mscale` bridge로 보낸다. 매 실제 step history를 갱신하며 기존
  6-step/phase-change replan과 invalid-plan fallback을 유지한다.
- strict atomic checkpoint와 초기화 스크립트, CPU 단위검사를 추가했다. 초기 체크포인트는
  학습 결과가 아니며 현재 MS18 Carry의 고정 agent-box 계약 때문에 role/assignment 선택은
  아직 구현 범위 밖이다.
- 검증: CUDA 비활성 Python compile, unit test 2개, checkpoint round-trip,
  WM supervised loss finite, `git diff --check` 통과. Isaac Gym GPU bridge smoke는 미실행이다.

### Top box 접촉면 평행 reward pilot

- 기존 `_box_parallel_error_deg`의 모서리/yaw 정렬 판정은 과거 ms75 replay를
  보존하도록 기본값으로 유지하고, opt-in `STACK_TOP_SURFACE_PARALLEL=1`에서만
  base 윗면과 top 아랫면의 signed normal 각도를 사용한다.
- `STACK_TOP_PARALLEL_REWARD_*`를 추가해 STACK phase의 목표 XY/Z 근처에서만
  `exp(-0.5 * (error_deg / sigma_deg)^2)` dense reward를 top agent에 지급한다.
  손 파지 수와 hand-clear는 reward gate에 사용하지 않는다.
- `train_ms75_faceparallel_local.sh`는 ms75와 같은 초기 checkpoint/설정을 재생하면서
  면 법선 성공 기준 10도, reward weight 0.25를 사용하고 release 관련 보상과
  hand-clear 요구는 0으로 고정한다.

### Local 환경의 Juan sequential-stack 평가기

- `juan_eval_sequential_stack.sh`와 전용 task/집계기를 추가해 ms75 sidecar와
  `HumanoidMASequentialStackRelease` 시나리오는 유지하면서 Juan의 env별 warm-up 제거,
  3x3 box grid, 540-episode 균형 집계와 성공 latch를 적용했다.
- 성공 모드는 `strict_done`, `box_radius`, `tokenhsi_carry`를 지원한다. 기존 player 변경은
  `MA_EVAL_ALL_AGENT_ROWS`/`MA_EVAL_FLUSH_DONE` opt-in일 때만 활성화된다.
- `bash -n`, Python compile, 합성 270-env/55열 집계 smoke를 통과했다.


### CLEAR 박스 39-D fade + 후퇴 goal 3-D 유지 및 goal reward

- opt-in `STACK_CARRY_OBS_KEEP_GOAL=1`은 ms75의 3-step Carry observation
  zero-fade에서 각 42-D 블록의 박스 상태 39-D만 fade하고, 마지막 goal 3-D는 고정된
  후퇴 endpoint를 매 frame root-local로 유지한다. local Z는 0으로 두어 낮은 box
  placement/putDown 목표로 해석되는 혼선을 줄였다.
- CLEAR에 endpoint 거리 감소, endpoint 도착 보상을 추가하고 CLEAR 진입 root 높이보다
  낮아지는 crouch와 배치한 박스 재접촉을 감점하는 opt-in reward를 추가했다. 기본값은
  모두 0이어서 기존 ms75/ms77 replay는 변하지 않는다.
- `STACK_RETREAT_GOAL_TOL`의 기본값은 기존 0.18m로 유지하고 새 wrapper만 0.05m를
  사용한다. 0.1m path cell보다 작은 tolerance도 실제 도달할 수 있도록 새 조건에서는
  마지막 활성 path cell을 요청된 retreat goal 좌표에 정확히 맞춘다. 따라서 감속 뒤
  goal 0.05m 안에서만 도착 latch, `M=0`, CLEAR→STACK 정지 확인이 시작된다.
- `train_ms75_goal3_boxfade_local.sh`는 ms75 sidecar를 재생해 ms18 epoch 9000,
  phase·10% carry rehearsal·freeze 계약을 보존하고 새 관측/reward만 켠다.
  `MA_TOKEN=mask`는 teammate를, train config의 `use_prior_knowledge: false`는 old-carry를
  attention에서 제외하므로 실제 입력 비교는 steering + goal-only new-carry다.
- 새 노브를 학습 sidecar 저장·재생 목록에 추가했다.
- Python `py_compile`, wrapper와 `train_local.sh`의 `bash -n`, `git diff --check`,
- 학습 shell의 PATH에 `rg`가 없어 시작 전 검사가 중단된 문제를 표준 `grep`으로 바꿨다.
  old-carry mask config 검사 통과와 동일 tag 재실행 가능 상태를 확인했다.
  기본 tag 미존재와 wrapper 실행 권한을 확인했다. 학습은 시작하지 않았다.

### 박스 없는 단일-agent steering 전용 뷰어

- 기존 `scripts/masteer/view.sh`를 변경하지 않는 `view_only_steering.sh`와
  viewer 전용 `HumanoidMAOnlySteering` 태스크를 추가했다. 래퍼는 원본을 `/tmp`에
  복사해 `--eval_task carry`만 `traj`로 치환하므로 Vulkan GPU guard, VNC,
  체크포인트 스냅샷 경로는 그대로 사용한다.
- A=1 `traj`로 carry box/platform과 carry 관측을 비활성화한다. 원래 TokenHSI
  traj와 custom steering 경로가 달라 4 m 이탈 시 조기 종료되던 문제를 막고,
  약 32 m steering 경로와 창을 600-frame 에피소드 내내 유지한다.
- `MS_CLIP=1`, `MS_ENDCLAMP=0`을 강제하며 낙상 또는 타임아웃 때만 새 위치와
  새 steering 경로로 리셋한다. 기존 학습·평가 태스크의 기본 동작은 바뀌지 않는다.

### Top WAIT의 최종 Carry goal 유지 zero-shot

- opt-in `STACK_TOP_WAIT_FINAL_GOAL=1`은 shared-goal Top의 safety path와 stage
  gate는 기존 중간점 W에 유지하면서, 42-D Carry observation/reward의
  `_box_tar_pos`만 nominal 최종 적층점 G로 복원한다. 관측 차원과 모델 ABI는
  바꾸지 않는다.
- 이에 따라 WAIT는 `distance(box,G)>0 + M=0`, 최종 placement는
  `distance(box,G)≈0 + M≈0`으로 구분된다. path 후보 생성에는 W가 필요하므로
  final G 복원은 `_reset_shared_goal_paths`가 끝난 뒤에만 수행한다.
- 기본값은 0으로 두어 기존 ms75/ms77 replay를 보존하고, 새 노브를 학습 sidecar
  저장·재생 목록에 추가했다.
- Python compile, `bash -n`, `git diff --check`를 통과했다. ms75 e12000에
  `STACK_TOP_WAIT_XY_ZERO=1 STACK_TOP_WAIT_FINAL_GOAL=1`을 zero-shot 적용한
  512-env 기본 3-repeat 실행은 2824행을 남겼지만 최종 rc 요약 전에 종료되어 진단
  결과로만 판정했다. 기존 waitxy1 대비 staged 0.442→0.011, XY 도착 후 freeze
  568/816→14/610으로 붕괴했고, 드문 freeze의 하강/반등 중앙값도
  0.748/0.780m로 유지됐다. full final G의 낮은 stack Z가 0.9m WAIT 높이와
  충돌하므로 이 zero-shot 표현은 기각한다.

## 2026-09-19

### Top WAIT XY 도착 즉시 M=0

- ms75 e12000의 512-env eval에서 freeze 성공 29개 중 23개가 safety-gate XY 도착
  이후 box center Z 중앙값 0.996m에서 0.200m까지 내려갔다가 0.993m로 다시 든 뒤
  freeze됐다. 목표 Z 0.900m 자체가 아니라 안정 5-frame을 기다리는 동안 계속된
  steering이 native Carry의 내려놓기·재집기 루프를 허용한 것으로 판정했다.
- opt-in `STACK_TOP_WAIT_XY_ZERO=1`은 grasp를 유지한 Top이 기존 safety-gate XY
  tolerance에 최초 진입하는 순간 steering `M=0`만 적용한다. 3-D WAIT 목표,
  안정 5-frame, CLEAR→STACK gate와 final goal commit/reset은 그대로 유지한다.

### Top WAIT 높이 하강·반등 eval 계측

- sequential stack eval metric 끝에 WAIT 목표 Z, safety-gate XY 최초 도착 시
  Top box Z, 도착 후 freeze 전 최저 Z, `M=0` freeze 순간 Z 4열을
  append했다. 정책·관측·보상·상태 전환은 바꾸지 않는 계측 전용이다.
- `stack_stage_summary.py`에 `STACK_TOP_WAIT_Z`를 추가해 XY 도착 Z 분포와
  도착 후 하강량(`descent`), 최저점에서 freeze까지 반등량(`rebound`)을
  바로 비교한다. 기존 106열 결과는 legacy로 계속 읽힌다.

## 2026-09-18

### 평가 반복 횟수를 실행별로 지정

- `trans_players.py`와 `amp_players.py`의 평가 반복 3회 기본값은 유지하면서
  `EVAL_NUM_REPEAT`로 덮어쓸 수 있게 했다. GUI 평가·녹화에서 더 많은
  reset 시나리오를 연속으로 보려는 용도이며, 1 미만 값은 거부한다.

## 2026-09-17

### STACK Base의 비행동성 box 보상 제거

- STACK에서 이미 배치된 Base box의 support/stability에 지급하던
  `0.40 * support + 0.25 * stable`을 제거했다. Base의 STACK 보상은 endpoint
  position-gated hold와 regrasp/foot-box 억제를 중심으로 두며 Top은 바꾸지 않는다.

### STACK Base hold reward를 endpoint 위치로 gate

- STACK의 Base hold reward에서 위치·속도·upright·양발 지지 항 전체를
  `exp(-8 * endpoint_error^2)`로 gate한다. endpoint에서 멀어진 Base가 자세와 양발
  지지만으로 hold reward를 받던 누수를 제거하며 Top reward는 바꾸지 않는다.
- TensorBoard에 `stack_state/base_hold_position_score`를 추가했다. 실행 중인 ms75는
  수정 전 코드를 이미 로드했으므로 이 변경이 적용되지 않는다.

### CLEAR endpoint 허용치 0.18 m

- ms74 epoch 10000의 end-cell-fix 평가에서 release 조건부 endpoint 도달이
  0.12 m에서는 0.625였지만, 기록된 최소 오차 기준 0.18 m에서는 0.819로 예상됐다.
  CLEAR→STACK 전환의 endpoint 허용치를 0.12 m에서 0.18 m로 완화했다.
- 실행 중인 ms74 프로세스에는 이미 로드된 0.12 m 조건이 유지되며, 새로 시작하는
  학습·평가부터 0.18 m 조건이 적용된다.
- `train_local.sh` sidecar에 `STACK_RETREAT_ENDPOINT_GATE`와
  `STACK_RETREAT_END_CELL_FIX`를 저장·재생하도록 추가했다.

### local ms68 PTH의 CLEAR gate zero-shot 대조

- 서버의 기존 endpoint-arrival 동작을 기본값으로 보존하면서,
  `STACK_RETREAT_ENDPOINT_GATE=0`일 때만 과거 local ms68의
  `arc 완료 + endpoint 감속/정지` CLEAR→STACK gate를 재생하는 평가용 opt-in을
  추가했다. reward·observation·정책 weight는 바꾸지 않는다.
- 같은 local ms68 e9800 PTH에 대해 legacy gate, 서버 0.12 m endpoint gate,
  endpoint gate + end-cell fix를 동일 seed/eval 설정에서 비교할 수 있다.

### ms70 retreat endpoint cell 정합 평가

- STACK_RETREAT_END_CELL_FIX를 기본 비활성 opt-in으로 추가했다. 활성화하면 retreat
  path의 끝을 기존 path[n_end-1]에서 다음 0.1 m cell까지 연장해, M=0 정지점과 원래
  clear goal의 0.12 m endpoint gate를 맞춘다. 기존 sidecar 재생 동작은 유지한다.
- CPU 1000-path 검사에서 기존 정지점의 goal 오차는 0.1000/0.1503/0.1998 m
  (min/median/max), 보정 후 0.0002/0.0513/0.0999 m였고 0.12 m 이내 비율은
  0.197에서 1.000으로 증가했다.
- ms70 e12000 checkpoint를 GPU 7, 512 env에서
  STACK_RETREAT_END_CELL_FIX=1로 재평가했다. release 조건부 endpoint 도달은
  0.513에서 0.702, CLEAR 통과는 0.278에서 0.395로 증가했다. 전체 base episode의
  STACK 진입은 0.211에서 0.290으로 증가했지만 strict final success는 여전히 0이다.
  남은 CLEAR 주 병목은 endpoint 이후 base 안정과의 동시 충족이다.

### CLEAR endpoint/stop 평가 funnel 계측

- sequential stack metrics의 기존 93열 뒤에 CLEAR 진단 13열을 append했다. 정책·보상·
  observation·gate는 바꾸지 않고, retreat arc 완료, endpoint 0.12 m 도달/arrival latch,
  humanoid stop 안정, stop candidate, pre-stop gate, 결합 gate, 최대 stop streak와
  endpoint/속도/자세/양발접촉 상태만 episode 단위로 누적한다.
- stack_stage_summary.py는 106열 결과에서 STACK_CLEAR_STOP_FUNNEL과
  STACK_CLEAR_STOP_STATE를 출력한다. 기존 93열 결과는
  STACK_CLEAR_STOP_DIAG unavailable=legacy_columns로 계속 읽는다.
- 두 Python 파일의 py_compile, 기존 ms70 e12000 93열 summary 호환성, 합성 106열
  summary 출력 경로를 GPU 없이 검증했다. checkpoint 재학습은 필요 없으며 새 suffix로
  동일 checkpoint를 재평가해야 한다.

## 2026-09-16

### ms70 하이브리드 CLEAR 회전 보상

- CLEAR 회전 중 yaw 오차 감소량에는 항상 계수 1.5를 주고, 남은 yaw 오차가
  90도 이하가 된 뒤에는 목표에 가까워질수록 0에서 2.0까지 증가하는 cosine
  gain을 추가한다. 새 보상은
  `STACK_CLEAR_YAW_PROGRESS=1`인 ms70에서만 활성화되어 기존 실험은 유지된다.
- RELEASE yaw 오차, CLEAR 최소 yaw 오차, retreat arc 0.1 m 최초 통과 step을
  metrics 마지막 3개 열에 추가했다. 기록은 새 보상 활성화 여부와 무관하다.
- ms70은 ms68 경로·fade·bypass 설정을 유지하고 Top wait와 Base hold 계수를
  각각 10.0으로 설정한다.

### inkyu-local ms67/ms68 서버 이식

- `inkyu-local`의 단일 CLEAR classic steering, heading-progress, 3-step carry 관측
  fade와 ms68 safety-gate bypass를 opt-in knob로 이식했다. 기존 서버 ms63/ms64의
  humanoid 기준 ±60도 안전 경로와 reward 기본값은 그대로 유지한다.
- ms67/ms68 wrapper에서는 Base box 기준 one-sided 0-60도 경로를 선택한다. ms67은
  1.2-2.0 m와 gate 유지, ms68은 2.0-3.0 m와 Base-first gate bypass를 사용한다.
- 로컬 GPU 1 하드코딩과 로컬 fallback 경로, 생성 YAML은 가져오지 않았다. 서버에서는
  기존 데이터·checkpoint 경로와 `MA_GPU=6|7`을 사용해 새 설정 파일을 생성한다.

### Base-first Top wait 생략과 CLEAR 후퇴 방향 보상

- shared-goal에서 Base가 strict CLEAR를 먼저 끝내면 Top의 safety wait 선행 조건을
  생략하고 현재 위치에서 실제 base box 위의 최종 goal로 즉시 재계획한다. Top이 먼저
  도착하면 기존처럼 safety gate에서 기다린다. 기존 sidecar는 기본값 0으로 재현된다.
- 랜덤 후퇴 거리 min/max와 후방 반각 knob를 추가했다. ms61 후속 wrapper는 env별
  2.0-3.0 m, 후방 중심 ±60도로 샘플링한다.
- CLEAR heading reward를 negative-clear 모드와 분리하고 후속 wrapper 가중치를 2.0으로
  설정했다. Python py_compile, 관련 Bash 구문 검사와 git diff --check를 통과했다.
- 총 120도 후퇴 범위에서 무작위 후보가 모두 Top box와 겹쳐 학습 전체가 종료된 문제를
  수정했다. 2도 간격의 후보 61개를 검사해 안전한 방향을 seed 기반으로 고르고, 경로가
  전혀 없는 환경만 실패로 기록해 reset한다.
- ms64부터 Top wait와 CLEAR 완료 후 Base hold reward 계수를 각각 0.50에서 1.0으로 높였다.

## 2026-09-15

### CLEAR를 기존 smooth steering + 도착 hold로 단순화

- ms60 try2의 별도 이동/방향 reward는 다음 실행부터 끈다. CLEAR는 기존
  부모 steering과 같은 속도 일치 양의항과 bounded 횡이탈 패널티를 계수 2.0으로 사용하며
  track/stall/reverse 패널티는 모두 0으로 둔다.
- 실제 1.2 m 후퇴 경로의 끝점 0.12 m 이내에 들어오고 arc가 0.8 m 이상이면 도착을
  env별로 latch한다. 그 즉시 steering 경로와 속도 명령을 0으로 바꿔 정지를 명령한다.
- 도착 뒤 CLEAR에서는 정지 reward 2.0을 매 frame 지급하고, 10-frame 정지 확인 뒤
  STACK에서도 기존 base hold reward 0.5를 계속 지급한다.
- 수정 시점에 실행 중이던 ms60 try2에는 로드된 이전 코드와 설정만 적용됐다. 실행 중인
  train_local.sh는 수정하지 않았다.
- Python py_compile, wrapper bash -n, git diff --check를 통과했다.

### ms59 후속 CLEAR 양의 경로 진행·방향 정렬 보상

- ms59의 frame당 최대 -40 경로 추종 패널티를 끄고, 후퇴 경로의 현재 접선 방향으로
  이동할 때 최대 +4를 주도록 STACK_CLEAR_MOVE_W=4.0을 적용했다.
- STACK_CLEAR_FACING_W를 추가해 humanoid 정면이 현재 경로 접선과 정렬될수록 최대
  +1을 준다. 기본값은 0이라 기존 sidecar 재생 결과는 바뀌지 않는다.
- 무작위 후퇴 경로 목표 길이는 0.8 m에서 1.2 m로, CLEAR 완료 arc는 0.6 m에서
  0.8 m로 늘렸다. Top 대기와 Base 정지 유지 reward는 각각 0.5로 높였다.
- 새 방향 보상은 train_local.sh sidecar 저장·재생 목록에 포함했다.
- Python py_compile, 관련 Bash 구문 검사, git diff --check를 통과했다. check_docs는 외부 구 경로 /home/hwanhee/CVPR2027의 기존 AGENTS.md·CLAUDE.md 링크 오류 2건으로 실패했다.

### Vulkan 숫자 selector와 물리 GPU 번호 분리

- 물리 GPU 2가 Vulkan 열거에서 빠지면서 `DRI_PRIME=7!`가 더는 물리 GPU 7을
  선택하지 못하고 8개 장치를 모두 노출해 GUI 가드가 중단되는 문제를 재현했다.
- `vulkan_gpu_guard.py`는 허용된 숫자 selector `6!`, `7!`만 시도하고 요청한
  `nvidia-smi` 물리 GPU UUID와 정확히 일치하는 한 장만 통과시켜 실제 selector를
  stdout으로 반환한다. `masteer/view.sh`도 반환값을 Isaac Gym 실행에 전달한다.
- 현재 서버에서는 GPU 7이 `6!`로 UUID 검증을 통과하고 GPU 6은 허용 selector로
  도달할 수 없어 안전하게 거부됨을 확인했다. 뷰어는 이후 CUDA 초기화까지 진행했지만,
  GPU 2의 `Unknown Error`로 신규 CUDA 컨텍스트 자체가 실패해 서버 복구가 필요하다.

## 2026-09-14

### 다양한 CLEAR 후퇴 경로와 경로 추종 패널티

- `STACK_RETREAT_RANDOM=1`에서 배치한 박스에서 멀어지는 연속 각도의 후퇴 경로를
  현재 위치 기준으로 생성한다. 다른 박스와의 기하학적 여유를 확인하고 양옆 방향을
  안전 후보로 포함한다. 리샘플링 후 실제 유효 경로 길이는 0.8 m 이상이다.
- `STACK_CLEAR_TRACK_PEN_W`는 현재 경로 접선 방향의 이동 부족과 경로 이탈 중
  큰 위반량을 음수 보상으로 차감한다. 기존 유예 및 감속·정지 구간은 제외한다.
  몸 방향을 제한하거나 회전 보상을 추가하지 않는다. 새 노브의 기본값은 0으로
  기존 저장된 실행 설정과 호환된다.
- `train_ms52_shared_goal_safe_local.sh`는 무작위 후퇴, 길이 0.8 m, 경로 패널티
  계수 40을 적용하고 기존 stall 패널티 계수는 0으로 해 중복을 피한다. 후퇴
  완료 arc 기준 0.6 m, 정지·박스 안정·Top 대기 조건과 carry 즉시 제로패딩은 유지한다.
  `train_local.sh`의 export와 재생 sidecar에도 새 노브를 포함한다.
- CPU에서 실제 경로 함수로 생성한 1,024개 경로의 최소 길이 0.8 m와 두 박스의
  기하학적 회피를 확인했다. 실제 CLEAR 보상식에서 경로 추종 성공 0, 이동 부족
  및 완전 이탈 -40, 유예·감속 구간 0을 확인했고 Python/Bash 구문 검사를 통과했다.
- 기존 ms58 e9500 정책으로 GPU 7·64환경 headless 검증을 실행해 종료 코드 0을
  확인했다. 결과 suffix는 `retreat_random40_e9500_smoke_20260914_v1`이다. 이는
  새 경로·보상 설정의 실행 검증이며 재학습이나 회전 동작 개선 검증은 아니다.

### masteer 뷰어 VNC 공유 메모리 부족 우회

- `scripts/masteer/view.sh`의 x11vnc 실행에 `-noshm`을 추가했다. 공유 메모리
  할당 오류(`shmget: No space left on device`)로 VNC만 종료되어 noVNC 연결이
  실패하는 문제를 우회한다.
- 기존 뷰어 디스플레이에서 오류를 재현하고 `-noshm` 적용 후 VNC 정상 시작과
  6100/5941 포트 수신을 확인했다. Bash 구문 검사를 통과했다.

## 2026-09-14 — ms52 shared-goal 안전 경로와 평행 STACK gate

- ms52의 340-D 관측·모델 ABI와 carry tokenizer 동결 계약을 유지한 전용 실행
  `train_ms52_shared_goal_safe_local.sh`를 추가했다. CLEAR 안정 정지 streak만
  15 frame에서 10 frame으로 완화하고, 학습 시작점은 ms18 epoch 9000을 유지한다.
- shared-goal에서 Base와 Top의 실제 box-carry arc만 비교해 0.8 m 이상 떨어지는
  safety-gate 경로쌍을 고른다. 접근 구간·경로 padding과 출발 후 0.6 m는 비교에서
  제외하며, 기준 미달 env만 최대 4회 다시 뽑는다. Top은 Base release·후퇴·안정
  정지 뒤에만 최종 적층 위치로 진입한다.
- STACK terminal success에 base/top box face-frame 평행 조건을 추가했다. 기본 호환값은
  180도이고 새 실행은 15도를 사용한다. 정육면체의 90도 quarter-turn은 허용하고
  45도 diamond 배치는 거부하므로, 조건을 통과해야 base/top 각각 +20을 한 번 받는다.
- 순수 tensor 검증에서 유효 carry arc/padding 분리와 0/15/45/90도 판정
  `pass/pass/fail/pass`를 확인했다. 대표 32개 경로 생성에서는 Base·Top pair 재탐색
  2회 안에 모두 0.8 m를 넘었다. Python/Bash 구문과 `git diff --check`를 통과했다.
  Isaac Gym class dummy 검증은 확장 모듈 segfault로 완료하지 못했으며 실제 학습은
  실행하지 않았다.

## 2026-09-14 — shared-goal W2S + late-STACK rehearsal 복구

### ms43 실행 설정에 STACK 성공 1회 보너스 +20 적용

- `scripts/masteer/train_release_continuity_local.sh`에
  `STACK_SUCCESS_BONUS=${PILOT_SUCCESS_BONUS:-20.0}`을 추가했다. 기존 완료 이벤트
  처리로 SUCCESS 전환 프레임에 base/top 각각 +20을 한 번 지급한다.
- carry tokenizer 동결, phase별 보상 및 성공 판정 조건은 유지한다. 현재 ms43
  성공 판정에는 top 손떼기가 필수가 아니며, 관측 수정과 함께 새 실행부터 적용된다.
- Bash 구문 검사와 학습 실행을 차단한 환경변수 전달 확인에서 보너스 20.0,
  carry 동결 1, 후퇴 측면각 45를 확인했다. 새 학습은 실행하지 않았다.


### 진행 중 reset 관측 캐시 오용 제거와 bootstrap 신체 상태 복원

- 단계 제어 중 선택 env 관측을 갱신하던 6곳을 실제 시뮬레이터 신체 상태로
  갱신하도록 수정했다. 일반 reset 직후 관측·AMP 초기화에 필요한 자세 캐시는 유지한다.
- bootstrap에 신체 링크 위치·회전·선속도·각속도를 함께 저장하고, 복원 시 reset
  자세 캐시도 일치시킨다. 외부 bank는 version 2로 저장하며, 신체 정보가 없는
  version 1 bank는 재수집을 요구한다. 학습 체크포인트 가중치는 변경하지 않는다.
- GPU 7 headless, ms57 현재 체크포인트, 256 env/seed 0 자연 전환 검증에서
  진행 중 신체 관측과 실제 상태의 최대 오차는 0이었다. 자연 STACK 진입은 15건,
  STACK 낙상 종료 1건, 성공 0건으로 기존 정책의 작업 성공까지 해결된 것은 아니다.
- 저장 bank 왕복·환경 원점 이동과 복원 3개 env의 첫 신체 관측·AMP 현재값 및
  이력 일치를 확인했다. Python 구문 검사와 diff check도 통과했다.
- 앞선 자연 전환 관측 소스 비교에서도 캐시 오용이 즉시 낙상의 주요 원인으로
  확인됐다. 과거 ms48의 정책 불안정성 우선 해석과 ms43 등 구 평가의 방법론 비교는
  수정된 관측으로 다시 검증해야 한다.
- 검증 원장: `runs/stack_bootstrap/production_obs_fix_20260914/verification.json`.
  기존 실행 중 프로세스에는 새 코드가 소급 적용되지 않는다.


- `bootstrap_fraction=0`의 원인은 설정값 0.15가 아니라 snapshot bank가
  비어 있던 것이었다. shared-goal의 stage gate가 이미 grasp와 box 안정
  5 frame을 요구하는데 추가 Top balance streak 8 frame까지 요구해 capture가
  한 번도 성립하지 않았다. 새 실행 스크립트에서는 중복 streak를 0으로 둔다.
- native carry rehearsal 20%와 전체 late-STACK reset 15%를 함께 맞추기 위해,
  non-carry 80% 안의 bootstrap 조건부 확률을 18.75%로 설정했다.
- snapshot은 현재 rollout의 물리 초기상태만 저장·복원한다. 복원 이후 action과
  reward는 현재 policy가 새로 생성하며, 과거 PPO `(s,a,r)` replay는 하지 않는다.
- TensorBoard에 `bootstrap_valid_fraction`과 설정·실제 reset 비율인
  `bootstrap_target_fraction`, `bootstrap_draw_probability`,
  `bootstrap_reset_fraction`, `bootstrap_eligible_draw_fraction`을 추가해
  snapshot bank 생성, reset sampling, 실제 phase 점유를 서로 분리했다.

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

## 2026-09-04

### stack-stage 평가에 release/clear/displacement/pickup 회귀 분리

`HumanoidMASequentialStackRelease`의 기존 49열 episode 지표 뒤에 base 박스가
배치 시점부터 움직인 최대 XY 거리를 추가했다. 관측·보상·action·phase 전환·reset은
바꾸지 않는다. `stack_stage_summary.py`는 release와 body-clear의 전체/조건부 성공률,
base displacement의 mean/median/p95를 출력하고, initial 평가가 있으면 역할별 strict
pickup 성공률 하락을 별도로 비교한다. 기존 49열 결과도 계속 읽을 수 있다.

### GUI viewer의 stale 물리 actor write 복원

로컬 정상본과 같은 ms18 epoch 9000 PTH를 서버에서 비교해 PTH SHA256이 동일함을 확인했다.
현재 서버 viewer는 `_update_marker()`에서 carry box와 platform을 매 렌더 프레임 stale
root-state로 다시 써 픽업을 방해하고 있었다. 물리 actor는 episode reset 직후 한 번만
동기화하고 이후에는 marker actor만 쓰도록 복원했으며, top camera 갱신 전에 root-state를
refresh하도록 맞췄다. 서버 `view.sh`도 실행 위치에서 ROOT를 계산하고 `tokenhsi_koo`를
사용하며 GPU 6/7 Vulkan UUID 검증을 통과해야 실행되도록 고쳤다. headless 학습 경로에는
영향이 없다.

### ms25 virtual-retreat adapter-only — 픽업 보존 실패로 기각

`ms25_virtualretreat_adapteronly_r50_s0_e1024`는 ms18 epoch 9000에서 시작해
기존 `internal_adapt_mlp`만 학습하고 actor 경로와 RMS를 동결했다.
CARRY/stack-release를 50:50으로 섞고 CLEAR에는 가상 박스 관측·signed retreat·loco AMP를 사용했다.

2 iteration smoke에서 adapter에만 gradient가 생기는 것은 확인했다.
GPU 7에서 1024 env × 2 agents로 본 학습을 시작했고 epoch 9100·9200 PTH를 저장했다.

epoch 9100 PTH를 viewer로 확인했을 때 기존 carry의 픽업 동작이 이미 깨지는 정성적 실패가 보였다.
이는 rehearsal과 actor 경로 동결만으로는 adapter가 만드는 최종 action 변화에서 픽업을 보존하지 못한다는 증거다.
사용자 판단으로 학습과 viewer를 중단했으며 별도 정량 평가는 수행하지 않았다.

**판정: 실패.** 이 adapter-only + reward 중심 설정은 현재 형태로 이어서 학습하거나 후속 기준선으로 사용하지 않는다.
완료 실험으로 해석하지 않고, 픽업 보존 장치가 없는 동일 설정의 재실행도 하지 않는다.

### CLEAR AMP를 carry로 복원

`origin/juan`의 구현과 대조해 CLEAR 이후 base agent의 두 carry token을 손 중심의
가상 박스로 바꾸는 방식은 그대로 유지했다. 가상 박스는 실제 박스 회전과 BPS를 쓰고,
목표는 retreat endpoint이며 CLEAR부터 SUCCESS까지 실제 박스를 다시 노출하지 않는다.

CLEAR에서 AMP task mask를 `traj`로 바꾸던 override와 실행 설정을 제거했다.
이제 부모 carry task의 AMP가 그대로 유지된다. reward 식과 계수는 변경하지 않았다.

### ms27 virtual-box + ms20 CLEAR reward + transition bonus 3.0

픽업이 깨진 ms26 실행을 중단하고, ms18 epoch 9000에서 다시 시작했다. 가상 박스,
carry AMP, CARRY/stack-release 50:50 rehearsal은 유지했다. 입력 RMS는 픽업 분포 보존을
위해 동결하되 tokenizer 동결은 해제했다(`MA_ADAPTER_ONLY=0`,
`MA_FREEZE_NEW_CARRY=0`). 기존 actor·transformer·composer와 old-carry tokenizer는
계속 동결된다.

CLEAR reward는 `STACK_CLEAR_STEER_W=0`으로 ms20 식을 그대로 사용하고,
손 떼기 전환 보너스는 `STACK_TRANSITION_BONUS=3.0`으로 설정했다.

256 env × 2 iteration smoke와 1024 env 본 실행 첫 backward 모두에서 steer/new-carry
tokenizer와 adapter gradient를 확인했다. 본 실행 태그는
`ms27_virtualbox_ms20clear_b3_tokunfreeze_s0`, GPU 7, 3000 iteration이다.

### ms28 bonus 0 재시작 + 단계별 pickup 진단

ms27 epoch 9300 viewer에서도 pickup이 깨져 실행을 중단하고 산출물은 보존했다.
픽업이 확인됐던 `ms20_stackrel_gate_bonus0_s0`에 맞춰 transition bonus와 CLEAR 진입
bonus를 모두 0으로 내렸다. 가상 박스, carry AMP, 50% rehearsal, tokenizer unfreeze와
ms20 CLEAR 식은 유지한다.

`HumanoidMASequentialStackRelease`의 episode 지표 뒤에 9열을 추가했다. 양손 OBB 접근,
10-frame strict pickup, 운반 중 10-frame break, base 배치, 손 떼기, body clear, 물리 실패,
최대 lift, rehearsal/base/top 구분을 직접 기록한다. `stack_stage_eval.sh`는 보존된 중간
checkpoint를 동일 sidecar로 평가하고 `stack_stage_summary.py`로 최초 정지 단계를 요약한다.
reward와 observation/model 구조는 바꾸지 않았다.

256 env × 2 iteration smoke에서 forward/backward, epoch 9002 저장, 49열 metrics와 요약
출력을 확인했다. 본 실행 `ms28_virtualbox_ms20bonus0_stage_s0`은 ms18 epoch 9000에서
GPU 7, 1024 env, 3000 iteration으로 시작했고 진단용 checkpoint를 50 iteration마다 남긴다.
첫 backward에서 steer/new-carry tokenizer와 adapter gradient 및 약 3.6만 step FPS를 확인했다.

### ms26~28 적층 PTH의 viewer 재검증 기준

학습 sidecar를 다시 대조한 결과 ms26·ms27·ms28은 모두 ms18 epoch 9000에서
시작한 `HumanoidMASequentialStackRelease`, `MA_TOKEN=mask` 계보다. 각 배치는
일반 CARRY rehearsal과 stack-release를 50:50으로 혼합하며, stack-release에서는
base 박스 배치·release·body clear 후 top 박스를 base 위에 놓는 전 단계를 학습한다.

PTH 경로를 `view.sh`에 직접 넘기면 학습 sidecar의 `MS_TASK`와 `STACK_*`를
자동 복원하지 않고 기본 `HumanoidMASteerCarry`로 실행된다. 따라서 일반
carry 동작 확인과 적층 정책 확인을 구분하고, 적층 확인 전에는 반드시
`runs/queue/logs/<tag>.env`를 source한다. 또한 stale 물리 actor write가 수정되기
전 viewer로 내린 ms26·ms27 pickup 정성 판정은 정책 회귀의 단독 근거로 쓰지
않고, 수정된 viewer와 학습 sidecar로 재확인한다.

### ms29 손 거리 전환 + 발-박스 penalty 비교

모델/head/tokenizer 구조는 바꾸지 않고 ms18 epoch 9000에서 시작한다. 실제 pickup 이력 뒤
Agent A의 양손이 박스 표면에서 0.10m 이상인 상태를 5 step 연속 만족하면 CLEAR로 전환한다.
box 위치·속도 안정화와 CLEAR→STACK hard gate는 끄고, Agent B 시작에는 A의 body-base-box
거리 0.60m만 유지한다. teammate token은 mask, new-carry와 steering tokenizer 및 adapter는
학습한다. release 후 기존 XY 이탈 실패 조건도 비활성화했다.

공통 CLEAR reward는 유지하고 한 쌍만 발 원점과 base-box OBB 표면 거리 0.20m 이내에
최대 0.25 penalty를 추가했다. 2048 env는 PhysX illegal memory access로 초기화 실패했고,
1024 env smoke는 두 iteration·checkpoint·gradient를 통과했다.

GPU 6에 1024 env, seed 0, 3000 iteration으로 두 실험을 시작했다.
- `ms29_hand10_clear06_base_e1024_s0`: foot penalty 0
- `ms29_hand10_clear06_foot025_e1024_s0`: foot penalty 0.25


## 2026-09-05

### ms30/ms31 all-feet placement gate + CLEAR motion-gate 비교

ms18 epoch 9000에서 동일하게 시작하는 1024-env, seed 0, +3000 iteration 두 런을
GPU 6에 배치했다. 공통으로 ms24의 안정 배치 후 RELEASE 순서를 복원하고 손 이격 0.15m,
body clear 0.60m, drop XY 100m를 사용한다. 목표 근처 CARRY 구간에서는 base box와 두
에이전트의 네 발 사이 최소 OBB 표면 거리에 최대 0.25 penalty를 주고, 0.20m 이격이
배치와 손 떼기 동안 유지되어야 다음 phase로 진행한다.

CLEAR→STACK gate는 CARRY/최종 적층 기준과 분리해 XY 0.15m, Z 0.10m, 선속도 0.15m/s,
각속도 0.40rad/s로 완화했다. 두 런의 유일한 비교축은 CLEAR의 support/stable 정적 보상을
기존대로 유지하는지, 후퇴 motion으로 gate하는지다.

- `ms30_hand15_clear06_allfoot_base_e1024_s0_try2`: 기존 CLEAR 보상
- `ms31_hand15_clear06_allfoot_motion_e1024_s0_try2`: CLEAR 정적 항을 motion-gate

`MS_SAVE_LATEST=100`은 기존 `Humanoid.pth` 갱신으로 유지하고,
`MS_SAVE_ARCHIVE=1000`을 학습 루프에 연결했다. 각 런은 절대 epoch
10000/11000/12000을 별도 저장하므로 종료 시 latest 1개와 archive 3개가 남는다.

64-env, 2-iteration smoke에서 task forward/backward와 종료를 확인했다. 본 런 둘 다
새 설정을 sidecar에서 재현했고, 첫 backward에서 active tokenizer와 adapt MLP gradient,
약 1.1~1.4만 total FPS를 확인했다. 최초 nohup-only 태그 두 개는 session 분리 전에
종료되어 학습하지 않았으며, 출력 보존 규칙 때문에 `try2` 새 태그로 재시작했다.


## 2026-09-06

### stack phase gate를 predicate별로 분해

`HumanoidMASequentialStackRelease`의 기존 50열 episode 지표 뒤에 22열을 append했다.
정책 입력·보상·action·phase 전환 조건은 바꾸지 않고 다음 값만 누적한다.

- CARRY: XY/Z/선속도/각속도/upright/foot/placeable/결합 gate의 step 통과율과
  결합 gate 최대 연속 프레임
- RELEASE: hand/foot/결합 gate 통과율과 최대 연속 프레임
- CLEAR: hand/body-distance/base-stable/결합 gate 통과율, 최대 body distance와
  실제 retreat path arc

`stack_stage_summary.py`는 기존 49·50열 결과를 계속 읽고, 새 72열 결과에서는 전체와
agent index 0/1을 분리해 출력한다. release→clear event의 프레임 지연도 함께 출력해,
이미 body-distance gate 밖에서 CLEAR에 진입한 즉시 통과와 실제 후퇴를 구분한다.

문법 검사와 기존 50열 결과 하위 호환을 확인한 뒤 GPU 6에서 ms30 64-env smoke를 실행했다.
72열 저장과 요약 출력을 확인했으며 reward/action 수치는 변경하지 않았다.

### ms30/ms31과 ms18-init ms24 두 정책의 gate 진단

동일한 최신-checkpoint, 512-env 평가를 `gate_diag_0906` suffix로 실행했다.

| tag | base n | place→release→clear | CARRY joint steps | RELEASE hand/foot/joint | CLEAR body | retreat arc p50/p95/max |
|---|---:|---:|---:|---:|---:|---:|
| ms30 allfoot base | 1132 | 11→7→3 | 86/616866 | .457/.055/.044 | .003 | .004/.139/.173 |
| ms31 allfoot motion | 1140 | 13→8→5 | 101/619410 | .430/.079/.041 | .017 | .032/.211/.271 |
| ms24 possteer .25 | 1160 | 66→62→0 | 435/607084 | .311/1/.311 | .000 | .025/.150/.258 |
| ms24 possteer .50 | 1159 | 47→44→1 | 297/613416 | .133/1/.133 | .004 | .013/.134/1.000 |

ms30/31은 안정 배치만 보면 step 통과율이 .218/.231이고 all-foot도 .571/.559지만,
둘이 같은 순간 겹친 비율은 약 0.00014/0.00016뿐이었다. 5-frame CARRY→RELEASE
gate가 첫 병목이다. RELEASE에 들어간 뒤에는 hand보다 all-foot가 훨씬 낮아 두 번째
병목도 foot hard gate다. 따라서 viewer에서 손을 뗐는데 phase가 바뀌지 않는 관찰과
정량 결과가 일치한다.

네 정책 모두 CLEAR의 실제 path 후퇴는 거의 없었다. ms24 possteer .50에서 한 episode만
arc 1.0m를 진행해 viewer에서 보였을 후보가 있으나, 그 episode는 body gate 통과 step
.080, base-stable .264, 결합 0이었고 base box가 0.256m 움직였다. 즉 실제로 물러났지만
박스 안정성과 body-distance가 동시에 성립하지 않아 STACK으로 전환되지 않았다.
유일한 ms24 .50 clear event도 release와 같은 frame에 발생해 학습된 후퇴 성공이 아니다.

### 다음 학습 순서 제안

1. CARRY→RELEASE에서 all-foot를 hard conjunction으로 쓰지 않고 안정 배치 5-frame을
   먼저 latch한다. foot distance는 dense penalty로 유지한다.
2. RELEASE→CLEAR도 손 떼기와 foot safety를 분리한다. 손 떼기를 phase 전환으로 쓰고,
   foot는 penalty 또는 별도 safety 판정으로 둬 서로 다른 시점의 조건을 강제로 겹치지 않는다.
3. CLEAR 성공은 현재 body-distance 대신 `retreat_arc >= 0.6m`를 필수로 해 이미 멀리
   서 있던 상태의 즉시 통과를 막고, base displacement/velocity를 dense penalty로 준다.
4. 첫 pilot은 후퇴 outlier가 실제로 나온 ms24 possteer .50에서 시작하되 CARRY 경로는
   동결하고 CLEAR에서만 활성인 residual을 쓴다. pickup 보호지표 0.90과 base displacement를
   함께 본다. 이 pilot이 실패할 때만 virtual-retreat 관측을 다시 검토한다.

위 제안은 아직 학습이나 큐 행으로 만들지 않았다.


### 모델 변경 없는 reward-only CLEAR pilot 구현

CLEAR 전용 residual은 모델 forward와 phase 입력을 바꿔야 하므로 사용하지 않았다.
`amp_network_builder_transformer_adapt.py`에는 변경이 없고, 기존 ms24
`possteer050 ... try3` 체크포인트와 내부 adapter 구조를 그대로 이어 쓴다. 대신
학습 환경의 25%를 기존 carry rehearsal로 유지해 새 CLEAR gradient에 의한 pickup/carry
붕괴를 완화한다. 이는 CARRY action 불변을 수학적으로 보장하지 않으므로 아래
100-iteration 단위 보호지표 평가가 필수다.

환경/controller와 reward에는 다음 opt-in 노브를 추가했다. 기본값은 기존 sidecar 동작을
보존하고, `train_clear_reward_pilot_local.sh`에서만 새 설계를 켠다.

- `STACK_ENTRY_FOOT_GATE=0`: CARRY→RELEASE는 안정 배치 5프레임만 latch한다.
- `STACK_RELEASE_FOOT_GATE=0`: RELEASE→CLEAR는 손 간격 5프레임만 사용한다.
- `STACK_CARRY_FOOT_GATE=1`, `STACK_FOOT_BOX_W=0.10`: all-foot는 hard gate가
  아니라 dense penalty와 73번째 episode 안전지표로 유지한다.
- `STACK_CLEAR_ARC_DIST=0.60`: CLEAR 진행 보상과 CLEAR→STACK 전환 모두 실제
  retreat path arc를 사용한다.
- CLEAR에서 latched base box의 XY 변위/선속도/각속도를 각각
  `0.10/0.05/0.05` 가중치로 정규화해 감점한다.
- `STACK_CLEAR_HARD_GATE=0`: 안정성은 hard conjunction 대신 위 dense penalty와
  별도 지표로 본다.
- `STACK_REHEARSAL_FRAC=0.25`: 나머지 75%만 새 순차 phase/reward를 경험한다.

파일럿은 500 iteration이며 archive를 100마다 저장한다. 평가는
`eval_clear_reward_pilot.sh`가 초기 ms24 기준선과 100·200·300·400·500 구간을
차례대로 평가한다. 보호 하한은 pickup 0.85, `release_given_place` 0.80,
release 이후 base displacement p95 0.15m이며, CLEAR foot-clear step 비율은 초기보다
최대 0.05p 하락까지 허용한다. 보호조건과 별도로 `retreat_arc >= 0.6m` episode
비율이 초기보다 증가했는지 `learning=UP`으로 출력한다. FAIL이어도 평가를 중단하거나
프로세스를 종료하지 않고 다섯 체크포인트를 모두 본다.

구현·문법·72열 하위 호환·73열 합성 판정만 확인했다. 학습과 GPU 평가는 실행하지 않았다.

## 2026-09-07

### legacy delivered gate와 동적 obj 적층 목표 추가

기존 안정 배치 결합 조건 대신 legacy carry 성공 이벤트인 `_ep_finish >= 0`을
CARRY→RELEASE 기준으로 선택할 수 있도록 `STACK_ENTRY_DELIVERED`를 추가했다.
기본값은 0이므로 기존 sidecar와 학습은 변하지 않는다. 전용 wrapper는 1 frame을 요청하지만 현재 try2의 실제 sidecar 값은 아래처럼 5 frame이다.

B의 적층 carry 목표는 A가 놓은 실제 아래 박스의 현재 XY와 윗면 Z를 매 step 사용한다.
기존에도 teammate token은 상대 박스의 현재 위치·속도를 관측했고 목표도 0.10 lerp로
추종했지만, 이제 lerp 지연 없이 실제 obj 위치를 즉시 반영한다.

`train_delivered_steer_local.sh`는 ms18 e9000에서 시작하며 가상 후퇴 박스를 끄고,
기존 steering 경로로 CLEAR 후퇴를 학습하는 500 iteration·25% carry rehearsal
파일럿을 실행한다. Python/Bash 문법과 diff whitespace만 확인했고 학습·GPU 평가는
실행하지 않았다.

### ms34 delivered-steer try2의 실제 RELEASE/CLEAR 조건과 reward

아래는 GPU 7에서 실행한 `ms34_ms18init_delivered_steer_s0_try2`의 생성 sidecar
기준이다. 전체 env의 25%는 carry rehearsal이고, 나머지 75%가 순차 phase를 사용한다.
가상 후퇴 박스는 꺼져 있으며(`STACK_VIRTUAL_RETREAT_BOX=0`) 기존 steering 창을 쓴다.

#### phase 전환

1. CARRY→RELEASE: base agent의 원래 carry 성공 latch `_ep_finish >= 0`만 사용한다.
   안정 배치 XY/Z/속도/upright와 foot 조건은 진단값일 뿐 hard gate가 아니다. 다만 현재
   try2 sidecar는 `STACK_ENTRY_STEPS=5`여서 이 값이 5 frame 연속 유지돼야 한다. 전용
   wrapper의 1 frame 설정이 기준 ms24 env source에 덮인 결과다.
2. RELEASE→CLEAR: base box와 양손 표면 사이 최소거리 `hand_dist >= 0.15m`를 5 frame
   연속 만족하면 전환한다. `STACK_RELEASE_FOOT_GATE=0`이므로 발 거리는 전환 조건이 아니다.
   이때 실제 base box에서 agent 반대 방향으로 1.5m retreat steering path를 생성하고 속도
   scale은 0.5로 둔다.
3. CLEAR→STACK: 손 간격 0.15m를 계속 유지하면서 실제 steering path 누적 진행거리
   `arc_root >= 0.60m`를 만족하면 전환한다. `STACK_CLEAR_HARD_GATE=0`이므로 base box의
   위치·속도 안정성 및 foot 거리는 hard gate가 아니다. STACK 진입 후 B의 carry goal은
   매 step 실제 A box의 현재 XY와 윗면 Z로 즉시 갱신된다.
4. RELEASE 이후 base box가 최초 고정 goal에서 XY 0.25m 초과 이동하거나 z가 -0.05m
   아래로 떨어지면 FAILED로 종료하고 base agent reward를 -1로 둔다.

#### reward

모든 phase는 먼저 ms18 carry reward를 계산한다. base agent는 RELEASE 이후 아래 reward로
교체된다. top agent와 rehearsal env에는 post-phase reward 교체가 적용되지 않는다. target 근처에서
박스를 집은 CARRY base와 RELEASE 이후에는 box-foot 거리 0.20m 안쪽에 최대 0.10의
dense penalty가 추가된다. 이는 hard gate가 아니다.

- 공통값: `h=clip(hand_dist/0.15,0,1)`,
  `support=exp(-20*xy_err^2-80*z_err^2)*upright`,
  `stable=exp(-10*||base_lin_vel||^2-0.5*||base_ang_vel||^2)`.
- RELEASE: `0.50*h + I(hand_clear)*(0.30*support + 0.20*stable) - 0.10*clip(1-foot_dist/0.20,0,1)`.
  즉 손을 멀리할수록 보상하고, 손을 0.15m 이상 뗀 동안에만 놓인 박스의 지지·정지를
  추가 보상한다.
- CLEAR: steering 진행량과 속도 일치를 `motion=0.7*signed_progress + 0.3*speed_match`로
  계산한다. 역방향 진행은 `signed_progress` 때문에 음수이고, path 횡오차는
  `path_quality=exp(-lat_error^2)`로 감점된다. 실제 reward는
  `I(hand_clear)*[motion*(0.40*support+0.25*stable)+0.35*clip(arc/0.60)]`에
  `0.50*I(hand_clear)*path_quality*motion`을 더한 뒤 base box penalty와 foot penalty를 뺀다.
  base box penalty 최대값은 위치변위 0.10, 선속도 0.05, 각속도 0.05로 총 0.20이다.
- CARRY→RELEASE 후 첫 CLEAR reward에는 transition bonus +3.0, CLEAR→STACK 후에는
  clear bonus +1.5, SUCCESS phase에는 매 step +0.5가 추가된다.

따라서 현재 설계의 RELEASE 핵심은 손 떼기이고, CLEAR 핵심은 손을 뗀 채 기존 steering
경로로 0.60m 실제 이동하는 것이다. 박스 안정성은 두 phase 모두 reward/penalty로만
유도하며 phase 전환을 막지는 않는다.

### ms35 delivered+lowered 진입과 RELEASE carry-reward bridge

기존 모델 구조와 중앙 phase controller는 유지했다. 새 opt-in `STACK_ENTRY_LOWERED=1`은
legacy carry 완료 latch `_ep_finish >= 0`에 `z_err <= STACK_Z_TOL`을 결합하며, ms35
wrapper는 이를 2 frame 유지한 뒤 CARRY에서 RELEASE로 전환한다.

`STACK_RELEASE_CARRY_BRIDGE=1`에서는 RELEASE 안에서만
`(1-h)*carry_r + h*(0.50 + 0.30*support + 0.20*stable)`을 사용한다. CARRY reward는
그대로이며, 손이 붙은 RELEASE 시작점은 현재 carry reward와 같고 손 거리 `h`가 증가하면
기존 hand-clear 완료 reward로 연속 전환된다. 두 옵션의 기본값은 0이라 기존 sidecar 동작은
변하지 않는다. pilot wrapper의 entry 설정이 기준 sidecar에 덮이지 않도록 `PILOT_*`
override를 source 뒤에 적용했다.

Python `py_compile`과 세 Bash 파일의 `bash -n`을 통과했다.
`ms35_ms18init_deliveredz2_releasebridge_s0`를 ms18 epoch 9000에서 GPU 7, 1024 env,
500 iteration으로 시작했다. 첫 backward에서 trainable `adapt_mlp` 4/4에 gradient가 있고
학습 step FPS가 출력되는 것을 확인했다.

#### 남아 있는 독립 병목: CLEAR 후퇴

ms35는 CARRY→RELEASE 진입과 RELEASE 시작 reward 연결만 수정한다. 기존 최신-checkpoint
512-env 진단에서 possteer025는 place→release가 58→57이었지만 실제 `retreat_arc >= 0.6m`는
0/57, possteer050은 45→43 뒤 0/43이었다. arc p50/p95/max도 각각
`0.020/0.082/0.453m`, `0.003/0.116/0.447m`라 RELEASE→CLEAR 전환 뒤 후퇴 실행이
별도의 두 번째 병목이다. 기록된 clear 각 1건은 delay 0 frame이며 실제 후퇴 성공으로 보지 않는다.

따라서 ms35의 판정은 (1) delivered+lowered 진입률, (2) release_given_place,
(3) CLEAR의 `retreat_arc >= 0.6m`를 순서대로 분리한다. 앞 두 항이 개선돼도 세 번째가
0이면 이번 reward bridge는 손 떼기까지만 해결한 것으로 판정한다. 현재 epoch 9100
체크포인트가 생성됐고 학습 프로세스는 계속 실행 중이다.

### ms36 TokenHSI식 순차 reward mask

ms35 512-env eval에서 ms18 initial 대비 epoch 9400의 place는 0.730→0.696으로 비슷했지만,
release_given_place는 0.391→0.216, RELEASE hand-clear step 비율은 0.016→0.007로 감소했다.
retreat arc p95도 0.116m→0.059m로 줄어 live carry reward를 손 이격에 따라 보간한 bridge가
손을 계속 붙잡는 방향으로 학습된 것으로 판정했다.

모델 구조와 중앙 controller는 유지하고 opt-in STACK_SEQUENTIAL_REWARD_MASK를 추가했다.
CARRY는 기존 reward를 그대로 사용하고, RELEASE는 carry_done+r_release, CLEAR는
carry_done+release_done+r_clear, STACK 이후는 carry_done+release_done+clear_done+r_stack을
사용한다. 완료 reward 기본값은 현재 reward 상한에 맞춘 1.6/1.0/(1.0+clear_steer_w)이며
새 ms36 wrapper는 1.6/1.0/1.5를 명시한다. 완료 reward는 손 거리와 무관한 상수라 CARRY
값을 보전하면서도 손 접촉 유인을 만들지 않는다. bridge와 reward mask의 동시 활성화는 거부한다.

기존 controller의 관측 전환도 그대로다. 배치 완료 후 RELEASE에서는 steer window를 0으로
만들고, 손 이격 5 frame이 확인된 뒤에만 CLEAR 후퇴 경로와 clear reward를 활성화한다.
전용 RELEASE token이나 observation/model shape 변경은 없다. 새 옵션들은 train sidecar에
저장·재생되며 scripts/masteer/train_sequential_reward_mask_local.sh는 ms18 epoch 9000,
delivered+lowered 2-frame 진입, bridge OFF, reward mask ON 설정을 준비한다.
Python py_compile, 관련 Bash 4개 bash -n, git diff --check를 통과했다.
### ms37 후측방 135도 CLEAR 경로
순수 후퇴 대신 상자 진행방향 기준 135도 후측방으로 비키도록 opt-in
STACK_RETREAT_SIDE_DEG를 추가했다. 값은 기존 후방 벡터에서 측면으로 회전하는 각도라
0도는 legacy 순수 후퇴, 45도는 진행방향 기준 135도, 90도는 순수 측면이다.
좌우 후보 중 top agent staging 위치와 반대인 쪽을 중앙 controller가 env별로 선택한다.
손 이격 5 frame 전에는 정지 steer를 유지하며, 확인 후에만 후측방 경로를 활성화한다.
요청에 따라 ms36 학습과 전용 로그 watcher를 종료했다. 종료 시 metrics는 56 episode였고
checkpoint 생성 전이라 비교 결과로 사용하지 않는다. 산출물은 삭제하지 않았다.
Python py_compile, 관련 Bash bash -n, git diff --check를 통과했다.
ms37_ms18init_seqrewardmask_side135_s0를 ms18 epoch 9000에서 GPU 7, 1024 env,
3000 iteration(최종 epoch 12000)으로 nohup+setsid 분리 실행했다.
첫 ms37 launch는 PhysX root-state tensor 초기화에서 CUDA global/shared-address 오류로
종료됐다. 후측방 controller가 호출되기 전의 GPU 초기화 오류이며 OOM은 아니었다.
같은 설정을 새 태그 ms37_ms18init_seqrewardmask_side135_s0_try2로 재시작했고,
PhysX 초기화와 첫 backward를 통과했다. adapt_mlp 4/4 gradient 및 학습 FPS를 확인했다.

### ms38 중앙 phase 기반 dynamic carry-token mask

TokenHSI long-horizon의 FSM task-token masking을 현재 중앙 controller에 맞춰
STACK_DYNAMIC_CARRY_MASK opt-in으로 구현했다. 별도 phase token이나 layer를 추가하지
않고 observation 340-D와 checkpoint parameter shape를 그대로 유지한다. 중앙 controller가
base agent를 CLEAR로 전환하면 CLEAR부터 SUCCESS까지 기존 두 carry observation window를
정확히 0으로 padding하고, adapt network가 raw zero window를 확인해 대응하는
new_carry와 old_carry attention key/value를 mask한다. top agent, carry rehearsal,
CARRY와 RELEASE phase의 관측 및 mask는 바뀌지 않는다.

현재 train cfg는 use_prior_knowledge=False여서 old_carry는 원래 비활성이므로, 실제
정책 차이는 post-CLEAR base agent의 new_carry를 끄고 [weight, self, steer]만 남기는
것이다. STACK_VIRTUAL_RETREAT_BOX와의 동시 사용은 서로 다른 CLEAR carry 표현이
겹치지 않도록 거부한다. 새 knob는 train sidecar에 저장·재생된다.

train_dynamic_carry_mask_from_ms18_local.sh는 ms37의 controller/reward/135도 후측방
설정을 그대로 source하되 정책은 요청대로 ms18 epoch 9000에서 시작한다. 태그
ms38_ms18init_seqrewardmask_side135_dynmask_s0를 GPU 7, 1024 env, 3000 iteration으로
분리 실행했으며 최종 epoch은 12000이다. 로그에서 ms18 checkpoint 로드, dynamic mask
활성, observation RMS (340,), 첫 backward의 adapt_mlp gradient 4/4와
fps step 31090.4를 확인했다. Python py_compile, 관련 Bash bash -n,
git diff --check를 통과했다.

### ms39 가상 박스 없는 steering-only CLEAR 500-iteration pilot

ms18 epoch 9000에서 시작해 ms38과 동일한 sequential reward, 135도 후측방 controller,
CLEAR 이후 dynamic carry-token mask를 유지하는 500-iteration wrapper를 추가했다. 가상
박스는 사용하지 않는다. ms38은 MA_ADAPTER_ONLY=1이라 steering tokenizer까지 동결한
채 adapter만 학습했으므로, 이번 비교에서는 pretrained backbone/composer와 new-carry
tokenizer는 계속 동결하고 steering extra tokenizer와 기존 internal_adapt_mlp만
학습 가능하게 한다(MA_ADAPTER_ONLY=0, MA_FREEZE_NEW_CARRY=1,
MA_NOFREEZE=0). 25% carry rehearsal과 나머지 task/reward 설정은 ms38 sidecar를 그대로
재생한다. 실제 base 박스를 발로 차는 행동은 기존 all-foot OBB 표면거리 penalty를
사용하되, 0.20m 안쪽 최대 감점을 ms38의 0.10에서 0.50으로 강화한다. 첫 backward의
tokenizer별 gradient로 실제 trainable 경로를 판정한다.

### ms41 reward-only CLEAR, zero observation + no carry-token mask

모델 구조와 observation 340-D shape는 바꾸지 않았다. `STACK_ZERO_CARRY_OBS=1`은
비-rehearsal base agent가 CLEAR부터 SUCCESS일 때 new/old carry observation window 두 개를
0으로 만들지만, `STACK_DYNAMIC_CARRY_MASK=0`이므로 두 carry token 위치는 frozen
Transformer attention에 그대로 남는다. 이는 ms38/ms39의 post-CLEAR carry-token attention
mask를 제거한 것이다. ms18에서 이어진 teammate token mask(`MA_TOKEN=mask`)는 별개의
기존 설정으로 유지한다.

`STACK_NEGATIVE_CLEAR_REWARD=1`은 arc ratchet 대신 실제 root XY velocity를 명령된
후측방 retreat 방향에 투영한 `v_along`을 사용한다. CLEAR 진입 뒤 두 손이 다시 0.15m
안쪽으로 들어오면 positive move reward를 0으로 만들고 즉시 -0.5 recontact penalty를 준다.
5-step grace 뒤 `v_along < 0.2*v_command`이면 최대 -0.5 stall penalty, 반대 방향이면
추가로 최대 -0.5 reverse penalty를 준다. 손이 clear이고 올바른 방향으로 이동할 때만
`path_quality*clip(v_along/v_command,0,1)`에 최대 +1.0을 준다. 기존 base 안정성 penalty와
0.10 foot-box penalty도 유지한다. 누적 완료 상수를 매 step 지급하던
`STACK_SEQUENTIAL_REWARD_MASK`는 끈다.

TensorBoard에는 `stack_reward/{clear_total,move_positive,hand_penalty,stall_penalty,
reverse_penalty,base_penalty,foot_penalty,transition_bonus}`, `stack_state/{hand_factor,
hand_clear_rate,recontact_rate,move_ratio,move_ok_rate,stall_rate,reverse_rate,v_along,
v_command,retreat_arc}`, `stack_phase/{release_fraction,clear_fraction,stack_fraction,
failed_fraction}`을 기록한다. phase 밖의 NaN은 observer가 제외해 해당 phase 표본만 평균낸다.

`tokenhsi_koo`에서 64-env 30-iteration 스모크를 완료했다. CLEAR scalar 18개가 기록됐고
마지막 값은 clear_fraction 0.0435, clear_total -0.3440, stall_rate 0.7978,
reverse_rate 0.5169였다. 따라서 현재 병목이 재접촉보다 정지와 역방향임을 분리해 볼 수 있다.
첫 backward에서 transformer 0/48, self/new-carry/old-carry/composer gradient가 모두 0이고,
steering tokenizer와 internal adapter에만 nonzero gradient가 있음을 확인했다.

`ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_s0`를 ms18 epoch 9000에서
GPU 7, 1024 env, 500 iteration(최종 epoch 9500), carry rehearsal 50%로 시작했다.
가상환경은 `tokenhsi_koo`, 분리 실행 PID는 3022738이다.

#### 3000-iteration 재실행 정정

최초 PID 3022738 실행은 1024 env였지만 500 iteration(`max_iterations=9500`)으로
잘못 시작한 것을 사용자 확인으로 발견했다. 승인 후 해당 session에 TERM을 보내 정상
종료했으며 생성된 로그·metrics 등 산출물은 삭제하지 않았다. 래퍼 기본값을 3000
iteration으로 수정하고 새 태그
`ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_3000_s0`를 ms18 epoch 9000에서
GPU 7, 1024 env, `tokenhsi_koo`, `max_iterations=12000`으로 재실행했다. 새 PID는
3041555이다. 첫 backward에서 Transformer 0/48과 frozen carry 경로를 재확인했고
steering tokenizer와 adapter에만 nonzero gradient가 있으며 첫 학습 FPS도 확인했다.

### ms42 completed-CARRY 0.50 reward continuity pilot

`humanoid_ma_sequential_stack_release.py`에 비-rehearsal base CARRY pickup 이후만 집계하는
TensorBoard 진단 7개를 추가했다. native carry reward, foot penalty, target distance,
grasp 상태, latched break, near-goal 미완료, path fraction이며 reward 값은 바꾸지 않는다.

`train_negative_clear_continuity_local.sh`를 추가했다. ms41의 freeze, zero carry observation,
attention mask OFF, negative CLEAR, rehearsal 0.50, foot penalty 0.10을 유지하고
`STACK_SEQUENTIAL_REWARD_MASK=1`, carry/release/clear 완료값 `0.50/0/0`만 적용한다.

첫 64-env smoke는 minibatch 2048이 AMP minibatch 4096보다 작아 학습 전 assert로 끝났고
산출물은 보존했다. 새 태그 smoke2는 minibatch 4096으로 2 iteration을 정상 완료했다.
Python compile, Bash syntax, diff check와 scalar 7개 기록을 확인했다. 본 학습은 GPU 7,
1024 env, ms18 epoch 9000에서 500 iteration을 정상 완료해 e9100~e9500을 보존했다.
첫 backward에서 steering tokenizer와 adapter만 gradient가 있고 frozen carry/Transformer/
composer는 0이었다.

512-env stage 평가에서 initial→e9500 base delivery는 0.716→0.652, break는
0.044→0.043으로 CARRY를 보존했다. `clear_given_release`는 0.024→0.492, 전체 CLEAR는
0.007→0.086으로 증가했다. 반면 `release_given_place`는 0.408→0.272로 감소해 다음
병목은 RELEASE의 5-frame hand-clear 전환으로 판정했다. TensorBoard에서도 +100→+500
동안 move OK 0.149→0.489, reverse 0.502→0.190, retreat arc 0.067→0.275 m로
negative CLEAR reward의 후퇴 학습 효과를 확인했다. 상세 수치는 `ANALYSIS.md`에 기록했다.

### ms43 RELEASE→CLEAR reward continuity + RELEASE diagnostics

`humanoid_ma_sequential_stack_release.py`에 RELEASE phase 조건부 TensorBoard scalar 12개를
추가했다. 실제 total/local reward와 hand/support/foot 기여, hand-clear factor/rate,
5-frame streak, support/stability/foot distance를 분리하며 reward 동작은 바꾸지 않는다.

`train_release_continuity_local.sh`를 추가했다. ms42의 구조·freeze·관측·mask 설정은
그대로 두고 carry/release/clear 완료값을 `0.50/1.00/0.00`, CLEAR stall penalty를
`1.50`으로 설정한다. CLEAR grace의 1.50 floor를 RELEASE 경계와 맞추고 grace 뒤 정지
상태는 -1.50 stall로 상쇄하는 설계다.

Python compile, Bash syntax와 diff check를 통과했다. 64-env 2-iteration smoke는 정상
완료했으나 RELEASE 표본이 없어, 30-iteration smoke로 scalar 12개가 각각 26회 기록되는
것을 확인했다. 본 학습
`ms43_ms18init_negclear_carry050_release100_stall150_zeroobs_nomask_3000_s0`를 GPU 7,
1024 env, ms18 epoch 9000에서 3000 iteration으로 분리 실행했다(PID 3840815). sidecar
설정, 첫 backward의 steering tokenizer/adapt_mlp gradient, frozen carry/Transformer/
composer의 zero gradient, rollout FPS와 metrics 저장을 확인했다.

## 2026-09-09

### ms46 stack-first bootstrap — 곡선 CLEAR와 실제 top release 보상

사용자 우선순위를 정밀 적재보다 실제 end-to-end 적재 발생으로 바꿨다. 모델·관측 shape는
유지하고 `HumanoidMASequentialStackRelease`에 기본 비활성 옵션만 추가했다.

- `STACK_CLEAR_ROUTE_AROUND=1`: base box에서 직선으로 뒷걸음치지 않고, 대기 중인 top
  carrier 반대편의 tangent waypoint를 거쳐 박스 옆으로 도는 steering path를 만든다.
- `STACK_CLEAR_STOP_ON_STACK=1`: 0.60 m CLEAR gate 통과 즉시 base의 남은 path를 현재
  위치 hold로 바꿔 STACK 중 계속 후퇴하다 넘어지는 controller mismatch를 제거한다.
- `STACK_TOP_SCALE`, `STACK_ABOVE_BONUS`: top box 접근은 느리게 하고 박스 위 영역에
  처음 들어온 사건을 한 번만 보상한다.
- top hand-separation progress/hold penalty와 별도 완화 tolerance를 추가했다. bootstrap
  SUCCESS는 두 손이 모두 떨어지고 위치·속도·자세 gate를 5 frame 유지해야 하며, 성공
  순간 두 agent에 `+20`을 한 번 지급한다.
- `train_stack_first_curved_clear_local.sh`: ms18 epoch 9000 초기화, 1024 env, 1000
  iteration, carry rehearsal 0.65, 작은 box 접촉은 soft penalty로 두는 ms46 pilot이다.

검증: Python/Bash syntax 통과. 256 env × 30 iteration smoke가 rc=0으로 끝났고 CLEAR
43 episode와 CLEAR→STACK 1 episode에서 새 controller 분기가 실행됐다. 첫 backward에서
steering tokenizer와 internal adapter gradient는 nonzero, frozen carry/Transformer/
composer gradient는 zero였다. 최종 성공 0/159는 짧은 smoke라 효과 판정에는 쓰지 않는다.

### ms45 phase-isolated RELEASE shaping + strict final success bonus

ms44의 고정 체크포인트 평가는 base place를 약 70--72%로 보존했지만
`release_given_place`가 initial 0.433에서 epoch 9500의 0.234로 하락했다.
학습 scalar도 RELEASE hand-clear가 0.0845에서 0.0026으로 감소하는 동안
release total reward는 증가해, live native-CARRY bridge가 안정적으로 계속 잡는
해법을 보상한 것으로 판정했다. 사용자 승인에 따라 ms44 학습만 종료했다.

모델/observation/controller shape은 바꾸지 않고 기본값이 0인 다음 reward knob를
추가했다.

- `STACK_RELEASE_PROGRESS_W`: RELEASE에서 signed `h_t-h_(t-1)` 보상. 손을 더 떼면
  양수이고 재접촉하면 음수다.
- `STACK_RELEASE_HOLD_PEN_W`, `STACK_RELEASE_HOLD_GRACE_STEPS`: grace 이후
  `(1-h)`에 비례해 계속 잡는 상태를 벌점 처리한다.
- `STACK_SUCCESS_BONUS`: top box가 목표 위치/높이/속도/자세 gate를
  `STACK_TOP_STEPS=20` frame 연속 통과해 SUCCESS로 바뀌는 순간, 두 agent 모두에게
  한 번만 주는 최종 stacking bonus다.

`STACK_RELEASE_CARRY_BRIDGE`와 RELEASE progress/hold shaping의 동시 사용은 거부한다.
새 평가 metrics의 마지막 열에는 strict final success step을 추가했고, 기존 73열 결과도
`stack_stage_summary.py`가 계속 읽는다.

`scripts/masteer/train_release_progress_success_local.sh`는 ms18 epoch 9000에서 시작해
live carry bridge를 끄고 progress/hold를 각각 0.50, hold grace를 5 frame,
strict final success bonus를 +10으로 설정한다. 그 밖의 1024-env/3000-iteration,
50% rehearsal, freeze, observation, CLEAR stall 설정은 ms44와 동일하다.

### ms44 CARRY->RELEASE reward continuity bridge

ms43의 3000-iteration stage 평가에서 `release_given_place`는 0.966까지
올랐지만 base PLACE는 0.574(+500)에서 0.426(+3000)으로 감소했다. 같은 구간의
grasp break 증가는 작고 `near_goal_not_delivered`가 0.113에서 0.265로 늘어,
남은 병목을 pickup/carry 붕괴보다 CARRY->RELEASE 경계의 보상 불연속으로 판정했다.

기존 `STACK_RELEASE_CARRY_BRIDGE`를 sequential completion reward와 함께 쓸 수
있게 확장했다. bridge가 켜지면 RELEASE total은 손 이격도 `h`에 따라
`h=0`에서 직전 native CARRY reward, `h=1`에서
`carry_done + 0.50 + 0.30*support + 0.20*stable`로 보간된다. bridge가 꺼진
기존 경로와 sequential mask 없이 쓰던 기존 bridge 동작은 그대로다. observation,
token/layer 수, controller와 gate는 바꾸지 않았다.

`train_carry_release_bridge_local.sh`는 ms43 sidecar를 기준으로 1024 env,
3000 iteration을 실행한다. ms43과 같은 freeze/mask를 유지하며 첫 backward의
gradient와 checkpoint별 stage 평가로 PLACE 보존 여부를 판정한다.

### MS18 물리 1-agent 시각화 호환 모드

`origin/local`에서 학습한 `ms18_maskteam_origscale_c06_s0` 환경을 실제 humanoid 한 명으로
재생하도록 `MS_SINGLE=1`을 추가했다. 뷰어 생성 config만 `numAgents: 1`로 바꾸고,
A=2 체크포인트가 기대하는 masked teammate 21-D 슬롯은 0으로 유지한다. 따라서 물리 actor와
박스·선반은 한 벌만 생성되면서도 체크포인트 observation 340-D와 tokenizer 입력
`[21, 12, 42, 42]`는 바뀌지 않는다. 이 모드는 `MA_TOKEN=mask` 이외의 체크포인트를 거부한다.
기존 A=2 기본값은 그대로다.

`bash -n`, Python compile, `git diff --check`를 통과했다. CPU 구조 검사에서
`task_obs_size=117`, 전체 observation 340-D, token dims `[21, 12, 42, 42]`를 확인해
MS18 체크포인트의 RMS 340-D와 teammate encoder 21-D에 일치시켰다.
GPU 7·임시 포트 6111의 실제 viewer smoke에서도 `env 1 x 1명`, `agents: 1`,
`num_obs: 340`과 MS18 PTH 복원을 확인했다. 1-env 평가 3회를 끝까지 완주했고
`fail_trials_because_terminate=0`이었다. 검증 viewer는 종료했다.
긴 시나리오 녹화를 위해 view 전용 `MS_EPISODE_LENGTH` 손잡이도 추가했다.
기본값은 기존 600 step을 보존하며, 1800을 주면 30 Hz 기준 60초다. 양의 정수가 아닌 값은
실행 전에 거부하고, GPU 없는 생성 검사에서 A=1·1 env·1800 step YAML을 확인했다.

발표용 속도 경로에는 선택적 `MS_DRAW_SPEED_BROWN=1` 팔레트를 추가했다. MS_MRAND의
속도 배수 0.25/0.50/0.75/1.00을 진한 갈색에서 기존 agent 경로색으로 선형 보간한다.
옵션을 끄면 기존 밝기 배율 색상을 그대로 사용한다. Python compile과 diff check를 통과했다.

### C5/C13 trajectory coordinator opt-in 이식

`traj` 브랜치의 C2 coordinator runtime과 `HumanoidMACoordCarry`를 현재
`TokenHSI-masteer`에 별도 모듈·별도 task로 이식했다. 기본 task
`HumanoidMASteerCarry`와 기존 GT `_gt_path` 생성은 바꾸지 않았으며,
`MS_TASK=HumanoidMACoordCarry`, `COORD_MODEL=c2`, `COORD_CKPT=<pth>`를 명시한
실행에서만 learned joint path/speed가 기존 320점 steering 버퍼를 대체한다.

m44의 sequential stacking 환경과 train/view/eval 스크립트는 수정하지 않았다.
GPU를 비활성화한 Python compile과 TRAJ 원본 coordinator 단위검사 75개를 통과했다.
전달받은 `runs/coord/imported/c5.pth`와 `c13.pth`도 strict schema/state-dict load를
통과했다. checkpoint의 실제 step과 설정은 각각 C5-r2 2000
(`unnecessary_slow=2`)과 C13 path-guard-30 300(`path_residual=30`)이다. 두 모델 모두
2-batch CPU forward에서 `[2,2,33,2]` path와 `[2,2,33]` speed를 내고, 선택·resample한
`[2,2,320,2]` 경로가 finite/valid임을 확인했다. 파일 해시와 실행법은
`runs/coord/imported/README.md`에 기록했다. GPU bridge smoke는 별도 실행 단계다.

## 2026-09-10

### ms47 reward-only CLEAR balance + strict stack success 40

ms46의 모델, 340-D observation, phase controller, 곡선 CLEAR 경로와 ms18 epoch-9000
초기화를 그대로 보존한 reward-only pilot 래퍼
`scripts/masteer/train_stack_first_clear_balance_local.sh`를 추가했다. RELEASE의 기존
`carry_done=1.0`은 유지하되 CLEAR부터 추가되는 `release_done`만 1.0에서 0.5로 낮추고,
stall penalty를 0.75에서 1.50으로, grace를 20에서 5 frame으로, 최소 이동 비율을
0.20에서 0.40으로 바꾼다. 따라서 grace 뒤 정지 CLEAR의 completion floor 1.5가 stall
1.5에 정확히 상쇄되고, ms46 command 0.375 m/s에서 stall 기준은 ms43과 같은 절대
0.15 m/s가 된다.

top 접근·release shaping과 5-frame strict success gate는 유지하고, 실제 물리 stack
성공 때 두 agent에 한 번 지급하는 `STACK_SUCCESS_BONUS`만 20에서 40으로 높였다.
단순 hand release에는 큰 보너스를 주지 않는다. steering tokenizer와 internal adapter는
계속 학습하고 carry tokenizer, Transformer와 composer는 기존 freeze 계약을 유지한다.

`bash -n`과 `git diff --check`를 통과했다. GPU 7에서
`ms47_ms18init_curveclear_balance_success40_1000_s0`를 detached session으로 시작했고,
launcher PID 2040962가 PID 1 아래 독립 session으로 유지되는 것을 확인했다. 생성된 sidecar와
첫 실행 로그에서 ms18 epoch-9000 초기화, success bonus 40, CLEAR stall 1.50,
move-min 0.40, grace 5, completion reward 1.00/0.50/0.00을 재확인했다. 첫 backward에서
steering tokenizer와 internal adapter에 각각 nonzero gradient가 들어왔고, freeze 대상은
zero/frozen 상태를 유지했다.


### ms48 1 m WAIT + 180-frame STACK budget and hold diagnostics

ms47 평가에서 strict stack success가 0인 직접 병목은 top carrier의 staging 거리 약 2 m와
STACK 진입 뒤 생존 중앙값 약 28.5 frame의 조합이었다. `train_wait1_stack180_holddebug_local.sh`는
모델·340-D observation·shared policy·carry/backbone freeze를 바꾸지 않고, top의 첫 목표를
base와 1 m 떨어진 고정 WAIT point로 둔다. 두 손 grasp, box 속도 안정, WAIT tolerance를
5 frame 연속 충족해야 stage latch가 생기며, base CLEAR가 끝나도 이 latch가 없으면 STACK으로
넘어가지 않는다. STACK 진입 순간에만 live base top pose로 한 번 retarget하고 180 physics
frame을 별도 보장한다. 전체 episodeLength는 600+180=780이다.

WAIT/approach 동안 top이 손을 먼저 떼면 한 손이라도 box surface에서 멀어진 정도에 비례해
`STACK_TOP_PREMATURE_RELEASE_PEN_W=0.50` 벌점을 준다. 목표까지 signed 거리 진전은 2.0
가중치로 보상하고, stable placement 8 frame 뒤에만 기존 release-progress/hold shaping을
활성화한다. success bonus +40과 strict 5-frame physical success gate는 유지한다.

학습 TensorBoard에는 `top_native`, `top_approach_progress`, `top_premature_release_penalty`,
`top_release_progress`, `top_hold_penalty`와 distance/grasp/settled/release-ready/stack-age를
추가했다. 평가 metrics는 stage/above/settled step, STACK frame·grasp·place count, 시작·최소
distance, reward 누적, 종료 사유를 더 기록하고 `stack_stage_summary.py`가 funnel/motion/reward/
termination 통계를 출력한다. 기존 74-column ms47 eval 포맷도 계속 읽는다.

GPU 학습은 사용자 요청에 따라 시작하지 않았다. Python compile, Bash syntax, diff check와
기존 ms47 e10000 eval summary의 backward-compatible 읽기를 통과했다.

GPU 7에서 64 env, 2 iteration smoke를 실행했다. 최초
`smoke_ms48_wait1_stack180_2it_20260910_s0`는 `minibatch_size=1024`가 기존
`amp_minibatch_size=4096`보다 작아 학습 시작 전 assertion으로 종료됐다. 출력은
지우거나 tag를 재사용하지 않았다. 새 tag
`smoke_ms48_wait1_stack180_2it_20260910_s0_r1`은 `PILOT_MB=4096`으로 rc=0
완료했다. 생성 config에서 64 env, episodeLength 780, minibatch/AMP minibatch 4096을
확인했고 sidecar에서 WAIT 1 m, 5-frame stage latch, PRE/STACK budget 600/180,
top settle 8을 재확인했다. 첫 backward의 steering tokenizer와 internal adapter
gradient 합은 각각 `9.276921e+02`, `1.712486e+03`이었고 frozen carry tokenizer,
self encoder, Transformer, composer는 0이었다.

```bash
# 64-env, 2-iteration smoke. 동일 tag는 재사용하지 않는다.
MA_GPU=7 PILOT_ENVS=64 PILOT_MB=4096 PILOT_ITERS=2 PILOT_SAVE_EVERY=1 \
  bash scripts/masteer/train_wait1_stack180_holddebug_local.sh <new-smoke-tag>

# ms18 epoch-9000에서 시작하는 1024-env, 1000-iteration ms48 pilot.
MA_GPU=7 bash scripts/masteer/train_wait1_stack180_holddebug_local.sh \
  ms48_ms18init_wait1_stack180_holddebug_1000_s0
```

### ms48 evaluation-only CLEAR→STACK transition trace

학습 없이 원인을 분리하도록 기본 비활성 디버그 옵션과 전환 trace를 추가했다. `STACK_TRACE`는 전환 직전·직후와 이후 60 frame 동안 두 agent의 action, task-observation 구간별 norm, command, root 속도·upright, fall 판정, box-target 거리와 hand 거리를 NPZ로 저장한다. `STACK_DEBUG_KEEP_WAIT`, `STACK_DEBUG_CARRY_LIVE_ON_STACK`, `STACK_DEBUG_REQUIRE_TOP_BALANCED`로 top retarget 제거, STACK에서만 base carry observation 복원, top 안정 gate 강화를 각각 독립 시험할 수 있다. 기본 환경변수에서는 기존 동작이 바뀌지 않는다.

`scripts/masteer/debug_stack_transition_eval.sh`와 `stack_transition_trace_summary.py`를 추가해 동일 checkpoint·seed에서 평가 전용 ablation을 반복하고 요약한다. ms48 epoch 9700, 128 env에서 baseline과 carry 복원, base hold 제거·장거리 이동, top 속도, WAIT 유지, 안정 gate, 조합, 지연 전환을 비교했지만 모두 strict stack success 0이었다. top 목표·steer·carry·command를 그대로 두고 base도 동일 장거리 경로를 유지한 대조군에서도 안정적으로 진입한 뒤 20–30 frame 사이 두 agent의 속도와 자세가 함께 붕괴했다. 따라서 view/render, 목표 순간이동, base zero-carry 또는 명시적 hold 하나가 단독 원인은 아니며, late CLEAR/WAIT 물리 상태에서 shared policy의 장기 안정성 부족이 주 병목이다.

학습은 실행하지 않았다. 결과와 trace는 `runs/results/masteer/debug_stack_ms48_ms18init_wait1_stack180_holddebug_1000_s0_e9700.log` 및 같은 디렉터리의 `trace_*__debug_*_e9700.npz`에 보존했다.

### ms49 model-preserving late-STACK stability curriculum

ms48 trace에서 task input과 action이 유지돼도 STACK 진입 20--30 frame 뒤 두 agent가 함께
무너졌고, 학습 마지막 100 scalar의 `stack_fraction` 평균은 0.00079, success bonus는
전체 구간 0이었다. 모델·340-D observation·token layout을 바꾸지 않고 실제
CLEAR→STACK 전환 물리 상태를 env별로 저장했다가 이후 reset 일부에서 재사용하는 기본
비활성 curriculum을 추가했다. `STACK_BOOTSTRAP_FRAC`이 재사용 비율을,
`STACK_BOOTSTRAP_KEEP_WAIT`가 bootstrap episode의 base/top hold를 제어한다. 복원 시
humanoid root/DOF와 두 box state, role별 goal/controller 상태를 함께 되살리고 AMP history는
복원된 현재 state로 채운다. 최종 end-to-end 평가에는 curriculum이 섞이지 않도록
`STACK_BOOTSTRAP_EVAL=0`이 기본이며 진단 평가에서만 명시적으로 켤 수 있다.

`STACK_HUMANOID_STABILITY_W`는 STACK에서 두 agent의 upright·root height·angular stability를
dense reward로 추가한다. 기존 box support와 top native reward는 유지한다. TensorBoard에는
`stack_phase/bootstrap_fraction`과 `stack_reward/{base_stability,top_stability}`를 추가했다.
`train_late_stack_stability_local.sh`는 ms48 epoch 9700에서 시작해 rehearsal 0.30,
bootstrap 0.50, WAIT hold, stability weight 1.00을 사용하며 기존 steering tokenizer/internal
adapter만 학습한다.

GPU 7 smoke에서 최초 `smoke_ms49_latehold_2it_20260910_s0`는 snapshot buffer를 simulator
tensor 생성 전 할당한 오류를 잡아 학습 시작 전에 종료됐다. 지연 할당으로 수정한
`smoke_ms49_latehold_2it_20260910_s0_r1`은 64 env, 2 iteration rc=0으로 완료했다.
`smoke_ms49_latehold_branch40_20260910_s0`은 128 env, 40 iteration rc=0으로 실제 저장·복원
분기를 통과했고 bootstrap/STACK 최대 점유율 0.012695/0.015625와 두 stability scalar를
기록했다. full pilot은 시작하지 않았다. Python compile, Bash syntax, diff check를 통과했다.

## 2026-09-11

### A2 WAIT endpoint와 STACK 재출발을 Juan carrier 방식에 맞춤

기존 실험의 재현성을 유지하도록 기본값은 그대로 두고 `STACK_STAGE_USE_HAND_Z`와
`STACK_STAGE_FORCE_ZERO`를 추가했다. 전자를 켜면 A2의 staging z를 reset 시점 양손의 평균
높이로 정하고, 후자를 끄면 stage latch에서 `mscale=0`을 강제하지 않는다.
`train_a2_transition_local.sh`는 ms50r1 epoch 9300에서 시작해 staging 거리 1.5 m, 손 높이
목표, 연속 endpoint steering, `STACK_TOP_SCALE=1.0`을 활성화한다. bootstrap 복원에서도
A2가 실제로 재출발하도록 WAIT 고정을 끈다. carry zero-padding, attention mask, 모델 구조와
freeze 설정은 ms50r1 그대로 유지한다. 학습은 실행하지 않았다.

### A1 CLEAR→DECEL→정지 전환 학습

모델·observation ABI를 바꾸지 않고 CLEAR endpoint 0.30 m부터 A1의 목표 속도를 선형으로
낮추는 감속 reward를 추가했다. 이동 reward와 stall/reverse penalty는 감속 구간에서 끄고,
root 선속도 0.10 m/s 이하, 각속도 0.50 rad/s 이하, upright 오차 15도 이하, 양발 접촉을
15 frame(30 Hz에서 약 0.5초) 연속 만족해야 STACK으로 전환한다. 조건을 만족하기 전에는
endpoint steering을 유지하므로 `_hold_steer()`의 급격한 정지 명령은 안정 정지 이후에만
적용된다. 기존 A2 WAIT/STACK 전환 수정은 그대로 유지한다.

`train_a1_w2s_transition_local.sh`는 ms18 epoch 9000에서 시작하며 CLEAR steering reward를
0.50에서 1.00으로 높이고 stop reward weight 1.00을 사용한다. 새 정지 구간을 실제로
통과해 학습하도록 stack bootstrap은 끈다. 모델 구조와 freeze 설정은 기준 ms50r1 sidecar를
그대로 상속하며 학습은 실행하지 않았다.

## 2026-09-12

### A2 carry-target-only zero-padding 진단

STACK_DEBUG_TOP_CARRY_TARGET_ONLY를 기본 비활성 옵션으로 추가했다. 활성화하면
non-rehearsal STACK의 top 역할에서 steering 12-D 값만 0으로 채우고, carry 두 토큰과
attention/token layout, A1 입력, controller와 목표점은 그대로 둔다. A1 carry
zero-padding과 마찬가지로 토큰을 attention에서 제거하지 않는다.
debug_stack_transition_eval.sh에는 재현 가능한 top_carry_target_only case를 추가했다.

ms52 latest checkpoint의 128-env, seed-0 평가에서 top steering norm이 전환 직후
0.768→0.000으로 바뀌어 옵션 적용을 확인했다. 그러나 baseline 대비 STACK 생존 p50은
32→32 frame, top-first 낙상은 10→11/16, 목표 접근 중앙값은 0.139→0.019 m,
top grasp frame 비율은 0.604→0.491로 개선되지 않았고 strict success는 둘 다 0이었다.
따라서 새 steering window 하나가 단독 원인은 아니며, carry-target-only를 쓰려면
STACK bootstrap으로 steer=0 + live carry 조합과 WAIT→재출발 상태를 직접 학습해야 한다.
Python compile, Bash syntax, diff check와 GPU 7 실제 평가를 통과했다.

### ms53 A2 direct carry-target reward + stable snapshot curriculum

정식 옵션 `STACK_TOP_CARRY_TARGET_ONLY`를 추가해 기존 debug alias와 같은 raw steering
zero-padding을 재현 sidecar에서 명시할 수 있게 했다. `STACK_TOP_DIRECT_CARRY_REWARD`는
STACK의 top row만 Juan native carry reward(`walk + carry + handheld + putdown - power`)로
치환한다. live stack target과 carry token은 유지되고 A1 row와 다른 phase reward는 그대로다.

ms52의 학습 TensorBoard에서 3000 iteration 내내 STACK/bootstrap fraction이 0인 것을
발견했다. 기존 snapshot은 실제 15-frame stop gate 통과 뒤에만 생성돼 stochastic 학습에서는
late-phase curriculum이 비활성이었다. `STACK_BOOTSTRAP_CAPTURE_STABLE`은 정상 phase gate를
바꾸지 않고, 나머지 CLEAR gate와 기존 A1 stop predicate를 한 frame 만족한 state만 snapshot
seed로 저장한다. 복원 시 A1 hold, A2 direct carry로 시작한다.

`train_a2_carry_target_bootstrap_local.sh`는 ms52 epoch 12000, A2 steering zero-padding,
direct carry reward, rehearsal 0.30, bootstrap 0.50, stable capture를 묶어 추가 3000 iteration을
GPU 7에서 실행한다. 모든 새 knob를 `train_local.sh` sidecar/export 목록에도 추가했다.

Python compile, Bash syntax, diff check를 통과했다. 256-env/80-iteration smoke는 epoch
12080까지 완료했고 bootstrap/STACK fraction 최대 0.02515, 마지막 0.00830을 기록했다.
본 학습 `ms53_ms52e12000_a2carryonly_boot50_3000_s0`은 1024 env로 첫 backward를
통과했으며 epoch 12000→15000을 실행 중이다.

## 2026-09-13

### 장기 학습 metrics 무한 누적·전체 재저장 방지

ms53r1 장기 학습에서 wrapper가 `MS_METRICS`와 `MA_METRICS`를 unset했지만
`train_local.sh`가 기본 npy 경로를 다시 만들어 metrics를 활성화했다. 완료 episode
tensor가 계속 `_metric_rows`에 쌓였고 25 reset batch마다 지금까지의 전체 배열을
`torch.cat`한 뒤 같은 npy를 덮어써, 파일 4.4 GiB·누적 쓰기 약 2.2 TB와 224-core
CPU 포화를 일으켰다. 사용자가 해당 학습을 종료했으며 코드는 프로세스를 종료하지 않았다.

`train_local.sh`의 metrics를 명시적 `MS_METRICS` 또는 `MA_METRICS` 경로가 있을 때만
켜도록 바꿨다. 환경은 metrics가 꺼진 실행에서 CPU tensor 복사와 `_metric_rows` 누적을
생략하고, 켠 실행도 `MA_METRICS_MAX_ROWS=262144` 기본 상한에서 수집·재저장을 멈춘다.
flush 주기는 `MA_METRICS_FLUSH_BATCHES=25`로 명시하고 두 값을 sidecar에 기록한다.
기존 4.4 GiB npy는 결과 보존 원칙에 따라 삭제하지 않았다. Python compile, Bash syntax,
`git diff --check`를 통과했으며 새 학습·평가는 실행하지 않았다.

### A1 steering-only CLEAR + A2 committed STACK 구조

기존 sidecar 재현성을 위해 기본값은 유지하고, `STACK_TOP_WAIT_AT_START`와
`STACK_TOP_COMMIT_GOAL`을 추가했다. 전자를 켜면 A2는 pre-STACK에서 자기 reset box 위치를
목표로 기다리고, 후자를 켜면 CLEAR 종료 순간 실제 base box pose로 top pose를 한 번 계산해
고정한다. STACK 동안 base 흔들림을 매 frame 따라가던 live 목표 갱신은 committed mode에서
중지하고, A2의 carry target과 기존 steering 경로를 같은 고정 목표로 동시에 retarget한다.

bootstrap snapshot은 `STACK_BOOTSTRAP_TOP_BALANCE_STEPS`와 humanoid root 선·각속도,
upright, top box 선·각속도 임계값을 모두 연속 만족한 env만 저장한다. Juan식 wait에서는
A2가 STACK 전 box를 들지 않으므로 `STACK_BOOTSTRAP_TOP_REQUIRE_GRASP=0`을 명시할 수 있다.

`MA_FINETUNE_NEWCARRY_RESIDUAL=1`은 모델 구조를 추가하지 않고 기존 actor 전체를 먼저
동결한 뒤 `new_carry` tokenizer와 기존 `internal_adapt_mlp`만 다시 연다. 별도 wrapper
`train_a2_committed_steer_bootstrap_local.sh`는 ms52 epoch 12000에서 시작하고, A1 CLEAR의
두 carry token을 exact attention mask해 steering-only로 만들며, A2에는 steering을 0으로
패딩하지 않는다. A2 snapshot 안정 streak는 12 frame이고 학습 metrics는 비활성이다.

Python compile, Bash syntax, `git diff --check`, checkpoint/sidecar 존재 검사를 통과했다.
새 학습·평가는 실행하지 않았다.
후속 점검에서 이 wrapper의 metrics/trace 저장은 실제로 비활성이고, 남은 디버그는 첫
backward에서 한 번 실행되는 `MS_GRADCHK`뿐임을 확인했다. 다음 실행부터는 wrapper가
`MS_GRADCHK=0`을 명시하며, agent도 문자열 `"1"`일 때만 검사를 실행하도록 바꿨다.
이미 실행 중인 Python 프로세스에는 영향을 주지 않는다.

### A2 carry tokenizer 동결 ablation 스위치

`train_a2_committed_steer_bootstrap_local.sh`에 `PILOT_FREEZE_A2_CARRY=1` 모드를 추가했다.
기본 ms54 모드는 바꾸지 않으며, 이 모드에서는 ms52 epoch 12000 시작점과 Juan식 wait,
committed top goal, A1 steering-only CLEAR, bootstrap 설정을 그대로 유지한다. 차이는
`new_carry` tokenizer를 동결하고 steering extra tokenizer와 기존 shared
`internal_adapt_mlp`만 학습한다는 점이다. 최종 action residual은 계속 바뀌므로 이는 carry
행동 전체 동결이 아니라 carry representation 동결 ablation이다.

## 2026-09-14

### ms18 shared-goal mixed sequence — carry 보존형 steering-only 학습

기존 순차 적층 환경에 `STACK_SHARED_GOAL_CARRY`를 추가했다. 두 역할은 같은 최종 XY를
공유하지만 Top은 시작부터 box를 들고 goal 앞 safety gate까지 이동한다. 먼저 도착하면 box를
든 채 HOLD하고, Base의 release·후퇴·15-frame 안정 정지가 모두 끝난 뒤에만 그 시점의 실제
base box pose로 Top 목표를 한 번 commit한다. STACK 전환 뒤에는 기존 `_hold_steer()`로 Base를
후퇴 endpoint에 고정하며, Top 대기 품질과 Base HOLD 품질을 각각 작은 dense reward와
TensorBoard state로 기록한다. strict success bonus는 40으로 유지하고 중간 bonus는 새 래퍼에서
0으로 둔다.

ms55의 carry 저하를 피하기 위해 `MA_FINETUNE_STEER_ONLY`를 추가했다. 새 레이어·token·head나
관측 차원은 만들지 않으며, ms18과 동일한 actor에서 기존 steering tokenizer만 학습한다.
carry/teammate tokenizer, Transformer backbone, composer, shared `internal_adapt_mlp`, actor RMS는
동결·eval 상태를 유지한다. critic은 기존 PPO 방식대로 학습한다.

`train_shared_goal_single_local.sh`는 ms18 epoch 9000에서 시작해 한 PPO run 안에 full sequence,
carry rehearsal 20%, 유효 physical snapshot 기반 late-STACK reset 15%를 섞는다. 학습률은
`MS_LR=5e-6`, 기본 추가 iteration은 6000, GPU는 허용된 7번이다. `train_local.sh`가 학습률과
새 knob를 생성 cfg 및 재현 sidecar에 남기도록 확장했다. Python compile, Bash syntax,
`git diff --check`, checkpoint/config 존재와 기본 tag 미사용 여부를 확인했으며 학습은 실행하지
않았다.
