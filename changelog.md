# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-09-20

### 17번 sampled OnTop edge context 연결

- 16번 scalar PRE/TERM·semantic/context fusion·GTA를 재사용해 `approach_distance_edge_context_ontop`을 별도 schema 3 / scratch 실험으로 추가했다. reset마다 각 agent가 Holding + NONE/AT/ON_TOP(.2/.5/.3)을 뽑고, 유효한 받침·chain·PRE/TERM·required-goal을 생성한다. A/B·물체 배정·edge 순서를 무작위화하고 그래프는 에피소드 동안 유지한다.
- OnTop은 회전 bbox의 월드 Z extent로 source 최하단과 support 최상단의 signed gap을 계산한다. k=10·현재 성공(phi≥.9, |gap|≤1mm)·own/TERM 성공 시 세 항 각각 .2 포화를 사용한다. PRE reward gate는 없으며 모든 그래프에 `.9 own + .1 other` task 공유를 적용한 뒤 각자 penalty와 기존 AMP/PPO 경로를 유지한다.
- 환경별 7-field graph/context packet을 실제 obs/next_obs에 저장하고 actor/critic이 해당 sample의 binding만 읽는다. 기본 관측은 615차원, context MLP 입력은 그대로 2차원이다. padding·bias 합산·RMS/clipping 우회·strict checkpoint 계약과 다른 E/M/O의 generic network forward를 지원한다. 기존 실험 경로는 보존한다.
- Ox는 dynamic box로 유지하며 target 플랫폼은 AT에만 활성화한다. 3단 적층 높이를 위해 새 실험의 실제 asset 크기를 0.2~0.4m로 제한하고, 높이·간격 제약 위반 시 같은 그래프의 물리 배치만 재시도한다. 실제 smoke에서 발견한 환경 클래스 reset dispatch 누락을 수정하고 reset 분포·플랫폼 비활성 검사를 추가했다.
- 전용 YAML·train/test/VNC, viewer preset 4개·역할 반전·loco 기본 초기화·stack 카메라·OnTop 비충돌 십자선을 연결했다. config 전체 목록/명령·structure와 `markdowns/edge_context_ontop.md`에 실행법·수식·호환성·물리 크기 차이를 기록했다.
- CPU 전체 **231개 통과**. 100,000 reset/12 outcome 분포, PRE/TERM·성공·공유, 회전 8-corner 대조, edge/batch/assignment 순열, padding/bias 합산, actor/critic·gradient·strict load·가변 크기·정규화·진단 분모를 확인했다.
- GPU **5만** 사용해 **2048환경** scratch epoch 2 저장과 resume epoch 3 저장을 확인했다. 실제 PPO 관측/next packet·혼합 보상·GAE 입력까지 assertion으로 검증했다. 두 run의 TensorBoard scalar **142개씩 모두 유한값**, scratch CSV 16행의 포화·공유·penalty 합계를 확인했다. 마지막 scratch 집계 21,199 reset의 second 비율은 .20133/.49750/.30117, 물리 재시도 471·실패 0이다.
- GPU 5의 16환경 시뮬레이션에서 네 preset 총 128 transitions·부분 reset·과거 graph replay·AT-only 플랫폼을 확인했다. 통제된 물리 fixture에서 box-box 적층 접촉과 dynamic Ox 이동을 확인했다. 이는 학습된 적층 행동 검증이 아니다.
- 전용 VNC wrapper로 GPU 5 / 포트 6081 / 1환경 / 역할 반전 / 2회 짧은 평가를 실행했다. noVNC HTTP 200·동일 CUDA/Vulkan GPU·graph 출력·평가 JSON 저장·정상 종료를 확인했다. 결과는 `output/approach_distance_edge_context_ontop_check/verification.json`과 하위 로그/metrics에 있다. 기존 학습은 종료·재시작하지 않았고 본학습은 자동 시작하지 않았다. 장기 수렴·자연스러운 적층/release·비켜주기·다인원 task 성능은 미검증이다.

### 기본 MPS 경로의 잔여 소켓 정리

- 다른 계정의 CUDA error 805 보고를 받아 기본 `/tmp/nvidia-mps`를 재점검했다. 제어 PID·응답·커널의 살아 있는 socket은 없었지만, 이전 전체 GPU 데몬의 본인 소유 소켓/lock 파일 3개가 남아 있었다.
- 소유권·파일 종류·미사용 상태를 확인하고 `/tmp/mps-test-1003/global-default-stale-20260920_191001/pipe`로 보관했다. 현재 기본 `/tmp/nvidia-mps` 경로는 없으며, GPU 6 전용 MPS와 기존 학습 client 연결은 유지된다.
- MPS 환경변수를 제거한 별도 프로세스에서 GPU 6에 대한 `cudaGetDeviceCount`가 정상 반환(0, 장치 1개)함을 확인했다. GPU 메모리 할당·커널 실행은 하지 않았다. 다른 계정/GPU의 실패 상황은 재현하지 않았으므로 과거 805의 정확한 원인이나 그 계정의 복구를 단정하지 않는다.

### MPS 사용 GPU 5·6 운영 범위 확인

- 사용자가 MPS를 GPU 5·6에서만 쓰겠다고 명시했다. GPU 4에 남아 있던 본인 소유 제어 데몬은 서버·client가 없음을 재확인하고 전용 경로로 정상 종료했다. 다른 사용자의 CUDA 작업과 GPU 6의 MPS 학습은 유지했다.
- 확인 시 실제 MPS 서버·학습 연결은 GPU 6에만 있으며, 기존 GPU 5 학습은 일반 CUDA 실행이다. GPU 0~4의 MPS는 자동 시작하지 않는다. 이 항목은 운영 방침이며 GPU 번호에 대한 하드 차단 설정을 추가한 것은 아니다.

### MPS 서버 즉시 시작과 학습 자동 연결

- `mps_start GPU`가 검증한 전용 데몬에 현재 UID의 `start_server`를 요청해 학습 없이도 해당 GPU의 MPS 서버까지 띄운다. 이미 떠 있는 서버는 재사용한다. 정상 종료 후 남는 알려진 log FIFO도 안전한 잔여 파일 보관 대상에 포함했다.
- 공통 `runtime_env.sh`에 `mps_auto_env.sh`를 연결했다. train/test/VNC의 지정 GPU에 내 MPS가 실행 중이면 새 터미널에서도 UUID·전용 pipe를 자동 적용한다. `mps_run`은 필요 없다. 다른 GPU/전역 pipe 상속·불명확한 데몬은 실행을 거부하며 자동으로 데몬을 시작하거나 종료하지 않는다.
- 검증: MPS 보호·즉시 서버 시작·자동 연결 테스트 **18개**, GUI GPU 테스트 **6개 통과**. 실제 GPU 6에서 학습 없이 서버를 시작했고, MPS 환경변수를 지운 새 셸에서 runtime을 통해 checkpoint로 64-step 시뮬레이션을 실행했다. 실제 client PID 연결과 서버가 GPU 6에만 표시됨을 확인했다. 기존 학습·다른 GPU 데몬은 변경하지 않았다.
- 개인 설치본·이식 문서·MPS README·config·structure를 갱신했다. 자동 연결은 이 레포의 공통 runtime을 사용하는 실행에 적용되며, 이미 실행 중인 학습은 전환되지 않는다.

### MPS 정상 종료 후 재시작 복구

- NVIDIA가 정상 종료 후 남긴 소켓 파일을 기존 보호 장치가 거부하던 문제를 수정했다. GPU 전용 경로에서 PID 없음·제어 응답 없음·커널의 살아 있는 소켓 없음·내 소유의 알려진 파일만 존재함을 확인한 경우에만 기존 pipe를 `pipe.stale.*`로 보관하고 새로 만든다. 살아 있는 socket·다른 사용자·잘못된 PID/설정·알 수 없는 파일은 계속 거부한다.
- repo·개인 설치본·이식 문서 코드를 동기화했다. client가 없을 때 `mps_status`는 대기 중임을 명확히 표시한다.
- 가짜 NVIDIA 도구 테스트 **13개 통과**(종료 잔여 socket 복구·살아 있는 socket 보호 포함). 실제 GPU 6의 종료 잔여 파일을 보관한 뒤 GPU 6 전용 MPS 시작·UUID/경로 검증·대기 상태 조회를 확인했다. 다른 GPU의 MPS와 기존 학습은 변경하지 않았다.

### 공유 서버용 MPS GPU 범위 검증 강화

- 시작·연결·상태·종료 전에 전용 경로와 PID 소유자, 데몬의 단일 GPU UUID·pipe·log 경로를 검증한다. 불일치하거나 확인할 수 없으면 기존 데몬을 건드리지 않고 실패한다. 전역 기본 경로 fallback은 없으며 동시 시작은 GPU별 lock으로 직렬화한다.
- 개인 설치본과 다른 서버용 설치 문서의 코드를 함께 갱신했다. `.bashrc`의 옛 MPS alias 블록은 백업 후 제거해 자동 로드 실패 시 전체 GPU 명령으로 돌아가지 않게 했다. 백업 접미사는 `20260920_170444_150084`다.
- 가짜 NVIDIA 도구 기반 보호 테스트 **10개 통과**, Bash 문법·새 터미널 자동 로드·기존 GPU 6 데몬의 UUID/경로를 읽기 전용으로 확인했다. 이번 강화 작업에서는 실제 MPS나 학습을 시작·종료하지 않았다.
- 같은 터미널의 이후 작업에만 설정이 적용되며 기존 학습은 자동 전환되지 않는다. 확실한 명령별 전달은 `mps_run GPU <command>`로 안내한다.

### 전체 GPU에 표시되던 기존 MPS 정리

- 옛 alias의 실패 실행이 GPU 제한 없이 `/tmp/nvidia-mps`에 제어 데몬을 남겼고, 그 서버가 GPU 0~6 모두에 표시되는 것을 확인했다. GPU별 helper의 전용 데몬과는 별개였다.
- 연결된 client가 없음을 확인한 뒤 기본 경로의 잘못 뜬 데몬만 `quit`으로 정상 종료했다. GPU별 데몬과 기존 학습은 유지했다.
- GPU 6 전용 경로로 짧은 CUDA 연산을 실행해 client 연결·연산 결과·MPS 서버가 GPU 6에만 표시되는 것을 확인했다. 확인용 CUDA 작업은 종료했다.

### MPS 단축 명령 자동 로드

- 사용자 요청으로 `mps/shell.sh`를 `~/.local/share/gpu-mps/shell.sh`에 설치하고 `.bashrc` 끝에 자동 로드를 등록했다. 새 Bash에서는 바로 `mps_start 6`을 쓸 수 있으며, 기존 터미널은 설치본을 한 번 source한다.
- 빈 경로 변수를 사용하던 기존 alias 원문은 보존하고 로드 시 함수로 대체한다. `.bashrc` 수정 전 백업은 개인 설치 디렉터리의 `bashrc.backup.20260920_165606_046792`다.
- 검증: Bash 문법, 새 interactive Bash에서 공개 함수 5개 자동 로드·alias 해제·GPU 6 UUID와 전용 경로 선택 확인. 실제 MPS 시작·종료나 기존 학습 변경은 하지 않았다.

### 16번 실험 전체 명령 목록 누락 보완

- `config.md`의 개별 실험 섹션에만 있던 16번 명령을 전체 학습·시각화 목록과 서버 VNC 섹션에도 추가했다. 실험 번호 범위와 scratch/resume 설명을 맞췄다.
- 검증: 명령의 스크립트 경로·문서 링크·diff 확인. 문서만 수정했으며 학습은 실행하지 않았다.

### Holding·At scalar PRE/TERM context와 edge 성공 포화

- 16번 `approach_distance_edge_context_success`를 별도 schema 2 / `state_relation_v1`로 구현했다. Holding/At k=10, 기존 distance progress·scene/reset·AMP·PPO를 유지하고 checkpoint 전이 없이 scratch 학습한다. 기존 k=5/k=10/OnTop 모드는 보존한다.
- 새 reward에서는 prerequisite 곱셈을 제거했다. edge의 현재 own success 또는 지정 TERM의 실제 own success일 때 state/progress/success를 각각 0.2로 포화한다. 재귀 TERM·latch·agent 추가 성공 보상은 없고, 2-edge/agent 상한은 1.2다. Raw state/context와 실제 성공은 보상 포화로 덮어쓰지 않는다. 패널티도 유지한다.
- Explicit graph compiler, binding 기반 Holding/At evaluator, owner 합산과 required-goal 평가를 추가했다. 정책은 저장된 `[q_pre,q_term]`을 context encoder·semantic/context fusion으로 처리해 기존 entity attention bias에 연결한다. 동일 pair bias는 합산, graph buffer는 non-persistent이며 generic 경로의 E=2M 의존을 제거했다. 새 checkpoint 평가의 E/M/O 변경과 동일 task training resume 계약을 분리했다.
- 전용 YAML·train/test/VNC, 3-edge 평가 override 예제, relation별 raw/실제 성공/포화 TensorBoard와 별도 CSV를 추가했다. config·structure·`edge_context_success.md`에 수식·명령·호환 범위를 기록했다. 운영 지시에 따라 실제 GPU 검증은 GPU 6만 사용했다.
- CPU 전체 **220개 통과**. 성공 경계·live 해제·비재귀 TERM·PRE와 reward 독립성·E=2/4/5/8/10·순서/owner·중복 pair·gradient·actor/critic strict load·RMS·패널티 보존을 확인했다. 셸 문법·문서 링크·diff도 확인했다.
- GPU 6 / **2048환경** scratch 확인에서 epoch 2를 저장하고 같은 모델 resume으로 epoch 3을 저장했다. 관측 595차원, scalar 92개 모두 유한값이며 CSV의 각 지급 항·agent total 합계를 검증했다. 같은 checkpoint로 M2/O3/E3 및 M3/O4/E6을 각각 16환경·64 step 실행해 partial reset·저장 context·minibatch 순열·RMS/clipping 우회·reward/penalty 일치를 확인했다.
- 서버 VNC wrapper로 GPU 6 / 1환경 / 짧은 평가를 실행하고 모델 로드·평가 JSON 저장·정상 종료를 확인했다. 결과는 `output/approach_distance_edge_context_success_check/verification.json`과 하위 run/metrics에 있다. 기존 학습은 종료·재시작하지 않았다. 장기 본학습, 새 graph에서의 task 성능, 자연스러운 접근→잡기→운반→배치→release 완주와 수렴은 미검증이다.

## 2026-09-19

### OnTop reward_terms 표시 분리

- 혼합 실험의 실제 OnTop 보상이 기존 `at_state`·`at_progress`에 합쳐 표시되던 문제를 수정했다. At/OnTop 마스크로 표시 성분만 나눠 `ontop_state`·`ontop_progress`를 추가하고 실제 reward·total·checkpoint 계약은 유지한다. k=5/k=10 carry의 기존 표시도 유지한다.
- OnTop viewer 로그는 env 0의 두 사람과 scenario/agent/role을 함께 출력한다. config·OnTop 문서에 평균 분모와 적용 시점(다음 실행/resume)을 명시했다. 기존 학습·event 파일은 변경하지 않았다.
- CPU 검증: 관련 diagnostics·OnTop·성공 포화 테스트 **48개 통과**. 입력 불변, 성분 분리 후 합계·기존 total 일치, At-only/OnTop-only/역할 혼합을 확인했다.
- GPU 0 / 2048환경 짧은 학습으로 새 TensorBoard OnTop 태그를 확인했다. scalar 169개 모두 유한값이고 분리한 reward 성분의 합과 total 오차는 1.5e-8 이하다. 저장 모델의 의존 시나리오 64-step 평가에서 두 사람의 role과 OnTop 항 출력·JSON 저장·정상 종료를 확인했다. 결과는 `output/approach_distance_success_ontop_reward_terms_check/`.

### Holding k=10 VNC 누락 보완과 실행 안내 통일

- 누락된 `approach_distance_success_holding_k10_vnc.sh`를 추가했다. 기존 공통 GUI wrapper로 k=10 test 스크립트를 연결하며 기본 평가 인자는 2/1/3/10, output은 `output/approach_distance_success_holding_k10_vnc`다.
- config.md 전체 목록에 14·15번 학습/로컬 추론/서버 VNC 링크를 표시하고, VNC 섹션에 두 실험의 전용 실행 명령을 추가했다. OnTop 섹션에도 로컬 viewer 명령과 실행 목적별 표를 보완했다.
- AGENTS.md에 새 실행용 config마다 train/test/VNC 스크립트와 세 실행 명령을 함께 제공하도록 명시했다. structure.md도 실제 VNC 파일 목록에 맞췄다.
- 검증: shell 문법·문서 링크·diff 확인. 사용자가 지정한 k=10 checkpoint를 GPU 4의 실제 viewer(C+G)로 로드했고, 확인용 포트 6081에서 noVNC HTTP 200, 1회 평가·JSON 저장·정상 종료를 확인했다. 확인용 VNC는 종료됐고 기존 학습은 유지했다.

### OnTop 혼합학습 구현과 checkpoint 전이

- 15번 `approach_distance_success_ontop_mixed`를 실행 코드에 연결했다. carry/carry 512 + 독립 OnTop 768 + 의존 OnTop 768, reset마다 A/B 역할 셔플, Holding k=5와 기존 초기화·AMP 설정을 사용한다. 원래 carry config와 스크립트는 유지한다.
- 회전된 바닥/윗면 중심 상태와 중심 XY progress, 이전 step의 `min(자기 Holding, 상대 At)` gate, 기존 로컬 성공 포화를 적용했다. B 성공에는 A 성공 조건을 추가하지 않는다. Ox는 기존 플랫폼 방식의 중력 off·고질량·속도 제한으로 고정하고 reset 시 재배치한다. 미사용 goal은 actor/critic에서 마스킹하고 해당 플랫폼도 비활성화한다.
- 저장 관측에 시나리오·역할을 추가해 PPO minibatch에서도 같은 그래프를 복원한다. 새 학습은 지정 epoch 18000 carry에서 텐서 177개·정규화 통계를 복사하고 actor/critic embedding만 8→9행으로 확장한다. 추가 freeze·optimizer/카운터 복원 없음; `transfer_report.json` 기록. OnTop resume은 전체 학습 상태를 복원한다.
- 전용 train/test/VNC 스크립트, 시나리오 선택 평가·TensorBoard 지표·평가 JSON을 추가했다. config.md 전체 목록·명령, structure.md와 ontop_mixed_config.md를 실제 실행 기준으로 갱신했다.
- 검증: 관련 CPU 테스트 **161개 통과**. carry 보상 동일성, gate·성공 포화, 회전 좌표, 역할/미니배치 순열, goal mask, 전이·gradient·진단 확인. shell 문법·문서 링크·diff 확인.
- GPU 4 / **2048환경** 확인용 전이 학습에서 epoch 2 저장 후 resume으로 epoch 3 저장. 두 실행의 TensorBoard scalar **167개씩 모두 유한값**, 시나리오 지표 36개 확인. 저장 모델의 16환경·64 step 검증에서 Ox pose 유지·부분 reset·보상/history·suffix·actor/critic 재배열 일치를 확인했다. 16환경 평가 스크립트도 JSON 저장까지 확인했다.
- 확인 결과는 `output/approach_distance_success_ontop_mixed_check/`에 저장했다. 장기 본학습·OnTop 수렴은 미검증이며, VNC wrapper는 셸 검증만 수행했다. 현재 혼합 구현은 사람 2·물체 3·세 시나리오 범위다.

### Holding k=10 비교 실험과 수식 문서

- `approach_distance_success_holding_k10` config·train/test를 추가했다. 기존 성공 시 포화 실험에서 Holding `hand_distance_scale`만 5→10으로 변경하고, 실험 이름·기본 output을 분리했다. 기존 k=5 파일과 reward 코드는 유지한다.
- `markdowns/approach_distance_success_holding_k10_reward.md`에 동일해진 Holding/At 상태함수·progress, 서로 다른 입력·선행 gate, 현재 성공·포화를 코드 블록으로 정리했다. config·구조 문서에 14번 실험과 실행법을 추가했다.
- 검증: 관련 CPU 테스트 **79개 통과**. YAML 설정 차이가 k 한 항목뿐임과 k=5 checkpoint 거부, k=10 phi/gate·기존 성공 포화 보존, 스크립트 문법을 확인했다.
- GPU 4 / 2048환경 / `MAX_ITERATIONS=1` 확인용 학습이 정상 종료됐고 epoch 2 checkpoint를 저장했다. 저장 metadata의 Holding k=10과 TensorBoard scalar 135개의 유한값을 확인했다. 결과는 `output/approach_distance_success_holding_k10_check/`; 장기 성능·수렴은 아직 검증하지 않았다.

## 2026-09-18

### 현재 실험 VNC 실행 단축

- `approach_distance_success_vnc.sh` 추가. 명령 앞의 `TOKENHSI_GPU`와 checkpoint만 지정하면 conda·VNC 경로·포트 6080·평가 인자 2/1/3/10을 기본으로 실행한다. 기존 환경변수와 평가 인자로 덮어쓸 수 있다.
- checkpoint 존재 여부를 VNC 시작 전에 확인한다. config 가이드와 구조 문서를 갱신했다.
- 검증: shell 문법, 기본값·사용자 인자·공백 포함 경로 전달·잘못된 checkpoint 거부 확인. GPU 2에서 단축 스크립트로 실제 checkpoint/viewer를 로드하고 64 step의 보상·history·관측 검증을 통과했다. 확인용 VNC는 종료했다.

### TensorBoard 핵심 pin 5개와 relation 표시 순서

- 추천 pin을 최종 배치율·배치 도달률·도달 후 유지율·현재 성공 비율·Holding 만족 비율의 5개로 줄였다.
- relation 태그를 `00_main` → `01_placement` → `02_reward` → `03_samples` → `04_state` → `05_motion` → `90_debug`로 정렬하고 그룹 내부에도 순서 번호를 붙였다. 같은 scalar를 중복 기록하지 않는다.
- 표시 이름만 변경하며 계산·reward·CSV·checkpoint 계약은 유지한다. 다음 시작/resume부터 적용된다. 과거 event와 실행 중인 학습, 브라우저 pin은 자동 변경하지 않는다.
- 검증: diagnostics/policy 테스트 **26개 통과**. 실제 TensorBoard event 쓰기/읽기로 61개 태그의 유일성, 핵심 5개와 그룹 순서, scalar 값·step 보존을 확인했다.

### 실험 2~8과 전용 코드 삭제, VNC GPU 지정 통일

- 기존 실험 2~8의 환경 YAML 7개, train/test 스크립트 14개, 전용 테스트 2개를 삭제했다. 남은 실험은 기존 ID 1, 9~13의 6개다.
- near/putdown 혼합 At, Gaussian velocity progress, At-only approach, 이전 self-gate 분기와 전용 config 필드를 제거했다. 남은 all-edge direction/distance, RSI, latch 기능은 유지한다.
- VNC는 명령 앞의 `TOKENHSI_GPU=5` 하나로 CUDA와 렌더링 GPU를 선택한다. 중간 `--gpu` 인자는 제거하고 README·config·운영 문서를 맞췄다.
- 검증: relation·RSI·GPU 테스트 144개와 policy 테스트 9개, **총 153개 통과**. 남은 relation config 5개의 수정 전후 reward·성공 이력·관측 suffix가 동일 입력에서 오차 없이 일치했다. 남은 YAML 설정값도 동일하다.
- 사용자 checkpoint를 GPU 2에서 환경변수 방식의 VNC로 로드해 64 step 실행했다. 보상·history·관측·reset 검증을 통과했고 확인용 viewer/VNC는 종료했다. GPU 5의 기존 학습은 유지했다.

### PRO 6000 학습 기준과 VNC GPU 선택

- 학습 환경 기본값과 문서를 2048로 통일했다. 이전 서버의 소규모 GPU 권고와 `SMOKE` 자동 축소 분기를 제거했다. 짧은 학습 검증도 2048 환경에서 반복 횟수만 제한한다. 평가·VNC 환경 수는 별도다.
- `USAGE.md`를 현재 PRO 6000 서버의 설치 환경·데이터·운영 안내로 축약했다.
- `gui_gpu_env.sh`에서 CUDA UUID와 Vulkan 선택을 맞추고 기존 `DRI_PRIME=0!` 고정을 제거했다. 이 단계에서 추가했던 CLI GPU 인자는 이후 위 변경에서 환경변수 방식으로 통일했다.
- Isaac Gym viewer에서 CUDA GPU 6 / graphics GPU 0 분리를 재현했다. 이 서버에서는 PCI 주소 선택이 적용되지 않아 NVIDIA ICD와 숫자 DRI_PRIME 선택을 사용한다.
- 검증: 관련 테스트 **109개 통과**. CLI `--gpu 6`과 환경변수 GPU 2를 각각 실제 viewer로 실행하여 지정 GPU에만 `C+G` 연결, 60프레임 렌더링 및 noVNC HTTP 200 확인. 모든 multi-agent train 스크립트 13개의 2048 기본값과 shell 문법도 확인했다.
- `SMOKE=1`이 남아 있는 셸에서도 GPU 6 / 2048 환경 / 본학습 batch 설정으로 2회 업데이트와 checkpoint 저장을 확인했다. 기록된 scalar 135개의 NaN/Inf 없음. 확인용 학습과 VNC는 종료했다.
- config와 잔여 코드의 삭제 후보·의존 관계를 `config.md`에 기록했다. 이 단계에서는 후보 config·기능 코드를 삭제하지 않았다.

### 문서 진입점 정리

- 루트 `AGENTS.md` 추가: 매 작업마다 최신 changelog, 레포 구조, config 요약을 확인하고 이미 읽은 내용은 재사용한다.
- 실험 가이드를 `markdowns/config.md`로 개명하고 상단에 짧은 현재 실험 요약을 추가했다. 전체 13개 config, 학습·평가·VNC 명령은 유지한다.
- `markdowns/structure.md`에 코드 역할과 작업별 최소 탐색 경로를 정리했다.
- 사용자 선택에 따라 과거 설계·분석·수정 기록 11개를 삭제했다. 임시 archive와 보관 목록도 제거하고 README·AGENTS·구조 문서에서 관련 참조를 정리했다. 현재 GTA 구조·설치·진단 문서는 유지한다.
- 삭제 후 남은 Markdown의 로컬 파일 링크 37개가 유효함을 확인했다.

### 성공 시 포화 on/off 비교

- `approach_distance_success_no_sat` 환경 YAML 및 train/test 스크립트 추가. 기존 `approach_distance_success`에서 edge saturation만 끄고 현재 At/Z 성공 시 매-step `+0.2`를 유지한다.
- reward runtime과 config 검증에서 현재 성공 보상을 포화 옵션과 독립시켰다. 기존 config와 checkpoint metadata 구조는 유지한다.
- 성공 중 원래 edge 계산, 반복 성공 보상, 성공 이탈/재진입, history 독립성, 잘못된 Z 오차 설정 및 checkpoint 불일치에 대한 테스트 추가.
- 검증: 관련 CPU 테스트 **103개 통과**, 환경 config **13개 검증 통과**. GPU 6 / 2048 환경 / 에이전트 2 / 물체 3으로 2회 학습 업데이트와 checkpoint 저장 성공. 지표 NaN/Inf 없음; 포화 지표는 0, 현재 성공 보상은 지급됨.
- 확인용 실행은 종료했으며 장기 학습 성능·수렴은 아직 검증하지 않았다. 포화 on/off는 서로 다른 reward config이므로 checkpoint를 직접 혼용하지 않는다.

### 데이터와 실행 환경

- 끊어진 dataset 심링크 12개를 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`로 재연결했다. carry YAML의 모션 경로 57개 모두 확인했다.
- 기존 포화 on 실험도 GPU 6에서 32환경 및 기본 2048환경 학습·checkpoint 저장을 확인했다.
- 원격 시각화: `run-gui.sh` + 해당 test 스크립트, `VNC_DIR="$HOME/opt/vnc"`, noVNC 6080 포워딩을 config 가이드에 문서화했다. VNC viewer 자체 실행 검증은 수행하지 않았다.
