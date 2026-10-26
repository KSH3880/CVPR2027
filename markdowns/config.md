# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번 및 SIT plane 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- `koo_1021`의 현재 GPU1 `stage1_push_finetune` 공유는 [팀 실행 가이드](../STAGE1_PUSH_FINETUNE_GUIDE.md)와 [설정 명세](stage1_push_finetune_settings.md)를 따른다. 실제학습은 CLI1200step이며 YAML기본600step과 구분한다. 복사본의 `stage1_push_rsi_uniform`과 여분상자SUPPORT 변경은 이 원본에 적용하지 않는다. 가이드에 데이터/78000전이/학습·평가·VNC 및 기존브랜치 push절차를 정리했다.

- PUSH eval 실패장면 viewer: GPU0 native·1인/상자1개·FT10000·저장된seed0/env3 초기상태로정책을다시실행한다. 아래두명령을한번에하나씩실행한다(1800step후종료). `direct`/`around`와뒤의seed/env인자를바꿀수있다. 새viewer의원본평가궤적과bitwise동일성은보장하지않으며실제GPU렌더링은이번명령준비에서미실행이다.

```bash
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo VNC_DIR="$HOME/opt/vnc" PORT=6117 RESOLUTION=1280x720x24 bash tokenhsi/scripts/multi_agent/run-gui.sh bash tokenhsi/scripts/multi_agent/view_push_eval_case.sh direct 0 3
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo VNC_DIR="$HOME/opt/vnc" PORT=6118 RESOLUTION=1280x720x24 bash tokenhsi/scripts/multi_agent/run-gui.sh bash tokenhsi/scripts/multi_agent/view_push_eval_case.sh around 0 3
```

접속포트는각각6117/6118이며 `http://localhost:6117/vnc.html?autoconnect=1&resize=remote` 형식이다. 이장면의원평가잔여거리는정면1.79m/돌아가기8.13m다. 출력은자동생성되는 `output_etc/check/push_eval_view_<조건>_seed<seed>_env<env>_<시각>`에분리저장한다.

- PUSH 정면/돌아가기 동시평가: [1인 시작배치 명세](evaluation_single_push_compare.md#정면돌아가기-시작-쌍-평가). GPU0·FT10000회·1800step·512환경×3seed, `bash output_etc/check/push_starts_ft10000_20261026/launch.sh`. tmux `koo_eval_push_starts10000`, 출력 `output_etc/check/push_starts_ft10000_20261026_run/{direct,around}/`. 실행상태는 출력 `run_manifest.json`과각조건 `status.json`, 결과는각조건 `summary.json`. 실행 중 중복 시작 금지.

- PUSH만1800step GPU0 재평가: `TOKENHSI_GPU=0 TOKENHSI_EVAL_TASKS=push_at TOKENHSI_EVAL_PUSH_EPISODE_LENGTH=1800 bash tokenhsi/scripts/multi_agent/evaluate_stage1_push_finetune.sh output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth output/stage1_base_push_finetune_eval9000_1800_20261026`. seed0/1/2·512환경·거리별집계를유지한다. tmux `koo_eval_push9000_1800`, 결과는위폴더 `RESULTS.md`/`summary.json`이다. 기존eval과병행시평가client검증/4GiB여유를조건으로 `TOKENHSI_EVAL_CONCURRENT=1`을명시한다. 현재실행중중복시작금지. [명세](evaluation_push_finetune.md#push-1800step-재평가).

- PUSH fine-tune9000회 GPU0 평가: [전용 프로토콜](evaluation_push_finetune.md). 기존4과제900step/PUSH1200step·512환경×3seed·2인/4상자 유지·점/영역ever 및 PUSH초기거리1m구간별집계다. 실행은 `TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/evaluate_stage1_push_finetune.sh`, tmux `koo_eval_push9000`, 결과는 `output/stage1_base_push_finetune_eval9000_20261026/{RESULTS.md,summary.json}`에 완료run마다 갱신한다. 현재실행중중복시작하지않는다. 1인single_match와환경/표본수가다르므로직접동등비교하지않는다.

- Stage1 78000회에서 PUSH 추가 파인튜닝: [PUSH 50% + 기존 4과제 50%](#stage1-push-finetune). 기존 과제의 RSI/모션은 원본 유지, 새 물성·방향 검사는 PUSH에 한정한다.

- 서버에서 복사한 object tokenizer `carry_chair_copied_adapter` 500epoch 로컬 resume: [전용 명령·검증 범위](#object-tokenizer-local-resume). `ports/ksh_object_finetune` 복사본·2048환경을 사용한다.

- 학습 중단 후 CUDA busy 오류와 MPS fatal 기록이 있으면 연결된 작업이 없는지 확인한 뒤 `source mps/shell.sh`, `mps_stop 0`, `mps_stop 1`, `mps_start 0`, `mps_start 1`로 사용자 MPS를 복구한다. helper가 확인된 비활성 소켓을 보관하므로 pipe 파일을 직접 삭제하지 않는다. 2026-10-13 두 GPU CUDA 전송/연산 확인.

- GPU별 MPS: `source mps/shell.sh` 후 `mps_start 1`·`mps_status 1`로 GPU1 시작/상태를 확인한다. 이 저장소 train/test/VNC는 `TOKENHSI_GPU=1` 지정 시 검증된 사용자 MPS에 자동 연결한다. 학습은 별도로 시작한다. [MPS 운영](../mps/README.md).

- 로컬 viewer/VNC 요청은 [로컬 VNC 뷰어](#local-vnc-viewer)의 현재 VRAM 옵션·검증 명령·Wayland 수정 사항부터 확인한다.

- ksh 세 방식 독립 이식: [코드 격리·checkpoint·명령](ksh_stage1_port.md). 기존 core 대신 `ports/ksh_stage1` 복사본을 실행하며 결과는 `output/ksh_*`로 분리한다.

- 5개 task의 고정 XY·크기/위치 Z 평가: [조건·명령·결과 해석](z_sweep.md). `evaluate_z_sweep.sh`는 학생 checkpoint의 headless 평가 전용이다.

- 후반 hybrid RSI: [설정·학습·평가·VNC 명령](hybrid_late_rsi.md). 같은 XYZ에서 80~95 → 70~80 → 60~70 → 50~60% 순으로 탐색하며 PD 검사와 현재 정책의 초기 rollout 관측을 결합한다.

- 학생 RSI 학습: [설정과 실행 명령](student_rsi_sampling.md). `approach_scenario_stage1_student_rsi_train.sh`의 agent별 task 비율은 HOLDING/SIT/CLIMB/ON_TOP/AT = **10/10/20/30/30%**다. ON_TOP·AT은 각각 HOLDING_ON_TOP·HOLDING_AT template이며 RSI skill 혼합과 별도다.

- 사용자 지정 조건부 AMP resume: `bash /home/hwanhee/koo_cvpr/runs/manual_resume_289953/launch_epoch7500_gpu6.sh`.
  별도 YAML의 `ampTaskConditioning: true`, `ampConditionedResume: true`로 프레임당 132차원 AMP를 복원한다.
  원본 unified 7500회 가중치/통계 전체, 초기화한 optimizer, 대상 random-binding reward/graph 계약을 사용하며 GPU 6·2048환경·총 40000회다.
  출력은 `output/shared_random_binding_resume1099002_full_amp_7500`, 로그/PID는 `../runs/manual_resume_289953/`에 저장한다. 기존 실행이 살아 있으면 중복 실행하지 않는다.

- 랜덤 크기 참조 RSI의 시간 후반/행동 단계 두 실험, distill 충돌 probe와 제한 사항:
  [reference_rsi.md](reference_rsi.md). 기준은 `unified_shared_random_binding`이며 학생 학습에 teacher를 사용하지 않는다.

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. AT goal 점도 해당 edge 담당 에이전트 색이며, goal 슬롯이 바뀌어도 배정을 따른다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용이며, 이 저장소는 심링크로 사용한다.

## 현재 실험

<a id="stage1-push-finetune"></a>

### Stage1 78000회 → PUSH 50% 파인튜닝

일반FT10000 PUSH viewer(원래학습크기/보행시작·2인/4상자/1환경·1800step)는아래명령을사용한다. GPU0 MPS가켜져있어도native viewer는MPS를우회한다. 이명령은시각화실행요청없이명령만제공한상태다.

```bash
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_EPISODE_LENGTH=1800 \
TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 \
PORT=6119 RESOLUTION=1280x720x24 OUTPUT_PATH=output/stage1_base_push_finetune_view10000_normal \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_vnc.sh \
output/stage1_base_push_finetune_stable_ep1200/Stage1PushFinetune_26-11-17-19/nn/Stage1PushFinetune_00010000.pth 2 1 4 100000
```

제로샷 XY 축소 viewer: `TOKENHSI_VIEWER_PUSH_XY_RANGE=1.2,1.6`은 X/Y를 각 .1m격자에서 독립 추출하고 Z는 원래167크기 샘플의1.45~1.70m를 보존한다. XY/Z의학습중바닥비율제약은이제로샷조건에적용하지않는다. 20kg/마찰.4, loco 시작·2인/4상자·평가1환경이다. 네상자크기는생성때배정되며매reset재추출하지않는다. GPU0에서아래명령으로실행하며기존eval병행시처리속도는변할수있다. CPU8192상자의실제asset크기분기·Z보존·문법/diff확인, 새GPU rollout은실행하지않았다.

```bash
env -u TOKENHSI_VIEWER_PUSH_XY_RANDOM -u TOKENHSI_VIEWER_PUSH_CUBE_SIDE \
  -u TOKENHSI_VIEWER_PUSH_CUBE_SIDES -u TOKENHSI_VIEWER_PUSH_BOX_RANGES \
  TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
  TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_PUSH_CUBES=1 \
  TOKENHSI_VIEWER_PUSH_XY_RANGE=1.2,1.6 TOKENHSI_VIEWER_PUSH_CUBE_FRICTION=0.4 \
  TOKENHSI_VIEWER_EPISODE_LENGTH=1800 TOKENHSI_VIEWER_RENDER_EVERY=2 \
  TOKENHSI_VIEWER_DRAW_SYNC=0 RESOLUTION=1280x720x24 PORT=6116 \
  OUTPUT_PATH=output/stage1_base_push_finetune_zeroshot_xy12to16 \
  bash tokenhsi/scripts/multi_agent/stage1_push_finetune_vnc.sh \
  output/stage1_base_push_finetune_stable_ep1200/Stage1PushFinetune_26-11-17-19/nn/Stage1PushFinetune_00010000.pth 2 1 4 100000
```

접속: `http://localhost:6116/vnc.html?autoconnect=1&resize=remote`.

학습1200step 전환은 `TOKENHSI_TRAIN_EPISODE_LENGTH=1200`으로 지정한다. 사용자 승인으로 `koo_train:0`에서 전용launcher `bash output/stage1_base_push_finetune_stable_ep1200/launch.sh`를 실행해 GPU1 MPS·2048환경·9000회에서 재개했다. 새run은 `output/stage1_base_push_finetune_stable_ep1200/Stage1PushFinetune_26-11-17-19`, 로그는 상위폴더 `train.log`다. 저장config episodeLength1200·epoch9001→9002·새에피소드지표37개 실제기록을확인했다. 실행중중복시작하지않는다. 다른저장본은 명령앞에 `RESUME_CHECKPOINT=<저장본>`을지정한다. 기존600step학습과동일lock을사용해중복실행을거부한다. 시각화길이는별도로아래옵션을지정한다. episodeLength는모델/PUSH checkpoint계약에포함되지않으며 기존optimizer/epoch/frame을복원한다. 라이브simulator/AMP replay는복원하지않는다.

상자→목표거리에는최소1m만있고별도최대거리설정은없다. 환경중심목표XY±4.5m·상자정면±45°·경로여유를만족하는목표를샘플링한다. seed42 CPU2048scene/4096상자의RSI전실측은평균7.824m·범위1.000~19.088m다(이최댓값은표본값). 사람→상자출발거리1~10m와다르며 RSI시에는이미진행한거리만큼실제남은거리가짧아진다.

시각화 에피소드 길이는 `TOKENHSI_VIEWER_EPISODE_LENGTH=1200`으로 평가/test/VNC에만 덮어쓴다. 학습YAML 기본은600step이며 현재학습은 위launcher의 CLI로1200step을적용한다. 예시(실행 전 GPU점유확인, 현재GPU1학습과분리해GPU0선택):

```bash
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_EPISODE_LENGTH=1200 \
TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 \
RESOLUTION=1280x720x24 PORT=6108 OUTPUT_PATH=output/stage1_base_push_finetune_view9000_1200 \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_vnc.sh \
output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth 2 1 4 100000
```

기존6108뷰어가 실행 중이면 그 뷰어 터미널에서 종료한 뒤 실행한다. 학습 tmux는 유지한다. 1200은최대길이이며 넘어짐 등 조기종료는 유지된다. 옵션 전달/CLI config 덮어쓰기 확인, 새GPU viewer실행은 하지 않았다.

2026-10-26 사용자 승인으로 `koo_train:0`에서9000회 저장본을 재개했다. 새 run은 `output/stage1_base_push_finetune_stable/Stage1PushFinetune_26-10-06-23`, 실행파일은 같은 상위폴더 `resume9000_20261026.sh`, 로그는 `resume9000_20261026.log`다. GPU1 MPS client·2048환경·9001→9002 및 아래 지표의 실제 TensorBoard 기록을 확인했다. 실행 중 중복 시작하지 않는다. 해당 실행파일은9000회부터 재시작하는 고정 명령이며 향후 최신상태 resume에는 `RESUME_CHECKPOINT=<새run의최신저장본>`을 사용한다.

2026-10-26 추가한 에피소드 기록은 **코드를 새로 로드한 학습/재개부터** TensorBoard `relation/episode/<과제>/`에 남는다. 기존 실행에 자동 반영되지 않으며 기록 전 구간을 복원하지 않는다. 과제는 `sit`, `climb`, `holding_at`, `holding_ontop`, `push_at`이고 `completed_agents`(완료된 agent 에피소드 수), `final_success`(종료 시 성공률), `ever_success`(초기 RSI 상태를 포함해 한 번이라도 성공한 비율)를 기록한다. 성공 조건은 기존 required-goal 판정을 그대로 쓰며 보상·정책·checkpoint 계약을 바꾸지 않는다. 여러 iteration을 합칠 때 성공률은 `completed_agents`로 가중한다. 완료 표본이 없으면 count만0으로 기록하고 성공률은 생략한다.

PUSH에는 `start_distance_m`, `remaining_distance_m`, `distance_reduction_m`, `progress_fraction`, `best_progress_fraction`도 기록한다. 모두 완료된 PUSH agent 에피소드의 평균이며 XY 상자→고정목표 거리 기준이다. 진행률은 `clamp((초기거리-잔여거리)/초기거리,0,1)`이고 최대진행률은 에피소드 중 최대값이다. 초기거리는 **RSI 적용 후**이므로 RSI로 미리 이동한 거리를 정책 실적으로 세지 않는다. 거리감소는 멀어지면 음수가 된다. `push_at_loco`와 `push_at_rsi`로 시작모드별 성공률·진행도도 분리한다. 기존 `relation/edge/*/own_success`는 스텝 상태 비율로, 위 에피소드 성공률과 다르다.

전용 env/train은 `ports/stage1_base_push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_finetune.yaml`이다. PUSH 50%, SIT 5%, CLIMB 12.5%, AT/ON_TOP 각16.25%로 기존 네 과제의 상대 비율을 유지한다. 기존 과제의 보상·RSI 확률/선별 및 원본 모션 묶음을 유지하고 `dataset_stage1_push_finetune.yaml`에 PUSH21개와 PUSH 전용 보행4개(`push_loco`)만 추가한다. 과거 PUSH100%/12000회 실행과 출력·checkpoint 계약을 분리한다.

PUSH 출발은 담당 사람으로부터1~10m, 목표는 환경 중심 XY각±4.5m 정사각형 안에서 사람/초기상자와각1m 이상·상자 로컬+Y 기준±45°를 동시에 만족하도록 거부 샘플링한다. 전체 경로/다른 사람과의 여유도 검사하므로 수용 분포는 완전 균등하지 않다. X/Y각1.60~2.20m(0.1m격자)·Z1.45~1.70m(0.05m격자), min(X,Y)≥1.1×Z인167종·크기무관20kg·마찰0.4이며 **기존 과제의 작은 상자 크기/물성은 원본 그대로**다. CONTACT state/progress/success각0.2(합0.6), 기존 AT각0.2를 사용한다. 바닥마찰1.0·600step·2048환경이다.

PUSH RSI는 승인된clip0/3/4/8의949후보·원본DOF/속도·loco/PUSH50/50·각클립후반30%를70% 확률로 요청한다. clip7만 제외하며 추가발2cm필터는 없다. 최초 목표는 고정하고 상자만 `initial + box_progress*(goal-initial)`로 옮긴다. 따라서 초기10m·RSI4m 진행이면 **6m가 남고 목표를 새로10m 앞에 두지 않는다**. 새bank는 `ports/stage1_base_push/tokenhsi/data/stage1_push_finetune_stable_rsi_bank.json`이며 새167크기/양담당자0.1초CPU PD316,966상태 및 ±45°경계/정면 기하475,449배치를 검사를 모두 통과했다. 실제 결과는 아래 보고서에 기록한다. 경로간격은3.5m로 늘렸다. 장기균형/학습성공 검증은 아니다.

원본 `ports/stage1_base_push/weights/Stage1V2Fixed_00078000.pth`에서 전이한 저장본은 `output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.pth`다. 모델162개텐서·기존과제embedding4행·기존AMP입력/정규화를 보존하고, 새PUSH embedding은carry_at행·새AMP family열/RMS는carry에서 초기화한다. optimizer 누적상태·epoch/frame·best reward는 초기화하며 live simulator/replay는 복원하지 않는다. 원본SHA는 전이JSON에 기록한다. 사용자 요청으로 GPU 확인용 학습/본학습은 실행하지 않고 명령만 제공한다. CPU 검사와 RSI 영상 생성만 수행한다. 과거18종/마찰0.5 bank·전이본·영상은 보존하며 현재실험과 계약이 다르다.

```bash
# PUSH만 CPU 기하/초기물리 검사 및 RSI 영상 재생성; 기존과제 RSI 재검사는 하지 않는다.
bash tokenhsi/scripts/multi_agent/check_stage1_push_finetune.sh
bash tokenhsi/scripts/multi_agent/review_stage1_push_finetune.sh
bash tokenhsi/scripts/multi_agent/check_stage1_push_finetune_player.sh
# 본학습 명령. 실행 상태/로그는 아래 보고서를 확인하고 중복 실행하지 않는다.
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=42 bash tokenhsi/scripts/multi_agent/stage1_push_finetune_train.sh 2 2048 4
# 확인용: 별도출력/2048환경, 전이본epoch0에서2회
TOKENHSI_GPU=1 OUTPUT_PATH=output/stage1_base_push_finetune_stable_check_train MAX_ITERATIONS=2 bash tokenhsi/scripts/multi_agent/stage1_push_finetune_train.sh 2 2048 4
# 로컬 추론 / 화면 없는 평가 / 서버 VNC (실행 전 로컬GPU 보호규칙 확인)
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_finetune_test.sh output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.pth 2 1 4 10
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage1_push_finetune_test.sh output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.pth 2 1 4 10
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6108 bash tokenhsi/scripts/multi_agent/stage1_push_finetune_vnc.sh output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.pth 2 1 4 10
```

학습재개는 `RESUME_CHECKPOINT=<이실험저장본>`으로 지정한다. 평가 기본은PUSH/loco시작이며 `TASK_GRAPH=random_scenario`로 혼합과제 평가를 선택할 수 있다. [PUSH RSI 영상](../output_etc/check/stage1_push_finetune_stable/push_rsi_candidates.mp4)·[검사보고서](../output_etc/check/stage1_push_finetune_stable/REPORT.md). 영상은CPU 초기자세열이며 정책/물리rollout이 아니다. 원본Stage1 pth·기존PUSH 저장본은 직접resume하지 않는다. 전이도구 `prepare_finetune.py`는 port cwd/PYTHONPATH에서 `bank`(최초1회), `transfer`(검증된bank필수)를 제공하며 기존파일을 덮어쓰지 않는다.

<a id="stage1-base-push"></a>

### 원본 Stage 1 네트워크·4상자 + PUSH_AT

`ports/stage1_base_push/`는 GitHub `stage1_stage2_base`의 `stage1_v2_fixed.yaml`과 task-shared embedding·Action MLP·GTA를 보존한 별도 실행본이다. 기존 SIT/CLIMB/carry_at/carry_ontop을 남기고 새 `PUSH_AT` 과제만100% 샘플링한다. 기존 네 과제 ID는0~3 유지, PUSH_AT은4로 추가하여 embedding이4→5행으로 늘어난다. AMP는 기존3family를 유지하고 별도 PUSH family3을 추가한다. 사람2·상자4·8토큰·정책634D·AMP1330D·2048환경이며 Stage2/coupled는 제외한다.

AT은 원본 state/progress/success각0.2·성공거리20cm/Z오차1mm·self_sum/본인100%를 사용한다. CONTACT는 현재 PUSH의 유한 면 양손 XY/반력 조건·접근 progress·0.4/0.2/0.2를 적용한다. 네 상자는 현재Y2m/X1~1.5m/Z1.3m·각20kg·마찰0.6이며 추가2개도 실제 actor/관측에 포함하고 경로 밖에 배치한다. loco/PUSH50/50·승인949후보·70%후반 RSI를 유지한다. [구현·차이·제한](../ports/stage1_base_push/README.md), [CPU 검사](../output/stage1_base_push_check/REPORT.md).

```bash
# CPU 회귀 및 실제 CPU reset/물리/player/checkpoint 연결 검사
WITH_SIM=1 bash ports/stage1_base_push/verify.sh
# 본학습: 현재 GPU1 학습을 자동으로 종료하지 않으며, 점유 중이면 시작을 거부한다.
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo OUTPUT_PATH=output/stage1_base_push bash tokenhsi/scripts/multi_agent/stage1_base_push_train.sh 2 2048 4
# 로컬 추론 / 화면 없는 평가 / 서버 VNC: 새 실험에서 학습한 저장본 사용
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_base_push_test.sh <Stage1BasePush.pth> 2 1 4 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage1_base_push_test.sh <Stage1BasePush.pth> 2 1 4 10
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6107 bash tokenhsi/scripts/multi_agent/stage1_base_push_vnc.sh <Stage1BasePush.pth> 2 1 4 10
```

RSI [검수 영상](../output_etc/check/stage1_base_push_rsi_review/stage1_push_rsi_review.mp4)·[보고서](../output_etc/check/stage1_base_push_rsi_review/REPORT.md): 실제4상자 reset8장면+PUSH949후보, CPU 초기자세55.6초다. 사용자 검수 대기이며 정책 rollout 영상은 아니다. 재생성: `bash output_etc/check/stage1_base_push_rsi_review/launch.sh`.

VNC 접속 `http://localhost:6107/vnc.html?autoconnect=1&resize=remote`. 2026-10-22 실제5500회 정책 viewer 실행은 `bash output/stage1_base_push_view_20261022/launch.sh`이며 tmux `koo_stage1_push_view_20261022`가 살아 있으면 중복 실행하지 않는다. GPU1 native/MPS 우회·저메모리1환경·loco 시작·1280×720이며 실제 정책 step/화면 및 실행 후 기존학습 진행을 확인했다. 로그/캡처는 같은 폴더 `viewer.log`/`preview.png`다. 관측 시 GPU1 여유523MiB·학습FPS 감소가 있었고 장기 병행 안전성을 보장하지 않는다.

12000회 정육면체 viewer: `bash output/stage1_base_push_cube_view_20261023/launch.sh`. GPU1 native·저메모리1환경·loco 시작·VNC6109이며 `TOKENHSI_VIEWER_PUSH_CUBES=1`로 실제 asset/크기관측/접촉 크기를 X=Y=Z, 한 변1.0~1.5m/0.1m격자6종에서 생성 시 배정한다(episode reset마다 크기가 바뀌지는 않는다). 네 상자 모두20kg/마찰0.6을 유지한다. 학습 설정/저장본 계약은 보존하며 native GUI/test·PUSH_AT/loco·16환경 이하만 허용한다. 크기가 바뀐 interaction RSI는 사용하지 않는다. tmux `koo_stage1_cube_view_20261023`, 로그는 같은 폴더 `viewer.log`. 접속: `http://localhost:6109/vnc.html?autoconnect=1&resize=remote`.

정육면체 크기/마찰 고정은 같은 viewer 전용 옵션 `TOKENHSI_VIEWER_PUSH_CUBE_SIDE=1.5 TOKENHSI_VIEWER_PUSH_CUBE_FRICTION=0.4`다. `TOKENHSI_VIEWER_PUSH_CUBES=1`과 함께 사용하며 네 상자 모두1.5m/20kg/마찰0.4로 만든다. 크기1~1.5m·유한한 비음수 마찰만 허용한다. 바닥마찰은 바꾸지 않는다. 예시(사용자 직접 실행, 기존 프로세스 종료 없음):

XY 독립 가변은 여기에 `TOKENHSI_VIEWER_PUSH_XY_RANDOM=1`을 추가한다. 네 상자의 X/Y를 각각1.0~1.5m/0.1m격자에서 생성 시 독립 배정하고 Z는 `TOKENHSI_VIEWER_PUSH_CUBE_SIDE` 값으로 고정한다. 크기는 reset마다 변경되지 않는다. 아래 명령에 이 옵션을 추가하고 `PORT=6112 OUTPUT_PATH=output/stage1_base_push_xy_friction04_view`로 실행하면 Z1.5m/20kg/상자마찰0.4의 별도 viewer다.

정육면체 후보크기 제한은 `TOKENHSI_VIEWER_PUSH_CUBE_SIDES=1.3,1.4,1.5`다. `TOKENHSI_VIEWER_PUSH_CUBES=1`과 함께 사용하고 `TOKENHSI_VIEWER_PUSH_XY_RANDOM=0`, `TOKENHSI_VIEWER_PUSH_CUBE_SIDE`는 unset해야 한다. 각 상자XYZ를 같은 값으로 배정한다. 아래 명령의 고정SIDE를 이 SIDES로 바꾸고 `PORT=6114 OUTPUT_PATH=output/stage1_base_push_cubes13to15_friction04_view`로 실행한다(20kg/상자마찰0.4 유지).

직육면체 범위·Y≥Z viewer는 다음과 같다. X1.2~1.7m, Y/Z각1.45~1.7m의 **연속값**으로 생성 시 배정하고 Y/Z를 큰값/작은값 순서로 놓아 Y≥Z를 보장한다(정육면체일 필요 없음). `BOX_RANGES` 순서는 xmin,xmax,ymin,ymax,zmin,zmax이며 현재Y/Z범위가같은 경우만 지원한다. 실제크기/질량/상자마찰은로그에 기록한다. 네 상자20kg/마찰0.4 유지, scene배치검사는 최대1.7m 크기로 수행하고 source YAML/bank는 변경하지 않는다. CPU10000상자범위/Y≥Z/20kg·최대크기2048scene 검사통과, 새GPU rollout은 미검증이다.

```bash
env -u TOKENHSI_VIEWER_PUSH_CUBE_SIDE -u TOKENHSI_VIEWER_PUSH_CUBE_SIDES \
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_PUSH_CUBES=1 TOKENHSI_VIEWER_PUSH_XY_RANDOM=0 \
TOKENHSI_VIEWER_PUSH_BOX_RANGES=1.2,1.7,1.45,1.7,1.45,1.7 \
TOKENHSI_VIEWER_PUSH_CUBE_FRICTION=0.4 \
TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 \
RESOLUTION=1280x720x24 PORT=6115 OUTPUT_PATH=output/stage1_base_push_ygez_view \
bash tokenhsi/scripts/multi_agent/stage1_base_push_vnc.sh \
output/stage1_base_push/Stage1BasePush_21-15-46-33/nn/Stage1BasePush_00012000.pth 2 1 4 100000
```

`BOX_RANGES`는 기존SIDE/SIDES/XY_RANDOM과동시사용하지않는다. 이전정육면체명령으로돌아갈때는 `TOKENHSI_VIEWER_PUSH_BOX_RANGES`를 unset한다.

```bash
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_PUSH_CUBES=1 \
TOKENHSI_VIEWER_PUSH_CUBE_SIDE=1.5 TOKENHSI_VIEWER_PUSH_CUBE_FRICTION=0.4 \
TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 \
RESOLUTION=1280x720x24 PORT=6111 \
OUTPUT_PATH=output/stage1_base_push_cube15_friction04_view \
bash tokenhsi/scripts/multi_agent/stage1_base_push_vnc.sh \
output/stage1_base_push/Stage1BasePush_21-15-46-33/nn/Stage1BasePush_00012000.pth 2 1 4 100000
```

짧은 학습은 별도 `OUTPUT_PATH=output/stage1_base_push_check_train MAX_ITERATIONS=2`와2048환경을 사용한다. 기존570D/6000회 PUSH 및 원본4과제 checkpoint 직접사용은 거부한다. 장기성능은 별도 평가하며 GPU 실행 전에 [로컬 보호 규칙](#local-vnc-viewer)을 따른다.

<a id="push-native-y2m"></a>

### PUSH Y길이2m·5000회 저장본 resume

`push_native_phase_y2m`은 기존 GPU1 `push_native_phase_xy_forward_2to5m_no_clip7`의5000회 저장본을 이어받는 별도 실험이다. **상자로컬Y=2m 고정**, X1~1.5m/0.1m격자6종·Z1.3m·20kg·마찰0.6이다. 실제 asset/관측/CONTACT 면 거리에 새크기를 적용하고 경로간격은2.8m다. 정면2~5m 목표·출발1~2m·원본DOF·clip7제외949후보·loco/PUSH50/50·후반70%/잔존30%·CONTACT+AT soft tilt·AMP21+4·570D·600step·2048환경은 유지한다. 기존 학습 config/bank/checkpoint는 변경하지 않는다.

전용bank `push_native_phase_y2m_rsi_bank.json`은 새크기로 CPU기하5,694배치·초기0.1초PD11,388상태(949후보×6종×2담당자)를 다시 통과했다. 0.4초고정PD 대표검사는48/144통과로 장기균형을 보장하지 않는다. 추가발2cm필터는 적용하지 않는다. CPU1000agent reset·담당자교환·부분reset·물리24step·AMP·Actor/Critic/Discriminator strict 복원을 확인했다. [검사보고서](../output_etc/check/push_native_phase_y2m/REPORT.md), [RSI초기자세 영상](../output_etc/check/push_native_phase_y2m/rsi_candidates_y2m.mp4). 영상은 CPU자세열이며 정책/물리rollout 녹화는 아니다.

전이본은 `output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth`다. 모델·모든RMS·optimizer168개상태·epoch5000/frame655360000·최고보상을 원본과 tensor단위 동일하게 보존했다. 새geometry/bank의 CPU검사 완료를 확인한 뒤 물리/RSI 계약과 저장config만 명시적으로 변경한다. live simulator/rollout/replay는 복원하지 않는다. 원본 checkpoint 직접 resume는 계약차이로 거부된다. 전이도구는 `transfer_push_native_y2m_checkpoint.py`이며 기존전이본을 덮어쓰지 않는다.

```bash
# 초기 물리/실행 연결 CPU 재검사: GPU/MPS 미연결
bash tokenhsi/scripts/multi_agent/check_native_push_y2m.sh
TOKENHSI_CHECKPOINT=output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth OUTPUT_PATH=output/push_native_phase_y2m_check/connection bash tokenhsi/scripts/multi_agent/check_push_native_y2m_scene.sh
# 본학습 resume: GPU1의 기존학습/viewer를 사용자가 종료한 뒤 실행
env -u MAX_ITERATIONS TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo \
RESUME_CHECKPOINT=output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth \
OUTPUT_PATH=output/push_native_phase_y2m \
bash tokenhsi/scripts/multi_agent/push_native_phase_y2m_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_native_phase_y2m_test.sh output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_native_phase_y2m_test.sh output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6106 bash tokenhsi/scripts/multi_agent/push_native_phase_y2m_vnc.sh output/push_native_phase_y2m/resume/PushNativePhaseY2m_from5000.pth 2 1 2 10
```

새 train wrapper는 지정GPU에Python CUDA client가 있으면 학습 시작 전에 거부하고 어떤프로세스도 종료하지 않는다. 학습2048환경은 유지한다. 짧은 학습 확인이 필요하면 `OUTPUT_PATH=output/push_native_phase_y2m_check_train MAX_ITERATIONS=5002`로 본학습과 분리한다(총epoch 상한이며 추가2회). GPU학습/PPO와 새크기의GPU viewer는 아직 검증하지 않았다. VNC 실행 전 [로컬GPU 보호규칙](#local-vnc-viewer)을 따른다.

<a id="push-native-phase-xy"></a>

### PUSH 원본 자세 RSI·XY 가변 상자

`push_native_phase_xy`의 현재 목표는 **RSI 적용 전 상자 초기 중심에서 박스 로컬+Y 정면으로만2~5m**이다. 출발1~2m·방향/배치/담당자 랜덤, loco/PUSH50/50·후반70%/잔존30%, 기존CONTACT+AT soft tilt 보상·600step·2H/2Box·2048환경을 유지한다. 상자X/Y는 각각독립1~1.5m/0.1m격자(36종), Z1.3m·20kg·마찰0.6이며 경로간격2.5m다.

RSI는 사용자가 확인한 MD 기준1,196후보에서 사용자검수로clip7전체247후보만 제외한 **clip0/3/4/8·949프레임**이다. 원본관절/속도를 보존하며 추가발2cm 조건은 적용하지 않으며 나머지949후보는 원본MD선별본과동일하다. 사용자가949후보영상을승인했으며학습실행은별도명령으로한다. AMP PUSH21개/보행4개는 유지한다. 현재bank는 `push_native_phase_xy_forward_2to5m_no_clip7_rsi_bank.json`이다. 3~5m 원본bank와 추가접지51후보bank는 과거검수용으로 보존하며 직접학습에 섞지 않는다. 새bank는 모션/asset/mesh/물성/보상/샘플링 계약을 검증한다. 기존저장본 직접resume는 변경된계약으로 거부되며 새scratch 실험이다.

[현재 직선목표/clip7제외 검수](../output_etc/check/push_native_phase_xy_no_clip7_review/REPORT.md). [검수용949후보영상](../output_etc/check/push_native_phase_xy_no_clip7_review/rsi_candidates_no_clip7.mp4). [이전MD RSI 영상](../output_etc/check/push_native_phase_xy/md_rsi_candidates_only.mp4)은clip7을포함한과거1,196후보의자세열이다. 재현: `CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/render_native_push_rsi.py --bank tokenhsi/data/push_native_phase_xy_rsi_bank.json --output output_etc/check/push_native_phase_xy/md_rsi_candidates_only.mp4 --candidates-only --expected-candidates 1196 --fps 30 --repeat 1`. 이 영상은 물리rollout이 아니다. 0.1초PD 검사는 장기균형/정책성공 검사가 아니며, 0.4초고정PD의 절대골반높이0.65m 기준 미달도 실제넘어짐 판정이 아니다. 학습/PPO/GPU 실행은 아직 하지 않았다.

목표 생성기`sampling: box_forward`는 초기박스의yaw를따른+Y방향만허용하며, 반대/옆/대각선목표를검사에서거부한다. 월드방향은박스랜덤회전을따라달라지며 각에피소드의목표점은고정된다. 후반RSI는같은직선상으로상자를진행시킨다. 다음 명령의 checkpoint는 이 설정으로 생성한 저장본으로 바꾼다. 실행 전 현재점유/PID를 확인하고 VNC는 [로컬GPU 보호규칙](#local-vnc-viewer)을 따른다.

```bash
# CPU 초기 PD 재검사(학습 없음)
OUTPUT_PATH=output/push_native_phase_xy_forward_2to5m_no_clip7_check bash tokenhsi/scripts/multi_agent/check_native_push_phase.sh
# 짧은 학습 검증: 환경 수2048 유지, 본학습과 출력 분리
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo OUTPUT_PATH=output/push_native_phase_xy_forward_2to5m_no_clip7_check_train MAX_ITERATIONS=2 bash tokenhsi/scripts/multi_agent/push_native_phase_xy_train.sh 2 2048 2
# 승인된949후보/정면2~5m scratch 본학습(직접실행하지않음)
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo OUTPUT_PATH=output/push_native_phase_xy_forward_2to5m_no_clip7 bash tokenhsi/scripts/multi_agent/push_native_phase_xy_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_native_phase_xy_test.sh <새실험.pth> 2 1 2 10
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_native_phase_xy_test.sh <새실험.pth> 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 bash tokenhsi/scripts/multi_agent/push_native_phase_xy_vnc.sh <새실험.pth> 2 1 2 10
```

평가는 기본loco 시작이며 RSI 직접 검수는 CPU 검사/영상을 사용한다. 새 bank는 source motion·asset/발mesh·물성·목표/보상 계약을 검사하며 물성/크기/선별이 바뀌면 다시 만들어야 한다.

기존 checkpoint로 미는 방향(상자 로컬Y)만2m로 늘려 보는 **평가 전용** 옵션은 `TOKENHSI_VIEWER_PUSH_LENGTH_Y=2`다. X1~1.5m/높이1.3m/질량20kg/마찰0.6은 유지한다. 실제 asset·관측 bbox·접촉/보상 크기 tensor와 장면 clearance에 적용한다. 경로간격은 커진 상자의 대각선+0.3m 이상(2m일 때2.8m)으로 늘린다. source YAML/RSI/checkpoint 계약은 원본 그대로 검사하고 실행 로그에 실제 viewer 크기를 별도 표시한다. 학습·headless·16환경 초과·interaction RSI·native 이외 모드는 거부한다. 이 크기의 physics rollout/GUI는 미검증이며 CPU검사만 통과했다.

```bash
env -u TOKENHSI_EVAL_TRAIN_RSI -u EPISODE_LENGTH \
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_PUSH_LENGTH_Y=2 TOKENHSI_VIEWER_LOW_VRAM=1 \
TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 TOKENHSI_VIEWER_QUIET=1 \
RESOLUTION=1280x720x24 PORT=6105 EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
OUTPUT_PATH=output/push_native_phase_xy_view_5000_y2m \
bash tokenhsi/scripts/multi_agent/push_native_phase_xy_vnc.sh \
output/push_native_phase_xy_forward_2to5m_no_clip7/PushNativePhaseXY_20-19-46-57/nn/PushNativePhaseXY_00005000.pth 2 1 2 10
```

접속 `http://localhost:6105/vnc.html?autoconnect=1&resize=remote`. 실행 전 현재GPU/학습PID/MPS를 확인하며 [로컬GPU 보호규칙](#local-vnc-viewer)을 따른다. 기본크기 영상과 혼동하지 않도록 output/포트를 분리한다.


<a id="push-distance-eval"></a>

### 3~5m 정책 loco 시작 거리별 평가

- 실행: `bash output/push_distance_eval_20261019/launch.sh`. 기존 `koo_push_distance_eval_20261019` tmux 또는 동일 평가 프로세스가 살아 있으면 중복 실행하지 않는다. GPU1·native/MPS 우회·`tokenhsi_koo`·저메모리1환경·2agent/2box·23000epoch·loco100%·seed20261019·episode600·50scene 목표/최대1시간이다. 검증된 native 경로 때문에1280×720 가상 화면은 유지하며 `TOKENHSI_VIEWER_RENDER_EVERY=1000000`으로 렌더를 줄인다. headless 스크립트 직접 실행은 MPS 자동 연결 경로이므로 대체하지 않는다. 실행 전후 학습 PID/iteration·로그/MPS 확인은 [로컬 규칙](#local-vnc-viewer)을 따른다.
- 로그 `output/push_distance_eval_20261019/eval.log`, 완료 표본 `metrics/distance_completed_results.json`, 집계 `metrics/distance_summary.json`(모두 해당 output 아래). `TOKENHSI_PUSH_DISTANCE_EVAL=1`은 실제 초기 상자→목표 XY 거리3~3.5/3.5~4/4~4.5/4.5~5m별 최초/종료 AT 성공, 최초 도달시간 및 종료 목표거리를 집계한다. 사람→상자 초기 접근거리도 보존하며 interaction RSI 시작은 거부한다. 완료 repeat마다 원자적으로 저장하여 timeout 시에도 완료분을 보존한다. 두 agent는 같은 scene을 공유하므로 독립 표본으로 간주하지 않는다. 학습 step 성공 비율과 별개인 첫 episode 평가다.
- 2026-10-19 12:00:39 전체50scene/100agent 완료·평가 종료. 최초/종료AT32%, 거리순27.3/37.5/39.3/23.1%, 두 agent 동시5/50scene다. 실패68건 중66건은 목표거리 감소0.5m 미만이다. [분석·한계](../output/push_distance_eval_20261019/REPORT.md) 참조. 종료 후 두 기존 학습epoch 진행·GPU1 MPS 로그 추가0byte를 확인했다.

같은23000저장본을 GPU1에서 loco 시작으로 시각화한다. 실행 전 학습/평가 PID·GPU 여유·포트6103을 다시 확인하고 [로컬 보호 규칙](#local-vnc-viewer)을 따른다. 별도 output이며 거리평가/RSI 관련 상속 옵션은 제거한다.

```bash
env -u TOKENHSI_EVAL_TRAIN_RSI -u TOKENHSI_PUSH_DISTANCE_EVAL -u TOKENHSI_PUSH_FACE_AUDIT -u TOKENHSI_PUSH_EVAL_AUDIT -u EPISODE_LENGTH \
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 TOKENHSI_VIEWER_QUIET=1 \
RESOLUTION=1280x720x24 PORT=6103 EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 SEED=20261019 \
OUTPUT_PATH=output/push_distance_view_20261019 \
bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_3to5m_vnc.sh \
output/push_contact_face_at_tilt_soft_3to5m/PushContactFaceATTiltSoft3to5m_18-18-25-28/nn/PushContactFaceATTiltSoft3to5m_00023000.pth 2 1 2 10
```

접속: `http://localhost:6103/vnc.html?autoconnect=1&resize=remote`.

### PUSH 면별 저메모리 평가

- 공통 면 AT soft tilt10000epoch의 같은 면/다른 면·기울기 평가는 `bash output/push_face_tilt_eval_20261017/launch.sh`다. GPU1 native·MPS 우회·저메모리1환경·1280×720·포트6102·보행 시작·seed20261017·40회 직렬/최대1시간이며 실행 전 학습 PID/로그/MPS·점유와 동일 세션 중복 여부를 확인한다. 로그/결과는 해당 output의 `eval.log`, `metrics/`에 쓴다. 집계: `python tokenhsi/scripts/multi_agent/summarize_push_faces.py output/push_face_tilt_eval_20261017/eval.log`.
- 2026-10-17 실행은1시간 timeout으로37scene/74agent 완료 후 종료됐으며 최종 metrics 저장 대신 완료 로그를 `completed_results.json`으로 복원했다. 최초/종료 AT5/74, 첫 성공 전 이동의 힘 proxy는 같은 면9333step·평균7.61도/10도 초과25.85%, 다른 면18step·평균0.41도다. 양면 주사용 episode가 없어 양면 성공 우월성은 미확인이다. [보고서](../output/push_face_tilt_eval_20261017/REPORT.md)와 `summary.json` 참조. 전후 기존 두 학습 진행·MPS 로그 추가0byte를 확인했다.
- opt-in 면 계측은 기존 XY proxy와 별도로 full quaternion OBB의 유한3D 측면18cm/모서리2cm tie·상하면 제외 분류를 기록한다. force proxy는 각 손의 해당 면 바깥 법선 방향 net force≥0.5N 및 박스 수평 net force≥0.5N을 추가한다. control step 경계 표본이며 정확한 충돌 쌍/physics tick OR가 아니다. 첫 AT 성공 전·속도0.02m/s 초과·현재 미성공 구간을 비교하고, 이 구간에서 한 분류가15step 이상/60% 이상이면 episode 주사용 면으로 분류한다(나머지는 mixed/insufficient). 성공 후 손 떼기를 분류 근거에서 제외하며 초기 목표 기하별로도 집계한다. 관측적 비교이며 같은 초기 상태의 강제 접촉 방식 인과 실험은 아니다.

- GPU1의30D·AT0.1m 저장본12500을 대상으로 `bash output/push_face_eval_20261016/launch.sh`를 사용한다. 동일 프로세스가 살아 있으면 중복 실행하지 않는다. GPU1 native·저메모리1환경·1280×720·보행 시작·seed20261016·40회 직렬 평가(최대1시간)이며, 학습 MPS 우회는 위 로컬 VNC 규칙을 따른다. 실행 전 GPU 점유/학습 PID를 확인한다.
- `TOKENHSI_PUSH_FACE_AUDIT=1`은 초기 목표 반대 ray가 면 중앙/모서리에 도달하는 경우와 매step 양손의 같은 면/다른 면/모서리 모호/근접 실패를 CPU에서 기록한다. 손 배치는 XY 근접 proxy이며 실제 접촉·힘 판정이나 같은 시작 상태의 인과 비교가 아니다. 학습/보상은 바꾸지 않는다.
- 결과는 `output/push_face_eval_20261016/{eval.log,metrics/}`. 완료 repeat만 중간 집계하려면 `python tokenhsi/scripts/multi_agent/summarize_push_faces.py output/push_face_eval_20261016/eval.log`; 최종 JSON도 같은 도구로 집계한다. 초기 기하 구간별 표본 수와 AT 최초/종료 성공률을 함께 해석한다. 새 train config가 아닌 평가 계측이다.

| 번호 | Config | 역할 |
| ---: | --- | --- |
| PUSH native Y2m resume | [push_native_phase_y2m.yaml](../tokenhsi/data/cfg/multi_agent/push_native_phase_y2m.yaml) | Y2m/X1~1.5m·949후보 재검사·5000회 모델/optimizer 유지 resume·GPU학습 미실행 |
| PUSH native phase XY | [push_native_phase_xy.yaml](../tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml) | 원본 관절·실제 밀기 구간 RSI/상자XY1~1.5m·Z1.3m/별도 scratch·GPU 학습 미실행 |
| PUSH common face soft tilt 3~5m | [push_contact_face_at_tilt_soft_3to5m.yaml](../tokenhsi/data/cfg/multi_agent/push_contact_face_at_tilt_soft_3to5m.yaml) | 상자 초기 중심 기준3~5m 랜덤 목표·기존 RSI/보상 유지·별도 전이/출력·본학습 미실행 |
| PUSH common face AT soft tilt | [push_contact_face_at_tilt_soft.yaml](../tokenhsi/data/cfg/multi_agent/push_contact_face_at_tilt_soft.yaml) | 고정1.3m CONTACT 유지·AT state/progress5~15도 감쇠·성공10도 이하·8000 저장본 명시적 전이 |
| PUSH common face CONTACT+AT | [push_contact_face_at.yaml](../tokenhsi/data/cfg/multi_agent/push_contact_face_at.yaml) | GPU1용 새 scratch·고정1.3m/20kg·60도 공통 면 선택·최대 손거리 state>0.9+힘·AT 기울기 감쇠 없음 |
| PUSH soft legacy40 | [push_soft_legacy40_at05.yaml](../tokenhsi/data/cfg/multi_agent/push_soft_legacy40_at05.yaml) | 이전40D/590D·AT0.5m 저장본 resume·전용 bank/output |
| PUSH CONTACT align + AT | [push_toy_contact_align_at.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_contact_align_at.yaml) | 양손 목표 반대편 표면 근접·사람 위치 정렬·근거리 접근 감쇠·AT 성공만10도·별도 scratch |
| PUSH AT success tilt | [push_toy_random_at_10deg_at_success_tilt.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_success_tilt.yaml) | AT 성공에만10도 제한·state/progress 감쇠 없음·GPU0 scratch |
| PUSH AT soft tilt | [push_toy_random_at_10deg_at_tilt_soft.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_tilt_soft.yaml) | AT state/progress 5~15도 감쇠·성공10도 이하·PUSH 각도 자유·scratch |
| PUSH AT state10deg | [push_toy_random_at_10deg_at_state.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_state.yaml) | PUSH 각도 제한 제거·AT state만10도 제한·별도 scratch 설정 |
| PUSH+AT grounded10deg | [push_toy_random_at_10deg_at_grounded.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_grounded.yaml) | PUSH·AT 모두10도/높이±8cm 보상 gate·GPU1 최신 저장본 명시적 전이·GPU0 별도 학습 |
| PUSH random AT 10deg | [push_toy_random_at_10deg.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg.yaml) | PUSH만10도·사람별 XY±2m 랜덤 AT·이동 경로 분리·별도 scratch 설정 |
| Local OPEN toy | [open_toy.yaml](../tokenhsi/data/cfg/multi_agent/open_toy.yaml) | OPEN 100%·후반 RSI·로컬 단독 정책/출력 |
| OPEN strong close | [open_toy_strong_close.yaml](../tokenhsi/data/cfg/multi_agent/open_toy_strong_close.yaml) | OPEN 토이·초기 닫힘 토크4N·m·강한 개방 저항/자동 닫힘·별도 RSI/출력 |
| PUSH + AT | [push_toy_only_random_scene_at.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_only_random_scene_at.yaml) | 문 없는2H/2Box·독립 PUSH/AT 보상 합산·기존 HOLDING+AT 방식·새 checkpoint |
| PUSH only | [push_toy_only_random_scene.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_only_random_scene.yaml) | 문 없는2H/2Box·6토큰·PUSH_ONLY=1·2048환경/최대100만epoch |
| PUSH random scene | [push_toy_midpoint_2m_random_scene.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_midpoint_2m_random_scene.yaml) | 고정1.3m·담당 박스/주변1~2m/방향/yaw 무작위·2m 목표·최대100만epoch |
| PUSH zone swap | [push_toy_midpoint_2m_zone_swap.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_midpoint_2m_zone_swap.yaml) | 2m 중점 XY 기준·reset마다 구역50% 교환·별도 scratch 학습 |
| PUSH midpoint 2m | [push_toy_midpoint_2m.yaml](../tokenhsi/data/cfg/multi_agent/push_toy_midpoint_2m.yaml) | 양손 중점 XY·뒷면 선분 XY 접촉·후진 차감 순이동 실적·2m 목표·로컬 PUSH 참조2개 |
| OPEN reference close | [open_toy_reference_close.yaml](../tokenhsi/data/cfg/multi_agent/open_toy_reference_close.yaml) | 참조 PD 후보 비교로 저항 조정·강성1/감쇠0.5/preload1/최대6·별도 RSI/출력 |
| Local PUSH toy | [push_toy.yaml](../tokenhsi/data/cfg/multi_agent/push_toy.yaml) | PUSH 100%·후반 RSI·로컬 단독 정책/출력 |
| ksh size RSI | [size_rsi.yaml](../ports/ksh_stage1/tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml) | 크기별 PD RSI·별도 actor/critic semantic MLP |
| Stage1 base PUSH | [stage1_base_push.yaml](../ports/stage1_base_push/tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml) | 원본 task-shared embedding/Action MLP/GTA·4상자·기존과제 유지/PUSH_AT100%·원본AT+현재CONTACT |
| ksh typed bias | [typed_bias.yaml](../ports/ksh_stage1/tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias.yaml) | size RSI + typed attention bias |
| ksh typed message | [typed_bias_message.yaml](../ports/ksh_stage1/tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml) | typed bias + typed value message |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 multi-agent carry 기준 |
| Stage1 PUSH fine-tune | [stage1_push_finetune.yaml](../ports/stage1_base_push/tokenhsi/data/cfg/multi_agent/stage1_push_finetune.yaml) | 원본78000회 전이·PUSH50%/기존4과제50%·[train/test/VNC](#stage1-push-finetune) |
| 27 | [approach_scenario_independent_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_independent_with_climb.yaml) | 독립 물체·goal binding, 점 기준 성공 |
| 28 | [approach_scenario_stage1_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_plane.yaml) | 27번의 ON_TOP·CLIMB을 상판 영역 성공으로 변경; Stage 2 기본 출발점 |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 checkpoint에서 협력 graph 학습 |
| Stage 2 plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | 34번 기반 SIT plane·보상·독립 과제 분포 |
| 30 | [approach_scenario_stage1_sit_plane_normalized.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_sit_plane_normalized.yaml) | 28번 + SIT 상판 성공·유효 edge 평균 보상 |
| 31 | [approach_scenario_stage1_skill_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum.yaml) | 30번 + 가까운 시작·후반 CLIMB RSI·어려운 과제 증량·보상 shaping |
| 32 | [approach_scenario_stage1_skill_curriculum_reward_preserved.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved.yaml) | 31번의 시작 상태·RSI·샘플링을 유지하고 state/progress는 30번 식 |
| 33 | [approach_scenario_stage1_at_goal_fix.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_at_goal_fix.yaml) | 32번 + AT 목표 slot reset 오류 수정 |
| 34 | [approach_scenario_stage1_self_sum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml) | 33번 + 자기 edge 보상 합계·동료 edge 평균, 과제 비율 10/10/10/35/35 |
| 35 | [approach_scenario_stage1_slow_near_start.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_slow_near_start.yaml) | 34번 + 가까운 시작 80%를 60만 step 유지하고 120만 step까지 30%로 감소 |
| 36 | [approach_scenario_stage1_placement_focus.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_placement_focus.yaml) | 34번 + HOLDING+AT/HOLDING+ON_TOP만 각 50% 샘플링 |
| 37 | [approach_scenario_stage1_ontop_putdown.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_ontop_putdown.yaml) | 36번 + ON_TOP에도 충돌 없는 중간 구간의 `putDown` RSI 10% 적용 |
| Paired AT | [approach_scenario_stage1_paired_at_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_at_no_near.yaml) | 2-agent/3-object, 매 scene AT+AT, 가까운 시작 없음 |
| Paired placement | [approach_scenario_stage1_paired_placement_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_no_near.yaml) | 2-agent/4-object, scene별 AT+AT 또는 ON_TOP+ON_TOP 각 50%, 가까운 시작 없음 |
| Paired owner HOLDING | [approach_scenario_stage1_paired_placement_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_owner_holding.yaml) | Paired placement + AT/ON_TOP edge에 담당자의 선행 HOLDING `φ` 입력 |
| Unified semantic | [approach_scenario_stage1_unified.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified.yaml) | 2H/4O 독립 5과제, canonical binding·토큰 순열·과제 조건 AMP·unified RSI |
| Unified shared edge | [approach_scenario_stage1_unified_shared_edge_encoder.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder.yaml) | Unified semantic의 actor·critic semantic EdgeEncoder 파라미터 공유 |
| OPEN/PUSH unified | [open_push_unified_amp.yaml](../tokenhsi/data/cfg/multi_agent/open_push_unified_amp.yaml) | 사람별 OPEN/PUSH 50/50 공동 학습·문 지지 및 바닥 박스 밀기·과제 조건부 AMP |
| OPEN/PUSH late RSI | [open_push_late_rsi.yaml](../tokenhsi/data/cfg/multi_agent/open_push_late_rsi.yaml) | 공동 과제·보행/조작50/50·PD검사 후보의 뒤쪽30%를70% 확률로 요청 |
| Door AMP OFF | [door_open_amp_off.yaml](../tokenhsi/data/cfg/multi_agent/door_open_amp_off.yaml) | 2H/2Door 독립 OPEN, shared EdgeEncoder·기존 MAAgent PPO, AMP 비활성 |
| Door AMP ON | [door_open_amp_on.yaml](../tokenhsi/data/cfg/multi_agent/door_open_amp_on.yaml) | 같은 OPEN/PPO·reward에 기존 비조건부 AMP 추가 |
| Door pull approach AMP | [door_pull_unified_approach_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_pull_unified_approach_amp.yaml) | 사람 쪽으로 열리는 −Z hinge·당김 접촉 공간, 별도 checkpoint/output |
| Door loco AMP | [door_open_unified_approach_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_unified_approach_amp.yaml) | 기존 unified·몸/손 계수0.2·전 구간 loco AMP |
| Door BONES left AMP | [door_open_bones_left_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_bones_left_amp.yaml) | 왼손·왼쪽 개방 human reference + loco30/door70 full AMP·loco-only reset |
| Door open/hold AMP | [door_open_hold_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml) | 사람1·문1, 밀어 열고 손으로5초 지지·놓으면 자동 닫힘 |
| Door shared loco AMP | [door_open_shared_edge_approach_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_shared_edge_approach_amp.yaml) | 같은 몸/손0.2·전 구간 loco AMP + Actor/Critic EdgeEncoder 공유 |
| Door unified AMP ON | [door_open_unified_amp_on.yaml](../tokenhsi/data/cfg/multi_agent/door_open_unified_amp_on.yaml) | 기존 unified의 Actor/Critic별 EdgeEncoder + Door proxy AMP/RSI |
| Door shared-edge AMP ON | [door_open_shared_edge_amp_on.yaml](../tokenhsi/data/cfg/multi_agent/door_open_shared_edge_amp_on.yaml) | 같은 Door proxy AMP/RSI + Actor/Critic EdgeEncoder 공유 |
| Shared random binding | [approach_scenario_stage1_unified_shared_random_binding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_random_binding.yaml) | 사용자 12옵션 조합: 무작위 binding·큰 상자·34번 near 시작·비조건부 AMP |
| Unified owner HOLDING | [approach_scenario_stage1_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml) | Unified semantic + AT/ON_TOP 담당 HOLDING φ 입력 |
| Hybrid late RSI | [approach_scenario_stage1_hybrid_late_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_hybrid_late_rsi.yaml) | student 기반, 같은 XYZ의 후반 PD/현재 정책 후보 선택·ON_TOP 받침별 putDown |

27~37번 Stage 1은 2-agent/3-object 독립 graph와 schema 9 semantic 관찰을 쓴다. ON_TOP 두 개를 동시에 샘플하지 않는다. 28~37번의 팀 보상은 자기 0.9, 동료 0.1이다. 30~33번은 유효 edge를 agent별로 평균 내어 단일 edge와 두 edge의 최대 task 보상을 0.6으로 맞춘다. 34~37번은 자기 edge를 합산하고 동료 edge만 평균 내며 state·progress·성공 식은 33번과 같다.

Paired AT/paired placement도 schema 9 semantic 관찰과 34번의 reward·RSI·AMP를 유지한다. 두 agent의 **과제 relation만 같은 타입**으로 샘플하고 실제 motor action은 각자 선택한다. Paired placement는 두 ON_TOP에 서로 다른 받침을 주기 위해 4개 물체를 쓴다. 두 기준 실험은 가까운 시작 확률이 처음부터 끝까지 0이며, 각자 별도 checkpoint/output에서 scratch로 시작한다.

Paired owner HOLDING은 paired placement의 샘플링·reward·RSI·AMP·가까운 시작 없음 설정을 유지한다. 관측만 edge당 5개에서 6개 필드로 늘리고 AT/ON_TOP에는 같은 owner의 `H→source O` HOLDING `φ`, 나머지 유효 edge에는 1을 기록한다. actor·critic의 기존 semantic 64차원 뒤에서 AT/ON_TOP에만 상태 잔차를 더한다. 기존 paired checkpoint와 관측·네트워크 형식이 달라 별도 output에서 scratch로 학습한다.

2026-09-30 공통 코드 수정 이후 scenario의 상자 속도 패널티는 현재 graph에 연결된 담당 물체를 따른다. 이전에는 legacy agent assignment를 사용해 랜덤 binding에서 다른 물체의 패널티가 들어갈 수 있었다. 모델/관측 shape는 같아서 기존 checkpoint를 로드할 수 있지만, 수정 전후 학습의 penalty 의미는 다르다. 이미 실행 중인 프로세스에는 자동 적용되지 않으며 이후 새 프로세스부터 적용된다.

33번은 32번의 보상·샘플링·RSI를 유지한다. AT graph가 두 번째 목표 slot을 가리킬 때 reset도 해당 목표를 배치하도록 공통 코드의 slot 선택을 수정했다. 수정 후 새로 시작하는 27~32번 실행에도 적용되지만, 이미 실행 중인 프로세스와 checkpoint는 바뀌지 않는다. 비교 실험은 33번 전용 output에 scratch로 시작한다.

<a id="object-tokenizer-local-resume"></a>

## Object tokenizer 500epoch 로컬 resume

`ports/ksh_object_finetune/resume_local.sh`는 `carry_chair_copied_adapter`를 `weights/resume_500.pth`에서 이어간다. 기본 GPU0·conda `tokenhsi_koo`·2048환경·seed42·headless다. 메인 학습 코드 대신 전용 스냅샷을 실행하며, `source_39000.pth`도 초기 모델 구성에 필요하다. `launch.py`의 무결성 검사는 서버의 `UPSTREAM.source` 대신 로컬 복사본188개 파일의 실행 전후 해시를 비교한다. 서버 원본과의 동일성 검사가 아니다.

**GPU0의 기존 학습을 정리하고 여유가 생긴 후** 저장소 루트에서 다음 순서로 수동 실행한다. 스크립트는 기존 학습/MPS를 종료하거나 재시작하지 않으며 자동 대기·자동 실행도 하지 않는다.

```bash
# 500에서 이어서 501/502epoch까지 짧게 확인
TOKENHSI_GPU=0 bash ports/ksh_object_finetune/resume_local.sh check
# 위 실행의 정상 종료·복원 로그를 확인한 뒤, 원래500 저장본에서 본학습
TOKENHSI_GPU=0 bash ports/ksh_object_finetune/resume_local.sh train
```

출력은 `output/object_tokenizer_finetune/carry_chair_copied_adapter/resume_local_500_{check,train}_<시각>`으로 분리한다. `OUTPUT_PATH`는 아직 없는 새 경로만 지정할 수 있다. `CHECKPOINT`로 다른 동일 실험 저장본, `MAX_ITERATIONS`로 본학습의 최종 epoch 한도를 지정한다(기본10000, 추가 반복 수가 아님). 현재 종료 조건은 `epoch > max_epochs`여서 한도를 한 번 초과할 수 있다. 확인 실행의 결과가 본학습에 자동 반영되지는 않는다.

resume 경로는 모델·optimizer·epoch500·frame65536000·정규화 통계를 복원한다. 시뮬레이션 상태/AMP replay의 연속 복원까지 보장하지 않는다. CPU에서 pth 로드·relation/task 계약·의자63개 fingerprint·모션136개 참조·Isaac Gym→PyTorch import를 확인했다. 2026-10-17 사용자GPU0 학습 종료 후 로컬PyTorch2.0.0·GPU0 MPS·2048환경으로500→502epoch/frame65798144·optimizer step24000→24096·물리/PPO/AMP 업데이트·저장·정상 종료를 검증했다. 동결actor76개/RMS/초기pth/코드 유지·93개 학습 텐서 변경·전체 모델 유한값 및GPU1 기존학습 PID/로그 갱신을 확인했다. 결과는 `output/object_tokenizer_finetune/local_validation_20261017/check/`, 로그는 상위 `check.log`다. 장기 성능/GUI는 미검증이다. GPU 메모리 부족을 이유로2048환경을 자동 축소하지 않는다. 평가/VNC wrapper는 이번 준비 범위에 포함하지 않았다.

<a id="local-vnc-viewer"></a>

## 로컬 VNC 뷰어

학습 병행 시 [AGENTS.md의 시각화 중 기존 학습 보호](../AGENTS.md#시각화-중-기존-학습-보호--필수)를 먼저 따른다. viewer/충돌 재현을 학습과 같은 MPS 서버에 연결하지 않는다. GPU 오류가 발생하면 재시도 전에 기존 학습 상태부터 확인한다. 아래 과거 실행 기록의 성공은 새로운 GPU/MPS 조합의 안전성을 보장하지 않는다.

2026-10-14 로컬3090에서 `GPU Pipeline: enabled` 직후 segfault는 Mesa 선택 레이어를 사용하는 Gym Vulkan 초기화에서 재현됐다. VNC 명령에 `TOKENHSI_GUI_GPU_MODE=native`를 추가하면 Mesa 레이어를 우회하고 CUDA/Vulkan을 같은 물리 GPU 인덱스로 선택한다. `--test` VNC 전용이며 뷰어만 MPS 자동 연결을 우회한다. CUDA 전체 장치 목록이 보이지만 정책·물리·기본 CUDA 장치는 지정 GPU로 고정한다. GPU0 CONTACT1000 및 GPU1 soft tilt5500 checkpoint 복원·짧은 rollout을 확인했다. 중간 검사에서 MPS 연결 뷰어의 CUDA 오류와 함께 GPU1 학습도 종료됐으므로 이 로컬 viewer는 아래 native 경로를 사용한다. MPS 데몬·학습 재시작은 별도 작업이다. 기본 `mesa` 경로는 유지한다.

2026-10-13 GPU1 PUSH goal-rear RSI의5500epoch 시각화는 `bash output/push_gpu1_view_20261013/launch.sh`로 재현한다. GPU1·저메모리1환경·2사람/2박스·1280×720·빠른 렌더·보행 시작이다. 접속은 `http://localhost:6093/vnc.html?autoconnect=1&resize=remote`, tmux는 `koo_push_gpu1_view_20261013`, 로그는 `output/push_gpu1_view_20261013/viewer.log`다. 기존 세션이 살아 있으면 중복 실행하지 않는다. checkpoint 복원/150step 및 기존 학습 유지 확인.

빠른 재생은 `TOKENHSI_VIEWER_RENDER_EVERY=2 TOKENHSI_VIEWER_DRAW_SYNC=0 TOKENHSI_VIEWER_QUIET=1`을 명령 앞에 추가한다. 물리/정책 step은 그대로 수행하며 viewer 렌더링만2틱마다 하고 draw 대기를 끄며 MA의 매step AMP debug 출력을 생략한다. 기본값은 매틱 렌더링·draw 대기이며 `V`는 렌더링 자체를 끄므로 빠른 화면 재생용이 아니다. 실제 속도 배율은 GPU/렌더링 부하에 따라 다르다.

학습과 함께 뷰어를 띄울 때 `TOKENHSI_VIEWER_LOW_VRAM=1`을 명령 앞에 추가하면 PhysX `max_gpu_contact_pairs`를2097152, `default_buffer_size_multiplier`를4로 덮어쓴다. `--test`·16환경 이하 전용이며 학습에서 사용하면 오류로 중단한다. checkpoint/물리 보상 계약은 유지하며 버퍼 용량만 줄인다. 로컬 viewer와 VNC 모두 전달된다. XIO는 X 연결 오류이므로 이 옵션만으로 해결된다고 보장하지 않는다.

Wayland 데스크톱에서도 VNC는 별도 Xvfb를 사용한다. `run-gui.sh`는 x11vnc에만 `WAYLAND_DISPLAY`·`XDG_SESSION_TYPE`을 제거해 잘못된 Wayland 감지로 인한 즉시 종료를 막는다.

현재2097152/4 설정은2026-10-08 GPU0 OPEN1000checkpoint·1환경30step·1280×720 VNC에서 정상 종료를 확인했다(`output/open_toy_strong_close_low_vram_double_check`). 버퍼 값을 각각2배로 올린 것이며 실제 VRAM 소비가 정확히2배라는 뜻은 아니다. 새 값은 뷰어 재실행부터 적용된다.

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_toy_strong_close_test.sh output/open_toy_strong_close/OpenToyStrongClose_08-16-36-28/nn/OpenToyStrongClose_00001000.pth 2 1 4 10
# 로컬 X 연결이 불안정하면 별도 가상 화면 사용
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6089 bash tokenhsi/scripts/multi_agent/open_toy_strong_close_vnc.sh output/open_toy_strong_close/OpenToyStrongClose_08-16-36-28/nn/OpenToyStrongClose_00001000.pth 2 1 4 10
```

접속 주소는 `http://localhost:6089/vnc.html?autoconnect=1&resize=remote`이며 원격이면6089포트를 포워딩한다. 예시1000epoch checkpoint는 검증용 기준이며 실제 요청 시 같은 실험의 저장된 checkpoint를 확인해 선택한다. GPU0은 예시로, 실행 전 점유와 여유 메모리를 확인한다. `x11vnc failed`면 `/tmp/tokenhsi_gui_6089_x11vnc.log`·`_xvfb.log`·`_websockify.log`를 확인한다. 과거의 작은 버퍼 값 대신 이 절의 현재 값과 실제 코드를 따른다.

## OPEN 참조 동작의 밀기 가능성 검사

`check_open_reference_push.sh`는 기존 strong-close YAML/asset/RSI를 읽고 CPU PhysX2환경에서 비학습 검사를 한다. 프로세스 메모리 안에서만 기존(강성2/감쇠0.5/최대4)·중간(6/1/12/preload2)·현재강화(16/4/32/preload4)를 비교한다. YAML·bank·checkpoint를 수정하지 않는다.

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/check_open_reference_push.sh
# 후보9개 추가 비교; 기존 보고서와 출력 분리
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo OPEN_REFERENCE_SWEEP=1 OUTPUT_PATH=output/open_reference_resistance_sweep bash tokenhsi/scripts/multi_agent/check_open_reference_push.sh
```

참조 시각0.633/1.433/2.233초와 닫힌 문/RSI 시작각을 조합한18조건에서 관절 PD 목표만 재생하고 root는 자유롭게 둔다. 실제 CPU 충돌 쌍을 control 경계에서 읽어 손/비손 문 접촉·첫 비손 접촉 이전 최대각·낙상을 기록한다. 사람을 멀리 둔9조건은 손잡이를 따라 이상적 접선 힘20/40/60N을4초 적용하고 무외력 환경과 비교한다. 이는 사람의 손 힘 측정이나 학습 정책 평가가 아니다.

결과는 [보고서](../output/open_reference_push_check/REPORT.md), `audit.json`, `reference_traces.csv`, `comparison.png`에 저장된다. 모든 저항에서 참조 PD가 낙상했으므로 저항 완화만으로 해결된다고 해석하지 않는다. 현재 저항에서40N 이상적 힘은90도 이상 개방했지만, 참조 PD의 손만으로90도 개방/5초 지지는 검증되지 않았다. 접촉 경계 사이의 짧은 충돌은 누락될 수 있다.

## 로컬 OPEN / PUSH 분리 토이 학습

2026-10-12 종료된 OPEN strong-close를 최신 저장6000epoch checkpoint에서 GPU0·2048환경·seed1015·기존 최대9000epoch로 재개했다. 실행/로그/새 run은 `output/open_resume_20261012/launch.json`, 재개 명령은 `output/open_resume_20261012/launch.sh`에 있다. tmux 접속은 `tmux attach -t koo_open_resume_20261012`다. 이전 마지막6354epoch 중6000 이후 미저장 진행은 복원되지 않는다. 기존 실행이 살아 있으면 중복 실행하지 않는다.

2026-10-11 리부트 후 OPEN strong-close는 GPU0·2048환경·seed1015·기존 최대9000epoch로5500epoch checkpoint에서 재개했다. 실행/로그/새 run은 `output/open_resume_20261011/launch.json`에 기록했다. tmux 접속은 `tmux attach -t koo_open_resume_20261011`이며 재개 명령은 `output/open_resume_20261011/launch.sh`에 있다. 기존 실행이 살아 있으면 중복 실행하지 않는다. GPU/PID는 실제 프로세스로 확인한다.

`open_toy`와 `push_toy`는 최신 `open_push_late_rsi`에서 각각 OPEN/PUSH 과제 확률을 100/0, 0/100으로 바꾼 별도 실험이다. **사람 2명 모두 해당 과제만 학습**하며 기존 H2/문2/박스2 구조를 유지한다. 물리·토큰 순열·조건부 AMP 네트워크·보행/조작 50/50 RSI·조작 후보 후반 요청 70%는 공동 설정과 같다. 로컬 토이 두 설정에는 아래의 손 조작 조건을 추가했다. 사용하지 않는 과제 물체와 모션은 구조/RSI bank 호환을 위해 로드하지만 해당 과제의 rollout/AMP 전문가 표본은 생성하지 않는다. OPEN의 `skillDiscProb`는 `[0.33,0.67,0]`, PUSH는 `[0.33,0,0.67]`이다.

각 실험에 전용 env/train YAML 및 `_train.sh`, `_test.sh`, `_vnc.sh`가 있다. train 기본 최대 반복 수는 `max_epochs=9000`이다. `MAX_ITERATIONS`를 지정하면 해당 값으로 덮어쓴다. 기존 엔진은 최대값을 넘는 시점에 종료하므로 9001epoch까지 실행할 수 있다. 환경 수 기본값은 **2048**이고 자동 축소하지 않는다. 출력은 `output/open_toy_hand_contact`, `output/push_toy_hand_contact`로 분리한다. 수정 전 토이 checkpoint도 새 손 조작 계약과 달라 직접 resume/test할 수 없다. task 확률이 checkpoint 계약에 포함되어 공동 정책 또는 반대 과제 checkpoint를 직접 resume/test할 수 없다. 같은 과제의 재개에는 `RESUME_CHECKPOINT`를 사용한다.

같은 실험의 GPU를 옮길 때는 저장된 checkpoint를 확인하고 기존 실행을 종료한 다음 `TOKENHSI_GPU=<대상 GPU> RESUME_CHECKPOINT=<같은 실험 checkpoint> TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/<실험명>_train.sh 2 2048 4`로 재개한다. 모델/optimizer/epoch/frame은 복원되며 저장 이후 진행·실행 중 환경/AMP replay는 그대로 이동되지 않는다. 기존 실험의 seed는 `relation_config.yaml`의 train seed를 `SEED`로 지정한다. 재개 실행은 같은 output 아래 새 run 디렉터리를 만든다. 2026-10-09 OPEN strong-close/PUSH의 GPU0/1 교체 재개 명령·checkpoint·로그는 `output/gpu_swap_20261009/launch.json`에 기록했다. 실제 현재 GPU/PID는 `nvidia-smi`와 프로세스로 확인한다.

두 train을 각각 실행하면 **Actor/Critic/D/optimizer/AMP replay가 실행별로 독립**이다. 같은 network 구조와 OPEN/PUSH 2차원 조건 라벨을 재사용하며 D 자체를 task별 다른 아키텍처로 바꾸지는 않는다. 한 실행 안의 사람2명은 해당 실행의 하나의 정책과 D를 공유한다.

`manipulationContact`는 현재 `open_toy`·`push_toy`에만 설정했다. 기존 공동 YAML과 SIT/ON_TOP/CLIMB는 기존 동작을 유지한다. 새 설정은 다음 조건을 사용한다.

- RSI 초기 문각도/박스 진행도는 그대로 인정한다. 이후 physics tick별 관측한 최고 진행 위치를 갱신하되, 같은 tick에서 유효 손 접촉·몸 접촉 없음(OPEN은 선 자세, PUSH는 grounded)을 만족한 새 진행만 추가 인정한다. 몸으로 먼저 진행한 뒤 손만 대거나 뒤로 갔다 다시 앞으로 와도 그 구간을 소급 인정하지 않는다.
- OPEN 개방 progress는 현재 손 접촉과 선 자세·몸 접촉 없음으로 gate한다. 90도/5초 지지는 기존 손 근접 grace0.1초를 유지하지만 몸 접촉 발생 시 즉시 공백 이력과 타이머를 초기화한다. RSI 시작각+유효 손 개방각이 목표각−0.02rad 이상이어야 지지 타이머를 누적한다. 실제 문각도는 여전히90도 이상이어야 한다.
- PUSH 조작 progress는 현재 양손 유효 접촉·몸 접촉 없음으로 gate한다. 성공은 RSI 진행+손으로 만든 진행이 목표 허용 영역에 도달하고, reset 이후 유효 손 접촉 중 새 전진5cm 이상 및 기존 목표/안정 조건을 만족하며 손 접촉 중 완료를 확인한 경우다. 이후 손을 떼어도 목표의 안정 상태와 몸 접촉 없음 조건을 유지하면 progress0.2+success0.2를 받는다. 손 거리 state 항을 최대값으로 덮어쓰지 않고 실제 거리로 계산한다. 목표/안정 조건을 벗어나거나 몸 접촉이 생기면 현재 완료가 해제된다.
- 몸 접촉은 담당 사람의 손 이외13개 body의 MJCF sphere/capsule/foot mesh 범위와 담당 문 panel/handle 또는 박스의 OBB 근접(추가2cm), 신체/물체 수평 반력2N 이상으로 추정한다. 수직 바닥 지지만으로 차단하지 않고 접촉만으로 episode를 종료하지 않는다. GPU net force는 충돌 pair가 아니므로 정확한 접촉 부위별 기여도/힘 비율을 보장하지 않는다.
- 기존 object 관측의 예약 슬롯에 인정 진행도·PUSH 손 전진 자격·완료/몸 접촉 이력을 넣어 관측684/AMP1310 shape를 유지한다. 새 의미와 MJCF hash는 checkpoint 계약으로 분리한다. RSI 후보 bank 자체는 기존 물리/기하 계약을 재사용한다.

TensorBoard `relation/90_debug/{open,push}/contact/{all,moving,completed}/` 아래의 `sample_count`, `hand_contact`, `clean_hand_contact`, `body_contact`, `body_horizontal_force_N`, `credited_progress`를 비교한다. `moving`은 미완료 상태에서 박스 속도0.02m/s 초과 또는 문 각속도0.02rad/s 초과인 표본이다. 완료 후 손 떼기를 이동 중 손 미사용과 구분하며 표본0의 평균0은 성능으로 해석하지 않는다. PUSH의 `contact/hand_credited_distance`는 reset 이후 손으로 인정받은 새 이동(m)이다.

로컬 전용 wrapper는 GPU 6 고정 제한이 없고 기본 GPU 0을 사용한다. GPU 번호/UUID는 `TOKENHSI_GPU`로 한 번 지정하며 VNC 내부에도 전달한다. 현재 로컬에는 RTX3090 두 장(0/1)이 있다. conda는 `TOKENHSI_CONDA_ENV` 지정값, 이미 활성화한 환경, `tokenhsi_koo` 순으로 선택하며 `runtime_env.sh`로 실행한다. 서버에서 이 wrapper를 사용할 때는 `TOKENHSI_CONDA_ENV=tokenhsi`를 지정한다. 저장소 루트는 스크립트 위치에서 계산하고 필요한 데이터 7개와 RSI bank는 저장소 내부 상대 경로로 참조한다. 원본 서버 경로를 추가하지 않는다.

```bash
# 실행 전 점유 확인; 아래 두 명령은 독립 학습이며 순차 실행해도 된다.
nvidia-smi
TOKENHSI_GPU=0 SEED=42 bash tokenhsi/scripts/multi_agent/open_toy_train.sh
TOKENHSI_GPU=0 SEED=42 bash tokenhsi/scripts/multi_agent/push_toy_train.sh
# 별도 확인 출력: 2048환경 유지, 짧은 반복만 실행
TOKENHSI_GPU=0 MAX_ITERATIONS=2 OUTPUT_PATH=output/open_toy_hand_contact_check bash tokenhsi/scripts/multi_agent/open_toy_train.sh
TOKENHSI_GPU=0 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_toy_hand_contact_check bash tokenhsi/scripts/multi_agent/push_toy_train.sh
# 로컬 viewer / 화면 없는 평가: 각 과제의 checkpoint를 사용
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_toy_test.sh <open_checkpoint.pth> 2 1 4 3
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_toy_test.sh <open_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_test.sh <push_checkpoint.pth> 2 1 4 3
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_test.sh <push_checkpoint.pth> 2 16 4 3
# 서버 VNC: 서버에 맞는 GPU/conda 지정, 기존 viewer와 포트 분리
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi PORT=6087 bash tokenhsi/scripts/multi_agent/open_toy_vnc.sh <open_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi PORT=6088 bash tokenhsi/scripts/multi_agent/push_toy_vnc.sh <push_checkpoint.pth> 2 1 4 10
```

손 조작 조건 추가 전 로컬 검증(2026-10-08): CPU42개 통과. RTX3090 GPU0·2048환경·MAX_ITERATIONS2에서 두 설정 각각3epoch 완료/checkpoint 저장, 담당 과제 rollout131072표본/epoch·반대 과제0 확인. 각 checkpoint2환경·30step headless 평가 복원 통과(`output/{open,push}_toy_{check,eval_check}`). GUI/VNC 실제 실행 및 장기 학습 성공은 미검증이다.

손 조작 조건 추가 후 검증(2026-10-08): CPU54개 통과. GPU0·2048환경·MAX_ITERATIONS2에서 각각3epoch/checkpoint 저장 및2환경/30step headless 평가 복원 통과(`output/{open,push}_toy_hand_contact_{check,eval_check}`). 새 접촉37개 태그×3epoch 유한값·담당131072/반대0 표본·수정 전 checkpoint 계약과의 차이 확인(`output/toy_hand_contact_tensorboard_audit.json`). 장기 손 사용 개선·실제 접촉 proxy 정확도·GUI/VNC는 미검증이다.

## PUSH midpoint 2m — 중점 XY·높이 자유·순이동 실적

사용자 전달 PUSH180/2m 기준의 PUSH 동작을 `push_toy_midpoint_2m`으로 분리했다. `relationReward.push.control_mode: midpoint_xy_net_credit`를 지정한 실험에서만 적용한다. 기존 `push_toy`/OPEN/공동 설정과 실행 중인 학습은 유지한다. 로컬 데이터는 사용자 지시에 따라 기존 PUSH180 참조2개를 그대로 사용하며21개로 늘리지 않는다. 기존 로컬 과제 분리를 유지해 사람2명 모두 PUSH를 하고 자기 박스를 민다(서버의 OPEN/PUSH50/50 혼합은 적용하지 않음). 두 모션 weight1씩, PUSH AMP 보행33%/밀기67%, 보행/조작 초기화50/50·조작 후반 요청70%,2048환경·600step·9000회, 자기90%/동료10% 및 task50%/AMP50%를 유지한다.

- 손 위치 보상은 박스 yaw로 계산한 **양손 중점 XY와 뒷면 중앙 `(0, -0.65)` 사이 거리**의 `exp(-10*거리²)`다. 손 높이·높이 차이·간격·오른손/왼손 고정점을 정책 보상으로 제한하지 않는다.
- 같은 physics tick에 각 손이 뒷면 XY 선분의18cm 이내이고 각 손의 밀기 반력0.5N 이상·박스 전방 힘0.5N 이상·바닥/기울기 조건을 만족해야 목표까지의 거리 감소를 인정한다. 몸 접촉은 진단하되 PUSH 이동/성공을 차단하지 않는다. 힘은 rigid body의 합산 접촉력이며 충돌 쌍별 힘은 아니다.
- 인정 이동은 전진 유효 접촉 tick에 더하고 후진은 접촉 여부와 관계없이 차감한다.0 아래로 내려가지 않으며 reset 직후부터 목표에 가까워진 순거리 이하로 제한한다. RSI로 미리 진행한 거리와 실제 밀기 실적을 별도로 저장한다.
- 성공은 원래 목표2m에 XY15cm 이내·높이±8cm·기울기0.35rad 이하·선속도0.2m/s 이하·각속도0.3rad/s 이하이며, `양손 인정 순거리 >= max(RSI reset 직후 남은 거리 - 0.15, 0.05)`를 요구한다. 조건을 유지하면 손을 떼어도 성공, 유지 시간은 없다. 진행 보상은 실제 기하 진행·RSI 시작 진행+인정 이동·바닥·유효 접촉으로 제한하며 성공 시 최대값을 지급한다.
- 박스1.3m 정육면체·20kg·마찰0.6·전방+Y 유지, 목표만1.5→2m다. RSI의 높이0.8m·좌우±0.2m·뒷면보다6cm 뒤 정렬은 **초기 자세용**이며 정책 목표점이 아니다. RSI 선행 이동은 `2m*0.85*phase`, 목표는 원래 시작점 기준이다.

기존 PUSH checkpoint는 reward/이동 실적/목표/bank 계약이 달라 직접 resume/test할 수 없다. 새 설정은 scratch 학습하며 output `output/push_toy_midpoint_2m`, bank `tokenhsi/data/push_toy_midpoint_2m_rsi_bank.json`을 사용한다. 실행 중인 GPU1의 기존 PUSH에는 파일 변경이 소급 적용되지 않는다.

```bash
nvidia-smi
# 새 본학습: GPU 확보 후 사용, 기존 PUSH를 자동 종료하지 않음
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=42 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_train.sh
# 짧은 학습 확인도2048환경 유지
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo MAX_ITERATIONS=2 OUTPUT_PATH=output/push_toy_midpoint_2m_check bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_train.sh
# 새 실험의 checkpoint를 사용한 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_test.sh <midpoint_2m_checkpoint.pth> 2 1 4 3
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_test.sh <midpoint_2m_checkpoint.pth> 2 16 4 3
# VNC: 서버에서는GPU/conda를해당머신에맞춰지정
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6092 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_vnc.sh <midpoint_2m_checkpoint.pth> 2 1 4 10
# checkpoint 없는 비학습 CPU2환경 후보 검사·1000reset/partial reset/AMP
TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=cpu bash tokenhsi/scripts/multi_agent/build_push_midpoint_2m_rsi.sh
# bank 재생성 없이 GPU reset/AMP·12step 물리 확인
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RSI_SIM=gpu RSI_REUSE_BANK=1 RSI_SMOKE_STEPS=12 OUTPUT_PATH=output/push_toy_midpoint_2m_gpu_check bash tokenhsi/scripts/multi_agent/build_push_midpoint_2m_rsi.sh
```

검증(2026-10-09): CPU 회귀77개 및 최종 새 설정/기존 토이 계약18개 통과. CPU2환경12step RSI 검사에서 OPEN13/13·PUSH90/171(모션별31/59) 후보가 통과했고1000reset·부분 reset·고정 목표·AMP가 유한함을 확인했다(`output/push_toy_midpoint_2m_rsi_check`). GPU0·2환경에서는 CPU bank를 재생성하지 않고1000reset·부분 reset·AMP 및12step 물리 실행/유한 보상·관측·인정 이동을 확인했다(`output/push_toy_midpoint_2m_gpu_check`). GPU 후보 재선별·새2048환경 학습·checkpoint·정책 성공·GUI는 미검증이며 기존 두 본학습은 유지했다.

VNC 접속 `http://localhost:6092/vnc.html?autoconnect=1&resize=remote`. 위 정책 viewer/test에는 새 checkpoint가 필요하며 장기 학습/GUI/성공률은 별도 확인한다. 학습 병행 시 [로컬 VNC 뷰어](#local-vnc-viewer)를 따른다.

## checkpoint 없는 AMP 물리 모션 뷰어

새 저항 `open_toy_reference_close`에서 AMP 참조 관절 목표를 시간에 따라 PD로 재생한다. 초기 정렬 이후 root는 자유이며 사람 순간이동·외력·학습 정책을 사용하지 않는다. 기본은 중간/후반/초반 RSI를 순환하며 문을 밀어 움직이는 과정을 보여준다. 낙상 또는 참조 끝+1초에서 멈춘 뒤2초 후 다음 시작으로 넘어간다. 문 통과/5초 지지 성공 시연은 아니다.

```bash
nvidia-smi
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6091 bash tokenhsi/scripts/multi_agent/open_reference_motion_view_vnc.sh
# 로컬 viewer
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 bash tokenhsi/scripts/multi_agent/open_reference_motion_view.sh
# 화면 없는 짧은 재생 검사
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 REFERENCE_VIEW_CYCLES=3 bash tokenhsi/scripts/multi_agent/open_reference_motion_view.sh
```

GPU0·저메모리 옵션·1280×720·VNC6091에서 RSI2회 재생/종료, CPU1환경 닫힌 문 시작1회 재생을 확인했다(`output/open_reference_motion_{vnc,closed}_check`). 세 경우 모두 이후 낙상했다.

접속 `http://localhost:6091/vnc.html?autoconnect=1&resize=remote`. `SPACE` 일시정지/재생, `N` 다음 참조 시작, `O` 닫힌 문 시작, `L` RSI 각도 시작. 첫 실행부터 닫힌 문을 보려면 `REFERENCE_START_MODE=closed`를 추가한다. 화면 손 힘은 모든 접촉의 합이며 문에 가한 힘이나 정확한 손/몸 충돌 쌍 판정이 아니다. 기본 로그는 `output/open_reference_motion_view/trace.jsonl`(재실행 시 덮어쓰기), 별도 경로는 `OUTPUT_PATH`로 지정한다. GPU·포트 선택/Wayland/VRAM 옵션은 [로컬 VNC 뷰어](#local-vnc-viewer)를 따른다.

## OPEN reference close — 참조 PD에 맞춘 닫힘 저항 후보

`open_toy_reference_close`는 strong-close 토이에서 자동 닫힘만 `clamp(-1*angle - 0.5*angular_velocity - 1, -6, 6)`N·m로 바꾼 별도 실행이다. 손 조작/비손 접촉 gate·90도/5초 지지 조건·문 asset·AMP 전문가·네트워크·RSI 혼합은 유지한다. 90도 정지 저항은29.13→2.57N·m(손잡이 접선 힘 약37.3→3.3N)다. 기본 최대9000회·2048환경, output `output/open_toy_reference_close`, 전용 bank `tokenhsi/data/open_toy_reference_close_rsi_bank.json`을 사용한다.

후보9개 중 초기 preload1N·m 이상을 유지하고 참조 중간/후반 RSI 두 시작에서 비손 접촉 전 최대각이95도 이상이며 선 자세로90도에 도달한 후보 중90도 저항이 가장 큰 값을 선택했다. 선택 후보는 두 시작 모두 최대97.4도, 더 강한2/0.5/preload1 후보는 후반 최대90.9도로 여유가 작았다. 닫힌 문 시작3조건은 미달이고 모든 자유 root 참조 PD는 이후 낙상했다. 따라서 **모든 시작에서 손만으로90도 개방/5초 지지를 해결한 설정은 아니다.** 참조의 순간 접촉 후 관성 개방도 포함되므로 학습 정책의 유효 손 진행·지지 성공과 구분한다. [전체 비교](../output/open_reference_resistance_sweep/REPORT.md)와 [선택 근거](../output/open_reference_resistance_sweep/SELECTED.md)를 참조한다.

기존 OPEN/strong-close checkpoint는 물리와 bank 계약이 달라 직접 resume/test할 수 없다. 새 설정으로 scratch 학습 후 해당 checkpoint를 사용한다. 실행 중인 기존 학습은 자동 변경되지 않는다. 아래 학습 명령은 GPU가 확보된 뒤 사용한다.

```bash
nvidia-smi
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=42 bash tokenhsi/scripts/multi_agent/open_toy_reference_close_train.sh
# 짧은 학습 확인도2048환경 유지
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo MAX_ITERATIONS=2 OUTPUT_PATH=output/open_toy_reference_close_check bash tokenhsi/scripts/multi_agent/open_toy_reference_close_train.sh
# 새 실험의 checkpoint 사용: 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_toy_reference_close_test.sh <reference_close_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_toy_reference_close_test.sh <reference_close_checkpoint.pth> 2 16 4 3
# 서버/가상 화면 VNC; 서버 conda는TOKENHSI_CONDA_ENV=tokenhsi로 지정
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6091 bash tokenhsi/scripts/multi_agent/open_toy_reference_close_vnc.sh <reference_close_checkpoint.pth> 2 1 4 10
```

실제 문 CPU 검사에서는90→10도2.03초, 2N·m/0.15초 충격 최대0.52도(기존5.28도),20N·m/0.15초 최대42.86도였다(`output/open_toy_reference_close_physics_check`). CPU69개 통과. 새 bank는 CPU2환경 후보 OPEN13/13·PUSH116/171 및1000reset/partial reset/AMP 검사 완료(`output/open_toy_reference_close_rsi_check`). GPU0·2환경에서 `RSI_SCREEN_SKILLS=doorOpen`으로 OPEN13/13을 재검사하고1000reset/partial reset/AMP를 확인했다. 미사용 PUSH116개는 계약이 같은 CPU bank에서 보존했다(`output/open_toy_reference_close_rsi_gpu_check`). 최종 bank의 CPU 계약8개도 통과했다. 부분 screen은 기존 bank의 계약/PD표시가 맞아야 허용되며 기본은 두 task 모두 검사한다. 새 checkpoint 생성·2048환경 학습·정책 성공·이 실험의 GUI/VNC는 아직 미검증이다. 현재 두 GPU에서 본학습이 실행 중이라 새2048환경 학습을 함께 실행하지 않았다.

## OPEN strong close — 가벼운 충격에 대한 개방 저항 강화

`open_toy_strong_close`는 OPEN 손 접촉 토이의 강화 변형이다. 닫힘 토크는 `clamp(-16*angle - 4*angular_velocity - 4, -32, 32)`N·m다. 기존 강성2/감쇠0.5/최대4에서16/4/32로 올리고, 닫힌 위치에서도4N·m preload를 걸어 첫 접촉에 저항한다. hinge damping0.5·마찰0, 문 질량/충돌 형상·보상/90도·손 지지5초 조건은 유지한다. preload가 없는 기존 설정은 기존 식을 사용한다.

전용 env/train YAML, train/test/VNC wrapper, `tokenhsi/data/open_toy_strong_close_rsi_bank.json`, 기본 출력 `output/open_toy_strong_close`를 사용한다. 최대9000회·2048환경이며 기존 OPEN/PUSH 실행에는 적용되지 않는다. 물리/RSI 계약이 바뀌어 기존 OPEN checkpoint 직접 resume/test는 거부된다. 새 scratch 학습과 해당 실험의 checkpoint를 사용한다.

```bash
# 본학습: GPU 점유 확인 후 실행
nvidia-smi
TOKENHSI_GPU=1 bash tokenhsi/scripts/multi_agent/open_toy_strong_close_train.sh
# 확인용 학습: 2048환경 유지
TOKENHSI_GPU=1 MAX_ITERATIONS=2 OUTPUT_PATH=output/open_toy_strong_close_check bash tokenhsi/scripts/multi_agent/open_toy_strong_close_train.sh
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_toy_strong_close_test.sh <strong_close_checkpoint.pth> 2 1 4 3
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_toy_strong_close_test.sh <strong_close_checkpoint.pth> 2 2 4 3
# VNC: Wayland 환경변수 상속으로 x11vnc가 종료되는 로컬 환경 대응
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 TOKENHSI_GPU=1 PORT=6090 bash tokenhsi/scripts/multi_agent/open_toy_strong_close_vnc.sh <strong_close_checkpoint.pth> 2 1 4 10
# 실제 문 asset만 사용하는 비학습 CPU PhysX 충격/놓기 비교
TOKENHSI_GPU=1 bash -c '. tokenhsi/scripts/multi_agent/runtime_env.sh; PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_door_autoclose.py'
```

검증: 실제 문 asset CPU PhysX8조건에서90도 정지 상태를 놓으면10도 미만까지1.93→0.63초, 초기 개방 각속도3rad/s에서는1.98→0.70초. 닫힌 문에2N·m/0.15초 입력 시 최대 개방5.28도→거의0도,20N·m/0.15초에서는52.80도→9.97도다(`output/open_toy_strong_close_physics_check/audit.json`). 사람이 가하는 충격/정책 성능과 같은 검사는 아니다. GPU1 비학습2환경에서 RSI OPEN13/13·PUSH145/171 후보,1000reset·partial reset·AMP 검사 통과(`output/open_toy_strong_close_rsi_check/audit.json`). 관련 CPU67개 통과. GPU1·2048환경·MAX_ITERATIONS2에서3epoch/checkpoint 저장(`output/open_toy_strong_close_check`), 해당 checkpoint2환경/30step headless 복원 통과(`output/open_toy_strong_close_eval_check`). 본학습·강화 설정 GUI/VNC·장시간 정책의 정상 손 지지5초 성공은 미검증이다.

## OPEN/PUSH Unified — 두 과제 공동 학습

현재 PUSH 상자는 **1.3m 정육면체**,20kg·마찰0.6이며 **양손 목표 높이0.8m**다. 깊이0.6m의 높은 상자는 뒤면을 밀 때 바닥에서 미끄러지기보다 기울어지는 물리 비교 결과에 따라 변경했다. 기본 output은 `output/open_push_unified_amp_cube_low`, 후반RSI는 `output/open_push_late_rsi_cube_low`이며 이전 크기/목표 checkpoint 직접 resume/test는 계약상 거부된다. 기존 실행 프로세스에는 자동 적용되지 않는다.

PUSH 손 목표는 박스 뒤면의 두 점이다: 오른손 local X=+0.2m, 왼손 X=-0.2m, Y=-depth/2=-0.65m, Z=0.8-height/2=0.15m(바닥 기준0.8m). 목표는 박스 회전/이동을 따른다. 손 상호작용은 `exp(-10*max(d_right²,d_left²))`에 바닥/기울기 조건을 곱한다. 밀기 접촉은 양손 모두 각자의 목표18cm 이내·각 손 전방 반력0.5N 이상·박스 전방 반력0.5N 이상을 같은 physics tick에 만족해야 한다. RSI는 몸통/발 및 팔 길이를 보존하고 양쪽 어깨에서 팔을 회전해 손Z를 목표에 맞춘 뒤 양손 XY를 강체 yaw/이동·6cm stand-off로 정렬한다. 어깨 DOF와 reset AMP 이력도 보정하며 AMP 전문가 원본은 유지한다. 물리적으로 목표에 닿지 못하거나 불안정한 후보는 PD screen에서 제외한다. OPEN의 한 손 조건은 유지한다.

물리 비교 검사는 다음 명령으로 실행한다. 학습하지 않는2환경 검사이며 YAML/RSI bank를 수정하지 않는다. 기존 상자의 목표 높이1.05/0.8/0.6/0.4m와1.3m 정육면체의1.05/0.8m를20kg·동일 마찰로 비교한다. 외력 검사는 대칭 양손 힘과 같은 합력80/120/160/200N을 뒤면 중앙에2초간 적용하고, 별도 환경은 사람을 멀리 둔 무외력 대조군이다. 이는 힘 높이/형상의 물리 비교이며 정책 성공 검사가 아니다. 사람 PD 검사는 clip별6개 동일 원본 자세를 각2환경에서 최대2초 유지하고 양손 접촉·시작 손 거리·기울기·바닥 edge 전진을 기록한다. 현재 사람 PD 검사는 어깨 회전 RSI를 적용한다. 아래의 이전 비교 결과는 팔 보정을 적용하기 전 원본 손Z를 유지한 검사다. `PHYSICS_FORCE_ONLY=1`로 외력 검사만 실행할 수 있다.

```bash
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_physics_comparison_verified bash tokenhsi/scripts/multi_agent/check_open_push_physics.sh
```

외력 분리 비교에서160N을 적용하면 기존 얕은 상자는 목표1.05/0.8/0.6/0.4m 모두 기울어졌고, 정육면체도1.05m에서는 기울어졌다. 정육면체+0.8m는 기울기0·바닥 전진31.56cm를 기록했다(`output/open_push_physics_comparison_isolated/comparison.json`). 무외력 대조군의 최대 전진은 기존 상자0.008mm/정육면체0이었다.같은 설정도200N에서는 넘어졌으므로 모든 힘에서 안정성을 보장하지 않는다. 현재 기본 설정은 이 정육면체/0.8m 조합이며 낮은 손 RSI 팔 보정을 적용했다. 새 학습 정책의 장시간 밀기/AMP 적합성은 별도 확인이 필요하다.

전체 검사24개 외력 조건·120개 사람PD 결과는 `output/open_push_physics_comparison_verified/comparison.json`이다. 원본 자세/1.05m 목표에서 기울기20도 이내·root 높이65cm 이상 구간의 바닥 edge 이동은 기존 상자 최대0.46cm, 정육면체 최대11.89cm였다.0.8m 목표의 첫 step 손 거리 중앙값은 약28cm로 현재18cm 접촉 허용보다 크다. 이 PD는 초기 자세를 고정하므로 목표 변경만으로 팔 동작을 낮추거나 보행/밀기 정책을 학습하지 않는다.

TensorBoard 진단(2026-10-07 추가)은 `relation/90_debug/{open,push}/{current_success,sample_count,distance_xy,xy_progress,manipulation_progress}`다. 담당 task의 전체 rollout 사람×step 표본으로 평균한다. `distance_xy`는 root→문 손잡이 또는 root→박스 중심의 수평 거리(m), `xy_progress`는 OPEN의 `1/(1+max(distance-approach_delta,0)/approach_sigma)`와 PUSH의 `1/(1+max(distance-0.5,0))`다. 현재 OPEN의 delta0.5/sigma1과 PUSH는 같은 식이며 XY거리0.5m 이내에서1로 포화한다. PUSH state의 접근 항도 같은 식에0.2를 곱한다(기존 `1/(1+distance)`에서 변경). `manipulation_progress`는 기존 progress 보상을0.2로 나눈 값으로 문 개방/지지 또는 박스 목표 이동 진행이며 접근 점수와 구분한다. 두 공동 실험(unified/late RSI)에 적용하며 조작 progress·성공·초기화·checkpoint 계약은 유지한다. 기존 실행 프로세스에는 자동 반영되지 않고 새 실행부터 변경 보상을 사용한다.

낮은 손/AMP 충돌 진단은 `relation/90_debug/push/` 아래에 기록한다. 모든 physics 값은 PUSH 담당 사람×control step 평균이며 `sample_count`와 함께 본다.

| 하위 태그 | 확인할 문제 |
| --- | --- |
| `physics/{right_distance,left_distance,worst_hand_distance}` | 손별 목표 거리(m), 한 손만 붙는 문제 |
| `physics/{right_height_error,left_height_error,both_hands_near,paired_contact}` | 목표 대비 손Z 오차(m), 양손 근접/동시 반력 접촉률 |
| `physics/{right_reaction_N,left_reaction_N,box_forward_force_N}` | 마지막 physics tick의 법선 반력(N); paired_contact는 control step 내 같은 tick의 양손 접촉을 수집 |
| `physics/{tilt_rad,tilt_violation,grounded,box_speed}` | 기울기(rad),20도 초과율,바닥/기울기 유효율,속도(m/s) |
| `physics/{box_forward_m,bottom_edge_forward_m,since_reset_forward_m,since_reset_bottom_edge_forward_m}` | 원래 시작/이번 reset 이후 전진(m),기울어짐에 의한 중심 이동과 바닥 edge 전진 구분 |
| `rsi/{right_reset_distance,left_reset_distance,retarget_count}` | 보정된 kinematic reset 손 거리(m)와 PUSH 조작 초기화 수 |
| `reward_balance/{task_reward,amp_reward,weighted_task_reward,weighted_amp_reward,combined_reward,sample_count}` | 실제 PPO task/AMP 보상 및 가중 후 기여; OPEN도 같은 태그 제공 |
| `reward_balance/{paired_contact,both_hands_near}_{amp_reward,task_reward,sample_count}` | 양손 접촉/근접 PUSH 표본에서 AMP와 과제 보상이 경쟁하는지; 표본0의 값0은 성능으로 해석하지 않음 |

바닥 edge 값은 reset의 수직·기본방향 상자를 기준으로 현재 최저 앞/뒤 edge 중점을 추적하는 진단이며 yaw/다른 축 회전에서는 단순 전진 proxy다. `since_reset_*`은 RSI가 박스를 미리 전진시킨 양을 제외한다. 현재 성공 조건/보상에 이 진단 값을 사용하지 않는다.

GPU6·2048환경·MAX_ITERATIONS2(실행 엔진상3epoch)에서 checkpoint 저장과 새 PUSH 진단37개×3회 유한값 기록 확인(`output/open_push_cube_low_train_check_v3/tensorboard_audit.json`). 보정 초기 손 거리 평균 약6.5cm이며 기존28cm 차이를 줄였다. checkpoint2환경/30step headless 복원 및 RSI 뷰어3sample/12step 확인 완료. 이 짧은 실행은 장기 정책 성공/AMP 적합성을 입증하지 않는다.

Actor/Critic의 실제 attention 가산 bias는 `relation_attention/{actor,critic}/edge_bias/{open,push,at}/{mean,rms,sample_count}`로 기록한다. 기존 진단 주기(기본100 forward)에 canonical graph의 유효 edge를 최대32 PPO 표본에서 추출하고 layer/head/edge 전체의 부호 평균·RMS를 계산한다. AT는 PUSH의 박스→목표 관계다. 표본이 없는 관계는 count/mean/rms가0이므로 count와 함께 해석한다. semantic-only 기존5과제도 holding/sit/climb/at/ontop 같은 이름으로 기록한다. 프로세스에 이미 import된 코드에는 자동 적용되지 않고 다음 실행/checkpoint 재개부터 적용한다. 기존 PID의 종료·재시작은 하지 않는다.

사용자 선택에 따라 기존 Unified처럼 사람2명 각각의 task를 매 reset마다 독립 샘플링한다. `relationGraph.shuffle_token_order: true`로 actor/critic의 매 forward마다 H2/O4/G2 전체8개 토큰 순서를 무작위로 섞는다. 토큰·관계 bias·GTA 위치/회전 정보를 함께 순열하고 출력 전에 복원하므로 담당 대상과 사람별 action/value 순서는 유지된다. OPEN50%·PUSH50%이며 OPEN/OPEN·OPEN/PUSH·PUSH/OPEN·PUSH/PUSH 모두 가능하다. 하나의 Actor/Critic·PPO 학습에서 graph가 담당 물체와 task를 알려준다. 기존 Unified와 같은 Actor/Critic별 별도 EdgeEncoder를 사용한다. 기존 HOLDING/SIT/CLIMB/AT/ON_TOP 및 Door Hold 이외의 문 단독 설정은 그대로 둔다.

환경 `HumanoidMAOpenPush`는 사람2·문2·박스2(총물체4)다. 문은 현재 Door Hold의 전용 `door_hold.urdf`·닫힘 spring·마찰0을 재사용하며90도(π/2rad) 이상 개방한 상태에서 손으로5초 지지해야 현재 성공이다. 유지 중 골반 높이는 asset 정상 선 자세의70% 이상, torso 수직축 기울기는40도 이내, 최소 한 발은 바닥 근처에서 위쪽 지지력을 내야 한다. 손 접촉 공백은0.1초까지 허용하고 이탈하면 유지 시간이 초기화된다. PUSH 박스는0.8×0.6×1.3m·20kg·마찰0.6이며 초기 위치에서+Y로1.5m 떨어진 고정 목표까지 민다. 목표XY15cm 이내·초기 바닥 높이±8cm·기울기0.35rad 이내·이동 속도0.2m/s 이하·각속도0.3rad/s 이하가 성공 조건이다. 담당 손이 박스 뒤 표면 근처에서 밀기 반력과 박스의 전방 반력을 내며 목표 쪽으로5cm 이상 전진한 접촉 이력이 추가로 필요하다. GPU net contact force 기반 공간/힘 판정이며 collision pair/grasp constraint는 아니다. 박스를 직접 움직이는 actuator는 없다.

사람별 초기화는 보행100%·`Start`, 담당 문 손잡이/박스 뒤면에서1.5~2.5m 떨어져+Y를 보도록 한다. 문/박스는 고정된 두 작업 공간에 함께 존재한다. graph는 OPEN의H→Door 또는 PUSH의H→Box 및Box→Goal을 사용한다. semantic packet20·intrinsic O40·총관측684차원이다. task 보상은 state0.4/progress0.2/success0.2와 기존 자기0.9/동료0.1 공유를 유지한다. 성공은 현재 상태이며 성공만으로 episode를 종료하지 않는다.

AMP는 담당 graph task를 따르는 조건부 단일 판별기다. 프레임당 human129+OPEN/PUSH one-hot2=131차원,10frame 총1310차원이다. 기존3-family AMP 기본 동작은 유지하고 label 크기만 선택 가능하게 확장했다. expert/replay는 해당 task의 실제 모션끼리 매칭한다.

| Task | AMP 전문가 | 초기화 |
| --- | --- | --- |
| OPEN | 보행4개33% + 기존 BONES 왼손 문 열기1개67% | 보행4개·Start |
| PUSH | 보행4개33% + 선택한 밀기2개67%(두 clip 동일 weight·각33.5%) | 보행4개·Start |

토큰 섞기 적용 후 CPU28개 통과:4가지 task 조합의 actor/critic 출력·gradient 순열 불변 및 자동 randperm 경로 확인. GPU6 비학습2환경 reset·AMP·actor/critic·4가지 task 조합 시뮬레이션 통과(`output/open_push_shuffle_check_gpu6/audit.json`). shuffle 설정도 checkpoint 계약에 기록한다. 이번 변경에서 학습은 실행하지 않았다.

원래 `open_push_unified_amp`는 loco100%·Start 초기화를 유지한다. 보행50%·조작50%와 후보 뒤쪽30%를70% 확률로 요청하는 별도 실행은 아래 OPEN/PUSH Late RSI 절을 참조한다. 초기화 비율과 AMP 전문가33/67 비율은 별개다.

현재33/67 비율에서 관련 CPU16개 통과(`/tmp/open_push_amp33_cpu.log`); 새 GPU/학습 실행은 하지 않았다. 두 task 모두 보행/조작 전문가를33/67로 맞췄다. OPEN/PUSH50/50 기준 명목 `skillDiscProb`는 `[0.33,0.335,0.335]`(loco/doorOpen/push)다. expert 비율은 checkpoint 계약에 포함되므로 이전 문100%·PUSH30/70 확인용 checkpoint의 직접 resume/test는 거부된다. 새 비율로 scratch 학습한다.

[dataset_open_push.yaml](../tokenhsi/data/dataset_open_push.yaml)은 기존 Door manifest의 보행4개·문1개와 새 밀기2개만 읽는다. 밀기는 `push_obstacle_180_101__A341`의0–12.2초(367frame), `push_obstacle_180_003__A342`의0–11.7초(352frame)를30FPS humanoid15body/32DOF로 변환했다. 준비와 지속 밀기를 남기고 손을 빼는 후반은 제외했다. 원본 두 BVH는 다운로드된 archive에서만 선별 추출했다. source에는 박스 궤적/접촉이 없으므로 human AMP만 사용하며 박스 RSI는 추가하지 않았다. 변환 보고서는 `tokenhsi/data/dataset_bones_push/motions/<clip>/phys_humanoid_v3/conversion_report.json`이다.

GPU는 사용자 지정 **6번만** 사용한다. 새 train/test/VNC/검증 wrapper는GPU6을 기본값으로 하고 다른 숫자를 거부한다. 기존 실행과 checkpoint는 변경하지 않는다. 문/박스 물리·expert manifest·task 비율·초기화·모델 공유 설정을 별도 checkpoint 계약으로 저장하므로 기존 Door/Unified checkpoint와 직접 resume/test할 수 없다. 기본 output은 `output/open_push_unified_amp`다.

```bash
# CPU 모션 변환 재생성: 원본 두 BVH 추출이 완료된 상태
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/convert_bones_push.py'
# 비학습 loader/reset/AMP/actor/critic/4가지 task 조합 검증
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_check_gpu6 bash tokenhsi/scripts/multi_agent/validate_open_push.sh
# scratch 본학습: 실행 전 nvidia-smi와 셸의 MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT 확인
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_train.sh 2 2048 4
# 짧은 확인용 학습: 본학습 output과 분리
TOKENHSI_GPU=6 MAX_ITERATIONS=2 OUTPUT_PATH=output/open_push_unified_amp_check bash tokenhsi/scripts/multi_agent/open_push_unified_amp_train.sh 2 2048 4
# 공동 정책 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <checkpoint.pth> 2 2 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_vnc.sh <checkpoint.pth> 2 1 4 10
# 평가에서 조합 고정: open / push / open_push / push_open (기본 random_open_push)
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=open_push bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <checkpoint.pth> 2 2 4 1
```

초기 전문가 비율(OPEN문100%·PUSH보행30/밀기70%)에서 CPU34개·GPU6 비학습2환경 검사 통과. PUSH 변환 관절 제한 위반0·최대 DOF속도6.95rad/s 미만·end-effector fit 최대1.55cm·발 mesh최소5mm 확인. reset30회에서두 사람 모두 OPEN/PUSH 샘플링 및4가지 조합·partial reset 보존·task별 AMP label·actor/critic/discriminator 유한값 확인. GPU6·2048환경·`MAX_ITERATIONS=2` 확인용 PPO/AMP 실행 및 checkpoint 저장, 같은 checkpoint의2환경/30step headless 평가 복원 완료. 실행 엔진은 max_epochs를 넘는 시점에 멈추므로 이 확인은3epoch까지 진행했다. 장기 본학습·학습된 정책의 성공·GUI/VNC 실행은 미검증이다. 비학습 결과 `output/open_push_check_gpu6/audit.json`, 확인용 checkpoint `output/open_push_unified_amp_check/OpenPushUnifiedAMP_07-16-40-29/nn/OpenPushUnifiedAMP.pth`, 로그 `/tmp/open_push_{cpu_tests_gpu6,sim_gpu6,train_check_gpu6,checkpoint_gpu6}.log`.

90도·자세 조건은 단독 Door Hold와 공동 OPEN에 동일하게 적용된다. 공동 PUSH 보상은 바꾸지 않았다. 자세가 무효이면 OPEN 손 상호작용·각도 진행·유지 진행·성공 보상을 지급하지 않고 접근 보상만 남긴다. 자세 설정은 doorHold와 checkpoint 계약에 포함되므로 이전 조건의 checkpoint는 직접 resume/test 계약과 다르다.

## OPEN/PUSH Late RSI — PD 후보 후반 시작

현재 양손 PUSH bank는 `tokenhsi/data/open_push_late_rsi_cube_low_bank.json`이며 출력 기본값은 `output/open_push_late_rsi_cube_low`다. GPU6 비학습2환경12step PD 검사에서 OPEN13/13·PUSH145/171개(clip별69/76개) 통과,1000reset 조작48.6%·후반 요청68.72%·부분 reset/고정 목표/보정 AMP history 유한값 확인(`output/open_push_cube_low_rsi_check/audit.json`). 후보 screen은 양손 근접·발 지지/안정성을 검사하며 양손 접촉력을 지속적으로 내는 정책 성공 검증은 아니다. 후보 계약revision3에 어깨 회전 retarget 방식을 포함하며 이전 크기/한 손/원본 손Z bank는 직접 사용할 수 없다. 이전 양손1.05m bank `open_push_late_rsi_two_hand_bank.json`와166개 통과 결과는 역사 artifact로 남긴다.

1.3m 높이만 변경했던 당시 bank는 `tokenhsi/data/open_push_late_rsi_h13_bank.json`이다. 재검증에서 OPEN13·PUSH155개 통과,1000reset 조작48.6%·후반 요청68.72%,partial reset/고정 목표/AMP 통과(`output/open_push_late_rsi_h13_check/audit.json`). 이전1.1m bank·확인용 checkpoint는 역사 artifact로 남기며 현재 크기 계약에는 사용할 수 없다. 당시 RSI 목표 손 높이는 실제 뒤면의 바닥+5cm~상단-5cm 범위로 clamp했다. 현재는 양손 목표높이0.8m와 어깨 회전 RSI를 쓴다. 기존0.55m 고정 중심 높이를 사용하던 PD 감사도 config의 절반 높이를 따르도록 수정했다.

사람·문·박스의 RSI 초기화를 눈으로 확인할 때는 checkpoint 없는 전용 검사 뷰어를 쓴다. 기본 `late` 모드는 사람0 OPEN/사람1 PUSH를 조작100%·후반 요청100%로 표시한다(검사 프로세스에서만 강제하며 학습 YAML/bank는 수정하지 않음). 모션clip/time/원본phase/후반 요청 여부가 화면에 표시되고 `latest_sample.json`·`samples.jsonl`에 기록된다. 한 physics frame 뒤 정지하며 **학습된 정책 행동이 아니다**. Space는 sampled DOF를PD로 최대12step 유지한다. 성공/장시간 안정성 확인과 구분한다.

PUSH 양손 목표는 오른손 빨강·왼손 청록 cross로 표시한다. `RSI_VIEW_FOCUS=1`로 PUSH 사람 카메라에서 시작할 수 있다.

키: `N` 새 샘플, `Space`12step PD 실행/일시정지, `C` 다른 사람 카메라, `L` 후반 조작 강제, `M` 학습의50/50·후반70% 혼합, `O` OPEN/OPEN, `P` PUSH/PUSH, `B` OPEN/PUSH. 카메라는 마우스로 조정할 수 있다. 일반 `open_push_late_rsi_test/vnc.sh`는 보행 시작 정책 평가라서 RSI 자세 검사용으로 쓰지 않는다.

```bash
# 로컬 RSI 초기화 검사(1환경, checkpoint 불필요)
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_rsi_view.sh
# 서버 VNC: VS Code PORTS에서6086 포워딩 후 localhost:6086/vnc.html 접속
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/open_push_rsi_view_vnc.sh
# 화면 없는3sample/12step PD 검사
TOKENHSI_GPU=6 HEADLESS=1 RSI_VIEW_CYCLES=3 bash tokenhsi/scripts/multi_agent/open_push_rsi_view.sh
```

검증 결과: CPU35개 통과. GPU6·비학습2환경에서 문13개·PUSH155개(clip별72/83)의12step PD 후보를 확보했다. 문 통과 후보의 원본phase는 약0.186~0.657이며 마지막30% 구간이 아니라 이 **유효 후보의 뒤쪽**을 우선 선택한다. PUSH는 원본 후반phase도 포함한다. 1000reset 실측 조작48.9%·조작 중 후반 요청69.33% 및4과제 조합·partial reset·고정 목표·AMP history 검증 통과(`output/open_push_late_rsi_reset_check/audit.json`). GPU6·2048환경 확인용3epoch 학습·checkpoint 저장·2환경30step 보행 시작 평가 복원 및XY/edge bias/RSI TensorBoard 생성 확인. 정책 성공·장기 학습·GUI/VNC는 미검증이다. 본학습은 실행하지 않았다.

`open_push_late_rsi.yaml`은 기존 공동 과제의 보상·90도/자세/5초 유지·AMP33/67·토큰 순열·모델을 유지한다. 예전 task별 보행/조작50/50 혼합과 Door/ksh 후보 후반 우선 방식을 참고해 사람별 OPEN/PUSH task에 맞는 초기화를 적용한다.

| Task | 학습 초기화 | 조작 후보 시간 선택 |
| --- | --- | --- |
| OPEN | 보행50%·doorOpen50% | 각 clip의 PD 통과 후보 뒤쪽30%를70% 확률로 요청; 나머지30%는 전체 통과 후보 |
| PUSH | 보행50%·push clip1 25%·clip2 25% | 각 clip 동일 비중, 위와 같은 후반 우선 선택 |

전체 초기화 중 후반 요청은35%다. 전체 후보에서 고르는15%에도 후반이 포함될 수 있어 실제 후반 선택 비율은35%보다 높을 수 있다. 후보 뒤쪽30%는 원본 모션의 마지막30%와 다르다. 보행은 기존0초 Start·1.5~2.5m 시작을 유지한다. 평가/test/VNC는 학습 RSI를 끄고 보행100% Start로 접근부터 평가한다.

사람 reference에 맞춰 OPEN은 문각도 `0.4+1.2*원본phase`rad의 손잡이 앞6cm에 손을 정렬하고 사람 전체 위치/회전/속도를 함께 변환한다. PUSH는 고정1.5m 목표 거리 중 `0.85*원본phase`만큼 박스를 미리 전진시키고 뒤면의 양손 목표점 앞6cm에 손 pair XY를 정렬한다. 원본에 물체 궤적이 없어 이 각도/위치는 **기하 proxy**이며 BONES에서 측정된 문/박스 상태가 아니다. 목표와 reset 기준 박스 위치는 이동 전 원래 상태를 유지한다. 문 유지 시간·접촉 유예/이력·박스 성공 eligibility는0부터 시작한다. 기존 성공 조건을 통과하는 실제 접촉/5초 유지가 필요하다.

`build_open_push_rsi.sh`는 원본모션 파일 hash·문/박스 물리·보상/어깨 retarget 방식에 묶인 `tokenhsi/data/open_push_late_rsi_cube_low_bank.json`을 만든다. geometry/AMP history 후보를12 control step(약0.4초)의 고정 reference DOF PD로 검사해 유한값·넘어짐/높이·root drift45cm 미만·발 지지·OPEN 손잡이20cm/PUSH 양손 목표 각각18cm 접근·PUSH 바닥 높이를 확인한다. OPEN은 현재 문 선 자세 판정으로 지지를 확인하고 PUSH는 별도의 foot body35cm/위쪽힘1N 기준을 쓴다. 각 clip에 통과 후보가 없으면 실행을 거부한다. 이는 짧은 reset 안정성 검사이며 접촉 유지·학습된 정책 성공을 보장하지 않는다.

기존 실행용 env는 변경하지 않았다. 새 env/output·RSI spec/후보 bank hash가 checkpoint 계약에 추가되므로 원래 loco-only checkpoint를 이 설정으로 직접 resume/test할 수 없다. 새 scratch 학습이며 기존 실행 PID를 종료/재시작하지 않는다. train/network YAML은 기존 공동 실험의 것을 재사용한다. 새 학습에서도 앞서 추가한 과제별 XY 접근·edge bias 로그가 기록된다. `relation/90_debug/{open,push}/rsi/{reset_count,interaction_count,late_request_count}`는 rollout 간 reset 표본 수다.

```bash
# 실행 전 GPU6 점유 확인; 기존 학습 유지
nvidia-smi
# 비학습2환경 후보 생성/PD검사·학습 reset 경로/AMP 감사
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_late_rsi_check bash tokenhsi/scripts/multi_agent/build_open_push_rsi.sh
# 새 scratch 본학습(자동 실행하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 별도 짧은2048환경 학습 확인
TOKENHSI_GPU=6 MAX_ITERATIONS=2 OUTPUT_PATH=output/open_push_late_rsi_train_check bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 로컬 viewer / 화면 없는 평가 / 서버 VNC: 보행 Start로 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <checkpoint.pth> 2 2 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_vnc.sh <checkpoint.pth> 2 1 4 10
```

## Door Open/Hold — 1인 밀기·지지

`HumanoidMADoorHold`는 사람1·문1의 별도 과제다. 90도(π/2rad) 이상 열고 자기 손을 손잡이12cm 이내에 두며 문을 밀어 받치는 접촉이5초 이어지면 현재 성공으로 판정한다. 접촉에는 손의 반력과 문짝의 개방 방향 힘이 각각0.5N 이상 필요하다. 손과 발 지지력을 각 physics tick에서 확인한다. 발은 해당 control step 내의 접촉 충격을 수집하고 다음 step으로 이월하지 않는다. 손은 최대0.1초의 짧은 접촉 공백을 허용한다. 손이 멀어지거나 공백이 길어지면 타이머와 현재 성공을 초기화한다. 성공 후에도 지지를 유지해야 한다. 추가 자세 조건은 골반 높이≥asset 정상 선 자세의70%, torso의 수직축 기울기≤40도, 최소 한 발의 body 높이≤18cm 및 위쪽 접촉력≥1N이다. 정상 높이는 reset/RSI 자세가 아닌 고정 asset 초기 자세에서 가져온다. 자세 조건이 깨지면 유지 시간을 즉시0으로 초기화하고 손 상호작용·열기 진행 보상을0으로 만든다. 접근 보상은 남긴다. 양손 중 어느 손이든 인정하며 몸통-문 접촉 금지·통과 판정·추가 넘어짐 종료는 적용하지 않았다. target_angle=π/2, angle_tolerance=0으로90도 미만 성공을 허용하지 않는다. 손가락 grasp constraint는 없으며 agent2의 운반/통과도 아직 없다.

전용 `door_hold.urdf`는 바닥과 문짝 사이4cm 틈을 두어 바닥 마찰에 걸리지 않도록 했다. 손잡이는 높이1m이며 기존 push 문 방향을 유지한다. 경첩 마찰0·감쇠0.5에 환경 닫힘 토크 `clamp(-2*angle-0.5*velocity, -4, 4)`를 적용한다. 정책은 사람 관절32개만 제어한다. 기존 문 asset/실험의 경첩 설정은 보존한다.

관측은288차원이며 물체 특징O38에 지지 시간·접촉 상태·마지막 접촉 후 시간이 포함된다. 사용자 요청으로 AMP 전문가를 BONES 왼손 open/walk100%(loco0%)로 변경했다. `skillDiscProb`와 `ampExpertMapping.OPEN`은 둘 다 `[0,1]`이다. RSI는 loco-only·`Start`를 유지하며 닫힌 문 손잡이 앞1.5~2.5m에서 매 reset마다 균등하게 거리를 뽑는다. 학습·평가 모두 적용하며 `doorSpawnDistanceRange: [1.5,2.5]`가 범위, `doorApproachAMP.spawn_distance: 2.5`는 재배치 기준인 상한이다. 문을 향한 방향·좌우 정렬은 유지한다. loco 파일은 초기화용으로 manifest에 남긴다. 긴 유지 동작의 전문가를 추가한 상태는 아니다. 새로운 관측·인원수·물리 계약 때문에 기존 checkpoint를 직접 사용할 수 없고 별도 output에서 scratch 학습해야 한다. 비율·시작 거리 범위도 계약에 포함되어 이전 고정 거리/30·70 설정 checkpoint와 직접 resume/test할 수 없다. 학습2048환경; 아래2환경 검사는 물리·loader 검증이며 학습된 정책의 성공을 의미하지 않는다.

```bash
# 비학습 검증: GPU6 / CPU 실제 손-문 collision pair
TOKENHSI_GPU=6 OUTPUT_PATH=output/door_hold_check_gpu bash tokenhsi/scripts/multi_agent/validate_door_hold.sh
TOKENHSI_GPU=6 AUDIT_SIM=cpu OUTPUT_PATH=output/door_hold_check_cpu bash tokenhsi/scripts/multi_agent/validate_door_hold.sh
# scratch 학습 (현재 실행하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_train.sh 1 2048 1
# 학습 후 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_test.sh <hold_checkpoint.pth> 1 1 1 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_test.sh <hold_checkpoint.pth> 1 16 1 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_vnc.sh <hold_checkpoint.pth> 1 1 1 10
```

기본 output은 `output/door_open_hold_amp`다. 유지 시간은 env YAML의 `doorHold.duration`으로 바꾼다. 실행 전 GPU6 점유 및 `MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT` 잔여값을 확인한다. 검증 결과는 `output/door_hold_check_{cpu,gpu}/physics_audit.json`을 확인한다.

## BONES Door — 왼손·왼쪽 개방 AMP

`door_open_bones_left_amp`는 기존 분리형 `door_open_unified_approach_amp`의 Door/PPO/reward·닫힌 문·1.5m loco reset을 유지한다. OPEN 전문가만 새 [dataset_door_bones_left.yaml](../tokenhsi/data/dataset_door_bones_left.yaml)의 loco30%/doorOpen70%로 바꿨다. `skill=[loco,doorOpen]`, 초기화 확률 `[1,0]`, expert `[0.3,0.7]`, gate 비활성/full AMP다. 비율은 최초 설정이며 학습으로 검증한 최적값이 아니다. 기존 YAML·output·checkpoint는 보존하고 새 manifest hash 계약으로 이전 loco/proxy checkpoint의 직접 resume/test는 거부한다.

원본 `inside_door_handle_right_side_open_walk_R_001__A515_M.bvh`는 **왼손→왼쪽** 반전 motion이다. Uniform77관절→phys_humanoid_v3의15body/32DOF, cm/Y-up→m/Z-up, constrained IK·손 고정·발 mesh grounding을 거쳐0–3.4초/30FPS/103frame으로 변환했다. 원본 full passage 대신 reach/open/초반 전진을 남겼다. source에는 문 위치/각도/접촉이 없으며 **human AMP reference로만 사용**한다. 문 상태는 simulator와 OPEN reward가 제공하며 source 문 RSI/그립 constraint를 새로 만들지 않았다.

CPU40개·GPU6 비학습2환경 loader/AMP64×1290·FK·52frame 재생 통과. 무접촉 Gym/Poselib 차이최대2.2mm, 문을 겹친 replay 차이최대9.5cm이다. CPU [문 비교 HTML](../tokenhsi/data/dataset_bones_door/motions/inside_door_handle_right_side_open_walk_R_001__A515_M/phys_humanoid_v3/door_preview.html)·GIF와 [검증 보고서](../output/door_bones_left_amp_check/REPORT.md)를 먼저 확인한다. preview의 문 각도는 손 방향으로 추정한 것으로 전문가/접촉 정답에 포함하지 않는다. GPU6 offscreen graphics 초기화는 실패했고 GPU0으로 대체하지 않았다. 아래 학습·policy test/VNC wrapper는 준비했으나 본학습·정책 성공·GUI 실행은 미검증이다.

```bash
# 원본 추출 완료 상태에서 CPU 변환 / 문 형상 비교 미리보기 재생성
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/convert_bones_door.py'
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/build_bones_door_preview.py'
# GPU6 전용 비학습 검증. CAPTURE_BONES는 지정하지 않는다.
TOKENHSI_GPU=6 OUTPUT_PATH=output/door_bones_left_amp_check bash tokenhsi/scripts/multi_agent/validate_bones_door_amp.sh
# 본학습: 별도 output, 2048환경, scratch. 이번 작업에서는 실행하지 않음.
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_train.sh 2 2048 2
# 해당 실험에서 새로 학습한 checkpoint만 사용
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_test.sh <bones_checkpoint.pth> 2 1 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_test.sh <bones_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_vnc.sh <bones_checkpoint.pth> 2 1 2 10
```

기본 output은 `output/door_open_bones_left_amp`다. 실행 전에 `nvidia-smi`와 `MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT` 잔여값을 확인한다. 변환 자체는 CPU이며 runtime을 통해 GPU6만 노출한다. 공유 원본 데이터에는 쓰지 않는다.

## Door unified / shared-edge — 같은 Door proxy AMP·RSI 비교

사용자 최종 선택은 기존 `approach_scenario_stage1_unified`와 `unified_shared_edge_encoder`의 비교다. 기존 unified는 **Actor/Critic별 별도 semantic EdgeEncoder**, shared-edge는 **같은 EdgeEncoder 파라미터 공유**다. 두 방식 모두 tokenizer·GTA Transformer·출력 head는 Actor/Critic별 별도다. ksh size RSI의 기본 semantic network도 별도 encoder이지만 이번 비교에 ksh 독립 코드나 typed bias/message를 추가하지 않는다.

새 `door_open_unified_amp_on` / `door_open_shared_edge_amp_on`의 env YAML은 동일하며, train YAML은 `transformer.share_edge_encoder: false/true`만 다르다. Door observation/reward/success·PPO·AMP·RSI는 동일하다. 기존 Door `amp_on/off` 실험은 유지한다.

실제 Door reference dataset이 없는 상태에서 [dataset_door_proxy.yaml](../tokenhsi/data/dataset_door_proxy.yaml)은 기존 loco·pickUp·carryWith 파일만 읽는다. **이들은 box 동작에서 고른 대체 참조이며 실제 문 열기 전문가가 아니다.** sit/climb/putDown·OMOMO 혼합은 새 두 실험에서 제외한다. AMP 전문가 비율은 loco/pickUp/carryWith=20/70/10%, RSI 초기화는30/60/10%다. `doorReference`는 pickUp/carryWith의 root 높이≥0.85m, root XY 속도≤0.8m/s, 양발 높이≤0.2m, 적어도 한 손의 높이1.0±0.25m·root heading 앞방향 reach≥0.2m를 검사한다. 전체 10프레임 AMP history가 조건을 만족하는 후보만 사용한다. 현재 참조 후보는 pickUp59·carryWith22개다.

RSI는 같은 후보에서 각 clip의 유효 후보 뒤쪽30%를70% 확률로 요청한다. 이는 ksh의 후반 후보 우선 방식에 맞춘 것이며, box 크기별 PD 후보 bank를 Door에 그대로 적용한 것은 아니다. Human을 +Y로 향하게 회전시키고 조건에 맞는 손을 **자기 handle XY에서 앞쪽8cm**에 정렬한다. root/손 높이는 원본을 유지하고 몸·root의 위치/회전/속도를 일관되게 변환한다. 문은 닫힌 상태로 별도 reset하며 binding은 바꾸지 않는다. loco 시작은 자기 문 앞에서 기존 loco 상태를 사용한다. 평가는 기존처럼 loco 시작이며 학습 RSI 시작 성공률과 구분한다.

```bash
# 동일 seed·2 humans·2048 학습환경·2 doors. 이번 작업에서는 실행하지 않음.
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_train.sh 2 2048 2
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_train.sh 2 2048 2

# 로컬 viewer / headless 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_test.sh <checkpoint.pth> 2 16 2 3

# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_vnc.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_vnc.sh <checkpoint.pth> 2 1 2 10
```

학습 전 `nvidia-smi`로 GPU6 점유와 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다. 기본 output은 `output/door_open_unified_amp_on`, `output/door_open_shared_edge_amp_on`이다. 같은 방식·같은 Door reference 계약의 checkpoint로만 resume/test한다. checkpoint에는 공유 여부·reference 조건·expert/RSI 비율·motion manifest SHA256을 기록해 혼용을 거부한다. 추후 실제 Door motion을 기존 MotionLib 형식으로 등록하고 manifest·expert row·reference 조건을 교체할 수 있다. TRUMANS 원본과 기존 ksh 이식본은 수정하지 않았다. 긴 학습의 문 열기 성능 개선은 아직 확인하지 않았다.

비학습 CPU physics check는 아래 Door ON/OFF 섹션의 명령에서 env/train YAML을 위 두 파일로, motion 파일을 `tokenhsi/data/dataset_door_proxy.yaml`로 바꾸면 된다. 새 reference 설정에서는 실제 interaction RSI 경로도 검사한다. CPU 회귀는 기존 Door 테스트와 `tokenhsi/tests/test_door_reference.py`를 함께 실행한다.

## Door 당김 방향 — 별도 approach AMP 설정

`door_pull_unified_approach_amp`는 기존 approach-only에서 경첩 축을−Z로 반전하고 handle offset을 `(0.78, -0.25, 1)`로 변경한다. 사람은 손잡이 앞1.5m·+Y 방향이며 문은 사람 쪽(−Y)으로 열린다. DOF 양수는 개방량으로 관측/reward에 사용하고 panel 물리/초기 관측 회전만 부호를 반영한다. 기존 밀기 설정과 checkpoint/output은 분리하며 밀기 checkpoint의 resume/평가는 거부한다.

물리 검증은 CPU/GPU 각2평가환경·한 코어·1스레드로 통과했다. 실제 손–손잡이 접촉, 무접촉20초, 접촉 해제4초, 관측/전체 reward 대조 결과는 [검증 보고서](../output/door_pull_physics_audit_20261006/REPORT.md)에 있다. 손은 sphere이고 grasp 제약은 없어서 스크립트 접촉으로 당김 방향의 힘 전달을 확인한 것이며 정책 성공/실제 grasp 검증은 아니다. 이 설정의 학습은 실행하지 않았다.

```bash
# 물리 검증만(각각 한 코어·1스레드, 학습 없음)
TOKENHSI_GPU=6 AUDIT_SIM=cpu DOOR_EXPERIMENT=door_pull_unified_approach_amp OUTPUT_PATH=output/door_pull_physics_check_cpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
TOKENHSI_GPU=6 AUDIT_SIM=gpu DOOR_EXPERIMENT=door_pull_unified_approach_amp OUTPUT_PATH=output/door_pull_physics_check_gpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
# 본학습(2048환경, scratch; 이번 작업에서는 실행하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_test.sh <pull_checkpoint.pth> 2 1 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_test.sh <pull_checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_vnc.sh <pull_checkpoint.pth> 2 1 2 3
```

## Door Open — 기존 unified + 전 구간 loco AMP

### Door 추가 물리 검증 (학습 없음)

아래 명령은 평가환경2개·CPU 한 코어 affinity·PyTorch/PhysX/BLAS 각1스레드로 제한한다. 무접촉20초 안정성, 스크립트 사람 접촉으로 passive hinge 개방, 물리/관측/전체 reward 일치를 검사한다. CPU에서는 실제 손–문 contact pair도 확인한다. 사람을 스크립트로 움직이므로 학습 정책 성공률 검증과 구분한다.

```bash
TOKENHSI_GPU=6 AUDIT_SIM=gpu OUTPUT_PATH=output/door_physics_check_gpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
TOKENHSI_GPU=6 AUDIT_SIM=cpu OUTPUT_PATH=output/door_physics_check_cpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
```

CPU/GPU 검증 통과 결과: [검증 보고서](../output/door_physics_audit_20261006/REPORT.md). 장기 학습 실행과 분리되며 checkpoint가 필요 없다.

### 2026-10-07 공유/분리 비교 — 같은 HOLDING 계수·전 구간 AMP

`door_open_unified_approach_amp`는 Actor/Critic EdgeEncoder **분리**(`share_edge_encoder: false`), 새 `door_open_shared_edge_approach_amp`는 **공유**(`true`)다. 환경 YAML은 같고 train YAML은 공유 플래그·실험 name만 다르다. 두 설정 모두 몸/손0.2·각도progress/success0.2·gate_enabled=false·loco expert/RSI·reset revision3·Start·task/disc0.5/0.5·PPO를 사용한다. output/checkpoint는 분리하고 공유 여부 계약이 다른 checkpoint의 resume/test는 거부한다. 아래 명령은 준비만 했으며 본학습은 실행하지 않았다.

```bash
# Actor/Critic EdgeEncoder 분리 — 터미널1
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_separate_full_amp_balanced_hinge bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# Actor/Critic EdgeEncoder 공유 — 터미널2
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_shared_full_amp_balanced_hinge bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <separate_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <separate_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_test.sh <shared_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_test.sh <shared_checkpoint.pth> 2 16 2 3
# 서버 VNC (포트 별도)
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <separate_checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_vnc.sh <shared_checkpoint.pth> 2 1 2 10
```

기본 shared output은 `output/door_open_shared_edge_approach_amp_holding_scale_full_amp_balanced_hinge`다. 기존 `door_open_shared_edge_amp_on`은 box proxy AMP/RSI 실험이므로 이번 loco-only 비교와 구분한다.

### 2026-10-07 현재 설정 — HOLDING 접근 계수·전 구간 AMP

사용자 요청으로 GPU6의 `reset_v2_run1/run2` 학습 두 개를 종료했다. 각 마지막 저장 checkpoint는 epoch10500이며 본학습은 재시작하지 않았다. 현재 미는 `door_open_unified_approach_amp`의 `state_reward_weight`는0.4로, 몸 접근·손 상호작용은 각각0.1→0.2이다. `doorApproachAMP.gate_enabled: false`로 거리 taper·손접촉/각도 latch를 비활성화해 에피소드 내내 AMP 가중치1을 사용한다. 접근 delta0.5/sigma1.0·손 거리 scale10·각도 progress0.2·success0.2·sharing0.9/0.1·AMP task/disc0.5/0.5·PPO/reset은 유지한다. 성공 시 edge 최대 보상은0.8이다. 다른 Door/당김·기존 HOLDING 설정은 유지한다.

기본 output은 `output/door_open_unified_approach_amp_holding_scale_full_amp_balanced_hinge`로 분리했다. 보상/gate 계약이 달라 기존0.1 또는 gate 활성 checkpoint는 현재 wrapper의 resume/test에서 거부한다. 아래는 새 계수·전 구간 AMP로 scratch 학습할 때의 명령이며 **이번 작업에서 실행하지 않았다**. 기존 checkpoint를 자동 변환하거나 덮어쓰지 않는다.

```bash
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 새 보상 계약의 checkpoint만 사용: 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <new_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <new_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <new_checkpoint.pth> 2 1 2 10
```

gate 수정 전 접근 정체 진단은 `output/door_approach_debug_20261007/REPORT.md`를 참조한다. 당시 기존 checkpoint를 저장된 원래 reward config로 비학습 평가했다. 이후 사용자 요청으로 거리 기반 AMP 보상 감소를 제거했으며, 전 구간 AMP에서의 학습 성능은 미검증이다. 새 설정의 물리/보상 검증은 `output/door_full_amp_check_20261007/`에 저장한다.

### 2026-10-06 reset·학습 이력 (이하 기존0.1 계수의 기록)

**아래 재개·평가 명령은 당시 이력이다. 현재0.2 계수 wrapper로 기존 checkpoint를 resume/test할 수 없다. 새 실행에는 위 현재 설정 명령을 사용한다.**

2026-10-06 미는 문 두 실행을 각각 epoch3000/frame393216000 checkpoint에서 resume했다. 원본 checkpoint는 보존하고 같은 run1/run2 output 아래 새 시간 디렉터리에 로그·checkpoint를 저장한다. Actor/Critic·AMP/discriminator·normalizer·optimizer·epoch/frame을 기존 restore 경로로 복원한다. 당김 asset/grasp는 사용하지 않는다. epoch3008 이후 진행을 확인했다. 운영 기록은 각 `latest_launch.json`, 검증/launch 기록은 `output/door_push_resume_20261006/`에 있다.

아래는 중단 후 재개용 명령이다. 같은 실행이 살아 있으면 중복 실행하지 않는다. 이후에는 각 실행의 가장 최신 checkpoint를 지정한다. CPU 사용을 줄이기 위해 OMP/MKL/OpenBLAS/NumExpr 연산 스레드를1로 제한하고 학습환경 수2048은 유지한다.

```bash
unset MAX_ITERATIONS
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run1 RESUME_CHECKPOINT=output/door_open_unified_approach_amp_reset_v2_run1/DoorOpenUnifiedApproachAMP_06-12-22-51/nn/DoorOpenUnifiedApproachAMP_00003000.pth bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 별도 터미널
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run2 RESUME_CHECKPOINT=output/door_open_unified_approach_amp_reset_v2_run2/DoorOpenUnifiedApproachAMP_06-12-22-51/nn/DoorOpenUnifiedApproachAMP_00003000.pth bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
```



현재 reset 수정본은 `stateInit: Start`로 네 loco clip의 시작 프레임을 사용한다. 무작위 시점 참조 유지 검사에서 일부 첫-step 물리 불안정성을 확인했고, 시작 프레임은 GPU2048환경×10회 모두 통과했다. own handle 앞1.5m·+Y 방향·env-local 관측, root/DOF 단일 setter·reset 중 즉시 refresh 제거·GPU index tensor 수명 보존을 적용한다. checkpoint `reset_revision: 3`와 `state_init: Start` 계약으로 수정 전 checkpoint의 평가/resume를 거부한다.

최종 재학습은 아래 별도 output에서 scratch로 실행한다(각 터미널에서 한 번씩):

```bash
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 별도 터미널
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run2 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 로컬 추론 / 서버 VNC (각 run의 수정 후 checkpoint)
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/door_reset_v2_eval bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 PORT=6086 OUTPUT_PATH=output/door_reset_v2_eval bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <checkpoint.pth> 2 1 2 10
```

운영 실행 기록은 각 output의 `latest_launch.json`·`launches/<날짜>/train.log`에 저장한다. 같은 실행이 살아 있으면 위 명령을 다시 실행하지 않는다. 검증 스크립트·로그는 `output/door_reset_v2_validation_20261006/`; 확인용 output은 `output/door_reset_v2_check`다.

2026-10-06 추가 수정: approach-only loco RSI의 사람 pose를 env-local own handle 앞1.5m로 재배치하고 +Y 방향으로 일관되게 회전한다. 이전에는 사람에만 arena origin이 더해져 다중환경에서 문과 멀어졌으며 단일환경 viewer로는 검출되지 않았다. 이 수정 전 시작한 `reset_fixed_run1/run2`에도 해당하므로 결과를 정상 학습 성능으로 해석하지 않는다. 새 프로세스부터 적용되며 기존 실행은 자동 갱신되지 않는다. GPU16환경 reset 검증 통과·수정 후 학습 성능은 미검증이다.

수정된 GPU reset으로 기존 두 실행과 같은 설정(seed42·2048환경·2Human/2Door)을 scratch 재학습하려면 서로 다른 터미널에서 아래 명령을 실행한다. 두 실행은 동일 설정 반복이며 ON/OFF 비교가 아니다. 이전 checkpoint는 resume하지 않는다.

```bash
# 터미널1
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_fixed_run1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 터미널2
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_fixed_run2 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
```


환경 reset만 확인하는 비학습 뷰어(체크포인트 불필요):

```bash
# 서버 VNC (6086 포트 전달 후 localhost:6086/vnc.html 접속)
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_reset_vnc.sh
# 로컬 화면
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_reset_view.sh
# GUI 없이 GPU reset 5회 확인
TOKENHSI_GPU=6 HEADLESS=1 DOOR_RESET_CYCLES=5 bash tokenhsi/scripts/multi_agent/door_reset_view.sh
```

실제 Door reset 후 참조 관절 자세를 PD target으로 1물리 step 적용하고, 그 화면을3초 유지한 뒤 재초기화한다. 학습 policy를 사용하지 않는다. terminal의 `DOOR_RESET_CHECK`는 reset 전후 root quaternion 유사도(1에 가까움)·골반 높이·문 각도를 출력한다. 초기 자세 적용 확인용이며 걷기 안정성·문열기 정책 성능 검증은 아니다. GPU6/1환경5회 headless 확인 통과. VNC에서 실제 viewer 생성·렌더링·reset2회 정상 종료를 확인했다. 로컬 viewer에는 DISPLAY가 필요하며, 없으면 안내와 함께 종료한다.

2026-10-06 GPU reset 진단: 사람과 Door의 indexed root/DOF setter를 통합했다. 이전 코드로 실행한 Door 학습은 사람 RSI root reset이 GPU 첫 step에 반영되지 않는 현상을 재현했다. 기존 실행에는 수정이 자동 적용되지 않으며, 이전 학습 current-success 지표를 정상 정책 성능으로 해석하지 않는다. 기존 5500 checkpoint는 수정 후 2환경 평가에서도 성공0·조기종료100%였다. 정상 reset에서의 재학습 성능은 미검증이다.

`door_open_unified_approach_amp`는 기존 unified의 Actor/Critic별 EdgeEncoder와 MAAgent PPO를 그대로 쓴다. H0→Door0/H1→Door1, Door 관측·성공 조건은 기존 Door 설정과 같으며 현재 몸 접근·손 상호작용 계수는 각각0.2이다. 실제 문 모션은 사용하지 않는다. `dataset_door_loco.yaml`의 기존 stand-to-walk·walk-to-stand·walk1 네 clip만 AMP expert와 loco RSI로 쓰고 box interaction proxy RSI는 제외했다. Human은 자기 handle 앞 1.5m에서 시작한다.

현재 push 설정은 `doorApproachAMP.gate_enabled: false`로 거리·손접촉·문각도와 무관하게 매 step AMP 가중치1을 사용한다. 에피소드 전 구간의 합산은 `0.5×task + 0.5×AMP`이며 discriminator·demo/replay·network는 유지한다. 기존 임계값은 spawn/reset 호환과 다른 설정을 위해 보존하지만 현재 push 보상에는 적용하지 않는다. 이 필드가 없거나true인 설정(당김 포함)은 기존1.2→0.6m taper·접촉/각도 latch를 유지한다. 실제 Door 모션 대신 기존 loco expert를 전 구간 사용한다.

TensorBoard의 `open/amp_weight`, `open/amp_active_fraction`은 현재 설정에서 모두1이어야 한다. checkpoint에는 gate·loco expert/RSI·manifest 해시·network 공유 여부를 기록한다. 기존 full-proxy AMP·접근0.1·gate 활성 checkpoint와 혼용하지 않는다. 현재 기본 출력은 `output/door_open_unified_approach_amp_holding_scale_full_amp_balanced_hinge`다.

```bash
# 점유 확인 후 본학습: 2Human / 2048환경 / 2Door
nvidia-smi -i 6
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <checkpoint.pth> 2 1 2 10
```

실행 전 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다. 2026-10-07 전 구간AMP 수정 후 CPU47개·GPU2환경 물리/보상 대조 및 근접/개방 상태의AMP1 검증을 통과했다. 이전 gated checkpoint의 GPU16환경 진단과 구분한다. 현재 설정의 학습 및 장기 성공률은 미검증이다.

## Door Open — AMP OFF / ON

`HumanoidMADoor`는 Stage 1 semantic 5필드 packet·shared EdgeEncoder·Actor/Critic Transformer와 MAAgent의 scene PPO를 재사용한다. `H0 --OPEN--> Door0`, `H1 --OPEN--> Door1` 고정 binding이며 각 문은 별도의 passive revolute hinge다. 협동 Door graph나 HOLDING 대체 relation을 사용하지 않는다. 두 config는 env의 `ampEnabled`, train의 `amp_enabled`만 다르고, 서로 일치해야 한다. 두 task multiplier는 기존처럼 0.5, ON의 AMP multiplier는 0.5다. OFF에서는 style reward·discriminator forward/loss·expert demo/replay 처리를 생략한다. PPO 설정과 network 구조는 동일하다.

Object node는 기존 O30에 hinge angle·angular velocity·panel 좌표계 handle XYZ를 추가한 O35다. 문 pose는 기존 GTA pose7로 전달한다. Human223·Goal1·6개 token·semantic suffix10을 유지하여 scene 관측은 570차원이다. 문 상태는 `door_hinge_angle`, `door_hinge_velocity`, `door_handle_position`, `door_pose`로 접근한다. `reset_door_angles(env_ids, angles)`는 `[B,2]` 문 각도만 바꾸고 binding은 유지한다. 기본 reset은 0rad, hinge 제한은 [0,1.7]rad다.

Reward는 owner의 root→handle XY 접근(기존 거리 progress 형태), 가까운 손→handle 지수 상호작용, `clamp(angle/1.2,0,1)` 열기 progress, 목표각 1.2rad·허용오차 0.05rad의 current success로 분리한다. approach/interaction은 각각 최대0.1, progress/success는 각각 최대0.2로 기존 edge 최대0.6을 유지한다. 현재 성공 중 term을 포화하며 성공이 풀리면 포화도 풀린다. 각 edge는 자기 문만 계산하고, 이후 기존 자기0.9·동료0.1 공유를 적용한다. power·Human collision penalty는 유지하고 box speed penalty는 사용하지 않는다. 성공 조기종료는 하지 않는다.

TensorBoard `reward_terms/door_approach`, `door_interaction`, `door_progress`, `door_success`, `total`에서 term을 확인한다. 네 Door term은 공유 전 owner 값이며 total은 공유·penalty 이후다. `open/hinge_angle`, `open/hinge_velocity`, `open/current_agent_success`, `open/current_scene_success` 진단도 기존 TensorBoard 경로로 전달한다. 평가 JSON은 기존 `metrics/edge_context_*.json`에 agent·scene·open_0/open_1 성공률을 기록한다.

AMP ON은 현재 `dataset_loco_sit_carry_climb.yaml`과 기존 MotionLib·AMP discriminator를 사용한다. Door motion 전문가는 아직 없으며 `ampExpertMapping.OPEN`은 기존 비조건부 expert 혼합의 명시적 설정이다. 향후 Door reference를 기존 motion YAML/skill 목록에 등록하고 이 row를 교체한다. TRUMANS repo·dataset은 이번 구현에서 사용하거나 수정하지 않았다. 기본 초기화는 기존 loco reference이며 기존 box/hybrid RSI 설정에는 영향을 주지 않는다.

```bash
# 학습 명령만 제공; 이번 작업에서는 학습하지 않음. 기본 2 2048 2.
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi bash tokenhsi/scripts/multi_agent/door_open_amp_off_train.sh 2 2048 2
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi bash tokenhsi/scripts/multi_agent/door_open_amp_on_train.sh 2 2048 2

# 로컬 viewer / headless 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_amp_off_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_amp_off_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_amp_on_test.sh <checkpoint.pth> 2 16 2 3

# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_amp_off_vnc.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_amp_on_vnc.sh <checkpoint.pth> 2 1 2 10
```

CPU 단위 검증은 `PYTHONPATH=tokenhsi python -c 'import isaacgym; import pytest; raise SystemExit(pytest.main(["-q", "tokenhsi/tests/test_door_open.py"]))'`로 실행한다. 실제 CPU PhysX 초기화·8step·부분 reset·passive hinge 회전·handle 위치·AMP demo를 확인하는 비학습 스크립트도 제공한다:

```bash
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_door_open_cpu.py \
  --task HumanoidMADoor \
  --cfg_env tokenhsi/data/cfg/multi_agent/door_open_amp_off.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/door_open_amp_off.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs 2 --num_agents 2 --num_objects 2 --headless --no_video \
  --test --eval --task_graph open --sim_device cpu --pipeline cpu --rl_device cpu
'
```

ON 검증은 위 두 YAML 이름의 `amp_off`를 `amp_on`으로 바꾼다. 이 스크립트는 CPU 평가 환경만 생성하고 trainer·학습은 실행하지 않는다.

Output은 각각 `output/door_open_amp_off`, `output/door_open_amp_on`이다. 기존 box checkpoint와 schema·Object tokenizer·relation embedding 크기가 다르므로 Door scratch checkpoint를 사용한다. 학습 resume는 같은 AMP 모드와 graph·reward 계약만 허용한다. GUI/VNC·GPU pipeline·장기 성공률은 별도 검증 대상이다.

## ksh 세 방식 독립 이식

기존 core를 유지하고 별도 `ports/ksh_stage1` 코드·config를 실행한다. 기본 output은 `output/ksh_size_rsi`, `output/ksh_size_rsi_typed_bias`, `output/ksh_size_rsi_typed_bias_message`다. 같은 방식 checkpoint로만 평가/resume한다. 데이터·캐시·테스트 경로는 [이식 안내](ksh_stage1_port.md).

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_vnc.sh <checkpoint.pth> 2 1 4 10

TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_vnc.sh <checkpoint.pth> 2 1 4 10

TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_vnc.sh <checkpoint.pth> 2 1 4 10
```

## 후반 hybrid RSI

별도 scratch 실험으로 XYZ20~65cm와 task 비율 10/10/20/30/30%를 유지한다. 후반 물리 후보를 현재 정책의 초기 2 step 관측으로 갱신한다. 같은 크기에서 80~95 → 70~80 → 60~70 → 50~60%를 사용하고 전부 없으면 같은 크기의 loco 시작으로 돌아간다. ON_TOP putDown은 source·받침 XYZ 조합을 따른다. 상세 조건·resume·진단은 [hybrid_late_rsi.md](hybrid_late_rsi.md).

```bash
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_vnc.sh <checkpoint.pth> 2 1 4 10
```

기본 output은 `output/approach_scenario_stage1_hybrid_late_rsi`다. 학습 resume는 동일 hybrid 후보/config 상태를 가진 checkpoint만 허용한다. 기존 학생 후보 파일은 SIT/CLIMB에 계속 필요하다.

## 학습

아래 `GPU`를 사용할 장치 번호로 바꾼다. 각 Stage 1 실험은 별도 output에 scratch로 시작한다.

```bash
GPU=5
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_train.sh 2 2048 3  # 27
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_train.sh 2 2048 3  # 28
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_train.sh 2 2048 3  # 30
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_train.sh 2 2048 3  # 31
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3  # 32
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_train.sh 2 2048 3  # 33
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_train.sh 2 2048 3  # 34
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_train.sh 2 2048 3  # 35
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3  # 36
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3  # 37
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3  # Paired AT
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4  # Paired placement
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4  # Unified semantic
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4  # Unified shared edge
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_train.sh 2 2048 4  # Shared random binding
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4  # Unified owner HOLDING
```

한 번에 필요한 실험 **한 줄만** 실행한다. 짧은 확인은 별도 output을 쓴다.

```bash
MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_scenario_stage1_skill_curriculum_reward_preserved_check \
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3
```

정기 checkpoint 저장 간격은 학습 설정에서 500 epoch다. `.pth`는 각 run의 `nn/`, TensorBoard는 `summaries/`, edge 진단은 `diagnostics/`에 저장된다.

## 로컬 평가와 서버 VNC

학습과 **같은 번호의 checkpoint와 스크립트**를 사용한다. `CKPT`를 실제 `.pth` 절대 경로로 설정한다.

```bash
GPU=5
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_test.sh "$CKPT" 2 1 3 10  # 27
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_test.sh "$CKPT" 2 1 3 10  # 28
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_test.sh "$CKPT" 2 1 3 10  # 30
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_test.sh "$CKPT" 2 1 3 10  # 31
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_test.sh "$CKPT" 2 1 3 10  # 32
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_test.sh "$CKPT" 2 1 3 10  # 33
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_test.sh "$CKPT" 2 1 3 10  # 34
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_test.sh "$CKPT" 2 1 3 10  # 35
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10  # 36
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10  # 37
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10  # Paired AT
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10  # Paired placement
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10  # Unified semantic
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10  # Unified shared edge
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_test.sh "$CKPT" 2 1 4 10  # Shared random binding
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10  # Unified owner HOLDING
```

화면 없는 학습 분포 평가는 해당 test 명령에서 `HEADLESS=1 TASK_GRAPH=random_scenario`와 평가 환경 수 `64`를 지정한다. 로컬 test와 서버 VNC는 모두 기본적으로 `output/<실험명>/metrics/`와 `diagnostics/`에 평가 결과를 쓴다. 반복 평가 시 진단 CSV는 같은 파일에 이어 쓰므로 실행별 분석은 `OUTPUT_PATH`를 따로 지정한다. 학습 checkpoint는 별도 run 디렉터리의 `nn/`에 저장된다. 아래는 서버 viewer 실행 명령이다.

```bash
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_vnc.sh "$CKPT"  # 27
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_vnc.sh "$CKPT"  # 28
TOKENHSI_GPU="$GPU" TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_vnc.sh "$CKPT"  # 30
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_vnc.sh "$CKPT"  # 31
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_vnc.sh "$CKPT"  # 32
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_vnc.sh "$CKPT"  # 33
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_vnc.sh "$CKPT"  # 34
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_vnc.sh "$CKPT"  # 35
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"  # 36
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"  # 37
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"  # Paired AT
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"  # Paired placement
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT"  # Unified semantic
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT"  # Unified shared edge
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_vnc.sh "$CKPT"  # Shared random binding
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT"  # Unified owner HOLDING
```

## Unified 독립 5과제 · semantic / shared edge / owner-HOLDING 비교

Semantic과 owner-HOLDING은 **edge 입력 상태 유무만 다르다**. Shared edge는 semantic과 과제·관측·보상이 같고 actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유한다. 토크나이저·GTA/Transformer·head는 분리한다. 기존 paired/focus 실험은 보존하며 각자 새 output에서 scratch로 시작한다. 새 AMP 입력은 프레임당 기존 129차원에 과제 one-hot 3차원을 더한 132차원, 10프레임 총 1320차원이다. 기존 paired checkpoint와는 AMP 및 variant 계약이 달라 이어 학습하지 않는다.

- 학습: 2명·4물체. 각 agent가 독립적으로 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 5/5/20/35/35%`를 샘플한다. 두 ON_TOP도 가능하며 명목 빈도는 12.25%다.
- 논리 할당(0-based): H₀→O₀→G₀, H₁→O₁→G₁. SIT/CLIMB도 각자의 Oᵢ를 사용한다. ON_TOP은 O₀→O₂, O₁→O₃. 물리 asset slot만 reset마다 섞이며 모든 관측·RSI·보상은 동일 logical-to-physical mapping을 따른다.
- H/O/G별 인코딩 뒤 transformer 입력을 모든 타입에 걸쳐 매 forward 무작위 순열로 섞는다. 한 forward의 batch는 같은 순열을 사용하며, GTA와 bias의 두 축도 함께 이동한다. 출력은 canonical 순서로 복원한 다음 human별 actor/value head에 전달한다. 순열 등가성에 따른 수치 동등성은 검증하되 추가 학습 이득을 보장하지 않는다.
- Goal GTA quaternion은 `(0,0,0,1)`로 고정한다. 상자 크기는 X=0.40m, Y=0.30m, Z=0.30~0.50m(0.05m 간격)다.
- loco RSI의 담당 상자는 **owner root XY에서 반경 1~2m**에 배치한다. AT 목표/ON_TOP 받침에 가까운 시작은 0%이며 별개 설정이다. pickUp/carryWith/putDown/sit/climb RSI의 참조 물체 위치는 유지한다. 물리 배치 부적합 시 graph를 유지한 채 reset을 다시 뽑는다.
- state/progress/success 0.2/0.2/0.2, 자기 edge 합계·동료 edge 평균의 0.9/0.1 공유, placement 성공 시 matching HOLDING 포화는 유지한다. TERM 입력은 추가하지 않는다.

| 과제 | RSI |
| --- | --- |
| HOLDING | loco 50%, pickUp 50% |
| SIT | loco 50%, sit 50% |
| CLIMB | loco 50%, climb 50%; 원래 허용 시점 사용, 후반 강제 샘플링 없음 |
| HOLDING+AT | loco 50%, pickUp 10%, carryWith 30%, putDown 10% |
| HOLDING+ON_TOP | loco 50%, pickUp 10%, carryWith 40%; putDown 없음 |

AMP는 단일 판별기에 **각 agent의 할당 과제 family**를 매 프레임 붙인다. HOLDING/AT/ON_TOP은 carry, SIT는 sit, CLIMB은 climb이다. RSI로 걷기를 뽑았어도 family는 할당 과제를 따른다. 전문가의 과제 내부 분포는 원본 unified에 맞춘다: carry=loco/OMOMO/pickUp/putDown 1/3·1/3·1/6·1/6, sit=loco/sit 1/3·2/3, climb=loco/climb/climbNoRSI 각 1/3. `carryWith`는 RSI에는 있지만 이 AMP 전문가 혼합에는 없다.

`skillDiscProb`는 리셋 시 family 비율 75/5/20으로 환산한 명목 전체 분포다. 실제 조건부 expert 생성은 `utils/unified_training.py`의 분포를 사용한다. 학습에서는 전문가·리플레이를 현재 rollout의 family와 **행별로 맞춰** 뽑아 PPO minibatch마다 family 빈도가 일치한다. 리플레이에 해당 family가 아직 없을 때만 같은 family의 현재 rollout을 사용한다. 과제 라벨은 AMP 정규화에서 통과시키며 참조 이력·부분 reset에도 유지한다. task/AMP reward 비중은 기존 0.5/0.5다.

Semantic 실험은 edge당 `[valid,src,dst,relation,owner]` 5필드이고 semantic type/relation만 bias에 인코딩한다. Owner 실험은 6번째 값에 동일 owner·source의 HOLDING φ를 넣고 AT/ON_TOP의 semantic64 뒤에서 상태 잔차를 더한다. 다른 유효 edge의 입력 값은 1이며, 상태 잔차는 placement edge에만 적용한다.

아래 학습 명령에서 필요한 실험을 선택한다. `TOKENHSI_GPU=0`은 로컬 단일 GPU 기준이다.

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4
```

각 checkpoint에는 같은 실험의 test/VNC를 사용한다. 로컬 viewer는 다음과 같다.

```bash
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
```

`TASK_GRAPH=holding|sit|climb|holding_at|holding_ontop`은 두 사람 모두 해당 과제로 평가한다. 기본 평가는 loco 시작이다. placement의 들고 있는 시작을 보려면 `TASK_GRAPH=holding_at` 또는 `holding_ontop`에 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 추가한다. `HEADLESS=1`은 화면 없는 평가다.

```bash
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
```

각 기본 output은 `output/approach_scenario_stage1_unified`, `output/approach_scenario_stage1_unified_shared_edge_encoder`, `output/approach_scenario_stage1_unified_owner_holding`이다. Shared edge는 별도 variant/checkpoint로 scratch 학습하며 기존 unified checkpoint를 resume하지 않는다. 짧은 검증은 별도 `_check` output과 `MAX_ITERATIONS=2`를 사용한다. 2명·4물체 sampler이며 가변 인원 평가 sampler 확장은 포함하지 않는다.

## Stage 1 AT/ON_TOP 집중 실험 · 36번

34번과 동일한 reward·RSI·가까운 시작 설정에서 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 0/0/0/0.5/0.5`로 샘플한다. 한 scene에 AT와 ON_TOP이 각각 하나씩 배정되고, 두 과제 모두 자기 HOLDING edge를 포함한다. 단독 HOLDING 및 SIT/CLIMB 과제는 학습 중 나오지 않는다. `templateRsi`의 AT/ON_TOP 행은 그대로 적용되고, CLIMB 후반 RSI는 CLIMB 과제가 없어 사용되지 않는다. `skillDiscProb`의 SIT/CLIMB 모션 비율은 34번과 동일하다. 새 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PlacementFocus.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"
```

## Stage 1 같은 과제 쌍 · 가까운 시작 없음

**Paired AT**은 3개 물체에서 두 agent에게 각각 다른 물체·목표를 배정한다. 각 agent의 edge는 `H→O HOLDING`, `O→G AT`이며 한 scene에 AT edge가 두 개다. **Paired placement**는 4개 물체에서 scene마다 `AT+AT` 또는 `ON_TOP+ON_TOP`을 50%씩 뽑는다. ON_TOP scene의 두 source와 두 support는 모두 다른 물체다. 두 실험 모두 34번의 RSI·AMP·보상식을 유지하고 `near_start` 확률만 0으로 고정한다. `TASK_GRAPH=holding_at`/`holding_ontop`으로 paired placement 평가 장면을 각각 고를 수 있다. 다른 물체 수와 variant는 서로 checkpoint 호환되지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PairedAtNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"

TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"
```

## Stage 1 Paired placement · 담당자 HOLDING 상태 입력

위 Paired placement와 동일한 2-agent/4-object·AT+AT/ON_TOP+ON_TOP·가까운 시작 없음·RSI·AMP·reward를 사용한다. AT/ON_TOP edge의 여섯 번째 관측 필드는 동일 owner와 source 물체의 직전 HOLDING edge `φ`다. HOLDING 등 다른 유효 edge에는 1을 넣고, 무효 edge는 0으로 마스킹한다. `φ`는 손–상자 중심 거리 기반의 0~1 상태이며 실제 물리적 파지 플래그는 아니다. 기존 paired checkpoint는 새 관측 폭과 맞지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementOwnerHolding.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"
```

## Stage 1 ON_TOP putDown RSI 실험 · 37번

36번과 동일한 과제·보상 분포에서 `HOLDING_ON_TOP` RSI 중 `carryWith` 10%를 `putDown`으로 옮긴다. ON_TOP `putDown`은 원본 모션의 45~70% 구간에서만 시작하고, 받침 상자를 모션의 최종 놓기 XY에 배치한다. 이 구간은 원본 모션의 후기 상자·받침 겹침을 피한다. 받침의 물리 크기는 기존 분포 그대로이며, reset 시 상자 또는 사람과 겹치면 재추첨한다. AT `putDown` RSI 10%와 나머지 설정은 36번과 같다. 별도 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1OntopPutdown.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"
```

## Stage 2 협력 실험 · 29번

29번은 지정한 Stage 1 `.pth`에서 actor/critic/AMP weight와 관찰 통계를 옮겨 새 실행을 시작한다. `RESUME_CHECKPOINT`는 이미 시작한 **같은 Stage 2 변형**을 이어 학습할 때만 쓴다. 기존 29번의 SIT 성공과 보상 정규화는 **28번 정의**다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/Stage1.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2Coordination.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2Coordination.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_vnc.sh "$CKPT"
```

Stage 2 plane은 34번의 SIT plane 성공·자기 edge 합계·동료 edge 평균을 적용한다. 협력 과제 비율은 `place_climb/place_sit/place_stack = 20/20/40%`, 독립 과제는 20%다. 독립 과제 내부의 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP은 34번과 같이 10/10/10/35/35이며 RSI·가까운 시작 일정도 34번과 같다. 협력 과제는 공유 물체에 맞춘 Stage 2 RSI를 유지한다. 별도 config·스크립트·output을 사용한다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/ApproachScenarioStage1SelfSum.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2CoordinationSitPlane.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_vnc.sh "$CKPT"
```

3·4인 평가 graph는 `tokenhsi/data/cfg/multi_agent/graphs/`에 있다. 입력 크기 검증용이며 성공은 별도 학습으로 확인한다.

## 원본 1번

원본 비교가 필요할 때 `ma_carry_train.sh`, `ma_carry_test.sh`, `ma_carry_watch.sh`를 사용한다. 실험별 GPU 지정과 데이터 경로는 위 공통 규칙을 따른다.

지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다. TensorBoard는 `tensorboard --logdir output --host 127.0.0.1 --port 6006`으로 실행한다.

## Shared random binding · 사용자 12옵션 조합

선택 순서 `2 2 2 1 2 1 2 1 2 1 2 2`를 별도 scratch 실험으로 구현했다. 기존 unified/shared/34/35 config와 output을 유지한다.

- Actor/critic semantic EdgeEncoder 공유, PRE/TERM·owner HOLDING 상태 입력 없음. H/O/G token 순열은 매 forward 유지한다.
- reset마다 논리 물체 4개와 goal 2개를 무작위 순열로 배정한다. 두 담당 물체와 두 ON_TOP 받침은 서로 분리된다. 물리 slot도 기존 방식으로 섞는다. graph·RSI·관측·보상·속도 penalty는 같은 binding을 사용한다.
- 실제 상자 크기는 XY 0.40~0.65m, Z 0.25~0.55m이며 34번 크기 풀·평가 크기를 사용한다. loco 담당 상자는 사람 root XY에서 1~2m다.
- 배치 가까운 시작은 34번 일정: per-env 0→300000 step에 80→30%, 이후 30%. AT 목표 XY는 source에서 0.75~1.5m, ON_TOP 받침 XY는 source에서 1.1~1.8m에 놓는다. 평가에는 적용하지 않으며 AT putDown RSI의 참조 목표는 보존한다.
- 과제 비율은 HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP 5/5/20/35/35다. RSI는 shared의 과제별 비율을 유지하고 CLIMB 후반 강제 RSI는 끈다. 가까운 목표 배치는 RSI 자세 초기화와 별개다.
- AMP는 34/35번 비조건부 혼합: loco/sit/climb/climbNoRSI/omomo/pickUp/carryWith/putDown = .26/.20/.08/.06/.10/.10/.10/.10. AMP 과제 one-hot·family 매칭을 끄며 10프레임 입력은 1290차원이다.
- Goal 회전은 identity, 속도 penalty는 graph 담당 물체 기준이다. state/progress/success .2/.2/.2와 자기 합계·동료 평균 .9/.1은 유지한다.
- 기본 정책 관측은 644차원이지만 AMP·binding·variant가 다르므로 기존 unified/shared checkpoint의 resume는 거부한다. 아래 전용 train에서 scratch로 시작한다.

```bash
nvidia-smi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_train.sh 2 2048 4
```

GPU는 실행 시 점유를 확인해 선택한다. 본학습 전에 셸에 남은 MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT를 해제한다. 평가에는 새 실험 checkpoint만 사용한다.

```bash
CKPT='/absolute/path/to/ApproachScenarioStage1UnifiedSharedRandomBinding.pth'
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_test.sh "$CKPT" 2 64 4 3
TOKENHSI_GPU=6 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_vnc.sh "$CKPT" 2 1 4 10
```

### 2026-10-07 push 경첩 저항 강화

현재 분리/공유 `door_open_*_approach_amp` 두 설정에 `env.door.hinge: {damping: 2.0, friction: 0.1}`를 적용했다. 기존 회전 경첩의0.5/0.05보다 저항을 높이고 DOF_MODE_NONE·stiffness0의 수동 문을 유지한다. 공용 URDF·pull·BONES 설정은 변경하지 않는다. 경첩 물리는 checkpoint 계약에 포함되어 기존 저항으로 학습한 checkpoint의 resume/test는 거부한다. 현재 wrapper는 새 물리 설정으로 scratch 학습하며 기본 output도 `_balanced_hinge`로 분리한다. 기존 프로세스에는 소급 적용되지 않는다.

물리 검증: `TOKENHSI_GPU=6 OUTPUT_PATH=output/door_hinge_resistance_check_20261007 bash tokenhsi/scripts/multi_agent/validate_door_physics.sh`. 비학습2환경에서 설정값·무접촉 안정성·충격 후 자유회전·지속 humanoid 접촉·관측/보상 대응을 검사한다. 손 사용은 여전히 성공의 필수 조건이 아니며 angle-only 성공 보상은 유지했다.

2026-10-07 로컬 학습 인계: 사용자 승인한 절충 경첩2.0/0.1을 분리/공유 loco full-AMP 두 설정에 반영했다. 로컬 인계 가이드는 [door_local_training.md](door_local_training.md). 5.0/0.5 결과는 이전 강화 설정의 기록이며 새 검증은 `output/door_balanced_hinge_check_20261007/`를 사용한다.

## PUSH midpoint 2m 구역 교환

`push_toy_midpoint_2m_zone_swap`는 2m 중점 XY 설정에서 `zoneAssignment.swap_probability: 0.5`만 추가한다. 환경마다 reset 시 두 사람의 아래/위 담당 구역을 교환하며 물체 배치는 유지한다. 부분 reset은 해당 환경만 바꾼다. 보상·물성·AMP·RSI 비율과 원래 2m RSI bank를 유지한다. 독립 RNG로 구역을 샘플한다. 기존 설정은 고정 배정이다.

2026-10-09 GPU1·2048환경·seed5306으로 scratch 학습 시작을 확인한 뒤 사용자가 종료했다. 이후 요청에 따라 이 실험의 기본 `max_epochs`를 **1,000,000**으로 변경했다. 아직 재실행하지 않았다. `MAX_ITERATIONS`가 설정돼 있으면 기본값을 덮어쓰므로 본학습 전에 해제한다. 실행 정보와 로그는 `output/push_toy_midpoint_2m_zone_swap/launch.json`에 기록한다. 기존 checkpoint는 구역 배정 계약이 달라 그대로 복원하지 않는다.

```bash
# 새 학습
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_zone_swap_train.sh 2 2048 4
# 학습 없는 연결/물리 검사: CPU2환경
TOKENHSI_GPU=1 bash tokenhsi/scripts/multi_agent/check_open_push_zones.sh
# 학습 후 로컬 추론
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_zone_swap_test.sh <checkpoint.pth> 2 1 4 10
# 학습 후 VNC
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6089 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_zone_swap_vnc.sh <checkpoint.pth> 2 1 4 10
```

전용 output은 `output/push_toy_midpoint_2m_zone_swap`다. 연결 검사 결과는 `output/push_toy_midpoint_2m_zone_swap_connections/audit.json`. TensorBoard 진단 `push/zones/agent{0,1}_{lower,upper}/`에 `sample_count`, `current_success`, `since_reset_forward_m`, `hand_credited_m`를 기록한다. 성공률은 현재 성공 상태의 step 평균이며 episode 성공률과 다르다. CPU 연결/12step 물리 검사와2048환경 학습 시작을 확인했다. 장기 학습 효과는 아직 검증하지 않았다.

## PUSH midpoint 2m 실제 배치 무작위화

`push_toy_midpoint_2m_random_scene`는 구역 교환 설정에서 기본 unified SIT/HOLDING의 **물리 담당 배정·보행 시작의 owner-relative 위치/yaw·참조 정렬 방식**을 적용한다. 두 사람은 자기 박스를 각각 민다. graph의 논리 연결은 canonical이며 물리 슬롯을 매 reset 50% 확률로 교환하고 관측/BPS/접촉력/보상은 같은 담당 물체를 조회한다. `shared_random_binding`의 추가 graph relation/goal slot 무작위화는 기본 unified 방식과 다르므로 도입하지 않는다.

| 항목 | 새 설정 |
| --- | --- |
| 박스 | 1.3×1.3×1.3m 고정·20kg·마찰0.6 |
| 보행 초기화 | 담당 사람 주변 무작위 방향·중심 거리1~2m에 박스 배치·박스 yaw와 사람 heading도 전 방향 샘플 |
| 담당 물체 | 실제 두 PUSH 박스의 owner를 reset마다 균등 교환; 부분 reset은 해당 환경만 변경 |
| 목표 | reset 당시 박스 yaw의 로컬 +Y 방향2m; RSI 및 실행 중에는 고정 |
| 조작 RSI | 샘플된 박스를 해당 방향으로 기존 phase만큼 전진시키고 참조 양손 XY를 새 뒷면에 정렬·자세/속도도 함께 yaw 회전 |
| 보상/실적 | 중점 XY·양손 힘·physics tick 순이동/후진 차감 식 유지; RSI 선행 이동 제외 |
| AMP/초기화 | 로컬 PUSH1802개·보행33/밀기67 전문가·보행/조작50/50·조작 후반 요청70% 유지 |
| 학습 | 2048환경·600step·최대1,000,000epoch |

별도 `pushScene.mode: owner_loco_random`, `box.reset.randomAssignment/randomRot: true`, `ownerLocoDistanceRange: [1,2]`를 사용한다. 보행 사람 기준점은 문과의 간섭을 줄이도록 (-4,-4)/(-4,+4)이고 사람 색과 구역은 고정하지 않는다. 두 박스의 밀기/RSI 뒤쪽 공간과 고정 문 영역을 포함한 경로가 겹치면 위치·yaw를 다시 샘플한다. 따라서 충돌 제외 이후 분포는 완전한 독립 균등 분포가 아니다. 1~2m 거리는 보행 시작에 적용하고 조작 RSI는 손 정렬로 가까이 시작한다. 미사용 문은 원래 위치/물성을 유지한다.

기존 2m PD bank를 유지한다. 모션/물리/보상/손 높이/stand-off 계약이 같고 새 배치는 수평 이동·yaw 회전이다. CPU·GPU 실제 초기화/물리 검증으로 새 정렬을 확인했다. checkpoint에는 새 배치 계약을 기록하므로 이전 고정/교환 설정 checkpoint를 직접 로드하지 않는다. 새 실험은 scratch로 실행한다. 2026-10-09 사용자 요청으로 GPU1·2048환경·seed5306·최대100만epoch 본학습을 시작하고 첫 학습 fps를 확인했다. 실행 정보/로그는 `output/push_toy_midpoint_2m_random_scene/launch.json`에 기록한다.

```bash
# 본학습: MAX_ITERATIONS와 RESUME_CHECKPOINT가 남아 있으면 해제 후 실행
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_train.sh 2 2048 4
# 학습 없는 CPU2환경 연결/RSI/물리 검사
TOKENHSI_GPU=1 bash tokenhsi/scripts/multi_agent/check_push_random_scene.sh
# 학습 없는 GPU2환경 검사
TOKENHSI_GPU=1 RSI_SIM=gpu OUTPUT_PATH=output/push_toy_midpoint_2m_random_scene_gpu_connections bash tokenhsi/scripts/multi_agent/check_push_random_scene.sh
# 해당 새 실험 checkpoint의 로컬 추론
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_test.sh <checkpoint.pth> 2 1 4 10
# 서버/로컬 VNC
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6089 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_vnc.sh <checkpoint.pth> 2 1 4 10
```

학습/평가 기본 output은 `output/push_toy_midpoint_2m_random_scene`다. CPU/GPU 검사 결과는 `output/push_toy_midpoint_2m_random_scene{,_gpu}_connections/audit.json`. 전진 진단은 고정 world Y 대신 episode 초기 밀기 방향에 투영한다. CPU 회귀78개·2048배치 샘플의 사분면/거리/경로·회전5조건의 접촉/성공/후진 차감 검사 통과. 실제 CPU·GPU 각2환경에서1,000개 agent reset·두 강제 배정 RSI·부분 reset·AMP·24step 물리 유한값/고정 목표 검사 통과. CPU 초기화 교환52.2%·조작49.9%·후반 요청69.5%. 새2048환경 본학습 시작은 확인했고 정책 성공 개선·checkpoint 복원·GUI/VNC는 아직 검증하지 않았다. 기존 OPEN 학습은 유지했다.

## 문 없는 PUSH 전용 옵션

2026-10-14부터 `push_only: true`의 박스 입력은 물리 정보30D(선속도3·각속도3·로컬 꼭짓점24)만 사용한다. 추가 조작/접촉 상태10D를 제거해 전체 관측은570D이며6토큰은 유지한다. soft tilt·CONTACT+AT 등 모든 문 없는 설정에 적용된다. 기존40D/590D checkpoint는 계약의 `object_obs_size`가 달라 직접 resume/test할 수 없다. 이미 실행 중인 학습은 기존40D를 유지하며, 변경된 코드로 새 학습을 시작할 때 적용된다. 이전590D 실행의 검증 기록은 당시 형식이다.

`PUSH_ONLY=1`을 `push_toy_midpoint_2m_random_scene_{train,test,vnc}.sh` 앞에 지정하면 전용 `push_toy_only_random_scene` env/train YAML·wrapper·output으로 전환한다. YAML에서는 `env.relationGraph.push_only: true`, `numObjects: 2`를 사용한다. 옵션을 생략하면 기존 문 포함 실험이다. 물체 수 인자는 이 모드에서2로 지정한다.

실제 사람2명과 박스2개만 생성하며 문 asset/actor/경첩 DOF·문 토크·문/손잡이 접촉·OPEN reward 계산을 제외한다. headless 환경의 rigid body는32개, DOF는64개다. clean-scene 관측은6토큰·570차원이며 graph는 `H0→Box0→Goal0`, `H1→Box1→Goal1`이다. 실제 박스 담당 배정은50% 교환하고 관측/BPS/힘/RSI/credit이 같은 배정을 따른다. 박스1.3m·20kg/마찰0.6·2m 목표·기존 PUSH 보상·AMP 혼합/참조2개·초기화 비율·무작위 위치/yaw·최대100만epoch는 유지한다. 배치 샘플링의 문 간섭 제외 검사는 문 없는 모드에서 생략한다.

공통 motion manifest와2m RSI bank 계약을 재사용하므로 설정/검증 metadata의 OPEN 항목은 남아 있으나 문 actor·OPEN 정책 과제·OPEN reward는 실행하지 않는다. 관측/graph 계약이 달라 이전 문 포함 checkpoint를 로드하지 않는다. 새 학습은 scratch이며 해당 전용 checkpoint로 평가한다.

```bash
# 문 없는 학습: 사람2·환경2048·박스2
PUSH_ONLY=1 TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_train.sh 2 2048 2
# 전용 wrapper를 직접 사용해도 동일
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/push_toy_only_random_scene_train.sh 2 2048 2
# 로컬 추론
PUSH_ONLY=1 TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_test.sh <checkpoint.pth> 2 1 2 10
# VNC
PUSH_ONLY=1 TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6089 bash tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_vnc.sh <checkpoint.pth> 2 1 2 10
# 학습 없는 CPU/GPU2환경 검사
TOKENHSI_GPU=1 bash tokenhsi/scripts/multi_agent/check_push_only_scene.sh
TOKENHSI_GPU=1 RSI_SIM=gpu OUTPUT_PATH=output/push_toy_only_random_scene_gpu_connections bash tokenhsi/scripts/multi_agent/check_push_only_scene.sh
```

2026-10-09 사용자 요청으로 GPU1·2048환경·2박스·seed5306·최대100만epoch scratch 본학습을 시작하고 실제 GPU 배치/정상 학습 로그를 확인했다. 전용 output은 `output/push_toy_only_random_scene`이며 실행 명령/로그는 `launch.json`에 기록한다. 기존78개 회귀·새3개 graph/보상 동일성/OPEN 미호출 검사 통과. 실제 CPU/GPU 각2환경에서1,000개 agent reset·두 배정 RSI·부분 reset·24step 유한 물리·bank 보존·Actor/Critic/AMP forward 통과(`output/push_toy_only_random_scene{,_gpu}_connections/audit.json`). 세 wrapper의 옵션 dispatch 및 Python/셸 문법 검사 통과. 사용자가 속도 비교를 생략하도록 요청했으므로 benchmark를 실행하지 않았다. 장기 성공률·checkpoint 복원·GUI/VNC는 아직 미검증이다.


## 문 없는 PUSH + AT 독립 보상

`push_toy_only_random_scene_at`은 기존 문 없는 무작위 배치 실험을 유지하면서, 보상만 기존 HOLDING+AT 방식으로 분해한 별도 실험이다. 네트워크·6토큰/570차원 관측·박스1.3m/20kg/마찰0.6·2m 목표·담당50% 교환·AMP/RSI 비율·2048환경/최대100만epoch는 유지한다. `PUSH_ONLY=1`은 종전 실험을 선택하므로 새 실험에는 아래 전용 wrapper를 사용한다.

- `H→Box PUSH`: state는 `exp(-10*d_hand_midpoint_xy²)`에 grounded를 곱한다. progress는 root→박스 XY거리로 `1/(1+max(d-0.5,0))`이며 grounded를 곱한다. 자체 성공은 state≥0.9다. 양손 중점과 yaw 기준 뒷면 중앙의 XY 근접 proxy이며 실제 접촉력이나 유효 밀기 거리 조건이 아니다. 기존 높이 자유 중점 기하를 유지한다.
- `Box→Goal AT`: state는 `exp(-10*||box_center-goal||₃D²)`, progress는 XY거리의 `1/(1+max(d-0.5,0))`다. 자체 성공은 state≥0.9 및 |Z오차|≤0.001m다. 높이가 맞으면 XY 약10.3cm 이내다. 기존 HOLDING+AT처럼 손 접촉/이동 실적/속도 gate 없이 목표 보상을 계속 지급한다. 기존 PUSH의15cm 허용값·속도 제한은 이 실험의 AT 성공 조건에 사용하지 않는다. 1mm 높이 기준은 엄격하므로 실제 정책 성공률은 별도 확인해야 한다.
- 각 edge의 state/progress/success 가중치는0.2/0.2/0.2다. 자체 성공 시 해당 edge를 포화하고, AT 현재 성공은 같은 박스의 PUSH도 포화한다. 자기2edge 합계×0.9 + 동료2edge 평균×0.1로 ksh와 같은 집계를 사용한다(최대1.14, penalty/AMP 혼합 전). 종전 단일 과제 최대0.8과 총 보상 크기가 다르다. power/collision penalty 및 task/AMP 혼합0.5/0.5는 유지한다.
- 최종 agent 성공/평가는 AT 현재 성공이다. 손을 떼어도 AT 조건을 유지하면 완료이고 목표를 벗어나면 해제된다. PUSH 자체 성공과 AT 자체 성공은 별도로 기록한다. 이는 손으로 목표까지 밀었다는 종전 순이동 자격을 보장하지 않는다. net-credit 계산은 관측/진단에 유지하지만 AT 보상을 차단하지 않는다.

checkpoint는 별도 `push_at_reward` 계약으로 분리하며 기존 PUSH checkpoint를 resume/test하지 않는다. `push_toy_only_random_scene_at_rsi_bank.json`은 동일 물리/모션/RSI 후보를 기존2m bank에서 복사했다. 새 보상 metadata 및 원본 hash/재사용 이유를 기록했으며 물리 후보를 재screen한 것은 아니다. 원본 bank는 수정하지 않는다.

```bash
# 점유 확인 후 실행; 이미 실행 중이면 중복 시작하지 않는다.
nvidia-smi
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/push_toy_only_random_scene_at_train.sh 2 2048 2
# 확인 학습: 환경 수 유지, 반복/output만 분리
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo MAX_ITERATIONS=2 OUTPUT_PATH=output/push_toy_only_random_scene_at_check bash tokenhsi/scripts/multi_agent/push_toy_only_random_scene_at_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_only_random_scene_at_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_only_random_scene_at_test.sh <checkpoint.pth> 2 16 2 3
# VNC: 로컬 환경, 서버에서는 TOKENHSI_CONDA_ENV=tokenhsi와 해당 GPU 지정
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6089 bash tokenhsi/scripts/multi_agent/push_toy_only_random_scene_at_vnc.sh <checkpoint.pth> 2 1 2 10
# 비학습 CPU2환경 연결 검사
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=cpu bash tokenhsi/scripts/multi_agent/check_push_at_scene.sh
```

VNC 주소는 `http://localhost:6089/vnc.html?autoconnect=1&resize=remote`다. 전용 output은 `output/push_toy_only_random_scene_at`. TensorBoard `relation/90_debug/push_at/{push,at}/{state,progress,success_bonus,own_success}`는 각각의 edge별 agent-step 평균이다. `push/current_success`와 평가 성공은 최종 AT 성공이며 `reward_terms/{state,progress,success}`는 두 edge 합계다. 새 분기의 기존 `push/manipulation_progress`는 합산 progress를0.2로 나눈 값으로 최대2이며 종전0~1지표와 직접 비교하지 않는다.

CPU81개 회귀와 실제 CPU2환경 reset/24step 물리·Actor/Critic/AMP 연결을 검증했다. 결과는 `output/push_toy_only_random_scene_at_connections/audit.json`에 있다. 2026-10-09 사용자 요청으로 기존 GPU1 PUSH를 종료하고 새 실험을 `tmux koo_train`의 `push_at` 창에서 GPU1·2048환경·2박스·seed5306·최대100만epoch scratch로 시작했다. 실제 GPU 배치·새 보상 설정·checkpoint 미복원·첫 정상 학습 fps를 확인했으며 GPU0 OPEN은 유지했다. 실행과 로그는 `output/push_toy_only_random_scene_at/launch.json`을 참조한다. 창 확인은 `tmux attach -t koo_train` 후 `tmux select-window -t koo_train:push_at`이다. checkpoint 복원·GUI/VNC·장기 정책 성공률은 아직 검증하지 않았다.

## PUSH·AT 모두10도/높이 조건: GPU0 전이 학습

`push_toy_random_at_10deg_at_grounded`는 기존 goal-rear RSI 설정에 `relationReward.at.require_grounded: true`를 추가한 별도 실험이다. PUSH와 AT 모두 박스 기울기≤10도·reset 전 중심 높이±8cm일 때 state/progress/성공 보상을 지급한다. 조건을 벗어나면 두 edge의 해당 보상은0이며 현재 AT 성공/PUSH 포화를 해제한다. 조건 복구 시 다시 지급한다. AMP·전력/충돌 패널티·episode 종료·랜덤 목표·손 중점 State·실제 접촉 gate 없음·도착 후 손 떼기는 유지한다.

보상 계약이 달라 기존 GPU1 checkpoint를 직접 resume/test하지 않는다. `transfer_push_at_grounded_checkpoint.py`는 원본 config/checkpoint/bank 계약을 검사하고 AT grounded 한 조건과 bank 경로 변경만 허용한다. actor/critic/discriminator와 입력·value·AMP 정규화 통계 및 원본 epoch/frame을 보존한다. optimizer 누적 상태·best reward는 초기화하고 rollout/AMP replay는 복원하지 않는다. 새 계약과 전이 출처를 기록한 별도 checkpoint를 생성하며 일반 resume 검사는 그대로 유지한다. 전용 RSI bank는 기존 후보/PD screen 데이터를 보존하고 새 보상 metadata 및 재사용 출처만 추가했다(재screen 없음).

사용한 원본은 GPU1의 `PushToyRandomAT10Deg_00004500.pth`이고, 변환본은 `output/push_toy_random_at_10deg_at_grounded/warmstart/from_gpu1_4500.pth`다. 본학습은 GPU0·2048환경·2사람/2박스·seed5306·최대100만epoch이며4501epoch부터 이어진다. 짧은2회 확인은 별도 `output/push_toy_random_at_10deg_at_grounded_check`에서 `MAX_ITERATIONS=4502`로 수행한다(최대epoch는 누적 번호). 학습용 실행에서는 viewer 저메모리/평가 전용 환경변수를 제거한다.

```bash
# 최초 전이: 새 output만 허용, 기존 변환본을 덮어쓰지 않는다
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/transfer_push_at_grounded_checkpoint.py --source output/push_toy_random_at_10deg_goal_rear_rsi/PushToyRandomAT10Deg_12-21-05-04/nn/PushToyRandomAT10Deg_00004500.pth --output output/push_toy_random_at_10deg_at_grounded/warmstart/from_gpu1_4500.pth
# 본학습: 실행 중이면 중복 실행하지 않는다
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=5306 RESUME_CHECKPOINT=output/push_toy_random_at_10deg_at_grounded/warmstart/from_gpu1_4500.pth bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_grounded_train.sh 2 2048 2
# 로컬 추론 / headless 평가 / 서버 VNC (변환본 또는 새 실험 checkpoint)
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_grounded_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_grounded_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6092 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_grounded_vnc.sh <checkpoint.pth> 2 1 2 10
```

## PUSH 10도 제한 / ksh식 랜덤 AT 목표

4500checkpoint의 GPU0 평가 명령은 `bash output/push_gpu0_eval_20261013/launch.sh`(보행 시작·16환경×10회), `bash output/push_gpu0_eval_20261013/launch_train_mix.sh`(학습 RSI 혼합·16환경×3회)다. `TOKENHSI_PUSH_EVAL_AUDIT=1`은 첫 에피소드의 살아 있는 scene만 집계해 all/미완료/moving/완료 접촉 step·에피소드 내 한 번이라도 양손 접촉·접촉 없이 도착·목표 고정·완료 시 보상 포화를 `metrics/edge_context_*.json`에 추가한다. `TOKENHSI_EVAL_TRAIN_RSI=1`은 평가에서만 학습의 조작 RSI 혼합을 허용하며 초기 조작 RSI 비율도 기록한다. 기본 평가는 기존처럼 보행 시작이다. 두 평가는 다른 seed로 실행하므로 동일 episode의 짝지은 비교는 아니다.

2026-10-13 평가 완료: 보행 시작 목표 도착0/320agent episode, 학습 RSI 혼합9/96. 혼합 조건의 이동 중 접촉proxy50.5%·완료 후 비접촉99.4%·완료 보상 포화 위반0, 두 조건 목표 drift0m. [평가 보고서](../output/push_gpu0_eval_20261013/REPORT.md). 접촉은 목표 방향 거리/힘 proxy이며 실제 충돌 쌍 측정이 아니다. RSI 혼합은 이미 박스를 일부 전진시킨 시작을 포함한다.

2026-10-13 사용자 요청으로4500epoch checkpoint의 GPU0 VNC를 실행했다. `bash output/push_gpu0_view_20261013/launch.sh`는 GPU0·평가1환경·2사람/2박스·저메모리·1280×720·6090포트·보행 시작을 사용한다. 접속은 `http://localhost:6090/vnc.html?autoconnect=1&resize=remote`, tmux는 `koo_push_view_20261013`, 로그는 `output/push_gpu0_view_20261013/viewer.log`다. 기존 세션이 살아 있으면 중복 실행하지 않는다.

2026-10-16 목표 반대편(`goal_opposite_surface_xy`) 접촉 진단은 선택된 같은 면의 XY 선분까지 각 손 거리 `d_i`로 `s_i=exp(-10*d_i²)`, CONTACT state `H=2*s_R*s_L/(s_R+s_L)`을 계산한다. 기존 양손18cm 조건을 **H>0.9**로 교체하고, 현재 박스→목표 방향의 각 손 반력≥0.5N·박스 전방 힘≥0.5N 조건은 유지한다. 손 Z 제한은 없으며 H는 완료 후 포화된 보상값이 아닌 현재 기하 점수다. 두 손 거리가 같으면 약10.26cm 미만에 해당한다. `both_hands_near`도 같은 H 판정을 사용한다. 힘·근접은 같은 physics tick에서 평가하며 control step 내 접촉 집계 방식은 유지한다. 이 변경은 목표 반대편 reference를 사용하는 공통 접촉/진단 경로에 적용되며, 기존 로컬 뒤면 reference는18cm를 유지한다. GPU1 soft tilt의 손 중점 state·AT/보상·면 선택·RSI·checkpoint 계약은 변경하지 않았다. 실행 중인 학습에는 자동 반영되지 않고 새 프로세스부터 적용되며 이번 작업에서 학습은 중단/재시작하지 않았다. `physics/paired_contact`와 `contact/{all,moving,completed}/hand_contact`는 agent-step 접촉 proxy다. moving은 미완료·박스 속도0.02m/s 초과, completed는 현재 AT 성공 상태다. 이전18cm 로그와 구분한다.

`push_toy_random_at_10deg`는 기존 GPU1 PUSH+AT에서 PUSH의 `max_tilt`만 정확히10도(0.17453292519943295rad)로 줄이고 AT 목표 샘플링을 분리한 새 실험이다. 10도 이하·초기 중심 높이±8cm 조건이 PUSH 손 근접/사람 접근 보상에 적용된다. AT의3D φ≥0.9(거리 약10.26cm)·목표 높이±1mm 성공 조건은 유지한다. **AT에는 기울기 제한을 추가하지 않았다. AT 성공 시 선행 PUSH 보상이 포화되는 기존 동작도 유지한다.** 접촉/손으로 민 거리/속도를 AT 성공에 요구하지 않는다.

참고 코드는 `ports/ksh_stage1/tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:_reset_task`다. ksh의 loco/pickUp/carryWith 시작 HOLDING+AT 목표는 사람별 기준점 주변 사각 영역의 랜덤 XY이며 사람/박스에서 최소1m를 띄운다. putDown RSI의 목표는 참조 동작 마지막 박스 위치이므로 모든 시작에서 무조건 랜덤인 것은 아니다. 해당 코드만 참고했으며 과거 높은 성공률의 원인을 검증한 것은 아니다.

새 PUSH 목표는 reset 시 담당 구역 기준점(-4,±4) 주변 **XY 각±2m**에서 뽑는다. 기준 사람 시작점과 초기 박스 중심에서 각각 최소1m, 초기 박스 yaw와 독립인 목표이며 목표 Z는 박스 초기 중심0.65m다. 두 박스–목표 선분과 그 뒤쪽1.5m 사람/RSI 경로의 간격이2.2m 이상이고 다른 사람 기준점에서 박스 외접반경+0.5m 이상인 배치만 수락한다. 불가능한 표본은 재샘플하며128회 후 실패하면 오류로 중단한다. 이는 초기 배치와 의도한 이동 경로의 여유 검사이며 실제 정책이 경로 밖으로 이동할 때의 충돌 방지는 보장하지 않는다.

보행 시작 박스는 기존처럼 사람 주변1~2m/전방향 yaw·담당50% 교환이며 AMP/물성/모션/보상 가중치/학습2048환경·최대100만epoch는 유지한다. 목표는 RSI 및 실행 중 고정하고 RSI 선행 이동은 샘플된 박스→목표 거리의 후보 phase 비율로 적용한다. RSI 초기 양손 중점도 보상과 같은 목표 반대편 표면에 맞춘다. 목표 방향에 수직인 양손 pair를 만들고 사람 자세·방향·선속도·각속도를 함께 yaw 회전하며 발 높이는 유지한다. 손 중점은 표면에서 stand_off(6cm)만큼 바깥에 놓는다. 목표는 RSI 전 이 reset에서 샘플한 담당 박스 목표를 사용한다. 초기 배치의 뒤쪽1.5m RSI 경로도 목표 방향을 따른다. 기존 고정2m checkpoint와 초기화/보상 계약이 달라 직접 resume/test할 수 없으며 새 설정은 scratch로 실행한다.

2026-10-12부터 PUSH 손 State는 `push_at.hand_reference: goal_opposite_surface_xy`를 사용한다. 매 step 현재 박스 중심→고정 목표의 XY 벡터를 박스 yaw 좌표로 변환하고, 반대 방향 ray와 박스 XY 직사각형 경계의 교점을 손 중점 기준으로 잡는다. 손 높이·박스 pitch/roll은 기준점 계산에서 제외하며 목표와 XY가 일치하면 박스 로컬 −Y 중앙으로 fallback한다. 목표가 옆/뒤에 있거나 박스가 움직이고 회전해도 현재 목표의 반대편 표면을 따른다. State는 `exp(-10*d_midpoint_xy²) × grounded`; 10도/±8cm·성공≥0.9·접근 Progress·AT 보상/포화는 유지한다. 각 손 접촉 강제는 추가하지 않는다. 기존 고정 목표 PUSH+AT 실험은 로컬 −Y 기준을 유지한다.

보상 및 RSI 정렬 계약이 바뀌어 수정 전 random-AT/goal-rear checkpoint는 직접 resume/test할 수 없다. train/test/VNC 기본 출력은 `output/push_toy_random_at_10deg_goal_rear_rsi`로 분리했다. 기존 학습 output은 보존한다. 전용 bank는 후보를 유지하고 `push_rsi_alignment: goal_opposite_surface_xy_v1`과 이전 hash를 기록했으며 후보 전체를 재screen하지 않았다.

전용 `push_toy_random_at_10deg_rsi_bank.json`은 기존 PUSH+AT의 동일 물리/모션 PD 후보를 복사하고 새 보상 계약과 재사용 출처를 기록했다. 후보를 새로 PD screen한 것은 아니며 새 랜덤 목표/RSI 배치의 reset·짧은 물리 연결 검사를 별도로 수행한다. 기존 설정/bank/checkpoint/output은 유지한다.

```bash
# 본학습: GPU 점유 확인 후 실행. 현재 작업에서는 본학습을 시작하지 않는다.
nvidia-smi
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=5306 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_train.sh 2 2048 2
# 로컬 시각화 / 화면 없는 평가: 새 설정으로 학습한 checkpoint 사용
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC: 서버 conda=tokenhsi, 로컬 conda=tokenhsi_koo
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6090 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_vnc.sh <checkpoint.pth> 2 1 2 10
# 비학습2환경 CPU/GPU 연결 검사
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=cpu bash tokenhsi/scripts/multi_agent/check_push_random_at_scene.sh
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=gpu OUTPUT_PATH=output/push_toy_random_at_10deg_gpu_connections bash tokenhsi/scripts/multi_agent/check_push_random_at_scene.sh
```

VNC는 `http://localhost:6090/vnc.html?autoconnect=1&resize=remote`. 학습 출력은 `output/push_toy_random_at_10deg_goal_rear_rsi`다. 짧은 학습 확인 시2048환경을 유지하고 `MAX_ITERATIONS=2 OUTPUT_PATH=output/push_toy_random_at_10deg_goal_rear_rsi_check`를 별도로 지정한다.

2026-10-12 목표 반대편 손 기준 수정 검증: 관련 CPU84개와 CPU2환경 reset/부분 reset/24step 물리·관측/보상·Actor/Critic/AMP 연결 통과. 결과는 `output/push_toy_random_at_10deg_goal_rear_connections/audit.json`. 새 GPU 실행·checkpoint 복원·GUI·장기 성공률은 미검증이며 본학습은 시작하지 않았다.

2026-10-12 RSI 목표 반대편 정렬 수정 검증: CPU82개 회귀·GPU1 2환경 1,000agent reset/양 배정 RSI/부분 reset/24step 유한 물리·관측·보상/Actor·Critic·AMP 연결·bank 보존 통과. 결과는 `output/push_toy_random_at_10deg_goal_rear_rsi_gpu_connections/audit.json`. 본학습은 시작하지 않았으며 새 checkpoint 복원·GUI·장기 정책 성공은 미검증이다.

## PUSH 기울기 제한을 AT state로 이동

`push_toy_random_at_10deg_at_state`는 goal-rear RSI 설정에서 `relationReward.push_at.require_tilt: false`, `relationReward.at.state_require_upright: true`를 사용하는 별도 실험이다. PUSH state/progress/성공에서 기울기 제한을 제거하되 reset 중심 높이±8cm 조건은 유지한다. AT state만 박스 기울기10도 초과 시0이며 목표 도착에 따른 포화 이후에도 적용한다. AT progress·성공·도착 시 PUSH 포화는 기존 판정을 유지한다. 각도 기준은 `relationReward.push.max_tilt`를 공유한다. 기존 두 실험의 기본 동작은 유지한다.

전용 RSI bank는 goal-rear RSI의 후보/PD screen을 보존하고 변경된 reward 계약과 재사용 출처를 기록했다(재screen 없음). 출력은 `output/push_toy_random_at_10deg_at_state`로 분리한다. 기존 checkpoint는 보상 계약이 달라 직접 resume/test할 수 없으며 아래 학습은 scratch다. 기존 학습 프로세스에는 자동 반영되지 않는다. PID174445는 작업 환경에서 조회되지 않아 학습 변경/재시작은 수행하지 않았다.

```bash
# scratch 학습; GPU 점유 확인 후 선택
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_state_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가 (새 실험 checkpoint 필요)
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_state_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_state_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6094 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_state_vnc.sh <checkpoint.pth> 2 1 2 10
```

CPU 보상/계약 회귀102개 통과. 실제 새 학습·시뮬레이션·checkpoint 복원·viewer는 미검증이다.

## PUSH 똑바로 밀기: AT 기울기 감쇠

### 이전40D·AT0.5m 저장본 GPU0 resume

`push_soft_legacy40_at05`는 이전 PID9090 학습과 같은 박스40D/총590D·AT progress_delta0.5m를 복원하는 전용 env 및 train/test/VNC다. train network YAML은 기존 soft tilt를 재사용한다. `pushObjectObsSize: 40`으로 이전 추가 상태10칸을 복원하며 기본30D 설정은 유지한다. 전용 bank는9000epoch checkpoint의 원본 SHA256과 동일하고, checkpoint 검사에서 SHA256 및 나머지 계약이 모두 같을 때 bank 파일 경로 이동만 허용한다. 출력은 `output/push_soft_legacy40_at05`로 분리한다. CPU61개·원본bank hash/보상 계약·셸 문법 확인; 실제 resume는 미실행이다.

```bash
# GPU0: 기존9000 저장본 resume, GPU1 새 학습과 별도 실행
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_GUI_GPU_MODE=mesa RESUME_CHECKPOINT=output/push_toy_random_at_10deg_at_tilt_soft/PushToyRandomAT10DegATTiltSoft_14-21-14-32/nn/PushToyRandomAT10DegATTiltSoft_00009000.pth MAX_ITERATIONS=1000000 SEED=356 OUTPUT_PATH=output/push_soft_legacy40_at05 bash tokenhsi/scripts/multi_agent/push_soft_legacy40_at05_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_soft_legacy40_at05_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_soft_legacy40_at05_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC: 기존 학습 보호 규칙/점유 확인 후
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6096 bash tokenhsi/scripts/multi_agent/push_soft_legacy40_at05_vnc.sh <checkpoint.pth> 2 1 2 10
```

### 현재30D·AT0.1m 설정

현재30D·AT0.1m 설정은2026-10-14 GPU1 MPS2환경에서 RSI1,000agent reset·양 담당 배정/부분 reset·24step 물리·570D Actor/Critic·AMP 전문가128개/Discriminator 연결을 확인했다. 조작 시작51.1%·후반 요청67.7%·reset 기울기0도·목표 고정·bank 보존 통과. 결과는 `output/push_soft_box30_at01_connections/audit.json`, 로그는 같은 폴더 `check.log`다. PPO/AMP optimizer 업데이트 및 본학습은 실행하지 않았다. 재검사 시 학습 없는 GPU를 확인하고 `TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=gpu OUTPUT_PATH=output/push_soft_box30_at01_connections bash tokenhsi/scripts/multi_agent/check_push_at_tilt_soft_scene.sh`를 사용한다.

2026-10-14 현재 soft tilt 설정의 AT `progress_delta`는0.5→0.1m다. 기울기 감쇠 전 progress는 `1/(1+max(거리XY-0.1,0))`이며 기울기 감쇠/성공 판정은 유지한다. 입력은 사람223D·박스30D·목표1D, pose/graph 포함570D다. RSI 후보/PD screen은 보존하고 bank reward 계약만 맞췄다. 기존40D/0.5m 저장본 직접 resume는 불가하며 새 학습 출력은 `output/push_soft_box30_at01`로 분리한다. 아래 명령은 GPU1 점유 확인 후 scratch로 실행한다.

`push_toy_random_at_10deg_at_tilt_soft`는 PUSH의 각도 gate를 제거하고 AT에 아래 조건을 적용한 독립 scratch 실험이다. 이전 `at_state`와 `at_grounded` 실험의 동작은 유지한다.

- 기울기 계수 `w = clamp((15도 - θ) / 10도, 0, 1)`: 5도 이하1, 10도0.5, 15도 이상0. AT state와 progress에 곱하며 목표 도착에 따른 포화 이후에도 적용한다.
- AT 성공은 기존 거리φ≥0.9·높이차≤1mm에 기울기≤10도를 추가한다. 거리φ와 감쇠된 state 보상을 구분하며, success 보너스에는 `w`를 곱하지 않는다. 따라서 목표 도착/10도에서는 AT state0.1+progress0.1+success0.2다.
- AT 성공 후10도 초과 시 현재 성공·AT bonus·AT에 의한 PUSH 포화를 해제한다. PUSH 자체의 손 근접 성공/접근 보상은 별개로 유지된다. 10도 이하로 복구하고 기존 도착 조건을 만족하면 재완료된다.
- PUSH의 초기 중심 높이±8cm 조건은 유지한다. 물리·참조 동작·목표 배치·RSI 정렬·AMP는 기존 goal-rear RSI와 같다.

RSI bank는 기존 후보/PD screen을 보존하고 보상 계약·원본 hash·재사용 출처만 갱신했다. 물리 설정 불변을 비교했으므로 전체 후보의 PD 재screen은 하지 않았다. GPU1 MPS 비학습2환경 검사에서1,000agent reset·초기 박스 기울기0도·양 담당 배정 RSI·부분 reset·24step 유한 물리/관측/보상·목표 고정·6토큰 Actor/Critic/AMP 연결을 확인했다. 결과는 `output/push_toy_random_at_10deg_at_tilt_soft_connections/audit.json`이다. CPU 회귀123개 통과. 본학습·새 checkpoint 복원·GUI·장기 학습 개선은 미검증이다.

출력/보상 계약을 별도로 분리했으므로 다른 실험의 checkpoint 직접 resume/test는 불가하다. 아래 scratch 명령은 checkpoint를 비워 시작하며 GPU1의 검증된 사용자 MPS에 자동 연결된다. 학습2048환경을 유지한다. 본학습 전 GPU 점유를 확인하고 viewer 전용 변수는 제거한다.

2026-10-14 같은 soft tilt 실험을5500epoch에서 재개한 뒤 사용자가 중단했고, 최신6000epoch 저장본에서 GPU1·2048환경·seed356·최대100만epoch로 다시 재개했다. 실행은 `bash output/push_soft_resume6000_20261014/launch.sh`, tmux는 `koo_push_soft_resume6000_20261014`, 로그는 `output/push_soft_resume6000_20261014/train.log`이며 이미 실행 중이면 중복 시작하지 않는다. launcher가 viewer/native 환경변수를 제거한다. checkpoint 복원·6001/6002epoch 진행·약12400FPS·실제 GPU1 배치·GPU0 기존 학습PID 유지를 확인했다. 6000 이후 미저장 진행·rollout/AMP replay는 복원되지 않는다. 새 run/세부 기록은 `output/push_soft_resume6000_20261014/launch.json`, 이전5500 재개 기록은 `output/push_soft_resume_20261014/launch.json`이다.

```bash
# scratch 본학습: 최대100만epoch, 별도 출력
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_GUI_GPU_MODE=mesa RESUME_CHECKPOINT= MAX_ITERATIONS=1000000 SEED=356 OUTPUT_PATH=output/push_soft_box30_at01 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_tilt_soft_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가: 이 실험에서 생성한 checkpoint
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_tilt_soft_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_tilt_soft_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6095 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_tilt_soft_vnc.sh <checkpoint.pth> 2 1 2 10
# RSI/reset/물리/네트워크 재검사 (비학습2환경)
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=gpu bash tokenhsi/scripts/multi_agent/check_push_at_tilt_soft_scene.sh
```

## AT 성공에만 기울기 제한: GPU0 scratch

`push_toy_random_at_10deg_at_success_tilt`는 `at.success_max_tilt: 0.17453292519943295`(10도), `push_at.require_tilt: false`를 사용한다. AT state/progress는 기존 거리식 그대로이며 기울기 감쇠를 적용하지 않는다. AT 성공에만 기존 거리φ≥0.9·높이차≤1mm와 기울기≤10도를 모두 요구한다. 완료 후10도 초과 시 현재 성공/AT bonus/AT에 의한 PUSH 포화를 해제한다. 이때 AT state와 progress는 실제 거리식으로 돌아가므로 목표 중심에 있으면 각각0.2가 유지될 수 있다. PUSH 자체 손 근접 성공/접근 보상·높이±8cm는 유지한다.

기존 soft/state/grounded 실험과 config/train/test/VNC·bank·output/checkpoint 계약을 분리한다. 물리/RSI/배치/참조는 그대로이며 기존 goal-rear RSI 후보/PD screen을 보존하고 reward 계약/출처만 갱신했다. GPU0 MPS2환경에서1,000agent reset·양 담당 배정 RSI·부분 reset·24step 물리/보상/관측·Actor/Critic/AMP 연결과 bank 보존 확인(`output/push_toy_random_at_10deg_at_success_tilt_connections/audit.json`). CPU 회귀145개 통과. GPU0 MPS는 `source mps/shell.sh` 후 `mps_start 0`, 상태는 `mps_status 0`으로 확인한다. GPU1의 기존 학습은 유지했다.

아래 본학습은 checkpoint 없이 scratch이며2048환경·최대100만epoch다. 사용자에게 명령만 제공했으며 본학습은 시작하지 않았다. 장기 성능·새 checkpoint 복원·GUI는 미검증이다. 기존 실험 checkpoint의 직접 resume/test는 불가하다.

```bash
# GPU0 MPS 자동 연결, scratch 본학습
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo RESUME_CHECKPOINT= MAX_ITERATIONS= OUTPUT_PATH=output/push_toy_random_at_10deg_at_success_tilt bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_success_tilt_train.sh 2 2048 2
# 이 실험의 checkpoint로 로컬 추론 / 화면 없는 평가
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_success_tilt_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_success_tilt_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6096 bash tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_success_tilt_vnc.sh <checkpoint.pth> 2 1 2 10
# 비학습 RSI/물리/네트워크 재검사
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=gpu bash tokenhsi/scripts/multi_agent/check_push_at_success_tilt_scene.sh
```

## CONTACT 양손 표면 근접·접근 정렬 + AT

2026-10-14 GPU1 제거 후 남은 GPU0에서 최신1500epoch checkpoint·2048환경·seed1485·최대100만epoch로 재개했다. 실행은 `bash output/contact_resume1500_single_gpu_20261014/launch.sh`, tmux는 `koo_contact_resume1500_single_gpu`, 로그/기록은 같은 폴더의 `train.log`·`launch.json`이다. 이미 실행 중이면 중복 시작하지 않는다. MPS는 변경하지 않았다. checkpoint 복원·1501/1502epoch 진행을 확인했다.

`push_toy_contact_align_at`는 `push_toy_random_at_10deg_at_success_tilt`를 기준으로 CONTACT(PUSH edge)의 세 항목만 바꾼 별도 실험이다. 실제 힘 접촉을 성공 조건으로 추가하지 않는다.

| 항목 | 정의 |
| --- | --- |
| CONTACT state | 현재 목표 반대 ray가 만나는 박스 면의 유한 XY 선분까지 각 손 거리로 `H_i=exp(-10*d_i²)` 계산, `H=2*H_L*H_R/(H_L+H_R)` 사용. 양손 중점 대신 각 손을 평가하며 면을 따라 이동/손 Z 자유. 두 점수 모두0이면 H=0 |
| CONTACT progress | `(1-w)*D*G+w*A`. 기존 사람–박스 거리 점수 D에 근거리 감쇠 G를 적용하고 위치 정렬 A를 별도로 더함. 프레임 변화량 보상은 아님 |
| 위치 정렬 A | 박스→사람과 박스→목표의 XY 각도 θ로 `exp(-(1-θ/π)²/alignment_sigma2)`. 반대편180도에서1. 사람과 박스 중심이 일치하면0. 목표 XY 일치 시 박스 yaw+Y 방향을 fallback으로 사용 |
| 근거리 감쇠 G | `near=clamp((gate_start_distance-d)/(gate_start_distance-gate_full_distance),0,1)`, `G=1-near*(1-A)`. 멀리서는1, 가까이서는A. 정렬 보상 A 자체는 감쇠하지 않음 |
| CONTACT 자체 성공 | H≥0.9 및 박스 reset 중심 높이±8cm. 힘/사람 정렬/손 높이/박스 기울기 조건 없음 |
| AT | 기준 실험 그대로: 거리 state/progress, 성공에만 거리φ≥0.9·높이차≤1mm·기울기≤10도. AT 성공 시 CONTACT state/progress/bonus도 포화하여 손을 떼도 유지 |

CONTACT 자체 성공 시에도 기존 규칙대로 CONTACT state/progress/bonus를 포화한다. 각 보상 최대0.2, CONTACT+AT 자기 합계·동료 평균 공유는 기존과 같다. `relationReward.push_at.contact_shaping`의 초기값은 `alignment_weight=0.5`, `alignment_sigma2=0.25`, `gate_full_distance=1.0m`, `gate_start_distance=2.0m`다. 거리는 사람 root–박스 중심 XY 기준이며 면까지 거리가 아니다. 박스 yaw를 목표에 정렬시키는 보상은 없다.

물리·참조 모션·목표 반대편 RSI 배치·후반 샘플링은 기준과 같아서 기존 후보/PD screen을 보존하고 전용 bank의 reward 계약/재사용 출처만 변경했다. 전용 env/train YAML·train/test/VNC·bank/output을 사용하며 기존 checkpoint는 새 reward 계약과 달라 직접 resume/test할 수 없다. 새 실험은 scratch로 시작하고 재개에는 이 실험의 checkpoint만 사용한다. `contact_shaping`이 없는 기존 설정은 기존 계산을 사용한다.

CPU176개 회귀 및 CPU2환경1,000agent reset·양 담당 배정 RSI·부분 reset·24step 물리/관측/보상·6토큰 Actor/Critic/AMP·목표 고정/bank 보존 검증 통과(`output/push_toy_contact_align_at_connections/audit.json`). Python/YAML/셸 문법·문서 링크·diff 확인. 본학습·새 checkpoint 복원·GUI·장기 성능 개선은 미검증이다.

```bash
# 실행 전 GPU 점유 확인; GPU0은 예시이며 기존 학습이 실행 중이면 중복 실행하지 않는다.
nvidia-smi
# scratch 본학습: 2048환경, 최대100만epoch(기준 설정 상속)
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo RESUME_CHECKPOINT= MAX_ITERATIONS= OUTPUT_PATH=output/push_toy_contact_align_at bash tokenhsi/scripts/multi_agent/push_toy_contact_align_at_train.sh 2 2048 2
# 짧은 학습 확인용(본학습과 output 분리)
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo RESUME_CHECKPOINT= MAX_ITERATIONS=2 OUTPUT_PATH=output/push_toy_contact_align_at_check bash tokenhsi/scripts/multi_agent/push_toy_contact_align_at_train.sh 2 2048 2
# 로컬 추론 / 화면 없는 평가: 이 실험에서 저장한 checkpoint 사용
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_toy_contact_align_at_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_toy_contact_align_at_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6097 bash tokenhsi/scripts/multi_agent/push_toy_contact_align_at_vnc.sh <checkpoint.pth> 2 1 2 10
# 비학습 2환경 연결 검사: CPU 물리·CPU 네트워크
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo RSI_SIM=cpu bash tokenhsi/scripts/multi_agent/check_push_contact_align_scene.sh
```


## PUSH 공통 면 CONTACT+AT — 고정 크기

`push_contact_face_at`는 사용자 제공 GPU4·6 방식의 문 없는 CONTACT+AT를 로컬 GPU1용 별도 실험으로 구현한다. 기존 GPU1 soft tilt 설정/실행은 유지하며 자동 전환·중단·본학습 시작은 하지 않았다. 박스는1.3×1.3×1.3m·20kg·마찰0.6 고정, 사람2/박스2·2048학습환경·관측570D/6토큰·기존 랜덤 AT 배치/AMP를 유지한다. 크기/질량 랜덤화는 하지 않는다.

- `pushGoalContact: true`와 `relationReward.contact_at`로 새 경로를 선택한다. 박스 yaw의4개 옆면 중 안쪽 법선과 현재 목표 방향 각도≤60도인 면을 후보로 두고, 양손 중 더 먼 손의 면 XY 선분 거리가 최소인 한 면을 선택한다. 두 손에 같은 면을 적용하고 매 물리 step 다시 고른다. 목표 XY 일치 시 박스 local+Y를 목표 방향으로 사용하며 동률은 고정 후보 순서다. 높이·손 순서·간격은 접촉 기하에 포함하지 않는다.
- 근접 state는 `exp(-10*max(d_R,d_L)^2)`다. **접촉은 이 값>0.9** 및 각 손 반력≥0.5N·상자 전방 힘≥0.5N을 모두 요구한다. 힘 투영은 선택 면의 안쪽 법선 기준이다. 각 물리 step에서 양손 조건을 동시에 만족해야 하고 control step의2회 물리 검사 결과는 OR로 누적한다. 손–박스 pair ID 대조는 하지 않는다. `hand_distance: 0.18`은 새 접촉 gate에 사용하지 않으며 RSI PD screen의 근접 허용과 기존 경로에 남긴다.
- CONTACT 접근 state는 힘 없이도 주어진다. progress는 root–박스 XY 거리 기준 `1/(1+max(d-0.5,0))`다. AT는3D 근접 state·XY 거리 progress `1/(1+max(d-0.1,0))`, 성공은 거리φ≥0.9 및 높이차≤1mm다. CONTACT/AT 각 edge의 state/progress/bonus 최대값은0.4/0.2/0.2다. 실제 접촉 또는 AT 성공이면 CONTACT를 포화하고 AT 성공이면 AT도 포화한다. `push_success`/CONTACT 자체 성공은 실제 누적 접촉을 기록하며 AT 포화와 구분한다. 자기 edge 합산·동료 edge 평균·0.9/0.1 공유는 유지한다.
- **기울기에 따른 state/progress 감쇠와 기본 AT 성공 기울기 제한은 없다.** 선택 옵션 `contact_at.max_tilt`를 명시하면 AT 성공에만 제한한다(현재 미설정). `push.max_tilt: 0.35`는 기존 진단/credit 경로용이며 새 CONTACT/AT 보상을 제한하지 않는다. `at`/`push_at`는 기존 공통 schema 호환 필드로 남지만 새 보상 계산은 `contact_at`를 사용한다.
- RSI는 보행/조작50/50, 조작의70%를 각 clip의 유효 후보 후반30%에서 요청한다. RSI 배치 전에는 양손 월드 위치가 아직 없으므로 목표 반대 ray를 seed로 허용 면을 선택하고 면 중심±접선0.2m·박스 바닥 기준0.8m 목표를 만든다. 팔 보정 후 몸 yaw/XY와6cm stand-off를 정렬하고 몸 Z는 이동하지 않는다. 초기 면 중심 정렬과 rollout의 실제 손 기반 면 선택은 구분한다. 고정점 추종 보상은 없다.

새 env/train YAML·train/test/VNC·RSI bank/output/checkpoint 계약을 분리했다. **기존 soft tilt/CONTACT checkpoint 직접 resume/test는 불가**하며 새 scratch로 시작한다. 현재 실행 중인 GPU1 학습에는 반영되지 않는다.

```bash
# 명령만 준비됨. 기존 학습이 점유한 GPU에 중복 실행하지 않는다.
nvidia-smi
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo RESUME_CHECKPOINT= OUTPUT_PATH=output/push_contact_face_at bash tokenhsi/scripts/multi_agent/push_contact_face_at_train.sh 2 2048 2
# 별도 짧은 학습 확인용; 아직 실행하지 않음
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo RESUME_CHECKPOINT= MAX_ITERATIONS=2 OUTPUT_PATH=output/push_contact_face_at_check bash tokenhsi/scripts/multi_agent/push_contact_face_at_train.sh 2 2048 2
# 새 실험에서 저장한 checkpoint의 로컬 추론 / 화면 없는 평가
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_contact_face_at_test.sh <new_checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_contact_face_at_test.sh <new_checkpoint.pth> 2 2 2 3
# 서버 VNC: 실행 전 local-vnc-viewer의 학습/MPS 보호 규칙 확인
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6098 bash tokenhsi/scripts/multi_agent/push_contact_face_at_vnc.sh <new_checkpoint.pth> 2 1 2 10
# CPU 검사: GPU 비노출·MPS 자동 연결 우회, 학습 실행 없음
TOKENHSI_GPU=-1 TOKENHSI_CONDA_ENV=tokenhsi_koo MPS_GPU_ROOT=/tmp/tokenhsi_cpu_no_mps RSI_SIM=cpu bash tokenhsi/scripts/multi_agent/check_push_contact_face_at_scene.sh
# 전용 bank를 CPU PD 재검사로 갱신; 기존 실험 bank는 변경하지 않음
TOKENHSI_GPU=-1 TOKENHSI_CONDA_ENV=tokenhsi_koo MPS_GPU_ROOT=/tmp/tokenhsi_cpu_no_mps RSI_SIM=cpu bash tokenhsi/scripts/multi_agent/screen_push_contact_face_rsi.sh
```

검증: CPU 회귀199개와 추가 면 선택/60도 경계3개 통과. CPU PD12step 재screen에서 기존90후보 중89개(clip별30/59)를 수용하고 자세 안정성 실패1개를 제외했다. 근접<18cm·발 지지·root/상자 안정성 검사이며 state>0.9/힘 접촉이나 정책 성공을 검증한 것은 아니다. 결과는 `output/push_contact_face_at_rsi_screen/screen.json`이다. CPU2환경1,000agent reset·양 담당 배정 RSI·partial reset·실제24step 물리/유한 보상·목표 고정·570D Actor/Critic/AMP 연결을 확인했다(`output/push_contact_face_at_connections/audit.json`). GPU 실행·본학습·새 checkpoint 복원·GUI·장기 성능은 미검증이다.


<a id="push-goal-3to5m"></a>

## PUSH 공통 면 soft tilt — 상자 기준 목표3~5m

`push_contact_face_at_tilt_soft_3to5m`은 상자의 **RSI 적용 전 초기 중심** 기준으로 거리3~5m와 방향0~360도를 뽑는다. 거리·방향 제안은 각각 균등하며, 두 담당자의 이동/RSI 경로 간격2.2m·다른 사람 시작점과의 여유·자기 사람 시작점에서 목표1m 이상 조건으로 공동 거부하므로 수용된 분포는 완전 균등이 아니다. 기존 사람 기준XY±2m 목표 설정은 유지한다.

박스1.3m/20kg/마찰0.6, CONTACT/AT 보상·기울기5~15도 감쇠·성공10도, AT progress 포화0.1m, 관측570D/6토큰, episode600step과 사람→상자1~2m는 유지한다. RSI는 동일 bank(`push_contact_face_at_tilt_soft_rsi_bank.json`)를 읽기 전용으로 재사용하며, 상자를 초기 거리의 `box_progress` 비율만큼 목표 방향으로 옮긴 뒤 공통 접촉면에 사람을 정렬한다. 따라서 후반 RSI의 **남은 거리까지3~5m로 강제하지 않는다**. bank의 물리/보상/모션/정렬 계약은 동일하다. arena 정규화 기준 `envSpacing: 7`도 유지하므로 멀어진 목표 관측은 기존보다 큰 값을 가질 수 있다.

2026-10-18 CPU 검증: 관련147테스트·2048환경 배치3seed의3~5m/경로 조건, 실제2환경1,000agent reset·두 담당 배정 RSI/목표 고정/부분reset·24step 물리·570D Actor/Critic/AMP 연결 통과. 초기 거리3.0005~4.9985m·RSI 후 남은 거리0.5190~4.9939m·조작 시작47.6%·후반 요청72.27%·reset 기울기0도였다. 기존89후보 모두 새 랜덤 배치의12step PD 지지/자세/손18cm근접 screen 통과(힘 접촉/정책 성공 검사가 아님), bank 파일은 변경하지 않았다.

CPU 전이본: `output/push_contact_face_at_tilt_soft_3to5m_transfer/PushContactFaceATTiltSoft_00017500_3to5m.pth`. 원본은 기존 soft tilt의17500epoch 저장본이며181개 모델/정규화 텐서와 epoch/frame을 그대로 보존하고 optimizer 누적상태·best reward·env state를 초기화했다. rollout/AMP replay는 복원하지 않는다. 새 배치 계약을 명시적으로 기록하며 원본 직접resume는 거부된다. 전이본 strict CPU network load/유한 forward까지 확인했다. **GPU resume/PPO·정책 rollout·장거리 성공률·600step 충분성은 미검증이며 본학습은 시작하지 않았다.** 실제 전환 시 기존 학습의 최종 저장본으로 다시 전이해야 한다(새 출력 파일 지정, 덮어쓰기 거부).

```bash
# CUDA 비노출·MPS 미연결 CPU 검사: 기존 학습은 변경하지 않음
bash tokenhsi/scripts/multi_agent/check_push_contact_face_at_3to5m_scene.sh
bash tokenhsi/scripts/multi_agent/screen_push_contact_face_at_3to5m_rsi.sh
# 명시적 CPU 전이 예시(이미 만든17500 전이본을 덮어쓰지 않음)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/transfer_push_goal_distance_checkpoint.py --source <기존_soft_tilt_최종저장본.pth> --output <새_전이저장본.pth>
# GPU1 기존 학습을 사용자가 종료하고 최종 저장본 전이/점유 확인 후 실행할 본학습 명령
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=356 RESUME_CHECKPOINT=output/push_contact_face_at_tilt_soft_3to5m_transfer/PushContactFaceATTiltSoft_00017500_3to5m.pth OUTPUT_PATH=output/push_contact_face_at_tilt_soft_3to5m MAX_ITERATIONS=1000000 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_3to5m_train.sh 2 2048 2
# 실행 검증은 별도 OUTPUT_PATH=output/push_contact_face_at_tilt_soft_3to5m_check_train·2048환경·MAX_ITERATIONS=<전이epoch+2> 사용
# 로컬 추론 / 화면 없는 평가 / 서버 VNC: 실행 전 로컬 VNC의 GPU/MPS 보호 규칙 필수
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_3to5m_test.sh <전이본_또는_후속저장본.pth> 2 1 2 10
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_3to5m_test.sh <전이본_또는_후속저장본.pth> 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_3to5m_vnc.sh <전이본_또는_후속저장본.pth> 2 1 2 10
```

검사 결과는 `output/push_contact_face_at_tilt_soft_3to5m_{check,checkpoint_check}/audit.json`, `output/push_contact_face_at_tilt_soft_3to5m_screen/screen.json`, 전이 폴더의 `validation.json`이다. 검사 wrapper는 CPU만 허용하고 runtime setup 전 CUDA/MPS 환경을 해제·차단한다. 원본89후보를 다시 screen하지만 bank를 쓰지 않는다. 전이 도구는 새 CPU audit와 원본의 전체 물리/네트워크/RSI 계약을 비교하고 목표 분포 이외 env 변경을 거부한다.

## PUSH 공통 면 CONTACT+AT — AT soft tilt 전이

`push_contact_face_at_tilt_soft`는 `push_contact_face_at`에서 AT state/progress만 기울기5°까지100%, 5~15° 선형 감쇠, 15° 이상0으로 바꾼다. AT 성공은 기존 위치 조건에 **기울기10° 이하**를 추가한다. 목표 도달로 포화된 state/progress에도 감쇠를 적용하므로10°의 AT state/progress는50%다. CONTACT 자체 근접/접근/힘 접촉·보상은 유지하며 CONTACT에는 기울기 제한을 추가하지 않는다. AT 성공을 통한 CONTACT 포화는 새 AT 성공 조건을 따른다. 박스1.3m·20kg·마찰0.6·570D/6토큰·2048환경·물리/RSI 후보는 유지한다.

보상 계약이 달라 원본 checkpoint 직접 resume는 거부한다. `transfer_push_contact_face_tilt_checkpoint.py`는 위 세 기울기 파라미터와 RSI bank 경로 외의 env 차이를 거부하고, 원본 보상/RSI/bank hash/graph 및 PD bank 계약을 검사한다. Actor/Critic/Discriminator·정규화·epoch/frame을 보존하고 optimizer state·best reward·env state를 초기화한다. rollout/AMP replay와 저장 이후 진행은 복원하지 않는다. bank는 물리/후보 변경 없이 보상 계약만 갱신한 별도 파일이다.

2026-10-17 준비한 전이본은 `output/push_contact_face_at_tilt_soft_transfer/PushContactFaceAT_00008000_tilt_soft.pth`이며 원본은 `output/push_contact_face_at/PushContactFaceAT_16-13-07-35/nn/PushContactFaceAT_00008000.pth`다. CPU74개 회귀·181개 가중치/정규화 텐서 동일·대상 metadata/원본 계약 거부·셸 문법/diff 확인. 전이 GPU 복원/PPO/물리 rollout·GUI는 미검증이다. 기존 학습 중단·새 학습 실행은 하지 않았다.

```bash
# 전이본 생성: CPU 전용, 기존 파일 덮어쓰기 거부. 위8000 전이본은 이미 생성됨.
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/transfer_push_contact_face_tilt_checkpoint.py --source <source_checkpoint.pth> --output <new_transfer_checkpoint.pth>
# 학습 재개: 기존 GPU1 학습 종료 확인 후 실행. 현재는 명령만 준비됨.
nvidia-smi
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo SEED=356 RESUME_CHECKPOINT=output/push_contact_face_at_tilt_soft_transfer/PushContactFaceAT_00008000_tilt_soft.pth OUTPUT_PATH=output/push_contact_face_at_tilt_soft MAX_ITERATIONS=1000000 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가 / 서버 VNC: 새 설정의 전이본 또는 후속 저장본 사용
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_test.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_test.sh <checkpoint.pth> 2 2 2 3
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6098 bash tokenhsi/scripts/multi_agent/push_contact_face_at_tilt_soft_vnc.sh <checkpoint.pth> 2 1 2 10
```

시각화 전에는 위 로컬 VNC의 학습/MPS 보호 규칙을 따른다. 전이 검증 기록은 `output/push_contact_face_at_tilt_soft_transfer/validation.json`이다.
