# 설치와 추가 실행 옵션

기본 명령은 [config](config.md), 구현 위치는 [structure](structure.md)를 따른다.

## 실행 환경

모든 wrapper는 `runtime_env.sh`로 기본 conda `tokenhsi`를 활성화하고 저장소 루트에서 실행한다. 다른 환경은 `TOKENHSI_CONDA_ENV=<이름>`, conda 설치 위치는 `CONDA_BASE`로 명시한다.

`TOKENHSI_GPU=0`처럼 물리 GPU를 한 번 지정한다. CUDA·기존 MPS·viewer Vulkan에 같은 GPU를 전달한다. `mps/`는 기존 daemon을 검사하며 실행 wrapper가 daemon을 시작하거나 종료하지 않는다.

PyTorch는 서버 GPU/CUDA 호환 설치본을 사용하며 Isaac Gym은 torch보다 먼저 import한다. `requirements.txt`는 남은 의존성이다. 현재 서버의 conda에는 Isaac Gym과 CUDA 라이브러리 활성화 설정이 있다.

## 데이터와 체크포인트

`tokenhsi/data/dataset_loco_sit_carry_climb.yaml`이 참조하는 `dataset_{amass_loco,amass_climb,sit,carry}/motions`를 공유 데이터에 심링크로 연결한다. 외부 데이터와 원본 checkout은 읽기 전용이다.

기본 전이 가중치는 `checkpoints/Stage1_00078000.pth`다. 현재 링크 대상은 `/home/cvlab/Desktop/CVPR2027/stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth`다. 다른 머신은 링크를 연결하거나 `STAGE1_CHECKPOINT=<절대경로>`로 BEFORE 새 학습의 source를 지정한다.

BEFORE checkpoint는 해당 실험 학습으로 생성한다. fixed 추론에는 기존 run의 `relation_config.yaml`도 필요하다. 확인 학습 저장본은 장기 성능 검증을 의미하지 않는다.

## 설정과 resume

- 새 학습은 선택한 환경·학습 YAML을 읽는다.
- 추론과 `RESUME_CHECKPOINT=<pth>`는 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다. 학습 YAML은 해당 실험 것을 사용한다.
- `ENV_CONFIG=<yaml>`은 명시적인 환경 설정이다. fixed/BEFORE 모두 같은 옵션을 쓴다. 저장 설정이 없으면 현재 YAML로 조용히 대체하지 않는다.
- 기존 fixed snapshot에 sim 전체가 없으면 native fixed sim을 보완한다. 부분 sim과 rest_offset 불일치는 거부한다.
- 인원·환경 수·토큰 수·episode 길이·평가 초기화·영상 CLI를 마지막으로 적용하며 실험 계약을 검사한다.
- 모든 wrapper는 기본 `SEED=42`를 쓴다. `EPISODE_LENGTH`는 추론 episode 길이다.

## 확인 실행과 캐시

`CHECK=1`은 기본 `MAX_ITERATIONS=1`의 확인 학습이다. 환경 수는 fixed2048, BEFORE1024를 유지한다. 확인용 `OUTPUT_PATH`는 `output_etc/check/<검사명>` 아래를 지정한다. wrapper와 직접 `run.py` 모두 경로를 검사한다.

본학습 checkpoint·공용 캐시는 `output/`, 확인 작업과 그 RSI 캐시는 `output_etc/check/`에 둔다. screening 코드 변경 시 캐시 키가 바뀌며 물리 검사를 다시 수행할 수 있다. 원본 캐시는 수정하지 않는다.

## BEFORE 추론 초기화와 거리

- `BEFORE_PICKUP_CARRY_RSI=loco`가 기본이다. RSI433 두 설정은 `mix`, `pickup`, `carry`를 지원한다. `mix`는 후행 ON_TOP의40/30/30 요청 분포이며 유효 프레임/배치 실패는 loco로 fallback한다. mix226 loco 설정은 loco만 지원한다.
- `BEFORE_FOLLOWER_NEAR=0`은 저장된 혼합을 쓴다.1은 near50의 상대 가까움 또는 mix226 중간 모드다.2는 mix226의 가장 짧은 모드이며 near50에서는 거부한다.
- `TASK_ROLE_SWAP=1`은 fixed preset의 역할 교환이다. 과제·스킬 값은 [추론 옵션](config.md#추론-옵션)을 따른다.

## 서버 viewer

로컬 viewer는 화면 세션이 필요하다. VNC는 Xvfb·x11vnc·noVNC·websockify가 필요하며 `PORT`, `RESOLUTION`, `VNC_DIR`, `X11VNC`, `NOVNC_DIR`, `WEBSOCKIFY`로 위치·화면을 지정한다. GUI 로그는 `output_etc/check/gui/`다. 종료 시 이번 wrapper가 띄운 VNC 서비스만 종료한다.

## 이전 이름의 저장본

기존 Stage1 78000과 정리 전 BEFORE 저장본은 그대로 읽을 수 있다. `checkpoint_compat.py`가 저장 설정의 reward 태그·RSI 경로·로컬 전이 링크만 현재 이름으로 변환한다. 모델 tensor와 보상 수치·실험별 계약 검사는 유지한다. 새 학습의 출력 경로와 실험 이름은 현재 컨피그 이름을 따른다.

## 정식 평가와 viewer 구분

`*_test.sh`/`*_vnc.sh`는 인원·RSI를 바꿔 관찰하는 viewer/진단이다. `HEADLESS=1`만 붙여도 정식 프로토콜로 바뀌지는 않는다. 일반 정식 평가는 `*_eval.sh`로 학습 인원을 유지하고, Stage1 전용1명 비교는 `stage1_single_eval.sh`로 실행한다. 시간은 정식 평가·viewer 모두 기본 Stage1 900step, Stage2 1200step이다. [실행 명령](config.md), [평가 규약](reference/evaluation.md)을 따른다.


## Stage1 PUSH

`stage1_push`는 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%의 사용자 지정 비율을 사용하는 스크래치 학습이다. 기본2명·2048환경·4물체·900step, 같은 `tokenhsi` runtime을 사용한다. 주어진21k finetune PTH의 일반 평가는 저장된 PUSH50% 분포를 복원하고 시간만900으로 맞춘다. 옆에 `relation_config.yaml`이 없으면 PTH의 `relation_experiment_config`를 읽는다. 서로 다른 분포/horizon의 PTH를 resume하지 않는다.

`stage1_push_single_eval.sh`는 실제1명·행동별 평가이며 PUSH도 포함한다. AT만9종 크기를 쓰고 PUSH는 학습167종 크기를 쓴다. 기존4과제 macro와 전체5과제 macro를 별도로 보존한다. [실행 명령](config.md), [구현 상세](reference/stage1_push.md)를 따른다.

PUSH 모션21개는 `tokenhsi/data/jointpush_motions/` 아래 읽기 전용 원본으로 연결한다. 기본 원본은 `/home/cvlab/Downloads/push_amp_raw_rsi_environment_final_20261024/data/motions/`이고 bank의 파일 해시를 검증한다. `checkpoints/Stage1PushFinetune_00021000.pth`는 Downloads의 원본 PTH를 가리킨다. 원격 파일 이름의 jointpush는 여기서는 독립 과제 통합이며 Stage2 공동 PUSH를 포함하지 않는다.

## BEFORE PUSH

`stage2_before_push`는 `stage2_before`의 독립 역할에 PUSH를 추가한다. 학습 진입점과 viewer/VNC/일반 평가 명령은 [config](config.md), 샘플링·물리·전이 설정은 [BEFORE PUSH](reference/stage2_before_push.md)에 있다.

새 학습은 기본 `checkpoints/Stage1PushFinetune_00021000.pth`에서 전이한다. `STAGE1_CHECKPOINT=/absolute/path/to/stage1_push.pth`로 다른 Stage1 PUSH 학습본을 선택할 수 있다. 기존 4과제 Stage1 가중치는 사용할 수 없다. resume는 `RESUME_CHECKPOINT`에 같은 BEFORE PUSH 실험의 PTH를 지정한다. viewer와 일반 평가에도 **Stage2 BEFORE PUSH PTH**를 사용한다. 전이 원본인 Stage1 PUSH PTH를 Stage2 저장본으로 대신 사용하지 않는다.

`TASK_GRAPH=push`는 4명 모두 독립 PUSH, `all_solo`는 다섯 행동 독립 샘플링이다. `before_sit`·`before_climb`·`before_stack`은 기존처럼 BEFORE 쌍을 고정한다. 일반 정식 평가는 저장된 학습 설정의 `random_scenario`를 사용하며, Stage2용 1명 단독 평가 wrapper는 제공하지 않는다.

## 공통 Stage2 PUSH와 coupled

`stage2_before_push`와 `stage2_coupled_push`는 동일한 5과제·4family·4종 사람 relation 모델을 사용한다. 관측은1272차원(4명·8토큰), AMP1330차원이며 두 Stage2의 제한 시간은1200step이다. 기존 BEFORE PUSH의5N suffix 저장본은 새 모델과 호환되지 않는다. Stage1 PUSH21k에서 각각 새 전이를 시작한다. 전이 원본은 `STAGE1_CHECKPOINT`, 같은 실험 resume은 `RESUME_CHECKPOINT`로 지정한다. 두 Stage2의 학습/추론/평가 명령은 [config](config.md)에 있다.

coupled viewer의 `TASK_GRAPH=all_coupled`는4명 공동 PUSH, `all_solo`는 독립5행동 혼합, `push`는 독립 PUSH4명이다. 본학습은 random_scenario 고정이며 새 학습은 자동으로 시작하지 않는다. [공통 구조·coupled 상세](reference/stage2_coupled_push.md).

공동 PUSH hard gate는 전원 양손의 뒤쪽 면18cm 접촉으로 판정하며 목표 방향 `aligned` 조건은 제거했다. 단독 PUSH와 현재 목표에 있을 때의 goal saturation 예외는 유지한다. Python 프로세스가 이미 로드한 gate 함수는 파일 수정으로 바뀌지 않으므로 새 코드 적용은 다음 실행부터다.

### Coupled 제로샷 viewer

Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 붙인 관찰용 저장본이다. 추가 optimizer step은0이며, 학습 재개용 optimizer는 포함하지 않는다. 기본1200step이고 아래 명령은4명 모두 같은 상자를 미는 장면을 표시한다.

```bash
# stage2_coupled_push
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth 4 1 8 10
```

독립 행동과2~4명 공동 그룹이 섞인 학습 분포를 보려면 `TASK_GRAPH=random_scenario`로 바꾼다. 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.

### BEFORE PUSH near50

`stage2_before_push_near50`은 `stage2_before_push`의 거리 변형이며 기본4명·1024환경·8토큰·1200step·loco다. `STAGE1_CHECKPOINT`로 Stage1 PUSH 전이 원본을 지정할 수 있고 viewer/평가는 near50 실험의 저장본을 사용한다. mix226 저장본과 혼용하지 않는다. near50의 정확한 거리 부등식은 [BEFORE PUSH](reference/stage2_before_push.md#near50-거리-변형), 실행 명령은 [config](config.md)를 따른다.

### 26k PUSH 평가와 Stage2 제로샷 관찰

사용자가 제공한 가중치는 `pth/Stage1V2Fixed_00078000.pth`(기존4과제), `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`(PUSH 포함)다.26k의 일반 Stage1 평가는 저장된 PUSH50% 분포를 복원하고900step을 사용한다. 현재 YAML의 새 학습 PUSH30%와 구분한다.

```bash
# stage1_push 일반 평가
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_eval.sh pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth --output output_etc/check/push_26k/general

# stage1_push 1명 행동별 단독 평가
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_single_eval.sh pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth --output output_etc/check/push_26k/single

# stage2_before_push_near50 제로샷 viewer
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_test.sh output_etc/check/push_26k/zeroshot/stage2_before_push_near50/ZeroShot_Stage1Push_26000.pth 4 1 8 10

# stage2_coupled_push 제로샷 viewer
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh output_etc/check/push_26k/zeroshot/stage2_coupled_push/ZeroShot_Stage1Push_26000.pth 4 1 8 10
```

두 Stage2 저장본은26k의 기존165 tensor를 그대로 복사하고 coordination을 identity로 초기화했다. 추가 학습0회이며 전이 원본·계약·1200step 장면 설정을 각각 저장한다. near50에서 BEFORE ON_TOP 쌍만 보려면 `TASK_GRAPH=before_stack`, coupled에서 독립/2~4명 그룹 혼합을 보려면 `TASK_GRAPH=random_scenario`를 사용한다. 관찰용 저장본에는 optimizer가 없다. 이 문서의26k 검증/관찰 명령은 기존 YAML의 기본21k 전이 경로를 변경하지 않는다.

PUSH가 포함된 설정의 viewer/VNC/영상 목표구는 반지름20cm로 표시한다(기존3cm). 목표 좌표와 성공 거리 판정은 그대로이며 기존 실행 명령을 다시 실행하면 적용된다.

## Stage2 actor attention 지도

현재1+3고정5000iter의관찰결과는 [interactive HTML](../output_etc/check/stage2_attention_1plus3_5000_20261031_131023/index.html)과 [평균PNG](../output_etc/check/stage2_attention_1plus3_5000_20261031_131023/stage2_attention_mean.png)다. 행=query agent, 열=key agent이며 H0는단독, H1~H3는공동그룹이다.4layer·2head를표시하고시점/전체평균·QK/관계bias/attention을전환한다.

프로젝트루트에서새output경로로수집한다. 이미원시배열이있는경로는보호를위해거부한다. 학습설정과모델은변경하지않는다.

```bash
ATTN_OUT="$PWD/ksh_jointpush/coupled_base/output_etc/check/attention_1plus3_new"
CKPT="$PWD/ksh_jointpush/coupled_base/output/stage2_coupled_26k_people_first_source2_10_goal2_5_20261030_235619/Stage2CoupledPushRegionGoals_30-23-56-23/nn/Stage2CoupledPushRegionGoals_00005000.pth"
TOKENHSI_CONDA_ENV=tokenhsi118 TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=push \
  TOKENHSI_RUN_ENTRY="$PWD/ksh_jointpush/scripts/coupled_stage2_attention_capture.py" \
  STAGE2_ATTN_OUTPUT="$ATTN_OUT" OUTPUT_PATH="$ATTN_OUT/rollout" \
  LD_LIBRARY_PATH=/home/injesus1010/anaconda3/envs/tokenhsi118/lib \
  OMP_NUM_THREADS=1 CXX=/usr/bin/g++-11 \
  bash ksh_jointpush/scripts/coupled_region_goals_test.sh "$CKPT" 4 1 8 3

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  ksh_jointpush/scripts/render_coupled_stage2_attention.py "$ATTN_OUT"
```

읽기전용수집entry는 `ksh_jointpush/scripts/coupled_stage2_attention_capture.py`이며 `STAGE2_ATTN_OUTPUT`, `STAGE2_ATTN_EPISODES`, `STAGE2_ATTN_STEPS`, `STAGE2_ATTN_INTERVAL`을받는다. 재수집에는새output경로를사용한다. actor의원본정규화입력을수집하고agent순서를복원해원본출력과대조한다. raw배열은frame/layer/head/query/key순서다. attention크기와행동인과기여량은같지않으며, QK표시는bias제거ablation이아니다.

### 1+3·2+2·4명 공동의 bias / raw QK / softmax 비교

[세 배치 비교 HTML](../output_etc/check/stage2_attention_comparison_5000_20261031_132618/index.html)은같은5000iter를사용한다. 배치별동일layer·head·시점의관계bias, 스케일전raw QKᵀ, 최종softmax를가로로배치한다. 기본화면은세배치의L4/H1평균이며전체layer/head와장면/step을선택할수있다. QK/√d전환과bias·QK동일색상척도옵션을제공한다. 세배치같은열의색상척도는공통이다.

`softmax(QKᵀ/√32+bias)`가실제확률이다. 평균화면은각항목의평균을따로표시하므로평균logit에softmax를취한값과같지않다. 원시NPZ의 `raw_qk_scores`는스케일전, `qk_scores`는√32로나눈값이다. 기존1+3수집은그대로재사용했으며다른두배치는추가물리rollout을수집했다. 성공사례필터링이나동일초기배치의대응비교는아니다.

프로젝트루트에서세배치를새경로에한번에수집·렌더링하려면:

```bash
ATTN_OUT="$PWD/ksh_jointpush/coupled_base/output_etc/check/attention_three_layouts_new"
CKPT="$PWD/ksh_jointpush/coupled_base/output/stage2_coupled_26k_people_first_source2_10_goal2_5_20261030_235619/Stage2CoupledPushRegionGoals_30-23-56-23/nn/Stage2CoupledPushRegionGoals_00005000.pth"
TOKENHSI_CONDA_ENV=tokenhsi118 TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=push \
  TOKENHSI_RUN_ENTRY="$PWD/ksh_jointpush/scripts/coupled_stage2_attention_capture.py" \
  STAGE2_ATTN_PATTERNS=1plus3,2plus2,4 \
  STAGE2_ATTN_OUTPUT="$ATTN_OUT" OUTPUT_PATH="$ATTN_OUT/rollout" \
  STAGE2_ATTN_EPISODES=3 STAGE2_ATTN_STEPS=600 STAGE2_ATTN_INTERVAL=10 \
  LD_LIBRARY_PATH=/home/injesus1010/anaconda3/envs/tokenhsi118/lib \
  OMP_NUM_THREADS=1 CXX=/usr/bin/g++-11 \
  bash ksh_jointpush/scripts/coupled_region_goals_test.sh "$CKPT" 4 1 8 3

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  ksh_jointpush/scripts/render_coupled_stage2_comparison.py "$ATTN_OUT"
```

여러 배치는동일player프로세스에서순차수집한다. `STAGE2_ATTN_PATTERNS`기본값은 `1plus3`이며단일배치는지정output직하위,여러배치는각그룹이름의하위폴더에저장한다. 비교렌더러는 `1plus3/`, `2plus2/`, `4/`의원본을읽고새 `analysis/`에검증·NPZ를저장하므로기존수집자료를덮어쓰지않는다.

## 로컬 정식 평가와 PUSH 종료 거리

프로젝트 루트에서 아래 명령을 사용한다. 로컬 `tokenhsi118`의 MHA/CUDA11 호환 진입점을 공식 seed driver의 `TOKENHSI_RUN_ENTRY`로 전달하며, 공식 평가 sampler·512환경·seed0/1/2·1200step은 그대로 유지한다. 기존 학습·viewer는 별도 프로세스다. 결과는 task별로 집계하며 Stage2를 task별 단독 장면으로 바꾸지 않는다.

```bash
CKPT="$PWD/ksh_jointpush/coupled_base/output/stage2_coupled_26k_people_first_source2_10_goal2_5_20261030_235619/Stage2CoupledPushRegionGoals_30-23-56-23/nn/Stage2CoupledPushRegionGoals_00015000.pth"
EVAL_OUT="$PWD/ksh_jointpush/coupled_base/output_etc/check/formal_eval_15000_tasks_new"
TOKENHSI_CONDA_ENV=tokenhsi118 TOKENHSI_GPU=0 HEADLESS=1 \
  LD_LIBRARY_PATH=/home/injesus1010/anaconda3/envs/tokenhsi118/lib \
  OMP_NUM_THREADS=1 CXX=/usr/bin/g++-11 \
  bash ksh_jointpush/scripts/coupled_region_goals_eval.sh "$CKPT" --output "$EVAL_OUT"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  ksh_jointpush/coupled_base/tokenhsi/rescore.py "$EVAL_OUT"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  ksh_jointpush/ksh_push_terminal_eval.py "$EVAL_OUT"
```

`summary.json`은 AT·ON_TOP·SIT·CLIMB·PUSH별 점/영역 ever 성공률을 평가 seed 평균·표본 표준편차로 기록한다. `REPORT.md`, `push_terminal_summary.json`, `push_terminal_agents.csv`, `push_terminal_boxes.csv`는 PUSH 종료 거리를 기록한다. 종료 시점은 환경의1200번째 step 또는 전원 낙상한 최초 step이며 성공·실패·낙상을 모두 포함한다. 개인 마지막 유효 시점(개인 낙상 또는 환경 종료)의 거리도 별도 CSV 열에 남긴다.

구역별 목표 모드의 주 거리는 구역 중심→개인 고정 목표의3D 거리다. 전체 물리 상자 중심→공통 목표 거리는 동일 폭 구역 중심들의 평균과 고정 목표들의 평균으로 복원하며 그룹당 한 번만 집계한다. solo에서는 두 거리가 같다. 최종20cm 비율은 종료 위치의 기하 판정이며 낙상 전 ever 성공률과 구분한다. 평균·중앙값·90백분위·XY거리·평가 seed별 거리 평균을 함께 기록한다.

평가 결과는 `output_etc/check/`의 `formal_eval_<iter>_tasks_terminal_push_<실행시각>/`에 저장한다. 최신15000iter 경로는 `formal_eval_15000_tasks_terminal_push_20261101_130229/`이며, 직전8000iter는 `formal_eval_8000_tasks_terminal_push_20261031_195033/`다. 이 가중치는 aligned를 포함한 hard gate로 학습했다. 새 평가 프로세스가 로드한 환경 코드만 aligned 제거 후 버전이며 가중치는 그대로다. 공식 성공 채점은 reward gate와 독립된 기하 판정이다. 실행 당시 source/checkpoint SHA256은 `source_manifest.json`에 보존한다. 보고서·그림의iter 표기는 source manifest의 `checkpoint_epoch` 또는 실제checkpoint 파일명에서 읽는다. 실행/완료 상태는 `runs/jointpush/ksh_jointpush_active_evaluation.json`을 따른다.

### 목표 통과 방향을 반영한 부호 거리

기존 평가의 `initial.pt`와 완전한 trace만 읽어 재집계한다. 학습·시뮬레이션을 다시 실행하지 않는다. 프로젝트 루트에서:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  ksh_jointpush/ksh_push_signed_eval.py "$EVAL_OUT"
```

reset 직후 초기 상자/구역 중심을p₀, 고정 목표를g라 하면 XY 단위 진행방향은 `u=(g−p₀)/‖g−p₀‖`다. `dot(p_end−g,u)>0`이면 기존3D 목표 거리의 크기에+, 그 외에는−를 붙인다. 정확히 목표에 있으면0이며 목표선 위에서 횡방향 오차만 있으면 넘지 않은 쪽(−)으로 분류한다. 현재 상자 방향이나 action 이후 첫frame으로 진행방향을 정하지 않는다.

부호만 붙인 거리와 함께 `signed_along_m=dot(p_end−g,u)`도 기록한다. 이 값은 횡방향 이탈을 제외한 실제 진행방향상의 미달/초과거리다. 종료 시 목표선 통과율과 진행방향으로20cm 이상 통과한 비율도 함께 저장하며, 평균은+/−상쇄가 있음을 명시한다. episode 종료·낙상 포함 기준은 기존 종료 거리와 같다.

산출물은 `PUSH_SIGNED_DISTANCE.md`, `push_signed_terminal_summary.json`, `push_signed_terminal_agents.csv`, `push_signed_terminal_boxes.csv`다. 기존 절대거리·성공률·원시trace는 유지한다. 구역은 개인목표 기준으로 사람당 한 번, 전체 상자/공통목표는 그룹당 한 번 집계한다.

### 종료 성공률과 첫 성공 이후 연속 유지율

기존 general 평가의 완전한 첫 episode trace에서 AT·ON_TOP·SIT·CLIMB·PUSH를 점/영역별 재채점한다. 프로젝트 루트에서:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  ksh_jointpush/ksh_terminal_success_eval.py "$EVAL_OUT"
```

`terminal`은 환경 종료 프레임에서 공식 기하 조건을 만족하고 해당 개인이 낙상하지 않은 경우다. 중간 이탈 후 재도달도 포함한다. `continuous`는 개인별 첫 성공부터 환경 종료까지 매 프레임 성공하고 낙상하지 않은 경우이며, 한 프레임 이탈도 실패다. 별도 최소 유지 시간은 요구하지 않는다. 환경 종료는1200step 또는 유효 인원 전원 낙상 최초step이며, 실패·낙상 사례를 분모에서 제외하지 않는다.

기존 ever 성공은 낙상 프레임까지 포함해 그대로 복원·대조한다. 기존 종료 거리의 최종20cm 비율은 낙상 이후 위치도 포함하는 순수 기하 지표이므로 `geometric_terminal`로 별도 보존한다. PUSH는 혼합 장면의1명 PUSH도 단독으로 집계하며 개인 구역 기준과 그룹 전원 기준을 분리한다. 그룹 종료 성공은 동일 종료 프레임에 전원이 각자 목표를 만족해야 한다.

산출물은 `TERMINAL_SUCCESS.md`, `terminal_success_summary.json`, `terminal_success_agents.csv`, `terminal_success_push_groups.csv`, `terminal_success_verification.json`이다. task 통계는 평가 seed 평균·표본 표준편차와 전체 표본 합산 비율을 함께 기록한다. 원시trace·기존 성공률/거리 JSON·CSV는 SHA256으로 불변을 확인하며 학습·시뮬레이션은 재실행하지 않는다.

## 전원 접촉 전 이동 벌점 실험

`stage2_coupled_push_premature_motion.yaml`은 구역 goal 설정에 `stage2JointPush.premature_motion_penalty`만 추가한다. 기본 YAML은 그대로이며 옵션이 없는 기존 checkpoint는 기존 계산을 사용한다. 내부 정책/학습 config 이름은 `stage2_coupled_push_region_goals`를 재사용한다. 새 학습 출력은 별도 `output/stage2_coupled_push_premature_motion_<timestamp>`이고 기본 전이 원본은 Stage1 PUSH26k다. 기존 Stage2 checkpoint를 이 변형으로 resume하면 보상 계약 불일치로 거부한다.

공동 PUSH에서 전원 양손이 상자 뒷면18cm 이내에 들어오지 않은 동안 물리 상자 중심의 수평 속도가0.05m/s를 넘으면 그룹당 `-0.2 * (speed-0.05)`의 보상 항을 만든다. 각 손의 접촉 반력을 반대로 뒤집어 상자 이동 방향에 투영한 양의 성분만 합산하고, 뒷면 근처의 손만 인정해 그룹 내부 힘 비율로 개인에게 분배한다. 단독/다른 과제·정지·전원 준비·접촉력 없는 관성 이동에는 벌점0이다. 목표 도달로 준비 조건을 우회하지 않는다. 목표/기존 CONTACT 성공 정의는 변경하지 않는다.

물리 simulate 호출마다 계산하고 control step 내 평균을 사용한다. 속도 단위는m/s, 힘은N이며 `weight=0.2`, `speed_threshold=0.05`, `force_epsilon=1e-6`은 초기 실험값이다. epsilon은 수치적 분모 보호이며2N 같은 행동 임계값이 아니다. 회전만 있는 움직임은 현재 제외한다. 정규화된 비율 때문에 그룹 전체 벌점은 인원수에 비례해 늘어나지 않는다.

접촉력은 Isaac Gym의 **손 rigid-body net contact force**이므로 상자와의 접촉쌍을 직접 측정하지 않는다. 뒷면 근접 조건으로 제한하지만 팀원/다른 물체 접촉이 섞일 가능성이 있는 귀속 추정치다. 정밀한 인과 기여도나 실제 전달 힘으로 해석하지 않는다.

TensorBoard `relation/premature_push/{penalty,moving_unready_agent_fraction,positive_hand_force_n}`은 공동 PUSH agent·물리 sample 기준 평균이다. penalty는 최종 task reward에 포함되며 AT/CONTACT 항에는 섞지 않는다. `reward_terms/total` 합산에도 반영된다.

현재 머신에서 짧은 통합 확인은 다음과 같이 실행한다(자동 종료).

```bash
TOKENHSI_CONDA_ENV=tokenhsi118 TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=2 \
  OMP_NUM_THREADS=1 CXX=/usr/bin/g++-11 \
  LD_LIBRARY_PATH=/home/injesus1010/anaconda3/envs/tokenhsi118/lib \
  bash tokenhsi/scripts/multi_agent/stage2_coupled_push_premature_motion_train.sh 4 1024 8
```

본학습·viewer·VNC 명령은 [config](config.md)를 따른다. viewer와 VNC는 이 변형으로 학습한 checkpoint의 저장 설정을 읽는다. A/B 비교 시 gate 코드(특히 aligned 제거 여부), 배치, seed와 나머지 보상을 동일하게 유지한다. 기존15500iter 학습은 aligned가 포함된 프로세스였으므로 현재 소스의 새 실험과 벌점 효과만의 A/B로 해석하지 않는다.


## 최초 출발 대기 평가

기존 trace에는 손 위치가 없다. 정식 평가를 `TOKENHSI_RUN_ENTRY=<CVPR2027>/ksh_jointpush/scripts/coupled_departure_capture.py`로 실행하면 `departure.npz`와 판정 metadata를 추가한다. 로컬 `coupled_region_goals_eval.sh`는 entry를 고정하므로 이 진단에서는 coupled_base의 `tokenhsi/scripts/multi_agent/stage2_coupled_push_region_goals_eval.sh`를 직접 사용한다. checkpoint 옆 저장 설정을 그대로 불러온다.

최초 출발은 수평속도0.05m/s 초과6control sample(약0.2초)의 시작이다. 전원 준비는 양손이 물리 상자 뒤 면18cm 이내인 기존 gate의 위치 조건이며 aligned/goal 예외/접촉력을 포함하지 않는다. 초기 reset을 포함하고 첫 팀원 낙상에서 그룹 관측을 중단한다. `ksh_departure_eval.py <eval-output>`이2/3/4명 물리 그룹별 선출발·출발 시 미준비·무출발·미준비·낙상 censor·준비 인원·준비 전 경로/변위·준비 후 출발 지연을 계산한다. 무준비 그룹의 거리는 관측 종료까지 누적한다.

이번 이전15k/현재7.5k 쌍 비교 실행은 `output_etc/check/departure_compare_15000_7500_20261102_112157/run.sh`, 상태는 CVPR2027 루트의 `runs/jointpush/departure_comparison.json`이다. 같은 실행을 중복 시작하지 않는다. 비교보고서는 완료 후 같은 폴더 REPORT.md에 저장된다.
