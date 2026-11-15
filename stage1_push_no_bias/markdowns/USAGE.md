# 설치와 추가 실행 옵션

기본 명령은 [config](config.md), 구현 위치는 [structure](structure.md)를 따른다.

## 실행 환경

모든 wrapper는 `runtime_env.sh`로 기본 conda `tokenhsi`를 활성화하고 저장소 루트에서 실행한다. 다른 환경은 `TOKENHSI_CONDA_ENV=<이름>`, conda 설치 위치는 `CONDA_BASE`로 명시한다.

`TOKENHSI_GPU=0`처럼 물리 GPU를 한 번 지정한다. CUDA·기존 MPS·viewer Vulkan에 같은 GPU를 전달한다. `mps/`는 기존 daemon을 검사하며 실행 wrapper가 daemon을 시작하거나 종료하지 않는다.

PyTorch는 서버 GPU/CUDA 호환 설치본을 사용하며 Isaac Gym은 torch보다 먼저 import한다. `requirements.txt`는 남은 의존성이다. 현재 서버의 conda에는 Isaac Gym과 CUDA 라이브러리 활성화 설정이 있다.

## 데이터와 체크포인트

`tokenhsi/data/dataset_loco_sit_carry_climb.yaml`이 참조하는 `dataset_{amass_loco,amass_climb,sit,carry}/motions`를 공유 데이터에 심링크로 연결한다. 외부 데이터와 원본 checkout은 읽기 전용이다.

이 머신의 공유 데이터는 `/home/cvlab/Desktop/CVPR2027/TokenHSI/tokenhsi/data/`에 있다. 네 `motions` 링크는 Git 관리 대상이 아니므로 새 checkout에서는 다시 연결해야 한다. `FileNotFoundError`가 모션 로딩 중 발생하면 YAML 변경 전에 해당 링크와 원본 파일의 존재를 확인한다.

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

`stage1_push_no_bias`는 같은 고정 비율 scratch 설정에서 actor/critic의 task attention bias만 제거한다. 전용 train/test/VNC·일반/단독 eval wrapper를 사용하고 기본 명령은 [config](config.md), 제거 범위는 [상세](reference/stage1_push_no_bias.md)를 따른다. 재개는 `RESUME_CHECKPOINT`에 같은 no-bias 학습의 PTH를 지정한다. 기존 bias 활성 PTH와의 재개·viewer·평가 혼용은 거부한다.

Stage1 PUSH의 기본 `random_scenario` viewer는 환경 생성 때 PUSH용 상자 슬롯 여부를 고정한다. 적은 환경에서 PUSH 슬롯이 뽑히지 않으면 reset을 반복해도 PUSH가 나타나지 않는다. PUSH만 보려면 기존 test/VNC 명령에 `TASK_GRAPH=push`를 붙인다. 예:

```bash
CKPT=output/stage1_push/Stage1Push_30-23-30-00/nn/Stage1Push.pth

# stage1_push: 4명·4물체, 가능한 행동으로 샘플링
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_test.sh "$CKPT" 4 1 4 10

# stage1_push: 4명·5물체, ON_TOP 1명과 재샘플링된 나머지 과제
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/stage1_push_test.sh "$CKPT" 4 1 5 10

# stage1_push: 8명 모두 독립 PUSH
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/stage1_push_test.sh "$CKPT" 8 1 8 10
```

Stage1 PUSH viewer/test/VNC는 사람 N≥1·물체 M≥N을 지원하며 물체 수 상한은 두지 않는다. 각 사람의 주 물체 N개를 먼저 확보하고, ON_TOP은 남은 M−N개에서 서로 다른 받침을 배정한다. 부족하면 초과 ON_TOP 과제만 크기에 맞는 다른 과제로 다시 뽑는다. `holding_ontop` 고정 요청에도 이 규칙을 적용하므로 M=N이면 전원이 대체 과제를 수행한다. 추가 물체는 distractor로 배치한다. 새 학습은 기존2명·4물체·2048환경, 정식 단독 평가는1명 규약을 유지한다.

PUSH 모션21개는 `tokenhsi/data/jointpush_motions/` 아래 읽기 전용 원본으로 연결한다. 기본 원본은 `/home/cvlab/Downloads/push_amp_raw_rsi_environment_final_20261024/data/motions/`이고 bank의 파일 해시를 검증한다. `checkpoints/Stage1PushFinetune_00021000.pth`는 Downloads의 원본 PTH를 가리킨다. 원격 파일 이름의 jointpush는 여기서는 독립 과제 통합이며 Stage2 공동 PUSH를 포함하지 않는다.

PUSH manifest의 `jointpush_motions/dataset_*/motions/.../ref_motion.npy` 경로마다 원본 패키지의 `push_001`–`push_021` 안에 있는 동일 SHA256 파일을 심링크로 연결한다. 원본 패키지는 디렉터리 구조가 다르므로 `jointpush_motions` 전체를 원본 `data/motions`에 연결하는 것만으로는 해결되지 않는다. 새 scratch 학습에도 기본 네 데이터셋과 PUSH 모션 링크가 모두 필요하다.

## BEFORE PUSH

`stage2_before_push`는 `stage2_before`의 독립 역할에 PUSH를 추가한다. 학습 진입점과 viewer/VNC/일반 평가 명령은 [config](config.md), 샘플링·물리·전이 설정은 [BEFORE PUSH](reference/stage2_before_push.md)에 있다.

새 학습은 기본 `checkpoints/Stage1PushFinetune_00021000.pth`에서 전이한다. `STAGE1_CHECKPOINT=/absolute/path/to/stage1_push.pth`로 다른 Stage1 PUSH 학습본을 선택할 수 있다. 기존 4과제 Stage1 가중치는 사용할 수 없다. resume는 `RESUME_CHECKPOINT`에 같은 BEFORE PUSH 실험의 PTH를 지정한다. viewer와 일반 평가에도 **Stage2 BEFORE PUSH PTH**를 사용한다. 전이 원본인 Stage1 PUSH PTH를 Stage2 저장본으로 대신 사용하지 않는다.

`TASK_GRAPH=push`는 4명 모두 독립 PUSH, `all_solo`는 다섯 행동 독립 샘플링이다. `before_sit`·`before_climb`·`before_stack`은 기존처럼 BEFORE 쌍을 고정한다. 일반 정식 평가는 저장된 학습 설정의 `random_scenario`를 사용하며, Stage2용 1명 단독 평가 wrapper는 제공하지 않는다.

## 공통 Stage2 PUSH와 coupled

`stage2_before_push`와 `stage2_coupled_push`는 동일한 5과제·4family·4종 사람 relation 모델을 사용한다. 관측은1272차원(4명·8토큰), AMP1330차원이며 두 Stage2의 제한 시간은1200step이다. 기존 BEFORE PUSH의5N suffix 저장본은 새 모델과 호환되지 않는다. Stage1 PUSH21k에서 각각 새 전이를 시작한다. 전이 원본은 `STAGE1_CHECKPOINT`, 같은 실험 resume은 `RESUME_CHECKPOINT`로 지정한다. 두 Stage2의 학습/추론/평가 명령은 [config](config.md)에 있다.

coupled viewer의 `TASK_GRAPH=all_coupled`는4명 공동 PUSH, `all_solo`는 독립5행동 혼합, `push`는 독립 PUSH4명이다. 본학습은 random_scenario 고정이며 새 학습은 자동으로 시작하지 않는다. [공통 구조·coupled 상세](reference/stage2_coupled_push.md).

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

## Stage1 PUSH 스케줄 실행 옵션

새 스케줄 실험은 `stage1_push_schedule` 전용 wrapper를 사용한다. 스케줄·체크포인트 분포는 [상세](reference/stage1_push_schedule.md), 기본 train/test/VNC 명령은 [config](config.md)를 따른다. 기존 `stage1_push` PTH를 스케줄 실험으로 resume하지 않는다.

```bash
RESUME_CHECKPOINT=/absolute/path/to/Stage1PushSchedule_00035000.pth TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_schedule_train.sh 2 2048 4
```


## Stage1 PUSH adaptive 실행·재개

기본 seed42의 스크래치 실행은 이전 실행 옵션을 해제한 뒤 전용 wrapper를 사용한다.

```bash
unset CHECK MAX_ITERATIONS OUTPUT_PATH RESUME_CHECKPOINT ENV_CONFIG
TOKENHSI_GPU=0 SEED=42 bash tokenhsi/scripts/multi_agent/stage1_push_adaptive_train.sh 2 2048 4
```

`stage1_push_adaptive_train.sh`는 전용 supervisor를 거쳐 자동 자산 재생성 때 동일 run의 체크포인트와 TensorBoard epoch/frame을 이어 간다. 수동 resume은 원본 run의 이후 PTH를 보존하도록 새 run 디렉터리를 만들며 전역 epoch/frame과 curriculum 상태를 복원한다. 직접 `run.py` 학습은 물리 자산 재생성 supervisor가 없으므로 거부한다. 본학습에는 `CHECK`·`MAX_ITERATIONS`를 설정하지 않는다. 중단 후 가장 최근 `Stage1PushAdaptive_adaptive_*.pth` 또는 일반 저장 PTH를 선택한다.

```bash
TOKENHSI_GPU=0 RESUME_CHECKPOINT=/absolute/path/to/Stage1PushAdaptive_adaptive_00002000.pth bash tokenhsi/scripts/multi_agent/stage1_push_adaptive_train.sh 2 2048 4
```

자동 평가는 영역(region) ever 성공률로 집중 대상·숙련·망각을 판단하며 `adaptive/eval_region_*`에 기록한다. 로그·trace·전환 판단은 `output_etc/check/stage1_push_adaptive/<run>/epoch_<epoch>/attempt_*/`에 저장한다. 평가 실패 시 해당 새 학습 실행은 오류를 드러내고 종료하며, 확률·성공 카운터를 임의 갱신하지 않는다. 중단된 평가의 `evaluation_snapshot.pth`로도 재개할 수 있다. 평가 지표와 수동 정식 평가의 구분·집중 구간 규칙은 [상세](reference/stage1_push_adaptive.md)를 따른다.
