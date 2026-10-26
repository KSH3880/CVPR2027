# koo_1021: 현재 GPU1 Stage1 PUSH 파인튜닝 가이드

이 문서는 `stage1_fix_koo`의 **현재 GPU1 학습 설정**을 공유하기 위한 가이드다. 사용할 브랜치는 `koo_1021`, config 이름은 **`stage1_push_finetune`**이다. 별도 복사본의 `koo_push_1026` / `stage1_push_rsi_uniform`과 구분한다.

현재 학습은 원본 Stage1 78000회 모델에 PUSH를 추가한 뒤, 파인튜닝 9000회 저장본에서 에피소드 길이를1200step으로 바꿔 재개한 실행이다. 이 문서 작성 과정에서는 학습 코드나 설정을 변경하거나 새 학습을 시작하지 않았다.

## 1. 진입점과 설정

| 용도 | 저장소 루트 기준 경로 |
| --- | --- |
| 실제 환경 config | [stage1_push_finetune.yaml](ports/stage1_base_push/tokenhsi/data/cfg/multi_agent/stage1_push_finetune.yaml) |
| PPO·AMP·네트워크 config | [stage1_push_finetune.yaml](ports/stage1_base_push/tokenhsi/data/cfg/train/rlg/stage1_push_finetune.yaml) |
| 내부 native RSI 계약 | [stage1_push_finetune_source.yaml](ports/stage1_base_push/tokenhsi/data/cfg/multi_agent/stage1_push_finetune_source.yaml) |
| 모션 manifest | [dataset_stage1_push_finetune.yaml](ports/stage1_base_push/tokenhsi/data/dataset_stage1_push_finetune.yaml) |
| 학습 | `tokenhsi/scripts/multi_agent/stage1_push_finetune_train.sh` |
| 로컬·headless 평가 | `tokenhsi/scripts/multi_agent/stage1_push_finetune_test.sh` |
| VNC | `tokenhsi/scripts/multi_agent/stage1_push_finetune_vnc.sh` |
| 설정 상세·복사본과의 차이 | [현재 학습 설정 명세](markdowns/stage1_push_finetune_settings.md) |

root의 wrapper는 `ports/stage1_base_push/`를 작업 디렉터리로 사용하고 이 port의 Python 코드를 import한다. root의 다른 실험용 `tokenhsi/run.py`를 직접 실행하지 않는다. `_source.yaml`은 bank 검증용 내부 입력이며 단독 학습 config가 아니다.

주요 설정은 PUSH50% + 기존 네 과제50%, 사람2명·실제상자4개·2048환경, PUSH 상자167종·20kg·마찰0.4다. **현재 실행의 길이는1200step**이며 YAML 기본600step과 다르다. 재현 명령에는 `TOKENHSI_TRAIN_EPISODE_LENGTH=1200`을 명시해야 한다.

## 2. GitHub에 없는 파일 준비

GitHub에는 코드·YAML·RSI bank JSON을 포함한다. 모션 NPY, PTH, `output/`, `output_etc/`는 제외되므로 별도로 준비해야 한다. 코드를 push하는 것만으로 현재 학습 가중치까지 전달되지는 않는다.

| 항목 | 준비 위치 |
| --- | --- |
| 원본 Stage1 78000 모델 | `ports/stage1_base_push/weights/Stage1V2Fixed_00078000.pth` |
| 기존 SIT·carry·보행·CLIMB 모션 | root `tokenhsi/data/dataset_{sit,carry,amass_loco,amass_climb}/` 아래 manifest와 같은 상대경로 |
| PUSH21개·PUSH용 보행4개 묶음 | root `tokenhsi/data/stage1_push_motion_data_20261020/tokenhsi/data/` 아래 원래 폴더 구조 |
| 학습한 PUSH checkpoint | 팀 내 별도 전달. 평가 또는 재개 시 해당 경로를 명시 |

port의 데이터 경로는 root 데이터에 대한 상대 심링크다. 원본 데이터는 읽기 전용으로 취급하며, 기존 디렉터리와 심링크를 덮어쓰지 않고 누락된 경로를 연결한다. humanoid XML과 mesh는 Git에 포함된 `tokenhsi/data/assets/`를 사용한다.

검증한 원본 78000 모델의 SHA256은 다음과 같다.

```text
f5b1637716611c673dd33f3dd894bf081a7dd88b7025943505d8287eba5ba13f
```

## 3. 새 clone에서 원본78000 전이본 만들기

이미 구성된 TokenHSI/Isaac Gym conda 환경을 사용한다. 로컬 환경 이름은 `tokenhsi_koo`이며, 다른 머신은 자신의 환경 이름으로 바꾼다. 아래 CPU 작업은 초기화된 PUSH 모델을 만들며 학습을 시작하지 않는다.

```bash
# 저장소 루트에서 실행
conda activate tokenhsi_koo
cd ports/stage1_base_push
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/tokenhsi:$PWD" python prepare_finetune.py transfer
cd ../..
```

생성 경로:

```text
output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.pth
output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.json
```

기존 대상은 덮어쓰지 않는다. 코드에 포함된 bank는 선별·검증된949후보를 사용하므로 bank 생성 명령을 다시 실행할 필요가 없다. 필요한 모션이나 asset이 빠져 있으면 파일을 먼저 연결해야 한다.

이 전이본은 기존 네 과제 가중치를 보존하고 PUSH embedding/AMP family를 추가한 **epoch0 모델**이다. 현재 학습 완료분의 PUSH 성능을 보려면 별도로 전달받은 학습 checkpoint를 사용한다.

## 4. 현재 설정으로 새 학습을 시작하는 명령

다른 머신 또는 사용 가능한 별도 GPU에서 실행하는 예시다. 현재 GPU1 학습이 실행 중인 머신에서 중복 실행하지 않는다. GPU 번호0은 예시이며 먼저 점유를 확인한다.

```bash
nvidia-smi
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_TRAIN_EPISODE_LENGTH=1200 SEED=42 \
OUTPUT_PATH=output/stage1_base_push_finetune_stable_ep1200 \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_train.sh 2 2048 4
```

이 명령은 준비한78000 전이본에서 새 파인튜닝을 시작한다. 현재 GPU1처럼 기존 파인튜닝 저장본에서 이어가려면 `RESUME_CHECKPOINT`를 명시한다.

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_TRAIN_EPISODE_LENGTH=1200 SEED=42 \
RESUME_CHECKPOINT=/전달받은/Stage1PushFinetune_checkpoint.pth \
OUTPUT_PATH=output/stage1_base_push_finetune_stable_ep1200 \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_train.sh 2 2048 4
```

학습 환경 수2048은 유지한다. 본학습 전 셸에 남아 있는 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT`를 확인한다. wrapper는 점유 GPU의 Python CUDA client를 발견하면 시작을 거부한다. 기존 프로세스나 MPS를 종료해 공간을 만들지 않는다.

## 5. 현재 로컬 실행의 재현 기준

다음은2026-10-26 확인한 실행 이력이며, 미래의 최신 checkpoint 경로를 뜻하지 않는다.

| 항목 | 확인값 |
| --- | --- |
| 코드 위치 | `/home/inkyu/projects/CVPR_stage1/stage1_fix_koo/ports/stage1_base_push` |
| 재개 입력 | `output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth` |
| 출력 run | `output/stage1_base_push_finetune_stable_ep1200/Stage1PushFinetune_26-11-17-19` |
| 실행 설정 | GPU1 MPS·2명·2048환경·4상자·seed42·1200step |
| 확인에 사용 가능한 학습본 예시 | 위 출력 run의 `nn/Stage1PushFinetune_00010000.pth` |
| 실행 설정 기록 | 위 출력 run의 `relation_config.yaml` |

9000회는 원본78000 이후의 PUSH 파인튜닝 학습 번호다. 현재 학습을 재현할 때는 원본78000과 파인튜닝9000을 혼동하지 않는다. 다른 머신에서는 파일을 별도 전달받아 자신의 경로로 지정한다. `output/`의 로컬 `launch.sh`는 Git에서 제외되며 절대경로·MPS 설정을 포함하므로 공유 실행 진입점으로 사용하지 않는다.

## 6. 평가와 VNC

`<checkpoint.pth>`에는 `stage1_push_finetune` 계약의 학습 저장본을 지정한다. 기본 평가는 PUSH·보행 시작이다. 학습과 같은 에피소드 길이를 보려면 `TOKENHSI_VIEWER_EPISODE_LENGTH=1200`을 명시한다.

```bash
# 로컬 viewer
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_GUI_GPU_MODE=native \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_EPISODE_LENGTH=1200 HEADLESS=0 \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_test.sh <checkpoint.pth> 2 1 4 10

# 화면 없는 평가
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_EPISODE_LENGTH=1200 HEADLESS=1 \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_test.sh <checkpoint.pth> 2 1 4 10

# 서버 VNC
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi_koo TOKENHSI_GUI_GPU_MODE=native \
TOKENHSI_VIEWER_LOW_VRAM=1 TOKENHSI_VIEWER_EPISODE_LENGTH=1200 \
RESOLUTION=1280x720x24 PORT=6108 \
bash tokenhsi/scripts/multi_agent/stage1_push_finetune_vnc.sh <checkpoint.pth> 2 1 4 10
```

VNC는 `http://localhost:6108/vnc.html?autoconnect=1&resize=remote`로 접속한다. 실행 직전에 GPU와 포트 점유를 확인한다. 학습 중 GPU의 MPS에 viewer를 연결하지 않고 별도 GPU/native 경로를 사용한다. 혼합과제 평가는 `TASK_GRAPH=random_scenario`로 선택할 수 있다. 정량 평가 프로토콜은 [평가 문서](markdowns/evaluation_push_finetune.md)를 따른다.

## 7. koo_1021에 commit/push하기

이 절의 명령은 사용자가 직접 실행한다. 문서 작성 시 현재 브랜치가 `koo_1021`, origin이 `https://github.com/KSH3880/CVPR2027`인 것을 확인했다.

```bash
cd /home/inkyu/projects/CVPR_stage1/stage1_fix_koo
git branch --show-current
git remote -v
git status --short
```

출력이 `koo_1021`인지 확인한다. 다른 브랜치라면 변경을 정리한 뒤 기존 브랜치로 이동하고, 이름이 같은 새 브랜치를 만들지 않는다. 아래 `git add -A`는 이 저장소의 기존 미커밋 평가/viewer 변경도 모두 포함하므로 목록을 확인한다.

```bash
git add -A
git --no-pager diff --cached --stat
git --no-pager diff --cached --check
git --no-pager diff --cached --name-only
git status --short
```

환경/학습/source YAML, 모션 manifest, RSI bank JSON, port의 관련 코드와 실행 스크립트, 이 가이드를 함께 포함한다. config만 커밋하면 새 clone에서 동작하지 않는다. `*.pth`, 모션 `*.npy`, `output/`, `output_etc/`가 포함되지 않았는지 확인한다.

```bash
git commit -m "Share running Stage1 PUSH fine-tuning baseline and guide"
git push -u origin koo_1021
```

위 조회 명령은 `--no-pager`를 사용하므로 `(END)` 화면으로 들어가지 않는다. 다른 Git 조회 화면에서 `(END)`가 보이면 `q`로 나온다. push가 원격 이력 차이로 거부되면 force push하지 않고 `git fetch origin` 후 이력을 먼저 확인한다.

## 8. 이번 문서의 확인 범위

실행 중 학습의 CLI·작업 디렉터리·GPU/MPS 환경과 저장된 `relation_config.yaml`을 읽기 전용으로 확인했다. 소스와 저장 설정의 graph/reward/상자/scene/RSI/extension 일치와 bank의949후보·`box_progress=0.85×phase`를 확인했다. 문서 링크·경로·Git 제외 규칙을 점검했다. 이번 작업은 문서 작성이며 새 시뮬레이션·학습·Git commit/push는 실행하지 않았다.
