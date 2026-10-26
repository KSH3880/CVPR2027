# Stage1 PUSH RSI Uniform 팀 실행 가이드

이 실험은 기존 Stage1 네 과제에 PUSH를 추가해 원본 78000회 모델을 파인튜닝한다. GPU1에서 사용하던 보상·정책 구조·상자 물리를 유지하면서, **PUSH RSI 시작 시 상자를 모션 진행률만큼 이동시키지 않고 목표 거리를 2~10m에서 제안**한다. 학습은 아직 시작하지 않았고 아래 초기 체크포인트는 학습 완료 모델이 아니다.

## 1. 사용할 파일

| 용도 | 저장소 루트 기준 경로 |
| --- | --- |
| 환경 config | [stage1_push_rsi_uniform.yaml](ports/stage1_base_push/tokenhsi/data/cfg/multi_agent/stage1_push_rsi_uniform.yaml) |
| PPO·AMP·네트워크 config | [stage1_push_rsi_uniform.yaml](ports/stage1_base_push/tokenhsi/data/cfg/train/rlg/stage1_push_rsi_uniform.yaml) |
| 내부 RSI bank 계약 | [stage1_push_rsi_uniform_source.yaml](ports/stage1_base_push/tokenhsi/data/cfg/multi_agent/stage1_push_rsi_uniform_source.yaml) |
| 모션 목록 | [dataset_stage1_push_finetune.yaml](ports/stage1_base_push/tokenhsi/data/dataset_stage1_push_finetune.yaml) |
| 초기화 준비 | `tokenhsi/scripts/multi_agent/prepare_stage1_push_rsi_uniform.sh` |
| 학습·평가·VNC | `tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_{train,test,vnc}.sh` |
| CPU 검사 | `tokenhsi/scripts/multi_agent/check_stage1_push_rsi_uniform.sh` |
| 기존 GPU1·JHH와의 차이 | [설정 비교](markdowns/push_rsi_uniform_comparison.md) |

실제 실행 코드는 `ports/stage1_base_push/tokenhsi/` 안에 있다. root의 wrapper가 작업 디렉터리와 Python 경로를 이 port로 고정한다. root의 다른 `tokenhsi/run.py`를 직접 실행하지 않는다. source config는 선별된 native RSI bank를 검증하는 내부 입력이며, 단독 학습용 config가 아니다.

## 2. 실험 설정

| 항목 | 값 |
| --- | --- |
| 사람·상자·학습 환경 수 | 2명·4상자·2048환경 |
| 에피소드 최대 길이 | 1200 제어 step, 낙상 등의 조기 종료 유지 |
| 과제 비율 | PUSH50%, SIT5%, CLIMB12.5%, carry AT/ONTOP 각16.25% |
| PUSH 시작 방식 | 보행50% / RSI50% |
| 보행 배치 | 사람 중심360°·상자 중심까지1~10m 제안. 목표는 환경중심XY±4.5m·최소거리1m·상자정면±45° 및 경로 조건으로 결정 |
| PUSH RSI 배치 | 목표 거리U(2,10)m·상자정면±45° 제안. 상자 위치는 유지하고 사람을 선택된 클립 시점의 접촉 자세로 정렬 |
| RSI 후보 | clip0/3/4/8의949개. 기존 후보·시점 선택 유지, clip7 제외, 후반30%를70% 확률로 요청 |
| PUSH 상자 크기 | X/Y1.60~2.20m·Z1.45~1.70m, min(X,Y)≥1.1Z를 만족하는167종 |
| PUSH 담당 상자 물성 | 크기와 무관하게20kg·마찰0.4, 별도밑판 없음. 바닥마찰1.0 |
| 받침/distractor 2개 | 원본 Stage1 SUPPORT: X/Y0.4~0.8m·Z0.25~0.5m, 밀도100kg/m³·원본마찰1.0 |
| 보상 | 기존 면 기반 CONTACT + AT. 각 state/progress/success0.2, 본인 보상100% |
| 정책·AMP | Transformer4층·2head·64D·GTA, 정책634D·8토큰, AMP1330D |
| 초기화 | 원본78000의 기존 행 보존. PUSH embedding은carry_at에서, PUSH AMP열은carry에서 복사. optimizer/epoch/frame 초기화 |
| 학습 설정 | 학습률2e-5, 입력/AMP 정규화 통계 갱신, 별도 frozen teacher 없음 |

거리·각도는 **제안 분포**다. 사람·상자·이동경로 간섭을 거부하므로 최종 수용 표본의 구간별 빈도가 완전히 같지는 않다. RSI에서는 기존 목표영역XY±4.5m 제한을 적용하지 않고 2~10m 거리 범위를 적용한다. 보행 시작 목표 배치는 기존 방식이다.

### 4상자 역할과 reset

실제 상자는 **담당 물체2개 + SUPPORT2개**다. PUSH 과제의 담당 물체에만 큰 PUSH 크기·20kg·마찰0.4를 적용한다. 담당 과제가 carry/SIT/CLIMB이면 그 과제의 원래 크기를 사용한다. SUPPORT는 PUSH 담당자의 옆에 있어도 큰 PUSH 상자로 만들지 않는다.

물리 actor는 생성 시 한 번 만들고 reset 때 위치와 역할을 배치한다. SUPPORT i+2는 해당 사람의 ON_TOP 과제에서는 받침이 되고, 다른 과제에서는 일반 distractor가 된다. ON_TOP 받침은 원본 목표/참조모션 배치를 유지한다. 사용하지 않는 SUPPORT는 원본 Stage1 원판영역 샘플러로 배치하며, PUSH 이동경로나 사람·다른 상자와 겹치면 다시 뽑는다. 여러 숨겨진 actor 중 교체하는 Stage2 pool을 추가한 것은 아니다.

PUSH 물성·보상·네트워크·과제비율·RSI2~10m/±45°는 이 변경에서 유지했다. SUPPORT 질량은 고정20kg이 아니라 `부피×100kg/m³`로 결정된다. `pushExtension.distractor_mode: stage1_support_v1`이 이 역할/물성 계약을 구분한다.

## 3. 실행 환경과 GitHub에 없는 파일

기존 TokenHSI/Isaac Gym 실행 환경을 사용한다. 이번 로컬 검사는 Python3.8·PyTorch2.0·Isaac Gym의 `gym_38` 바인딩으로 수행했다. conda 환경 이름과 Isaac Gym 설치 위치는 팀원 머신에 맞춘다. VNC 사용 시 저장소의 `run-gui.sh`가 요구하는 VNC/noVNC 도구도 필요하다.

```bash
# 모든 명령은 clone한 저장소 루트에서 실행한다.
export TOKENHSI_CONDA_ENV=tokenhsi  # 본인 머신의 conda 환경 이름으로 변경
```

GitHub에는 코드·YAML·RSI bank JSON을 올린다. **모션 NPY와 checkpoint PTH, 학습 출력은 Git에서 제외**되어 별도 준비가 필요하다. 기존 데이터 원본은 읽기 전용으로 두고 아래 위치에 연결한다.

| 필요 항목 | 배치 위치 |
| --- | --- |
| 원본 Stage1 78000 checkpoint | `ports/stage1_base_push/weights/Stage1V2Fixed_00078000.pth` |
| 기존 SIT·carry·보행·CLIMB 데이터 | root의 `tokenhsi/data/dataset_{sit,carry,amass_loco,amass_climb}/` 아래, manifest의 상대경로와 동일하게 |
| PUSH21개와 PUSH용 보행4개 묶음 | `tokenhsi/data/stage1_push_motion_data_20261020/tokenhsi/data/` 아래 원래 디렉터리 구조 유지 |
| humanoid XML·mesh | root의 `tokenhsi/data/assets/` — Git에 포함된 파일 사용 |

port 안 데이터 경로는 root 데이터로 연결된 상대 심링크다. 기존 디렉터리와 심링크를 덮어쓰지 말고 누락된 `motions/` 또는 파일만 자신의 데이터 위치에 연결한다. 예를 들어 carry의 motions 폴더가 없으면 다음처럼 연결한다.

```bash
ln -s /실제/데이터/dataset_carry/motions tokenhsi/data/dataset_carry/motions
```

원본 checkpoint SHA256:

```text
f5b1637716611c673dd33f3dd894bf081a7dd88b7025943505d8287eba5ba13f
```

준비 스크립트는 필요한 파일과 원본 checkpoint 해시를 읽기 전용으로 확인한다. 아래 명령은 파일 누락 목록만 확인하고 전이본을 만들지 않는다.

```bash
CHECK_ONLY=1 bash tokenhsi/scripts/multi_agent/prepare_stage1_push_rsi_uniform.sh
```

## 4. 초기 PUSH 체크포인트 준비

필요 파일을 연결한 다음 실행한다. GPU와 MPS 연결을 비활성화한 CPU 작업이며 학습은 시작하지 않는다.

```bash
bash tokenhsi/scripts/multi_agent/prepare_stage1_push_rsi_uniform.sh
```

생성 파일:

```text
output/stage1_base_push_rsi_uniform/resume/Stage1PushRsiUniformSupport_from78000.pth
output/stage1_base_push_rsi_uniform/resume/Stage1PushRsiUniformSupport_from78000.json
```

별도 위치의 원본을 읽으려면 `SOURCE_CHECKPOINT=/경로/Stage1V2Fixed_00078000.pth`를 명령 앞에 지정한다. 이미 생성된 대상은 덮어쓰지 않는다. 원본78000과 기존GPU1/JHH PUSH 저장본을 새 실험에 직접 resume하지 않는다. 설정 계약이 다르면 실행 전에 거부한다. 큰 여분상자를 사용하던 종전 `Stage1PushRsiUniform_from78000.pth`도 직접 resume할 수 없으며, 새 `Stage1PushRsiUniformSupport_from78000.pth`를 사용한다.

## 5. CPU 검사

```bash
bash tokenhsi/scripts/multi_agent/check_stage1_push_rsi_uniform.sh
```

회귀 검사와 실제 CPU Isaac Gym reset·상자 물성·checkpoint 복원·정책 물리24step을 검사한다. 검사 환경 수16은 평가/검증용이며, 학습 환경 수2048과 별개다. 결과는 `output_etc/check/push_rsi_uniform/`에 저장한다. 실제 GPU/PPO 학습과 장기 성공률을 검증하는 명령은 아니다.

## 6. 학습

GPU 번호는 자신의 머신에 맞춘다. 실행 전에 GPU 점유와 남아 있는 `RESUME_CHECKPOINT`, `OUTPUT_PATH`, `MAX_ITERATIONS` 값을 확인한다. wrapper는 점유 중인 GPU의 Python CUDA client를 발견하면 시작을 거부하며 기존 프로세스를 종료하지 않는다.

```bash
nvidia-smi
# 초기 전이본에서 시작. GPU0은 예시이며 학습은2048환경이다.
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_train.sh 2 2048 4
```

학습 출력은 `output/stage1_base_push_rsi_uniform/<run>/`이다. TensorBoard와 checkpoint도 이 경로에 저장된다. 짧은 학습 확인은 별도출력과 반복수 제한을 함께 사용한다.

```bash
TOKENHSI_GPU=0 OUTPUT_PATH=output/stage1_base_push_rsi_uniform_check MAX_ITERATIONS=2 \
bash tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_train.sh 2 2048 4
```

이 실험에서 학습한 checkpoint를 재개하려면 다음처럼 명시한다.

```bash
TOKENHSI_GPU=0 RESUME_CHECKPOINT=output/stage1_base_push_rsi_uniform/<run>/nn/<checkpoint.pth> \
bash tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_train.sh 2 2048 4
```

## 7. 추론·VNC

아래 `<checkpoint.pth>`에는 **이 실험에서 학습한** 저장본을 지정한다. 준비용 epoch0 전이본을 사용해도 동작은 검사할 수 있지만 PUSH를 파인튜닝한 정책은 아니다. 평가의 기본 시작 방식은 보행이며, 목표 배치도 기존 보행 square/cone 방식을 사용한다.

```bash
# 로컬 viewer — 원본 학습과 분리한 GPU 사용
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 \
bash tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_test.sh <checkpoint.pth> 2 1 4 10

# 화면 없는 평가
TOKENHSI_GPU=0 HEADLESS=1 \
bash tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_test.sh <checkpoint.pth> 2 1 4 10

# 서버 VNC, 1280×720·1환경
TOKENHSI_GPU=0 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 \
RESOLUTION=1280x720x24 PORT=6116 \
bash tokenhsi/scripts/multi_agent/stage1_push_rsi_uniform_vnc.sh <checkpoint.pth> 2 1 4 10
```

VNC 접속은 `http://localhost:6116/vnc.html?autoconnect=1&resize=remote`이다. 원격 서버라면 SSH 터널 등 기존 접속 방식을 사용한다. 학습 중 GPU와 같은 MPS에 viewer를 연결하지 않는다. 학습용 GPU를 비우려고 기존 학습이나 MPS를 종료하지 않는다.

## 8. GitHub 전달 시 포함할 내용

전용 YAML과 실행 wrapper뿐 아니라 `ports/stage1_base_push/`의 PUSH task·scene sampler·계약 검사·준비 도구 등 이번 변경에 필요한 코드도 함께 커밋해야 한다. 현재 작업은 미커밋 변경을 포함하므로 **config 파일만 올리면 새 clone에서 동작하지 않는다.** port 안의 상대 심링크도 포함한다.

```bash
git status --short
git diff --check
```

이 가이드와 config, 코드, RSI bank JSON은 Git에 포함하고 `output/`, `output_etc/`, `*.pth`, 모션 `*.npy`는 기존 제외 규칙을 유지한다. 팀원에게 별도 데이터 위치와 원본78000 파일을 안내한다. 저장소에 복사된 과거 학습 PID·절대경로·launcher는 다른 머신의 실행 명령으로 사용하지 않는다.

공유용 명령으로 회귀16개·PUSH reset1280agent·정책 물리24step과 기존과제/PUSH 혼합 reset2560agent를 확인했다. 실제 RSI645건의 거리는2.017~9.999m, 각도는최대44.93°였다. SUPPORT의 ON_TOP/일반distractor 배치도 확인했다. 새 초기전이본은직전uniform실험의모델/정규화tensor와동일하다. GPU/PPO·장기성공률은미검증이며 Git commit/push는실행하지 않았다. 상세 차이는 [비교 문서](markdowns/push_rsi_uniform_comparison.md)에 있다.
