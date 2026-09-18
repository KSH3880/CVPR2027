# 현재 서버 실행 환경

이 저장소의 학습 기준은 **NVIDIA RTX PRO 6000 Blackwell / 2048 환경**이다.
실험별 학습·평가·VNC 명령과 checkpoint 규칙은 [config.md](config.md)를 따른다.

## 환경과 데이터

| 항목 | 현재 서버 |
| --- | --- |
| 저장소 | `/home/hwanhee/ksh/approach_distance_success` |
| conda 환경 | `tokenhsi`, Python 3.8 |
| PyTorch / CUDA build | `2.4.0a0+gitee1b680` / `12.8` (현재 설치본 확인) |
| Isaac Gym | `/home/hwanhee/CVPR2027/isaacgym` |
| 공유 데이터 원본 | `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/` |
| 학습 환경 수 | **2048** |
| VNC | Xvfb + `$HOME/opt/vnc`의 x11vnc + `$HOME/opt/novnc` |

현재 설치된 conda 환경을 사용한다. 공유 데이터는 심링크로 읽으며 원본을 수정하지 않는다.
새 서버로 이관할 때는 [PORTING_GUIDE.md](PORTING_GUIDE.md)의 경로·의존성 확인 절차를 따른다.

```bash
cd /home/hwanhee/ksh/approach_distance_success
conda activate tokenhsi
nvidia-smi
```

학습 스크립트가 `runtime_env.sh`를 통해 conda와 CUDA 환경을 준비한다.
GPU는 명령마다 `TOKENHSI_GPU`로 명시한다.

## 학습과 짧은 검증

```bash
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_train.sh 2 2048 3
```

짧은 검증도 2048 환경과 본학습 설정을 사용하고 반복 횟수만 제한한다.

```bash
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi MAX_ITERATIONS=1 \
RESUME_CHECKPOINT= OUTPUT_PATH=output/approach_distance_success_no_sat_check \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_train.sh 2 2048 3
```

`SMOKE`에 따른 환경 수·학습 설정 자동 축소는 제거했다. 본학습 전에 `MAX_ITERATIONS`,
`OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여 설정을 확인한다.

## 평가와 원격 viewer

평가·VNC는 학습 환경 수와 별개다. viewer는 보통 환경 1개로 확인한다.
`TOKENHSI_GPU=5`를 명령 맨 앞에 두면 CUDA와 렌더링 GPU에 함께 적용된다.
같은 터미널에서 계속 사용할 경우 `export TOKENHSI_GPU=5`를 먼저 실행해도 된다. 실행 명령, checkpoint 선택, 6080 포워딩은
[config.md의 VNC 절](config.md#원격-서버에서-vnc로-시각화)에 있다.

## 문제 확인

| 증상 | 확인할 것 |
| --- | --- |
| GPU 선택이 예상과 다름 | `nvidia-smi`의 PID와 GPU 번호, GUI 시작 로그의 물리 GPU. 내부 `cuda:0`은 선택한 GPU의 로컬 번호다. |
| CUDA 메모리 부족 | 선택한 GPU의 다른 프로세스, 실제 환경 수, actor/object 수와 batch 설정. 기준 환경 수는 2048이다. |
| `libpython3.8.so.1.0` 로드 실패 | `tokenhsi` 환경과 `runtime_env.sh` 적용 여부 |
| 데이터 파일 누락 | `tokenhsi/data/dataset_*/` 심링크와 YAML 상대 경로 |
| VNC 실행 실패 | `VNC_DIR`, `NOVNC_DIR`, 포트 점유와 `/tmp/tokenhsi_gui_<PORT>_*.log` |
| 검은 VNC 화면 | 서버 터미널의 모델 로딩·checkpoint 오류, viewer 창 생성 여부 |

학습 출력과 지표는 `output/<실험>/<run>/`에 저장된다. 지표 의미는
[relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다.
