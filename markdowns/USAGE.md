# TokenHSI — 설치 구성 및 사용법

아래 내용은 추론 13종 + 학습 2종을 실제로 돌려 검증한 결과입니다.
검증에 사용한 참고 사양은 **RTX 4080 16GB / CUDA 11.7 / Ubuntu 22.04**이며,
표에 적힌 VRAM·FPS 수치는 그 기준입니다. 경로와 환경 이름은 머신마다 다르므로
이 문서에서는 placeholder로 씁니다.

이 문서에서 쓰는 표기:

| 표기 | 의미 |
|---|---|
| `$TOKENHSI_ROOT` | 이 repo를 clone한 경로 |
| `$TOKENHSI_CONDA_ENV` | 사용하는 conda 환경 이름 (기본 `tokenhsi`) |
| `$VNC_HOME` | 원격 GUI 도구를 설치한 경로 (예: `~/opt/vnc`) |

`tokenhsi/scripts/multi_agent/` 스크립트들은 이 값들을 **동일한 이름의 환경 변수**로
읽습니다. repo 경로는 스크립트 위치에서 자동 계산하므로 지정할 필요가 없습니다.

---

## 1. 환경 요약

| 항목 | 값 |
|---|---|
| conda 환경 | Python 3.8 (이름은 자유, 기본 가정은 `tokenhsi`) |
| PyTorch | 2.0.0+cu118 |
| IsaacGym | Preview 4, editable 설치 (`pip install -e <isaacgym>/python`) |
| pytorch3d | 0.7.7 (소스 빌드) |
| 저장소 | `$TOKENHSI_ROOT` |
| 데이터 | `$TOKENHSI_ROOT/tokenhsi/data/` (약 2GB, 별도 다운로드) |
| 체크포인트 | `$TOKENHSI_ROOT/output/` (13개, 약 1GB, 별도 다운로드) |

### 시스템 환경을 건드리지 않고 설치하는 방법

- **시스템 CUDA 툴킷을 바꿀 필요가 없습니다.** PyTorch 휠은 자체 CUDA 런타임을 번들로
  포함하므로 시스템 툴킷 버전과 무관하게 동작합니다. 실제 제약은 드라이버 버전입니다
  (검증 환경은 드라이버 535 / CUDA 12.2 지원, 시스템 툴킷은 11.7 그대로 두었습니다).
- pytorch3d 컴파일용 CUDA 11.8 nvcc는 **conda 환경 내부에만** 설치하면 됩니다.
- `LD_LIBRARY_PATH`는 `~/.bashrc`가 아니라 conda activate 훅
  (`$CONDA_PREFIX/etc/conda/activate.d/tokenhsi_env.sh`)에서 설정하세요.
  → 다른 conda 환경에 영향이 없습니다.
- 원격 GUI 스택(TurboVNC/VirtualGL/noVNC)은 **root 권한 없이** 홈 디렉터리
  (`~/.local`, `$VNC_HOME`)에 설치할 수 있습니다.

### 시작하기

```bash
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi}"
cd "$TOKENHSI_ROOT"
```

---

## 2. 추론 (Inference / Test)

### 2.1 헤드리스 실행 (SSH에서 바로)

테스트 스크립트는 `--headless`가 없으면 GUI 창을 띄웁니다. SSH에서 화면 없이 돌리려면
`--headless`를 붙이세요.

```bash
# 예: path-following
python ./tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml \
    --checkpoint output/single_task/ckpt_traj.pth \
    --test --num_envs 16 --headless
```

> **테스트는 무한 루프입니다.** 스스로 종료되지 않으니 `Ctrl+C`로 끊으세요.

### 2.2 원격 GUI로 보기

→ [5. 원격 GUI](#5-원격-gui-vscode-브라우저에서-보기) 참고.

repo에 포함된 헬퍼만으로도 충분합니다. 임시 Xvfb 디스플레이를 띄우고 noVNC로 노출합니다.

```bash
sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/single_task/traj_test.sh
```

상시 VNC 데스크톱을 따로 운영한다면 그쪽 스크립트를 쓰면 됩니다.

```bash
"$VNC_HOME"/start-vnc.sh                                        # 최초 1회
"$VNC_HOME"/run-gui.sh tokenhsi/scripts/single_task/traj_test.sh
```

### 2.3 검증 완료된 전체 스크립트

모두 실제로 실행해 정상 동작을 확인했습니다.

| 스크립트 | 기본 num_envs | 최초 로딩 시간 |
|---|---|---|
| `single_task/traj_test.sh` | 16 | ~30초 |
| `single_task/sit_test.sh` | 16 | ~40초 |
| `single_task/climb_test.sh` | 16 | ~40초 |
| `single_task/carry_test.sh` | 16 | ~60초 |
| `tokenhsi/stage1_test.sh` | 16 | ~90초 |
| `tokenhsi/stage2_comp_traj_carry_test.sh` | 16 | ~60초 |
| `tokenhsi/stage2_comp_sit_carry_test.sh` | 16 | ~60초 |
| `tokenhsi/stage2_comp_climb_carry_test.sh` | 16 | ~60초 |
| `tokenhsi/stage2_object_chair_test.sh` | 16 | **~8분** |
| `tokenhsi/stage2_object_table_test.sh` | 16 | ~90초 |
| `tokenhsi/stage2_terrain_traj_test.sh` | 1 | ~2분 |
| `tokenhsi/stage2_terrain_carry_test.sh` | 1 | ~5분 |
| `tokenhsi/stage2_longterm_test.sh` | 1 | **~12분** |

> `object_chair`와 `longterm`은 데이터·메시 로딩만 8~12분 걸립니다.
> 멈춘 게 아니니 기다리세요. `nvidia-smi`로 프로세스가 살아있는지 확인하면 됩니다.

### 2.4 정량 평가 (Eval)

```bash
sh tokenhsi/scripts/tokenhsi/stage1_eval.sh carry   # traj | sit | climb | carry
```

eval은 헤드리스로 512 trial을 돌고 **스스로 종료**합니다. 결과는
`output/metrics/ckpt_stage1/<task>/metrics_<시각>.json`에 저장됩니다.

실측 결과 (carry):

```
ObjectSet_test_0  success_rate 0.994
ObjectSet_test_1  success_rate 0.969
ObjectSet_test_2  success_rate 0.959
```

---

## 3. 학습 (Training)

### ⚠️ 반드시 `--num_envs`를 낮추세요

저장소 스크립트의 기본값은 **4096**이며, 16GB VRAM에서는 **OOM으로 실패**합니다.

| num_envs | 결과 |
|---|---|
| 4096 (기본값) | ❌ OOM |
| 2048 | ⚠️ 위험 |
| **1024** | ✅ 권장 — stage1 기준 VRAM 약 11.4GB |

```bash
# 단일 태스크 (traj) — 약 21k FPS
python ./tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml \
    --num_envs 1024 --headless

# TokenHSI 통합 트랜스포머 (stage1) — 약 16k FPS
python ./tokenhsi/run.py --task HumanoidTrajSitCarryClimb \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs 1024 --headless
```

`--output_path <dir>`로 결과 경로를 지정할 수 있습니다 (기본 `output/`).
실행마다 `Humanoid_<날짜-시각>/` 디렉터리가 생기고 그 안에 `nn/`(체크포인트)와
`summaries/`(TensorBoard)가 저장됩니다.

### 학습 모니터링

```bash
tensorboard --logdir output/          # 브라우저에서 localhost:6006
```

로그에는 `fps step:` 줄만 찍히고 보상은 TensorBoard로만 나갑니다.
정상 학습 시 `episode_lengths`와 `reward_*`가 상승합니다 (stage1 139 iteration 기준
episode_lengths 31 → 57, reward_climb −2.8 → 15.2 확인).

### 학습 재개 / 파인튜닝

```bash
--checkpoint <경로.pth> --resume 1
```

---

## 4. 뷰어 단축키

| 키 | 기능 |
|---|---|
| `F` | 휴머노이드에 카메라 고정 |
| 우클릭 + `WASD` | 시점 이동 |
| `Shift` + 우클릭 + `WASD` | 시점 빠르게 이동 |
| `K` | 디버그 라인 표시 |
| `L` | 스크린샷 녹화 시작/중지 (`output/imgs/`) |

녹화한 이미지를 mp4로:

```bash
python lpanlib/others/video.py --imgs_dir output/imgs/<경로> --delete_imgs
```

---

## 5. 원격 GUI (VSCode 브라우저에서 보기)

로컬 머신 → VS Code Remote-SSH 환경에서 IsaacGym 창을 브라우저로 보는 구성입니다.
전부 **root 권한 없이** 홈 디렉터리에 설치할 수 있습니다.

### 가장 간단한 방법: repo의 `run-gui.sh`

별도 설치나 상시 데스크톱 없이, 실행할 때마다 임시 X 디스플레이를 띄웁니다.
필요한 것은 `Xvfb`, `x11vnc`, `noVNC`, `websockify` 네 가지이고, 스크립트가 `PATH`와
표준 설치 위치를 자동 탐색합니다. 못 찾으면 아래 변수로 알려주면 됩니다.

```bash
# 전부 선택 사항 — 자동 탐색이 실패할 때만
export X11VNC=/path/to/x11vnc          # 또는 VNC_DIR=<사용자 설치 prefix>
export NOVNC_DIR=/path/to/novnc        # vnc.html이 있는 디렉터리
export WEBSOCKIFY=/path/to/websockify
export PORT=6080                       # 웹 포트 (기본 6080)

sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/multi_agent/ma_carry_watch.sh 2 4 3
```

### 상시 VNC 데스크톱을 쓰는 경우 (선택)

여러 세션에서 창을 계속 유지하고 싶을 때의 구성입니다. 아래 경로는 예시이며
설치 위치는 자유입니다 (`$VNC_HOME`은 본인이 정한 경로).

| 구성요소 | 위치 예시 | 역할 |
|---|---|---|
| TurboVNC 3.3 | `~/.local/opt/TurboVNC` | 가상 데스크톱 X 서버 |
| noVNC + websockify | `$VNC_HOME/novnc` | VNC → 웹(WebSocket) 게이트웨이 |
| VirtualGL 3.1.5 | `~/.local/opt/VirtualGL` | OpenGL 앱 GPU 가속 (IsaacGym엔 불필요) |
| twm | `~/.local/bin/twm` | 창 관리자 |
| 실행 스크립트 | `$VNC_HOME/` | start / run / stop |

### 사용 순서

**1) 서버에서 VNC 데스크톱 시작** (한 번만)

```bash
"$VNC_HOME"/start-vnc.sh              # 기본 :2, 1920x1080
"$VNC_HOME"/start-vnc.sh 2 1600x1000  # 해상도 지정
```

**2) VS Code에서 포트 포워딩**

`PORTS` 탭 → `Forward a Port` → **6080** 입력.
(Remote-SSH가 자동 감지하는 경우도 있습니다.)

**3) VS Code 내장 브라우저로 열기**

`Cmd+Shift+P` → `Simple Browser: Show` → 아래 주소 입력:

```
http://localhost:6080/vnc.html?autoconnect=1&resize=remote
```

**4) IsaacGym 실행**

```bash
"$VNC_HOME"/run-gui.sh tokenhsi/scripts/single_task/traj_test.sh
"$VNC_HOME"/run-gui.sh tokenhsi/scripts/tokenhsi/stage1_test.sh
```

**5) 종료**

```bash
"$VNC_HOME"/stop-vnc.sh
```

### 동작 원리 (중요)

- **IsaacGym은 OpenGL이 아니라 Vulkan으로 렌더링합니다.** 따라서 VirtualGL은
  IsaacGym 가속에 관여하지 않습니다. 핵심은 Vulkan 로더가 NVIDIA ICD를 쓰도록
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`을 지정하는 것이고,
  이건 `run-gui.sh`가 자동으로 설정합니다.
  (검증 머신에는 ICD가 6개 — intel, llvmpipe, radeon 등 — 깔려 있어서 지정하지 않으면
  소프트웨어 드라이버가 선택되었습니다. `ls /usr/share/vulkan/icd.d/`로 확인하세요.)
- 검증: `nvidia-smi`에서 프로세스가 `C+G`(Compute+Graphics)로 표시되면 GPU 렌더링입니다.
- 순수 OpenGL 앱은 `USE_VGL=1`로 VirtualGL을 켜세요:
  ```bash
  USE_VGL=1 "$VNC_HOME"/run-gui.sh glxgears
  ```
  VirtualGL 없이 `:2`에서 OpenGL을 돌리면 llvmpipe(소프트웨어)로 떨어집니다.

### 보안

VNC 데스크톱에는 **비밀번호가 없습니다.** 대신 Xvnc(5902)와 noVNC(6080) 모두
`127.0.0.1`에만 바인딩되어 있어 LAN에서 접근할 수 없고, SSH 터널을 통해서만 열립니다.
`0.0.0.0`으로 바꾸지 마세요. 외부 노출이 필요하면 먼저 VNC 비밀번호를 설정하세요:

```bash
<TurboVNC 설치 경로>/bin/vncpasswd
# 이후 start-vnc.sh의 -securitytypes 를 TLSVnc,VncAuth 로 변경
```

---

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 학습 시 `CUDA out of memory` | `--num_envs 1024`로 낮추세요. 기본값 4096은 16GB에서 불가. |
| 추론/학습 동시 실행 시 OOM | **IsaacGym 작업은 한 번에 하나만** 돌리세요. |
| 테스트가 몇 분째 출력 없음 | 정상입니다. `object_chair` ~8분, `longterm` ~12분 로딩. |
| 테스트가 끝나지 않음 | test 모드는 무한 루프입니다. `Ctrl+C`로 종료. |
| GPU 메모리가 안 돌아옴 | 죽다 만 프로세스 확인: `nvidia-smi` → `kill -9 <pid>`. |
| `ImportError: libpython3.8.so.1.0` | conda 환경을 다시 activate 하세요 (훅이 `LD_LIBRARY_PATH` 설정). |
| `No module named 'pytorch3d'` | conda 환경 activate 누락. pytorch3d는 **모든** 태스크의 필수 의존성입니다 (`tokenhsi/utils/parse_task.py`에서 무조건 import). |
| `Error: FBX library failed to load` | 무해한 경고입니다. 무시하세요. |
| `Error for key= global_root_yaw_rotation` | poselib의 정상 출력입니다. 오류 아닙니다. |
| VNC 화면이 검게만 나옴 | 앱이 아직 로딩 중이거나 창이 없는 상태입니다. `DISPLAY=:2 xdpyinfo`로 서버 확인. |
| `libvglfaker.so cannot be preloaded` | `LD_LIBRARY_PATH`에 VirtualGL의 `lib` 디렉터리가 필요합니다. `run-gui.sh`는 자동 처리. |

---

## 7. 데이터 재생성 (선택)

기본적으로는 HuggingFace 전처리 데이터를 쓰므로 **필요 없습니다.**
원본(AMASS/SAMP/OMOMO)에서 다시 만들려면 추가로 필요합니다:

1. [SMPL body models](https://smpl.is.tue.mpg.de/) 다운로드 (로그인 필요) → `body_models/smpl/`에 배치
2. `tokenhsi/data/dataset_cfg.yaml`에 원본 데이터셋 경로 기입
3. `bash tokenhsi/scripts/gen_data.sh`

> SMPL body model은 데이터를 직접 재생성할 때만 필요하고, 전처리 데이터를 쓰는
> 학습·추론에는 없어도 됩니다.
