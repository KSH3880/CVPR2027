# TokenHSI on the 8× RTX PRO 6000 Blackwell box

## 실행

```bash
conda activate tokenhsi
cd ~/CVPR2027/TokenHSI

# GPU 지정은 반드시 CUDA_VISIBLE_DEVICES 로 (--sim_device cuda:N 은 죽음, 아래 주의사항 참고)
CUDA_VISIBLE_DEVICES=0 sh tokenhsi/scripts/tokenhsi/stage1_test.sh
CUDA_VISIBLE_DEVICES=1 sh tokenhsi/scripts/single_task/traj_train.sh
```

README의 모든 스크립트가 그대로 동작한다. 단, 이 서버에는 쓸 수 있는 데스크톱 세션이 없어서
test 스크립트는 뷰어를 못 띄운다 → `--headless` 를 붙이거나 아래 뷰어 항목을 참고.

학습 성능: `traj_train` 기준 4096 envs 에서 **약 52,000 FPS** (GPU 1장).

## 멀티 GPU 학습

```bash
bash ~/CVPR2027/scripts/train.sh 1 tokenhsi/scripts/single_task/traj_train.sh   # 평소
bash ~/CVPR2027/scripts/train.sh 8 tokenhsi/scripts/single_task/traj_train.sh   # 동일 레시피, 8장
bash ~/CVPR2027/scripts/train.sh 8 tokenhsi/scripts/single_task/traj_train.sh --fast  # 배치 8배, 빠름
```

`train.sh` 는 GPU 수만 바꿔도 학습이 원본과 같아지도록 `num_envs`, `minibatch_size`,
`amp_minibatch_size` 를 GPU 수로 나눠준다 (1/2/4/8 실행 검증 완료). `--dry-run` 으로 계산 결과만
확인할 수 있고, 나누어떨어지지 않는 GPU 수는 에러로 막는다. `--fast` 는 GPU마다 원본 envs를
그대로 주는 모드로, 배치가 N배 커지는 대신 훨씬 빠르다.

측정값은 [BENCHMARK.md](BENCHMARK.md) 에 정리돼 있다. 요약하면:

| 모드 | 8 GPU 속도 | 학습 |
|---|---|---|
| 동일 레시피 (기본) | 1.16× | 원본과 같음 |
| `--fast` | 7.5× | 배치 8배, lr 조정 필요 |

**동일 레시피 모드는 속도가 거의 안 붙는다.** 총 envs가 4096으로 고정이라 GPU당 512개밖에
안 돌고, 그 정도 일감으로는 Blackwell 한 장을 채울 수 없기 때문이다. 하나의 학습을 빨리 끝내려면
`--fast`, 여러 실험을 빨리 보려면 GPU 1장에 실험 1개씩 8개를 동시에 돌리는 편이 낫다.

**동작 방식** — rl_games 1.1.4는 horovod로 분산 학습을 하는데, horovod는 여기서 소스빌드한 torch에
붙여서 빌드할 수 없다. 그래서 rl_games가 실제로 쓰는 몇 개 함수만 `torch.distributed` 위에 구현한
대체 모듈을 `~/CVPR2027/ddp_shim/horovod/torch/` 에 두고 PYTHONPATH로 주입한다.
`train.sh` 가 알아서 해준다. collective는 NCCL 2.27로 처리한다 (torch 2.4에 들어있는 NCCL 2.20은
sm_120에서 멈춘다. `TOKENHSI_DDP_BACKEND=gloo` 로 CPU fallback 가능).

**알아둘 것** — `--max_iterations` 로 끝나는 실행에서는 rank 0이 먼저 종료하면서 나머지 rank가
`Connection closed by peer` 를 뱉는다. 체크포인트는 이미 저장된 뒤라 무해하다.

## 3D 보기

**브라우저에서 마우스로 조작 (인터랙티브)**

```bash
bash ~/CVPR2027/scripts/tokenhsi_gui.sh --port 6080 sh tokenhsi/scripts/tokenhsi/stage1_test.sh
```
VS Code PORTS 패널에서 6080 포워딩 → `http://localhost:6080/vnc.html`.
옵션: `--port N`, `--res 1920x1080x24`, `--display :N` (기본은 비어있는 디스플레이 자동 선택).
포트만 다르게 주면 여러 개 동시 실행 가능.

**영상으로 녹화**

```bash
xvfb-run -a -s "-screen 0 1280x720x24" \
  python ./tokenhsi/run.py --task HumanoidTraj ... --test --num_envs 1 --record
ffmpeg -framerate 30 -pattern_type glob -i "output/imgs/*/frame_*.png" \
  -c:v libx264 -pix_fmt yuv420p demo.mp4
```

구조: `Xvfb` 가상 디스플레이 → `x11vnc`(loopback only) → `websockify + noVNC`.
IsaacGym 뷰어는 X를 "창" 용도로만 쓰고 렌더링은 GPU가 하므로 가상 디스플레이로 충분하다.

## 주의사항

**GPU 지정** — `--sim_device cuda:N` (N≠0) 을 주면 첫 reset 에서
`CUDA error: an illegal memory access was encountered` 로 죽는다. IsaacGym 쪽 문제이고
`CUDA_VISIBLE_DEVICES=N` 으로 우회한다 (GPU 5에서 확인).

**느린 스크립트** — `stage2_terrain_carry_test` 는 지형 생성 때문에 첫 rollout 까지 20분 이상 걸린다.
멈춘 게 아니다.

**무시해도 되는 경고** — `FBX library failed to load`, `Error for key= ...` 는 정상 출력이다.

## 구성

| 항목 | 값 |
|---|---|
| conda env | `tokenhsi` (python 3.8) |
| PyTorch | 2.4.1 **소스 빌드** (CUDA 12.8, sm_120) |
| IsaacGym | Preview 4 → `~/CVPR2027/isaacgym` |
| pytorch3d | 0.7.7 소스 빌드 |
| 데이터 | HF `lianganimation/TokenHSI` → `tokenhsi/data/` |
| 체크포인트 | `output/single_task/`, `output/tokenhsi/` |
| 빌드 산출물 | `~/CVPR2027/pytorch-src/dist/torch-*.whl` |

### 왜 PyTorch를 빌드했나

이 GPU는 sm_120(Blackwell)인데 python 3.8용 공식 torch 휠은 최대 CUDA 12.4(sm_90)까지만 지원해서
`no kernel image is available for execution on the device` 로 GPU 연산이 전부 실패한다.
IsaacGym Preview 4는 `gym_38.so` 까지만 있어서 python 3.9+ (sm_120 지원 휠)로 올라갈 수도 없다.
그래서 python 3.8 + CUDA 12.8 + `TORCH_CUDA_ARCH_LIST=12.0` 으로 직접 빌드했다.

재빌드가 필요하면 `~/CVPR2027/build_torch.sh`.

### 적용한 패치

| 파일 | 내용 |
|---|---|
| `pytorch-src/cmake/.../select_compute_arch.cmake` | arch 정규식이 한 자리 major만 매칭 → `12.0` 인식 못함. Blackwell 항목 추가 |
| `torch/utils/cpp_extension.py` | `arch[0] + arch[2:]` 로 `12.0` → `1.0` 오파싱. `replace(".", "")` 으로 수정 + supported_arches 에 12.0 추가 |
| `tokenhsi/env/tasks/base_task.py` | `--record` 로 녹화 자동 시작 (헤드리스라 `L` 키를 누를 수 없어서). 2줄, 기본 동작 변화 없음 |

`cpp_extension.py` 패치는 **CUDA 확장을 빌드할 때마다 필요**하다. 소스 트리에도 반영해뒀으니
torch를 재설치하면 다시 적용해야 한다.

### 설치 안 한 것

- **SMPL body models** — 로그인이 필요해서 못 받았다. HF 전처리 데이터를 쓰면 실행에는 불필요하고,
  `tokenhsi/scripts/gen_data.sh` 로 데이터를 직접 재생성할 때만 필요하다.
  받으면 `body_models/smpl/` 에 넣으면 된다.
- **FBX SDK** — 경고만 뜨고 쓰이지 않는다.

## 검증 결과 (13/13)

| 스크립트 | reward |
|---|---|
| single_task: traj / carry / climb / sit | 286 / ok / 87 / 19 |
| tokenhsi: stage1 | 77 |
| comp: traj_carry / sit_carry / climb_carry | 247 / ok / ok |
| objShape: chair / table | ok / ok |
| terrainShape: traj / carry | 282 / 802 |
| longterm | 315 |
| **traj_train** (4096 envs, 20 iter) | 52k FPS, 체크포인트 저장 확인 |

로그: `~/CVPR2027/runs/verify_logs/`
