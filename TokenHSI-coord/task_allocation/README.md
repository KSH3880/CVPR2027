# Frozen ms18 위의 주기적 box 할당

초기 구현은 2 agents / 2 boxes다. 모델 자체는 가변 token 수를 지원하지만,
물리 환경과 공동 action은 identity `[0,1]` / swap `[1,0]` 두 permutation이다.
box j의 goal j는 episode 동안 고정하고, 접근 중인 box의 담당만 변경한다.
한 번 grasp proxy가 검출되거나 한 box가 배송 완료되면 현재 permutation을
episode 끝까지 고정한다. 운반 중 인계나 배송 후 추가 box 선택은 포함하지 않는다.

## 구조와 입력

`coordinator/task_allocation.py`의 agent self-attention → box/goal 쌍 내부
masked self-attention → agent query/task key·value cross-attention을 사용한다.
공유 pair score를 합해 두 공동 할당의 categorical logits를 만든다.
중복 box 선택은 불가능하며, argmax 대신 공동 action을 샘플링해 PPO로 학습한다.
GT trajectory / scenario id / entity ID embedding은 입력하지 않는다.

- agent `[B,2,16]`: root xyz/quaternion/linear·angular velocity 13개,
  현재 담당 box의 held proxy, phase/3, 남은 episode 비율.
- box `[B,2,20]`: 실제 box pose/velocity 13개, size 3개, 현재 owner one-hot 2개,
  held proxy, 배송 완료 여부. goal `[B,2,3]`는 box 순서와 대응한다.
- 위치의 xy는 env 초기 agent 중심에 대해 표현한다. z와 world 회전/속도는 유지한다.

6개 위치(agent 2, box 2, goal 2)를 기본 ±3m 영역에서 rejection sampling한다.
최소 간격은 1.2m와 box 대각선+0.3m 중 큰 값이다. box는 바닥에서 시작한다.
reset 시 humanoid/box/goal tensor를 실제 actor 축으로 쓰고 기존 single-commit
reset을 사용한다. goal 높이는 box 반높이이며 randomHeight는 꺼져 있다.

실행 경로는 root→box→goal의 두 직선이며 grasp 이후 root→goal 직선으로 갱신한다.
속도 명령은 1.5m/s, 담당 또는 phase 변경 때 기존 경로 커서를 초기화한다.
장애물 회피/안전한 인계/학습된 경로 계획은 없다.

## 보상과 시간축

기본 30 simulator step마다 판단한다. ms18은 deterministic/no_grad 실행하고
optimizer는 할당 모델과 critic만 갱신한다.

```text
매 실제 step: -1 * dt + 10 * 새로 배송된 box 수 - 40 * 실패 종료
판단 시:      -0.1 * 담당이 변경된 agent 수
```

첫 할당에는 변경 비용이 없다. 유지 보너스는 없다. 배송은 grasp 이력이 있는
물리 box가 고정 goal의 3D 거리 0.3m 이내, 속도 0.2m/s 미만, 미보유 상태를
10 step 유지한 geometry proxy다. 접촉 기반 성공 판정을 검증한 것은 아니다.
모든 box 배송 시 성공 종료하며, native fall/timeout 종료는 미배송이면 실패 비용을 준다.

구간 보상은 각 step의 gamma 할인 합과 시작 시 변경 비용이다. gamma=0.999,
lambda=0.995는 simulator step 기준이다. GAE bootstrap은 실제 실행 길이의
`gamma ** duration`, trace는 `(gamma*lambda) ** duration`을 사용한다.
종료 transition은 bootstrap하지 않는다. 구간 중 완료된 env는 이후 보상에서
제외하고 구간 끝에서 row ID로 reset한다. reset 이후 episode를 이전 action에
합산하지 않는다. 다음 판단 전까지 새 episode를 진행하지 않는다.

## 실행

Isaac Gym 사용 가능한 conda 환경을 활성화한 상태에서 workspace 루트에서 실행한다. launcher는 `mps/shell.sh`의 `mps_use 6`으로 기존 GPU6 MPS를 확인하고 `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY`와 GPU UUID를 자동 적용한다. 기본 root는 `/tmp/mps-test-${UID}`이며 `MPS_GPU_ROOT`로 다른 전용 root를 지정할 수 있다. MPS를 시작·종료하지 않으며 연결 검증이 실패하면 학습 전에 종료한다. MPS 환경변수는 run.env에 기록한다.
GPU는 6만 허용하며 다른 번호는 launcher에서 거부한다. 점유 상태는 실행 전에 확인한다. 새 tag만 허용하고 기존 출력은 덮어쓰지 않는다.
기본 ms18은 `TokenHSI-masteer/output/stack/`, Stage1은 `TokenHSI-masteer/output/tokenhsi/`의 checkpoint다. 다른 호환 weight는 두 번째 인자와 `MS_CKPT`로 지정한다. motion/학습 cfg는 기존 ms18 데이터의 절대 경로를 사용한다.

```bash
conda activate tokenhsi_juan
MA_GPU=6 ALLOC_MODE=train ALLOC_ENVS=64 ALLOC_ITERS=100 ALLOC_SEED=0 \
  bash TokenHSI-coord/task_allocation/launch.sh allocation_s0

# 4개의 실제 박스, 2명의 agent. GPU6 기존 MPS에 연결해 짧게 검증한다.
# 운반 중 재할당은 금지하며 배송 완료 후 다음 박스를 선택한다.
ALLOC_BOXES=4 ALLOC_MPS=1 ALLOC_ENVS=2 ALLOC_ITERS=1 ALLOC_HORIZON=40 ALLOC_VERIFY=1 \
  bash TokenHSI-coord/task_allocation/launch.sh allocation_four_mps_probe

# 비-MPS 설정 확인. 존재하지 않는 pipe 경로로 기존 MPS 연결을 우회한다.
ALLOC_MPS=0 ALLOC_ENVS=2 ALLOC_ITERS=1 \
  bash TokenHSI-coord/task_allocation/launch.sh allocation_nomps_probe '' --dry-run

# GPU6 MPS 연결과 실제 설정만 확인. 출력 폴더/학습 프로세스를 만들지 않는다.
MA_GPU=6 ALLOC_ENVS=64 ALLOC_ITERS=100 \
  bash TokenHSI-coord/task_allocation/launch.sh allocation_s0 '' --dry-run

# 다른 호환 checkpoint 지정
MA_GPU=6 ALLOC_ENVS=64 MS_CKPT=/absolute/path/ckpt_stage1.pth \
  bash TokenHSI-coord/task_allocation/launch.sh allocation_s0 /absolute/path/ms18.pth

# 짧은 실제 연결 검증: 2 env, 2 macro decisions, 1 PPO update
MA_GPU=6 ALLOC_ENVS=2 ALLOC_ITERS=1 ALLOC_HORIZON=2 ALLOC_EPOCHS=1 \
  MS_CKPT=/absolute/path/ckpt_stage1.pth \
  bash TokenHSI-coord/task_allocation/launch.sh allocation_smoke /absolute/path/ms18.pth
```

산출물: `runs/task_allocation/<tag>/env.yaml`, `run.env`, `config.json`,
`run.log`, `metrics.jsonl`, `allocation_latest.pth`, 10 iteration 간격 numbered PTH.
run.env에는 기본값까지 실제 적용한 설정이 기록된다. `ALLOC_INIT`으로 할당 checkpoint와
optimizer를 재개할 수 있다. simulator episode/RNG는 새로 시작한다.
`ALLOC_MODE=eval ALLOC_INIT=...`은 deterministic matching과 로그만 수행한다.
재개/평가 시 같은 env 설정을 재현해야 하며 checkpoint는 interval, reward, 할인,
layout, episode 길이와 두 executor checkpoint 경로 불일치를 거부한다.

## 검증

```bash
cd TokenHSI-coord
python -m unittest discover -s task_allocation/tests -v
python -m unittest coordinator.tests.test_task_allocation -v
```

CPU 8개 + attention 3개 테스트 통과: 유효 permutation/운반 lock, 최초 할당·전환 비용,
step 시간/배송/실패 보상, 실제 duration 할인과 terminal bootstrap 차단, 랜덤 배치 간격,
PPO attention weight 갱신, 구간 중 종료의 보상 격리 및 row reset, attention mask와
순서 등변성. Python import/CLI 및 shell syntax 확인.
2026-10-02 GPU 6 실제 검증 완료:

| 실행 | 실제 env-step 합 | 배송 box | 팀 완료 | 실패 | 재할당 |
|---|---:|---:|---:|---:|---:|
| 2 env / 60 step / PPO 1 update | 120 | 0 | 0 | 0 | 1 |
| 저장 모델 / 720 step 평가 | 1438 | 0 | 0 | 2 | 17 |
| 초기 거리 기반 고정 담당 / 720 step 평가 | 1406 | 8 | 4 | 0 | 0 |

팀 완료는 reset을 포함한 완료 episode 수이며 2개의 env 개수가 분모가 아니다.
미완료 episode도 있어 위 표로 성공률을 추정하지 않는다. 단일 seed 기능 검증이다.
배송은 앞서 정의한 geometry proxy다. `held_box_steps`는 누적 grasp lock의
box-step 수로 실제 손 접촉 시간은 아니다. `root_displacement`는 macro 시작/끝의
agent 이동 거리 합이며 reset transition은 제외해 전체 이동 거리와 다르다.

PPO smoke에서 ms18 model state 불변, allocation 갱신, checkpoint 재로드의
확률/critic 출력 일치를 assert했다. eval에서는 model parameter 불변과
재할당 시 물리 root/box/goal 불변을 확인했다. 초기 거리 기반 고정 담당 대조군은
`ALLOC_MODE=eval ALLOC_BASELINE=nearest_initial ALLOC_INIT=...`으로 실행한다.
`ALLOC_VERIFY=1`은 위 고정/재로드 검증 및 verification.json 기록을 켠다.
기록은 `runs/task_allocation/readiness_20261002.json` 및 각 tag의 run.log,
metrics.jsonl, verification.json에 있다. 물리 실행/학습 연결은 준비됐지만,
동적 할당 모델의 수렴/성능은 아직 검증하지 않았다.

초기 실행 a는 상대 motion 경로 누락으로 실패했고 b는 사용자 GPU6 전용
지시 직후 해당 GPU3 검증 프로세스만 중단했다. 두 산출물과 종료 이유를 보존했다.
