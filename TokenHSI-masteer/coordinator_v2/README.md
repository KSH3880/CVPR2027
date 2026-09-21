# Coordinator V2

Frozen MS18 앞에서 동작하는 opt-in 상위 planner다.

- actual-state fixed-window GRU encoder
- low-dimensional joint path/speed proposal
- recurrent ensemble executor-aware world model
- uncertainty-aware MPPI
- BOOM-style value-weighted proposal alignment
- 기존 33-point root/box/goal 및 MS18 bridge 계약 유지

현재 Carry 환경은 agent별 box/goal이 고정이므로 V2도 우선 joint path, speed,
pickup dwell만 계획한다. `task_dim`과 `scene_dim`은 role/assignment 및 scene 입력을
추가하기 위한 명시적 확장점이며, 현재 단계에서 자율 role 발견을 주장하지 않는다.

초기 체크포인트 생성(학습 결과가 아님):

```bash
CUDA_VISIBLE_DEVICES='' python3 ../scripts/masteer/init_wm_mppi_checkpoint.py \
  ../runs/coord/v2/init.pth
```

MS18 연결:

```bash
MS_TASK=HumanoidMACoordCarry COORD_PROVIDER=learned COORD_MODEL=v2 \
COORD_CKPT=/absolute/path/to/trained-v2.pth ...
```
