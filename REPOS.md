# CVPR2027 — Repo 구성

TokenHSI 기반 multi-agent HOI 연구. 아래 3개는 모두 **원본 TokenHSI의 독립 복사본**이며,
목적별로 분리해서 서로 간섭 없이 실험한다.

| 레포 | 역할 | 과거 상태 스냅샷 (2026-08-16) |
|---|---|---|
| [TokenHSI/](TokenHSI/) | 원본 baseline / reference. **수정 금지** | pristine |
| [TokenHSI-steer/](TokenHSI-steer/) | **Steerable Carry** — action policy에 steering input 추가 | 작업중. F11 최종 확정 / F13 관측 위상 제거 진행 |
| [TokenHSI-ma/](TokenHSI-ma/) | **Multi-agent base** — 2-agent 환경/베이스 실험 | 작업중. **M0 에서 막힘** — 텐서 평탄화가 A≥2 에서 view 가 안 됨 |

현재 상태는 이 표가 아니라 [plan/PLAN.md](plan/PLAN.md)와 트랙 PLAN을 따른다. 새 세션은
[docs/OPERATIONS.md](docs/OPERATIONS.md)에서 시작하며, `HANDOFF_ma.md`는 과거 스냅샷이다.

---

## 멀티 GPU

TokenHSI 소스는 건드리지 않는다. 멀티 GPU는 전부 레포 **바깥**에 있다.

- [ddp_shim/horovod/](ddp_shim/horovod/) — horovod API(`rank/size/local_rank/allreduce/broadcast_*/DistributedOptimizer`)를
  `torch.distributed` 위에 구현한 shim. backend는 `TOKENHSI_DDP_BACKEND`(기본 `nccl`).
  PYTHONPATH로 주입해서 TokenHSI의 기존 `--horovod` 경로를 그대로 태운다.
- [scripts/train.sh](scripts/train.sh) — 주 entry point. 1/2/4/8 GPU. 기본 `same` 모드는 `num_envs`/`minibatch_size`/
  `amp_minibatch_size`를 GPU 수로 나눠서 원본 single-GPU recipe와 동일한 batch·update 수를 유지한다.
  `--fast`는 GPU마다 full `num_envs`(batch N배), `--dry-run`은 설정만 출력.
- [scripts/train_multigpu.sh](scripts/train_multigpu.sh) — recipe 조정 없이 torchrun으로만 감싸는 단순 버전.

**대상 레포는 `TOKENHSI_REPO`로 고른다** (기본 `TokenHSI`):

```bash
TOKENHSI_REPO=TokenHSI-steer bash scripts/train.sh 8 tokenhsi/scripts/single_task/carry_train.sh
TOKENHSI_REPO=TokenHSI-ma    bash scripts/train.sh 4 tokenhsi/scripts/single_task/carry_train.sh --dry-run
```

같은 변수를 [scripts/verify_all.sh](scripts/verify_all.sh), [scripts/verify_rerun.sh](scripts/verify_rerun.sh),
[scripts/sweep_knobs.sh](scripts/sweep_knobs.sh), [scripts/tokenhsi_gui.sh](scripts/tokenhsi_gui.sh)도 받는다.
생성물은 전부 `runs/` 아래로 모이고 레포 이름으로 나뉜다 (`runs/gen_cfgs/<repo>/`, `runs/verify_logs/<repo>/`, `runs/sweep_logs/<repo>/`, `runs/sweep_cfgs/<repo>/`).

---

## 왜 나눠져 있나 (최종 그림)

전체 방법론은 [METHOD_V2.md](METHOD_V2.md)에 있다 (Notion `2. Methodology Draft · V2`의 로컬 사본). 요약하면:

```
Task/Role spec → Shared World Model → K개 coordinated future traj τ¹..τᴷ
→ Planner (dependency / collision / completion) → τ*
→ steering command 추출 → TokenHSI-based Action Policy → Simulator → replan
```

- WM/Planner는 **trajectory space**에서 planning한다. (target space 아님)
- Action Policy는 traj 전체를 받지 않고, 매 step **steering command**만 받아 실행한다.
- traj는 실행 중에도 계속 갱신된다 → online / receding-horizon.

`-steer`는 이 파이프라인의 **아래쪽 절반(action policy가 steering에 반응하는가)** 을,
`-ma`는 **위쪽 절반(여러 agent가 공유 환경에서 coordination 하는가)** 을 각각 떼어서 먼저 검증한다.

---

## TokenHSI-steer

### 목표
기존 TokenHSI Carry 성능을 유지한 채, **외부 steering input으로 이동 경로를 조종**할 수 있게 만든다.
"dynamic trajectory를 실행한다"가 아니라 **"내가 원하는 대로 steer되는가 + 그 결과 traj를 잘 따라가는가"** 가 핵심.

입력 구성:
```
현재 state (human / object / scene) + skill=Carry + final target + steering input
```
- **final target** = 결국 물체를 어디에 놓을 것인가 (기존 그대로 유지)
- **steering input** = 지금 어느 방향/어느 지점으로 움직일 것인가 (신규, 매 step 갱신 가능)

이 둘을 **분리**하는 것이 요점. 기존 Path Following + Carry(comp task)는 고정 traj를 따라가는 설정이라
그대로 쓰지 않는다. 참고만 한다.

### 이미 검증된 것 (학습 없이 target만 직접 움직인 probe) — 이 수치가 기준
- 매 step 현재 위치 기준 traj 위 **약 1.2 m 앞 point**를 target으로 사용
- traj 이탈 보정항을 더하면 tracking이 조금 더 좋아짐
- **목적지 약 1.5 m 전**부터는 steering point 대신 원래 final target을 줘야 placement가 정상 동작
- lookahead가 **1.0 m보다 짧으면 placement 동작이 일찍 발생**
- **1.5 m 구간 내 46° 이상 급커브**는 피할 것
- 전체 traj 정확도보다 **현재 위치 앞쪽 local steering point의 정확도**가 중요
- 20종 traj에서 무튜닝으로 **R² 0.91–0.997, 성공률 100%**

> 이 규칙을 그대로 쓸 필요는 없다. steering point 대신 direction 등 다른 representation도 가능.

### 코드 앵커 (경로는 TokenHSI-steer 기준)
- Carry subtask obs = **42-D**: 등록 [`humanoid_traj_sit_carry_climb.py:450`](TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L450), packing [`:2167`](TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2167)
  `box_vel(3) + box_ang_vel(3) + box_pos(3) + box_rot_6d(6) + box_bps(24) + box_tar_pos(3)`
- **final target은 `local_box_tar_pos` 3-D 하나뿐** ([`:2149`](TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2149), root-local). probe는 이 3-D를 덮어쓴 것.
- Traj subtask obs = `2 × num_traj_samples` (=20), 등록 [`:348`](TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L348), 생성 [`_fetch_traj_samples:902`](TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L902)
  → `progress_buf * dt`로 **시간 인덱싱**해서 `_traj_gen.calc_pos` 호출. 이 시간 의존성이 곧 "pre-defined traj" 구조.
  **표현(root-local 2D point N개)은 그대로 재사용 가능**, 소스만 외부 주입 buffer로 교체하면 online이 된다.
- 토큰화: [`amp_network_builder_transformer.py:33-38`](TokenHSI-steer/tokenhsi/learning/transformer/amp_network_builder_transformer.py#L33-L38)이 `each_subtask_obs_size`별 tokenizer 생성
  → steering은 **carry 토큰을 42→45로 넓히는 게 아니라 새 토큰으로 추가**해야 pretrained tokenizer가 안 깨진다.
- 그 메커니즘이 이미 있음: [`amp_network_builder_transformer_adapt.py:153-160`](TokenHSI-steer/tokenhsi/learning/transformer/amp_network_builder_transformer_adapt.py#L153-L160)의 `has_extra` / `new_extra`
  (backbone·composer·기존 tokenizer는 `requires_grad_(False)`, extra obs tokenizer만 학습)

### 주의할 점
- **pre-grasp 구간**: 잡기 전에는 target 조정이 잘 안 먹는다. obs 문제가 아니라 (tar_pos는 항상 들어감)
  학습된 behavior가 "잡기 전엔 box로 간다"이기 때문. reward가 grasp 이후 지배적
  ([`:1004-1008`](TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L1004-L1008)).
  → steering token만 추가해선 안 고쳐진다. **steering tracking reward를 approach 구간에도 걸어야 함.**
  목표는 approach → grasp → carry → placement **전 구간** steering.

### 평가 설계
global GT trajectory를 하나 만들고, 각 step마다 **현재 위치 주변 일정 구간은 GT와 동일 / 그 이후 구간은 조금씩 변형**한
traj를 입력한다. 실행 중 traj가 바뀌는 상황에서 policy가 새 steering command에 제대로 반응하는지 확인하는 용도.

---

## TokenHSI-ma

Multi-agent base 실험용. 2-agent가 공유 scene에서 동시에 동작하는 환경을 만들고,
teammate-aware collision / interference avoidance를 붙이는 쪽.

V2 draft 기준 이쪽 담당 범위:
- shared state `S_t` (agent A/B + object + scene) 구성
- per-agent role / skill / skill goal / dependency 세팅
- 예제 task: **Corridor-Clearing** — A가 obstacle box를 치워야 B가 통로를 지나감
  (핵심 제약: `T_clear(A) < T_enter_corridor(B)`)
- Action policy에 teammate context conditioning 추가

`-steer`에서 만든 steerable Carry가 최종적으로 여기에 들어간다.

---

## 열려 있는 것

- WM이 같은 `S_t`에서 의미 있는 K개 future traj를 어떻게 생성할지
- feasibility constraint를 generation에 넣을지 Planner filtering으로 뺄지
- WM/Planner replan 주기 : steering 갱신 주기 : action policy 제어 주기 비율
- multi-agent fine-tuning 이후에도 traj→steering→path following correspondence가 유지되는지
- Skill Goal이 region일 때 최종 placement point를 언제 commit할지
- Carry 외 skill의 steering interface (skill마다 다를 수 있음)
