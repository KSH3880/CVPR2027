# TokenHSI stage1 동작 구조

`ckpt_stage1.pth`로 TokenHSI를 돌릴 때 무슨 일이 벌어지는지. 실행 진입점은
[stage1_test.sh](../TokenHSI-steer/tokenhsi/scripts/tokenhsi/stage1_test.sh).

모든 수치는 `TokenHSI-steer` 코드와 `output/tokenhsi/ckpt_stage1.pth`에서 직접 확인한 값이다.

- [1. 전체 구조](#1-전체-구조)
- [2. 실행 순서와 리셋](#2-실행-순서와-리셋)
- [3. 입력 354차원](#3-입력-354차원)
- [4. Transformer 입출력 상세](#4-transformer-입출력-상세)
- [5. AMP는 어디서 도는가](#5-amp는-어디서-도는가)
- [6. TensorBoard 지표](#6-tensorboard-지표) → [별도 문서](TOKENHSI_TENSORBOARD.md)
- [7. steer 작업 관점](#7-steer-작업-관점)

---

# 1. 전체 구조

`ckpt_stage1.pth`는 **하나의 정책 네트워크**가 traj/sit/carry/climb 4개 과제를 전부 수행하도록
학습된 결과물이다. env가 매 스텝 354차원 벡터 하나를 만들어 주면, 네트워크가 32차원 관절 목표각을
뱉고, 그걸 PD 컨트롤러가 토크로 바꿔 물리엔진을 굴린다. 초당 30번 반복.

```
                        Isaac Gym (num_envs, 30 Hz control / 60 Hz sim)
                        humanoid(phys_humanoid_v3, 32 DoF) + traj·chair·box·climb-obj
                                        │
                     ┌──────────────────┴───────────────────┐
                     │  obs_buf  (B, 354)                   │
                     │                                      │
   self obs 223 ─────┤  humanoid proprioception             │
   task obs 127 ─────┤  traj 20 | sit 38 | carry 42 |climb 27│ ← 비활성 task 구간은 0으로 마스킹
   onehot     4 ─────┤  task_mask                           │
                     └──────────────────┬───────────────────┘
                                        │ RunningMeanStd (onehot 4dim은 다시 원복)
                                        ▼
  ┌──────────────────────── actor (Transformer) ────────────────────────┐
  │  self_encoder      MLP 223→256→128→64 ┐                             │
  │  task_encoder[0]   MLP  20→256→128→64 │                             │
  │  task_encoder[1]   MLP  38→…→64       ├─ 토큰 6개 (B, 6, 64)         │
  │  task_encoder[2]   MLP  42→…→64       │  [weight, self, traj,       │
  │  task_encoder[3]   MLP  27→…→64       ┘   sit, carry, climb]        │
  │                                                                     │
  │  src_key_padding_mask: weight·self + "지금 활성 task 1개"만 살림      │
  │                        ↓                                            │
  │  TransformerEncoder  4 layers × 2 heads, d=64, ffn=512, pos_embed 없음│
  │                        ↓                                            │
  │  x[:,0] (weight token) → Composer MLP 64→1024→512→32 (identity)      │
  └────────────────────────────┬────────────────────────────────────────┘
                               │  mu (B,32),  sigma = const exp(-2.9)
                               ▼
                        action a ∈ [-1,1]^32
                               │  pd_tar = offset + scale·a
                               ▼
                        DoF position target → PhysX 2 substep
                               │
        ┌──────────────────────┼─────────────────────────┐
        ▼                      ▼                         ▼
   새 obs_buf            reward (task별 분기)        reset / terminate
                         + AMP disc reward           (fall, IET, fail_dist)

  ── 학습 시에만 쓰이는 곁가지 ──
   critic : obs 354 → MLP 2048→1024→512 → value 1   (transformer 아님, 별도 MLP)
   disc   : amp_obs 10 step × 133 = 1330 → 1024→512→1
```

## 파라미터 분포

| 블록 | 파라미터 | 비중 |
|---|---|---|
| self_encoder | 98,496 | 8.0% |
| task_encoder ×4 | 198,144 | 16.0% |
| **transformer (4층)** | **332,032** | **26.8%** |
| **composer** | **607,776** | **49.1%** |
| weight_token / pos_embed / sigma | 480 | 0.04% |
| **actor 합계** | **1,236,928** | |
| critic | 3,350,529 | (test 미사용) |
| disc | 1,888,257 | (test 미사용) |

절반이 composer에, 4분의 1이 트랜스포머에 있다. 트랜스포머는 d=64로 아주 좁고 "self와 활성 task를
섞는 라우터"에 가깝다. 실제 관절 제어 매핑은 그 뒤 1024→512 MLP가 담당한다.

---

# 2. 실행 순서와 리셋

## 프로그램이 켜질 때

```
1) --cfg_env  yaml 로드   → 물리/환경/과제 설정
   --cfg_train yaml 로드  → 네트워크 구조/알고리즘 설정
2) Isaac Gym sim 생성 (dt = 1/60, substeps 2)
3) HumanoidTrajSitCarryClimb env 생성
   ├ humanoid asset(phys_humanoid_v3.xml) 로드 → 15 body, 32 DoF
   ├ env 각각에 actor를 여러 개 심는다:
   │    humanoid / 의자 / 박스 / 발판2개 / climb 물체 / (마커들)
   ├ obs 크기 계산 → 354, action 크기 계산 → 32
   └ 4개 skill별 motion dataset 로드 (dataset_loco_sit_carry_climb.yaml)
4) rl_games Runner 생성 → algo 'trans' → TransPlayer (--test니까 agent 아님)
5) 네트워크 build → AMPTransformerMultiTaskBuilder.UnifiedNetworkClass
6) restore(ckpt_stage1.pth)
   ├ model         → actor(transformer) + critic + disc 가중치
   ├ running_mean_std   → obs 354차원 정규화 통계
   ├ amp_input_mean_std → disc 입력 133차원 통계
   └ epoch/frame/optimizer (test에선 안 씀)
7) player.run() 루프 시작
```

**네트워크 구조는 yaml이 정하고, 가중치만 pth에서 온다.** yaml의 obs 구성이 하나라도 달라지면
shape mismatch로 restore가 터진다.

## 리셋 때 env 하나가 갖게 되는 상황

`_reset_envs(env_ids)`
([:1727](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L1727))
가 순서대로 호출하는 것:

### (a) `_reset_task_indicator` — 이 env는 무슨 과제인가
```
task_indicator[env] ~ Categorical([0.1, 0.3, 0.3, 0.3])   # traj/sit/carry/climb
task_mask[env] = one-hot                                  # 예: carry → [0,0,1,0]
```
그다음 그 과제 안에서 **어떤 시작 자세로 시작할지(skill)** 를 또 뽑는다. carry면
`loco_carry 0.5 / pickUp 0.1 / carryWith 0.3 / putDown 0.1`. 뽑힌 skill의 모션 데이터셋에서
모션 하나 + 그 모션 안의 랜덤 시각 t를 샘플링해 둔다.

### (b) `_reset_ref_state_init` — 휴머노이드 자세를 모션에서 복사
`stateInit: "Random"` = **RSI (Reference State Initialization)**. 위에서 뽑은 (모션, 시각)의
root pos/rot, 전 관절 각도·각속도를 그대로 시뮬레이터에 써넣는다. 캐릭터는 T-포즈로 서서 시작하는 게
아니라 **걷는 도중 / 박스를 든 도중 자세로 뚝 떨어져서 시작**한다.

### (c) `_reset_task_*` — 4개 전부 호출된다
과제가 carry여도 traj/sit/climb 리셋 함수가 다 돈다. env에는 의자도 박스도 climb 물체도 물리적으로
다 존재하기 때문(Isaac Gym은 env마다 actor 구성이 같아야 배치 텐서로 다룰 수 있다). 자기 과제가
아닌 물체는 방해되지 않게 치운다:

```python
# _reset_task_carry, carry가 아닌 env들
self._box_states[not_task_env_ids, 2] = 5.0   # 박스를 공중 5m로 띄워 격리
```

carry인 env는:
- 박스 크기를 랜덤 스케일(0.5~1.5배, 0.125 간격)로 뽑고
- 박스를 캐릭터 기준 **반경 1~10 m 랜덤 방향**에 놓고, z축 랜덤 회전
- 50% 확률로 발판 위(최대 1.2 m 높이)에 올림
- 목표 위치 `box_tar_pos`를 (±4.5, ±4.5, 0.5~1.0)에서 뽑음 → 박스를 저기까지 옮겨 그 높이에 놓아라

> yaml의 `mode: "train"`이라 `--test`여도 `testSizes` 고정 박스 9종이 아니라 랜덤 크기가 쓰인다.
> `--test`가 이 값을 덮어쓰지 않는다. 고정하려면 yaml을 고쳐야 한다.

traj면 `_traj_gen`이 속도 0.5~1.5 m/s, 가속 ≤2, 2% 확률로 90° 급회전이 섞인 경로를 생성해 둔다.
이 경로는 **시간의 함수**라 아무 시각 t를 넣으면 위치가 나온다.

### (d) `_compute_observations` + `_init_amp_obs`
초기 obs를 채우고, AMP용 과거 10프레임 버퍼를 채운다.

## 한 스텝(1/30초)

```
[obs_buf] (B, 354)
   │ ① _preproc_obs : RunningMeanStd, 단 마지막 4차원(one-hot)은 역정규화로 0/1 복원
   │ ② 네트워크 forward → mu (B,32)
   │ ③ sigma = exp(-2.9) ≈ 0.055, sample 또는 deterministic이면 mu
   │ ④ clip[-1,1] → rescale
   ▼
[action] (B, 32)
   │ ⑤ pre_physics_step: pd_tar = _pd_action_offset + _pd_action_scale * action
   │    gym.set_dof_position_target_tensor(pd_tar)
   │ ⑥ 물리 2회 (controlFrequencyInv=2, sim dt=1/60)
   │    각 sim step마다 PhysX 내장 PD가 τ = kp(pd_tar − q) − kd·q̇ 를 인가
   │ ⑦ post_physics_step: progress_buf+1 → refresh → obs → reward → reset
   ▼
다음 스텝
```

**네트워크는 완전히 memoryless다.** RNN도 없고 프레임 스택도 없다. 매 스텝 현재 상태 354개만 보고
결정한다. 과거 정보가 필요한 건 AMP disc뿐(10프레임).

## action → 실제 움직임

`_build_pd_action_offset_scale`
([humanoid.py:343](../TokenHSI-steer/tokenhsi/env/tasks/humanoid.py#L343)):

- 3-DoF 관절(구관절, exp-map 3축): 관절 한계 최댓값 ×1.2, π로 clip → 대칭 범위 `[-s, s]`
- 1-DoF 관절(힌지): 관절 한계 그대로 → `offset = 중앙값`, `scale = 반폭`

**action은 토크가 아니라 관절 목표각이다.** `a=0`이면 관절 중립 자세. 실제 힘은 PhysX 내장 PD가
만든다. 정책이 30 Hz로 목표각만 갱신하면 PD가 60 Hz로 추종한다.

32 = `_dof_offsets`의 마지막 값. 12개 관절이 3+3+3+1+3+1+3+3+3+3+3+3으로 쪼개져 총 32 DoF.

## 보상과 종료

`_compute_reward`가 task_indicator로 마스크를 잘라 과제별로 다른 함수를 적용한다:

| 과제 | 보상 |
|---|---|
| traj | `exp(-2·‖root_xy − traj(t)‖²)` — 시각 t에 있어야 할 지점과의 거리 |
| sit | 목표 접근 속도 + 방향 정렬, 속도/각속도 과다 시 페널티 |
| carry | `walk_r`(박스 접근) + `carry_r`(목표로 운반) + `handheld_r`(손이 닿음) + `putdown_r` |
| climb | 목표 접근 + 발이 물체 위에 |

공통으로 `power_reward = −0.0005·Σ|τ·q̇|`.

종료 조건:
- 낙상: 머리 외 body가 0.15 m 아래, 머리는 0.3 m
- traj: 목표에서 4 m 이상 이탈
- 시간: 600 step (= 20초)
- **IET**: 목표 0.2 m 이내에 60 step 유지 → 성공 판정하고 조기 종료

`--eval`에서는 IET를 끄고(`_enable_IET = False`) 대신 `_success_buf` / `_precision_buf`로
성공률과 최종 오차를 기록한다.

---

# 3. 입력 354차원

## self obs `obs[0:223]`

`compute_humanoid_observations_max`
([humanoid.py:647](../TokenHSI-steer/tokenhsi/env/tasks/humanoid.py#L647)).
전부 **root의 heading(요각)만 제거한 좌표계** — 캐릭터가 어느 방향을 보든 같은 자세면 같은 숫자.
위치도 root 기준 상대값이라 월드 어디에 있든 동일.

| 인덱스 | 내용 | 비고 |
|---|---|---|
| `[0:1]` | root height | `rootHeightObsPolicy: False` → **항상 0** (죽은 차원) |
| `[1:43]` | body 14개의 local 위치 (root 제외, 14×3) | root는 자기 자신이라 항상 0이므로 제거 |
| `[43:133]` | body 15개의 local 회전, tan-norm 6D (15×6) | 쿼터니언 대신 6D(불연속 없음) |
| `[133:178]` | body 15개 선속도 (15×3) | |
| `[178:223]` | body 15개 각속도 (15×3) | |

`localRootObsPolicy: False`라서 `[43:49]`(root 회전)만은 heading 제거 없이 **전역 회전**이 들어간다.
→ 정책은 몸이 기울었는지(pitch/roll)는 알지만 **월드에서 어느 방향을 보는지는 모른다.**
방향 정보는 오직 task obs를 통해서만 온다.

## task obs `obs[223:350]`

`compute_location_observations`
([:2088](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2088)).
역시 전부 root heading 좌표계.

**traj `[223:243]` (20)**
```
미래 waypoint 10개의 (x,y)
시각: t, t+0.5, t+1.0, ..., t+4.5   (t = progress_buf * dt)
```
매 스텝 `_traj_gen.calc_pos`로 새로 뽑으므로 캐릭터가 전진하면 창이 같이 미끄러진다. z는 버린다.

**sit `[243:281]` (38)**
```
[243:246] 앉을 목표 지점 (3)
[246:270] 의자 bounding point 8개 (8×3)
[270:272] 의자가 바라보는 방향 2D (2)
[272:275] 의자 root 위치 (3)
[275:281] 의자 root 회전 6D (6)
```

**carry `[281:323]` (42)** — steer에서 건드릴 부분
```
[281:284] 박스 선속도 (3)
[284:287] 박스 각속도 (3)
[287:290] 박스 위치 (3)
[290:296] 박스 회전 6D (6)
[296:320] 박스 꼭짓점 8개 (8×3)   ← 크기·모양 정보가 여기 들어있음
[320:323] 박스를 놓을 목표 위치 (3)
```

bps는 박스 로컬 좌표계 8개 꼭짓점(±sx/2, ±sy/2, ±sz/2)을 월드로 보냈다가 캐릭터 로컬로 다시
가져온 것
([_build_box_bps](../TokenHSI-steer/tokenhsi/env/tasks/comp_interaction_skills/humanoid_comp_traj_carry.py#L245)).
크기를 스칼라 3개로 주는 대신 이렇게 주면 **위치·자세·크기가 한 표현에 섞여** 있어 잡는 지점을
추론하기 쉽다.

**climb `[323:350]` (27)**
```
[323:326] 올라설 목표 root 위치 (3)
[326:350] 물체 bounding point 8개 (8×3)
```

**one-hot `[350:354]` (4)** — `[traj, sit, carry, climb]`

## 마스킹이 두 번 걸린다

1. **env 쪽**: `enableApplyMaskOnTaskObs: True` → carry env면 `[223:281]`과 `[323:350]`이 전부
   **0으로 곱해짐**
   ([:2171](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2171))
2. **네트워크 쪽**: 그 구간을 인코딩한 토큰 자체를 attention에서 **제외**

1만 하면 "0을 인코딩한 토큰"이 여전히 attention에 참여해 노이즈가 된다. 2까지 해야 진짜로 없는 것과
같아진다. `plan/PLAN.md`의 "마스킹 아웃 vs obs만 zero" 두 검증이 갈리는 지점이 정확히 이것.

## 정규화

`normalize_input: True`. RunningMeanStd(354)를 통과시킨 뒤 **마지막 4차원(onehot)만 역정규화해서
0/1로 되돌린다**
([trans_players.py:186-210](../TokenHSI-steer/tokenhsi/learning/transformer/trans_players.py#L186-L210)).
네트워크가 이 4차원의 argmax로 padding mask를 만들기 때문. 정규화된 값으로도 argmax가 보통
보존되지만 보장이 없어서 명시적으로 복원한다.

---

# 4. Transformer 입출력 상세

배치 `B=16` (test 기준) 예시.

## 전체 지도 (shape만)

```
obs (16, 354)
  ├─ [:, :223]      ──► self_encoder      ──► (16, 64)  ──unsqueeze──► (16, 1, 64)
  ├─ [:, 223:243]   ──► task_encoder[0]   ──► (16, 64)  ┐
  ├─ [:, 243:281]   ──► task_encoder[1]   ──► (16, 64)  │ stack(dim=1)
  ├─ [:, 281:323]   ──► task_encoder[2]   ──► (16, 64)  ├──► (16, 4, 64)
  ├─ [:, 323:350]   ──► task_encoder[3]   ──► (16, 64)  ┘
  └─ [:, 350:354]   ──► (mask 만드는 데만 사용, 인코딩 안 함)

weight_token (1,1,64) ──expand──► (16, 1, 64)

concat(dim=1):  (16,1,64) + (16,1,64) + (16,4,64)  ──►  x (16, 6, 64)
                 weight      self        4 tasks

src_key_padding_mask (16, 6)  bool

TransformerEncoder(x, mask)  ──►  (16, 6, 64)      # shape 불변
                                     │
                                  x[:, 0]  ──►  (16, 64)
                                     │
                              Composer MLP  ──►  (16, 32)  = mu
```

핵심: **트랜스포머는 shape을 바꾸지 않는다.** `(B, 6, 64)` 들어가서 `(B, 6, 64)` 나온다.
하는 일은 6개 64차원 벡터를 서로 섞는 것뿐.

## Tokenizer — 가변 차원을 전부 64로

`_build_mlp(units=[256,128] + [64])`가 만드는 실제 모듈
(체크포인트 키 index 0/2/4 = Linear, 1/3/5 = ReLU):

```
Linear(in, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 64) → ReLU
```

**마지막에도 ReLU가 붙는다.** 즉 모든 토큰은 **non-negative 64차원 벡터**로 들어간다.
LayerNorm/BatchNorm은 없다 (`normalization: None`).

| 모듈 | in | hidden | out | 파라미터 |
|---|---|---|---|---|
| `self_encoder` | 223 | 256→128 | 64 | 98,496 |
| `task_encoder[0]` traj | 20 | 256→128 | 64 | 46,528 |
| `task_encoder[1]` sit | 38 | 256→128 | 64 | 51,136 |
| `task_encoder[2]` carry | 42 | 256→128 | 64 | 52,160 |
| `task_encoder[3]` climb | 27 | 256→128 | 64 | 48,320 |

4개 encoder는 서로 **완전히 독립된 가중치**다. 입력 차원이 20/38/42/27로 달라 공유할 수도 없다.
공유되는 건 그 뒤 트랜스포머부터.

```python
# amp_network_builder_transformer.py:135
task_token = torch.stack([
    self.task_encoder[i](task_obs_real[:, idx[i]:idx[i+1]])
    for i in range(4)
], dim=1)     # (16, 4, 64)
```

**4개 전부 forward된다.** carry env여도 traj/sit/climb encoder가 다 돈다. 입력이 이미 0으로
마스킹돼 있고 출력 토큰도 attention에서 제외되므로 계산 낭비지만, 배치 처리를 위해 그냥 다 돌린다.

## weight_token

```python
self.weight_token = nn.Parameter(torch.zeros(1, 1, 64))   # 학습되는 파라미터, 64개
weight_token = self.weight_token.expand(B, -1, -1)         # (16, 1, 64)
```

**입력에 의존하지 않는 상수 벡터.** 모든 env, 모든 스텝에서 동일. ViT의 `[CLS]`와 같은 역할 —
"여기 결과를 모아라"는 빈 슬롯. 트랜스포머를 통과하며 self/task 토큰의 정보를 attention으로
빨아들이고, 최종적으로 이 자리만 읽어서 action을 만든다.

## 토큰 배열과 pos_embed

```
x = cat([weight, self, task_token], dim=1)   # (16, 6, 64)

index:   0        1       2       3      4       5
       weight    self    traj    sit   carry   climb
```

```python
if self.use_pos_embed:            # yaml: False → 실행 안 됨
    x = x + self.pos_embed        # (1, 6, 64)
```

`pos_embed` 파라미터 384개는 체크포인트에 있지만 **사용되지 않는다.** 각 슬롯이 전용 encoder를
갖고 있어 정체성이 이미 구분되기 때문.

## src_key_padding_mask — (16, 6) bool

PyTorch 규약: **True = 무시(가림)**, False = 사용.

```python
mask = torch.ones((16, 6), dtype=bool)     # 전부 True = 전부 가림
mask[:, [0, 1]] = False                    # weight, self 는 항상 열기

idx = one_hot.argmax(-1) + 2               # (16,)  0~3 → 2~5
mask[one_hot(idx, num_classes=6).bool()] = False
```

carry env(one-hot `[0,0,1,0]` → argmax 2 → +2 = 4)의 마스크 행:

```
        idx:   0      1      2      3      4      5
             weight  self   traj   sit  carry  climb
mask  :      False  False  True   True  False   True
의미  :       사용   사용   무시   무시   사용    무시
```

env마다 행이 다르다. **언제나 열린 토큰은 정확히 3개.**

## TransformerEncoderLayer 한 층의 내부 shape

`d_model=64, nhead=2, dim_feedforward=512, dropout=0, activation=relu, batch_first=True,
norm_first=False(기본)` → **post-LN** 구조.

```
입력 x (16, 6, 64)

┌─ Multi-Head Self-Attention ────────────────────────────────────┐
│  in_proj_weight (192, 64)   192 = 3 × 64  (Q,K,V 한 번에)       │
│  x @ Wᵀ + b  →  (16, 6, 192)  →  split 3  →  Q,K,V 각 (16,6,64)│
│                                                                │
│  head 분할: nhead=2, head_dim = 64/2 = 32                       │
│      Q,K,V →  (16, 2, 6, 32)                                   │
│                                                                │
│  scores = Q @ Kᵀ / √32     →  (16, 2, 6, 6)                    │
│                                ↑     ↑  ↑                      │
│                              batch head (query, key)           │
│                                                                │
│  마스크 적용: key_padding_mask (16,6) → (16,1,1,6) 브로드캐스트   │
│      True인 key 열에 -inf 대입                                  │
│      carry env: key 2,3,5 열이 전부 -inf                        │
│                                                                │
│  softmax(dim=-1)          →  (16, 2, 6, 6)                     │
│      각 query 행은 key 0,1,4 세 곳에만 확률 분배 (합 1)          │
│                                                                │
│  attn @ V                 →  (16, 2, 6, 32)                    │
│  head 병합                 →  (16, 6, 64)                       │
│  out_proj (64, 64)        →  (16, 6, 64)                       │
└────────────────────────────────────────────────────────────────┘
                          │
x = LayerNorm(x + attn_out)     norm1 (64,)     →  (16, 6, 64)
                          │
┌─ Feed-Forward ────────────────────────────────────────────────┐
│  linear1 (512, 64)  →  (16, 6, 512)                            │
│  ReLU               →  (16, 6, 512)                            │
│  linear2 (64, 512)  →  (16, 6, 64)                             │
└───────────────────────────────────────────────────────────────┘
                          │
x = LayerNorm(x + ff_out)       norm2 (64,)     →  (16, 6, 64)

출력 (16, 6, 64)
```

**한 층 파라미터 83,008개** = in_proj 12,480 + out_proj 4,160 + linear1 33,280 +
linear2 32,832 + norm1/norm2 256. 이게 4층(총 332,032). 층 간 마스크는 **매 층 동일하게 재적용**된다.
`nn.TransformerEncoder`의 최종 `norm`은 `None`이라 마지막 추가 LayerNorm은 없다.

### 마스킹 주의점 2가지

**(1) 가려지는 건 key/value 축이지 query 축이 아니다.**
`scores`가 `(16, 2, 6, 6)`인데 마스크는 마지막 축(key)에만 걸린다. traj 토큰(index 2)도 **query로서는
여전히 계산되어 출력값을 갖는다** — 다만 index 0만 읽으므로 무의미하다. `x[:, 2]`가 0이 아니라고
놀랄 필요 없음.

**(2) 정보는 흘러들어가지 않는다.**
key로 완전히 차단되므로 index 0(weight)이 계산될 때 traj/sit/climb 토큰 값은 어떤 층에서도 반영되지
않는다. 그래서 `src_key_padding_mask`로 끄면 **비트 단위로 동일한 결과**가 보장된다. 반면 obs만
0으로 만들면 토큰이 여전히 key로 참여해서 값이 달라진다.

## 출력 추출 → action

```python
weights = self.composer(x[:, 0])
```

```
x            (16, 6, 64)
x[:, 0]      (16, 64)        ← weight token 자리만
```

Composer MLP (`extra_mlp_units: [1024, 512]`, `output_size = 32`, `activation = identity`):

```
Linear(64, 1024)   →  (16, 1024)     66,560
ReLU
Linear(1024, 512)  →  (16,  512)    524,800
ReLU
Linear(512, 32)    →  (16,   32)     16,416     ← 마지막 층엔 ReLU 없음
Identity()         →  (16,   32)
                                    ─────────
                                    607,776
```

이름이 `Composer`이고 다른 stage에서는 `activation="sigmoid"`로 skill blending weight를 만들지만,
**stage1에서는 identity라 그냥 action을 직접 뱉는 MLP**다.

```python
mu    = weights                 # (16, 32)
sigma = sigma_act(self.sigma)   # (32,) 상수, exp(-2.9) ≈ 0.055
```

## 왜 이 구조인가

- **task마다 별도 encoder**: traj는 20차원, carry는 42차원으로 의미도 스케일도 다르다. 분리하면 각
  과제 정보가 **같은 64차원 공간**으로 사영되고, 트랜스포머는 "지금 켜진 게 뭐든" 동일하게 처리한다.
- **확장성**: 새 과제 추가 시 `task_encoder`에 MLP 하나 + 토큰 하나만 늘리면 되고,
  **기존 encoder/transformer/composer 가중치는 그대로 재사용**된다. stage2들이 전부 이 구조 위에서
  새 토큰만 붙여 fine-tune하는 이유.
- **`use_pos_embed: False`**: 각 슬롯이 전용 encoder를 가져 정체성이 이미 구분됨.
- **dropout 0**: 코드에 `# using dropout will make training failed [BUG]` 주석이 달려 있다.

---

# 5. AMP는 어디서 도는가

AMP는 **정책(354 → 32)과 완전히 분리된 두 번째 파이프라인**이다. 정책 obs를 전혀 공유하지 않고,
자기만의 관측(`amp_obs`, 1330차원)과 자기만의 네트워크(disc)를 갖는다.

AMP는 **보상 함수의 절반**이다. "목표를 달성했나"(task reward)는 사람이 손으로 짠 식이 계산하고,
**"사람처럼 움직였나"(style reward)는 판별기가 계산한다.**

```
최종 보상 = 0.5 × task_reward + 0.5 × disc_reward
```

## 두 파이프라인

```
                        시뮬레이터 상태 (root, dof, rigid bodies, 물체들)
                                    │
              ┌─────────────────────┴──────────────────────┐
              │                                            │
    ┌─────────▼──────────┐                     ┌───────────▼────────────┐
    │  정책 관측         │                     │  AMP 관측              │
    │  obs_buf (B, 354)  │                     │  amp_obs_buf(B,10,133) │
    │                    │                     │                        │
    │  self 223          │                     │  물체/목표 정보 없음!  │
    │  + task 127        │                     │  순수 "몸의 움직임"만  │
    │  + onehot 4        │                     │  최근 10프레임         │
    └─────────┬──────────┘                     └───────────┬────────────┘
              │                                            │
              ▼                                            ▼
      actor (transformer)                          disc MLP (1330→1024→512→1)
      → action (B, 32)                             → logit (B, 1)
              │                                            │
              │                                     style reward
              ▼                                            ▼
         물리 시뮬레이션 ────► task reward ────► 0.5·task + 0.5·style
```

**결정적 차이**: `amp_obs`에는 박스도, 의자도, 목표 위치도 없다. 오직 관절 각도·속도·발끝 위치 같은
**몸의 상태**만 있다. 판별기가 "과제 성공 여부"가 아니라 "동작의 자연스러움"만 보게 하려는 설계.

## amp_obs 1330차원

### 한 프레임 = 133차원

`build_amp_observations`
([:2052](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2052))의
`cat` 순서 그대로:

| 인덱스 | 내용 | 차원 |
|---|---|---|
| `[0:1]` | root height | 1 (`rootHeightObs: False` → **항상 0**) |
| `[1:7]` | root 회전 tan-norm 6D | 6 (`localRootObs: True` → **heading 제거됨**) |
| `[7:10]` | root 선속도 (local) | 3 |
| `[10:13]` | root 각속도 (local) | 3 |
| `[13:85]` | dof_obs — 관절각을 회전 표현으로 변환 | 72 |
| `[85:117]` | dof_vel — 관절 각속도 원본 | 32 |
| `[117:129]` | key body 4개 위치 (양손·양발, root 상대) | 12 |
| `[129:133]` | **task one-hot** | 4 |

`129 = 13 + 72 + 32 + 12`. yaml 주석의 `28 + 2*2`가 dof_vel 32를 뜻한다(v3는 28+4 DoF).

### 정책 obs와 미묘하게 다른 점

```yaml
localRootObs: True         # AMP disc용  → root 회전에서 heading 제거
localRootObsPolicy: False  # 정책용      → root 전역 회전 유지
```

판별기는 **캐릭터가 어느 방향을 보는지 몰라야** 한다. 북쪽으로 걷든 남쪽으로 걷든 "걷기 스타일"은
같아야 하니까. 반대로 정책은 몸이 기울었는지 알아야 균형을 잡는다.

또 정책은 `local_body_pos/rot/vel`로 15개 body 전부를 보지만, AMP는 `dof_pos/dof_vel` + key body
4개만 본다. 표현 자체가 다르다.

### 왜 10프레임인가 (`numAMPObsSteps: 10`)

```
amp_obs_buf (B, 10, 133)
   [:, 0]     ← 현재 프레임        (_curr_amp_obs_buf)
   [:, 1:10]  ← 과거 9프레임       (_hist_amp_obs_buf)

flatten → (B, 1330)  ← disc 입력
```

한 프레임만 보면 "포즈가 그럴듯한가"만 판별한다. 10프레임(= 1/3초)을 보면 **속도 프로파일, 발 접지
타이밍, 스텝 리듬** 같은 시간적 스타일까지 판별할 수 있다. 원조 AMP 논문은 2프레임을 썼는데
TokenHSI는 10으로 늘렸다.

## env가 매 스텝 하는 일 (--test에서도 돈다)

```python
# humanoid_traj_sit_carry_climb.py:1185
def post_physics_step(self):
    super().post_physics_step()        # obs_buf, reward, reset 갱신

    self._update_hist_amp_obs()        # ① 버퍼 한 칸씩 밀기
    self._compute_amp_observations()   # ② 현재 프레임 채우기

    amp_obs_flat = self._amp_obs_buf.view(-1, 1330)
    self.extras["amp_obs"] = amp_obs_flat   # ③ agent에게 전달
```

**① `_update_hist_amp_obs`** — 링버퍼가 아니라 통째로 시프트한다:
```python
for i in reversed(range(9)):
    self._amp_obs_buf[:, i+1] = self._amp_obs_buf[:, i]
```
`[8]→[9]`, `[7]→[8]`, ... `[0]→[1]`. 그다음 `[0]`을 새 프레임으로 덮어쓴다.

> 이건 `post_physics_step` 안에 있으니 `--test`에서도 **매 스텝 계산된다.** 다만 player는 이 값을
> 보상에 쓰지 않고, viewer가 켜져 있을 때 `_amp_debug`로 disc 점수를 찍어보는 데만 쓴다.

### 리셋할 때 과거 채우기

막 리셋된 env는 과거 9프레임이 없다. 두 가지로 채운다
([`_init_amp_obs`](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L1854)):

- **`_init_amp_obs_default`**: 현재 프레임을 9번 복제 (정지 상태로 시작한 env)
- **`_init_amp_obs_ref`**: RSI로 시작한 env는 **참조 모션에서 시각 t-dt, t-2dt, … t-9dt의 프레임을
  실제로 뽑아** 채운다

후자가 중요하다. 그냥 복제하면 "속도 0으로 얼어붙은 궤적"이 되어 판별기가 즉시 가짜라고 판정한다.

## 학습 루프에서의 위치

```
┌─ play_steps() : horizon_length=32 스텝 롤아웃 ───────────────┐
│  for n in range(32):                                         │
│      action = policy(obs)                                    │
│      obs, r, done, infos = env.step(action)                  │
│      experience_buffer['rewards'][n] = r          ← task만   │
│      experience_buffer['amp_obs'][n]  = infos['amp_obs']     │
│                                                              │
│  롤아웃 끝난 뒤 한 번에:                                       │
│      disc_r     = _calc_disc_rewards(mb_amp_obs)             │
│      mb_rewards = 0.5*task_r + 0.5*disc_r   ← 여기서 합침    │
│      GAE로 advantage 계산 (gamma 0.99, tau 0.95)              │
└──────────────────────────────────────────────────────────────┘
                            │
┌─ disc 학습용 데이터 3종 ──────────────────────────────────────┐
│  amp_obs        = 방금 롤아웃한 것          (가짜, 최신)      │
│  amp_obs_replay = replay buffer에서 샘플    (가짜, 과거)      │
│  amp_obs_demo   = demo buffer에서 샘플      (진짜, 모션캡처)  │
└──────────────────────────────────────────────────────────────┘
                            │
┌─ calc_gradients() : mini_epochs=6 회 ────────────────────────┐
│  PPO loss (actor+critic)  +  5 × disc_loss                   │
│  한 optimizer로 동시에 업데이트 (lr 2e-5)                     │
└──────────────────────────────────────────────────────────────┘
```

### disc reward
```python
# amp_agent.py:645
disc_logits = eval_disc(amp_obs)
prob = sigmoid(disc_logits)                   # "진짜일 확률"
disc_r = -log(max(1 - prob, 1e-4)) * 2        # disc_reward_scale = 2
```
- 상한: `-log(1e-4) × 2 ≈ 18.4` (판별기를 완전히 속임)
- 하한: `0` (완전히 들킴)
- 판별기가 반반이면: `-log(0.5)×2 ≈ 1.39`

`torch.no_grad()` 안에서 계산된다 — 보상일 뿐 gradient가 흐르지 않는다.

### disc loss
```python
# amp_agent.py:513
disc_loss = 0.5 * (BCE(agent_logit, 0) + BCE(demo_logit, 1))
          + 0.01 * ‖disc_logits.weight‖²            # logit_reg
          + 5.0  * mean(‖∂demo_logit/∂demo_obs‖²)   # grad_penalty
          + 1e-4 * ‖모든 disc weight‖²              # weight_decay
```
**grad penalty(계수 5)가 특히 크다** — 진짜 데이터 지점에서 판별기 기울기를 억눌러 판별기가 너무
빨리 이겨버리는 걸 막는다. GAN 학습 안정화의 핵심 장치.

### 3개 버퍼

| 버퍼 | 크기 | 내용 |
|---|---|---|
| `_amp_obs_demo_buffer` | 200,000 | 모션캡처에서 뽑은 진짜 궤적 |
| `_amp_replay_buffer` | 200,000 | 과거 정책이 만든 궤적 (`keep_prob 0.01`로 희소 저장) |
| `experience_buffer['amp_obs']` | 32×4096 | 이번 롤아웃 |

replay를 섞는 이유: 판별기가 최신 정책만 상대하면 **과거에 이미 걸러낸 나쁜 동작을 잊어버리고**,
정책이 그 동작으로 되돌아가는 순환이 생긴다.

## demo 데이터 출처

`fetch_amp_obs_demo(num_samples=512)`
([:1280](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L1280)):

```
① skillDiscProb 로 skill 하나 뽑기
     loco 0.1 / loco_sit 0.1 / sit 0.2 / loco_carry 0.1 / omomo 0.1
     / pickUp 0.05 / carryWith 0.0 / putDown 0.05
     / loco_climb 0.1 / climb 0.1 / climbNoRSI 0.1
② 그 skill의 motion_lib에서 모션 + 시각 t 랜덤 샘플
③ t, t-dt, ..., t-9dt 의 10프레임을 뽑아 build_amp_observations
④ 그 skill이 속한 task의 one-hot 4차원을 붙임
     예: loco → traj,  sit/loco_sit → sit,  pickUp/putDown/omomo → carry
⑤ (512, 1330) 반환
```

`skillDiscProb`은 **정책 리셋용 `skillInitProb`과 완전히 다른 것**이다. 전자는 "판별기에게 어떤
진짜 데이터를 보여줄까", 후자는 "env를 어떤 자세로 시작시킬까".

`carryWith: 0.0` — 박스를 들고 걷는 모션은 disc 학습 데이터에서 제외됐고, `omomo 0.1`이 그 역할을 한다.

## task one-hot 4차원이 붙은 이유

`enableTaskSpecificDisc: True` → 133의 마지막 4차원.

없다면 판별기는 "앉기 + 나르기 + 오르기 데이터"를 **하나의 뭉뚱그린 분포**로 학습한다. 그러면 앉는
중인 정책이 "걷기처럼 보이게" 움직여도 통과해버린다. one-hot을 넣으면 판별기가 조건부 분포
`p(motion | task)`를 학습하고, **carry env는 carry 스타일로만 평가**된다.

정책 쪽 `_compute_amp_observations`는 `self._task_mask`를, demo 쪽 `fetch_amp_obs_demo`는
skill→task 매핑으로 만든 one-hot을 붙인다. 둘이 같은 규칙이어야 판별기가 제대로 짝지어 비교한다.

## 정규화

정책 obs와 **별개의** RunningMeanStd를 쓴다:
```python
# amp_agent.py:60
self._amp_input_mean_std = RunningMeanStd((133,))   # 프레임 단위!
```
1330이 아니라 **133 단위**로 통계를 잡고, `_preproc_amp_obs`가 `(N,1330)` → `(N,10,133)`로 보고
정규화한 뒤 되편다. 10프레임이 같은 통계를 공유하는 게 맞으니까.

## 파일별 역할

| 파일 | AMP 관련 역할 |
|---|---|
| [humanoid_traj_sit_carry_climb.py](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py) | `amp_obs` 생성·버퍼 관리, demo 샘플링, one-hot 부착 |
| [amp_network_builder.py](../TokenHSI-steer/tokenhsi/learning/amp_network_builder.py) | `_disc_mlp` (1330→1024→512) + `_disc_logits` (→1) |
| [amp_agent.py](../TokenHSI-steer/tokenhsi/learning/amp_agent.py) | disc reward, 보상 결합, disc loss, replay/demo 버퍼 |
| [amp_players.py](../TokenHSI-steer/tokenhsi/learning/amp_players.py) | test 시 `_amp_debug`로 disc 점수만 출력 (보상 미사용) |
| [trans_agent.py](../TokenHSI-steer/tokenhsi/learning/transformer/trans_agent.py) | `amp_agent` 상속, transformer용 obs 전처리 조정 |

---

# 6. TensorBoard 지표

학습 중 기록되는 스칼라 태그 전부(`rewards0`, `reward_<task>`, `losses/*`, `info/disc_*`,
`performance/*`)와 각각을 어떻게 읽는지는 별도 문서로 뺐다.

→ **[TOKENHSI_TENSORBOARD.md](TOKENHSI_TENSORBOARD.md)**

---

# 7. steer 작업 관점

## 토큰을 하나 추가하면 shape이 어떻게 변하나

```
                            현재            토큰 추가 후
task_encoder 개수            4               5  (steer용 MLP 하나 추가)
num_tokens                  6               7
x                     (B, 6, 64)      (B, 7, 64)
pos_embed             (1, 6, 64)      (1, 7, 64)   ← shape mismatch! (미사용이지만 load 시 걸림)
src_key_padding_mask  (B, 6)          (B, 7)
attention scores      (B,2,6,6)       (B,2,7,7)
x[:, 0]               (B, 64)         (B, 64)      ← 변화 없음
composer 입출력        64 → 32         64 → 32      ← 변화 없음
```

**바뀌지 않는 것**: `self_encoder`, 기존 `task_encoder[0..3]`, 트랜스포머 전 층, `composer`.
트랜스포머는 토큰 개수에 대해 shape-agnostic이라 `num_layers`, `d_model`이 그대로면 가중치가 전부
재사용된다. 이게 TokenHSI가 stage2에서 새 능력을 붙일 때 stage1 가중치를 통째로 물려받는 이유.

**걸리는 것**: `pos_embed`가 `(1,6,64)` → `(1,7,64)`로 커져 `load_state_dict`가 strict 모드에서
실패한다. 실제로는 `use_pos_embed: False`라 안 쓰이므로 새로 만들거나 앞 6개만 복사하면 된다.
새 `task_encoder[4]`도 체크포인트에 없으니 랜덤 초기화 대상.

## "껐을 때 동일" 검증

`plan/PLAN.md`의 두 가지 방법이 갈리는 이유가 [3절의 이중 마스킹](#마스킹이-두-번-걸린다) 때문이다.

- `src_key_padding_mask[:, 6] = True` → attention scores의 key 축 7번째 열이 전부 -inf가 되어
  6-토큰일 때의 softmax와 **수치적으로 동일한 분포**. `x[:, 0]`도 그대로. → action이 bit-exact하게
  일치해야 한다. 안 맞으면 배선이 틀린 것.
- obs만 0 → 토큰은 살아있고 값만 0 → **거의 동일해야 함.** 크게 달라지면 attention이 새 토큰의 존재
  자체에 민감하다는 신호.

## AMP 쪽

steering 입력을 추가해도 **AMP는 원칙적으로 손댈 게 없다.** `amp_obs`는 정책 obs와 독립이고
물체·목표 정보를 안 담으니까. 다만:

1. **`_num_tasks`가 늘어나면** one-hot이 4→5가 되어 `_num_amp_obs_per_step`이 133→134,
   disc 입력이 1330→1340이 된다. → `_disc_mlp.0.weight`가 `(1024,1330)`→`(1024,1340)`이라
   **체크포인트 로드가 깨진다.** 새 task를 등록하지 않고 carry의 task obs만 42→42+K로 늘리면
   이 문제는 안 생긴다.
2. **frozen 단계에서 disc는 학습 안 하지만 보상 계산에는 쓰인다.** steering으로 경로가
   부자연스러워지면 style reward가 떨어져 학습이 그쪽으로 끌려간다. "경로는 맞는데 동작이 어색"한
   실패 모드가 나오면 `disc_reward_w 0.5`를 의심해볼 지점.

## 텐서보드 추가 주의

frozen 단계에서는 backbone이 얼어 있으니 `losses/a_loss`·`c_loss`가 거의 안 움직여야 정상이다.
움직이면 얼리기가 제대로 안 된 것.

**`plan/PLAN.md`의 지표(R², lateral error, 성공률)는 텐서보드에 안 찍힌다.** `--eval` 경로의
`_compute_metrics_evaluation_*`가 별도로 계산해 파일로 떨군다. 학습 중에 보려면 로깅을 직접
추가해야 한다.
