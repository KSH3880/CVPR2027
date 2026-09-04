# MAStop carry A/B/C/D 실험 기록

마지막 갱신: 2026-08-26

## 목표

두 에이전트가 각자 박스를 운반하다 충돌이 예상되면 정확히 한 명만 박스를 든 채
정지하고, 상대가 통과한 뒤 운반을 재개하여 두 박스를 모두 배달한다.

이 문서는 `HumanoidMAStopCarry`의 A/B/C/D curriculum 설계, 실행 설정, 평가 결과와
판정을 계속 갱신하는 실험 원장이다. 자동 결과 원장이 아니므로 수치는 반드시 원시
로그와 함께 기록한다.

## 모델 계보와 용어

```text
TokenHSI Stage 1 prior
  -> MASteer ms12_base4L_s0_ep3000
     -> A: 추가학습 없는 multi-agent carry 대조군
     -> A+B: solo stop 학습
     -> A+B+C: fixed-oracle yield 학습
     -> A+B+C+D: dynamic-ETA yield 학습
```

- Stage 1은 `traj/sit/carry/climb`을 학습하며 carry 목표 속도는 1.5 m/s로 고정이다.
- Stage 1 motion에는 사람의 `walk_to_stand`가 있지만, 박스를 든 채 경로 중간에서
  명령에 따라 `stop -> hold -> resume`하는 학습은 없다.
- MASteer `ms12`의 랜덤 속도창은 배율 `{1.0, 0.75, 0.5, 0.25}`를 사용한다. 최저
  명령 속도는 약 0.375 m/s이며, 목적지 끝 이외의 운반 중 `M=0` 정지는 학습하지 않았다.
- 따라서 A는 raw Stage 1이 아니라, Stage 1 prior와 MASteer adapt policy를 사용하는
  **2-agent, 2-box 운반 기준선**이다.

사용 checkpoint:

- Stage 1 prior: `TokenHSI/output/tokenhsi/ckpt_stage1.pth`
- A/MASteer 시작점:
  `TokenHSI-masteer/output/masteer/ms12_base4L_s0_ep3000/Humanoid_snap/nn/Humanoid.pth`

## 시나리오 정의

모든 시나리오는 에이전트 2명과 에이전트별 박스 1개, 총 박스 2개를 사용한다. 두
에이전트는 같은 정책 파라미터를 공유한다.

| 시나리오 | 목적 | 정지 조건과 배치 | 기본 학습 mix |
|---|---|---|---:|
| A `free` | 기존 pickup/carry 보존 | 정지 명령 없음. 두 에이전트가 각자 정상 운반 | `1,0,0,0` |
| B `solo` | 운반 중 stop/hold/resume | 두 경로를 약 9 m 분리. 무작위 한 명이 strict pickup 후 1.5 m 운반하면 `M=0`, 0.75~1.5초 hold 후 resume | `.5,.5,0,0` |
| C `fixed oracle` | exactly-one yield | 직각 교차 경로. 둘 다 pickup한 뒤 고정된 한 명이 교차점 약 1.2 m 전부터 정지하고 상대가 clearance 1.0 m를 넘으면 resume | `.35,.25,.40,0` |
| D `dynamic ETA` | 충돌 예측 기반 yield | 둘 다 pickup한 뒤 root/box 속도 EMA로 ETA를 계산. horizon 3.0초 안이며 ETA 차가 1.0초 미만이면 늦게 도착할 한 명을 정지 | `.35,.20,.20,.25` |

B는 충돌 회피 단계가 아니다. 두 agent를 멀리 분리하여 `M=0`이 박스를 든 물리적
정지로 이어지는지만 먼저 학습한다. C부터 교차 충돌과 clearance를 사용한다.

## Reward 구성

모든 단계에서 MASteer의 기존 pickup/carry/steering reward를 유지한다. 정지 대상
agent의 정지 구간에만 다음 항을 추가한다.

```text
+ 0.5 * exp(-4 * root_speed^2)
+ 0.5 * exp(-4 * box_speed^2)
+ 0.5 * held
- 0.2 * (root_speed + box_speed)
- 1.0 * dropped
```

C/D에서는 conflict 시 사람과 박스 footprint가 겹치면 agent별로 `-1.0` collision
penalty를 추가한다. 별도의 stop 모션 demonstration은 없으며 PPO reward fine-tuning으로
학습한다.

## 실행 설정

| 항목 | 값 |
|---|---:|
| GPU | 6, 7 DDP |
| 총 env | 2048 (GPU당 1024) |
| agent 수 | env당 2, 총 4096 rows |
| seed | 0 |
| 추가 iteration | 단계당 3000 |
| `MS_MRAND` | 4 |
| `MS_VEL_W` | 5 |
| `MS_CLIP` | 1 |
| PPO minibatch | rank당 8192 |
| AMP minibatch | rank당 2048 |

현재 실행 스크립트 `scripts/run_ma_stop_abcd.sh`는 다음과 같은 **순차 resume** 구조다.

```text
A checkpoint -> B checkpoint -> C checkpoint -> D checkpoint
```

하지만 앞으로 비교하려는 실험은 각 누적 reward/시나리오 조합을 동일한 A checkpoint에서
독립적으로 시작해야 한다.

```text
A
A checkpoint -> A+B
A checkpoint -> A+B+C
A checkpoint -> A+B+C+D
```

따라서 C/D를 다시 실행하기 전에 runner의 시작 checkpoint를 이 독립 분기 구조로
수정하고, 각 branch가 해당 mix 전체를 처음부터 학습하도록 검증해야 한다.

## 2026-08-25~26 실행 결과

### 실행 상태

| 단계 | tag | 상태 | checkpoint/중단 이유 |
|---|---|---|---|
| A | `mastop_a_ms12_3k` | 평가 완료 | 추가학습 없음 |
| A+B | `mastop_ab_s0` | 학습·평가 완료, 기각 | `GATE_FAIL free carry regression` |
| A+B+C | `mastop_abc_s0` | 미실행 | B gate 실패로 chain 중단 |
| A+B+C+D | `mastop_abcd_s0` | 미실행 | B gate 실패로 chain 중단 |

A+B checkpoint:

`TokenHSI-mastop/output/mastop_carry/mastop_ab_s0/Humanoid_25-20-31-44/nn/Humanoid.pth`

### 하네스 full-eval success

0회차는 reset 정착 문제가 있으므로 기존 규칙대로 repeat 1·2 평균을 사용한다.

| 모델 | repeat 0 | repeat 1 | repeat 2 | stable mean |
|---|---:|---:|---:|---:|
| A | 0.7881 | 0.7109 | 0.7168 | **0.7139** |
| A+B | 0.5215 | 0.4756 | 0.4482 | **0.4619** |

A+B는 A보다 stable success가 약 **25.2%p 하락**했다.

### strict metric 요약

| 모델/시나리오 | rows | pickup | finish | finish\|pickup |
|---|---:|---:|---:|---:|
| A / A | 2274 | 0.8892 | 0.7036 | 0.7690 |
| A+B / A | 1184 | 0.8598 | 0.5971 | 0.6601 |
| A+B / B | 1134 | 0.9012 | 0.3060 | 0.3268 |

### B 정지 동작

| metric | 값 |
|---|---:|
| stop episode rate | 0.8448 |
| stop 중 held fraction | 0.9528 |
| stop 중 root speed | 0.584 m/s |
| stop 중 box speed | 0.587 m/s |
| release 후 resume | 0.9582 |
| collision episode | 0.0265 |
| median clearance | 6.588 m |

B 학습 전 smoke에서 `M=0` 구간 root/box 속도는 약 0.90 m/s였다. 학습 후 약
0.58 m/s로 감소하여 정지 신호와 hold/resume는 일부 학습했지만, 실제 정지라기보다
gate 상한 0.6 m/s를 간신히 만족한 수준이다.

## 현재 판정

1. **pickup 자체가 붕괴한 것은 아니다.** B 시나리오 pickup은 0.9012다.
2. **운반 완료가 붕괴했다.** B finish는 0.3060이며, 정지 없는 A subset도
   0.7036에서 0.5971로 하락했다.
3. **B는 충돌 때문에 실패한 것이 아니다.** 경로가 분리되어 collision episode가
   0.0265이고 median clearance가 6.588 m다.
4. 현재 결과는 기존 정지 기능이 망가진 것이 아니라, 원래 없던 운반 중 정지를 새로
   학습하면서 기존 carry policy가 함께 drift한 것으로 해석한다.
5. 현 `mastop_ab_s0`는 채택하지 않으며 C/D 학습 시작점으로 사용하지 않는다.

가능성이 큰 원인은 3000 iteration 일괄 fine-tuning, B env 비율 50%, 강한
`MS_VEL_W=5`와 stop reward의 결합, 그리고 free 행동을 보존하는 teacher/L2-SP 제약이
없는 것이다.

## 평가 주의사항

strict metric 파일의 행 수가 완전하지 않다.

```text
기대: 512 env * 2 agents * 3 repeats = 3072 rows
A:    2274 rows
A+B:  2318 rows
```

현재 metric dump가 reset된 episode만 기록하여 평가 종료 시 살아 있는 episode를
누락하는 것으로 보인다. 따라서 위 strict pickup/finish 수치는 **잠정값**이며 row-count
검증을 추가한 뒤 다시 측정해야 한다. 다만 별도 하네스 stable success도 0.7139에서
0.4619로 하락했으므로 A+B의 운반 퇴행이라는 핵심 판정은 유지된다.

현재 gate 조건은 다음과 같다.

- free: `pickup >= 0.868`, `finish >= 0.699`
- 해당 stop scenario: `stopRate >= 0.5`, `heldStop >= 0.5`, `resumed >= 0.3`
- 해당 stop scenario: `rootV <= 0.6`, `boxV <= 0.6`

A+B는 stop 조건은 만족했지만 free finish가 0.5971이어서 gate에서 중단됐다.

## 다음 실행 전 할 일

| 순서 | 작업 | 통과 조건 |
|---:|---|---|
| 1 | metric 종료 시점 직접 capture 및 3072행 검증 | 누락·중복 없이 정확히 3072행 |
| 2 | 같은 config로 A와 현재 A+B 재평가 | 하네스와 strict 지표 모두 기록 |
| 3 | A에서 B를 짧게 재학습 | 500 iteration 단위로 free/solo 평가 |
| 4 | B mix와 reward 강도 조정 | free carry gate 유지 + 실제 stop 속도 개선 |
| 5 | A+B 통과 후 C 독립 branch | exactly-one stop, clearance 후 resume |
| 6 | C 통과 후 D 독립 branch | predictor 기반 yield와 both delivery 확인 |

B 재설계 후보는 다음 실험을 시작하기 전 하나씩 분리해 결정한다.

- B 비율을 0.50에서 0.20~0.30으로 낮춘다.
- 3000 iteration 일괄 실행 대신 500 iteration마다 평가한다.
- stop bonus와 기존 `M=0` velocity reward의 합산 크기를 다시 맞춘다.
- free scenario에서 MASteer teacher action KL 또는 L2-SP를 적용한다.
- 필요하면 현재 B 앞에 한 명만 active carrier인 B0 진단 단계를 둔다. 네트워크 관측
  호환성을 위해 두 번째 agent/box를 물리적으로 삭제하지 않고 inactive mask를 설계한다.

## 원시 자료

- chain log: `runs/queue/logs/mastop_abcd_chain.log`
- A raw eval: `runs/queue/logs/eval_mastop_a_ms12_3k.raw.log`
- A+B raw eval: `runs/queue/logs/eval_mastop_ab_s0.raw.log`
- A metrics: `runs/results/mastop_carry/eval_mastop_a_ms12_3k.npy`
- A+B metrics: `runs/results/mastop_carry/eval_mastop_ab_s0.npy`
- task: `TokenHSI-mastop/tokenhsi/env/tasks/adapt_interaction_skills/humanoid_ma_stop_carry.py`
- train: `TokenHSI-mastop/tokenhsi/scripts/tokenhsi/stage2_ma_stop_carry_train.sh`
- eval: `TokenHSI-mastop/tokenhsi/scripts/tokenhsi/stage2_ma_stop_carry_eval.sh`

## 갱신 규칙

- 새 결과는 실행 날짜, tag, 시작 checkpoint, mix, reward 변경점, seed와 함께 기록한다.
- 하네스 success와 strict metric을 구분한다.
- strict metric은 기대 row 수가 맞지 않으면 잠정값으로 표시한다.
- gate 실패는 삭제하지 않고 실패 원인과 함께 남긴다.
- C/D 결과가 생기기 전에는 완료로 표시하지 않는다.

## 2026-08-26 · Pickup 원인 분리 선행 실험

정지 curriculum 전에 세 축을 분리한다. f22x2는 추가학습 없는 single-policy×A=2,
pickupdiag_zero/live_3k_s0는 같은 구조·학습량에서 상대 정보만의 효과,
pickupdiag_zero/live_25k_s0는 3K 학습량 부족 여부를 본다.

f22x2는 f22의 정책 계약 [self 223 | dual-steer 24 | carry 42]=289를 유지하고 MA의
A=2 actor/reset 물리만 사용한다. 2-env smoke에서 checkpoint strict-load와 47열 strict
metric 저장까지 통과했으나 pick=0/6, close=0/6이었다. 표본이 작으므로 성능 결론이
아니라 호환 경로 동작 확인으로만 기록한다.

zero/live 학습은 둘 다 Stage 1에서 시작하고 네트워크 차원을 같게 유지한다. zero는
21D teammate 입력값만 0으로 만들며, 25K는 각 조건의 3K checkpoint를 resume한다.
strict pickup은 손 거리 0.35m 미만, 초기 지지면 대비 lift 0.10m 초과가 10 policy step
연속일 때만 성공이다. 새 산출물은 모두 runs/*/masteer_pickup/ 아래에 둔다.

### 2026-08-26 14:32 · zero/live 3K 실행 및 조건부 25K

Codex 종료와 무관하게 유지되도록 프로젝트 표준인 setsid+nohup 독립 세션으로 시작했다.

| 실험 | GPU | session PID | 설정 |
|---|---:|---:|---|
| pickupdiag_zero_3k_s0 | 6 | 2497412 | A=2, 2048 env, teammate 21D 값을 0 |
| pickupdiag_live_3k_s0 | 7 | 2497413 | A=2, 2048 env, teammate 21D live |
| pickupdiag_followup | CPU wait | 2499916 | 두 3K 종료 후 평가와 조건부 25K |

첫 nohup 시도는 서버 실행 관리기가 자식을 정리해 학습 시작 전에 종료됐다. 그때의
작은 config/log는 삭제하지 않고
runs/logs/masteer_pickup/failed_launch_20260826_143254/에 보존했다. 재실행본은 두 GPU에서
각각 약 18 GB를 사용하고 첫 PPO iteration 및 47열 metric flush까지 확인했다.

followup의 판정 순서는 다음과 같다.

1. 두 3K exit가 모두 rc=0인지 확인한다.
2. GPU 6/7에서 512-env strict 평가를 동시에 실행한다.
3. 두 정책 모두 pick < 0.85 또는 bothPick < 0.70이면 각자의 3K checkpoint에서
   총 25K까지 resume한다.
4. 한 정책이라도 gate를 통과하면 teammate 입력 효과를 먼저 분석하기 위해 25K를
   자동 시작하지 않는다.
5. 25K가 실행된 경우 완료 뒤 같은 512-env strict 평가를 자동 실행한다.

저장 위치:

- checkpoint: runs/output/masteer_pickup/<tag>/*/nn/Humanoid.pth
- 재현 config: runs/gen_cfgs/masteer_pickup/<tag>.yaml
- 학습 로그/env/exit: runs/logs/masteer_pickup/<tag>.{log,env,exit}
- 학습 metric: runs/results/masteer_pickup/train_<tag>.npy
- 3K 평가: runs/logs/masteer_pickup/eval_<tag>_gate3k.log 및
  runs/results/masteer_pickup/eval_<tag>_gate3k.npy
- 25K 평가: 같은 디렉터리의 eval_<tag>_full25k.{log,npy}
- 자동 후속 상태: runs/logs/masteer_pickup/pickupdiag_followup.status

### 2026-08-26 · 실행 계획 변경: 15K 고정, 3K 간격 저장

이 절이 위의 3K gate/조건부 25K 계획을 대체한다. 사용자 요청으로 실행 중이던
pickupdiag_zero/live_3k_s0와 pickupdiag_followup watcher를 모두 정상 종료했다.
두 3K 중단본은 약 38 iteration까지 진행됐고 checkpoint가 생기기 전이며, 로그와 metric은
삭제하지 않고 보존한다. 조건부 25K는 더 이상 자동 실행되지 않는다.

새 실행:

| 실험 | GPU | session PID | 총 iteration | checkpoint 간격 |
|---|---:|---:|---:|---:|
| pickupdiag_zero_15k_s0 | 6 | 2504000 | 15000 | 3000 |
| pickupdiag_live_15k_s0 | 7 | 2504001 | 15000 | 3000 |

두 run 모두 Stage 1에서 새로 시작하고 2048 env, A=2, seed 0을 사용한다. 차이는
teammate 21D 입력이 zero인지 live인지뿐이다. run별 train config에서
save_frequency=3000, save_intermediate=True를 함께 설정해 최신 Humanoid.pth 외에 다음
중간 checkpoint를 별도 보존한다.

- Humanoid_00003000.pth
- Humanoid_00006000.pth
- Humanoid_00009000.pth
- Humanoid_00012000.pth
- Humanoid_00015000.pth

저장 루트는 runs/output/masteer_pickup/pickupdiag_{zero,live}_15k_s0/*/nn/ 이다.
학습 로그와 strict metric은 각각 runs/logs/masteer_pickup/ 및
runs/results/masteer_pickup/에 저장된다.
