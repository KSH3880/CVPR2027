# 원본 TokenHSI Stage 1 Carry reward

원본 baseline [`TokenHSI/`](../TokenHSI/)의 Stage 1 **Carry 보상만** 탑다운 방식으로 정리한다.
앞부분은 전체 구조와 행동 관점의 요약이고, 뒷부분은 현재 코드의 수식, 게이트, 기본 계수를 그대로
풀어 쓴 상세 설명이다.

기준 실행과 구현:

- 학습: [`stage1_train.sh`](../TokenHSI/tokenhsi/scripts/tokenhsi/stage1_train.sh)
- 환경: [`humanoid_traj_sit_carry_climb.py`](../TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py)
- 환경 설정: [`amp_humanoid_traj_sit_carry_climb.yaml`](../TokenHSI/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml)
- 학습 설정: [`amp_imitation_task_transformer_multi_task.yaml`](../TokenHSI/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml)

---

## 1. 한눈에 보는 Carry 보상

Stage 1의 Carry 학습 보상은 크게 세 층이다.

```text
Carry task reward
├─ box로 접근한다                 r_walk
├─ box를 최종 목표로 이동한다      r_transport
├─ 두 손을 box 가까이에 유지한다   r_handheld
└─ 목표 위치에 정확히 내려놓는다   r_putdown
        + box 과속 penalty
        + 관절 power penalty
        + AMP motion-style reward
```

PPO가 실제로 사용하는 최종식은 다음과 같다.

```math
r_{train}
=0.5\left(r_{walk}+r_{transport}+r_{handheld}+r_{putdown}+r_{power}\right)
+0.5r_{AMP}
```

여기서 `r_transport` 안에는 box 과속 패널티가 포함되어 있다. AMP를 제외하고 환경이 직접 반환하는
보상은 다음과 같다.

```math
r_{env}=r_{walk}+r_{transport}+r_{handheld}+r_{putdown}+r_{power}
```

기본 설정에서 패널티를 제외한 Carry task reward의 스텝당 양의 최댓값은 `1.0`이다.

| 구성 | 최대값 | 주된 역할 |
|---|---:|---|
| `r_walk` | 0.2 | humanoid가 box로 가고 가까이 머무름 |
| `r_transport`의 양의 항 | 0.4 | box를 목표 방향으로 운반하고 목표 3D 위치에 맞춤 |
| `r_handheld` | 0.2 | 두 손의 평균 위치를 box 중심 가까이에 유지 |
| `r_putdown` | 0.2 | box를 목표 높이 1 mm, 수평 10 cm 이내에 배치 |

`r_train`의 두 가중치가 각각 0.5라는 뜻이지, task와 AMP의 실제 수치 기여량이 항상 정확히 절반씩
같다는 뜻은 아니다. 두 보상의 값 범위가 다르다.

---

## 2. 행동 단계로 보는 보상

Carry를 행동 순서대로 보면 다음처럼 이해할 수 있다.

### 2.1 Box에 접근

- humanoid가 box 방향으로 약 `1.5 m/s`로 움직이면 `r_walk`가 커진다.
- box 수평 0.5 m 이내에 들어오면 `r_walk`는 최대 `0.2`로 고정된다.
- 손이 box 중심에 가까워지면 `r_handheld`도 커진다.

이 구간에서는 humanoid를 최종 placement target으로 직접 끌어가는 항이 없다. humanoid의 직접적인
이동 목표는 먼저 box다.

### 2.2 Pickup과 손 유지

- 별도의 `grasp_success` 또는 contact binary reward는 없다.
- `r_handheld`는 두 손 위치의 평균과 box 중심의 3D 거리를 본다.
- humanoid root와 box가 수평 0.7 m보다 멀어지면 `r_handheld=0`이다.

즉, pickup은 명시적 성공 플래그보다 손-박스 근접, box 이동 reward, AMP motion prior의 결합으로
유도된다.

### 2.3 목표로 운반

- box가 목표 방향으로 약 `1.5 m/s`로 이동하면 transport 속도 reward가 커진다.
- box가 목표의 정확한 3D 위치에 가까워질수록 별도의 near-position reward가 커진다.
- box의 3D 속력이 `2.5 m/s`를 넘으면 과속 패널티를 받는다.

최종 목표를 향한 신호는 humanoid root가 아니라 **box의 위치와 속도**를 기준으로 한다. 그래서 box가
움직이지 않는 pre-grasp 구간에서는 최종 target의 직접적인 영향이 약하다.

### 2.4 목표에 내려놓기

- 목표와의 수평 거리가 0.5 m 이내이면 transport 속도 항은 조건과 무관하게 최대값으로 덮어써진다.
- 목표의 정확한 3D 위치에 가까우면 near-position reward가 최대에 가까워진다.
- 높이 오차 1 mm 이하, 수평 오차 10 cm 이하이면 `r_putdown=0.2`다.

`r_putdown`은 한 번만 주는 terminal bonus가 아니다. 조건을 만족하는 매 스텝에 주는 reward이며,
Carry는 성공했다고 즉시 끝내는 IET가 없다.

> 위 단계는 이해를 위한 구분이다. 코드에는 approach/pickup/transport/place 상태를 판별해 reward를
> 전환하는 state machine이 없다. 네 양의 항과 패널티를 **모든 Carry 스텝에서 동시에 계산**한다.

---

## 3. 기본 설정

Stage 1 Carry에서 실제 활성화된 주요 설정은 다음과 같다.

| 설정 | 값 | 의미 |
|---|---:|---|
| `onlyVelReward` | `True` | walk의 거리 항과 transport의 far-distance 항을 최종식에서 제외 |
| `onlyHeightHandHeldReward` | `False` | 손-박스 reward에 높이만이 아니라 3D 거리 사용 |
| `box_vel_penalty` | `True` | box 과속 패널티 활성화 |
| `box_vel_pen_threshold` | 2.5 m/s | box 과속 기준 |
| `box_vel_pen_coeff` | 1.0 | box 과속 패널티 계수 |
| 접근/운반 목표 속도 | 1.5 m/s | 코드 호출부에서 전달하는 값 |
| `power_reward` | `True` | 관절 power 패널티 활성화 |
| `power_coefficient` | 0.0005 | 관절 power 패널티 계수 |
| `task_reward_w` | 0.5 | 환경 reward 결합 가중치 |
| `disc_reward_w` | 0.5 | AMP reward 결합 가중치 |
| `disc_reward_scale` | 2 | AMP reward scale |

`onlyVelReward=True`라는 이름과 달리 전체 Carry 보상이 속도만 보는 것은 아니다. 현재 최종식에도
box의 3D 목표 위치, 손-박스 거리, putdown 위치가 들어간다. 이 옵션이 제거하는 것은 두 개의
far-distance shaping 항뿐이다.

---

## 4. 표기

| 기호 | 의미 |
|---|---|
| `p_h`, `p_h^-` | 현재/직전 humanoid root 위치 |
| `p_b`, `p_b^-` | 현재/직전 box 중심 위치 |
| `g_b` | box의 최종 placement target |
| `p_R`, `p_L` | 오른손과 왼손 위치 |
| `p_H=(p_R+p_L)/2` | 두 손 위치의 평균 |
| `h_b` | box 높이 |
| `dt` | control step, 기본 `1/30 s` |
| `v_h=(p_h-p_h^-)/dt` | humanoid root의 유한차분 3D 속도 |
| `v_b=(p_b-p_b^-)/dt` | box 중심의 유한차분 3D 속도 |

코드의 거리 변수는 대부분 Euclidean distance가 아니라 **squared distance**다.

---

## 5. `r_walk`: humanoid가 Box로 접근

구현: [`compute_walk_reward()`](../TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2306)

먼저 humanoid에서 box로 향하는 수평 단위 방향과 그 방향의 root 속도를 구한다.

```math
\hat d_{h\rightarrow b}
=\operatorname{normalize}(p_{b,xy}-p_{h,xy})
```

```math
v_{h\rightarrow b}=\hat d_{h\rightarrow b}^{\mathsf T}v_{h,xy}
```

목표 속도는 `1.5 m/s`다.

```math
r_{walk\_vel}=\exp\left[-5(1.5-v_{h\rightarrow b})^2\right]
```

그 뒤 다음 게이트를 적용한다.

```text
if v_h→b <= 0:
    r_walk_vel = 0

if ||p_b,xy - p_h,xy|| < 0.5 m:
    r_walk_vel = 1
```

두 번째 조건이 나중에 실행되므로, box 0.5 m 이내에서는 현재 이동 방향이나 속도와 무관하게 1이다.
기본 설정의 최종 walk reward는 다음과 같다.

```math
r_{walk}=0.2r_{walk\_vel}
```

코드는 아래 위치 reward도 계산한다.

```math
r_{walk\_pos}=\exp\left[-0.5\|p_{b,xy}-p_{h,xy}\|^2\right]
```

하지만 `onlyVelReward=True`이므로 원본 Stage 1 최종식에서는 사용하지 않는다. 이 이름은 접근용으로
붙었지만 `r_walk`는 pickup 이후에도 계속 계산되며, humanoid가 들고 있는 box 가까이에 있으면 보통
계속 최대값을 받는다.

---

## 6. `r_transport`: Box를 최종 목표로 이동

구현: [`compute_carry_reward()`](../TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2340)

### 6.1 목표 방향 속도 reward

```math
\hat d_{b\rightarrow g}
=\operatorname{normalize}(g_{b,xy}-p_{b,xy})
```

```math
v_{b\rightarrow g}=\hat d_{b\rightarrow g}^{\mathsf T}v_{b,xy}
```

```math
r_{box\_vel}=\exp\left[-5(1.5-v_{b\rightarrow g})^2\right]
```

이 값에는 아래 게이트가 **표시된 순서대로** 적용된다.

```text
1. if v_b→g <= 0:
       r_box_vel = 0

2. if p_b,z <= h_b/2 + 0.2 m:
       r_box_vel = 0

3. if ||g_b,xy - p_b,xy|| < 0.5 m:
       r_box_vel = 1
```

2번은 낮은 box를 밀거나 차서 목표로 보내는 행동을 억제하려는 height gate다. 하지만 3번이 나중에
실행되므로 목표 수평 0.5 m 이내에서는 진행 방향과 height gate를 덮어쓰고 `r_box_vel=1`이 된다.

### 6.2 목표 3D 위치 reward

```math
r_{box\_near}=\exp\left[-10\|g_b-p_b\|^2\right]
```

이 항은 x, y뿐 아니라 z까지 본다. 별도의 box height gate는 적용되지 않는다.

코드는 수평 far-position reward도 계산한다.

```math
r_{box\_far}=\exp\left[-0.5\|g_{b,xy}-p_{b,xy}\|^2\right]
```

그러나 기본 `onlyVelReward=True`에서는 최종식에서 빠진다. 따라서 활성화된 양의 transport reward는
다음 두 항이다.

```math
r_{transport}^{+}=0.2r_{box\_vel}+0.2r_{box\_near}
```

### 6.3 Box 과속 패널티

box의 3D 속력 `||v_b||`가 `2.5 m/s`를 넘으면 패널티를 받는다.

```math
P_{box\_vel}
=-\left(1-\exp\left[-2\max(\|v_b\|-2.5,0)^2\right]\right)
```

이 패널티에는 목표까지의 거리 게이트가 없다. box가 어디에 있든 과속하면 적용된다.

```math
r_{transport}=0.2r_{box\_vel}+0.2r_{box\_near}+P_{box\_vel}
```

`P_box_vel`은 0에서 시작해 속력이 커질수록 `-1`에 가까워진다.

---

## 7. `r_handheld`: 손과 Box의 근접

구현: [`compute_handheld_reward()`](../TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2288)

기본 `onlyHeightHandHeldReward=False`이므로 두 손 평균 위치와 box 중심의 3D 거리를 쓴다.

```math
p_H=\frac{p_R+p_L}{2}
```

```math
r_{handheld}=0.2\exp\left[-5\|p_H-p_b\|^2\right]
```

단, humanoid root와 box 중심의 수평 거리가 0.7 m보다 크면 강제로 0이다.

```text
if ||p_b,xy - p_h,xy|| > 0.7 m:
    r_handheld = 0
```

주의할 점:

- 양손 각각이 box 표면의 grasp point에 닿았는지는 보지 않는다.
- contact force나 grasp constraint도 보지 않는다.
- 두 손의 **평균 위치**와 box **중심**의 거리만 본다.
- 함수에 `g_b`가 인자로 전달되지만 활성 코드에서는 handheld 계산에 사용하지 않는다.

---

## 8. `r_putdown`: 목표에 정확히 배치

구현: [`compute_putdown_reward()`](../TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L2389)

```math
r_{putdown}=0.2\,\mathbf{1}
\left[
|p_{b,z}-g_{b,z}|\le0.001
\;\land\;
\|p_{b,xy}-g_{b,xy}\|\le0.1
\right]
```

즉 두 조건을 동시에 만족해야 한다.

- box 중심 높이 오차: `1 mm` 이하
- box 중심 수평 오차: `10 cm` 이하

box의 회전, 속도, 접촉 상태는 putdown 판정에 직접 사용하지 않는다. 조건을 유지하면 여러 스텝에 걸쳐
반복해서 `0.2`를 받을 수 있다.

---

## 9. `r_power`: 공통 관절 Power 패널티

구현: [`_compute_reward()`](../TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L1017)

```math
r_{power}=-0.0005\sum_j|\tau_j\dot q_j|
```

- `tau_j`: simulator가 기록한 DoF force
- `qdot_j`: 관절 속도
- `j`: 모든 DoF

코드에는 별도의 `dt` 곱이 없다. 이 값은 Carry task reward에 먼저 더해져 `r_env`가 되고, 이후 전체
환경 reward weight `0.5`가 적용된다.

---

## 10. `r_AMP`: Carry 동작의 Motion-style reward

구현: [`AMPAgent._calc_disc_rewards()`](../TokenHSI/tokenhsi/learning/amp_agent.py#L645)

환경 reward만으로는 자연스러운 보행, pickup, 운반 자세를 충분히 지정하지 않는다. AMP 판별기는
10-frame motion window가 reference motion처럼 보이는지를 logit `D`로 출력한다.

```math
p_{real}=\sigma(D)
```

```math
r_{AMP}=-2\log\left(\max(1-p_{real},10^{-4})\right)
```

- `2`는 `disc_reward_scale=2`다.
- `D=0`, `p_real=0.5`이면 `r_AMP=2 log 2`, 약 `1.386`이다.
- clamp에 따른 최댓값은 약 `18.42`다.
- AMP reward는 `torch.no_grad()` 안에서 계산되며 PPO용 scalar reward로만 들어간다.

Stage 1은 하나의 task-conditioned discriminator를 사용한다. Carry 환경의 AMP frame에는 Carry
one-hot이 붙으며, 한 frame은 133차원, 10-frame 입력은 총 1330차원이다.

Carry로 라벨링되는 reference skill의 discriminator sampling 확률은 다음과 같다.

| Carry reference skill | `skillDiscProb` |
|---|---:|
| `loco_carry` | 0.10 |
| `omomo` | 0.10 |
| `pickUp` | 0.05 |
| `carryWith` | 0.00 |
| `putDown` | 0.05 |

특히 `carryWith`는 환경의 RSI 시작 skill로는 사용되지만, AMP demo sampling 확률은 `0.0`이다.

---

## 11. 최종식을 완전히 펼치면

원본 Stage 1 기본 설정에서 활성 Carry 보상을 한 식으로 쓰면 다음과 같다.

```math
r_{train}=0.5\Big(
0.2r_{walk\_vel}
+0.2r_{box\_vel}
+0.2r_{box\_near}
+P_{box\_vel}
+r_{handheld}
+r_{putdown}
+r_{power}
\Big)
+0.5r_{AMP}
```

여기서 각 항은 다음 범위를 갖는다.

| 항 | 범위 |
|---|---:|
| `0.2 r_walk_vel` | `[0, 0.2]` |
| `0.2 r_box_vel` | `[0, 0.2]` |
| `0.2 r_box_near` | `(0, 0.2]` |
| `P_box_vel` | `(-1, 0]` |
| `r_handheld` | `[0, 0.2]` |
| `r_putdown` | `{0, 0.2}` |
| `r_power` | `<= 0`, 고정 하한 없음 |
| `r_AMP` | `[0, 약 18.42]` |

---

## 12. 성공과 종료는 Reward와 별개

- Carry에는 목표 달성 시 주는 별도의 terminal success bonus가 없다.
- `r_putdown`을 받았다고 episode가 즉시 종료되지 않는다.
- Stage 1의 IET target은 Sit과 Climb에만 지정되며 Carry에는 적용되지 않는다.
- Carry episode의 time limit은 기본 600 control step, 즉 약 20초다.
- 낙상은 episode를 끝내지만 별도의 낙상 penalty를 reward 식에 직접 더하지 않는다.
- 평가의 success 판정도 학습 reward를 바꾸지 않는다.

따라서 Carry 완료는 학습 중에는 `r_box_near`와 반복 가능한 `r_putdown`으로 보상되며, 종료 bonus가
아니라 남은 episode 동안 받을 수 있는 누적 reward의 형태로 반영된다.

---

## 13. 핵심 해석

1. **Pre-grasp에서 최종 target 영향이 약한 이유**: humanoid를 직접 움직이는 `r_walk`는 box만 향하고,
   최종 target reward는 box가 실제로 움직여야 커진다.
2. **명시적 grasp reward가 없다**: 손-박스 중심 근접, box 이동, AMP가 함께 pickup을 유도한다.
3. **`onlyVelReward=True`여도 위치 reward가 남는다**: `r_box_near`, `r_handheld`, `r_putdown`은 모두
   위치 기반이다.
4. **목표 0.5 m 이내의 우선순위가 강하다**: `r_box_vel`이 방향·속도·height gate와 무관하게 1로
   덮어써진다.
5. **Putdown은 매우 엄격하지만 terminal이 아니다**: 높이 1 mm와 수평 10 cm 조건을 매 스텝 검사한다.
6. **자연스러운 동작의 상당 부분은 AMP에 맡긴다**: task reward는 목표 성취를, AMP는 사람다운 motion
   prior를 제공하며 결합 가중치는 각각 0.5다.

---

## 14. 실제 진행 단계별 Reward 활성 강도

코드에는 approach, pickup, transport, putdown을 구분하는 reward state machine이 없다. 모든 항을
매 Carry 스텝에 동시에 계산하지만, 거리와 높이 게이트 때문에 실제 지배적인 항이 아래처럼 바뀐다.

| 진행 상태 | 주로 작동 | 보조적으로 작동 | 거의 없거나 0 |
|---|---|---|---|
| Box에서 멀리 있음 | `r_walk` | AMP, power penalty | handheld=0, transport≈0, putdown=0 |
| 사람-Box 0.7 m 부근 | `r_walk`, `r_handheld` | AMP | transport≈0, putdown=0 |
| 사람-Box 0.5 m 안, 아직 안 듦 | pinned `r_walk`, `r_handheld` | AMP | box velocity=0, near≈0, putdown=0 |
| Box를 들어 올리는 중 | pinned `r_walk`, `r_handheld` | AMP, transport가 열리기 시작 | putdown=0 |
| Box를 들고 운반 | `r_walk`, `r_handheld`, `r_box_vel` | AMP | target이 멀면 `r_box_near`≈0 |
| Target 0.5 m 안 | pinned `r_box_vel`, `r_box_near` | walk, handheld | putdown은 대부분 0 |
| 정확한 Target에 배치 | `r_box_vel`, `r_box_near`, `r_putdown` | walk, handheld | — |

### 14.1 Box에서 멀리 있을 때

사람-Box 수평 거리가 0.7 m보다 크면 `r_handheld=0`이다. Box가 아직 정지해 있고 Target도 멀다면
`r_box_vel=0`, `r_box_near≈0`, `r_putdown=0`이므로 실질적인 task 신호는 `r_walk`뿐이다.

| Box 방향 humanoid 속도 | 실제 `r_walk` |
|---:|---:|
| 0 또는 반대 방향 | 0 |
| 0.5 m/s | 약 0.0013 |
| 1.0 m/s | 약 0.057 |
| 1.5 m/s | 0.200 |
| 2.0 m/s | 약 0.057 |
| 2.5 m/s | 약 0.0013 |

따라서 이 구간의 명령은 사실상 “Box 방향으로 약 1.5 m/s로 접근하라”다.

### 14.2 사람-Box 0.7 m 안

root-Box 수평 거리 gate가 풀리면서 손 평균 위치와 Box 중심의 3D 거리에 따른 `r_handheld`가 들어온다.

| 양손 평균-Box 중심 거리 | `r_handheld` |
|---:|---:|
| 0.5 m | 약 0.057 |
| 0.3 m | 약 0.128 |
| 0.2 m | 약 0.164 |
| 0.1 m | 약 0.190 |
| 0 m | 0.200 |

이 단계에서는 `r_walk`가 몸을 Box까지 보내고, `r_handheld`가 손을 Box 중심으로 가져가며, AMP가
reference pickup과 유사한 자세를 유도한다.

### 14.3 사람-Box 0.5 m 안, 아직 들기 전

`r_walk=0.2`로 pin된다. 이제 이 항은 방향이나 속도 shaping을 하지 않고 “Box 가까이에 머물라”는
상수성 보상이 된다. Box가 여전히 낮으면 height gate 때문에 `r_box_vel=0`이고, Target이 멀면
`r_box_near`도 사실상 0이다. 따라서 실제 grasp 직전에는 `r_handheld`와 AMP가 핵심 신호다.

예를 들어 사람-Box 수평 거리 0.4 m, 손 평균-Box 중심 거리 0.3 m, Box가 정지해 있고 Target이 멀면:

```text
r_walk       = 0.200
r_handheld   ≈ 0.128
r_transport  ≈ 0
r_putdown    = 0
──────────────────
raw task     ≈ 0.328
```

### 14.4 Box를 들고 운반할 때

Box 중심이 `h_b/2 + 0.2 m`보다 높고 Target 방향으로 움직이기 시작하면 `r_box_vel`이 주요 운반
신호가 된다. 일반적인 정상 운반 상태의 양의 task reward는 대략 다음과 같다.

```text
r_walk       ≈ 0.2    # 사람과 Box가 가까워 pin
r_handheld   ≈ 0.2    # 손과 Box가 가까움
r_box_vel    ≈ 0.2    # Target 방향 1.5 m/s
r_box_near   ≈ 0      # Target이 아직 멀면 매우 작음
r_putdown    = 0
──────────────────
raw task     ≈ 0.6
```

`r_box_near`는 항상 계산되지만 full 3D 거리에 대한 지수함수라 멀리서는 거의 사라진다.

| Box-Target 3D 거리 | 실제 `0.2 r_box_near` |
|---:|---:|
| 2.0 m | 사실상 0 |
| 1.0 m | 약 0.000009 |
| 0.7 m | 약 0.0015 |
| 0.5 m | 약 0.0164 |
| 0.3 m | 약 0.0813 |
| 0.2 m | 약 0.1341 |
| 0.1 m | 약 0.1810 |
| 0 m | 0.2000 |

### 14.5 Target 0.5 m 안

`r_box_vel=0.2`로 pin되어 운반 속도 요구가 사라지고, `r_box_near`가 정확한 3D 위치로 수렴시키는
주요 가변 신호가 된다. 예를 들어 Target 수평 거리 0.3 m, 높이 오차 0, 손 평균 거리 0.1 m라면:

```text
r_walk       = 0.200
r_handheld   ≈ 0.190
r_box_vel    = 0.200    # Target 0.5m pin
r_box_near   ≈ 0.081
r_putdown    = 0        # 아직 수평 10cm 밖
──────────────────
raw task     ≈ 0.671
```

이 구간에서는 Box를 낮춰 height gate 조건에 들어가도, 뒤에서 실행되는 Target 0.5 m pin이
`r_box_vel`을 다시 1로 덮어쓴다. 따라서 감속하고 내려놓는 동안에도 속도 항의 실제 기여 `0.2`가
유지된다.

### 14.6 정확한 위치에 도달했을 때

사람과 손이 Box 가까이에 있고 Box가 Target 중심과 정확히 일치하면:

```text
r_walk       = 0.2
r_box_vel    = 0.2
r_box_near   = 0.2
r_handheld   = 0.2
r_putdown    = 0.2
──────────────────
양의 raw task = 1.0
```

반대로 Box만 정확히 배치하고 사람이 0.7 m보다 멀리 물러나면 `r_walk`와 `r_handheld`를 잃어
약 `0.6/step`만 남는다. 보상만 보면 배치 후 완전히 release하고 떠나는 것보다 Box와 손을 가까이
유지하는 쪽이 더 높은 return을 만들 수 있다.

---

## 15. Pin, Near, Putdown의 정확한 차이

세 표현을 혼동하지 않도록 정리하면 다음과 같다.

| 항 | 종류 | 조건과 효과 |
|---|---|---|
| 사람-Box `walk` pin | hard overwrite | 수평 0.5 m 미만이면 `r_walk_vel=1`, 실제 `r_walk=0.2` |
| Box-Target velocity pin | hard overwrite | 수평 0.5 m 미만이면 `r_box_vel=1`, 실제 기여 0.2 |
| `r_box_near` | continuous shaping | Box-Target 3D 거리에 따라 `0.2 exp(-10d²)` |
| `r_putdown` | per-step binary bonus | 수평 0.1 m 이하이면서 높이 오차 1 mm 이하이면 0.2, 아니면 0 |

Target 0.5 m pin은 전체 `carry_r`나 전체 Carry reward를 고정하지 않는다. `r_box_vel`이라는 한
subterm만 1로 덮어쓴다. `r_box_near`와 `r_putdown`은 그 뒤에도 각자의 조건으로 독립 계산된다.

`r_putdown`은 이름과 달리 실제 put-down event detector가 아니다. 다음은 확인하지 않는다.

- 손이 Box에서 떨어졌는가
- grasp/contact가 해제됐는가
- Box가 바닥이나 플랫폼에 접촉했는가
- Box가 정지했는가
- Box 회전이 올바른가
- 이전에 Box를 들어 올린 적이 있는가

실제 구현 의미는 “현재 Box 중심이 지정된 Target 중심의 수평 10 cm, 높이 1 mm 허용 범위 안에
있는가”이다. 조건을 만족하는 동안 매 스텝 0.2를 받고, 다음 스텝에 벗어나면 즉시 0이 된다. 한 번
켜진 뒤 유지되는 latch도 아니고, 성공 episode를 끝내는 terminal bonus도 아니다.

예를 들어 `Target=(2.00, 1.00, 0.2500)`일 때:

| Box 중심 | 수평 오차 | 높이 오차 | `r_putdown` |
|---|---:|---:|---:|
| `(2.08, 1.04, 0.2505)` | 약 8.9 cm | 0.5 mm | 0.2 |
| `(2.08, 1.04, 0.2520)` | 약 8.9 cm | 2.0 mm | 0 |
| `(2.11, 1.00, 0.2500)` | 11 cm | 0 mm | 0 |

---

## 16. Scratch 학습에서 “숙였다가 못 잡고 발로 차기”가 나오는 이유

관찰된 현상:

```text
Box까지 접근 성공
→ 몸을 숙이고 손을 가져감
→ 실제 grasp/lift에는 실패
→ 다시 일어나 Box를 발로 참
→ Box를 Target까지 보냄
```

이는 현재 보상에서 가능한 전형적인 shortcut 또는 reward hacking으로 해석할 수 있다.

### 16.1 접근과 숙이기까지는 직접 신호가 있음

- `r_walk`가 Box 접근을 직접 보상한다.
- `r_handheld`가 손 평균 위치를 Box 중심으로 가져가는 것을 보상한다.
- AMP가 pickup reference와 비슷한 숙이기 자세를 보상할 수 있다.

따라서 coarse navigation과 pickup 직전 자세는 비교적 쉽게 배운다.

### 16.2 실제 grasp와 lift 사이에 Reward gap이 있음

다음에 대한 명시적 양의 reward는 없다.

```text
양손 contact 성공
안정적인 grasp 성립
Box가 손에 고정됨
Box가 처음 바닥에서 떨어짐
Box를 처음 일정 높이 이상 들어 올림
```

더구나 Box 중심이 world z 기준 `h_b/2 + 0.2 m`보다 낮은 동안 transport velocity reward는 0이다.
지면에서 시작한 Box라면 정책은 정밀한 양손 접촉을 만들고 Box를 약 20 cm 이상 들어 올린 뒤에야
transport reward를 얻는다. 이미 높은 플랫폼 위에 있는 Box에는 같은 해석이 적용되지 않는다.
scratch 정책에는 이 credit-assignment 구간이 어렵다.

### 16.3 발차기는 즉시 Target reward를 만들 수 있음

`r_box_near`에는 grasp나 lift gate가 없다. 따라서 Box를 손으로 들든, 바닥으로 밀든, 발로 차든
Target에 가까워지면 같은 near reward를 받는다.

또한 바닥에 있는 Box가 Target 수평 0.5 m 안에 들어가면 마지막 pin이 height gate를 덮어써서
`r_box_vel=0.2`가 된다. Ground target에서 위치가 충분히 정확하면 실제 release 없이
`r_putdown=0.2`도 받을 수 있다.

사람과 손이 Box에서 멀어진 상태라도 정확한 Target에 Box가 머물면 조건에 따라 다음 보상이 남는다.

```text
r_box_vel pin = 0.2
r_box_near    = 0.2
r_putdown     = 0.2
──────────────────
raw task      = 0.6 / step
```

Carry에는 성공 IET가 없으므로 이 값은 one-shot bonus가 아니라 남은 episode 동안 반복될 수 있다.
어려운 grasp와 지속 운반으로 최대 1.0을 얻는 경로보다, 쉬운 발차기로 반복 가능한 0.6에 도달하는
경로가 학습 초기에 더 안정적인 local optimum이 될 수 있다.

### 16.4 Power와 AMP의 영향

- 정상 Carry는 여러 스텝 동안 Box를 지지하고 균형을 유지하므로 power를 지속 사용한다.
- 발차기는 순간적인 힘만 사용하고 끝날 수 있어 누적 power penalty가 더 작을 가능성이 있다. 실제
  크기는 로그로 확인해야 한다.
- AMP는 발차기를 부자연스럽다고 판별할 수 있지만 scratch 학습 초기에는 판별기도 함께 학습된다.
  초기의 약한 판별 신호보다 반복 가능한 task shortcut이 먼저 자리 잡을 수 있다.
- Stage 1 discriminator demo에서 `carryWith`의 `skillDiscProb`은 0이고 `omomo`가 0.1이므로, 실제
  carry gait에 대한 style supervision 구성도 함께 확인할 필요가 있다.

### 16.5 변경한 입력이 Grasp 난도를 높였을 가능성

원본 Carry 입력에는 다음 정밀 object 정보가 있다.

```text
Box 선속도 3D
Box 각속도 3D
Box 위치 3D
Box 회전 6D
Box 꼭짓점/BPS 24D
최종 Target 위치 3D
```

Box까지 접근하려면 대략적인 방향과 거리만 있어도 되지만, grasp에는 Box 높이·크기·회전·표면과
손의 정밀한 상대관계가 필요하다. 새 입력에서 BPS, 회전, 높이, local relative pose가 빠졌거나
표현/정규화가 약해졌다면 다음과 같은 분리가 나타날 수 있다.

```text
coarse navigation 성공
pickup 모양 흉내 성공
정밀 손 배치와 물리 grasp 실패
```

따라서 현재 현상은 보통 두 원인의 결합으로 보는 것이 타당하다.

1. 원래 reward에 kicking shortcut이 존재한다.
2. 변경한 입력 또는 scratch optimization 때문에 정상 grasp 경로가 원본보다 어려워졌다.

---

## 17. 원인 확인용 계측

먼저 reward를 바꾸기보다 아래 항을 스텝별로 따로 기록하면 가설을 확인할 수 있다. 수정 실험은 원본
`TokenHSI/`가 아니라 해당 실험 레포에서 해야 한다.

```text
human_box_xy_dist
hands_box_3d_dist
box_height_gate_margin = box_z - (box_height/2 + 0.2)
box_target_xy_dist
box_target_3d_dist
box_speed_3d

walk_r
box_vel_positive_r
box_near_r
box_vel_penalty
handheld_r
putdown_r
power_r
amp_r
```

발차기 shortcut이면 다음 시간 순서가 예상된다.

```text
1. Box 접근        → walk_r 상승
2. 몸 숙임         → handheld_r와 AMP 상승
3. grasp 실패      → ground Box에서 box_height_gate_margin이 0을 넘지 못함
4. 발차기          → handheld_r 감소, box_target distance 감소
5. Target 0.5m 안  → 낮은 Box인데도 box_vel positive r가 0.2로 점프
6. Target 정착     → box_near≈0.2, 조건 충족 시 putdown_r=0.2 반복
```

추가로 확인할 항목:

- Ground target과 elevated target을 나눠서 kicking 비율을 비교한다. Ground target에서 shortcut이 더
  강하면 putdown의 위치-only 판정과 잘 맞는 증거다.
- 변경한 입력에 Box의 z, 크기/BPS, 회전, 손-Box 상대 위치를 복원할 수 있는 정보가 있는지 확인한다.
- RSI 설정이 원본의 Carry skill 초기화 확률
  `loco_carry/pickUp/carryWith/putDown/omomo = 0.5/0.1/0.3/0.1/0.0`을 유지하는지 확인한다.
- 정상 carry와 kicking rollout의 `raw task`, `power`, `AMP`, 최종 PPO reward를 별도로 합산해 실제
  어느 경로의 return이 큰지 비교한다.

### 수정 방향 후보

아래는 진단 이후 검토할 후보이며 이 문서 작성 과정에서는 구현하지 않았다.

1. `r_box_near` 또는 Target 0.5 m pin을 `grasped`/`lifted` 조건으로 gate한다.
2. Target pin이 height gate를 덮어쓰지 않도록 조건 또는 적용 순서를 바꾼다.
3. `lifted_once`, 양손 contact, 안정적 grasp에 대한 중간 reward를 추가한다.
4. `r_putdown`에 `previously_lifted`, 낮은 Box 속도, support contact 또는 release 조건을 추가한다.
5. 완료 reward를 one-shot으로 만들거나 성공 후 종료하여 반복 reward farming을 줄인다.
6. 변경한 입력에 정밀 grasp에 필요한 Box geometry와 상대 pose가 보존되는지 먼저 검증한다.

---

## 18. 원본 Carry 최종 요약

```text
r_walk
→ 사람을 Box로 보냄

r_handheld + AMP
→ 손을 Box에 가져가고 pickup 모양을 만듦

r_box_vel
→ 들린 Box를 Target 방향으로 운반

r_box_near
→ Target 근처에서 정확한 3D 위치로 수렴

r_putdown
→ 수평 10cm·높이 1mm 위치 조건에 대한 반복 가능한 binary bonus
```

현재 reward는 정상적인 pickup-carry-putdown도 보상하지만, “반드시 손으로 들어서 옮겼다”는 history나
event를 확인하지 않는다. 그 결과 scratch 정책은 grasp가 어려울 때 Box를 발로 차서 `r_box_near`,
Target pin, `r_putdown`을 얻는 더 쉬운 경로를 선택할 수 있다.

---

## 19. F22 reward

이 절의 F22는 현재 steer 기준 체크포인트인 `f22_long_c20`을 뜻한다. `f22_long_ctrl`은 아래의
경로 이탈 벌점이 없는 대조군이므로 구분해야 한다.

### 19.1 한 줄 요약

F22는 원본 Carry의 목적을 유지하면서, 사람과 Box의 속도를 최종 목적지 직선 방향이 아니라
**각자 경로의 앞쪽 조준점 방향**으로 측정하고, 경로에서 옆으로 벗어나면 강한 음수 보상을 준다.

```text
접근: 사람은 1.5 m/s로 approach path를 따라 Box로 간다
운반: Box는 1.5 m/s로 transport path를 따라 Target으로 간다
공통: 경로 이탈은 latpen으로 벌한다
완료: box-near, handheld, putdown은 원본과 같다
```

F22는 가변속도 모델이 아니다. `STEER_MRAND=0`, `STEER_TIME=0`인 기준 설정에서 목표 속도는
항상 `1.5 m/s`다. `STEER_DUAL=1`은 사람용 창과 Box용 창을 모두 관측에 주는 설정일 뿐,
reward를 두 배로 만들거나 phase를 고르는 설정은 아니다.

### 19.2 실제 task reward 식

다음을 정의한다.

```text
V(v, u; 1.5) = exp[-5(1.5 - u·v)^2]
               단, u·v <= 0이면 0

L(d_lat) = exp(-0.5 d_lat^2) - 1
           경로 위에서는 0, 벗어날수록 음수, 멀리서는 -1에 수렴

N = exp[-10 ||Box - Target||_3D^2]
H = exp[-5  ||손 평균 - Box||_3D^2]
P = 정확한 putdown 조건의 0/1 값
```

`V_root`는 사람 속도를 사람 경로의 조준점 방향으로 투영한 값이고, `V_box`는 Box 속도를 Box
경로의 조준점 방향으로 투영한 값이다. F22 설정은 `STEER_POS=latpen`, `STEER_POS_C=2.0`,
`onlyVelReward=True`, walk/carry 바깥 배율 `2`이므로 실제 식은 다음과 같다.

```text
r_walk = 0.2 V_root + 2.0 L(d_lat_root)

r_carry = 0.2 V_box
          + 0.2 N
          + 2.0 L(d_lat_box)
          - BoxSpeedPenalty

R_task,F22 = 2 r_walk + 2 r_carry + 0.2 H + 0.2 P - PowerPenalty
```

완전히 펼치면 다음과 같다.

```text
R_task,F22 = 0.4 V_root
             + 0.4 V_box
             + 0.4 N
             + 0.2 H
             + 0.2 P
             + 4.0 L(d_lat_root)
             + 4.0 L(d_lat_box)
             - 2 BoxSpeedPenalty
             - PowerPenalty
```

경로 이탈과 과속 penalty가 0일 때 양의 task reward 최대치는 `1.6/step`이다. 학습기에 들어가는
최종 reward는 원본과 같이 다음 비율이다.

```text
R_train,F22 = 0.5 R_task,F22 + 0.5 R_AMP
```

즉 AMP 혼합 비율 자체는 동일하지만, F22 task reward의 크기와 음수 latpen이 원본보다 커서 실제
학습에서 style과 task가 체감하는 상대 크기는 원본 Carry와 완전히 같지는 않다.

### 19.3 Pin

F22도 두 개의 원본 pin을 유지한다.

```text
사람-Box XY < 0.5 m:
    V_root = 1
    root latpen = 0

Box-Target XY < 0.5 m:
    V_box = 1
    box latpen = 0
```

따라서 endpoint 0.5 m 안에서는 실제 속도와 경로 이탈을 더 따지지 않는다. 이때 F22의 바깥 배율까지
포함한 속도 pin 값은 각 항마다 `0.4/step`이다.

### 19.4 Latpen의 실제 강도

F22에서 각 다리의 최종 경로 벌점은 `4[exp(-0.5d^2)-1]`이다.

| 경로 횡방향 이탈 `d` | F22의 한 다리 latpen |
| ---: | ---: |
| 0.1 m | 약 -0.020 |
| 0.3 m | 약 -0.176 |
| 0.5 m | 약 -0.470 |
| 1.0 m | 약 -1.574 |
| 아주 멀리 | -4에 수렴 |

사람과 Box의 latpen은 동시에 계산된다. 다만 사람-Box 또는 Box-Target이 각자의 pin 반경 안이면
해당 latpen은 0이 된다. F22가 `c20`인 이유가 이 강한 경로 구속이다.

### 19.5 단계별로 실제로 무엇이 강한가

#### Box에 접근하기 전

주력은 `0.4 V_root`와 `4 L(d_lat_root)`다. 사람은 경로 방향으로 1.5 m/s에 맞춰 가면서 경로
옆으로 벗어나지 않아야 한다. Box가 정지해 있으면 `V_box=0`이고, Target이 멀면 near도 거의 0이다.

#### Box 0.5 m 안

사람 속도 reward가 pin되어 `0.4`가 되고 root latpen은 0이 된다. 손이 가까워질수록 `0.2H`와
AMP pickup style이 커진다. 그러나 grasp 성공 자체를 판정하는 별도 reward는 없다.

#### Box가 움직이기 시작할 때

F22 phase-free 구현에는 원본 Carry의 Box height gate가 들어 있지 않다. 따라서 Box가 아직 지면에
있어도 경로 방향으로 움직이면 `0.4V_box`를 받을 수 있다. 정상적으로 들고 운반해도 같은 reward를
받지만, 발로 차거나 밀어서 경로 방향 속도를 만든 경우도 reward 조건만 보면 통과한다.

이 점은 이후 masteer의 MS16에서 원본 height gate를 복구하며 명시적으로 수정됐다.

#### Target에 접근할 때

`0.4V_box`가 운반 속도를, `0.4N`이 정확한 3D 목표 위치 수렴을 담당한다. `N`은 멀리서는 매우
작고 Target 바로 근처에서 강해진다. Box 속도가 `2.5 m/s`를 넘으면 과속 penalty가 생긴다.

#### Target 0.5 m 안과 정확 배치

Box 속도는 실제 움직임과 무관하게 pin되어 `0.4`, box latpen은 0이 된다. 정확한 3D 위치에서는
near가 `0.4`에 가까워지고, XY 10 cm 및 z 1 mm 조건까지 맞으면 putdown `0.2`가 추가된다.

사람이 Box에서 멀어져 손·walk reward를 잃더라도 지면의 Box를 Target에 정확히 보내면 다음
`1.0/step`이 남을 수 있다.

```text
V_box pin = 0.4
box near  = 0.4
putdown   = 0.2
────────────────
합계       = 1.0 / step
```

그래서 F22는 원본 Carry보다도 지면 밀기·발차기 shortcut을 더 직접 강화할 여지가 있다.

### 19.6 Box 과속 penalty

임계 속도는 `2.5 m/s`, 계수는 `1.0`이다.

```text
Q = 1 - exp[-2(max(2.5, ||v_box||_3D) - 2.5)^2]
```

F22에서는 `Q`가 carry 안에 들어간 뒤 바깥 배율 2를 받으므로 최종 task reward에는 `-2Q`로
적용된다. 예를 들어 Box 속도가 3.0 m/s면 약 `-0.787`, 4.0 m/s면 약 `-1.978`이다. 발로 너무
세게 차는 행동은 이 항으로 벌하지만, 2.5 m/s 이하의 밀기나 발차기는 이 항이 잡지 못한다.

---

## 20. MS18 reward

이 절은 `ms18_maskteam_origscale_c06_s0`의 저장된 sidecar 설정을 기준으로 한다.

### 20.1 한 줄 요약

MS18은 F22 계열의 경로 추종 reward를 두 명의 agent에 적용하되,

```text
고정 1.5 m/s → 경로 구간별 0.375 / 0.75 / 1.125 / 1.5 m/s
F22 최종 lat 계수 4.0 → MS18 최종 lat 계수 0.6
walk/carry 바깥 배율 2 → 1
누락됐던 Box height gate → 복구
```

한 모델이다. `MS18`에서 새로 바뀐 것은 teammate 토큰의 exact attention mask이며, reward와
가변속도는 MS17과 동일하다.

### 20.2 가변속도 명령이 만들어지는 방식

경로를 네 구간으로 나누고, 각 구간마다 다음 네 배수 중 하나를 무작위로 뽑는다.

```text
MS_MRAND = 4
MS_M_LO  = 0.25

배수       = {0.25, 0.50, 0.75, 1.00}
창 길이 M = 2.4 m × 배수
           = {0.6, 1.2, 1.8, 2.4} m

목표 속도 v_cmd = M / 1.6 s
                 = {0.375, 0.75, 1.125, 1.5} m/s
```

`MS_MRAND=4`가 속도 종류가 네 개라는 뜻은 아니다. 구현상 **경로를 네 구간으로 나눈다**는 뜻이고,
각 구간이 위 네 속도 중 하나를 독립적으로 고른다. 이웃 구간에서 같은 속도가 다시 나올 수도 있다.

정책에는 6개의 local waypoint가 주어진다. 전체 창 길이가 `M`이므로 인접 waypoint 간격은 다음과
같다.

| `M` | 점 간격 `M/6` | reward 목표 속도 `M/1.6` |
| ---: | ---: | ---: |
| 0.6 m | 0.1 m | 0.375 m/s |
| 1.2 m | 0.2 m | 0.750 m/s |
| 1.8 m | 0.3 m | 1.125 m/s |
| 2.4 m | 0.4 m | 1.500 m/s |

따라서 별도의 scalar speed command가 붙는 것이 아니라 **눈앞의 경로 점들이 촘촘하면 천천히,
멀찍이 놓이면 빨리** 가라는 명령이다. reward는 같은 `M`으로 목표 속도를 계산하므로 관측과 정답이
연결된다. 속도는 매 physics step마다 랜덤으로 바뀌는 것이 아니라, episode reset 때 만든 구간별
프로파일을 경로 진행에 따라 읽는다.

### 20.3 실제 task reward 식

MS18의 속도 kernel은 다음과 같다.

```text
V(v, u; v_cmd) = exp[-5(v_cmd - u·v)^2]
                 단, u·v <= 0이면 0
```

사람과 Box는 각자의 현재 경로 위치에서 서로 다른 `M`을 읽을 수 있다. 사람은 approach 경로의
조준점과 속도 명령을, Box는 transport 경로의 조준점과 속도 명령을 사용한다.

```text
r_walk,MS18 = 0.2 V_root(v_cmd_root)
               + 0.6 L(d_lat_root)

r_carry,MS18 = 0.2 V_box,gated(v_cmd_box)
                + 0.2 N
                + 0.6 L(d_lat_box)
                - BoxSpeedPenalty

R_task,MS18 = r_walk,MS18
              + r_carry,MS18
              + 0.2 H
              + 0.2 P
              - PowerPenalty
```

즉 완전히 펼치면 다음과 같다.

```text
R_task,MS18 = 0.2 V_root
              + 0.2 V_box,gated
              + 0.2 N
              + 0.2 H
              + 0.2 P
              + 0.6 L(d_lat_root)
              + 0.6 L(d_lat_box)
              - BoxSpeedPenalty
              - PowerPenalty
```

이탈·과속·power penalty가 없을 때 양의 task reward 최대치는 원본 Carry와 같은 `1.0/step`이다.
최종 PPO reward는 다음과 같다.

```text
R_train,MS18 = 0.5 R_task,MS18 + 0.5 R_AMP
```

### 20.4 속도 reward의 민감도

속도 오차 `Δv = v_cmd - v_parallel`에 대한 kernel과 MS18의 가중 결과는 다음과 같다.

| `|Δv|` | kernel `exp(-5Δv²)` | MS18 속도 reward `0.2×kernel` |
| ---: | ---: | ---: |
| 0.0 m/s | 1.000 | 0.200 |
| 0.1 m/s | 0.951 | 0.190 |
| 0.2 m/s | 0.819 | 0.164 |
| 0.3 m/s | 0.638 | 0.128 |
| 0.5 m/s | 0.287 | 0.057 |
| 1.0 m/s | 0.0067 | 0.0013 |

예를 들어 현재 명령이 `0.375 m/s`인데 실제로 경로 방향 `0.75 m/s`로 가면 속도 reward는 약
`0.099`다. 정확히 `0.375 m/s`로 맞추면 `0.2`를 받는다. 반대 방향이나 완전 정지는 현재 MS18의
네 양수 속도 명령에서 `0`이다.

### 20.5 복구된 Box height gate

MS18은 F22와 달리 원본 Carry의 height gate를 명시적으로 복구했다.

```text
Box center z <= Box height/2 + 0.2 m
    → V_box = 0

Box center z > Box height/2 + 0.2 m
    → 명령 속도에 따른 V_box 사용
```

그 뒤에 Box-Target XY 0.5 m pin을 적용한다. 즉 순서는 다음과 같다.

```text
1. 경로 방향 가변속도 reward 계산
2. Box가 낮으면 0으로 gate
3. Box가 Target XY 0.5 m 안이면 최종적으로 1로 pin
```

따라서 Target에서 멀리 있는 지면 Box를 차서 보내는 동안에는 transport 속도 reward `0.2V_box`를
받지 못한다. 하지만 `box-near`에는 lift/grasp gate가 없고, Target 0.5 m pin은 height gate 뒤에
적용되므로 발차기 shortcut이 완전히 사라진 것은 아니다.

### 20.6 단계별로 실제로 무엇이 강한가

#### Box에 접근하기 전

주력은 `0.2V_root(v_cmd)`와 `0.6L(d_lat_root)`다. 정책은 waypoint 간격을 읽어 현재 구간의
속도에 맞춰 사람을 이동시켜야 한다. Box가 정지해 있고 Target이 멀면 carry 쪽 양의 reward는 거의
없다.

#### Box 0.5 m 안과 pickup 구간

사람 속도 reward는 `0.2`로 pin되고 root latpen은 0이 된다. 손이 가까워지면 `0.2H`와 AMP style이
작동한다. 여전히 실제 contact/grasp/lift event 자체에 대한 별도 bonus는 없다.

#### Box를 들어 운반할 때

Box가 height gate를 넘은 뒤부터 `0.2V_box(v_cmd)`가 본격적으로 작동한다. 이때 사람의 현재 구간이
아니라 **Box 자신의 transport 경로 진행 위치**에서 속도 명령을 읽는다. `0.2N`은 Target이
가까워질 때 점차 강해지고, `0.6L(d_lat_box)`은 운반 경로 이탈을 벌한다.

#### Target에 도달할 때

Box-Target XY 0.5 m 안이면 Box 속도 reward는 `0.2`로 pin되고 box latpen은 0이다. 정확한 3D
위치에서 near가 `0.2`, 엄격한 XY/z 조건에서 putdown이 `0.2`가 된다. 사람이 멀어져도 지면의
Box가 정확한 Target에 남아 있으면 원본과 같은 최대 `0.6/step`이 남을 수 있다.

```text
V_box pin = 0.2
box near  = 0.2
putdown   = 0.2
────────────────
합계       = 0.6 / step
```

### 20.7 MS18의 latpen과 F22 비교

MS18에서 각 다리의 최종 latpen 계수는 `0.6`이다.

| 경로 이탈 `d` | F22 한 다리 | MS18 한 다리 |
| ---: | ---: | ---: |
| 0.1 m | -0.020 | -0.003 |
| 0.3 m | -0.176 | -0.026 |
| 0.5 m | -0.470 | -0.071 |
| 1.0 m | -1.574 | -0.236 |
| 아주 멀리 | -4.0 | -0.6 |

MS18의 `MS_REWARD_OUTER=1`, `MS_POS_C=0.6`은 집기·운반 신호가 경로 이탈 벌점에 묻히지 않게
원본 reward scale 쪽으로 되돌린 설정이다.

### 20.8 Multi-agent와 teammate mask는 reward가 아님

MS18은 agent가 두 명이지만 저장된 설정은 다음과 같다.

```text
MA_TOKEN = mask
MA_C     = 0
MA_BETA  = 0
MS_SCEN  = free
```

- teammate 21-D 토큰은 관측 크기에는 남지만 actor Transformer의 attention key/value에서
  제외된다.
- 이 mask는 reward를 변경하지 않는다. MS18과 MS17의 reward 식은 같다.
- team reward 계수 `MA_C`, `MA_BETA`가 모두 0이라 `_apply_team_reward()`는 아무 효과가 없다.
- 따라서 각 agent의 직접 task reward는 자기 사람, 자기 Box, 자기 Target, 자기 path에 대한 Carry
  reward다. MS18 자체에 별도의 충돌 회피나 협동 bonus가 추가된 것은 아니다.

---

## 21. F22와 MS18 핵심 비교

| 항목 | F22 `f22_long_c20` | MS18 `ms18_maskteam_origscale_c06_s0` |
| --- | --- | --- |
| agent 수 | 1 | 2 |
| 속도 명령 | 고정 1.5 m/s | 구간별 0.375/0.75/1.125/1.5 m/s |
| 속도 전달 | 6점 경로 창의 방향 | 6점 경로 창의 방향과 간격 |
| walk/carry 바깥 배율 | 2 | 1 |
| 최종 latpen 계수/다리 | 4.0 | 0.6 |
| Box height gate | phase-free 코드에 없음 | 원본 순서로 복구됨 |
| 속도항 최대/다리 | 0.4 | 0.2 |
| near 최대 | 0.4 | 0.2 |
| handheld/putdown 최대 | 각각 0.2 | 각각 0.2 |
| 양의 task reward 최대 | 1.6 | 1.0 |
| AMP 혼합 | task 0.5 + AMP 0.5 | task 0.5 + AMP 0.5 |
| teammate 정보 | 해당 없음 | actor attention에서 mask |
| 명시적 team reward | 해당 없음 | 계수 0이라 없음 |

가장 짧게 정리하면 다음과 같다.

```text
F22  = 강한 경로 구속 + 고정속도 + 지면 Box 속도 reward 가능
MS18 = 약한 경로 구속 + 구간별 가변속도 + 원본 Box 높이 gate 복구
```
