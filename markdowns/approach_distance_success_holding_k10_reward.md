# approach_distance_success_holding_k10 보상 수식

**이 config에서는 Holding과 At의 상태함수가 계수까지 동일하다. Progress도 수식과 계수가 동일하다.** 관계별 입력 좌표와 선행 gate는 다르므로, 두 edge의 최종 보상이 항상 같다는 뜻은 아니다.

수식은 렌더링 문제를 피하기 위해 모두 일반 코드 블록으로 적었다. 이 문서는 현재 Holding·At 실험에 대한 설명이며 OnTop은 아직 포함하지 않는다.

## 원본과 달라진 설정

| 항목 | 원본 approach_distance_success | 새 holding_k10 |
| --- | --- | --- |
| Holding 상태함수 k | 5 | **10** |
| At 상태함수 k | 10 | 10 |
| 상태 보상 가중치 | 0.2 | 동일 |
| Progress 보상 가중치 | 0.2 | 동일 |
| Progress delta / sigma | 0.5m / 1.0m | 동일 |
| Gate beta / center | 30 / 0.8 | 동일 |
| 상태 만족 기준 | 0.9 | 동일 |
| 현재 성공 조건 | At 만족 + Z 오차 ≤ 1mm | 동일 |
| 현재 성공 시 포화·추가 보상 | 두 edge 합 0.8 + 추가 0.2 | 동일 |

YAML 설정값은 `env.relationReward.holding.hand_distance_scale` 한 항목만 바꿨다. 기존 k=5 실험 파일은 유지한다.

## 1. 관계 그래프

각 에이전트 i의 관계는 다음과 같다.

```text
Hi ──Holding──▶ Oi ──At──▶ Gi

Hi: 에이전트
Oi: 해당 에이전트에게 할당된 상자
Gi: 해당 상자 중심이 도달해야 하는 목표 좌표
```

## 2. 상태함수: Holding·At 완전히 같은 형태와 계수

```text
state_error_squared
    = (source.x - target.x)²
    + (source.y - target.y)²
    + (source.z - target.z)²

phi = exp(-10 × state_error_squared)
```

| 관계 | source_state | target_state |
| --- | --- | --- |
| Holding | 오른손·왼손 위치의 평균 | 할당된 상자 중심 |
| At | 할당된 상자 중심 | 목표 좌표 |

풀어 쓰면 다음과 같다. 모든 위치의 단위는 m다.

```text
hand_midpoint = (right_hand_position + left_hand_position) / 2

phi_H(t) = exp(-10 × ||hand_midpoint(t) - object_center(t)||²)
phi_A(t) = exp(-10 × ||object_center(t) - goal_position(t)||²)
```

여기의 거리는 **XYZ를 모두 포함한 3D 거리**다. Holding 상태함수에는 사람 root가 들어가지 않는다.

두 상태 모두 `phi >= 0.9`가 만족 기준이다. k=10에서 이는 두 입력점 사이의 거리가 약 **10.265cm 이하**라는 뜻이다. Holding 점수는 양손 평균 위치를 보는 대리 지표이며 실제 파지 접촉 자체를 판정하지는 않는다.

## 3. Progress: Holding·At 완전히 같은 형태와 계수

```text
d_xy = sqrt((source.x - target.x)² + (source.y - target.y)²)

P = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
```

| 관계 | source_progress | target_progress |
| --- | --- | --- |
| Holding | 사람 root | 할당된 상자 중심 |
| At | 할당된 상자 중심 | 목표 좌표 |

```text
d_H(t) = ||human_root(t).xy - object_center(t).xy||
d_A(t) = ||object_center(t).xy - goal_position(t).xy||

P_H(t) = 1 / (1 + max(d_H(t) - 0.5, 0) / 1.0)
P_A(t) = 1 / (1 + max(d_A(t) - 0.5, 0) / 1.0)
```

- XY 거리 0.5m 이내에서는 P=1이다.
- Z, 속도, 이동 방향은 이 progress 식에 들어가지 않는다.
- 이름은 progress이지만 이전 step보다 가까워진 양이 아니라 **현재 거리 점수**다.

## 4. Gate: 관계별 적용이 다름

공통 soft gate 함수:

```text
sigmoid(x) = 1 / (1 + exp(-x))

g(phi) = sigmoid(30 × (phi - 0.8))
```

현재 edge의 보상 활성화 계수는 다음과 같다. t는 보상을 계산할 현재 step이다.

```text
G_Holding(t) = 1
G_At(t)      = g(phi_H(t-1))
```

Holding에는 선행 edge가 없고, At에는 Holding이 선행 조건이다. At의 gate는 **이전 step Holding 상태**로 계산한다. At 자신의 점수로 At 보상을 gate하지 않는다.

`phi >= 0.9` 만족 판정과 gate는 별개다. phi=0.8에서 gate=0.5, phi=0.9에서 gate≈0.953이며 0.9 미만이라고 보상이 갑자기 0이 되지는 않는다.

## 5. 현재 성공 전의 edge 보상

공통 형태:

```text
r_edge_raw(t) = G_edge(t) × (0.2 × phi_edge(t) + 0.2 × P_edge(t))
```

관계별로 쓰면:

```text
r_H_raw(t) = 0.2 × phi_H(t) + 0.2 × P_H(t)

r_A_raw(t) = g(phi_H(t-1)) × (0.2 × phi_A(t) + 0.2 × P_A(t))
```

즉 **상태함수와 progress는 같지만, 입력과 G가 달라 최종 edge 보상은 다를 수 있다.**

## 6. 현재 성공 조건과 성공 중 포화

각 에이전트의 현재 성공 조건:

```text
z_error(t) = abs(object_center(t).z - goal_position(t).z)

S(t) = [phi_A(t) >= 0.9] AND [z_error(t) <= 0.001]
```

Holding 만족 여부, 과거 achieved/done 이력은 이 현재 성공 조건에 포함하지 않는다. 성공한 step에는 gate 계산 후 실제 지급 보상을 덮어쓴다.

```text
if S(t):
    r_H_paid(t) = 0.4
    r_A_paid(t) = 0.4
    current_success_bonus(t) = 0.2
else:
    r_H_paid(t) = r_H_raw(t)
    r_A_paid(t) = r_A_raw(t)
    current_success_bonus(t) = 0

relation_task_reward(t)
    = r_H_paid(t) + r_A_paid(t) + current_success_bonus(t)
```

성공 중 relation task reward는 에이전트당 **1.0**이다. 성공에서 벗어나면 즉시 원래 계산으로 돌아온다. 과거 성공 이력이 포화를 계속 유지하지 않으며, first-success 보너스는 0이다.

포화는 지급 보상만 바꾼다. 실제 phi, gate, satisfied, achieved, done 관측은 그대로 유지한다.

## 7. 기존 페널티와 학습 설정

```text
task_reward
    = relation_task_reward
    + power_penalty
    + agent_collision_penalty
    + box_speed_penalty
```

이 페널티와 AMP·네트워크·reset 설정은 원본과 같다. 위 1.0은 페널티와 AMP 결합 이전의 relation task reward다.

## 파일과 실행

- 새 config: [approach_distance_success_holding_k10.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success_holding_k10.yaml)
- 학습: [approach_distance_success_holding_k10_train.sh](../tokenhsi/scripts/multi_agent/approach_distance_success_holding_k10_train.sh)
- 평가: [approach_distance_success_holding_k10_test.sh](../tokenhsi/scripts/multi_agent/approach_distance_success_holding_k10_test.sh)
- 수식 구현: [relation_reward.py](../tokenhsi/env/tasks/multi_agent/relation_reward.py)
- 입력 좌표 구성: [relation_task.py](../tokenhsi/env/tasks/multi_agent/relation_task.py)
- 실행·checkpoint 규칙: [config.md](config.md#비교-실험-holding-k10)

```bash
cd /home/hwanhee/ksh/approach_distance_success
nvidia-smi -i 4

TOKENHSI_GPU=4 TOKENHSI_CONDA_ENV=tokenhsi \
RESUME_CHECKPOINT= MAX_ITERATIONS= OUTPUT_PATH=output/approach_distance_success_holding_k10 \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_holding_k10_train.sh 2 2048 3
```

새로 학습하는 명령이다. 기존 k=5 checkpoint는 reward config가 달라 이 실험에 직접 resume/evaluate할 수 없다.
