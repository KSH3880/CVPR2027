# Cuboid PUSH 물성·리워드 설계

기준: `stage1_v2_fixed_push_360_raw_cuboid`의 현재 설정과 코드.

## 물성

| 항목 | 설정 |
| --- | --- |
| 크기 X | 1.2~1.7m 균등 추출 |
| 크기 Y/Z | 1.45~1.7m에서 두 값을 균등 추출한 뒤 Y ≥ Z로 정렬 |
| 질량 | 부피에 선형 대응하는 15~20kg |
| 본체 마찰 | 0.9 |
| 밑판 마찰 | 0.3 |
| 밑판 형상 | 두께 1cm, 상자와 같은 가로·세로 |
| 지면 정지/동마찰 | 1.0 / 1.0 |
| 상자·지면 반발계수 | 0 |
| 선형/각 감쇠 | 0.01 / 0.01 |

```text
V = X × Y × Z
m = 15 + 5 × (V - 2.523) / (4.913 - 2.523)  [kg]
```

크기·질량은 환경 생성 시 정하며 reset 때 유지한다. 본체와 밑판은 하나의 강체에 속한 두 충돌 형상이다. 질량 중심은 상자 중심, 관성은 전체 직육면체 기준이다.

```text
Ixx = m(Y² + Z²)/12
Iyy = m(X² + Z²)/12
Izz = m(X² + Y²)/12
```

## 리워드

사람마다 담당 상자에 대한 **PUSH 접촉 edge와 AT 목표 도달 edge**를 함께 계산한다.

### 1. PUSH 접촉

상자 로컬 `-Y` 면을 지정 접촉 면으로 사용한다. `n`은 상자 로컬 `+Y`의 세계 방향이고, `d_R`, `d_L`은 각 손의 XY 위치에서 지정 면의 XY 투영까지의 최소 거리다.

```text
φ_PUSH = exp(-10 × max(d_R, d_L)²)
p_PUSH = 1 / (1 + D_route)
```

`D_route`는 사람 root에서 지정 면 뒤 1m 지점까지의 접근 경로 길이다. 상자 외곽에 0.45m 여유를 두고 모서리를 우회하는 경로를 계산한다. 접촉 점수에는 손 중점·손 높이 오차를 사용하지 않는다.

유효 접촉 조건은 다음과 같다. `f_R`, `f_L`, `f_box`는 양손과 상자의 net contact force다.

```text
C = (d_R ≤ 0.18m) AND (d_L ≤ 0.18m)
    AND (f_R · (-n) ≥ 0.5N) AND (f_L · (-n) ≥ 0.5N)
    AND (f_box · n ≥ 0.5N)
```

한 제어 step의 두 simulate 호출 중 한 번이라도 만족하면 해당 step의 `C=1`이다.

### 2. AT 목표 도달

`c`는 상자 중심, `g`는 목표 중심이다.

```text
φ_AT = exp(-10 × ||c - g||²)
p_AT = 1 / (1 + max(||c_xy - g_xy|| - 0.1, 0))
A    = (φ_AT ≥ 0.9) AND (|c_z - g_z| ≤ 0.001m)
```

목표 성공은 중심 거리 약 10.3cm 이내·높이 오차 1mm 이하다. 회전·속도·손 접촉은 별도 성공 조건에 포함하지 않는다.

### 3. 상태·접근도·성공 보상 합산

```text
s_PUSH = C OR A
s_AT   = A

R_e = 0.2 × (s_e이면 1, 아니면 φ_e)
    + 0.2 × (s_e이면 1, 아니면 p_e)
    + 0.2 × s_e

담당 edge 합 = R_PUSH + R_AT    # 최대 1.2
```

각 edge는 포화 전 `0.2φ + 0.2p`, 포화되면 0.6이다. 목표 도달 시 두 edge가 모두 포화된다. 성공 보상은 현재 조건을 만족하는 매 step에 지급하며, 과거 성공만으로 영구 유지하지 않는다.

### 4. 페널티와 AMP 합산

`τ`, `q̇`는 관절 force/torque와 속도, `d_human`은 두 사람 root의 XY 거리다.

```text
R_power     = -0.0005 × Σ_j |τ_j × q̇_j|
R_collision = -0.5 × max(0.7 - d_human, 0) / 0.7
R_box_speed = 0

R_env   = R_task + R_power + R_collision
D       = sigmoid(discriminator_logit)
R_AMP   = 2 × [-ln(max(1 - D, 0.0001))]
R_train = 0.5 × R_env + 0.5 × R_AMP
```

`R_task`는 런타임이 산출한 task reward다. PUSH의 사람 간 페널티는 현재 거리식이며, 상자 속도 페널티는 최종 보상에서 제거된다. 낙상은 별도 상수 감점 없이 조기 종료로 처리한다.

근거: [환경 설정](../../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid.yaml), [학습 설정](../../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw_cuboid.yaml), [상자 샘플링·접근 거리](../../tokenhsi/utils/push_navigation.py), [밑판 물성](../../tokenhsi/utils/push_box_asset.py), [접촉·edge 보상](../../tokenhsi/env/tasks/multi_agent/push_reward.py), [환경 보상 적용](../../tokenhsi/env/tasks/multi_agent/humanoid_ma_stage1_push.py), [파워·근접 페널티 합산](../../tokenhsi/env/tasks/multi_agent/edge_context_task.py), [AMP 보상](../../tokenhsi/learning/amp_agent.py).
