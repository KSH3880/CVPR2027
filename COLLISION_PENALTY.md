# Collision Penalty 계산 방식

현재 사람–사람 구현: `tokenhsi/env/tasks/multi_agent/collision_reward.py`, 적용 경로: `humanoid_ma_carry.py` → `edge_context_task.py`. `agentCollisionMode` 기본은 `static`, Shared9 CPA 전용 config는 `cpa`다. 아래 Agent–Object와 crossing scenario 항목은 현재 미구현 설계이며, 사람–상자 보상 패널티는 없다.

## 공통 사항

- 모든 거리/속도는 **XY 평면**(root 위치 `[0:2]`, root 선속도 `[7:9]`)만 사용한다.
- 에이전트 $i$의 violation $V_i \in [0, 1]$은 상대 엔티티 $j$들에 대한 값의 **최댓값**이다.
- 보상: $r_i^{\text{col}} = -k \cdot V_i$

| 파라미터 | config key | 기본값 |
|---|---|---|
| $d_{\min}$ | `agentCollisionDist` | 0.7 |
| $k$ (agent) | `agentCollisionCoeff` | 0.5 |
| $\gamma$ | `agentCollisionTTCDiscount` | 0.99 |
| $k$ (object) | `otherObjectCollisionCoeff` | 1.0 |
| on/off | `agentCollisionPenalty` / `otherObjectCollisionPenalty` | True / False |

## 1. 정적 거리 페널티

`compute_agent_collision_penalty`

$$
d_{ij} = \lVert \mathbf{p}_i - \mathbf{p}_j \rVert_{xy}, \qquad
V_i = \max_{j \ne i} \frac{\max(d_{\min} - d_{ij},\ 0)}{d_{\min}}
$$

- 현재 root XY 거리만 보며, $d_{ij} < d_{\min}$일 때 선형으로 증가 (root XY 위치가 일치하면1; 실제 형상 접촉 여부를 측정하지 않음).

## 2. CPA(Closest Point of Approach) 페널티

현재 사람–사람은 `compute_agent_cpa_collision_penalty`; cross-entity/Agent–Object 확장은 아래 설계식만 정의한다.

상대 위치/속도:

$$
\mathbf{p} = \mathbf{p}_i - \mathbf{p}_j, \qquad \mathbf{v} = \mathbf{v}_i - \mathbf{v}_j
$$

각 항:

$$
\begin{aligned}
c_{ij} &= \mathrm{clip}\!\left(\frac{-\mathbf{p}\cdot\mathbf{v}}{\lVert\mathbf{p}\rVert\,\lVert\mathbf{v}\rVert},\ 0,\ 1\right) && \text{(closing: 접근 방향 정도)} \\
t^*_{ij} &= \max\!\left(\frac{-\mathbf{p}\cdot\mathbf{v}}{\lVert\mathbf{v}\rVert^2},\ 0\right) && \text{(최근접 시각)} \\
d^{\text{cpa}}_{ij} &= \lVert \mathbf{p} + \mathbf{v}\, t^*_{ij} \rVert && \text{(최근접 거리)} \\
r_{ij} &= \frac{\max(d_{\min} - d^{\text{cpa}}_{ij},\ 0)}{d_{\min}} && \text{(distance risk)} \\
u_{ij} &= \gamma^{\,t^*_{ij} / \Delta t} && \text{(urgency)}
\end{aligned}
$$

$$
V_i = \max_{j \in \mathcal{M}_i} \; c_{ij}\, r_{ij}\, u_{ij}
$$

- 분모는 `1e-6`으로 clamp.
- 멀어지거나($\mathbf{p}\cdot\mathbf{v} \ge 0$) 상대속도가 0이면 $c_{ij}=0$ → **이미 겹쳐 있어도 페널티 0**.
- 충돌이 임박할수록($t^*$ 작을수록) $u_{ij} \to 1$.

### 2a. Agent–Agent

`compute_agent_cpa_collision_penalty`: $j$ = 다른 휴머노이드, $\mathcal{M}_i = \{ j \ne i \}$.

### 2b. Agent–Other object

`compute_agent_object_cpa_collision_penalty`: $j$ = box, 속도는 box 선속도. 마스크 $\mathcal{M}_i$(`_get_other_object_collision_mask`)에서 자신에게 할당된 box는 제외.

## 현재 실행 설정

Shared9 CPA는 `agentCollisionMode: cpa`, `agentCollisionPenalty: true`, `agentCollisionCoeff: 0.5`, `agentCollisionDist: 0.7`, `agentCollisionTTCDiscount: 0.99`를 사용한다. 일반 config는 기본 `static`이다. CPA는 기존 정적 항을 교체하며 둘을 합산하지 않는다. 시간 할인은 제어 step의 `dt`를 사용하고, 사람당 위험도의 최댓값을 한 번만 지불한다.

Collision은 edge 보상 공유 이후 task에 더해진다. Task/AMP 혼합0.5/0.5·reward shaper1에서는 `-0.5 V`가 최종 보상에 `-0.25 V`로 반영된다. 정지·멀어짐·root XY 일치 시 CPA는0일 수 있으며 물리 형상 관통 벌점이 아니다. 실행 명령은 [Stage 2 가이드](markdowns/config_stage2.md)에 있다.

## Crossing/Agent–Object 설계 모드 (현재 미구현)

| `collisionScenario.mode` | Agent–Agent | Agent–Object |
|---|---|---|
| `none` | 정적 거리 (1) | CPA (2b)* |
| `two_agent_crossing`, `two_agent_crossing_ontop` | CPA (2a) | CPA (2b)* |

\* `otherObjectCollisionPenalty: True` 이고 계수 > 0 일 때만.
Agent–Agent 항은 `agentCollisionPenalty: True` 이고 $M > 1$ 일 때만.

## 로깅용 거리 (보상 아님)

crossing 모드에서 env별 휴머노이드 쌍의 최소 XY 거리 $d^{\text{env}}_{\min}$를 기록한다.
이벤트 발생 후 $d^{\text{env}}_{\min} < d_{\min}$이면 proximity로 카운트.
