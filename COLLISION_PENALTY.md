# Collision Penalty 계산 방식

구현: `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` (`_compute_reward`, `compute_*_collision_penalty`)

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

- 현재 거리만 보며, $d_{ij} < d_{\min}$일 때 선형으로 증가 (접촉 시 1).

## 2. CPA(Closest Point of Approach) 페널티

`compute_cross_entity_cpa_collision_penalty`

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

## 모드별 적용

| `collisionScenario.mode` | Agent–Agent | Agent–Object |
|---|---|---|
| `none` | 정적 거리 (1) | CPA (2b)* |
| `two_agent_crossing`, `two_agent_crossing_ontop` | CPA (2a) | CPA (2b)* |

\* `otherObjectCollisionPenalty: True` 이고 계수 > 0 일 때만.
Agent–Agent 항은 `agentCollisionPenalty: True` 이고 $M > 1$ 일 때만.

## 로깅용 거리 (보상 아님)

crossing 모드에서 env별 휴머노이드 쌍의 최소 XY 거리 $d^{\text{env}}_{\min}$를 기록한다.
이벤트 발생 후 $d^{\text{env}}_{\min} < d_{\min}$이면 proximity로 카운트.
