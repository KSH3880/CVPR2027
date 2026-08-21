# Method V2 — Shared World Modeling for Role-Aware Multi-Agent HOI

> Notion `2. Methodology Draft · V2` (+ V2 Details)의 로컬 사본. 매번 링크 열지 않고 여기서 바로 읽는다.
> Notion을 고치면 이 파일도 같이 갱신할 것. 레포별 실행 계획은 [REPOS.md](REPOS.md).

각 agent의 role·skill·skill goal이 주어진 상태에서, Shared World Model이 coordinated future
trajectory candidates를 생성하고 Planner가 dependency·collision/interference·team task completion을
고려해 best future trajectory를 선택한다. 선택된 trajectory는 고정된 Trajectory-to-Steering rule을 통해
현재 Steering Target으로 변환되고, TokenHSI-based Multi-Agent Action Policy가 이를 실제 full-body
motion으로 실행한다.

```
Task / Role Specification → Shared World Model → Future Trajectory Planning
→ Trajectory-to-Steering → Multi-Agent Action Policy → Physics Simulator → Replanning
```

---

## 1. Overall Framework

```
High-Level Team Task
        ↓
Task / Role Specification
        ↓
Shared World Model
        ↓
Coordinated Future Trajectory Candidates  τ¹, τ², ... τᴷ
        ↓
Planner → Best Future Trajectory τ*
        ↓
Trajectory-to-Steering Controller
        ↓
Current Steering Target g_t
        ↓
TokenHSI-based Multi-Agent Action Policy
        ↓
Physics Simulator
        ↓
Actual Shared State → Replanning
```

### Key idea

- **Skill Goal**은 현재 skill segment가 최종적으로 달성해야 하는 objective다.
- World Model의 planning representation은 **future trajectory**이며, Steering Target을
  WM input이나 planning variable로 두지 **않는다**.
- Planner는 여러 coordinated future trajectories를 직접 평가해 best trajectory τ*를 선택한다.
- 선택된 τ*는 deterministic Trajectory-to-Steering Controller를 통해 현재 Steering Target g_t로 변환된다.
- Action Policy는 trajectory 전체를 보지 않고, TokenHSI와 동일한 target interface로 g_t를 실행한다.
  다만 policy 자체는 Frozen TokenHSI가 아니라 **TokenHSI-based Multi-Agent Action Policy**로 추가 학습된다.
- Simulator가 actual next state를 만들면 그 상태에서 다시 prediction / planning한다.

### 1.1 End-to-End I/O

```
STATIC / TASK CONDITION
──────────────────────────────────────────────
Per-agent Role / Interaction Object
Per-agent Final Goal or Goal Constraint  G_i
Task Dependency / Completion Condition   D
Current Skill Structure                  Skill
                    ↓
ACTUAL SHARED STATE FROM SIMULATOR
──────────────────────────────────────────────
S_t
├─ Human A current state / root state
├─ Human B current state / root state
├─ Task-relevant object states
└─ Scene geometry
                    ↓
SHARED WORLD MODEL
──────────────────────────────────────────────
Input : S_t + Current Skill / Skill Goal G + Task Dependency D + Scene condition
Output: K coordinated future trajectory candidates τ^1_(t+1:t+H) ... τ^K_(t+1:t+H)
        each candidate includes
        ├─ Human A/B future root trajectories
        ├─ Object future trajectories
        └─ Future task / event states (corridor_clear, B_enter_corridor, progress, ...)
                    ↓
PLANNER
──────────────────────────────────────────────
Input   : { τ^k } for all k + Skill Goal / Task Condition G + Dependency D
Evaluate: dependency / team-task progress / collision-interference /
          valid placement / completion time
Output  : Best future trajectory τ*
                    ↓
TRAJECTORY-TO-STEERING CONTROLLER
──────────────────────────────────────────────
Input : selected trajectory τ* + current actual position/state
Rule  : nearest point on τ* → path lookahead point → current Steering Target g_t
        → near Skill Goal, converge/switch g_t to Skill Goal
Output: Current Steering Target g_t
                    ↓
        ┌───────────┴───────────┐
        ▼                       ▼
AGENT A ACTION POLICY    AGENT B ACTION POLICY
─────────────────────    ─────────────────────
actual self state        actual self state
actual object state      actual object state
teammate context         teammate context
Current Skill_A          Current Skill_B
Steering Target g_A,t    Steering Target g_B,t
        ▼                       ▼
Full-body action a_A,t   Full-body action a_B,t
        └───────────┬───────────┘
                    ▼
PHYSICS SIMULATOR
──────────────────────────────────────────────
Input : S_t + {a_A,t, a_B,t}
Output: Actual S_(t+1)  ────────→ REPLAN from S_(t+1)
```

**핵심 시간축**: WM/Planner는 t에서 t+1 … t+H의 future trajectory를 보고 판단하지만,
Action Policy는 trajectory 전체를 입력받지 않는다. 선택된 τ*에서 현재 Steering Target g_t를
계산해 실행하고, Simulator가 만든 actual S_(t+1)에서 다시 prediction / planning한다.

---

## 2. Task & Coordination Formulation

### 2.1 Task / Role Specification

초기 setting에서는 semantic task structure를 사전에 제공하여 coordination 문제에 집중한다.
Agent role / Skill segment / Interaction object / Skill goal 또는 goal constraint /
Completion condition / Inter-agent dependency.

**Example: Corridor-Clearing Task**
- Agent A: `Carry(Obstacle Box)` / Skill goal = clear-passage placement **region**. 정확한 placement point는 사전에 고정하지 않음
- Agent B: `Carry(Task Box)` / Skill goal = fixed `G_B`

B는 자신의 box를 정해진 goal까지 운반한다. A는 통로를 막는 obstacle box를 옮겨 B가 지나갈 수 있게
해야 하지만, 정확히 어디에 놓을지는 사전에 주어지지 않는다.

### 2.2 State-based Dependency

Dependency는 반드시 skill-to-skill relation일 필요가 없다. 한 agent의 행동이 만든 shared-world
state가 다른 agent의 실행 가능성을 결정할 수 있다.

```
A removes obstacle → corridor_clear = True → B can pass through corridor
```

핵심 constraint: `corridor_clear`가 `B_enter_corridor`보다 먼저 만족될 것.

---

## 3. Shared World Coordination

### 3.1 Shared World Model

**Input**: current actual shared state `S_t` (all agents' human/root states, task-relevant object
states, scene geometry) + current agent-wise skill/segment + skill goal / goal constraint +
task dependency / completion condition. **Steering Target은 WM input이 아니다.**

**Output**: K coordinated future trajectory candidates τ¹ … τᴷ — future human root trajectories,
future object trajectories, task/event states and timing (corridor_clear, B_enter_corridor,
grasp/release, progress), coarse collision/interference 관련 spatial evolution.

각 candidate는 Planner가 team-level future를 직접 비교하기 위한 representation이며,
Action Policy의 직접 입력은 아니다. 구체적 candidate generation / supervision 방식은 후속 설계 항목.
핵심은 **target space가 아니라 trajectory space에서 team-level future를 모델링·planning한다**는 것.

### 3.2 Trajectory-to-Steering Execution Bridge

```
Selected Future Trajectory τ*
        ↓  current actual position에서 τ*의 nearest point
        ↓  trajectory를 따라 lookahead point 선택
Current Steering Target g_t
        ↓
TokenHSI-based Multi-Agent Action Policy
```

Carry steering probe에서 trajectory 위 lookahead point를 target으로 주는 일관된 pure-pursuit rule로
다양한 경로를 추종할 수 있음을 확인했다 (수치는 §4.2). Skill Goal 근처에서는 Steering Target을 실제
Skill Goal로 전환/수렴시켜 native placement behavior를 이용한다.

따라서 Steering Target은 WM input이나 Planner의 planning variable이 아니라,
**selected trajectory를 Action Policy command로 바꾸는 execution interface**다.

### 3.3 Future Evaluation & Planning

Planner 평가 기준: dependency satisfaction / collision·interference / team task completion·progress /
valid object placement·clearance / efficiency·completion time.

가장 좋은 future τ*를 선택한 뒤, τ* 자체를 Action Policy에 넣지 않고 Trajectory-to-Steering
Controller가 현재 Steering Target을 계산한다.

---

## 4. Physical Execution & Learning

### 4.1 Multi-Agent Action Policy

Pretrained TokenHSI의 physical skill capability를 initialization으로 사용하되, 최종 policy는
Frozen TokenHSI가 **아니다**. Shared multi-agent environment에서 teammate-aware
collision/interference avoidance를 추가 학습한다. Control interface는 가능한 한 TokenHSI와 동일한
target-conditioned 형태를 유지한다.

**Per-agent input at t**: actual humanoid state `s^i_t` (Simulator) / actual interaction-object state
(Simulator) / current skill `skill^i_t` / **current target = Trajectory-to-Steering Controller가 계산한
`g^i_t`** / teammate state·local interaction context (Simulator, 추가 conditioning)

**Output**: current full-body joint action `a^i_t`

**Responsibility**: native HOI skill execution (approach/grasp/carry/placement/balance/contact),
local collision·interference avoidance, selected Steering Target의 physical execution.
WM이 생성한 future trajectory 전체는 Action Policy 입력으로 쓰지 않는다.

> **Pre-grasp note**: pretrained Carry에서는 box를 집기 전 target steering 영향이 약하다.
> V2에서는 이 구간의 세밀한 global steering을 target에 의존하지 않고, Multi-Agent Action Policy의
> local collision avoidance + native approach/pickup capability가 담당하도록 한다.
> Post-grasp부터 trajectory-derived Steering Target이 강한 transport control로 작동한다.

### 4.2 Closed-Loop / Receding-Horizon Execution

```
Actual S_t → World Model → τ¹…τᴷ → Planner → τ*
→ Trajectory-to-Steering Controller → g_t
→ TokenHSI-based Multi-Agent Action Policy → a_t
→ Physics Simulator → Actual S_(t+1) → Replan from Actual State
```

### 4.3 Training

- **Action Policy**: pretrained TokenHSI에서 시작해 native target-conditioned HOI skill capability를
  유지하면서 teammate-aware collision/interference avoidance를 학습
- **Trajectory-to-Steering Bridge**: Carry에서는 pretrained TokenHSI probe로 lookahead target 기반
  일관된 path following이 가능함을 확인. **multi-agent fine-tuning 이후에도 이 correspondence가
  유지되는지 재검증 필요**
- **World Model**: current shared state + task/skill condition → coordinated future trajectories 생성
- **Planner**: trajectory-level dependency / collision / task progress / efficiency 평가

> Independent single-agent trajectory는 V2의 runtime input이 아니다. Steering Target 역시 WM runtime
> input이 아니다. Runtime planning representation은 future trajectory이며, Steering Target은 selected
> trajectory 이후 execution 단계에서 계산한다.

---

# V2 Details

## 1. Control Interface

| | 정의 | 누가 쓰나 |
|---|---|---|
| **Skill Goal** `G_i^m` | agent i의 현재 skill segment m이 최종 달성해야 하는 objective. Carry에서는 최종 object placement point/region. long-horizon에서 segment가 바뀌면 같이 바뀜 | WM / Planner 조건 |
| **Future Trajectory** | 여러 agent·object의 future evolution을 trajectory/event로 표현 | WM / Planner의 planning representation. Action Policy는 전체를 직접 받지 않음 |
| **Steering Target** `g^i_t` | 선택된 future trajectory를 TokenHSI-compatible target command로 변환한 현재 execution target | Action Policy 입력. WM input도 planning variable도 아님 |

Action Policy 입력 형태:
```
Current Humanoid State + Current Interaction-Object State + Current Skill
+ Current Target (= Steering Target) + Teammate / Local Interaction Context
        ↓
TokenHSI-based Multi-Agent Action Policy → Full-body Joint Action
```

## 2. World Model: Temporal I/O

Planning horizon H. Input at t: `S_t` + agent-wise skill/segment + Skill Goal/Goal Constraint G +
task/dependency condition D + scene condition. Output: K candidates τ^1_{t+1:t+H} … τ^K_{t+1:t+H}
(human root trajectories, task-relevant object trajectories, event/task states).

WM은 Steering Target을 condition으로 future를 rollout하는 control-conditioned predictor가 **아니라**,
team-level future trajectory space를 직접 모델링하는 planning model이다.

## 3. World Model Output Representation

초기 버전에서는 full-body future joint trajectory를 예측하지 않는다. Planner가 coordination을
판단하는 데 필요한 최소 표현으로 시작한다.

- **Human**: per-agent root position trajectory (필요시 root orientation / velocity)
- **Objects**: task-relevant object position / orientation trajectory (필요시 velocity)
- **Events / Task State**: `corridor_clear`, `B_enter_corridor`, grasp/release, skill·task progress, completion
- **Collision / Interference**: 초기에는 root distance / coarse spatial overlap 수준.
  정확한 full-body contact와 local avoidance는 Action Policy + Simulator가 담당

## 4. Trajectory-to-Steering Controller

Steering Target을 별도로 proposal / sampling하지 않는다. Planner가 선택한 trajectory에서
일관된 deterministic rule로 현재 target을 계산한다.

### 4.1 Carry Rule (probe에서 검증)

1. 현재 character 위치에서 selected trajectory의 nearest point를 찾는다.
2. trajectory를 따라 약 **1.2 m** 앞의 point를 Steering Target으로 사용한다.
3. Skill Goal까지 남은 거리가 약 **1.5 m** 이하가 되면 carrot을 해제하고 actual Skill Goal을 target으로 쓴다.

```
Selected Trajectory τ*
      ·
      ·       ×  ← 1.2 m lookahead = Steering Target
     ·
Current ●
        ↓
Target-conditioned Action Policy
```

heading 방향으로 단순 offset을 주는 것이 아니라 **trajectory 위 point**를 target으로 쓰기 때문에,
lateral drift가 생겨도 path로 복귀하는 closed-loop pure-pursuit behavior가 나온다.

### 4.2 Empirical Feasibility Range

- lookahead: 약 **1.2 m** (1.0 m보다 짧아지면 placement 동작이 일찍 발생)
- release / Skill Goal switch: 약 **1.5 m**
- trajectory point spacing: **≤ 0.5 m** 권장
- 약 1.5 m 구간 내 **46° 이상 급커브**는 피할 것 (≈ minimum turning radius 1.6 m)
- initial tangent가 current heading과 자연스럽게 이어질수록 안정적
- 전체 traj 정확도보다 **현재 위치 앞쪽 local steering point의 정확도**가 중요

→ 20종 trajectory에서 무튜닝으로 **R² 0.91–0.997, 성공률 100%**.
이 조건은 WM trajectory feasibility constraint / Planner filtering criterion으로 쓸 수 있다.

### 4.3 Scope

이 virtual steering handle은 **Carry에서만** 강하게 검증되었다. 다른 skill은 control interface가
다를 수 있으므로 skill-specific execution bridge가 필요할 수 있다.

## 5. Planning & Receding-Horizon Execution

Main cost / constraint: dependency satisfaction, team task completion·progress, efficiency·makespan,
valid placement·clearance, coarse collision·interference.
현재 corridor task의 핵심 temporal constraint는 `T_clear(A) < T_enter_corridor(B)`.

Planner가 τ*를 선택하면 Controller가 current actual state 기준으로 g_t를 계산한다.
따라서 **별도의 learned trajectory → target inverse model은 필요하지 않다.**

## 6. Execution Frequency & Compute

주파수는 분리 가능하다 — Action Policy는 high-frequency full-body control, Controller는 selected
trajectory + current actual state로 target 갱신, WM/Planner는 lower-frequency로 future trajectory
생성·선택. 한 번 선택된 trajectory를 짧은 planning interval 동안 유지하면서 Steering Target만 자주
갱신할 수도 있고, 매 interval마다 actual state에서 새 trajectory를 생성할 수도 있다. 비율은 후속 실험.

초기 compute-friendly setting: WM output은 root + object + event만 / short–moderate horizon /
작은 K / full-body future joints는 예측 안 함 / small vector·token-based world model부터 시작.
이미지·voxel 기반 WM보다 상태 표현이 작고, execution target 계산은 deterministic rule이라
추가 inference cost가 매우 작다.

## 7. Training Strategy

**7.1 Action Policy Initialization** — pretrained TokenHSI의 physical skill capability와
target-conditioned execution을 최대한 유지. 추가 학습의 핵심은 teammate/local interaction context
conditioning, collision·interference avoidance, multi-agent shared-scene robustness.
approach/grasp/carry/balance를 처음부터 다시 배우는 게 아니라 local multi-agent execution capability를
**추가**하는 것. 중요한 검증 항목: fine-tuning 이후에도 Carry의 trajectory → lookahead target →
path following correspondence가 유지되는가.

**7.2 World Model Training** — 해결해야 할 항목: 동일 입력에서 meaningful한 K future candidates를
만드는 multimodal generation, generated trajectory의 feasibility·diversity, actual multi-agent
Action Policy + Simulator rollout을 supervision으로 어떻게 구성할지, fine-tuned Action Policy 변화에
따른 WM adaptation.

**7.3 Training Data Coverage** — Planner가 좋은 future만 보게 되면 안 되므로 collision,
dependency violation, inefficient coordination 등 **bad future**도 충분히 포함한다.
successful team rollout뿐 아니라 다양한 coordination failure / near-failure case를 포함해
trajectory-level cost가 학습·평가 가능하도록 구성.

## 8. Current Open Questions

- WM이 동일한 `S_t` + Task/Skill에서 K개의 meaningful coordinated future trajectories를 어떻게 생성할지
- feasibility constraint를 generation 단계에 넣을지, Planner가 filtering할지
- WM prediction horizon H와 candidate 수 K
- WM/Planner replanning : Steering Target update : Action Policy control 주파수 비율
- Multi-Agent Action Policy에 teammate context를 어떤 방식으로 넣을지
- pre-grasp에서 Steering Target 영향이 약한 동안 local collision avoidance만으로 충분한지
- multi-agent fine-tuning 이후에도 Carry의 trajectory-to-target execution bridge가 유지되는지
- Skill Goal이 region인 경우 final placement point를 언제/어떻게 commit할지
- Carry 외 skill에서 trajectory와 low-level control을 연결할 skill-specific execution interface
