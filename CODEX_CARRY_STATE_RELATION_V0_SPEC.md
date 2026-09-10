# CVPR2027 — Carry-only State–Relation Reward v0
## Codex 구현 명세: `edge_a2_gta` 기준

### 2026-09-10 대화에서 확정한 구현 기준 (아래 원문보다 우선)

- 성공 시 scene을 종료하지 않는다. `terminate_when_all_subgoals_done: false`로 두고 기존 timeout/fall 종료를 유지한다. subgoal별 bonus는 1회, 완료 후 해당 task reward는 0이며 AMP/regularizer는 계속 적용한다.
- 공통 velocity progress 및 sigmoid gate 수식은 원문 그대로 사용한다. 0.5m 같은 추가 reward cutoff, pinning 복원, 감속 곡선, height mask는 넣지 않는다. 목표 근처 속도·오차·gate·AMP/합산 보상과 sampled time-series를 기록하여 확인한다.
- putdown 허용 오차는 TokenHSI와 같은 XY 0.1m / Z 0.001m. Holding은 두 손 midpoint의 `exp(-5*error²)`와 satisfaction 0.9이며, 기존 absolute reward 및 root–box 0.7m cutoff는 복사하지 않는다.
- H/O RMS는 유지하고 Target/GTA pose/relation suffix는 RMS 및 flat observation clipping을 우회한다. 학습 중 static/dynamic/QK attention scale과 entropy를 반복 기록한다.
- 새 학습 기본값은 **env 2048 / PPO minibatch 16384 / mini-epochs 6**. 기본 scene은 비교에 사용한 M=2, O=3이며 positional arguments로 변경할 수 있다. 기존 legacy 설정과 스크립트는 유지한다.
- 구현·실행 검증 결과 및 명령은 [구현 보고서](markdowns/CARRY_STATE_RELATION_V0_IMPLEMENTATION.md)에 정리했다. 원문의 success-termination 요구/체크리스트는 위 합의로 대체한다.

> **이 문서는 코드 수정 작업 지시서다.** 아래 명세를 읽고 실제 저장소를 다시 확인한 뒤, 구현·테스트·실행 스크립트·구현 결과 보고까지 완료하라. 설계 설명이나 TODO만 작성하고 끝내지 말 것.
>
> **대상 저장소:** `https://github.com/KSH3880/CVPR2027`  
> **대상 브랜치:** `edge_a2_gta`  
> **명세 작성 시 확인한 원격 HEAD:** `9b68fca23fb65842c8a9a64cfd6fc066c8dcee76`  
> **확인일:** 2026-09-10  
> **이번 구현:** Carry-only, `Holding(H,O)` 및 `At(O,G)`, 상태 변화량 + 공통 velocity progress, soft prerequisite, achieved 이력, subgoal별 1회 성공 보너스.  
> **이번에 구현하지 않음:** `OnTop`, `Beside`, `Push`, `Pull`, operator/action embedding, 계단 쌓기 시나리오, world model/planner.
>
> 문서에서 **확인된 기존 구현**, **이번 v0의 구현 결정**, **검증해야 할 가설/한계**를 구분한다. 여기 제시된 새 reward는 아직 학습 성능을 검증한 결과가 아니다. “곱셈이면 유지가 보장된다”, “soft gate면 exploit이 사라진다”처럼 주장하지 말 것.

---

## 0. 먼저 읽을 요약

### 0.1 목표

기존 Carry의 task reward를 이름만 바꾸는 것이 아니라, 두 개의 **상태 relation edge**에 정렬한다.

```text
Carry subgoal 전체

Human H ── Holding ──▶ Object O ── At ──▶ Goal G

Holding(H,O)
  state component    = 손–물체 상태 점수의 변화량
  progress component = H의 XY 속도가 O를 향하는 정도         [Walk]

At(O,G)
  state component    = box_near + putdown으로 만든 상태 점수의 변화량
  progress component = O의 XY 속도가 G를 향하는 정도         [Transport]
```

`Walk`, `Transport`, `Putdown`을 추가 relation edge로 만들지 않는다. `At(H,O)` 접근 edge도 이번에는 추가하지 않는다.

각 edge의 공통 계산 문법은 다음이다.

$$
R_e^t = A_e^t\left[\lambda_s(\phi_e^{t+1}-\phi_e^t)
+\lambda_v(1-g_e^t)P_e^{t\to t+1}\right],
\qquad
A_e^t=\prod_{p\in Pre(e)}g_p^t.
$$

- $\phi_e$: relation별 상태 만족도, `[0,1]`.
- $g_e=C(\phi_e)$: 공통 sigmoid로 만든 soft gate.
- $P_e$: 모든 edge가 공유하는 **XY velocity-only** progress.
- $Pre(e)$: subgoal이 지정하는 prerequisite relation 목록.
- 빈 prerequisite 집합의 곱은 `1`.
- `State`는 `+phi`가 아니라 **signed `delta_phi`**다.
- `1-g_e`는 **progress 성분에만** 곱한다. state 변화량까지 일괄 끄지 않는다.

Carry에서는:

$$
R_H^t=\lambda_s\Delta\phi_H^t+
\lambda_v(1-g_H^t)P(H,O),
$$

$$
R_A^t=g_H^t\left[\lambda_s\Delta\phi_A^t+
\lambda_v(1-g_A^t)P(O,G)\right].
$$

최종 목표의 유효 완료와 subgoal 성공은:

$$
valid_i^{t+1}=c_{A,i}^{t+1}\land a_{H,i}^{t},
$$

$$
first_i^{t+1}=valid_i^{t+1}\land\neg done_i^t,
$$

$$
r_{task,i}^t=\mathbf1[\neg done_i^t](R_{H,i}^t+R_{A,i}^t)
+B_{success}\mathbf1[first_i^{t+1}].
$$

`done_i`는 **그 subgoal의 보너스 지급/완료 이력**이다. 물리 상태가 영구히 유지된다는 뜻이 아니다.

### 0.2 이번에 바꾸지 않을 것

- one-scene / one-X 구조, H/O/T tokenizer, GTA geometry, full self-attention.
- actor와 critic의 별도 encoder, Human readout, shared action/value head.
- humanoid asset, PD 제어, simulator, AMP observation/discriminator/reference motion.
- 기존 기본 학습 hyperparameter. 새 reward와 동시에 학습률·sigma·batch 등을 대규모 변경하지 않는다.
- 기존 legacy 실행 경로와 checkpoint 해석.
- 기존 실험 산출물·로그·checkpoint. 새 실행은 별도 output 디렉터리.

### 0.3 특히 금지하는 오구현

1. `Holding`을 달성하자마자 state 항 전체를 `(1-c_H)`로 끄는 구현.
2. 모든 edge에 `+phi`를 매 step 지급하는 구현.
3. `g_H` 대신 **과거 achieved `a_H`**로 운반 reward를 영구 활성화하는 구현.
4. `At`만 만족하면 성공 보너스를 지급하는 구현.
5. `0→1`이 반복될 때마다 성공 보너스를 재지급하는 구현.
6. env 전체 reset 시 다른 env의 `prev_phi`, `achieved`, `done`까지 지우는 구현.
7. 기존 `walk/carry/handheld/putdown` 합계 위에 새 reward를 추가해 이중 지급하는 구현.
8. `Holding/At` truth에 따라 attention graph에서 edge를 매 step 생성/삭제하는 구현.
9. 모델 forward에서 live simulator의 achieved 버퍼를 직접 읽는 구현. PPO 재학습 시에는 **rollout에 저장된 관측**을 읽어야 한다.
10. 과거 대화에서 언급된 `842D`, `300D suffix`, `72D level encoder`를 현재 브랜치의 사실로 가정하는 구현. 아래 실제 확인 결과를 따른다.

---

## 1. 작업 전 저장소 점검 및 보존 정책

먼저 다음을 실행하고 결과를 구현 보고서에 기록하라.

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git log -1 --oneline
```

명세 작성 시 HEAD와 로컬 HEAD가 다르면 최신 코드를 읽어 차이를 반영한다. **명세의 SHA로 hard reset하지 않는다.** 사용자의 uncommitted 변경을 덮어쓰지 않는다.

저장소/상위 디렉터리의 `AGENTS.md`, `CLAUDE.md` 등 작업 지침이 있으면 읽는다. 실행·GPU 자동화 관련 파일이 있더라도 기존 실험을 임의 종료하거나 GPU 전체를 점유하지 않는다.

### 1.1 추가형 구현

환경 설정에 명시적 모드를 추가한다.

```yaml
env:
  relationReward:
    mode: legacy_tokenhsi  # 기존 config에서 key가 없을 때도 반드시 이 값
```

이번 새 config에서만:

```yaml
env:
  relationReward:
    mode: state_relation_v0
```

- `legacy_tokenhsi`: 기존 reward, 입력 크기, relation matrix, checkpoint loading 동작 유지.
- `state_relation_v0`: 본 명세의 새 reward 및 필요한 상태 관측 경로 사용.
- 새 모드는 우선 `policyObsMode: clean_scene`만 지원한다.
- `legacy_multirow + state_relation_v0`는 조용히 잘못 실행하지 말고 설명적인 오류를 낸다.

새 config와 script를 추가하되 기존 명령어의 의미를 바꾸지 않는다. 원격 push, 기존 브랜치 reset, 대규모 학습 실행은 작업 범위에 포함하지 않는다.

---

## 2. 실제 브랜치에서 확인한 구조

아래 정보는 `9b68fca23fb65842c8a9a64cfd6fc066c8dcee76`의 코드를 읽어 확인했다. 최종 구현 시 로컬 파일을 다시 확인한다.

### 2.1 중요한 파일

| 파일 | 확인된 역할 / 수정 시 검토 지점 |
|---|---|
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | `HumanoidMACarry`, 물체 assignment, scene observation, reward, reset, evaluation |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma.py` | multi-agent tensor layout, base reset/done 및 scene 환경 인터페이스 |
| `tokenhsi/env/tasks/multi_agent/scene_features.py` | GTA pose record 구성 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | `build_relation_matrix`, `EdgeEncoder`, `RelationEncoder`, `RelationTransformerLayer`, actor/critic |
| `tokenhsi/learning/multi_agent/scene_normalizer.py` | H/O RMS, Target/pose bypass |
| `tokenhsi/learning/multi_agent/ma_agent.py` | PPO scene rollout, network config, normalizer, AMP 결합 |
| `tokenhsi/learning/multi_agent/ma_players.py` | 테스트 player의 config/normalizer, reward debug 출력 |
| `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml` | 환경 config |
| `tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml` | network/PPO/AMP config |
| `tokenhsi/scripts/multi_agent/ma_carry_train.sh` | 기존 train 실행 및 `runtime_env.sh` 사용 |
| `tokenhsi/tests/test_ma_scene_policy.py` | 기존 network/GTA/가변 entity 수 회귀 테스트 |
| `tokenhsi/tests/test_ma_scene_features.py` | scene feature 테스트 |

### 2.2 현재 scene layout

`M` agents, `O >= M` objects, `M` goals:

```text
[H_0 ... H_(M-1) | O_0 ... O_(O-1) | G_0 ... G_(M-1)]

Human raw content : 223D
Object raw content: 30D
Goal raw content  : 1D constant
Pose record       : 7D per token
L                 : 2*M + O
```

현재 관측 폭:

$$
D_{legacy}=223M+30O+M+7(2M+O)=238M+37O.
$$

예: `M=2, O=2`이면 **6개 token, 550D observation**이다. 이번에 B의 goal을 없애거나 O–O 쌓기 task로 바꾸지 않는다. `OnTop`은 후속 범위다.

물리 box assignment는 reset마다 달라질 수 있고, 논리 object slot `[0,M)`은 owner 순서다. `_assigned_box_values`, `_logical_box_values`를 재사용해야 한다. `physical box index == agent index`로 가정하지 않는다.

### 2.3 현재 relation encoder

원격 HEAD의 relation taxonomy는 다음과 같다.

```text
REL_NONE        = 0
REL_SELF        = 1
REL_TEAMMATE    = 2
REL_OWN_OBJECT  = 3
REL_OWN_GOAL    = 4
REL_OBJECT_GOAL = 5
```

이는 **소유/연결 관계**이지, 본 명세의 `Holding/At` 현재 상태 graph가 아니다.

현재 A2 edge MLP:

```text
source type 16 + relation type 32 + target type 16 = 64
64 -> 64 -> 64
layer/head bias projection
```

현재 static relation matrix는 `[L,L]`, bias는 `[layers,heads,L,L]`이고 batch dimension이 없다. 새 `achieved`/현재 relation 상태를 입력하면 batch별 차이가 생기므로 아래 §12의 관측 및 bias 확장이 필요하다.

### 2.4 현재 reward와 설정

`HumanoidMACarry._compute_reward`는 각 agent의:

```text
walk_r + carry_r + handheld_r + putdown_r
+ power penalty
+ agent collision penalty
```

를 계산한다. 이 브랜치의 실제 합계는 앞선 다른 main/steer 구현에서 보았던 `2*walk + 2*carry`라고 가정하면 안 된다.

환경 config의 `onlyVelReward: True`에서:

```text
Walk      = 0.2 * velocity_term
Transport = 0.2 * velocity_term + 0.2 * near_xyz_term
Handheld  = 0.2 * hand_geometry_term
Putdown   = 0.2 * geometric_putdown_indicator
```

단, 원본의 pinning/height mask/box speed penalty도 적용되므로 위는 성분 구분이며 최종 숫자의 전부가 아니다.

기본 네트워크: `64D`, `4 layers`, `2 heads`, GTA enabled, 별도 actor/critic. 학습 config는 `gamma=0.99`, `horizon_length=32`, `minibatch_size=8192`, `mini_epochs=6`, `task_reward_w=0.5`, `disc_reward_w=0.5`. 다른 대화의 실험값으로 조용히 바꾸지 않는다.

---

## 3. 용어와 책임 경계

| 용어 | 이번 구현에서의 의미 |
|---|---|
| Entity token | Human, Object, Goal의 내용/기하 표현 |
| Relation edge | 특정 entity pair에 대한 상태 predicate instance |
| State evaluator | 해당 pair의 상태 점수 `phi`를 계산하는 함수 |
| Progress evaluator | source가 target을 향하는 XY 속도를 계산하는 공통 함수 |
| Prerequisite | 현재 subgoal 실행 문맥에서 선언한 relation 간 의존성 |
| Soft gate `g` | 현재 상태가 다음 reward를 얼마나 활성화하는지 |
| Current satisfaction `c` | 지금 relation이 기준을 만족하는지 |
| Achieved `a` | 이번 subgoal에서 relation의 유효 만족을 경험했는지 |
| Subgoal done | 성공 보너스를 이미 지급했고 해당 subgoal을 완료 처리했는지 |
| Carry | 집기/운반/목표 배치를 포함한 subgoal 전체 |
| Walk | `Holding(H,O)` edge에 딸린 H→O velocity 성분의 설명용 이름 |
| Transport | `At(O,G)` edge에 딸린 O→G velocity 성분의 이름; 기존 코드의 `carry_r` |

### 3.1 유지할 독립성

`phi_Holding(H,O)`는 goal G를 읽지 않는다. `phi_At(O,G)`는 human H를 읽지 않는다. 공통 progress는 relation 이름을 보지 않는다.

다른 edge와의 의존성은 reward engine이 `Pre(e)`로 처리한다. 따라서 **evaluator의 모듈성**은 확보하지만, 실행과 최종 reward가 물리적으로 독립이라는 뜻은 아니다.

### 3.2 “모든 edge마다 함수를 새로 만든다”가 아님

- `Holding(H_A,O_A)`와 `Holding(H_B,O_B)`는 하나의 evaluator를 공유한다.
- `At(O_A,G_A)`와 `At(O_B,G_B)`도 하나의 evaluator를 공유한다.
- relation별로 다른 것은 state evaluator다.
- 실제 edge instance 수는 agent 수에 따라 늘어나지만 reward 함수 종류는 늘어나지 않는다.

---

## 4. 이번 Carry subgoal과 정적 graph

각 agent i에 대해:

```text
subgoal_i:
  actor: H_i
  operator: CARRY  # task 정의에만 존재. 이번 network 입력에는 넣지 않음.
  object: O_i
  target_edge: At(O_i, G_i)

edges:
  e_H_i = Holding(H_i, O_i)
  e_A_i = At(O_i, G_i)

prerequisite:
  Pre(e_H_i) = {}
  Pre(e_A_i) = {e_H_i}
```

`Holding -> At`은 **relation instance 간 dependency**다. entity graph의 `H -> O -> G` 화살표와 다른 종류의 연결임을 코드 자료형에서도 구분한다.

이번에는 각 agent가 자기 box를 자기 goal에 놓는 Carry를 병렬 수행한다. A가 놓은 box 위에 B가 쌓는 시나리오는 구현하지 않는다.

### 4.1 자동화의 정확한 의미

`compile_carry_subgoal(H_i,O_i,G_i)`가 위 두 edge와 dependency를 생성하게 한다. evaluator가 자동으로 물리적 prerequisite를 발견하는 것은 아니다.

`At` relation 자체에 `Holding`이 항상 필요하다고 전역 하드코딩하지 않는다. 향후 Push에서는 다른 prerequisite를 쓸 수 있기 때문이다.

### 4.2 권장 정적 tensor schema

캐노니컬 edge 순서:

```text
[Holding_0, At_0, Holding_1, At_1, ..., Holding_(M-1), At_(M-1)]
E = 2*M
```

```python
edge_src:       LongTensor[E]
edge_dst:       LongTensor[E]
edge_relation:  LongTensor[E]
edge_owner:     LongTensor[E]       # reward를 받는 agent index
edge_terminal:  BoolTensor[E]
edge_mask:      BoolTensor[E]
prereq_mask:    BoolTensor[E, E]    # prereq_mask[e, p] == True: e가 p에 의존
subgoal_target: LongTensor[M]       # 각 subgoal의 At edge index
```

- `Holding_i`: `src=i`, `dst=M+i`.
- `At_i`: `src=M+i`, `dst=M+O+i`.
- `edge_owner[2*i] = edge_owner[2*i+1] = i`.
- `prereq_mask[2*i+1, 2*i] = True`.
- 나머지 prerequisite는 false.
- SELF/TEAMMATE/NONE attention pair는 task reward edge로 합산하지 않는다.

graph compiler/registry를 simulator와 network가 공유할 수 있는 가벼운 모듈에 둔다. 순환 import나 Isaac Gym import가 없어야 한다.

---

## 5. 좌표 및 velocity 규칙

### 5.1 대표 위치

| Entity | progress 위치 | progress velocity |
|---|---|---|
| Human | humanoid root position | root의 연속 control-step 위치 차분 |
| Box | box actor 중심 위치 | 해당 assigned box의 연속 위치 차분 |
| Goal | 목표 box 중심 좌표 `_tar_pos` | 이번에는 사용하지 않음 |

- progress는 **XY 평면**만 사용한다. 3D center 방향을 쓰면 Human이 바닥의 box 중심을 향해 아래로 움직이라는 신호가 될 수 있다.
- Holding/At state evaluator는 **XYZ**를 사용한다.
- world 좌표 혹은 env-local 좌표 중 동일한 convention으로 source/target을 모두 계산한다. 한쪽에만 env origin을 빼지 않는다.
- source velocity는 TokenHSI와 맞춰 **위치 차분 / control dt**를 사용한다.
- target velocity를 빼는 relative velocity로 이번에 바꾸지 않는다. 이는 추후 moving target에서 별도 평가할 문제다.
- Goal의 z는 바닥/플랫폼 surface z가 아니라 **box center의 desired z**라는 현재 코드 convention을 확인하고 유지한다.

### 5.2 frame 및 assignment 안전성

reset/reassignment 전후 위치를 연결해서 velocity를 만들면 안 된다. reset된 env는 `prev_position = reset_position`으로 맞춘다.

training의 reference-state reset 직후 simulator refresh와 `_kinematic_humanoid_rigid_body_states` 중 어떤 tensor가 유효한지 현재 경로를 확인한다. 무효한 초기 tensor에서 phi/velocity를 계산하지 않는다.

---

## 6. 공통 progress: velocity-only

transition `s_t -> s_(t+1)`에 대해, 원본 함수와 같은 post-step 방향 convention을 사용한다.

$$
\delta p_{XY}^{t+1}=p_Y^{xy,t+1}-p_X^{xy,t+1},
\quad
u_{XY}^{t+1}={\delta p_{XY}^{t+1}\over
\max(\|\delta p_{XY}^{t+1}\|_2,\epsilon)}.
$$

$$
v_X^{xy,t\to t+1}={p_X^{xy,t+1}-p_X^{xy,t}\over\Delta t},
\quad
v_\parallel=u_{XY}^{\top}v_X^{xy,t\to t+1}.
$$

$$
\boxed{
P_e^{t\to t+1}=
\mathbf1[\|\delta p_{XY}^{t+1}\|_2>\epsilon]
\mathbf1[v_\parallel>0]
\exp\{-k_v(v^*-v_\parallel)^2\}.
}
$$

v0 초기값:

```yaml
target_speed: 1.5
velocity_scale: 5.0
normalization_epsilon: 1.0e-6
```

`P`는 `[0,1]`이다. weight `lambda_v`는 밖에서 곱한다. 함수 내부에서 0.2를 곱하고 engine에서도 다시 곱하지 않는다.

### 6.1 포함하지 않는 것

- `exp(-0.5 * xy_distance_squared)` far position 성분.
- relation별 walk 함수 또는 transport 함수의 중복 구현.
- goal 근처에서 velocity reward를 강제로 1로 고정하는 legacy pinning.
- Carry height mask.
- Push/Pull용 앞·뒤 target offset.
- steering/path-following reward.

원본 함수들은 legacy 경로에 그대로 남는다. 새 모드의 공통 progress는 독립 helper로 구현한다.

### 6.2 이 progress의 의미와 한계

이것은 **거리 감소량 자체**가 아니라 “target 방향 source speed를 선호하는 shaping”이다. moving target이면 source velocity만으로 두 entity의 상대 거리가 줄었다고 단정할 수 없다. 이번 고정 goal Carry에서 먼저 검증한다.

`v_parallel <= 0`이면 0이다. “후진에 음수 penalty를 준다”라고 잘못 설명하지 않는다.

`v_parallel=0`에서 0이 되는 indicator는 원본의 특성이다. 모든 항이 완전히 미분 가능한 reward라는 주장도 하지 않는다. PPO는 이 reward를 미분해 action을 업데이트하는 방식이 아니다.

---

## 7. State satisfaction evaluator

### 7.1 Holding: legacy handheld geometry 기반 v0

각 agent의 right/left hand 위치 평균:

$$
\bar p_{hand}={p_{right}+p_{left}\over2},
\quad
\epsilon_H=\|\bar p_{hand}-p_O\|_2^2.
$$

$$
\boxed{\phi_H=\exp(-k_H\epsilon_H),\qquad k_H=5.0.}
$$

초기 구현에서는 `onlyHeightHandHeldReward=False`의 XYZ 형태만 지원한다. root–box 0.7m legacy cutoff는 새 evaluator에 자동 복사하지 않는다. legacy 경로에서만 기존대로 유지한다.

상태 evaluator는 다음을 반환하도록 한다.

```python
phi_holding: Tensor[N, M]   # [0,1]
aux:
  hands_mean_error_xyz
  root_box_distance_xy
  right_hand_box_distance
  left_hand_box_distance
```

`aux`는 진단용이며 core reward에 몰래 가중하지 않는다.

#### 반드시 인정할 한계

이 식은 **실제 grasp/support의 완전한 판정기가 아니다.** 손 평균이 box 중심 근처인 자세, 양손이 멀리 벌어졌지만 평균만 맞는 자세, 바닥 box 위에 손만 가져다 댄 자세도 높은 값을 만들 수 있다.

따라서 `achieved_Holding`은 이번 v0에서 **이 proxy를 만족한 이력**이다. 이를 ground-truth “실제로 잡고 운반했다”라고 표현하지 않는다.

실험 진단에서 이 false-positive가 확인되면 동일 evaluator 내부의 접촉/relative motion/grasp-point 정의를 개선해야 한다. 이번 v0에서 새로운 contact gate, lift gate, torso-frame attachment 규칙을 임의로 추가하지 않는다. 필요한 강화는 후속 변경안으로 보고한다.

### 7.2 At: box_near와 putdown을 하나의 score로 통합

$$
q_{near}=\exp\{-k_A\|p_O-p_G\|_2^2\},\qquad k_A=10.0.
$$

원본 putdown의 기하 판정:

$$
q_{put}=\mathbf1[\|p_O^{xy}-p_G^{xy}\|_2\le\epsilon_{xy}]
\mathbf1[|z_O-z_G|\le\epsilon_z].
$$

원본과 맞춘 초기값:

```yaml
putdown_xy_tolerance: 0.10
putdown_z_tolerance: 0.001
```

통합 상태 점수:

$$
\boxed{
\phi_A=q_{near}[\alpha+(1-\alpha)q_{put}],\qquad\alpha=0.5.
}
$$

- 멀면 `q_near≈0`, `phi_A≈0`.
- 근처에 있지만 putdown 기준 전이면 `phi_A <= 0.5`.
- 목표 기하 기준까지 맞으면 `phi_A = q_near`, 1에 가까워진다.

`q_near`와 `q_put`는 하나의 At evaluator 내부 성분이다. 별도 Putdown edge, 별도 positive putdown reward를 추가하지 않는다.

#### 이 합성에 대한 주의

1. `q_put`가 binary이므로 **phi_A 자체에 jump가 존재한다**. 뒤에 sigmoid를 사용해도 전체가 완전히 smooth해지는 것은 아니다.
2. 1mm z tolerance는 원본 값의 차용이지 모든 물체/physics 설정에 적합하다는 보장이 아니다. 실제 도달 가능성과 chattering 빈도를 측정해야 한다.
3. 원본 putdown 식은 **위치 조건**이다. support contact, 속도 안정성, hands released를 자동 검증하지 않는다.
4. 따라서 v0 success 이름은 `valid_geometric_carry_success`처럼 실제 측정 내용을 드러내라. 영상/물리 진단 없이 “stable released placement 성공률”이라고 보고하지 않는다.
5. 필요 시 향후 같은 evaluator 안에서 `q_put`를 support/settled score로 개선할 수 있다. 이번 기본 식을 변경했다면 반드시 변경 이유·수식·config 차이를 보고해야 한다.

### 7.3 상태와 completion을 통일한다는 의미

completion을 다른 수작업 reward로 만드는 것이 아니라, **동일한 phi를 thresholding**한다.

$$
c_e^t=\mathbf1[\phi_e^t\ge\tau_{success,e}].
$$

하지만 `phi`, `soft gate`, `current bool`, `achieved bool`을 같은 변수 하나로 덮어쓰지 않는다. 공통 evaluator에서 파생되지만 시간적 의미가 다르다.

---

## 8. Soft gate와 현재 만족 판정

공통 soft gate:

$$
\boxed{g_e^t=\sigma[\beta(\phi_e^t-\tau_{gate})].}
$$

초기값:

```yaml
gate_center: 0.80
gate_beta: 30.0
satisfaction_threshold: 0.90
```

- `tau_gate`는 sigmoid 중심이다. hard threshold가 아니다.
- `tau_success`는 이력/최종 성공 판정을 위한 bool 기준이다.
- v0는 모든 relation에 동일한 기본값을 적용하되, config 구조는 relation별 calibration을 향후 허용할 수 있어야 한다.
- score 분포가 다르면 같은 threshold의 물리 의미도 다를 수 있다. 한 숫자라서 자동으로 general하다고 주장하지 않는다.

예시 (`beta=30`, center `0.8`):

| phi | soft gate g (근사) |
|---:|---:|
| 0.0 | 0.000000000038 |
| 0.5 | 0.000123 |
| 0.7 | 0.047426 |
| 0.8 | 0.500000 |
| 0.9 | 0.952574 |
| 1.0 | 0.997527 |

sigmoid는 정확한 0/1이 아니다. `g_H≈0`을 “운반 reward가 수학적으로 완전 0”이라고 설명하지 않는다.

Hysteresis/EMA/counter는 이번 reward gate의 기본값에 추가하지 않는다. 단계가 늘어나는 대신 일단 sigmoid만 사용한다. 향후 추가하면 그 memory 역시 observation/reset에 반영해야 한다.

---

## 9. 정확한 reward 수식과 시간 인덱스

### 9.1 공통 edge reward

`phi_prev = phi(s_t)`, `phi_next = phi(s_(t+1))`:

$$
\Delta\phi_e^t=\phi_e^{t+1}-\phi_e^t.
$$

**v0에서는 prerequisite gate와 자기 progress gate 모두 pre-transition 값**을 사용한다.

$$
A_e^t=\prod_{p\in Pre(e)}g_p^t.
$$

$$
\boxed{
R_e^t=A_e^t\big[\lambda_s\Delta\phi_e^t+
\lambda_v(1-g_e^t)P_e^{t\to t+1}\big].
}
$$

이 convention을 택하는 이유는, 해당 transition 중 prerequisite를 잃었을 때 post-state gate가 0이 되어 음의 변화량까지 즉시 가리는 구현을 피하기 위해서다. 이것이 모든 exploit을 없앤다는 뜻은 아니다.

### 9.2 Carry 두 edge

$$
\boxed{
R_H^t=\lambda_s(\phi_H^{t+1}-\phi_H^t)
+\lambda_v(1-g_H^t)P(H,O).
}
$$

$$
\boxed{
R_A^t=g_H^t\left[
\lambda_s(\phi_A^{t+1}-\phi_A^t)
+\lambda_v(1-g_A^t)P(O,G)
\right].
}
$$

**별도 maintain reward는 추가하지 않는다.** Holding이 감소하면 `delta_phi_H < 0`이고, 앞으로의 At reward도 soft gate를 통해 약해진다. 이는 보상 유도이지 물리적 유지 보장이 아니다.

### 9.3 signed delta를 보존할 것

```python
delta_phi = phi_next - phi_prev
```

다음을 하지 않는다.

```python
# 금지: 나빠지는 transition을 없애 버림.
delta_phi = (phi_next - phi_prev).clamp_min(0)

# 금지: 상태가 높으면 정지한 채 매 step reward 지급.
state_reward = phi_next

# 금지: state 항도 자기 completion과 함께 꺼짐.
state_reward = (1.0 - gate_prev) * delta_phi
```

success 후 task 종료 mask는 별도다. 그것은 **이미 끝난 subgoal을 다시 보상하지 않는 처리**이지 미완료 relation의 음수 변화량을 지우는 처리가 아니다.

### 9.4 초기 가중치: 실험 출발값

```yaml
state_delta_weight: 1.0
velocity_progress_weight: 0.2
subgoal_success_bonus: 5.0
```

- `velocity_progress_weight=0.2`는 원본 velocity 항의 scale을 차용한다.
- `state_delta_weight=1.0`은 `phi`를 delta로 바꾼 뒤의 **새 초기값**이다. 원본 `0.2*phi`와 같은 시간 누적 보상이라고 주장하지 않는다.
- success bonus 5.0도 검증된 최적값이 아니다.
- 모든 edge에 같은 기본 state/velocity weight를 적용한다. Carry/Holding별 특수 weight 분기는 만들지 않는다.
- 학습 전후에 component별 평균/표준편차/누적합을 측정하고, 변경 시 config 및 비교 결과를 남긴다.
- `lambda_s`, `lambda_v`, `B`를 함수 안팎에서 중복 적용하지 않는다.

### 9.5 왜 단순 soft switching만으로 성공이 보장되지는 않는가

아래는 구현 버그가 아니라 **제안 objective의 알려진 한계**다.

- sigmoid로 부드럽게 연결해도 다음 상태의 기대 return이 충분하지 않으면 정지할 수 있다.
- delta state는 동일 state에서 0이지만, velocity 성분·AMP·종료 처리까지 포함한 총 return의 최적 행동은 별도로 평가해야 한다.
- `A_e(t) * delta_phi_e`는 일반적인 단일 potential의 차분과 같지 않다. 따라서 gate를 바꾸면서 같은 상태를 왕복하는 cycle에 양의 reward가 남을 수 있다.
- 양의 velocity shaping은 되돌아가는 구간을 강하게 벌하지 않으므로 왕복/비비기 exploit 가능성이 남는다.

**이번에는 합의한 식을 구현하되, 이를 이론적으로 exploit-free라고 포장하지 말 것.** §17의 counterexample와 rollout 진단으로 위험을 드러내고 후속 조정 근거를 만든다. 임의로 potential-based 방식이나 별도 maintain penalty로 바꾸지 않는다.

---

## 10. Achieved 이력과 subgoal success

### 10.1 현재 만족과 과거 달성의 구분

```text
phi_e : 현재 연속 상태 점수
g_e   : 현재 reward의 soft 활성화 값
c_e   : 현재 threshold 만족 여부
a_e   : 이번 subgoal에서 유효 만족을 경험한 적 있는지
```

Holding을 놓으면 `phi/g/c`는 감소할 수 있지만, `a_H`는 reset 전까지 1로 남는다.

**reward의 prerequisite에는 현재 `g_H`**, **최종 유효 완료에는 과거 `a_H`**를 사용한다. 이 둘을 바꾸면 안 된다.

### 10.2 valid completion의 공통 정의

v0는 과거 prerequisite 달성을 요구하는 dependency를 구현한다.

$$
valid_e^{t+1}=c_e^{t+1}\land
\bigwedge_{p\in Pre(e)}a_p^t,
$$

$$
a_e^{t+1}=a_e^t\lor valid_e^{t+1}.
$$

부모 이력은 **업데이트 전 `a_prev` snapshot**으로 읽는다. 같은 step에 graph를 순차 순회하면서 원치 않게 여러 단계를 즉시 통과하지 않도록 한다.

Carry:

$$
a_H^{t+1}=a_H^t\lor c_H^{t+1},
$$

$$
valid_A^{t+1}=c_A^{t+1}\land a_H^t.
$$

### 10.3 target 유효 완료가 subgoal 성공

$$
first_i^{t+1}=valid_{target_i}^{t+1}\land\neg done_i^t,
$$

$$
done_i^{t+1}=done_i^t\lor valid_{target_i}^{t+1}.
$$

$$
bonus_i^t=B_{success}\mathbf1[first_i^{t+1}].
$$

`first`는 **이번 episode/subgoal에서 최초 한 번**이다. 단순히 `c_A`의 `0→1`만 보면 흔들릴 때마다 bonus가 나갈 수 있으므로 반드시 `done` latch를 사용한다.

### 10.4 Carry 전체 task reward

$$
\boxed{
r_{task,i}^t=(1-done_i^t)(R_{H,i}^t+R_{A,i}^t)
+B_{success}first_i^{t+1}.
}
$$

- 첫 성공 transition의 dense reward와 bonus는 지급한다.
- 이후 해당 agent/subgoal의 task reward는 0이다.
- release 후 Holding이 낮아졌다고 해당 Carry가 재활성화되지 않는다.
- 다른 agent의 미완료 subgoal reward는 계속 계산한다.

### 10.5 이것이 막는 것과 막지 못하는 것

**막는 것:** Holding proxy를 한 번도 만족하지 않은 채 box가 goal에 들어갔을 때 **성공 bonus**를 지급하는 오류.

**막지 못하는 것:**

```text
잠깐 Holding proxy 만족 -> achieved_H = 1
-> 놓음 / 던짐 / 발로 참
-> 나중에 At 만족
```

이 경로도 `a_H * c_A`만으로는 성공 처리될 수 있다. 기존 대화의 “achieved 하나로 정상 Carry 의미를 완전히 보장한다”는 해석은 채택하지 않는다.

또한 soft gate는 정확한 0이 아니므로, achieved가 없어도 작은 shaping reward는 남을 수 있다. “발로 차면 전체 reward가 반드시 0”이라고 주장하지 않는다.

이번 v0는 이를 **history-qualified geometric success**로 명명한다. 연속적인 손 지지, actual carry route, 안전한 release까지 보장하려면 향후 더 강한 validator/temporal contract가 필요하다. 이번 구현에 그러한 새 규칙을 몰래 추가하지 않는다.

### 10.6 Holding과 release의 경계

이번 success는 `At 만족 + Holding 이력`이다. 최종 순간에 `Holding=1`을 요구하지 않는다.

그러나 현재 geometric `At`는 “손을 확실히 떼었다”까지 확인하지 않는다. 성공 후 reward 종료로 release를 방해하지 않도록 할 수는 있지만, **release를 적극적으로 성공 조건으로 강제한 것은 아니다.** 평가에서는 이 차이를 분리한다.

### 10.7 향후 prerequisite의 다른 의미

미래 `OnTop`에서 받침이 **한 번 존재했음**과 **지금 존재함**은 다르다. 모든 미래 dependency에 `a_parent`만 쓰면 이미 무너진 받침을 허용할 수 있다.

이번 구현은 `Holding -> At`의 history prerequisite까지만 지원한다. 미래 invariants/current-state prerequisite는 별도 semantics로 확장해야 하며 지금 구현하지 않는다. 파일명이나 API 이름에서 “all dependencies solved”라는 표현을 쓰지 않는다.

---

## 11. 상태 업데이트, reset, rollout 순서

### 11.1 runtime tensor

```python
phi:           FloatTensor[N, E]
gate:          FloatTensor[N, E]
satisfied:     BoolTensor[N, E]
achieved:      BoolTensor[N, E]
subgoal_done:  BoolTensor[N, M]
prev_root_pos: FloatTensor[N, M, 3]
prev_box_pos:  FloatTensor[N, M, 3]
```

`phi_prev`, `achieved_prev`, `done_prev`를 필요 시 snapshot/copy한다. view alias 때문에 reward 계산 중 prev가 next로 덮어써지지 않게 한다.

### 11.2 reset 초기화

reset된 `env_ids`에 대해서만:

1. 기존 assignment, actor, box, goal, reference-state reset 완료.
2. 올바른 물리/kinematic state로 `phi_init`, `gate_init`, `c_init` 계산.
3. `phi_prev = phi_init`. **0으로 두고 첫 step에 공짜 positive delta를 만들지 않는다.**
4. `prev_root_pos`, `prev_box_pos`를 reset된 실제 위치로 초기화.
5. `achieved` 초기값은 현재 초기 상태를 기준으로 설정한다. `Holding`이 이미 만족된 reference-state에서는 `a_H=1`로 seed할 수 있다. 스킬 이름만 보고 무조건 1로 만들지 않는다.
6. At이 초기부터 valid한 reset에서는 성공 bonus를 지급하지 않는다. 해당 subgoal을 `done_init=True`로 표시한다.
7. 모든 subgoal이 초기부터 done이면 학습용 reset을 다시 샘플한다. 반복 횟수는 제한하고 실패 시 명확한 오류를 낸다. 초기부터 성공하는 episode를 무한 보너스 생성기로 만들지 않는다.
8. 새 관측의 history suffix와 phi가 이 초기 state를 반영해야 한다.

`stateInit: Random`과 carry reference reset을 쓰는 기존 경로에서 중요한 처리다. 새 reward를 위해 RSI를 임의로 없애지 않는다.

### 11.3 한 control step의 순서

현재 코드의 `post_physics_step`은 observation을 reward보다 먼저 계산한다. 새 history suffix를 넣으면 이 순서에서 한 frame 오래된 achieved/done이 들어갈 수 있다.

새 모드에서는 다음 논리 순서를 구현한다.

```text
(1) pre_physics_step
    phi/gate/achieved/done의 s_t snapshot 유지
    root/assigned-box의 previous position 기록
    기존 action/PD 제어 실행

(2) physics step 완료

(3) refresh simulator tensors

(4) post-state phi_next, gate_next, satisfied_next 계산

(5) source displacement velocity / generic progress 계산

(6) gate_prev를 사용하여 edge reward 계산

(7) achieved_prev를 사용하여 valid completion 및 first_success 계산

(8) achieved_next, subgoal_done_next commit

(9) reward buffer, 성공/실패/timeout 정보 commit

(10) s_(t+1)의 node/pose/relation suffix observation 생성

(11) 기존 AMP history/observation, extras, reset accounting 처리
```

기존 순서를 전체 경로에서 무조건 바꾸지 않는다. 새 모드에서 필요한 관계 update helper를 분리하여 legacy 경로가 그대로 동작하게 한다.

### 11.4 reward와 observation 함수는 이력 업데이트를 중복 수행하지 않는다

`evaluate_relations(state)`는 순수 함수다. 여러 번 호출해도 achieved/done이 바뀌면 안 된다.

`advance_relation_runtime(...)`만 한 transition에 한 번 이력을 갱신한다. evaluation metrics나 debug print가 bonus/flag를 변경하면 안 된다.

### 11.5 multi-agent termination

- 한 agent의 subgoal 성공으로 모든 env/agent를 reset하지 않는다.
- scene 내 모든 subgoal의 `done`이 true이면 success termination을 할 수 있도록 새 모드에 구현한다.
- `M=1`이면 그 하나가 valid success할 때 episode 종료.
- 부분 완료 agent는 action을 계속 출력한다. 물리를 강제로 freeze/teleport하지 않는다.
- 기존 실패/fall/timeout 처리는 보존한다.
- success terminal은 새로운 absorbing terminal로 처리하여 다음 value를 잘못 bootstrap하지 않게 한다. 현 `infos['terminate']`가 next-value mask로 쓰임을 확인하라.
- timeout/truncation은 기존의 bootstrap convention과 구분한다.
- per-agent reward/action/done accounting은 기존 `N*M` 순서를 유지하고, scene 성공 mask는 그 scene의 모든 agent slot에 일관되게 broadcast한다.

`all(done)`는 “각 subgoal이 한 번 유효 완료됨”의 뜻이다. 모든 goal state가 동시에 계속 유지된다는 뜻은 아니므로 현재 At 동시 만족률을 별도로 기록한다.

---

## 12. Policy 입력: 새 operator 없이 relation/history를 전달

### 12.1 지금 Carry operator embedding은 넣지 않는다

모든 sample이 Carry이므로 `CARRY` constant one-hot/token/embedding을 추가하지 않는다. `Push/Pull`을 위한 입력 slot도 지금 폭만 미리 늘리지 않는다.

하지만 **achieved와 subgoal_done은 물리 상태만으로 복원할 수 없는 이력**이므로 actor/critic 관측에 포함한다. 이력을 env 안에만 숨겨 reward를 바꾸면 feed-forward policy/value가 구분하지 못하는 상태가 생긴다.

### 12.2 observation suffix 제안 — 이번 구현 결정

기존 node 및 pose block 뒤에 추가한다.

```text
[ Human nodes | Object nodes | Goal nodes | GTA pose records
  | relation_state[E,4] | subgoal_done[M] ]

relation_state columns = [phi, gate, satisfied, achieved]
```

bool은 observation 저장 시 float 0/1로 변환한다.

추가 폭:

$$
D_{suffix}=4E+M=9M,
\quad
D_{new}=247M+37O.
$$

예:

| M | O | token 수 L | legacy obs | 새 obs |
|---:|---:|---:|---:|---:|
| 1 | 1 | 3 | 275 | 284 |
| 2 | 2 | 6 | 550 | 568 |
| 3 | 4 | 10 | 862 | 889 |

물리 node 차원 `223/30/1`, GTA pose7 자체는 바뀌지 않는다. **새 state token을 추가하지 않는다.**

`gate`와 `satisfied`는 phi에서 파생되지만, env와 policy 사이 설정 불일치를 막고 디버깅하기 쉽게 v0 suffix에 명시적으로 싣는다. 두 값의 일관성 테스트를 추가한다.

### 12.3 normalizer 및 clipping

- 기존 H/O의 normalization 범위는 그대로.
- Goal constant, pose7, relation suffix, done은 RMS를 우회한다.
- suffix를 전체 raw feature와 함께 running statistics로 정규화하지 않는다.
- `SceneRunningMeanStd`가 현재 `nodes + L*kinematic_size`만 기대하므로 `extra_passthrough_size` 또는 동등한 구조를 추가한다. default 0으로 legacy assertion 유지.
- `RelationEncoder.forward`의 `obs[:, offset:]` 전체를 pose로 reshape하는 부분을 명시적 pose slice와 suffix slice로 나눈다.
- env/vec-env/player의 `clip_obs` 경로도 확인한다. state/history 값은 그대로 보존되어야 하고 pose를 clipping하는 기존 문제를 새로 만들지 않는다.
- PPO rollout에 suffix가 저장되고 minibatch reshuffle 이후에도 같은 transition의 값이 사용되어야 한다.

### 12.4 상태 relation의 static semantic schema

기존 enum 0~5를 **재번호화하지 않는다**. legacy checkpoint를 보호한다.

새 모드에서 사용하는 relation ID를 별도로 append한다.

```python
REL_HOLDING = 6
REL_AT = 7
```

- legacy encoder의 relation embedding row 수는 기존 6개 그대로 유지.
- state-mode encoder는 8개 row를 사용할 수 있다.
- task edge는 canonical `H_i -> O_i`, `O_i -> G_i` 두 개뿐이다.
- diagonal `REL_SELF`는 유지.
- 새 state graph에서 나머지 pair는 기본 `REL_NONE`로 둔다. 기존 OWN_GOAL/TEAMMATE shortcut을 조용히 task edge로 유지하지 않는다.
- full attention은 유지한다. `REL_NONE`은 attention 차단 mask가 아니다.
- reverse edge를 reward edge로 복제하지 않는다. 방향은 predicate argument 역할이다.
- `Holding=0`이어도 해당 desired relation edge는 존재한다. 목표/필요 관계를 표현하는 edge의 존재와 현재 truth 값을 혼동하지 않는다.

기존 static A2 경로를 보존하기 위해 `build_state_relation_matrix(...)` 같은 새 helper를 추가하고 모드로 분기한다. legacy의 `build_relation_matrix(...)` 출력을 바꾸지 않는다.

### 12.5 dynamic relation feature를 A2 bias에 주입

현재 static A2 계산은 유지하되 새 모드에서만 작은 추가 경로를 둔다.

```text
static semantic bias:
  source type16 + relation type32 + target type16
  -> 기존 A2 edge MLP64
  -> [layers, heads, L, L]

dynamic state bias (새 모드):
  [phi_e, gate_e, satisfied_e, achieved_e, owner_done_e]  # 5D
  -> shared MLP: 5 -> 32 -> 64
  -> layer/head projection
  -> sparse edge별 [N, E, layers, heads]
  -> canonical (src,dst)에 scatter
  -> [layers, N, heads, L, L]
```

최종 bias:

```python
bias_new = static_bias[:, None, ...] + dynamic_bias
```

- dynamic MLP는 모든 relation instance/agent에 공유한다.
- `owner_done_e = subgoal_done[:, edge_owner[e]]`.
- edge state를 계산하는 evaluator는 신경망 parameter가 아니다. 학습되는 것은 이를 받아 attention을 조절하는 MLP다.
- dynamic 출력 projection은 zero-init하여 새 feature가 처음부터 attention을 크게 흔들지 않게 한다.
- 기존 network 전체 linear initializer가 zero-init을 덮어쓸 수 있으므로 최종 initialization 순서를 확인한다.
- zero-init projection이면 첫 backward에서 이전 MLP layer gradient가 0일 수 있다. 두 번째 optimizer step 이후 gradient를 검사한다.
- 추가 token이나 pairwise K/V tensor `[N,heads,L,L,d_head]`를 만들지 않는다. 이미 필요한 attention bias 크기만 추가한다.
- actor와 critic은 각자 별도의 dynamic encoder parameter를 갖는다.

### 12.6 attention layer의 batch bias 처리

현재 `RelationTransformerLayer.forward`는 `rel_bias.unsqueeze(0)`을 가정한다.

새 경로에서는 다음 두 경우를 명시적으로 지원한다.

```python
# legacy/static
rel_bias.shape == [heads, L, L]

# state-mode batch-specific
rel_bias.shape == [N, heads, L, L]
```

rank에 따라 **한 번만** 올바르게 broadcast한다. 이미 batch가 있는 bias에 다시 unsqueeze하지 않는다.

GTA `g_i/g_i^-1`, Q/K/V transform, `head_dim % 4` 규칙, Human readout은 변경하지 않는다.

### 12.7 state+reward가 공유해야 하는 metadata

환경에서 컴파일한 graph와 network에 들어가는 graph가 달라지면 안 된다.

- shared pure-Python schema/helper를 사용한다.
- `relation mode`, `E`, `edge src/dst/relation/owner`, `prerequisite`, `terminal edge`를 동일한 compiler에서 만든다.
- graph는 이번에는 env별로 static/shared하되 state tensor만 env별이다.
- `M/O` 변경 시 `set_entity_counts`가 relation matrix, sparse edge list, suffix size, scatter index를 함께 갱신해야 한다.
- optimizer에 저장되는 parameter 수는 agent/object 수에 의존하지 않아야 한다.

---

## 13. AMP, 물리 regularizer, task reward의 합산

새 reward가 기존 task reward를 **대체**하지만 AMP를 대체하지는 않는다.

환경 reward:

$$
r_{env,i}=r_{task,i}+r_{power,i}+r_{collision,i}+r_{box\_speed,i}.
$$

그 뒤 기존 AMP agent의 `_combine_rewards`에서:

$$
r_{train,i}=w_{task}r_{env,i}+w_{disc}r_{AMP,i}.
$$

이 단계가 이미 존재하므로 env 안에서 AMP를 또 더하지 않는다.

### 13.1 유지할 regularizer

- power penalty: 기존 식 및 config 유지.
- inter-agent collision penalty: 기존 식 및 config 유지.
- box speed penalty: 기존 Transport 함수 속에 들어 있으므로 새 task path에서 자동 소실되지 않도록 **별도 regularizer 성분으로 분리**해 동일 식을 재사용한다.

기존 box speed penalty의 형태:

$$
r_{box\_speed}=-k_{box}\left[1-\exp\{-2(\max(\|v_O\|_2,v_{lim})-v_{lim})^2\}\right].
$$

이는 relation state나 progress가 아니라 기존 motion-quality/안정성 regularizer다. legacy Transport 함수와 새 regularizer를 동시에 호출해서 두 번 지급하지 않는다.

성공한 agent도 scene이 계속되는 동안 물리 regularizer와 AMP는 기존처럼 계산될 수 있다. 해당 subgoal의 task reward만 `done`으로 종료한다. 이러한 차이는 총 return과 종료 유인에 영향을 주므로 로그로 확인한다.

### 13.2 height gate와 pinning 정책

이번 새 기본 모드:

```text
height gate: OFF
legacy pin-to-one: OFF
```

이것은 **간단한 v0를 검증하기 위한 선택**이다. achieved나 completion이 그 물리적 역할을 완전히 대체했다는 뜻이 아니다.

특히 원본은 height mask를 적용한 다음 0.5m 근처 pinning으로 vel을 다시 1로 덮는다. 따라서 원본도 “낮은 box는 항상 velocity reward 0”이라고 단순 설명하면 틀린다.

새 설계에서 목표 근처 아직 Holding/At이 완성되지 않았으면 progress gate가 남아 있을 수 있다. 1.5m/s를 계속 선호하여 overshoot/비비기/착지 방해가 발생하는지 반드시 측정한다. 문제가 확인되면 **공통 near-goal slowdown** 같은 후속 수정이 필요할 수 있다. 이번 명세를 구현하며 임의로 추가하지 말고 결과 보고서에 구체적인 재현과 수정안을 제시한다.

---

## 14. 권장 파일 구성 및 파일별 수정 사항

이름은 아래를 기본으로 한다. 기존 유사 모듈이 있으면 중복 생성하지 말고 역할을 유지하며 통합한다.

### 14.1 새 모듈

```text
tokenhsi/utils/relation_task_spec.py
  relation ID / immutable graph specification / Carry compiler
  simulator와 learning 양쪽에서 import 가능

tokenhsi/env/tasks/multi_agent/relation_reward.py
  torch-only evaluator / progress / gate / reward / achieved update
  Isaac Gym 없이 unit test 가능

tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation.yaml
  새 environment mode, reward/evaluator 설정

tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml
  기존 PPO/AMP/Transformer config를 유지한 새 실험 config

tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh
tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh
  새 config/output 경로를 쓰는 실행 스크립트

tokenhsi/tests/test_relation_reward.py
tokenhsi/tests/test_relation_runtime.py
tokenhsi/tests/test_ma_relation_policy.py
  수식/이력/reset/observation/encoder regression

markdowns/CARRY_STATE_RELATION_V0_IMPLEMENTATION.md
  실제 변경·검증·실행법·남은 한계 보고서
```

### 14.2 기존 파일 수정

**`humanoid_ma_carry.py`**

- mode/config는 `super().__init__` 이전, 관측 크기 산정에 필요한 시점에 파싱.
- 실제 `N/M/O/device`가 확정된 뒤 runtime buffer 할당. 부모 생성자 내부 reset/obs 호출 순서를 확인.
- logical assignment 기반 graph mapping.
- 새 mode의 observation size 및 suffix.
- `_compute_reward`의 legacy/new 분기.
- pure evaluator 호출 및 일관된 update 순서.
- per-env reset 초기화.
- 새 success metrics 및 all-subgoal scene termination.
- 기존 `_success_buf`의 거리 만족 frame count와 새 valid-success를 혼동하지 않게 별도 key 사용.

**`amp_network_builder_ma.py`**

- 새 static state relation schema 및 enum append.
- `EdgeEncoder`의 num relation types를 mode별로 parameterize; legacy default 6.
- suffix parsing 및 dynamic state bias.
- batch bias broadcasting.
- `set_entity_counts`, actor/critic config 전달.
- 기존 GTA 계산 변경 금지.

**`scene_normalizer.py`**

- suffix passthrough size 지원.
- legacy 기본값 0, 기존 assertion/결과 유지.

**`ma_agent.py`, `ma_players.py`**

- 새 observation metadata를 둘 다 동일하게 network/normalizer에 전달.
- rollout memory/suffix 처리.
- success terminal 및 next-value mask 확인.
- player의 hard-coded `walk/carry/handheld/putdown` print를 모드별 처리.
- `REWARD_TERM_NAMES`와 `reward_terms` 열 순서 일치.

### 14.3 function contract

```python
def evaluate_holding(hands_pos, object_pos, *, hand_scale):
    """... -> phi in [0,1], auxiliary geometry."""


def evaluate_at(object_pos, goal_pos, *, near_scale, alpha,
                xy_tolerance, z_tolerance):
    """... -> phi, q_near, q_put."""


def velocity_progress(source_prev, source_next, target_next, dt,
                      *, target_speed, velocity_scale, eps):
    """All edge types share this function. XY, source displacement velocity."""


def relation_gate(phi, *, beta, center):
    """Sigmoid gate only; no state mutation."""


def compute_relation_reward(phi_prev, phi_next, gate_prev, progress,
                            prereq_mask, edge_owner, done_prev,
                            *, state_weight, velocity_weight):
    """Return per-edge components and per-agent sum; do not update history."""


def advance_relation_history(satisfied_next, achieved_prev, done_prev,
                             prereq_mask, subgoal_target):
    """Return achieved_next, done_next, valid_next, first_success."""
```

- 위 함수는 `torch.no_grad`로 호출하거나 reward path에 autograd graph가 쌓이지 않게 한다.
- policy/critic network의 gradient는 정상적으로 유지한다. reward가 no-grad인 것과 actor의 gradient가 끊기는 것은 다른 문제다.
- `.item()`, `.cpu()`, Python per-env loop를 매 step 핵심 경로에서 사용하지 않는다.
- 고정 relation type별 소수의 분기는 가능하지만 env/agent별 반복은 vectorize한다.

---

## 15. Config 예시 — 단일 기준안

아래 block을 새 environment YAML에 넣는다. 원본 물리/asset/초기화 설정은 복사하여 유지한다. 설정 키를 실제로 읽는 구현과 YAML이 반드시 일치해야 한다.

```yaml
env:
  policyObsMode: clean_scene

  relationReward:
    mode: state_relation_v0
    schema_version: 1

    state_delta_weight: 1.0
    velocity_progress_weight: 0.2
    subgoal_success_bonus: 5.0

    soft_gate:
      center: 0.80
      beta: 30.0
    satisfaction_threshold: 0.90

    progress:
      target_speed: 1.5
      velocity_scale: 5.0
      normalization_epsilon: 1.0e-6

    holding:
      hand_distance_scale: 5.0

    at:
      near_distance_scale: 10.0
      near_fraction: 0.5
      putdown_xy_tolerance: 0.10
      putdown_z_tolerance: 0.001

    observation:
      include_relation_state: true
      relation_state_fields: [phi, gate, satisfied, achieved]
      include_subgoal_done: true

    success:
      require_achieved_target_prerequisites: true
      once_per_subgoal: true
      terminate_when_all_subgoals_done: true
      seed_achieved_from_valid_reset_state: true
      suppress_bonus_for_initial_success: true

    diagnostics:
      enabled: true
      log_interval: 30
```

### 15.1 고정 의미를 임의 config 조합으로 흩뜨리지 말 것

위 boolean들은 v0의 의미를 기록하기 위한 설정이다. 예를 들어 `require_achieved_target_prerequisites:false`를 정상 v0 모드로 조용히 허용하지 않는다. 불필요하게 20개의 gate 조합을 제공하는 범용 프레임워크를 만들지 않는다.

unsupported relation/operator는 명확한 오류를 낸다. `OnTop`을 자동으로 `At`으로 바꿔 실행하지 않는다.

### 15.2 Hyperparameter 조정과 구조 변경의 구별

다음은 측정 후 조정 가능한 수치다.

```text
state_delta_weight
velocity_progress_weight
subgoal_success_bonus
gate beta / center
satisfaction threshold
state tolerance / distance scale
```

반면 다음은 별도 설계 변경이다.

```text
delta_phi 대신 absolute phi 지급
state reward에 stop gate를 곱함
current gate 대신 achieved로 dense reward 활성화
height mask를 전체 At에 추가
pin-to-one 복원
Holding에 contact/lift 의미 추가
final release 또는 continuous carry history 강제
```

구조 변경을 “weight tuning”으로 숨기지 않는다.

---

## 16. 핵심 계산 reference code

아래는 **pure PyTorch 수식 reference**다. simulator 통합 코드를 그대로 대체하는 완성 패치는 아니다. 실제 구현은 buffer lifecycle과 metadata를 위 명세대로 연결한다.

```python
# REFERENCE_CORE_BEGIN
from typing import Dict
import torch
from torch import Tensor


def holding_phi(hands_pos: Tensor, object_pos: Tensor,
                hand_scale: float = 5.0) -> Tensor:
    # hands_pos: [..., 2, 3], object_pos: [..., 3]
    mean_hand = hands_pos.mean(dim=-2)
    error = (mean_hand - object_pos).square().sum(dim=-1)
    return torch.exp(-hand_scale * error)


def at_phi(object_pos: Tensor, goal_pos: Tensor,
           near_scale: float = 10.0, alpha: float = 0.5,
           xy_tolerance: float = 0.10,
           z_tolerance: float = 0.001) -> Dict[str, Tensor]:
    error = object_pos - goal_pos
    q_near = torch.exp(-near_scale * error.square().sum(dim=-1))
    q_put_bool = (
        (error[..., :2].square().sum(dim=-1) <= xy_tolerance ** 2)
        & (error[..., 2].abs() <= z_tolerance)
    )
    q_put = q_put_bool.to(object_pos.dtype)
    phi = q_near * (alpha + (1.0 - alpha) * q_put)
    return {"phi": phi, "q_near": q_near, "q_put": q_put}


def soft_gate(phi: Tensor, beta: float = 30.0,
              center: float = 0.8) -> Tensor:
    return torch.sigmoid(beta * (phi - center))


def velocity_progress(source_prev: Tensor, source_next: Tensor,
                      target_next: Tensor, dt: float,
                      target_speed: float = 1.5,
                      velocity_scale: float = 5.0,
                      eps: float = 1.0e-6) -> Tensor:
    if dt <= 0.0:
        raise ValueError("control dt must be positive")
    delta = target_next[..., :2] - source_next[..., :2]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    direction = delta / distance.clamp_min(eps).unsqueeze(-1)
    source_velocity = (source_next[..., :2] - source_prev[..., :2]) / dt
    speed = (direction * source_velocity).sum(dim=-1)
    reward = torch.exp(-velocity_scale * (target_speed - speed).square())
    valid = (distance > eps) & (speed > 0.0)
    return torch.where(valid, reward, torch.zeros_like(reward))


def prerequisite_product(values: Tensor, prereq_mask: Tensor) -> Tensor:
    # values: [N,E]; prereq_mask: [E,E], [child,parent]
    # Empty parent set -> product of ones -> 1.
    factors = torch.where(
        prereq_mask.unsqueeze(0),
        values.unsqueeze(1),
        torch.ones_like(values).unsqueeze(1),
    )
    return factors.prod(dim=-1)


def prerequisite_all(flags: Tensor, prereq_mask: Tensor) -> Tensor:
    factors = torch.where(
        prereq_mask.unsqueeze(0),
        flags.unsqueeze(1),
        torch.ones_like(flags, dtype=torch.bool).unsqueeze(1),
    )
    return factors.all(dim=-1)


def relation_step(
    phi_prev: Tensor,
    phi_next: Tensor,
    progress: Tensor,
    achieved_prev: Tensor,
    done_prev: Tensor,
    prereq_mask: Tensor,
    edge_owner: Tensor,
    subgoal_target: Tensor,
    state_weight: float = 1.0,
    velocity_weight: float = 0.2,
    success_bonus: float = 5.0,
    beta: float = 30.0,
    gate_center: float = 0.8,
    satisfaction_threshold: float = 0.9,
) -> Dict[str, Tensor]:
    # Shapes:
    # phi/progress/achieved: [N,E]
    # done: [N,M]; owner:[E]; target:[M]
    gate_prev = soft_gate(phi_prev, beta, gate_center)
    activation = prerequisite_product(gate_prev, prereq_mask)
    delta_phi = phi_next - phi_prev

    state_component = state_weight * activation * delta_phi
    velocity_component = (
        velocity_weight * activation * (1.0 - gate_prev) * progress
    )
    edge_live = (~done_prev[:, edge_owner]).to(phi_prev.dtype)
    state_component = state_component * edge_live
    velocity_component = velocity_component * edge_live
    edge_reward = state_component + velocity_component

    satisfied_next = phi_next >= satisfaction_threshold
    valid_next = satisfied_next & prerequisite_all(achieved_prev, prereq_mask)
    # History does not keep evolving for an already completed subgoal.
    achieved_next = achieved_prev | (valid_next & edge_live.bool())
    target_valid = valid_next[:, subgoal_target]
    first_success = target_valid & ~done_prev
    done_next = done_prev | target_valid

    agent_reward = torch.zeros_like(done_prev, dtype=phi_prev.dtype)
    agent_reward.scatter_add_(
        dim=1,
        index=edge_owner.unsqueeze(0).expand(phi_prev.shape[0], -1),
        src=edge_reward,
    )
    bonus = success_bonus * first_success.to(phi_prev.dtype)
    agent_reward = agent_reward + bonus
    return {
        "agent_task_reward": agent_reward,
        "edge_reward": edge_reward,
        "state_component": state_component,
        "velocity_component": velocity_component,
        "activation": activation,
        "delta_phi": delta_phi,
        "gate_next": soft_gate(phi_next, beta, gate_center),
        "satisfied_next": satisfied_next,
        "achieved_next": achieved_next,
        "valid_next": valid_next,
        "first_success": first_success,
        "done_next": done_next,
        "success_bonus": bonus,
    }
# REFERENCE_CORE_END
```

### 16.1 Reference 사용 시 주의

위 code block은 수식 대조용이다. 실제 모듈에서는 shape/dtype/device, finite 값, relation index 범위를 검사하라. reset/scene termination/AMP 결합/관측 생성은 이 코드 외부에서 명세대로 처리한다.

- 이 함수는 `achieved_prev`와 `done_prev`를 in-place 수정하지 않는다.
- `subgoal_done` 이후 phi는 계속 물리 상태를 관찰할 수 있지만 history/reward는 재활성화하지 않는다.
- `edge_live.bool()`은 subgoal 활성 여부이며 relation의 current truth가 아니다.
- inference/training 모두 저장된 동일 state metadata를 사용한다.
- 호출자에서 `torch.no_grad()`를 적용한다.

---

## 17. 필수 테스트 — 학습 전에 수식부터 검증

### 17.1 테스트 층을 나눈다

1. **수식/shape unit test:** CPU PyTorch만으로 실행. Isaac Gym import 금지.
2. **network/normalizer regression:** 기존 torch/rl_games 테스트 환경에서 실행.
3. **simulator smoke test:** asset와 Isaac Gym이 있는 환경에서 실행.
4. **학습/행동 검증:** 실제 rollout과 비교 실험. 1~3 통과를 4의 성공으로 보고하지 않는다.

### 17.2 Progress 함수

아래를 `Holding`과 `At` 양쪽 instance에 같은 helper로 검사한다.

| 입력 상황 | 기대 |
|---|---|
| target 방향 속도 1.5m/s | `P≈1` |
| 정지 | `P=0` |
| target 반대 방향 이동 | `P=0` |
| 수직 방향만 이동 | XY 속도가 0이면 `P=0` |
| 목표 좌표와 source XY 동일 | finite, `P=0`, NaN 없음 |
| XY target을 향해 1.0 또는 2.0m/s | `exp(-1.25)` |
| source/target에 동일 XY translation | 같은 P |
| source/target/속도에 동일 yaw rotation | 같은 P |
| target은 정지인데 source만 이동 | source displacement 기준 |
| 이동 target | target velocity를 빼지 않는 현재 convention 확인 |

구버전 walk/transport와의 동등성은 **pinning 바깥**, anti-kick mask가 false인 구간, `onlyVelReward=True`일 때 velocity 성분을 꺼내 비교한다. 전체 함수 합계와 새 generic P가 같다고 assert하지 않는다.

### 17.3 State evaluator

**Holding**

- hands 평균이 box center와 같으면 `phi=1`.
- 거리 증가에 따라 phi 감소.
- goal 위치만 바꿔도 phi_H는 불변.
- hand center midpoint false-positive fixture를 만들고 해당 v0 proxy의 한계를 문서화.
- 원본 handheld geometry와 비교할 때 0.2 scale 및 0.7m mask 여부를 구분.

**At**

- 정확한 목표 center: `q_near=1`, `q_put=1`, `phi_A=1`.
- z가 1mm를 넘어가면 q_put이 0이 되는 경계 검사.
- XY가 0.1m를 넘어가면 q_put이 0이 되는 경계 검사.
- q_put=0인 경우 phi_A가 0.5를 넘지 않음.
- human state만 바꿔도 phi_A는 불변.
- q_put boundary의 discontinuity를 숨기지 말고 숫자로 기록.
- simulator steady state가 1mm 조건에 실제로 도달하는지 별도 smoke test.

### 17.4 Gate / prerequisite

- sigmoid의 유한성 및 단조성.
- `phi=0.8`에서 `g=0.5`.
- `phi=0.9`에서 `g≈0.952574`.
- 빈 prerequisite set의 product=1.
- 두 prerequisite가 있는 synthetic graph에서 product가 올바른지 검사. 이 테스트가 OnTop task 구현을 뜻하지는 않는다.
- `prereq_mask[child,parent]` 축을 거꾸로 해석하지 않았는지 검사.
- parent state가 작은 폭으로 변하면 sigmoid input에 따라 reward가 변하고 hard 0/1 switch를 쓰지 않음.
- sigmoid가 0/1에 정확히 도달한다는 잘못된 assert 금지.

### 17.5 Reward / signed delta

- `phi_next == phi_prev`, source velocity=0이면 **task dense reward=0**. AMP/regularizer 포함 total과 혼동하지 않는다.
- Holding score 하락 시 Holding state 성분 음수.
- At score 증가 시 gate 크기에 비례한 state 성분 양수.
- reward gating은 pre-transition 값이며, next-state 값을 섞지 않음.
- `delta_phi`를 positive clamp하지 않았는지 검사.
- state 성분에 `(1-g_self)`가 곱해져 있지 않은지 검사.
- state+velocity weight 이중 적용 없음.
- sum of edge reward가 scatter된 per-agent reward와 일치.
- relation evaluator를 두 번 호출해도 history가 안 바뀜.

### 17.6 Achieved / success / reset

| 시나리오 | 기대 |
|---|---|
| Holding 이력 없이 At만 성공 | bonus=0 |
| Holding 달성, 다음 step At 성공 | bonus 정확히 1회 |
| At threshold 주변 반복 왕복 | done 이후 bonus 재지급 없음 |
| Holding 달성 후 놓음 | a_H 유지, g_H 감소 |
| 성공 후 손을 뗌 | 이전 Carry task reward 재활성화 없음 |
| 동일 step에 처음 Holding과 At을 동시에 만족 | a_prev 기준이므로 그 step target valid는 false |
| 다음 step도 At 유지, 이전 a_H=true | target valid 처리 |
| RSI로 이미 Holding 상태에서 시작 | 초기 phi와 이력 seed가 일관됨 |
| 초기부터 At valid | 공짜 success bonus 없음 |
| env 0만 reset | env 1의 phi/history/done 불변 |
| assignment 변경 | prev position/phi를 새 물체 state로 초기화 |
| agent A만 완료 | B의 reward/rollout 정상 지속 |
| 모든 agent 완료 | scene success termination 및 bootstrap mask 일관 |

### 17.7 Observation / network

- `legacy mode`의 관측 폭과 출력이 기존과 동일.
- `M=1,O=1`: 284D; `M=2,O=2`: 568D; `M=3,O=4`: 889D.
- H/O RMS는 정상, Goal/pose/suffix는 값 그대로 유지.
- same physical state, different achieved/done의 두 sample이 **서로 다른 suffix**를 갖는지 검사.
- zero-init dynamic projection 때문에 초기 출력이 같을 수 있으므로, history가 모델에 실제 연결되는지는 projection에 비영 값 또는 2회 학습 step을 준 뒤 검사.
- batch별 history가 달라도 다른 env bias에 누출되지 않음.
- tensor shape는 actor/critic 각각 `[N,M,D]` readout, action/value는 기존 flatten convention.
- 관측 permutation에 맞춰 H/O/G, edge index, owner, history를 함께 permutation하면 출력도 해당 순서로 변함.
- agent/object 수가 달라져도 parameter count가 일정.
- GTA 관련 기존 테스트 통과.
- dynamic edge MLP의 gradient는 두 번째 optimizer step 이후 확인.
- actor/critic parameter 공유가 새로 생기지 않음.
- PPO minibatch에서 stored suffix를 쓰고 live env의 최신 버퍼를 읽지 않음.

### 17.8 알려진 exploit에 대한 진단 테스트

아래 테스트는 “현재 설계가 이를 막는다”를 assert하는 용도가 아니다. **현재 규칙이 무엇을 허용하는지 수치로 드러내는 diagnostic**이다.

#### A. Pick-once-then-kick

```text
Holding proxy 달성 -> a_H=true
Holding 감소
At 성공
```

현재 식은 성공 bonus를 허용할 수 있다. 그 사실을 보고서의 limitation에 남긴다. 물리 rollout에서도 이 행동이 나오는지 별도로 검사한다.

#### B. Goal-first-then-touch

```text
Holding 없이 box가 goal로 감
이후 그 자리에서 Holding proxy를 만족
At 상태가 계속 유지됨
```

`a_H*c_A`는 이 경로도 후속 step에 valid로 만들 수 있다. “정확한 행동 순서 검증”이라고 주장하지 않는다.

#### C. Gate-toggle cycle

다음 closed cycle에서 state shaping 합계를 계산한다.

```text
(phi_H, phi_A):
(1,0) -> (1,0.4) -> (0,0.4) -> (0,0) -> (1,0)
```

Holding delta 합은 0이지만 At 증가분은 큰 g_H에서, At 감소분은 작은 g_H에서 계산되어 양의 잔여 reward가 남을 수 있다.

- `gamma=1` 합과 실제 training `gamma=0.99` discounted sum을 둘 다 계산.
- 이로부터 현재 reward를 “potential-based이므로 cycle reward 0”이라고 설명하지 말 것.
- 이번 구현에서 몰래 다른 수식으로 변경하지 않고, 필요 시 후속 수정안으로 제시.

#### D. Near-goal velocity conflict

box가 target XY에 가까우나 z 조건을 아직 만족하지 못하는 rollout을 수집한다.

```text
phi_A / g_A
XY distance / z error
velocity-to-goal reward
overshoot / repeated contact / target crossing count
```

completion gate가 pinning/감속을 자동 대체했는지 **실제로 확인**한다.

### 17.9 테스트 실행 명령 예시

기존 runtime Python 환경을 사용한다.

```bash
PYTHONPATH=tokenhsi python -m pytest -q \
  tokenhsi/tests/test_relation_reward.py \
  tokenhsi/tests/test_relation_runtime.py

PYTHONPATH=tokenhsi python -m pytest -q \
  tokenhsi/tests/test_ma_scene_features.py \
  tokenhsi/tests/test_ma_scene_policy.py \
  tokenhsi/tests/test_ma_relation_policy.py
```

실행 환경에서 torch/rl_games/Isaac Gym이 없으면 이를 구분해 보고한다. 실행하지 않은 테스트를 pass라고 쓰지 않는다.

---

## 18. 로그 / 시각 확인 / 성능 평가

### 18.1 핵심 로그

agent별 또는 집계형으로 다음을 기록한다.

```text
relation/holding_phi
relation/at_phi
relation/holding_gate
relation/at_gate
relation/holding_satisfied
relation/holding_achieved
relation/at_satisfied
relation/at_valid
relation/subgoal_done

reward/holding_state_delta
reward/holding_walk_velocity
reward/at_state_delta
reward/at_transport_velocity
reward/subgoal_success_bonus
reward/task_relation_total
reward/power
reward/agent_collision
reward/box_speed
reward/env_total
reward/amp
reward/train_combined

geometry/hand_midpoint_error
geometry/left_hand_box_distance
geometry/right_hand_box_distance
geometry/root_box_distance_xy
geometry/box_goal_distance_xy
geometry/box_goal_error_z
geometry/q_near
geometry/q_put
geometry/box_bottom_height

success/raw_geometric_goal
success/history_qualified_goal
success/first_success_count
success/current_all_goals_satisfied
success/all_subgoals_ever_completed
success/time_to_first_holding
success/time_to_valid_goal
```

가능하면 episode별 time-series CSV를 일부 env에 대해 저장한다. 수천 env 전체를 매 step CPU로 복사하지 않는다. debug sampling과 interval을 사용한다.

### 18.2 물리 진단과 reward proxy의 구분

hand contact, object lift, actual release, stable placement 여부를 얻을 수 있으면 별도 진단값으로 저장한다. `net_contact_force`만으로 정확한 상대 물체와의 grasp를 판정할 수 있다고 가정하지 않는다. 해당 sensor/API가 제공하는 의미를 확인한다.

현재 측정 가능한 정보로 확정하지 못하면 `unknown`으로 기록하고 영상 확인 항목으로 남긴다. phi 기반 성공을 “완벽한 Carry”로 대체 보고하지 않는다.

### 18.3 단계별 rollout 확인

최소 다음 장면을 확인한다.

```text
1. box까지 접근
2. 손을 box 근처로 가져감
3. 실제 box 들기/운반
4. goal XY 근처 감속 및 자세 조정
5. box-near -> putdown proxy 전환
6. success bonus 1회
7. release / 다음 agent 동작 중 behavior
8. 잘못 잡음 / 중간 drop / 다시 접근
```

주요 관찰 질문:

- hand midpoint만 맞추고 물체를 들지 않는가?
- Holding gate 경계에서 앞뒤로 반복하는가?
- goal 근처에서 1.5m/s reward 때문에 멈추지 못하는가?
- intermediate achievement 후 kick/throw로 끝내는가?
- At putdown z threshold를 실제로 넘는가?
- 성공 이후 reward가 재켜지거나 bonus가 반복되는가?
- 한 agent 완료가 다른 agent의 관측/버퍼를 망가뜨리는가?

### 18.4 비교 실험 원칙

1. **Legacy regression:** 새 모드를 끈 채 기존 checkpoint/test path가 그대로 동작하는지 확인.
2. **Scalar shadow evaluation:** 같은 저장된 transition 또는 scripted tensor fixture에 legacy/new reward를 함께 계산해 성분을 비교. 실제 학습 reward에 두 개를 동시에 더하지 않는다.
3. **M=1 smoke:** reward/observation/reset을 충분히 확인.
4. **M=2 independent Carry smoke:** assignment, scene reset, agent별 success 확인.
5. 승인된 compute 범위 안에서 학습 비교. seed, 초기 분포, frame budget, PPO 설정을 맞춘다.

한 번에 reward, optimizer, batch, architecture width를 모두 바꾼 뒤 어느 변경 덕분인지 단정하지 않는다. 새 history 입력은 objective의 memory를 노출하기 위한 변경이므로 반드시 보고한다.

평균 reward만 아니라 valid success, 실제 carry quality, stall/drop/overshoot, wall-clock throughput을 함께 비교한다. 신뢰도 있는 성능 주장은 반복 실험을 거친 뒤에만 한다.

---

## 19. 실행 스크립트 및 checkpoint

### 19.1 새 train script

기존 `ma_carry_train.sh`와 동일하게 `runtime_env.sh`를 source하고, positional argument 의미를 유지한다.

```text
sh tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh \
    [num_agents] [num_envs] [num_objects]
```

새 config와 output 경로만 선택한다.

```text
cfg_env:
  tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation.yaml
cfg_train:
  tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml
output:
  output/ma_carry_relation_v0/<unique_run_name>
```

기존 script의 `MAX_ITERATIONS`와 runtime GPU 선택 방식을 유지한다. 고정 GPU 번호를 코드에 박지 않는다.

### 19.2 Smoke 명령 예시

아래는 새 script 구현 후의 예시다. 현재 존재하는 스크립트라고 주장하지 않는다.

```bash
MAX_ITERATIONS=5 sh tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh 1 32 1
MAX_ITERATIONS=5 sh tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh 2 32 2
```

sample 수가 train config의 minibatch와 맞지 않으면 **smoke 전용 config**에서 batch 크기를 줄이고 그 사실을 기록한다. production PPO 설정을 몰래 바꾸지 않는다. 위 명령이 그대로 돌아가도록 script가 smoke config를 지원하거나, 실제 유효한 명령을 구현 보고서에 제시하라.

예: `M=1,N=32,horizon=32`이면 per-agent rollout은 1024 sample이다. 기본 minibatch 8192가 맞지 않으므로 smoke에서는 minibatch 1024 이하의 유효한 값이 필요하다.

### 19.3 기존 checkpoint 보존

- 기존 checkpoint는 `legacy_tokenhsi` 경로에서 strict load가 계속 가능해야 한다.
- 새 모드 checkpoint에는 `reward_mode`, `schema_version`, suffix layout, relation taxonomy, 해당 config를 기록한다.
- 새 입력/semantic ID가 들어가므로 old/new checkpoint를 무조건 동일하다고 취급하지 않는다.
- `strict=False`로 모든 mismatch를 무시하는 방식 금지.

### 19.4 선택적 warm-start

기존 weight에서 새 모드로 warm-start를 제공한다면 명시적 변환 경로로 만든다.

```text
copy:
  physical tokenizers
  type embeddings
  GTA/Transformer layers
  action head
  가능한 기존 A2 weight

initialize:
  HOLDING relation row <- 기존 OWN_OBJECT row를 초기값으로 사용할 수 있음
  AT relation row      <- 기존 OBJECT_GOAL row를 초기값으로 사용할 수 있음
  dynamic state bias   <- zero output projection
```

이것은 semantic 초기화 힌트일 뿐 forward output 동일성을 보장하지 않는다. 새 directed graph와 history 입력이 기존 ownership graph와 다르기 때문이다.

누락/추가 parameter 이름을 보고한다. 구조가 바뀐 optimizer state는 무분별하게 복원하지 않는다. 새 training config는 기본적으로 기존처럼 `load_checkpoint: False`를 유지할 수 있으며, warm-start는 별도 명령으로 요청할 수 있게 한다.

---

## 20. 후속 확장 — 이번에는 설명만

### 20.1 상태 relation `OnTop`, `Beside`

향후 추가 가능한 형태:

```text
OnTop(X,Y)
Beside(X,Y)
```

공통 reward engine은 그대로 두고 relation state evaluator를 추가하는 방향이다. 그러나 실제 적용에는 물체 크기, support, target anchor, 이동 대상의 현재 상태, invariant semantics 등을 다시 정해야 한다.

이번에 하지 않는 것:

- 실제 evaluator 구현.
- graph preset이나 default edge 생성.
- OnTop을 위해 O_A를 moving goal로 바꾸는 환경 수정.
- 새로운 object/goal token 축소 또는 추가.
- A place -> B stack을 수행하는 계단 task.
- 현재 두 edge Carry의 성공을 OnTop 성공으로 해석.

미래 상태 이름은 문서에만 남긴다. 지금 unsupported name을 받으면 `NotImplementedError` 또는 config validation 오류를 명확히 낸다.

### 20.2 행동 operator `Push`, `Pull`

향후 edge-conditioned network 입력에 relation과 **별개 attribute**를 추가할 수 있다.

```text
relation = At
operator = Carry / Push / Pull
```

- `At_Carry`, `At_Push`, `At_Pull`처럼 relation semantics를 행동별로 복제하지 않는다.
- 현재 Carry-only에서는 operator embedding을 추가하지 않는다.
- 나중에 여러 행동이 동일 relation graph/target을 공유할 때 operator 입력이 구분 정보를 준다.
- operator label만 입력한다고 push/pull 방식이 자동 보장되는 것은 아니다. 실제 행동별 interaction constraints 또는 학습 데이터/보상 정의가 필요할 수 있다.
- pull에는 단순 압축 contact만으로 충분하지 않을 수 있으며 grasp/hook/접촉 모델을 확인해야 한다.
- 앞·뒤 interaction target을 명시할지 학습에 맡길지는 그때 평가한다.

### 20.3 보류하는 일반화

```text
multi-attribute relation slots
operator token / operator edge embedding
new learned target-point generator
additional Near/Walk/Reach state edges
current-invariant / historical prerequisite 혼합 계약
continuous grasp history / strict release success
PBRS 또는 다른 reward objective로의 교체
world model / future rollout / planner
```

확장을 쉽게 하는 인터페이스는 유지하되, 이번 범위를 구현하기 위해 이 기능들까지 만들지 않는다.

---

## 21. Codex가 완료 후 제출할 결과

`markdowns/CARRY_STATE_RELATION_V0_IMPLEMENTATION.md`에 다음을 기록한다.

### 21.1 코드 변경

- 작업 시작 branch/HEAD와 실제 변경 base.
- 파일별 변경 요약.
- legacy 보존 방식과 새 mode 진입 경로.
- 새 관측 layout/폭, static relation graph, dynamic state bias shape.
- 실제 구현한 reward/evaluator 수식 및 문서 대비 변경 사항.

### 21.2 검증

- 실행한 테스트 명령 및 결과.
- 실행하지 못한 테스트와 이유.
- pure reward fixtures 및 알려진 exploit diagnostic 결과.
- simulator smoke 결과, 영상/로그 경로.
- one-time bonus 및 reset/assignment 검증 결과.
- legacy checkpoint regression 결과.

### 21.3 실행

- 새 train/test 명령.
- smoke 전용 valid minibatch 설정.
- checkpoint 호환성/변환 명령이 있다면 그 사용법.
- 로그 key와 해석.

### 21.4 남은 한계

반드시 다음을 포함한다.

```text
Holding proxy != verified grasp
At geometric proxy != verified stable released placement
achieved history != continuous Carry execution
soft gate != guaranteed switching / maintenance
no pinning != guaranteed safe stopping
signed delta with gates != guaranteed cycle-free reward
```

구현 완료와 학습 성공을 구분한다. 모델이 아직 학습되지 않았다면 “구현 및 테스트 완료, 학습 성능 미검증”이라고 쓴다.

---

## 22. 완료 체크리스트

- [ ] baseline 코드/기존 config 및 실험 결과 보존.
- [ ] 새 `state_relation_v0` mode와 별도 실행 config/script.
- [ ] entity/graph compiler는 실제 논리 assignment 사용.
- [ ] Carry edge는 Holding과 At 두 개뿐.
- [ ] progress는 공통 XY source velocity-only 함수.
- [ ] state는 signed delta_phi.
- [ ] Holding geometry 및 At near+putdown state 구현.
- [ ] soft gate는 공통 sigmoid.
- [ ] prerequisite product는 pre-transition current gate 사용.
- [ ] success는 target current satisfaction × prior achieved prerequisite.
- [ ] bonus는 subgoal별 최초 1회.
- [ ] post-success reward 재활성화 없음.
- [ ] reset/RSI/assignment 초기화가 first-step reward를 오염시키지 않음.
- [ ] achieved/done이 actor와 critic 관측에 들어감.
- [ ] suffix는 RMS/pose reshape를 오염시키지 않음.
- [ ] batch dynamic relation bias와 GTA가 함께 정상 작동.
- [ ] per-agent reward/scene PPO grouping 유지.
- [ ] AMP/physics regularizer 이중 적용 없음.
- [ ] 원본 box speed penalty의 우발적 소실 없음.
- [ ] success terminal과 timeout의 bootstrap 처리 구분.
- [ ] unit/regression/smoke 테스트와 limitation diagnostic 보고.
- [ ] OnTop/Beside/Push/Pull은 이번에 구현하지 않음.
- [ ] 동작 성능을 실제 측정 없이 보장했다고 주장하지 않음.

---

## 23. 확인한 소스 근거

아래는 명세 작성 시 확인한 source permalink다. 문서에 제시한 **새 설계의 성능을 증명하는 출처가 아니라**, 기존 코드의 사실을 확인하는 근거다.

1. [원격 기준 commit](https://github.com/KSH3880/CVPR2027/commit/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76)
2. [HumanoidMACarry 초기화·관측 크기](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py#L1-L230)
3. [Scene observation·reward·pre/post physics·metrics](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py#L650-L860)
4. [기존 handheld/walk/transport/putdown 함수](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py#L1550-L1710)
5. [기존 relation enum/matrix/A2 encoder 및 attention](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/learning/multi_agent/amp_network_builder_ma.py#L40-L250)
6. [RelationEncoder observation parsing 및 actor/critic](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/learning/multi_agent/amp_network_builder_ma.py#L530-L840)
7. [SceneRunningMeanStd](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/learning/multi_agent/scene_normalizer.py)
8. [MAAgent 관측·PPO·AMP 처리](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/learning/multi_agent/ma_agent.py#L1-L240)
9. [MAPlayer의 network config 및 reward 출력](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/learning/multi_agent/ma_players.py)
10. [Carry 환경 config](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml)
11. [PPO/AMP/Transformer train config](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml)
12. [기존 train script](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/scripts/multi_agent/ma_carry_train.sh)
13. [기존 scene/GTA tests](https://github.com/KSH3880/CVPR2027/blob/9b68fca23fb65842c8a9a64cfd6fc066c8dcee76/tokenhsi/tests/test_ma_scene_policy.py)

---

## 24. 마지막 작업 지시

본 명세의 우선순위는 다음과 같다.

```text
1. 기존 실행·checkpoint를 보존한다.
2. 합의한 Carry-only 두-edge reward를 정확하게 구현한다.
3. history가 있는 objective에 맞게 observation/reset/termination을 일관되게 연결한다.
4. 수식 및 알려진 실패 경로를 테스트한다.
5. 검증한 범위만 보고한다.
```

설계가 수식 몇 줄로 표현된다는 이유로 물리 동작까지 자동 보장한다고 가정하지 말 것. 반대로 모든 가능한 미래 행동을 지금 구현하며 scope를 키우지도 말 것.

**이번 결과물은 `Holding + At`, 공통 velocity progress, soft prerequisite, achieved-qualified target success를 실제 `edge_a2_gta` 코드에서 실행할 수 있는 Carry-only v0다.**

---

## 부록 A. 명세 작성 시 수행한 검증 범위

이 문서의 `REFERENCE_CORE` 코드에 대해 CPU PyTorch로 아래 11개 sanity-check 그룹을 실행했다.

```text
PASS  velocity target/stop/reverse/speed fixtures
PASS  zero-distance and XY-only fixtures
PASS  state evaluator exact and putdown boundary fixtures
PASS  soft-gate calibration fixtures
PASS  no prerequisite history -> no success bonus
PASS  one-time success, post-success release and no reactivation
PASS  pre-transition achieved snapshot ordering
PASS  stationary state without completion -> zero task reward
PASS  signed holding delta preserved
PASS  known gated-delta cycle quantified (not claimed eliminated)
PASS  multi-agent owner scatter and shape
```

§17.8-C의 알려진 cycle에서, velocity를 0으로 두었을 때 reference state reward는 다음과 같았다.

```text
transition rewards:
[+0.399010951, -1.0, approximately 0.0, +1.0]

undiscounted sum = +0.399010951
gamma=0.99 sum   = +0.379309951
```

즉 해당 예제는 **현재 수식에 reward-cycle 가능성이 남아 있음을 확인한 결과**이지 그것을 해결했다는 결과가 아니다. generic rule의 구현 정확성과 objective의 바람직함은 별개로 검증해야 한다.

원격 저장소를 실제로 수정하거나, Isaac Gym simulation을 실행하거나, policy를 학습한 것은 아니다. Codex는 §17~21의 저장소 통합 테스트와 실제 행동 검증을 별도로 수행해야 한다.
