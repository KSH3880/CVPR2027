# Clean Scene Node / Geo 입력 명세

> 이 문서는 이전 15-D pairwise Geo ablation의 기록이다. 현재 GTA clean-scene 경로는
> [`ma_clean_scene_gta.md`](ma_clean_scene_gta.md)를 따른다.

이 문서는 현재 `edge_geo` 작업 트리의 `policyObsMode: clean_scene` 입력을 기준으로 한다.
`legacy_multirow`는 기존 A1 baseline의 230-D Human / 39-D Object / 6-D Target 구조를
그대로 사용하며, 아래 변경의 대상이 아니다.

## 1. 설계 원칙

```text
Node
= entity 자신의 local/self state
+ entity 자신의 shared scene pose

Geo
= 두 entity 사이의 directed relative geometry

Semantic edge (A2)
= source type + task relation + target type
```

여기서 scene 좌표는 Isaac Gym 전체 world 좌표가 아니라 다음 env-local 좌표다.

```text
p_env = p_world - env_origin
```

Human, Object, Target 모두 같은 env origin과 같은 위치 scaling을 사용한다.

---

## 2. Human node: 227-D

최종 순서는 다음과 같다.

```text
Human 227-D
= heading-local body/self state 222-D
+ shared scene pose               5-D
```

### 2.1 Body/self state: 222-D

원본 TokenHSI maximum-coordinate self observation 223-D에서 첫 번째 root-height
scalar를 제거한다. 15개 body 기준 상세 구성은 다음과 같다.

| 구간 | 차원 | 좌표계 / 의미 | 정규화 |
|---|---:|---|---|
| root를 제외한 body position | 14 x 3 = 42 | Human root-relative, Human heading-local | Human RMS |
| 전체 body rotation | 15 x 6 = 90 | Human heading-local tangent/normal 6-D | Human RMS |
| 전체 body linear velocity | 15 x 3 = 45 | Human heading-local | Human RMS |
| 전체 body angular velocity | 15 x 3 = 45 | Human heading-local | Human RMS |
| 합계 | 222 | | |

Human self state는 모든 Human slot이 하나의 `RunningMeanStd(222)`를 공유한다.

### 2.2 Shared scene pose: 5-D

```text
[
  (root_world_x - env_origin_x) / arena_scale,
  (root_world_y - env_origin_y) / arena_scale,
  (root_world_z - env_origin_z),
  cos(root_heading),
  sin(root_heading),
]
```

이 5-D에는 RunningMeanStd를 적용하지 않는다. XY에는 모든 entity가 공유하는 fixed
scale만 적용하고, Z는 env-local metre 단위를 유지한다. `cos/sin`도 그대로 유지한다.

---

## 3. Object node: 39-D

최종 feature 순서는 다음과 같다.

| 구간 | 차원 | 좌표계 / 의미 | 정규화 |
|---|---:|---|---|
| object-local linear velocity | 3 | `R_object^-1 v_world` | Object RMS |
| object-local angular velocity | 3 | `R_object^-1 w_world` | Object RMS |
| object-local bbox corners | 8 x 3 = 24 | box 중심 기준 local corner | Object RMS |
| env-local object position | 3 | scaled XY + metre Z | 없음 |
| full object orientation | 6 | world/env tangent/normal 6-D | 없음 |
| 합계 | 39 | | |

앞의 local 30-D는 모든 Object slot이 하나의 `RunningMeanStd(30)`를 공유한다.
뒤의 scene pose 9-D에는 RMS를 적용하지 않는다.

Object scene position은 Human/Target과 동일하게 계산한다.

```text
[
  (object_world_x - env_origin_x) / arena_scale,
  (object_world_y - env_origin_y) / arena_scale,
  (object_world_z - env_origin_z),
]
```

Object orientation 6-D는 quaternion을 직접 넣지 않고 TokenHSI의
`quat_to_tan_norm`과 같은 tangent/normal 표현을 사용한다.

---

## 4. Target node: 3-D

Carry Target은 orientation과 dynamics가 없는 point target이다.

```text
Target 3-D
= [
    (target_world_x - env_origin_x) / arena_scale,
    (target_world_y - env_origin_y) / arena_scale,
    (target_world_z - env_origin_z),
  ]
```

Target node 전체에는 RunningMeanStd를 적용하지 않는다. Human/Object scene position과
동일한 deterministic scaling만 사용한다.

---

## 5. Node normalization 요약

```text
Human 227-D = RMS(앞 222-D) + passthrough(뒤 5-D)
Object 39-D = RMS(앞 30-D)  + passthrough(뒤 9-D)
Target  3-D = passthrough(전체 3-D)
```

Scene pose를 entity별 RMS로 정규화하지 않는 이유는 같은 물리 위치가 Human/Object/Target
타입에 따라 다른 숫자로 바뀌는 것을 막기 위해서다. 회전의 `cos/sin` 및 6-D 구조도 그대로
보존한다.

---

## 6. Stored scene observation

`M = Human 수`, `O = Object 수`, `L = 2M + O`일 때 저장 순서는 다음과 같다.

```text
[
  M x Human node 227-D,
  O x Object node 39-D,
  M x Target node 3-D,
  L x Geo-construction raw state 13-D,
]
```

전체 observation width:

```text
M*227 + O*39 + M*3 + (2M+O)*13
= 256M + 52O
```

예:

| M | O | token 수 L | scene observation width |
|---:|---:|---:|---:|
| 1 | 1 | 3 | 308 |
| 2 | 2 | 6 | 616 |
| 2 | 3 | 7 | 668 |
| 3 | 4 | 10 | 976 |

---

## 7. Geo construction용 raw state: entity당 13-D

13-D block은 node tokenizer 입력이 아니며 forward에서 Geo를 계산하기 위한 원본 상태다.
Node RMS를 우회하고 별도의 RunningMeanStd도 적용하지 않는다.

```text
raw kinematics 13-D
= position 3 + quaternion 4 + linear velocity 3 + angular velocity 3
```

| Entity | position | quaternion | linear/angular velocity |
|---|---|---|---|
| Human | unscaled env-local root XYZ | full root orientation | world-frame 실제 값 |
| Object | unscaled env-local object XYZ | full object orientation | world-frame 실제 값 |
| Target | unscaled env-local target XYZ | identity `[0,0,0,1]` | zeros(6) |

Human/Object quaternion을 full orientation으로 저장하는 이유는 relative rotation에서 roll/pitch까지
사용하기 위해서다. Position/velocity 계산 시에는 이 quaternion에서 heading만 별도로 추출한다.

---

## 8. Pairwise Geo input: 15-D

모든 directed pair `(i -> j)`가 같은 schema와 같은 shared Geo Encoder를 사용한다.

```text
Geo(i,j) 15-D
= relative position          3
+ relative rotation          6
+ relative linear velocity   3
+ relative angular velocity  3
```

### 8.1 H/O source

Human과 Object source에서는 position/velocity에 source의 heading만 사용한다.

```text
dp_ij = R_heading(i)^-1 (p_j - p_i)
dv_ij = R_heading(i)^-1 (v_j - v_i)
dw_ij = R_heading(i)^-1 (w_j - w_i)
```

Object가 기울어져도 forward/left/height 좌표가 같이 기울어지지 않는다.

Relative rotation만 source/target의 full orientation을 사용한다.

```text
q_rel = q_i^-1 q_j
dR_ij = quat_to_tangent_normal(q_rel)  # 6-D
```

### 8.2 Target source

Target에는 heading이 없으므로 T-source relative position은 shared env-local 축의 단순 차이다.

```text
dp_Tj = p_j - p_T
```

### 8.3 Target 및 self masking

Target이 source 또는 target인 모든 pair는 position만 유지한다.

```text
Geo(H,T) = [actual dp, zeros(12)]
Geo(O,T) = [actual dp, zeros(12)]
Geo(T,H) = [actual dp, zeros(12)]
Geo(T,O) = [actual dp, zeros(12)]
Geo(T,T) = [actual dp, zeros(12)]
```

모든 self edge는 `REL_SELF`가 의미를 담당하도록 전체 15-D를 0으로 만든다.

```text
Geo(i,i) = zeros(15)
```

### 8.4 Pair별 actual / zero 표

| Directed pair | position 3 | rotation 6 | linear velocity 3 | angular velocity 3 |
|---|---|---|---|---|
| H -> H | actual, source heading | actual, full rotation | actual | actual |
| H -> O | actual, source heading | actual, full rotation | actual | actual |
| H -> T | actual, source heading | zero | zero | zero |
| O -> H | actual, source heading | actual, full rotation | actual | actual |
| O -> O | actual, source heading | actual, full rotation | actual | actual |
| O -> T | actual, source heading | zero | zero | zero |
| T -> H | actual, scene axes | zero | zero | zero |
| T -> O | actual, scene axes | zero | zero | zero |
| T -> T | actual, scene axes | zero | zero | zero |

표의 모든 diagonal pair는 all-zero self 예외를 따른다.

---

## 9. Geo scaling

Geo에는 RunningMeanStd를 적용하지 않고 물리량별 fixed scaling만 적용한다.

| Geo component | scale | 현재 `arena_scale=5`일 때 |
|---|---|---|
| relative position XY | `1 / arena_scale` | `x 0.2` |
| relative position Z | `1.0` | metre 유지 |
| relative rotation 6-D | `1.0` | 원래 `[-1,1]` 범위 유지 |
| relative linear velocity | `0.25` | 약 4 m/s를 1로 매핑 |
| relative angular velocity | `0.25` | 약 4 rad/s를 1로 매핑 |

Train config의 `position_scale: null`은 network 생성 시 task의 실제 `arena_scale`을 받아
자동으로 다음 값을 만든다.

```text
position_scale = [1/arena_scale, 1/arena_scale, 1]
```

따라서 `envSpacing`을 바꿨을 때 `0.2` 같은 상수를 수동으로 고칠 필요가 없다.

RMS를 사용하지 않는 이유:

- Target/self padding의 정확한 0을 유지한다.
- M/O 변화로 pair type 비율이 달라져도 입력 의미가 변하지 않는다.
- rotation 6-D의 구조를 통계 정규화로 변형하지 않는다.
- checkpoint를 다른 entity count에 적용할 때 running statistics 의존성을 만들지 않는다.

---

## 10. Geo ablation

모든 모드에서 tensor shape과 Geo Encoder 입력은 15-D로 고정한다.

```yaml
geometry:
  mode: full15  # position3 | pose9 | full15
```

| mode | 실제 사용 component | zero masking |
|---|---|---|
| `position3` | position 3 | dimensions 3:15 |
| `pose9` | position 3 + rotation 6 | dimensions 9:15 |
| `full15` | position + rotation + linear/angular velocity | Target/self validity mask만 적용 |

모든 ablation이 같은 `Linear(15, 64)` Geo Encoder를 사용하므로 parameter shape이 바뀌지 않는다.

---

## 11. A2 semantic edge와 Geo의 결합

A2 semantic edge는 Geo와 별도 branch다.

```text
source entity type embedding 16 ─┐
relation type embedding      32 ─┼─ concat 64 -> shared MLP -> semantic edge 64
target entity type embedding 16 ─┘

semantic edge 64 -> layer/head projection -> scalar relation bias
```

A2 입력은 discrete embedding이므로 RMS나 fixed physical scaling을 적용하지 않는다.
최종 bias projection은 0으로 초기화된다.

최종 attention:

```text
score(i,j) = Q_i K_j / sqrt(d_head)
           + A2_semantic_bias(i,j)
           + Geo_score(i,j)

message(i,j) = V_j + Geo_message(i,j)
```

`REL_NONE`이어도 Geo는 실제 물리 geometry를 계산한다. A2는 별도로 source/target entity type과
`NONE` relation을 embedding한다.

Config:

```yaml
relation_bias: true
relation_bias_mode: edge_mlp  # edge_mlp=A2, lookup=A1
```

`legacy_multirow`는 config와 무관하게 A1 lookup으로 강제된다.

---

## 12. 학습 실행 후 확인할 항목

다음 항목은 unit test만으로 적정 범위를 확정할 수 없으므로 실제 rollout에서 확인한다.

1. Scaled `dp`, `dv`, `dw`의 mean/std와 absolute p95/p99를 기록한다. p99가 지속적으로
   2~3보다 훨씬 크면 velocity scale을 다시 조정한다.
2. Human/Object quaternion norm이 1에 가까운지, Target quaternion이 정확히 identity인지 확인한다.
3. Object local-X가 거의 수직인 심한 tilt에서 projected heading이 불안정해지는지 확인한다.
4. Target 관련 dimensions 3:15 및 모든 self Geo가 실제 rollout에서도 정확히 0인지 확인한다.
5. Actor/Critic의 Geo Encoder, Geo score/message head에 유한한 nonzero gradient가 들어오는지 확인한다.
6. A2 final projection은 0 초기화이므로 첫 backward에서 A2 embedding/MLP gradient가 0인 것이
   정상이다. Projection update 이후에는 embedding/MLP gradient가 nonzero여야 한다.
7. `clip_observations`를 유한값으로 설정하지 않는다. Wrapper 단계에서 raw 13-D quaternion이나
   velocity를 먼저 clip하면 Geo 계산이 손상된다.
8. 기존 clean-scene checkpoint는 node width 및 Geo/A2 semantics가 달라 strict-load 호환되지
   않는다. 새 checkpoint로 학습한다.

---

## 13. 관련 구현

- Node 생성: `tokenhsi/env/tasks/multi_agent/humanoid_ma.py`
- Object/Target 및 raw 13-D 생성: `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`
- Scene position 공통 변환: `tokenhsi/env/tasks/multi_agent/scene_features.py`
- 부분 RMS 정규화: `tokenhsi/learning/multi_agent/scene_normalizer.py`
- A2 semantic edge 및 Geo 15-D: `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`
- 기본 train config: `tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml`
- 검증: `tokenhsi/tests/test_ma_scene_policy.py`, `tokenhsi/tests/test_ma_scene_features.py`
