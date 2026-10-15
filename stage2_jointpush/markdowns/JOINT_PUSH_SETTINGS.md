# Joint Push 환경·모델·학습 설정

기준일: 2026-10-15. 대상은 **2인·60kg·직선 Push·agent 시작 위치/방향 360° 랜덤·AT soft gate** 학습이다.

환경 값은 아래 run의 `relation_config.yaml`, 학습 seed는 `random_seed.txt`, 실행 동작은 현재 구현을 기준으로 정리했다.

- 실험 이름: `JointPushAlignedSurround60SoftGateCa4`
- run: `runs/jointpush/aligned_surround60_softgate_training/JointPushAlignedSurround60SoftGateCa4_14-11-37-12/`
- env config: `ports/koo_push/tokenhsi/data/cfg/multi_agent/joint_push_aligned_surround60_softgate_ca4.yaml`
- train config: `ports/koo_push/tokenhsi/data/cfg/train/rlg/joint_push_aligned_surround60_softgate_ca4.yaml`
- task: `HumanoidMAJointPush`

이 문서의 코드·config 경로는 별도 표시가 없으면 `stage2_jointpush/` 기준이다. 실행 명령은 `/home/hwanhee/juan/CVPR2027`에서 사용한다.

## 1. 태스크와 학습 범위

두 humanoid가 동일한 상자의 같은 면에 접근해 자리를 나누고, 함께 상자를 하나의 목표 위치까지 미는 태스크다. 목표 방향은 초기 상자 정면과 정렬한다. 의도적으로 목표 방향을 틀어 놓는 곡선 Push 조건은 포함하지 않는다.

Stage1에서 가져온 사람·장면 표현과 Push 동작 초기값을 활용하고, Stage2의 task grounding과 cross attention을 통해 두 agent의 협동을 학습한다. 기존 Stage2 checkpoint를 이어받는 resume로 시작한 run은 아니다. Stage1 전이와 함께 새 Stage2 optimizer·counter·CA4에서 시작한다.

## 2. 실행 환경과 시간 간격

| 항목 | 설정 |
|---|---|
| 시뮬레이터 | Isaac Gym, PhysX GPU 시뮬레이션 |
| GPU | physical GPU **6** |
| 실행 준비 | `ports/koo_push/tokenhsi/scripts/multi_agent/runtime_env.sh` |
| 실행 Python 환경 | `tokenhsi_juan`, Python 3.8 계열 |
| 병렬 환경 | 2048 |
| 환경당 agent | 2 |
| 전체 agent 슬롯 | 4096 |
| `envSpacing` | 5 |
| 물리 timestep | 1/60초 |
| `substeps` | 2 |
| `controlFrequencyInv` | 2 |
| 정책 제어 간격 | 약 1/30초, 30Hz |
| episode 길이 | 600 control steps, 약 20초 |
| 실제 학습 seed | **6397** |
| YAML seed 기본값 | -1: 신규 실행에서 seed를 생성하며 `SEED`로 지정 가능 |
| mixed precision / multi-GPU | false / false |

MPS를 사용하는 실행 환경이다. 제공하는 joint push wrapper는 GPU6을 지정한다. MPS 서버 설정 변경은 학습/평가 절차에 포함하지 않는다.

### PhysX 설정

| 항목 | 값 |
|---|---:|
| `solver_type` | 1 |
| `num_threads` | 4 |
| `num_position_iterations` | 4 |
| `num_velocity_iterations` | 0 |
| `contact_offset` | 0.02m |
| `rest_offset` | 0m |
| `bounce_threshold_velocity` | 0.2 |
| `max_depenetration_velocity` | 10.0 |
| `default_buffer_size_multiplier` | 10.0 |
| `contact_collection` | 2 |
| `max_gpu_contact_pairs` 기본값 | 8 × 1024 × 1024 |

물리 timestep과 contact-pair 기본값은 `utils/config.py::parse_sim_params`에서 설정하고 env YAML을 적용한다. CLI override가 있으면 해당 실행값을 별도로 확인해야 한다.

## 3. 상자·바닥·object 슬롯

### 실제 공동 상자

| 항목 | 값 |
|---|---:|
| 크기 XYZ | **2.6 × 2.6 × 2.6m** |
| 부피 | 17.576m³ |
| 학습 질량 | **60kg** |
| 질량/부피로 계산한 밀도 | 약 **3.413kg/m³** |
| 상자 shape 마찰 계수 | **0.6** |
| 상자 restitution | 0 |
| linear / angular damping | 0.01 / 0.01 |
| 크기·밀도 랜덤화 | 사용하지 않음 |

60kg은 총질량이다. 밀도는 크기에 맞춰 asset을 생성하기 위한 값이며, 바닥 마찰력 자체를 뜻하지 않는다.

### 바닥

| 항목 | 값 |
|---|---:|
| static friction | **1.0** |
| dynamic friction | **1.0** |
| restitution | 0 |

바닥의 정지 마찰을 동마찰보다 크게 설정하지 않았다. 상자 shape의 0.6과 바닥의 1.0을 별도로 설정하며, 이 문서에서는 최종 접촉 마찰 계수를 임의로 곱하거나 평균하여 단정하지 않는다.

### 관측 호환용 슬롯

Stage1 관측 구조를 유지하기 위해 object 슬롯은 총 4개다.

| object 슬롯 | 용도 | reset XY 위치 |
|---|---|---|
| 0 | 미사용 door | (5, -5) |
| 1 | 미사용 door | (-5, 5) |
| **2** | **두 agent가 공유하는 실제 상자** | **(0, 0)** |
| 3 | 미사용 box | (5, 5) |

공유 상자 중심 Z는 초기 1.3m다. goal 슬롯은 2개를 유지하되 두 agent의 목표 위치를 같게 설정하고, 관계 그래프에서는 같은 goal node를 참조한다.

미사용 슬롯은 관측과 물리 actor 구조에서 유지된다. 촬영용 뷰어에서는 해당 물체의 visual만 숨긴다. 충돌체까지 삭제하거나 관측에서 제거한 구성은 아니다.

## 4. 목표와 미는 면

| 항목 | 설정 |
|---|---|
| 상자 초기 yaw | 0–360° 균등 랜덤 |
| 목표 방향 | 초기 box local **+Y** |
| 목표 거리 | **2.2–5.0m 균등 랜덤** |
| 목표 높이 | 초기 상자 중심 높이와 동일 |
| 목표 재샘플링 | 환경 reset마다 독립 샘플링 |
| episode 도중 목표 | world 위치 고정 |
| 곡선 목표 각도 | 없음 |
| 상자 회전 자유도 | yaw 회전을 물리적으로 잠그지 않음 |
| 접촉면 | 초기 상자 **local −Y** 면 |
| 접촉면 유지 | `face_mode: reset_fixed` |

현재 초기 정렬 조건에서는 reset 때 목표 방향으로 선택한 뒷면이 local −Y 면이다. 선택한 면의 로컬 법선을 저장하고, episode 도중 목표의 상대 방향이 달라져도 다른 면을 다시 선택하지 않는다. 두 agent가 같은 면을 공유한다.

고정하는 것은 **상자의 물리적 면 ID**다. 상자가 이동·회전하면 손 보상 기준 면의 중심과 법선도 따라 움직인다. 손 목표를 초기 world 좌표에 고정하는 것은 아니다. 부분 reset에서는 해당 환경의 면만 갱신한다.

## 5. Agent 시작 위치·방향과 RSI

### 360° 시작 위치

`random_spawn.mode: surround_independent`를 사용한다.

- 각 agent의 상자 주변 방위각을 0–360°에서 독립 샘플링한다.
- 위치 반경은 상자 XY 외접원 반경에 1–2m를 더해 샘플링한다.
- 현재 외접원 반경은 `sqrt(2.6² + 2.6²)/2 ≈ 1.838m`다.
- 따라서 상자 중심–agent root의 초기 수평 거리는 약 **2.838–3.838m**다.
- 두 root의 수평 간격이 **0.8m 미만**이면 두 위치를 재샘플링한다. 최대 128회 시도한다.
- heading은 위치와 별개로 각 agent마다 0–360°에서 독립 샘플링한다.

앞면·옆면·뒷면 쪽 모두에서 시작할 수 있다. 상자를 등지고 시작하거나 두 사람이 같은 쪽에서 시작하는 것도 가능하다. agent별 좌/우 담당 자리를 미리 배정하지 않는다.

YAML의 `spawn_distance: [1, 2]`는 이 모드에서 **외접원 바깥에 더하는 반경**이다. 모든 면까지의 최단 거리가 정확히 1–2m라는 의미가 아니다. 두 사람 간격을 보장하기 위한 rejection 때문에 최종 위치 분포에는 간격 제약이 반영된다.

### RSI

| 항목 | 설정 |
|---|---|
| `jointPush.rsi` | `loco_only` |
| `stateInit` | `Random` |
| skill 순서 | loco / doorOpen / push |
| `skillInitProb` | **[1.0, 0.0, 0.0]** |
| motion 초기화 | loco motion에서 샘플링한 자세·속도 |
| 배치 | 샘플링한 위치·heading으로 root 및 body pose/velocity를 함께 변환 |
| Push interaction RSI | 이 프로필에서는 사용하지 않음 |
| AMP 초기 history | 실제 reset AMP state를 10프레임 반복 |

`hybridInitProb: 0.5`가 YAML에 남아 있지만 `stateInit: Random`이므로 Hybrid 초기화 확률로 동작하지 않는다. 별도 koo Push interaction RSI 프로필과 구분한다.

## 6. 관측·행동·관계 그래프

| 항목 | 값 |
|---|---|
| policy observation 모드 | `clean_scene` |
| 장면 observation 크기 | **684** |
| 행동 크기 | **agent당 32** |
| scene node | human 2 + object 4 + goal 2 = **8개** |
| 관절 제어 | PD target 제어 |
| `powerScale` | 1.0 |
| actor/critic | 별도 encoder |
| 장면 위치·회전 관계 | GTA, `se3_direct_sum` |
| 객체 bbox 관측 | 사용 |
| 물체 intrinsic 관측 | object-local 선속도·각속도·bbox corners |
| goal intrinsic 관측 | 상수 placeholder, 위치/회전은 GTA pose로 전달 |

사람과 물체 intrinsic feature를 token화하고, entity pose를 통한 기하 관계와 semantic 관계를 함께 처리한다. 현재 정책은 실제 2.6m 상자를 관측한다. 무학습 비교에서 사용한 1.3m 가상 상자 관측을 이 프로필에 적용한 것은 아니다.

### 공유 관계 그래프

node index는 human0/1, object0–3에 해당하는 2–5, goal0/1에 해당하는 6–7이다. 공유 상자 object2는 node4, 공유 goal0은 node6이다.

| owner | source | relation | target |
|---|---|---|---|
| agent0 | human0(node0) | PUSH **12** | shared box(node4) |
| agent0 | shared box(node4) | AT **7** | shared goal(node6) |
| agent1 | human1(node1) | PUSH **12** | shared box(node4) |
| agent1 | shared box(node4) | AT **7** | shared goal(node6) |

모든 edge는 유효하다. 두 agent의 AT edge가 최종 goal subgoal이다. semantic packet 필드는 `[valid, src, dst, relation, owner]`다. task/scene token 순서를 섞어 특정 배열 순서에 의존하는 것을 줄인다.

## 7. Stage1 전이와 Stage2 모델

### Stage1 초기값

- 원본 계열: koo `open_push_push180_contact_at_warmstart`
- source checkpoint epoch: **29000**
- 로컬 checkpoint: `runs/jointpush/stage1_sources/OpenPushPush180ContactAtWarmstart_00029000.pth`
- 원본 Stage1은 1.3m 상자 기반이며, 현재 환경은 2.6m 상자다.

| 모듈 | 초기화 | Stage2 학습 |
|---|---|---|
| Actor scene encoder | Stage1 가중치 | **동결** |
| Actor semantic edge embedding/MLP | Stage1 actor encoder에 포함 | **동결** |
| Observation RMS | Stage1 통계 | **동결** |
| Task grounding MLP | 새 초기값 | 학습 |
| CA4 | 새 초기값 | 학습 |
| Action head | Stage1 head 확장 | 학습 |
| Critic encoder/value head | Stage1 초기값 | 학습 |
| AMP discriminator | Stage1 초기값 | 학습 |
| AMP input RMS | Stage1 통계로 초기화 | 학습 시 갱신; one-hot label은 보존 |
| Optimizer/counter | 새 상태 | 이전 Stage2 상태를 가져오지 않음 |

따라서 “Stage2를 새로 학습”은 전체 정책을 무작위 초기화한다는 뜻이 아니다. 기존 Stage1 encoder뿐 아니라 action head·critic·discriminator 초기값도 사용한다.

### 네트워크 구성

| 항목 | 설정 |
|---|---|
| Stage1 encoder feature width | 64 |
| tokenizer MLP hidden | 256, 128 |
| Stage1 attention layers / heads | 4 / 2 |
| Stage1 FFN width | 512 |
| Stage1 semantic bias 방식 | `edge_mlp` |
| GTA | enabled, translation scale 1.0 |
| 별도 geometry branch | disabled |
| Stage2 grounding | **320 → 128 → 64**, ReLU |
| Stage2 CA layers / heads | **4 / 2** |
| Stage2 CA FFN width | **128** |
| CA dropout | 0 |
| Action head | **128 → 1024 → 512 → 32** |
| Value head hidden | 1024, 512 |
| Discriminator hidden | 1024, 512 |
| 활성화 | ReLU |
| policy sigma | fixed, 학습하지 않음; config 초기 log-sigma -2.9, Stage1 저장값 전이 |

### Task token과 협동 bias

각 agent의 token 입력은 다음 5개 64차원 feature를 이어 붙인 것이다.

```text
human feature + shared box feature + shared goal feature
             + PUSH semantic feature + AT semantic feature
                         ↓ grounding MLP
                    agent별 task token 64D
```

PUSH/AT semantic feature는 Stage1의 source type embedding, relation embedding, target type embedding을 edge MLP에 통과시킨 결과다. 해당 encoder는 동결한다.

CA4에서는 human feature가 Query이고 두 agent의 task token이 Key/Value다.

| Stage2 bias ID | 의미 |
|---|---|
| 0 | 기타 task |
| 1 | 자기 task |
| 2 | 같은 상자·목표를 공유하는 파트너 task |

각 층·head마다 3개의 학습 가능한 scalar bias를 갖고 초기값은 0이다. 같은 층·head·ID에서는 task 상태와 무관하게 같은 bias를 선택한다. task semantic embedding이 이 scalar를 매번 생성하는 구조는 아니다. embedding은 task token, 즉 Key/Value를 통해 attention에 영향을 준다.

```text
동결 Stage1 encoder → human feature 64D ──────────────────┐
                    → scene features → task tokens → CA4 → context 64D
                                                        ↓ concat
                                             action head 입력 128D
```

Action head 첫 층은 `[W_stage1, 0]`로 확장하고 기존 bias를 복사한다. 처음에는 새 context 경로의 가중치가 0이며 이후 학습한다.

## 8. Task reward

각 agent가 PUSH와 AT 두 관계의 State·Progress·Success 보상을 받는다. 아래 모든 수식은 control step마다 계산한다.

### 8.1 기호

- `m_i`: agent i의 양손 위치 중점.
- `c`: 고정된 미는 면의 중심 XY, Z는 상자 바닥 + 0.8m.
- `n`, `t`: 현재 상자 pose를 따라 움직이는 미는 방향 법선과 가로 접선.
- `l_i = dot((m_i-c)_XY, t)`: 손 중점의 면 중심 기준 가로 좌표.
- `d_i = dot((m_i-c)_XY, n)`: 미는 면으로부터의 깊이 오차.
- `h_i = (m_i-c)_Z`: 손 높이 오차.
- `b`, `g_goal`: 상자 중심과 목표 중심.
- `w`: 면 폭, 현재 2.6m.

### 8.2 PUSH / Contact State

```text
outside_i = max(|l_i| − w/2, 0)
surface_i = exp[−10 × (d_i² + outside_i² + h_i²)]
symmetry = exp[−10 × ||(m_0+m_1)/2 − c||²]
separation = |l_0 − l_1|
separation_score = clip(separation / 0.65, 0, 1)
φ_PUSH_i = surface_i × symmetry × separation_score
R_PUSH_state_i = 0.2 × φ_PUSH_i
```

손 목표를 고정 앵커에 두는 방식은 아니다. 두 agent 손 중점의 평균이 면 중심에 가깝고, 간격이 충분한 배치를 평가한다.

0.65m는 `면 폭 × min_separation_fraction = 2.6 × 0.25`에서 나온다. 간격이 작으면 별도 음수 penalty 대신 State 점수의 계수가 선형으로 줄어든다. 각 손을 개별 접촉 판정하는 대신 양손 중점을 쓰며, 실제 접촉력·미는 힘은 이 점수에 포함하지 않는다.

### 8.3 PUSH Progress / Success

```text
r_box = sqrt(size_x² + size_y²)/2
D_agent_i = ||root_i_XY − b_XY||
R_PUSH_progress_i = 0.2 / [1 + max(D_agent_i − r_box − 0.2, 0)/1.0]
R_PUSH_success_i = 0.2 × 1[φ_PUSH_i ≥ 0.9]
```

Progress는 몸통 root가 상자에 가까울수록 커진다. 현재 상자에서는 root 거리가 약 2.038m 이내일 때 포화한다. 포화했다고 실제로 상자를 밀었거나 손이 닿았다는 뜻은 아니다.

Success bonus는 조건을 만족하는 매 step에 지급한다. 한 번만 주는 종료 보상이 아니다.

### 8.4 AT soft gate

```text
opposite = 1[l_0 × l_1 < 0]
g_soft = min(φ_PUSH_0, φ_PUSH_1) × opposite
```

두 사람이 **같은 미는 면의 좌우**에 있는지 확인한다. 상자 앞면과 뒷면에 한 명씩 배치한다는 의미가 아니다. 같은 쪽이면 0, 좌우에 있으면 약한 쪽의 손 배치 품질을 사용한다.

연속값인 부분은 φ의 최솟값이며, `opposite`는 여전히 0/1 조건이다. 이 gate는 AT State·Progress에만 곱한다.

### 8.5 AT State / Progress

```text
D_3D = ||b − g_goal||
D_XY = ||b_XY − g_goal_XY||
φ_AT = exp(−10 × D_3D²)
R_AT_state = 0.2 × g_soft × φ_AT
R_AT_progress = 0.2 × g_soft / [1 + max(D_XY − 0.1, 0)/1.0]
```

State는 목표 근처에서 급격히 커지고, Progress는 멀리서 접근할 때도 점수를 준다. 둘 다 **현재 거리 기반**이다. `이전 step 거리 − 현재 거리`가 아니므로 가까운 상태를 유지해도 계속 지급된다. 실제 이동량·속도·힘에 대한 추가 보상은 없다.

### 8.6 AT Success / 최종 성공

```text
strict_contact = (φ_PUSH_0 ≥ 0.9) AND (φ_PUSH_1 ≥ 0.9)
                 AND opposite AND (separation ≥ 0.65)
AT_success = (φ_AT ≥ 0.9)
             AND (|b_Z − g_goal_Z| ≤ 0.001)
             AND strict_contact
R_AT_success = 0.2 × 1[AT_success]
```

`φ_AT ≥ 0.9`는 상자–목표의 **3D 거리 약 0.10265m 이내**에 해당한다. 별도로 높이 차이 **1mm 이하**, 두 사람의 엄격한 접촉 조건도 만족해야 한다. soft gate를 도입해도 최종 성공 조건은 완화하지 않는다.

### 8.7 Power / CPA / 최종 합성

```text
R_power_i = −0.0005 × Σ_j |torque_ij × joint_velocity_ij|
R_task_i = R_PUSH_state_i + R_PUSH_progress_i + R_PUSH_success_i
           + R_AT_state + R_AT_progress + R_AT_success
           + R_power_i + R_CPA_i
R_train_i = 0.5 × R_task_i + 0.5 × R_AMP_i
```

CPA는 두 사람의 수평 상대 위치·속도로부터 향후 최접근 시 충돌 위험을 계산한다. 계수 **0.5**, 거리 기준 **0.7m**, TTC discount **0.99**를 사용한다. Power penalty는 torque²가 아닌 위의 절대 power 합이다.

별도로 상대의 task reward를 더하는 비율은 `self=1, teammate=0`이다. 다만 PUSH의 대칭성 점수와 AT gate 자체에 공동 상태가 포함돼 있으므로 agent 간 의존성은 보상 안에 존재한다.

## 9. AMP와 motion data

| 항목 | 설정 |
|---|---|
| motion YAML | `tokenhsi/data/dataset_open_push_push180.yaml` |
| YAML 수록 motion 수 | loco 4, doorOpen 1, push 21 |
| 현재 task family | PUSH만 사용 |
| PUSH expert 혼합 | **loco 0.33 / doorOpen 0 / push 0.67** |
| 초기화 RSI 혼합 | **loco 1.0 / doorOpen 0 / push 0** |
| discriminator | 공유 1개, task-conditioned |
| family label | OPEN/PUSH 2차원 one-hot |
| motion feature | 129차원/frame |
| discriminator 입력 | (129+2) × 10 frames = **1310차원** |
| AMP 정규화 | 사용, one-hot label 값은 보존 |

**RSI와 AMP expert는 별개**다. loco-only RSI이지만 AMP reference에는 Push motion도 사용한다. motion YAML에 doorOpen이 포함되어 있어도 현재 PUSH expert에서 가중치가 0이므로 선택되지 않는다.

agent의 AMP 관측과 expert 관측 모두 frame마다 task family label을 붙인다. label은 discriminator의 별도 인자가 아니라 `amp_obs` 끝에 포함된다. 현재 두 agent는 모두 PUSH label이다.

```text
R_AMP = 2 × [−log(max(1 − sigmoid(D(amp_obs)), 0.0001))]
```

앞의 2는 `disc_reward_scale`이다. 이후 최종 학습 보상에 합칠 때 `disc_reward_w=0.5`를 곱한다.

## 10. PPO·optimizer·discriminator 학습 설정

### PPO

| 항목 | 값 |
|---|---:|
| learning rate | 2e-5 |
| LR schedule | constant |
| gamma / GAE tau | 0.99 / 0.95 |
| horizon length | 32 |
| rollout당 scene steps | 2048 × 32 = 65536 |
| rollout당 agent steps | 2048 × 2 × 32 = 131072 |
| minibatch size | 16384 |
| mini epochs | 6 |
| PPO clip `e_clip` | 0.2 |
| critic coefficient | 5 |
| entropy coefficient | 0 |
| bounds loss coefficient | 10 |
| normalize input / value / advantage | true / true / true |
| reward shaper scale | 1 |
| clip value | false |
| truncate gradients | false |
| grad norm 설정값 | 1.0, 다만 truncate_grads=false |
| seq_len 설정값 | 4 |
| epsilon-greedy | false |
| max_epochs | 1000000 |

### Discriminator

| 항목 | 값 |
|---|---:|
| AMP demo buffer | 200000 |
| AMP replay buffer | 200000 |
| replay keep probability | 0.01 |
| AMP batch size | 512 |
| AMP minibatch size | 4096 |
| discriminator coefficient | 5 |
| logit regularization | 0.01 |
| gradient penalty | 5 |
| reward scale | 2 |
| weight decay | 0.0001 |

### 저장

- 정기 checkpoint는 **500 iter마다** 저장한다.
- `save_intermediate: true`, `save_best_after: 50`이다.
- run에는 `relation_config.yaml`, `stage2_transfer_report.json`, `random_seed.txt`, `summaries/`, `nn/`을 저장한다.
- 같은 profile의 resume는 checkpoint에 저장된 환경·optimizer·counter를 복원한다.
- 평가·뷰어도 저장 환경을 복원하므로, 최신 YAML만 수정한다고 과거 모델의 물리 조건이 바뀌지 않는다.

## 11. Episode 종료와 성공 처리

- 발 이외 body의 contact와 body 높이 저하를 조합해 넘어짐을 검출한다.
- `terminationHeight=0.15m`, 허용 contact body는 right_foot/left_foot이다. 초기 몇 step은 넘어짐 판정을 억제한다.
- 어느 한 agent가 넘어지면 공유 환경 전체를 reset한다. 넘어진 agent 자신의 terminate flag와 상대의 truncation은 구분한다.
- 600step timeout에서 reset한다.
- `terminate_when_all_subgoals_done: false`이므로 성공만으로 즉시 episode를 끝내지는 않는다.

성공률을 정의할 때는 episode 중 한 번이라도 AT_success를 만족했는지와 마지막 순간에도 만족하는지를 구분한다. 이 문서에는 해당 수치를 넣지 않는다.

## 12. 학습·평가·뷰어 실행

작업 디렉터리: `/home/hwanhee/juan/CVPR2027`.

```bash
# 신규 Stage2: Stage1 초기값 전이, 60kg, 2048환경, GPU6
bash stage2_jointpush/scripts/joint_push_softgate_train.sh

# 동일 softgate profile 학습 재개
RESUME_CHECKPOINT=/absolute/path/checkpoint.pth \
  bash stage2_jointpush/scripts/joint_push_softgate_train.sh

# 저장 설정 그대로 headless 평가
bash stage2_jointpush/scripts/joint_push_softgate_test.sh /absolute/path/checkpoint.pth

# 저장 설정 그대로 VNC 뷰어, 일반적인60kg모델용
PORT=6086 bash stage2_jointpush/scripts/joint_push_softgate_view.sh /absolute/path/checkpoint.pth

# 동일60kg학습모델을40kg에서진단하는뷰어
PORT=6086 bash stage2_jointpush/scripts/joint_push_softgate_mass40_view.sh /absolute/path/checkpoint.pth
```

신규 학습 기본 output은 `runs/jointpush/aligned_surround60_softgate_training/`이다. `OUTPUT_PATH`, `SEED`, `STAGE1_CHECKPOINT`, 양수 `MAX_ITERATIONS`를 지정할 수 있다. `MAX_ITERATIONS=0`은 상한 변경으로 처리되지 않으므로 wrapper에서 거부한다.

평가는 `NUM_ENVS`, `REPEATS`, `EPISODE_LENGTH`, `SEED`, `OUTPUT_PATH`를 지정할 수 있다. VNC 뷰어 기본은 1환경이며 `PORT`를 지정한다.

### 40kg 진단 구분

- 학습의 60kg을 변경하는 기능이 아니다.
- checkpoint의 network/reward/reference 계약을 정상적으로 검증한다.
- GPU simulation 준비 전에 공유 box2를 40kg에 해당하는 밀도로 생성한다. 관성도 같은 형상과 질량에 맞춘다.
- 미사용 box3은 원래 60kg을 유지한다.
- 모델·마찰·목표·초기화 분포·보상은 동일하게 유지한다.
- `mass_preview.json`에 원래 checkpoint 질량, 실제 생성 질량, 적용 단계를 저장한다.
- 실행 중 rigid-body property setter로 질량을 변경하는 이전 방식은 사용하지 않는다.

### 촬영용 뷰어

처음에만 옆쪽의 자유 카메라로 배치한다. agent focus는 해제하고 episode reset에서 카메라를 되돌리지 않는다. 사용자가 조절한 카메라 위치를 유지한다. F는 추적 전환, L은 PNG frame 저장 시작/정지다. 노란 목표 영역 overlay는 표시하지 않고, 초기·현재 방향 화살표는 유지한다.

## 13. 기존 profile과 혼동하기 쉬운 항목

| profile | 질량 | agent 시작 위치 | 목표 거리 | AT gate |
|---|---:|---|---|---|
| aligned_loco_rsi | 40kg | 뒷면 좌우 lane 고정, 앞뒤 거리 랜덤 | 2.2–2.8m | hard |
| aligned_random_spawn60 | 60kg | 뒷면 쪽만, 좌우·heading 랜덤 | 2.2–2.8m | hard |
| aligned_surround60 | 60kg | 주변360°·heading 랜덤 | 2.2–5m | hard |
| **aligned_surround60_softgate** | **60kg** | **주변360°·heading 랜덤** | **2.2–5m** | **soft_min** |

이전 checkpoint의 face_mode 유무 등은 저장 설정이 기준이다. 현재 파일 값을 과거 run에도 적용된 것으로 해석하지 않는다.

YAML에 남아 있는 단독 Push용 `relationReward.push`의 hand force, box force, speed, tilt 등이나 door 설정, `eval.successThreshold=0.2`는 joint_push_reward의 성공식을 덮어쓰지 않는다. **현재 JointPush 성공 판정에는 접촉력 기준, 속도 기준, 최종 yaw 정렬 조건이 포함되지 않는다.**

## 14. 구현 참조

- Task/reset/고정면: [humanoid_ma_joint_push.py](../ports/koo_push/tokenhsi/env/tasks/multi_agent/humanoid_ma_joint_push.py)
- 시작 위치: [joint_push_spawn.py](../ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_spawn.py)
- Reward: [joint_push_reward.py](../ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_reward.py)
- Stage2 token/CA4: [push_at_coordination.py](../ports/koo_push/tokenhsi/learning/multi_agent/push_at_coordination.py)
- Stage1 전이: [joint_push_transfer.py](../ports/koo_push/tokenhsi/learning/multi_agent/joint_push_transfer.py)
- 조건부 AMP: [humanoid_ma_open_push.py](../ports/koo_push/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py)
- 질량 진단: [joint_push_mass_preview.py](../ports/koo_push/tokenhsi/utils/joint_push_mass_preview.py)
