# TokenHSI 멀티 에이전트 확장 — 구현 문서

TokenHSI의 carry 태스크를 **M명의 휴머노이드, O개(O≥M)의 박스, M개의 목표** 구조로 확장한
구현입니다. 각 에이전트에는 서로 다른 박스 하나를 무작위 배정하고, 남은 O-M개는 미할당
distractor로 둡니다. 엔티티(사람 / 객체 / 목표)를 타입별 토크나이저로 분리하고, 할당 관계를
비학습 relation matrix로 표현해 학습 가능한 attention bias로 주입합니다.

기존 체크포인트(`output/`)와 데이터셋(`tokenhsi/data/dataset_*`)은 **일절 건드리지 않았습니다.**
모션 데이터는 읽기 전용으로 재사용하고, 학습 산출물은 별도 경로에 씁니다.

---

## 0. 쉽게 읽는 요약

### 무엇을 만들었나

원래 TokenHSI의 carry는 **한 장면에 사람 1명, 박스 1개**였습니다. 이제 **사람 M명, 박스 O개,
목표 M개**를 지원합니다. `O≥M`이며 사람마다 서로 다른 담당 박스와 목표가 하나씩 있고, 남는
박스는 담당자가 없습니다. 물리 박스와 담당자는 에피소드 리셋마다 무작위로 다시 연결됩니다.
렌더링에서는 담당 박스를 에이전트와 같은 색으로 표시하고, 미할당 박스는 회색으로 표시합니다.

### 왜 "토큰"으로 쪼갰나

기존 TokenHSI는 관측을 `[내 몸 상태] + [태스크 정보]` 두 덩어리로 넣었고, 태스크 정보 안에
박스 상태와 목표가 뭉쳐 있었습니다. 이걸 **세 종류로 분리**했습니다.

```
사람(humanoid) 토큰   ← 내 몸이 어떤 자세로 어디에 있나
객체(object) 토큰      ← 박스가 어디에 어떻게 놓여 있나
목표(goal) 토큰        ← 박스를 어디로 옮겨야 하나
```

이렇게 쪼개면 **사람이나 박스가 늘어날 때 토큰만 늘리면 됩니다.** 토큰 수는 `2M+O`이며,
**네트워크 구조와 파라미터는 전혀 바뀌지 않습니다.**

### 그럼 "누구 박스가 누구 것"인지 어떻게 아나

토큰이 6개 있으면 네트워크 입장에서는 그냥 벡터 6개일 뿐입니다. 그래서 **관계 표(relation matrix)**
를 같이 줍니다. "이 사람과 이 박스는 주인-소유물 관계", "이 박스와 이 목표는 짝" 같은 걸
숫자로 적어둔 6×6 표입니다.

이 표는 **학습하지 않습니다.** 환경이 이미 알고 있으니까요. 대신 각 관계 종류마다 **작은 실수
하나씩**을 학습해서 attention 점수에 더해줍니다. "주인-소유물 관계면 서로 좀 더 주목해라" 같은
걸 네트워크가 스스로 배우는 겁니다. 학습되는 건 **총 48개 숫자**뿐입니다.

### agent A / agent B를 왜 구분하지 않나

구분하지 **않는 게** 핵심입니다. 사람 토큰은 전부 똑같은 종류로 취급하고, "지금 누구의 행동을
뽑는가"는 **몇 번째 토큰을 읽느냐**로 정합니다. 항상 0번 토큰을 읽고, 그 0번 자리에 지금 행동할
사람을 놓습니다.

덕분에 네트워크는 "1번 에이전트", "2번 에이전트" 같은 걸 아예 모르고, **사람이 몇 명이든
똑같이 동작**합니다. 이게 나중에 M=2로 학습해서 M=4로 그냥 돌려보는 게 가능한 이유입니다.

### 좌표계가 왜 그렇게 중요한가 (`obsFrame`)

사람 형태의 로봇을 제어할 때, 정책이 내놓는 건 **관절 각도**입니다. 관절 각도는 "내 몸 기준"이라
내가 북쪽을 보든 동쪽을 보든 **똑같습니다**.

그래서 관측도 "내 몸 기준"으로 주면, 북쪽으로 걸어가는 상황과 동쪽으로 걸어가는 상황이
**같은 입력**이 되어 정책이 한 번만 배우면 됩니다. 이걸 **heading 불변성**이라고 하고,
AMP 계열 휴머노이드 제어에서 가장 중요한 공짜 이득입니다.

세 가지 선택지가 있습니다.

| 모드 | 비유 | heading 불변 | 한 번에 M명 |
|---|---|---|---|
| **`owner`** (기본값) | 각자 자기 눈으로 본 자기 물건 | ✅ | ✅ |
| `global` | CCTV 하나로 경기장 전체를 봄 | ❌ | ✅ |
| `ego` | 매번 "A의 눈"으로 전부 다시 봄 | ✅ | ❌ |

- **`global`** 은 좌표계가 하나라 깔끔하고 누가 어디 있는지 정확히 압니다. 대신 "북쪽 걷기"와
  "동쪽 걷기"가 다른 입력이 되어 네트워크가 회전 보정을 **직접 배워야** 합니다.
- **`ego`** 는 매 사람마다 장면 전체를 그 사람 시점으로 다시 계산합니다. 계산량이 M배 늘고,
  "B의 박스를 A 시점에서 본 것"은 1인 학습 때 본 적 없는 낯선 입력이 됩니다.
- **`owner`** 는 각 물건을 **그 주인의 시점**으로 기술합니다. 그러면 이 값은 **누가 보든 똑같아서**
  한 번만 계산하면 되고, 동시에 각자에게는 "내 몸 기준"이라 heading 불변성도 유지됩니다.
  1인 학습 때와 완전히 같은 입력이기도 합니다.

  단점 하나: 각자 자기 시점만 갖고 있으면 **A가 B의 위치를 모릅니다.** 그래서 사람 토큰에
  "경기장 어디쯤에 어느 방향으로 서 있는지" 4개 숫자를 덧붙여, 네트워크가 뺄셈으로 상대 위치를
  알아낼 수 있게 했습니다.

### "한 번의 연산으로 모든 액션"은 되는가

**됩니다.** `owner`와 `global` 둘 다 토큰 값이 관측자와 무관하므로, 장면을 한 번만 인코딩하고
i번째 토큰 출력을 읽으면 i번 사람의 행동이 나옵니다(수학적으로 정확히 성립).

다만 **코드에서는 아직 사람 수만큼 forward가 돕니다.** 이유는 좌표계가 아니라 **PPO 때문**입니다.
PPO는 사람별로 value·advantage를 따로 계산해야 해서 `(환경, 사람)`마다 한 줄씩 데이터를 만들고,
학습할 때 그 줄들을 섞습니다. 섞인 줄 하나만 봐도 계산이 되어야 하므로 줄마다 인코더가 돕니다.
실측상 이 부분이 전체 시간의 1/3 정도이고 나머지는 물리 시뮬이라 손해가 크지 않아 지금은
그대로 뒀습니다.

### 그래서 결과는?

- **1인 관측이 원본 TokenHSI와 수치적으로 완전히 동일**함을 확인했습니다 (오차 0.0).
- **파라미터 수가 사람 수와 무관**합니다 (M=1,2,3,4 모두 4,100,770개).
- M=2로 학습한 가중치가 M=3 네트워크에 **그대로 로드**됩니다.
- 1인 학습에서 사람이 넘어지지 않고 버티는 시간이 **30 → 71 스텝**으로 늘었습니다
  (원본 TokenHSI 레퍼런스보다 빠른 속도).

---

## 1. 한눈에 보기

```
                     obs row 1개 = (env, agent) 1쌍,  ego-first 정렬
   ┌──────────────────────────────────────────────────────────────────┐
   │ [ h₀ h₁ … h_{M-1} │ o₀ o₁ … o_{O-1} │ g₀ g₁ … g_{M-1} ]          │
   │      230 each           39 each           6 each                 │
   └────────┬────────────────────┬──────────────────┬─────────────────┘
            │ T_h                │ T_o              │ T_g        (타입별 토크나이저)
            │ +E_humanoid        │ +E_object        │ +E_goal    (타입 임베딩)
            └────────────────────┴──────────────────┘
                                 │  (2M+O)개 토큰, 64-d
                    ┌────────────┴────────────┐
                    │  Relation Transformer   │  ← 고정 relation matrix R을
                    │  (4 layers, 2 heads)    │     학습 가능한 head별 attention bias로
                    └────────────┬────────────┘
                                 │  token 0 = ego
                    ┌────────────┴────────────┐
              Actor encoder              Critic encoder   (가중치 분리)
                    │                          │
              Action Head                 Value Head
                    │                          │
                   μ, σ  →  a^i              V^i
```

**핵심 성질**: self/other 구분도, agent-ID 임베딩도 없습니다. "누가 무엇을 소유하는가"는
relation matrix가 전달하고, ego는 **readout 위치**가 결정합니다. 그 결과 네트워크는 완전한
permutation-equivariant이며 **모든 파라미터가 M과 O에 무관**합니다.

| M | O | obs / row | 토큰 수 | 파라미터 | state_dict 키 |
|---|---|---|---|---|---|
| 1 | 1 | 275 | 3 | 4,100,770 | 155 |
| 2 | 2 | 550 | 6 | 4,100,770 | 155 |
| 2 | 3 | 589 | 7 | 4,100,770 | 155 |
| 3 | 3 | 825 | 9 | 4,100,770 | 155 |
| 4 | 4 | 1100 | 12 | 4,100,770 | 155 |

→ `M=2,O=2` 체크포인트가 `M=2,O=3` 또는 `M=3,O=3` 네트워크에
`load_state_dict(strict=True)`로 그대로 올라갑니다.

---

## 2. 파일 목록

### 신규

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `tokenhsi/env/tasks/multi_agent/humanoid_ma.py` | 666 | env당 M체 휴머노이드를 지원하는 `Humanoid` 서브클래스. 텐서 뷰, 리셋, PD 타깃, 관측 프레임, 동영상 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | — | MA carry 태스크. O개 박스, 무작위 assignment, 에이전트별 보상 / AMP obs / 목표 |
| `tokenhsi/env/tasks/multi_agent/vec_task_wrapper_ma.py` | 46 | rl_games에 `(env, agent)`별 row를 노출 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | 267 | relation-transformer actor + critic |
| `tokenhsi/learning/multi_agent/ma_agent.py` | 108 | PPO/AMP 에이전트. entity-wise 정규화, 동영상 훅 |
| `tokenhsi/learning/multi_agent/ma_players.py` | 50 | 테스트 플레이어 |
| `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml` | — | env 설정 |
| `tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml` | — | 학습 설정 |
| `tokenhsi/scripts/multi_agent/ma_carry_{train,test}.sh` | — | 실행 스크립트 |

### 기존 파일 수정 — 총 26줄, 전부 additive

| 파일 | 변경 |
|---|---|
| `tokenhsi/run.py` | `ma` algo / player / `amp_multi_agent` network 등록, `env_info['agents']` 노출 |
| `tokenhsi/utils/parse_task.py` | MA 태스크 import, `num_agents` 속성이 있으면 MA 래퍼 사용 |
| `tokenhsi/utils/config.py` | `--num_agents` CLI 인자 |

기존 태스크 경로는 그대로입니다. 유일한 동작 변화는 `env_info['agents']`가 명시적으로 1이 되는
것인데, 이는 rl_games의 기본값과 동일합니다.

---

## 3. 실행

```bash
# 각 스크립트가 tokenhsi_jhh를 자동 활성화합니다.
# repo 밖에서 실행해도 현재 checkout을 기준으로 동작합니다.

# 학습 (M=1, 1024 envs — 16GB 기준 권장)
sh tokenhsi/scripts/multi_agent/ma_carry_train.sh 1 1024

# 학습 (M=2, 512 envs — 휴머노이드 총 개수를 맞춤)
sh tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 512

# headless 테스트 (2 agents / 16 env / 기본 3회 반복)
sh tokenhsi/scripts/multi_agent/ma_carry_test.sh output/ma_carry/<run>/nn/HumanoidMA.pth 2 16 0 3

# 2 agents / 3 boxes / 1 env / 20회 반복 (기존 2-agent checkpoint 재사용)
sh tokenhsi/scripts/multi_agent/ma_carry_test.sh output/ma_carry/<run>/nn/HumanoidMA.pth 2 1 3 20

# 2 agents / 3 boxes / 4 env 시각화 학습 resume
PORT=6080 sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/multi_agent/ma_carry_watch.sh \
  2 4 3 output/ma_carry/HumanoidMA_02-17-14-51/nn/HumanoidMA.pth

# 임시 noVNC viewer에서 20회 보기 (GPU 0 고정)
PORT=6080 sh tokenhsi/scripts/multi_agent/run-gui.sh \
  sh tokenhsi/scripts/multi_agent/ma_carry_test.sh \
  output/ma_carry/<run>/nn/HumanoidMA.pth 2 1 3 1024
```

`ma_carry_test.sh`의 위치 인자는 차례대로
`checkpoint`, `num_agents`, `num_envs`, `num_objects`, `num_repeats`,
`eval_skills`, `eval_skill_probs`입니다.
`num_repeats`를 생략하면 기본값은 3이며, 총 평가 에피소드 수는
`num_envs * num_repeats`입니다.
평가 skill과 확률의 기본값은 각각
`loco,pickUp,carryWith,putDown`과 `0.5,0.1,0.3,0.1`입니다. 원하는 분포는 다음처럼 지정합니다.

```bash
sh tokenhsi/scripts/multi_agent/ma_carry_test.sh \
  output/ma_carry/<run>/nn/HumanoidMA.pth 2 1 3 20 \
  "pickUp,carryWith,putDown" "0.2,0.6,0.2"
```

VS Code의 **PORTS** 탭에서 `6080`을 포워딩한 다음,
`Simple Browser: Show`로 아래 주소를 엽니다.

```text
http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=remote
```

`run-gui.sh`는 Xvfb·x11vnc·websockify를 임시로 실행합니다. 실행 명령을
`Ctrl+C`로 종료하면 GUI 인프라도 같이 정리됩니다.

추론에서도 `box.reset.randomHeight: True`와 `randomHeightProb`가 적용됩니다. 높이가 선택된
source/target 플랫폼은 원본 TokenHSI와 같은 계산식으로 담당 박스/목표의 XY와 바로 아래 Z에
놓입니다. 원본은 높이가 선택되지 않은 플랫폼을 지면 안쪽에 두지만, 이 PC의 GPU PhysX에서는
동적 플랫폼이 바닥과 관입한 뒤 위로 밀려납니다. 따라서 이 경우에는 플랫폼을 비활성 위치에 두고
동일한 역할을 하는 바닥 plane이 박스를 지지합니다. reference motion에서 필요하지 않은 플랫폼과
미할당 distractor 박스에도 플랫폼을 붙이지 않습니다. 이 호환 구현은 학습과 추론에 공통 적용되며
별도 실행 옵션은 필요하지 않습니다.
플랫폼끼리는 전용 collision filter로 충돌하지 않아 여러 플랫폼이 겹쳐도 물리가 발산하지 않습니다.
원본의 fixed platform처럼 회전하지 않도록 quaternion은 항상 수평으로 초기화되고 각속도는 0으로 제한됩니다.

산출물은 `output/ma_carry/HumanoidMA_<시각>/` 아래에 `nn/`(체크포인트),
`summaries/`(TensorBoard)로 저장됩니다. 학습 중간 동영상은 생성하지 않습니다.

> `minibatch_size`는 `horizon_length(32) × num_envs × M`을 나눠떨어지게 해야 합니다.
> 기본값 8192는 `1024×1×32 = 32768`, `512×2×32 = 32768` 모두 만족합니다.
> 작은 규모로 스모크 테스트할 때는 `--minibatch_size`를 함께 낮추세요
> (단 `amp_minibatch_size(4096) ≤ minibatch_size` 제약이 있습니다).

---

## 4. 관측 프레임 모드 — `obsFrame`

**이 문서에서 가장 중요한 부분입니다.** 세 모드의 차이는 "각 엔티티를 어느 좌표계로 기술하는가"
하나이며, 그 선택이 (a) 연산량, (b) heading 불변성, (c) M=1→M 전이 난이도를 전부 좌우합니다.

핵심 판별 기준은 **"토큰의 내용이 관측자에 의존하는가"** 입니다.

| | 토큰 내용이 관측자에 의존 | heading 불변 | 1회 인코딩으로 M개 액션 | 엔티티 변환 횟수 | M=1 분포와 일치 | 에이전트 간 기하 |
|---|---|---|---|---|---|---|
| **`owner`** (기본값) | 아니오 | **예** | **예** | M | **예** | arena 4-d 경유 |
| `global` | 아니오 | 아니오 | **예** | M | 아니오 | **직접·정확** |
| `ego` | **예** | 예 | 아니오 | **M²** | 아니오 | **직접·정확** |

### `owner` — 소유자 프레임 (기본값)

각 엔티티를 **그것을 소유한 에이전트의 heading frame**으로 기술합니다.

```
s^A = A의 몸을 A 자신의 프레임으로     o^A = A의 박스를 A의 프레임으로     g^A = A의 목표를 A의 프레임으로
s^B = B의 몸을 B 자신의 프레임으로     o^B = B의 박스를 B의 프레임으로     g^B = B의 목표를 B의 프레임으로
```

- 이 값들은 **누가 보든 동일**합니다. row가 달라지는 건 순서(ego-first 순열)뿐입니다.
- relation matrix R은 "인덱스가 같은가"만 보고 만들어지므로 **에이전트 인덱스 순열에 불변**입니다.
  따라서 트랜스포머의 equivariance에 의해
  **`canonical 순서 1회 forward의 position i` = `ego-i-first 순서 forward의 position 0`** 이 성립합니다.
  즉 한 번의 씬 인코딩으로 M개 액션이 정확히 나옵니다.
- `s^B`는 B 자신의 프레임이므로 공용 action head가 그대로 `a^B`로 디코딩합니다.
- **검증됨**: M=1일 때 이 표현의 휴머노이드 토큰은 원본 TokenHSI 관측(223-d)과
  **수치적으로 완전히 동일**합니다 (root 자리 3차원이 정확히 0으로 삽입, max diff = 0.0).

**대가**: A의 토큰이 전부 A 프레임, B의 토큰이 전부 B 프레임이면 A는 B가 어디 있는지 알 수 없습니다.
이를 보완하려고 humanoid 토큰에 **arena-local 4차원**(env 원점 기준 root xy / envSpacing, heading cos·sin)을
붙였습니다. 트랜스포머가 뺄셈으로 상대 위치를 복원할 수 있습니다. → 226 + 4 = **230**

### `global` — 전지적 단일 좌표계

모든 엔티티를 **env 중앙 기준, 전역 방향**으로 기술합니다 (위치는 `envSpacing`으로 정규화).
소유자/자기자신 개념이 어디에도 없는 완전 중앙집중 표현입니다.

- `owner`와 **똑같은** one-pass 성질을 갖습니다 (토큰 내용이 관측자 무관).
- 에이전트 간 기하가 토큰 안에 직접 들어 있습니다.
- **대가: SE(2)(heading) 불변성을 잃습니다.** 정책이 내놓는 건 자기 몸 프레임의 PD 관절 타깃인데,
  "북쪽을 보고 박스로 걸어가기"와 "동쪽을 보고 걸어가기"는 같은 제어 문제이고 정답 액션도 같습니다.
  글로벌 좌표계에서는 두 관측이 달라지므로 네트워크가 "회전만큼 출력을 돌려라"를 **직접 학습**해야
  합니다. AMP 휴머노이드는 학습 자체가 병목이라 체감될 가능성이 큽니다.
- env 중앙 정규화는 **translation** nuisance만 제거하고 **rotation**은 남습니다.

구현상으로는 `compute_humanoid_observations_rel_max` / `compute_object_goal_observations`에
관측 프레임으로 **env 중앙 + 항등 사원수**를 넘기면 그대로 됩니다. 코드 재사용 100%.

### `ego` — 관측자 프레임

모든 엔티티를 **관측하는 에이전트의 프레임**으로 재표현합니다.

- 에이전트 간 기하가 직접적이고 정확합니다.
- 하지만 토큰 내용이 관측자에 의존하므로 **one-pass 성질이 깨지고**, 엔티티 변환이 **M²번** 필요합니다.
- 그리고 "B의 박스를 A 시점에서 본 것"은 M=1 토크나이저가 한 번도 못 본 분포입니다.

**결론: `owner`가 `global`과 `ego`를 실용적으로 지배합니다.** `global`의 one-pass 성질을 그대로
가지면서 heading 불변성과 M=1 분포 일치를 유지합니다. 세 모드 모두 구현되어 있으니 **YAML 한 줄로**
A/B 하실 수 있습니다.

---

## 5. 관측 레이아웃

한 row = `(env, agent)` 한 쌍. **env-major / agent-minor** 순서 (`row = env * M + agent`).
이 순서는 rl_games가 `all_done_indices[::num_agents]`를 하기 때문에 강제됩니다.

```
[ humanoid 블록 (M × 230) │ object 블록 (O × 39) │ goal 블록 (M × 6) ]
```

**엔티티 블록 단위**로 배치한 이유: 네트워크가 타입별로 한 번의 view로 잘라낼 수 있고,
정규화도 타입별로 묶을 수 있기 때문입니다.

| 토큰 | 차원 | 내용 |
|---|---|---|
| humanoid | **230** | root height(1, carry에서는 0) + body pos(15×3) + body rot tan-norm(15×6) + lin vel(15×3) + ang vel(15×3) + **arena pose(4)** |
| object | **39** | lin vel(3) + ang vel(3) + pos(3) + rot tan-norm(6) + bbox 8점(24) |
| goal | **6** | 관측 프레임 기준 목표 위치(3) + **박스→목표 벡터(3)** |

원본 TokenHSI carry와의 차이:
- 원본은 `local_body_pos`에서 root 3차원(항상 0)을 버립니다. 여기서는 **유지**합니다 —
  그래야 ego든 아니든 모든 humanoid 토큰이 같은 레이아웃을 갖고 토크나이저를 공유할 수 있습니다.
- 원본 goal은 `local_tar_pos`(3)만 줍니다. 여기에 **박스→목표 상대 벡터(3)** 를 추가했습니다
  (설계안 §4의 "object에서 target까지의 상대 target"). 이게 실제로 행동을 유발하는 양입니다.

---

## 6. Relation matrix

토큰 순서 `[h₀..h_{M-1}, o₀..o_{O-1}, g₀..g_{M-1}]`에 대해 비학습
`(2M+O, 2M+O)` 정수 행렬을 만듭니다. 앞의 M개 object 슬롯에는 에이전트 순서대로 담당 박스를
재배열하고, 뒤의 O-M개 미할당 박스는 다른 엔티티와 `none` 관계를 갖습니다. 따라서 물리 박스 ID가
매 reset마다 바뀌어도 relation matrix 자체는 배치 공통으로 유지됩니다.

| 값 | 의미 |
|---|---|
| 0 | `none` — 아무 관계 없음 |
| 1 | `self` — 자기 자신 (대각) |
| 2 | `teammate` — 휴머노이드 ↔ 휴머노이드 |
| 3 | `own_object` — 휴머노이드 ↔ 자기 박스 |
| 4 | `own_goal` — 휴머노이드 ↔ 자기 목표 |
| 5 | `object_goal` — 박스 ↔ 자기 목표 |

M=2 실제 출력:

```
        h0   h1   o0   o1   g0   g1
  h0:    1    2    3    0    4    0
  h1:    2    1    0    3    0    4
  o0:    3    0    1    0    5    0
  o1:    0    3    0    1    0    5
  g0:    4    0    5    0    1    0
  g1:    0    4    0    5    0    1
```

학습되는 것은 `rel_embed: (num_layers, num_heads, 6)` — 총 **48개 스칼라**(인코더당)뿐이며,
attention logit에 head별로 더해집니다:

```
S_ij = q_i·k_j / √d_h + b_h(R_ij)
```

**0으로 초기화**했습니다. 그래서 학습 중 한 번도 등장하지 않은 relation type은 "bias 없음 =
평범한 attention"으로 동작합니다. M=1에서는 타입 {1,3,4,5}만 등장하므로, **M=1 → M≥2로 갈 때
새로 학습되는 파라미터는 타입 {0, 2}에 해당하는 `4층 × 2헤드 × 2타입 = 16개 스칼라`**
(actor+critic 합쳐 32개)뿐입니다.

`rel_matrix`는 `persistent=False` 버퍼입니다. 체크포인트에 들어가지 않으므로 M/O가 다른 네트워크에
가중치를 그대로 로드할 수 있습니다. `network.set_entity_counts(M', O')`로 실행 중 재타겟도 가능합니다.

---

## 7. 학습 스택이 거의 그대로 동작하는 이유

TokenHSI의 rl_games 스택은 이미 멀티 에이전트를 전제로 배선되어 있습니다
(`common_agent.py:323` `done_indices = all_done_indices[::self.num_agents]`,
`:519` `num_seqs = num_actors * num_agents`).

`env_info['agents'] = M`으로 알려주고 env가 `(num_envs*M, ...)`를 내보내면 다음이 **수정 없이** 동작합니다:

- 에이전트별 value / advantage / GAE / log-prob
- 에이전트별 task reward
- **에이전트별 AMP style reward** — `infos['amp_obs']`가 `(num_envs*M, 1290)`이 되면
  discriminator가 그대로 휴머노이드별 style reward를 냅니다.
- **centralized per-agent critic** — row가 이미 씬 전체를 담고 있으므로
  `V^i = V(s^{1..M}, o^{1..M}, g^{1..M}; ego=i)`가 자동으로 성립합니다.

직접 손봐야 했던 것은 네 군데뿐입니다:

1. **`env_info['agents']`** — rl_games는 `get_number_of_agents()`가 아니라
   `env_info['agents']`에서 에이전트 수를 읽습니다. TokenHSI의 `RLGPUEnv.get_env_info()`가
   이 키를 넣지 않아 `num_agents`가 항상 1이었습니다.
2. **row → env 인덱스 매핑** — rl_games는 `all_done_indices[::num_agents]`를 리셋에 넘기는데,
   이 값은 **row 인덱스**(`env*M`)입니다. env 인덱스로 나눠줘야 합니다
   (`vec_task_wrapper_ma.py`).
3. **`_build_rand_action_probs`** — `amp_agent.py:496`이 `num_envs` 길이 텐서를 만드는데
   실제 배치는 `num_envs*M`입니다. 현재 cfg는 `enable_eps_greedy: False`라 값은 전부 1.0이지만
   shape은 여전히 틀립니다.
4. **entity-wise 관측 정규화** — 아래 §8.

---

## 8. Entity-wise 관측 정규화

`normalize_input: True`는 flat obs 벡터를 **차원별로** 정규화합니다. 그대로 두면 ego 슬롯의
230차원과 teammate 슬롯의 230차원이 **서로 다른 mean/std**를 갖게 되어, 같은 물리량이 슬롯에 따라
다른 스케일로 공유 토크나이저에 들어갑니다. 통일 설계의 permutation equivariance가 입력 스케일
단계에서 깨집니다.

`EntityRunningMeanStd`(`ma_agent.py`)가 이를 타입별로 묶습니다:

```
RunningMeanStd(230) × 1   # 모든 humanoid 슬롯이 공유
RunningMeanStd(39)  × 1   # 모든 object 슬롯이 공유
RunningMeanStd(6)   × 1   # 모든 goal 슬롯이 공유
```

`(B, M·size)` → `(B·M, size)` reshape → 정규화 → 복원. 부수 효과로 **정규화 통계가 M과 무관**해져
M=2 체크포인트를 M=3에서 그대로 쓸 수 있습니다.

TokenHSI 원본은 태스크별 토크나이저가 전부 분리되어 있어 이 문제가 없었습니다. 가중치를 공유하기
시작하면서 생기는 이슈입니다.

---

## 9. 시뮬레이션 쪽에서 일반화한 것

`env/tasks/humanoid.py`는 "env당 휴머노이드 1체 = actor 0"을 인라인으로 하드코딩하고 있어
서브클래스 훅으로는 우회할 수 없습니다(예: `dof_force_tensor.view(num_envs, num_dof)`가
M>1에서 즉시 예외). 그래서 `HumanoidMA`가 `Humanoid.__init__`을 거치지 않고 `BaseTask.__init__`을
직접 호출하며 텐서 뷰를 M-aware로 다시 만듭니다. **원본 파일은 수정하지 않았습니다.**

env당 actor 레이아웃 (순서가 곧 root-state 인덱싱):

```
[ humanoid_0 … humanoid_{M-1} , box_0 … box_{M-1} , (marker_0 … marker_{M-1}) ]
```

휴머노이드를 먼저 만들어야 rigid body가 `[0, M*num_bodies)`를 차지합니다.

M-aware로 다시 만든 텐서 뷰 (전부 sim 텐서의 **view**라 in-place 쓰기가 그대로 전파됩니다):

| 텐서 | shape |
|---|---|
| `_humanoid_root_states` | `(N, M, 13)` |
| `_dof_pos` / `_dof_vel` | `(N, M, num_dof)` |
| `_rigid_body_{pos,rot,vel,ang_vel}` | `(N, M, num_bodies, ·)` |
| `_contact_forces` | `(N, M, num_bodies, 3)` |
| `_box_states` | `(N, M, 13)` |
| `obs_buf` | `(N*M, obs_size)` |
| `rew_buf`, `_terminate_buf` | `(N*M,)` |
| `reset_buf`, `progress_buf` | `(N,)` — 에피소드는 env 단위로 공유 |

슬롯은 `(env_ids, agent_ids)` 인덱스 쌍으로 지정합니다(`_expand_slots`). 리셋은 항상 env 단위입니다.

**충돌**: `col_group = env_id`을 그대로 두어 같은 env의 휴머노이드끼리 **물리적으로 충돌**합니다.

---

## 10. 중간 동영상

학습이 제대로 되고 있는지 TensorBoard 곡선 말고 눈으로도 확인할 수 있게, headless에서도
off-screen 카메라 센서로 env 0을 녹화합니다.

```yaml
video:
  enable: True
  everyNEpochs: 50
  numFrames: 300      # 30Hz 기준 10초
  width: 1024
  height: 768
  focus: "all"        # "all" = env 0의 모든 캐릭터를 담음 | "agent0" = 한 명만 추적
  camDistance: 3.0    # 기준 거리. 에이전트가 흩어질수록 자동으로 커짐
  camHeight: 1.5      # 기준 높이. 같은 방식으로 커짐
  fov: 45.0           # IsaacGym 기본 화각은 1.8m 캐릭터에 너무 넓음
```

- `MAAgent.train_epoch()`이 N epoch마다 env에 녹화를 요청합니다.
- env는 `post_physics_step`에서 프레임을 모으고, `numFrames`가 차면 mp4로 씁니다 (OpenCV).
- 카메라는 env 0의 **휴머노이드들**만 프레이밍합니다
  (`dist = camDistance + 1.2 × 퍼짐반경`, `height = camHeight + 0.5 × 퍼짐반경`).
  박스를 프레이밍 대상에 넣으면 카메라가 뒤로 밀리고, 박스가 캐릭터를 가리는 일이 잦습니다.
- `enable: True`이면 headless여도 목표 marker actor를 생성합니다(영상에서 목표가 보이도록).
- 저장 위치: `<output_path>/HumanoidMA_<시각>/videos/epoch_XXXXXX.mp4`
- 테스트 실행(`--test`)에서도 `videos/test.mp4` 한 편을 남깁니다.

**알려진 한계 / 함정** (실제로 다 밟았습니다):

1. `gym.set_camera_location(cam, env, pos, tgt)`는 env를 넘기면 좌표를 **env-local**로
   해석하는데 root state는 글로벌입니다. env 원점을 빼야 합니다.
   (env 0은 원점이 `(0,0,0)`이라 증상이 안 드러나서 더 헷갈립니다.)
2. `gymapi.CameraProperties()`의 **기본 화각이 매우 넓어서** 명시적으로 좁히지 않으면
   캐릭터가 수십 픽셀로 찍힙니다.
3. `gym.attach_camera_to_body(..., FOLLOW_POSITION)`는 이 환경에서 **검은 화면**이 나왔습니다.
   절대 좌표 방식(`set_camera_location`)을 씁니다.
4. **카메라 센서를 켜면 프로세스 종료 시 segfault(exit 139)가 납니다.**
   `MAX EPOCHS NUM!` 출력과 체크포인트 저장이 **끝난 뒤** 발생하는 IsaacGym teardown 크래시라
   학습 결과에는 영향이 없습니다. 다만 학습을 스크립트로 체이닝한다면 exit code를 무시해야 합니다.
5. 초기 학습 단계에서는 캐릭터가 곧바로 넘어져 바닥에 눕기 때문에 화면에서 작게 보입니다.
   `focus: "agent0"`으로 더 가까이 볼 수 있습니다.

비용: 녹화 중인 스텝에서만 `step_graphics` + `render_all_camera_sensors`가 돕니다
(50 epoch마다 300 스텝 ≈ 전체 스텝의 약 19%).

---

## 11. 검증 결과

### 관측 정확성

```
original TokenHSI obs (ego-only) : (6, 223)
owner-frame relative form        : (6, 226)
root-position 자리가 정확히 0     : True   (max|·| = 0.0)
그 3차원을 빼면 원본과 일치        : True   (max diff = 0.0)
```

### M 무관성

```
M=1: 4,100,770 params, 155 keys, obs=275,  tokens=3
M=2: 4,100,770 params, 155 keys, obs=550,  tokens=6
M=3: 4,100,770 params, 155 keys, obs=825,  tokens=9
M=4: 4,100,770 params, 155 keys, obs=1100, tokens=12

keys/shapes identical across M      : True
M=2 -> M=3 load_state_dict(strict)  : OK
M=1 -> M=2 미학습 스칼라             : 32개 (actor+critic 합계)
```

### 학습 (RTX 4080 16GB, 각 200 iteration, `obsFrame: owner`)

| | M=1 | M=2 |
|---|---|---|
| num_envs | 1024 | 512 |
| 휴머노이드 총 개수 | 1024 | 1024 |
| obs / row | 275 | 550 |
| VRAM | 10.6 GB | 11.8 GB |
| fps (total) | ~16,000 | ~12,500 |
| `episode_lengths` | 30.7 → **71.1** (최고 84.0) | 29.4 → **43.3** |
| `rewards` | 0.64 → **13.2** | −0.10 → **4.09** |
| `losses/disc_loss` | 18.3 → **1.66** | 18.7 → **1.77** |
| `info/disc_agent_acc` | 0.86 → 0.96 | 0.60 → 0.96 |
| exit code | 0 | 0 |

참고: USAGE.md의 TokenHSI stage1 레퍼런스는 139 iteration에서 `episode_lengths 31 → 57`입니다.
M=1은 128 iteration에서 이미 84.0에 도달했습니다.

M=2가 느린 것은 예상된 현상입니다 — 에피소드를 공유하므로 둘 중 하나만 넘어져도 리셋되고,
초기 보상이 음수인 것은 에이전트 간 충돌 패널티(§12-(3)) 때문입니다.

---

## 12. **결정이 필요한 항목** (지금은 제 추천안으로 넣어둔 것들)

아래 항목들은 합리적인 기본값을 골라 넣었지만, 연구 방향에 따라 달라질 수 있는 선택입니다.

### (1) 관측 프레임 — `obsFrame`
현재 `"owner"`. §4 참조. `"global"`이 원래 의도하신 전지적 표현이지만 heading 불변성을 잃습니다.
`owner`가 one-pass 성질을 그대로 가지면서 heading 불변을 유지하므로 기본값으로 두었습니다.
**A/B 실험 가치가 가장 큰 항목입니다.**

### (2) 에피소드 / 종료 조건
현재: **env 단위 공유 에피소드**. 누구든 넘어지면 env 전체가 리셋됩니다. 단 `terminate` 플래그는
**에이전트별**로 유지해서, "안 넘어졌는데 팀메이트 때문에 끝난" 에이전트는 truncation으로
올바르게 부트스트랩됩니다(`next_vals *= (1 - terminated)`).
- 대안: 에이전트별 부분 리셋. IsaacGym에서 `progress_buf`가 env 단위라 지저분해집니다.
- 영향: M이 커질수록 에피소드가 짧아집니다(누구 하나만 넘어져도 끝). M=4 이상에서 문제될 수 있습니다.

### (3) 에이전트 간 충돌 패널티
현재: `agentCollisionPenalty: True`, `coeff 0.5`, `dist 0.7`.
`max_j(clamp(0.7 - d_ij, 0) / 0.7)`를 두 에이전트 모두에게 뺍니다.
- 계수·거리 모두 **임의로 정한 값**입니다. 튜닝 필요.
- 애초에 패널티를 줄지, 아니면 물리 충돌만으로 충분한지도 결정 사항입니다.
- M=2에서 초기 reward가 음수로 시작하는 원인입니다.

### (4) arena-local 4차원
`owner` 모드에서 에이전트 간 상대 위치를 복원 가능하게 하려고 humanoid 토큰에 추가한
`(root xy / envSpacing, heading cos, heading sin)`입니다.
- **장점**: 없으면 A가 B의 위치를 전혀 알 수 없어 (3)의 충돌 패널티가 학습 불가능합니다.
- **단점**: 절대 위치 정보가 관측에 들어옵니다. TokenHSI가 의도적으로 피하는 것입니다.
- `global` 모드에서는 중복이므로 제거 가능합니다(현재는 체크포인트 포맷 통일을 위해 유지).

### (5) goal 표현
현재 6-d = `[관측 프레임 기준 목표 위치(3), 박스→목표 벡터(3)]`.
원본 TokenHSI는 3-d(`local_tar_pos`)만 씁니다. 뒤의 3차원이 제가 추가한 것입니다.
- 회전이 선형이므로 `R(tar-box) = R(tar-ego) - R(box-ego)`로 계산합니다(quat_rotate 1회 절약).
- 목표 방향(orientation)은 넣지 않았습니다. carry는 위치만 평가하기 때문입니다.
  설계안 §4의 `ΔR_{goal,obj}`가 필요하면 goal 토큰을 확장하면 됩니다.

### (6) Relation type 집합
현재 6종. 대칭 관계만 표현합니다.
- 비대칭 구분(A→O와 O→A를 다른 타입으로)을 추가할 수 있습니다.
- 기하 정보(거리 구간 등)를 relation에 넣는 것은 설계안 §4에서 의도적으로 배제했으므로 넣지 않았습니다.
- 미할당 방해물 박스는 기존 `none` relation으로 지원합니다. 여러 박스를 한 에이전트가 동시에
  소유하도록 만들려면 assignment 표현을 추가로 일반화해야 합니다.

### (7) 원본 carry 대비 **누락된 기능** — 파리티가 필요하면 복원해야 합니다
- **`randomHeight` 플랫폼**: 박스를 공중(테이블 위)에서 집거나 놓는 시나리오.
  env당 `2M`개의 고정 actor와 두 번째 리셋 경로가 필요해서 첫 골격에서 제외했습니다.
  현재 `assert`로 막혀 있습니다.
- **`randomDensity` 박스 질량 관측**: object 토큰 폭이 고정이어야 해서 제외했습니다.
  넣으려면 object 토큰을 39 → 40으로 늘리면 됩니다.
- 두 기능 모두 TokenHSI carry 성능에 기여하므로, **원본과 직접 비교하려면 복원이 필요합니다.**

### (8) 스폰 / 목표 샘플링 기하
- 에이전트는 반지름 `agentSpawnRadius: 2.5`의 원 위에 배치.
- 목표 샘플링 반경: M=1은 ±4.5(원본과 동일), M>1은 `4.5 - spawnRadius = 2.0`으로 축소하고
  각자 자기 오프셋을 중심으로 샘플링.
- 박스 랜덤 배치 반경: 소유자로부터 `[1, tar_reach]`.
  **원본은 `[1, 10]`이었는데** envSpacing이 5인 것을 감안하면 원본 쪽이 이상해 보여 좁혔습니다.
- 전부 임의값입니다. M이 커지면 반경 배치가 비좁아집니다.

### (9) Critic 구조
현재: **actor와 동일한 구조의 별도 relation-transformer** + value head (가중치 분리).
- 설계안 §8의 "policy transformer 출력을 critic에 넣지 않는다"를 따랐습니다.
- 대안 A: actor와 트렁크 공유 (연산 절반, 하지만 설계안 위배).
- 대안 B: flat MLP (원래 이렇게 만들었다가 **M 무관성이 깨져서** 교체했습니다 —
  `critic_mlp.0.weight`가 M에 따라 542 vs 813).
- 현재 방식은 트랜스포머 트렁크 연산이 2배입니다. 트렁크가 작아서(64-d, ≤12토큰) 감내 가능합니다.

### (10) AMP discriminator
현재: **모든 에이전트가 하나의 discriminator를 공유**하고, style reward는 에이전트별로 나옵니다.
- 배치가 `(num_envs*M, 1290)`이 되어 자동으로 이렇게 됩니다.
- 에이전트별 discriminator를 두는 것도 가능하지만, 스타일은 공통이어야 하므로 공유가 맞다고 봅니다.

### (11) 롤아웃 dedup 최적화 (미구현)
`owner`/`global` 모드에서는 한 env의 M개 row가 서로 순열이므로, 롤아웃에서 트랜스포머를
env당 1회만 돌리고 M개 토큰 출력을 흩뿌릴 수 있습니다.
- **PPO 업데이트에서는 불가능합니다** — 미니배치가 셔플되므로 각 row가 독립 평가되어야 합니다.
- 실측상 업데이트가 wall time의 약 1/3이고 롤아웃 2/3의 대부분은 물리 시뮬이라 이득이 작습니다.
- 필요하면 추가하겠습니다.

### (12) M=1 → M≥2 전이 프로토콜
현재는 각각 처음부터 학습합니다.
- 권장: M=1로 carry 능력 확보 → **M=2 zero-shot 성능을 먼저 측정**(미학습 파라미터가 32개뿐) →
  그 다음 finetune. 그 갭 자체가 논문에 쓸 수 있는 수치입니다.
- `--checkpoint <M=1 ckpt> --resume 1`로 이어받을 수 있습니다.

### (13) 규모 / VRAM
16GB 기준 M=1은 1024 envs, M=2는 512 envs로 잡았습니다(휴머노이드 총 개수 고정).
M=4까지 가면 256 envs가 됩니다. env 다양성이 줄어드는 것과 총 휴머노이드 수를 유지하는 것
사이의 트레이드오프입니다.

### (14) 동영상 프레이밍
현재 env 0의 휴머노이드들을 담도록 자동으로 거리를 잡습니다(`camDistance + 1.2 × 퍼짐반경`).
- 캐릭터가 화면 정중앙이 아니라 한쪽으로 치우쳐 잡히는 경우가 있습니다.
  `set_camera_location`의 좌표 해석이 문서와 미묘하게 다른 것으로 보이며, 정밀 보정은 하지
  않았습니다. 학습 모니터링 용도로는 충분히 보입니다.
- M이 커지면 캐릭터가 작아집니다. `focus: "agent0"`로 한 명만 가까이 볼 수 있습니다.
- 완성도 있는 영상이 필요하면 `--test`로 체크포인트를 불러와 뷰어(VNC)로 보는 쪽을 권합니다.

---

## 13. 구현 중 발견한 것 / 주의사항

### TorchScript 레거시 실행기 퓨전 버그
IsaacGym의 `BaseTask.__init__`이 `torch._C._jit_set_profiling_executor(False)`를 켭니다.
이 레거시 실행기에서 **같은 사원수로 `quat_rotate`를 두 번 호출하고 결과를 `cat`하면**
`vector::_M_range_check` 런타임 에러가 납니다(개별 호출은 정상, 프로파일링 실행기에서도 정상).

`compute_object_goal_observations`에서 이 패턴에 걸렸고, 회전이 선형이라는 점을 이용해
`R(tar - box) = R(tar - ego) - R(box - ego)`로 바꿔 `quat_rotate` 호출 하나를 없애 해결했습니다.
결과적으로 더 빠르기도 합니다. **새 jit 함수를 추가할 때 주의하세요.**

### `_M_range_check` 류 에러의 디버깅
IsaacGym에서 CUDA device-side assert는 비동기라 엉뚱한 줄을 가리킵니다.
`CUDA_LAUNCH_BLOCKING=1`로 재현하면 실제 위치가 나옵니다.

### 파라미터 삭제
`AMPBuilder.Network`가 만든 `actor_mlp`/`mu`/`mu_act`/`critic_mlp`/`value`/`value_act`를
`del`로 제거하고 트랜스포머로 대체합니다. `sigma`와 discriminator는 그대로 씁니다.

---

## 14. 다음 단계 제안

| 단계 | 내용 | 검증 기준 |
|---|---|---|
| **M1** | M=1을 수렴까지 학습 | 기존 `output/single_task/ckpt_carry.pth` eval과 성공률 비교 |
| **M2** | M=1 가중치로 **M=2 zero-shot 측정** | 미학습 파라미터 32개뿐 — 갭 자체가 결과 |
| **M3** | M=2 finetune, 충돌 패널티 튜닝 | 두 에이전트 모두 목표 도달, M=1 대비 하락폭 |
| **M4** | M=3, 4 zero-shot | **핵심 결과** — cardinality 외삽 |
| **M5** | `obsFrame` A/B (`owner` vs `global`) | heading 불변성의 실제 기여도 정량화 |
| **M6** | 원본 파리티 복원 (§12-(7)) | TokenHSI carry와 직접 비교 가능 |
