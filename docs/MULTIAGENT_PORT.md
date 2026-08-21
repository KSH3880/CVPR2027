# 단일 → 다중 에이전트 이식 안내

TokenHSI 태스크를 A=1 에서 A≥2 로 옮길 때 무엇을 바꾸고 어디서 터지는지.
carry 를 이식하며 실제로 나온 버그만 적는다. sit/climb 추가나 steer 통합 때 이걸 본다.

## 0. 가장 중요한 것

**A=1 에서 항등이 되는 모든 것이 위험 지점이다.**

```
row // 1        == row          env↔row 혼동이 사라진다
_rows           == num_envs     크기가 같아 shape 오류가 안 난다
terminated      == reset        분리될 수가 없다
agent_rows(ids) == ids          헬퍼가 항등 함수가 된다
```

그래서 **A=1 회귀 테스트를 통과하면서 A=2 에서만 조용히 틀린다.** 크래시도 경고도 없다.
실제로 이 방식으로 성공률이 0.96 → 0.61 로 떨어진 적이 있고, 원인을 찾는 데 하루가 걸렸다.

## 1. 단위 규칙 (먼저 정하고 시작한다)

| 대상 | 단위 | 이유 |
|---|---|---|
| 태스크·스킬 | **env** | 물체가 env 당 하나면 에이전트별로 다른 태스크를 주면 배치가 모순된다 |
| 모션 시작 프레임 | **agent** | 둘이 다른 자세에서 시작해야 한다 |
| 관측·행동·보상·지표 | **row** | row = env*A + a. 사람마다 하나 |
| 물체 (박스·받침) | **row** | 에이전트마다 자기 것 |
| reset_buf·progress_buf | **env** | 씬이 공유물이라 에피소드 경계도 공유 |

`humanoid.py` 의 헬퍼:
```
agent_rows(env_ids)   env id -> 그 env 들의 모든 row id
all_rows()            전체 row id
humanoid_rows(t)      (E,A,...) -> (E*A,...)   **읽기 전용**
agent_axis(t)         agent 축을 되살린 창       **쓰기는 반드시 이걸 통해서**
per_env_rows(t, ids)  env 당 하나인 값을 row 수만큼 복제
```

## 2. 실제로 터진 것들

### env id 를 row 로 쓰기 — 가장 비쌌다
`_reset_ref_env_ids` 는 **env id** 를 담는다(스킬은 env 단위로 뽑고 모션만 env 당 A 개 뽑는다).
그런데 `loco_carry` 분기만 `ids // num_agents` 로 row 취급했다. 바로 위 `pickUp` 분기는
env 로 읽는다 — **한 파일에서 같은 값을 두 가지로 해석**하고 있었다.
env 5 를 row 로 읽으면 `env 2, agent 1` 에 박스를 놓고 나머지 에이전트 박스는 안 놓는다.

→ **인덱스 배열을 볼 때마다 출처를 따진다.** 만든 곳까지 거슬러 올라가 확인한다.

### view vs copy — 쓰기가 조용히 버려진다
`(E,A,...)` 를 `.reshape`/`.view` 로 펴면 torch 가 **사본을 반환**한다.
flat index `i = e*A + a` 의 오프셋이 `(i//A)*num_actors*D + (i%A)*D` 라 i 에 선형이 아니고,
어떤 stride 로도 표현이 안 되기 때문이다. 그 사본에 쓰면 시뮬레이터에 도달하지 않는다.

→ 쓰기는 `agent_axis(t)[e, a, ...]` 형태로만. 읽기만 `humanoid_rows(t)`.

### actor id 표 크기
박스·플랫폼이 에이전트당 하나가 되면 그 actor id 표도 **row 크기**다.
`_box_actor_ids[env_ids]` 처럼 env 로 인덱싱하면 **절반이 누락되는데 크래시가 안 난다.**
`set_actor_root_state_tensor_indexed` 에 안 들어간 물체는 리셋 위치가 시뮬에 반영되지 않는다.

### 물체를 옮길 때 딸린 것들
박스를 옮기면 **그 아래 platform·tar_platform 도 같이** 옮겨야 한다.
안 그러면 `carryResetRandomHeight` 에서 박스가 공중에서 떨어져 굴러간다.
증상은 "리셋 직후 박스 속도가 0 이 아님" 으로 관측에 나타난다.

### terminate 와 reset 의 분리
`reset_buf` 를 "전원이 끝나야"로 바꾸면, 넘어진 ragdoll 이 씬에 남아 **매 스텝 넘어짐 판정을
다시 만족**한다. `_terminate_buf` 가 계속 1 이 되고 그게 `extras["terminate"]` 로 나가
`next_vals *= (1 - terminated)` 에서 **파트너의 value bootstrap 까지 매 스텝 0** 으로 만든다.
실측: terminate=1 인 스텝의 99.7 % 가 리셋이 아닌 스텝이었다.

→ **실패는 한 명만 나도 즉시 env 를 끝낸다.** terminate 는 "이 스텝에 실패로 끝났다" 여야 한다.
   성공(IET)만 전원이 끝나야 하는 것으로 둔다.

### 초기 배치가 겹친다
원본 리셋은 env 하나에 사람 하나를 전제로 "env 원점 근처 랜덤"이면 충분했다.
사람이 둘이 되면 **둘 다 원점 근처로 뽑혀 겹친다.** 실측: 15 % 가 tau(0.3m) 안에서 시작,
박스는 16 % 가 관통 상태로 시작.

→ 피할 수 없는 벌점은 학습 신호가 아니라 잡음이다. 최소 간격을 강제한다
   (`enforce_spawn_gap`). 사람을 옮길 때 **박스·받침·목표를 같은 벡터로 함께** 옮긴다.

### 학습 경로는 별도로 뚫어야 한다
`--test` 가 돌아도 학습은 안 될 수 있다. 실제로 다섯 군데가 막혀 있었다:
- `run.py:get_env_info()` 가 `agents` 를 안 넘기면 rl_games 가 batch 를 절반으로 계산
- `_build_rand_action_probs` 가 env 크기 -> row 크기 액션과 shape 불일치
- `num_actors` 는 cfg 의 `numEnvs` 에서 온다. `--num_envs` 로 덮어도 안 따라온다
- `--checkpoint` 만으로는 복원이 **조용히 건너뛰어진다.** `--resume 1` 이 같이 있어야 한다
- `--resume` 은 epoch 카운터도 복원한다. stage1 이 40,000 이면 `--max_iterations 6000` 은 즉시 종료

### 뷰어도 별도다
`_draw_task` 의 `_box_states[:, 0:3]` 이 A=2 에서 **에이전트 축**을 자른다.
`_init_camera` 의 `_humanoid_root_states[0, 0:3]` 은 xy z 가 아니라 **앞 세 에이전트**를 고른다.
마커는 env 당 하나인데 궤적 표본은 row 당 하나다.

## 3. 검증 방법

**A=1 회귀만으로는 부족하다.** 위 항목이 전부 A=1 에서 항등이기 때문이다. 추가로:

1. `MA_DUMP=1` — 모든 텐서를 env / (E,A,·) / row 로 분류해 한 번에 찍는다.
   크래시를 하나씩 잡으면 남은 개수를 모른 채 진행하게 된다.
2. `MA_POSDUMP=1` — 리셋 직후 실제 좌표. 물체가 딸려 왔는지 눈으로 본다.
3. **관측 덤프 비교** — A=1 과 A=2 의 관측을 차원별로 대조한다.
   실제로 이 방법이 "박스가 안 옮겨졌다"를 dim 321 로 짚어냈다.
4. **독립 검토** — `terminate` 버그는 혼자서는 못 잡았다. 알려진 함정 목록으로
   다른 관점이 훑게 하는 게 값을 한다.
5. `rc=0` 도 `warn=0` 도 통과 조건이 아니다. 로그에 `=> loading checkpoint` 가 있는지,
   `fps step` 줄 수가 요청한 만큼인지 확인한다.

## 4. 학습 설정

```
A=1  env 4096 × 1명 = 4096 줄   minibatch 16384  ->  48 업데이트/롤아웃
A=2  env 2048 × 2명 = 4096 줄   minibatch 16384  ->  48 업데이트/롤아웃
```
**줄 수를 맞추면 레시피를 안 바꿔도 된다.** env 를 올리려면 minibatch 를 같은 배수로
올려야 한다 — 안 그러면 업데이트 수가 튀어 학습이 통째로 달라진다 (ENV_SCALING §1).

실측(env 2048 vs 4096): 성공률 차이 없음. 처리량은 4096 이 샘플당 2.3 배 유리.
