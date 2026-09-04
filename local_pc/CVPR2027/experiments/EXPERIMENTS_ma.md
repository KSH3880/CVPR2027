# ma 실험 기록

현재 계획은 [PLAN_ma](../plan/PLAN_ma.md), 방법론은 [METHOD_V2](../METHOD_V2.md)를 따른다.
자동 수집 결과는 [AUTO_RESULTS_ma](AUTO_RESULTS_ma.md)이며,
**이 파일은 검토된 비교·판정의 기준이다.** 단계가 끝날 때마다 해석과 결론을 갱신한다.

**프레이밍**: adapt 확장이 아니다. TokenHSI의 기존 스킬(Carry 수행 능력)을 **초기화로만** 가져가고,
그 위에 **팀 폴리시를 새로 형성**한다. frozen은 진단 스캐폴딩이지 최종 상태가 아니다.

<!-- AUTO_RESULTS -->

## 자동 결과 (최신순)

> 자동 생성 영역이다. 최신 결과가 위, 가장 오래된 결과가 아래다.
> 비교·채택·기각 해석은 이 블록 아래의 수동 기록에 남긴다.

| 완료 시각 | 상태 | tag | 자동 요약 | 큐 질문 |
|---|---|---|---|---|
| 2026-08-19 17:28 | 완료 | `g1_live_s1` | MA_EVAL_SUMMARY tag=g1_live_s1 rc=0 sr=0.8989 eps=2348 fin=0.8301 t50=315 colEp=0.2751 colStep=17.96 enter=0.46 still=185.40 path=12.89 pathA=13.07,12.64 | live 시드 1. 910/3000 에서 재개 |
| 2026-08-19 17:17 | 완료 | `g1_zero_s0` | MA_EVAL_SUMMARY tag=g1_zero_s0 rc=0 sr=0.9067 eps=2190 fin=0.8936 t50=308 colEp=0.2868 colStep=19.38 enter=0.46 still=217.41 path=13.14 pathA=13.23,13.06 | spawn gap 적용 대조군. 942/3000 에서 재개 |
| 2026-08-19 16:38 | 완료 | `g1_c5_live` | MA_EVAL_SUMMARY tag=g1_c5_live rc=0 sr=0.8862 eps=2334 fin=0.8419 t50=304 colEp=0.3051 colStep=16.74 enter=0.49 still=199.83 path=12.38 pathA=12.52,12.22 | c=5 재확인. 1602/3000 에서 재개 |
| 2026-08-19 16:26 | 완료 | `t6_nf_c5` | MA_EVAL_SUMMARY tag=t6_nf_c5 rc=0 sr=0.8950 eps=2266 fin=0.8698 t50=315 colEp=0.3098 colStep=15.40 enter=0.52 still=199.92 path=13.22 pathA=13.44,12.97 | 동결 해제 + c=5. 1891/3000 에서 재개 |
| 2026-08-19 16:24 | 완료 | `g1_live_s0` | MA_EVAL_SUMMARY tag=g1_live_s0 rc=0 sr=0.8931 eps=2294 fin=0.8439 t50=321 colEp=0.2868 colStep=16.93 enter=0.47 still=187.39 path=13.35 pathA=13.49,13.11 | 토큰만 켠다. 1717/3000 에서 재개 |
| 2026-08-19 16:13 | 완료 | `t6_nf_c0` | MA_EVAL_SUMMARY tag=t6_nf_c0 rc=0 sr=0.8643 eps=2254 fin=0.8469 t50=325 colEp=0.3035 colStep=18.62 enter=0.48 still=194.24 path=13.55 pathA=13.71,13.34 | **동결 해제 · 대조군.** 1969/3000 에서 재개. 중간 지표 `path=8.62` — 동결 런들(12.3~13.0)보다 낮고 A=1 기준선 9.07 에 가깝다 |
| 2026-08-19 15:22 | 완료 | `g1_zero_s1` | MA_EVAL_SUMMARY tag=g1_zero_s1 rc=0 sr=0.8740 eps=2270 fin=0.8599 t50=312 colEp=0.3057 colStep=21.54 enter=0.48 still=199.58 path=13.28 pathA=13.10,13.58 | zero 시드 1. 1735/3000 에서 재개 |
| 2026-08-19 08:53 | 완료 | `t1_live_e4096` | MA_EVAL_SUMMARY tag=t1_live_e4096 rc=0 sr=0.8687 eps=2210 fin=0.8674 t50=325 colEp=0.3855 colStep=24.64 enter=0.63 still=205.79 path=13.06 pathA=13.17,12.98 | **env 학습 안정성.** `t1_live_s0` 과 env 수만 다르다. minibatch 를 같은 배수로 올려 롤아웃당 업데이트 48 회를 유지한다 — 안 올리면 96 회가 되어 학습 자체가 달라진다(ENV_SCALING §1, steer 16k 가 걸린 지점). 성공률·근접이 크게 흔들리면 2048 확정을 재검토한다 |
| 2026-08-19 07:46 | 완료 | `t4_c2_live` | MA_EVAL_SUMMARY tag=t4_c2_live rc=0 sr=0.8867 eps=2340 fin=0.8393 t50=316 colEp=0.3940 colStep=20.66 enter=0.64 still=199.47 path=12.74 pathA=12.78,12.61 | **c 를 실효 크기로.** c=0.2/0.5 는 벌점이 보상의 0.08~1.8 % 라 아무 일도 안 했다(실측: 에피소드 보상합 312, 벌점 총량 0.26). c=2 면 ~7 % |
| 2026-08-19 07:42 | 완료 | `t4_c2_zero` | MA_EVAL_SUMMARY tag=t4_c2_zero rc=0 sr=0.8809 eps=2410 fin=0.8116 t50=296 colEp=0.3884 colStep=21.50 enter=0.62 still=208.52 path=12.17 pathA=12.21,12.10 | 같은 c=2 에서 토큰만 0. **c 가 실제로 물릴 때 토큰이 쓸모를 갖는가** — 1단계에서 이 질문을 못 던진 건 c 가 죽어 있었기 때문이다 |
| 2026-08-19 06:57 | 완료 | `t3_mkspn` | MA_EVAL_SUMMARY tag=t3_mkspn rc=0 sr=0.8809 eps=2346 fin=0.8325 t50=313 colEp=0.3725 colStep=18.74 enter=0.61 still=201.23 path=12.43 pathA=12.42,12.43 | **3단계 ①.** dense 팀 항 없이 **종단 makespan 만**. `success_all·(294/T_last)`. 600 프레임에 신호 한 번이라 이중 credit assignment 에 정면으로 걸린다 — **안 되는 걸 보이는 것이 ②③를 쓰는 근거**가 된다 |
| 2026-08-19 06:44 | 완료 | `t5_r12` | MA_EVAL_SUMMARY tag=t5_r12 rc=0 sr=0.8784 eps=2380 fin=0.8210 t50=308 colEp=0.3756 colStep=19.97 enter=0.66 still=199.53 path=12.12 pathA=12.22,11.91 | **입력축.** 상대 root·속도·heading 만(12). 반응적 회피에 그것만으로 충분한가. 양 끝(zero=0, live=21)은 t1 과 t2_c05_b00 이 겸한다 |
| 2026-08-19 06:33 | 완료 | `t5_r18` | MA_EVAL_SUMMARY tag=t5_r18 rc=0 sr=0.8833 eps=2330 fin=0.8416 t50=315 colEp=0.3682 colStep=19.72 enter=0.59 still=202.66 path=12.64 pathA=12.86,12.39 | **입력축.** + 상대 박스(18). 상자가 독립적으로 길을 막는 게 중요한가. r18→live 에서 안 좋아지면 *끝점을 알아도 부족하다* = WM 논거 |
| 2026-08-19 06:32 | 완료 | `t3_minp` | MA_EVAL_SUMMARY tag=t3_minp rc=0 sr=0.8848 eps=2352 fin=0.8261 t50=320 colEp=0.3639 colStep=20.71 enter=0.60 still=190.26 path=12.47 pathA=12.73,12.27 | **3단계 ②.** β 자리를 가중평균 → `min_i(progress)` 로 **교체**. c=0 이라 충돌 벌점 없이 **팀 결합 형태만** 본다. 게으름이 구조적으로 불가능한 형태 |
| 2026-08-19 04:49 | 완료 | `t4_c5_zero` | MA_EVAL_SUMMARY tag=t4_c5_zero rc=0 sr=0.8647 eps=2414 fin=0.8227 t50=290 colEp=0.3819 colStep=20.86 enter=0.61 still=215.04 path=11.90 pathA=11.73,11.99 | 같은 c=5 에서 토큰만 0 |
| 2026-08-19 04:38 | 완료 | `t4_c5_live` | MA_EVAL_SUMMARY tag=t4_c5_live rc=0 sr=0.8740 eps=2362 fin=0.8273 t50=310 colEp=0.3929 colStep=20.93 enter=0.63 still=201.76 path=12.23 pathA=12.59,11.90 | c=5. 벌점이 보상의 ~18 %. **여기서도 근접이 안 내려가면 벌점 형태 자체를 다시 봐야 한다** |
| 2026-08-19 00:43 | 완료 | `t1_live_s0` | MA_EVAL_SUMMARY tag=t1_live_s0 rc=0 sr=0.8838 eps=2294 fin=0.8588 t50=312 colEp=0.4037 colStep=21.95 enter=0.68 still=209.07 path=12.83 pathA=12.88,12.75 | 보상 변경 없이 teammate 21-D 입력만으로 회피가 생기는가 |
| 2026-08-19 00:42 | 완료 | `t2_c05_b25` | MA_EVAL_SUMMARY tag=t2_c05_b25 rc=0 sr=0.8652 eps=2294 fin=0.8461 t50=324 colEp=0.3670 colStep=22.21 enter=0.62 still=201.56 path=12.98 pathA=12.84,13.08 | 강한 벌점·보상 공유에서 lazy-agent가 생기는지 본다 |
| 2026-08-19 00:41 | 완료 | `t1_zero_s0` | MA_EVAL_SUMMARY tag=t1_zero_s0 rc=0 sr=0.8818 eps=2348 fin=0.8360 t50=306 colEp=0.3731 colStep=21.44 enter=0.61 still=207.50 path=12.51 pathA=12.64,12.37 | **대조군.** 토큰 자리는 있고 값만 0. 관측 크기가 live 와 같아 같은 네트워크로 비교된다 |
| 2026-08-19 00:36 | 완료 | `t2_c02_b25` | MA_EVAL_SUMMARY tag=t2_c02_b25 rc=0 sr=0.8584 eps=2372 fin=0.8162 t50=319 colEp=0.3735 colStep=21.15 enter=0.63 still=189.45 path=12.60 pathA=12.75,12.38 | 약한 벌점에 보상 공유를 더했을 때 실효 세기가 변하는가 |
| 2026-08-19 00:35 | 완료 | `t2_c02_b00` | MA_EVAL_SUMMARY tag=t2_c02_b00 rc=0 sr=0.8760 eps=2382 fin=0.8296 t50=309 colEp=0.3896 colStep=19.93 enter=0.60 still=201.09 path=12.33 pathA=12.38,12.25 | 충돌 벌점만 약하게. `c`가 회피를 만드는 최소선인가 |
| 2026-08-19 00:35 | 완료 | `t2_c05_b00` | MA_EVAL_SUMMARY tag=t2_c05_b00 rc=0 sr=0.8936 eps=2290 fin=0.8655 t50=316 colEp=0.4052 colStep=23.99 enter=0.68 still=201.87 path=12.90 pathA=13.14,12.68 | 강한 충돌 벌점이 회피와 makespan에 주는 영향을 본다 |
| 2026-08-19 00:28 | 완료 | `t1_zero_s1` | MA_EVAL_SUMMARY tag=t1_zero_s1 rc=0 sr=0.8628 eps=2336 fin=0.8318 t50=307 colEp=0.3793 colStep=23.91 enter=0.61 still=207.39 path=12.38 pathA=12.50,12.27 | zero 대조군의 학습 시드 변동 |
| 2026-08-18 21:46 | 완료 | `t1_live_s1` | MA_EVAL_SUMMARY tag=t1_live_s1 rc=0 sr=0.8696 eps=2424 fin=0.7995 t50=316 colEp=0.3680 colStep=21.45 enter=0.62 still=187.06 path=12.55 pathA=12.46,12.62 | live 의 학습 시드 변동 |

<!-- /AUTO_RESULTS -->

## 운영 복구 (2026-08-18)

동일 tag를 재사용한 학습이 구 종료·평가 기록과 충돌해 현재 실행이 조기 완료 처리된 사고를
복구했다. 구 아티팩트는 `runs/results/ma/superseded_tag_reuse/`에 보존했고 현재 8개 실행은
활성 큐와 상태 원장에 다시 연결했다. 이후는 기존 출력 디렉터리가 있는 tag의 재실행을
거부하고, 실행 중인 프로세스를 구 종료 표식보다 먼저 판정한다.

## 전환 시 보존한 확정 설계 근거 (2026-08-18)

- 실행 기준은 A=1에서 4,096 env, A=2에서 2,048 env로 총 row 수를 4,096에 맞춘다.
  `envSpacing=5`, contact pairs 4 M, minibatch 16,384를 기준으로 쓴다.
- 단독 기준선 `b_solo`는 성공률 0.9600, 완료 중앙값 294 step, 이동 거리 9.07 m다.
  순차 실행 상한은 588 step이며 A=1 충돌 0.000은 계측 대조군이다.
- teammate 입력은 상대 root 위치·속도 6D, heading 6D, box 위치·속도 6D,
  target 3D의 총 21D다. ego-root yaw 프레임, 10 m clip, pairwise 구조를 쓴다.
- zero/live 비교는 입력 크기와 네트워크 구조를 같게 두고 값만 바꾼다.
- 충돌은 두 사람의 15×15 body 최소거리 `d_min`으로 재며 같은 정의를 지표와 벌점에 쓴다.
- 충돌 벌점은 `c·(pin(d_min)-1)`로 원본 보상 천장을 유지하고, 여러 상대는 합이 아닌
  최댓값으로 집계한다.
- 팀 보상은 `(1-β)·r_i + β·r_j`를 첫 후보로 쓴다. 에이전트별 성공률·이동 거리·
  자기 보상 비중·deadlock으로 lazy-agent를 판정하며 실패하면 `min_i(progress)`를 검토한다.
- 현재 문제는 각자 자기 물체를 운반하며 공간만 공유하는 space-time scheduling이다.
  steer와는 각 설계를 독립 확정한 뒤 결합한다.

## 초기 사다리 스냅샷 (2026-08-14)

아래 표는 M0 착수 당시 기록이며 현재 상태가 아니다. 현재 큐와 열린 질문은
[PLAN_ma](../plan/PLAN_ma.md)를 따른다.

| 단계 | 내용 | 상태 | 결과 |
|---|---|---|---|
| M0 | humanoid 2명 배선 + 멀리 배치 (학습 없음) | **재설계 필요** | 평탄화가 A≥2에서 복사본이 됨. agent 축 유지해야 |
| M1 | 가까이 배치 — baseline 만들기 (학습 없음) | 대기 | |
| M2 | teammate 입력 추가, 꺼둔 상태 | 대기 | |
| M3 | 충돌 페널티만 — wait 실패 모드 확인 | 대기 | |
| M4 | makespan 공유 보상 | 대기 | |
| M5 | 태스크 1 — 경로 교차 | 대기 | |
| M6 | 태스크 2 — 병목 통로 | 대기 | |
| M7 | 태스크 3 — Corridor-Clearing | 대기 | |

상태: `대기` / `진행중` / `통과` / `실패→분기` / `보류`

## 기록 규칙

각 단계가 끝나면 아래 형식으로 결과를 남긴다. **실패한 배치도 반드시 남긴다.**

```
## M<N> 결과 (YYYY-MM-DD)

설정: 체크포인트 / cfg / num_envs / iteration
슬롯별 결과:
  1 (대조군) : 둘다성공률 / makespan / 충돌 / deadlock / 단독대비지연
  2          : ...
판정: 통과 / 실패
다음: 어느 분기로 갔는지, 왜
```

M1부터는 **항상 같은 5개 지표**를 잰다: 둘 다 성공률 / makespan / 충돌 횟수·지속 /
deadlock rate / 단독 수행 시간 대비 지연.

---

## 초기 설계 기록

> 워크플로(TeamHOI 인프라·학습·transformer + TokenHSI 포팅 표면 + 선행연구 정독
> → 3안 설계 → 합성 → 적대적 검증) 결과 반영 예정.
> 현재 확정 내용은 [PLAN_ma](../plan/PLAN_ma.md)가 기준이다.

---

## 결과 로그

(아직 없음)


## M0 착수 (2026-08-14 02:00)

steer가 5시간 이상 무인으로 안정적으로 돌아서 계획대로 ma를 병행 시작했다.

### numAgents 지원 — A=1에서 기존과 동일하도록

`tokenhsi/env/tasks/humanoid.py`에 29줄 변경. 핵심은 **평탄화**다:

```python
# humanoid는 매 env의 앞쪽 A개 actor 슬롯을 차지한다.
# 모든 humanoid 텐서는 agent 축을 가진 뒤 (num_envs * A, ...)로 평탄화된다.
# A == 1이면 shape이 기존 단일 에이전트 코드와 완전히 같다.
self._humanoid_root_states = self._root_states.view(
    self.num_envs, num_actors, 13)[:, :A].reshape(self.num_envs * A, 13)
```

같은 방식으로 dof(`[:, :A*num_dof]`), rigid body(`[:, :A*num_bodies]`),
contact force를 바꿨다. actor id는 `env_id*num_actors + a`로 env당 A개 생성.

**평탄화를 고른 이유**: TeamHOI처럼 `(E, A, ...)` 축을 유지하면 학습 루프 전체를
고쳐야 한다(TeamHOI는 rl_games의 ExperienceBuffer를 직접 재구성한다).
평탄화하면 rl_games는 "env가 A배 많다"고만 보게 되고, **A=1에서는 아무것도 안 바뀌므로
회귀 검증이 성립한다.** 이게 M0의 통과 조건이다.

### 진행

- [x] 텐서 레이아웃 (root/dof/rigid body/contact force/actor id)
- [ ] A=1 회귀 테스트 — 실행중 (`runs/queue/logs/ma_regress_A1.log`)
- [ ] `_build_env`에서 A개 actor 생성 + 겹치지 않는 초기 배치
- [ ] `get_number_of_agents()` → rl_games에 A 알리기
- [ ] A=2, 멀리 배치, pretrained 정책으로 성공률이 단일과 같은지

**A=1 회귀를 통과해야만 A=2로 넘어간다.** 통과 못 하면 텐서 슬라이싱이 틀린 것이고,
그 상태로 진행하면 이후 모든 결과가 무의미해진다.

## M0 배선 완료 — A=1 회귀 3연속 통과 (02:15)

한 단계 고칠 때마다 A=1 회귀를 다시 돌렸다. 원본 baseline은 0.964다.

| 단계 | 변경 | A=1 회귀 |
|---|---|---|
| 1 | 텐서 평탄화 (root/dof/rigid body/contact force/actor id) | **0.958** |
| 2 | + `_build_env`에서 A개 actor 생성 | **0.971** |
| 3 | + 버퍼 평탄화 (obs/rew/reset/progress) + VecTask에 A 전달 | **0.958** |

셋 다 원본과 구분되지 않는다(런 간 노이즈 ±0.02 수준).
**평탄화 설계가 A=1에서 기존 코드 경로를 전혀 안 건드린다는 것이 확인됐다.**

### 왜 단계마다 회귀를 돌렸나

A=2에서 문제가 생겼을 때 "어느 단계에서 깨졌나"를 즉시 짚기 위해서다.
세 단계를 한꺼번에 바꾸고 A=2로 갔다면, 실패했을 때 후보가 셋이고
각각을 되돌려가며 이분 탐색해야 한다. 회귀 한 번이 5분인데
잘못된 방향으로 한 시간을 쓰는 것보다 싸다.

### 변경 요약

**`tokenhsi/env/tasks/humanoid.py`**
- `self.num_agents = cfg["env"].get("numAgents", 1)` — `super().__init__()` 전에 설정해야
  BaseTask가 버퍼를 할당할 때 쓸 수 있다
- humanoid 텐서를 `[:, :A]`로 자른 뒤 `(num_envs * A, ...)`로 평탄화
- actor id를 `env_id * num_actors + a`로 env당 A개 생성
- `_build_env`에서 A개 actor를 반지름 1.5 m 원형으로 배치(생성 시점 겹침 방지),
  에이전트별 색을 달리해 시각적으로 구분

**`tokenhsi/env/tasks/base_task.py`**
- `self._rows = num_envs * num_agents`를 도입하고 per-agent 버퍼를 전부 `_rows`로 할당.
  A=1이면 `_rows == num_envs`라 기존과 동일

**`tokenhsi/env/tasks/vec_task.py`**
- `self.num_agents = getattr(task, 'num_agents', 1)` — rl_games가 에이전트 수를 알게 한다

**`amp_humanoid_traj_sit_carry_climb.yaml`**
- `numAgents` 키 추가 (기본 1)

### 다음: A=2

첫 시도 실행 중(`ma_A2_first.log`, 64 env x 2 agent = 128 row).

**이 단계에서 기대하는 것은 "성공률이 좋다"가 아니다.** 지금 carry 태스크는
env당 상자가 하나뿐이라 두 에이전트가 같은 상자를 두고 경쟁한다.
확인하려는 것은 **크래시 없이 돌고, 128개 row가 서로 다른 관측을 갖고,
두 humanoid가 물리적으로 존재하는가**다.

per-agent 상자·목표는 M0b에서 다룬다.


## M0 설계 결함 발견 (02:16) — 평탄화는 A>=2에서 성립하지 않는다

A=2 첫 시도가 두 번 죽었다. 첫 번째는 힘 센서/dof force 텐서를 평탄화 안 한 것이라
쉽게 고쳤지만, 두 번째가 진짜 문제였다:

```
RuntimeError: view size is not compatible with input tensor's size and stride
```

`.reshape()`로 바꾸면 크래시는 사라진다. **그래서 위험하다.**

### 실증

```python
E, A, OBJ, D = 4, 2, 1, 13          # env당 actor = humanoid A개 + 물체 1개
flat = ...                           # (E*num_actors, D)  시뮬 메모리
v = flat.view(E, num_actors, D)[:, :A]     # humanoid만 슬라이스
v.is_contiguous()   # False
v.view(E*A, D)      # RuntimeError
r = v.reshape(E*A, D)
r[0,0] = -999
flat[0,0] == -999   # False  ← 복사본이다
```

**A=1에서는 view가 성립한다** (`[:, :1]`은 크기 1 차원이라 언제나 병합 가능).
그래서 A=1 회귀가 세 번 다 통과했는데도 A>=2에서 무너진다.

### 왜 치명적인가

이 텐서들은 IsaacGym 메모리를 가리키는 **창**이어야 한다.
- 리셋에서 `_humanoid_root_states[env_ids] = ...`로 쓴 값이 시뮬에 전달되어야 하고
- 매 스텝 `refresh_*` 후 갱신된 값이 읽혀야 한다

복사본이면 둘 다 끊긴다. **크래시 없이 조용히 틀린 결과**가 나온다.
`reshape`로 "고쳤다면" 이번 밤 내내 그럴듯한 숫자를 뽑으면서 전부 무의미했을 것이다.

### 왜 수학적으로 불가능한가

평탄 인덱스 `i = e*A + a`에 대응하는 메모리 오프셋은
`(i // A) * num_actors * D + (i % A) * D` 이다.
i에 대해 선형이 아니므로 **어떤 stride 조합으로도 표현할 수 없다.**
`num_actors == A`일 때만(물체가 없을 때만) 선형이 되어 평탄 view가 존재한다.

### 결론: agent 축을 유지해야 한다

TeamHOI가 `(E, A, ...)` 축을 그대로 두는 이유가 이것이었다. 우회가 아니라 필연이다.
`[:, :A]` 슬라이스 자체는 정상적인 view이므로 **(E, A, ...) 형태로는 안전하다.**
평탄화가 필요한 곳은 **읽기 전용 지점**(관측 계산, 학습 배치 구성)뿐이고,
**쓰기 경로(리셋)는 반드시 (E, A, ...) view를 통해야 한다.**

대신 학습 루프가 `(E*A)` 배치를 기대하므로, TeamHOI처럼
rl_games 쪽에 손을 대거나 관측/보상 버퍼만 별도로 평탄 텐서로 관리해야 한다.
이건 M0의 남은 작업이고, 2시에 급하게 할 일이 아니라 제대로 설계해야 한다.

### 현재 상태

`numAgents: 1`로 되돌려 두었다. A=1 경로는 회귀 3연속 통과 상태로 정상이다.
힘 센서/dof force 평탄화(`num_envs * num_agents`)는 A=1에서 무해하므로 남겨둔다.

**교훈: "크래시가 사라졌다"는 고쳤다는 뜻이 아니다.**
`view`가 거부한 것을 `reshape`으로 통과시키는 것은 대개 의미가 바뀐다는 신호다.

## 2026-08-19 · 1단계(토큰축) · 2단계(c×β) — 8행, 각 3000 iter

기준선: 상한 `0.9463`(간섭 없음) · 출발점 `0.8604`(sep0) · 되찾을 몫 **8.6 포인트**.
신호 문턱 `Δ ≥ 0.05` (시드 1~2개, 학습 시드 변동 0.046).

| tag | sr | 근접에피 | t50 | 이동 | 근접/이동 | 게으름비 |
|---|---|---|---|---|---|---|
| 학습 전 | 0.8604 | 0.259 | — | 9.07 | 1.75 | — |
| `t1_zero_s0` | 0.8818 | 0.373 | 306 | 12.51 | 1.71 | 1.02 |
| `t1_zero_s1` | 0.8628 | 0.379 | 307 | 12.38 | 1.93 | 1.02 |
| `t1_live_s0` | 0.8838 | 0.404 | 312 | 12.83 | 1.71 | 1.01 |
| `t1_live_s1` | 0.8696 | 0.368 | 316 | 12.55 | 1.71 | 1.01 |
| `t2_c02_b00` | 0.8760 | 0.390 | 309 | 12.33 | 1.62 | 1.01 |
| `t2_c05_b00` | **0.8936** | 0.405 | 316 | 12.90 | 1.86 | 1.04 |
| `t2_c02_b25` | 0.8584 | 0.373 | 319 | 12.60 | 1.68 | 1.03 |
| `t2_c05_b25` | 0.8652 | 0.367 | 324 | 12.98 | 1.71 | 1.02 |

### 1단계 — 토큰 무시. 확정

`zero` 0.8723 vs `live` 0.8767, **Δ = +0.004**. 문턱 0.05 에 한참 못 미친다. 더 결정적인 건
**조건 내 시드 편차(0.019 / 0.014)가 조건 간 차이(0.004)보다 크다**는 것이다. 신호가 없다.

PLAN 이 "리워드 먼저, 입력 나중"으로 순서를 뒤집은 근거가 실측으로 확인됐다 —
**상대를 신경 쓸 이유가 보상에 없으면 정책은 토큰을 본다 해도 쓰지 않는다.**

### 2단계 — 승자 없음. 장기런 넣지 않는다

최고 `t2_c05_b00` 0.8936, `t1_live` 대비 **Δ=+0.017** → 승자 조건 ①(+0.05) **미달**.
②(t50 588 로 안 밀림)와 ③(게으름 없음)은 전부 통과했다. 미리 정한 규칙대로 장기런은 보류.

**β=0.25 는 두 c 에서 모두 성능을 떨어뜨렸다** (0.8760→0.8584, 0.8936→0.8652).
게으름은 없었으므로(비 1.02~1.04) 게으름 때문이 아니라, 상대 보상이 섞여
**자기 행동과 자기 보상의 연결이 흐려진 것**으로 보인다.

### 왜 c 가 아무 일도 안 했나 — 원인 확정

**벌점이 보상의 0.08 % 다.**

```
에피소드 보상합   312      (스텝당 0.52)
근접 스텝 중앙값   1       (평균 22)
c=0.5 벌점 총량   0.26    →  보상의 0.08 %  (평균 근접으로 잡아도 1.8 %)
```

`근접/이동` 이 전 조건에서 **1.62~1.93 으로 평평**하다 — 학습 전 1.75 를 포함해서.
`c` 가 회피를 전혀 만들지 않았다. **c=0.2/0.5 는 축을 탐색한 게 아니라 0 근처만 두 번 찍은 것이다.**

`근접에피`가 25.9 % → 37~40 % 로 오른 것은 벌점 실패가 아니라 **활동량 증가의 부산물**이다
(이동 9.07 → 12.3~13.0). 그래서 이동으로 정규화한 `근접/이동` 을 같이 봐야 한다.

### 다음 — c 를 실효 크기로

`t4_c2_*` / `t4_c5_*` 투입. c=2 면 보상의 ~7 %, c=5 면 ~18 % 다.
**zero 팔을 함께 넣는다** — 1단계에서 "c 를 걸면 토큰이 쓸모를 갖나"를 못 물었던 건
c 가 죽어 있었기 때문이다. c 가 실제로 물릴 때 다시 묻는 게 맞다.

> 미리 정한 분기는 "차이 없음 → `t2` 에 zero 팔 4칸"이었다. 그런데 그 규칙은
> **c 가 효과를 낸다는 전제**로 쓴 것이고, 측정이 그 전제를 부정했다.
> 죽은 c 위에서 zero 를 더 돌리면 "아무것도 아닌 것 vs 아무것도 아닌 것"이 된다.
> 그래서 같은 질문을 **살아있는 c** 로 옮겼다. 규칙을 어긴 게 아니라 전제가 무너진 경우다.
