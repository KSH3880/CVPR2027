# coord 실험 계획

`TokenHSI-coord/`는 완성된 ms18 action executor를 frozen으로 두고, 실제 shared state에서
joint path와 속도를 생성하는 상위 coordinator를 검증한다. 아래 A/B와 C2는 기존 결과를
덮어쓰지 않는 독립 실험 트랙이다.

## 고정 계약

- A=2 Carry
- 입력: actual root/box/goal state와 measured phase만
- 출력: agent별 33점 path와 smooth speed
- 실행: 0.1m 간격 320점으로 resample 후 ms18의 6점/12-D steer token 사용
- replan: 기본 6 action step, phase 변화 시 즉시
- ms18 executor PTH는 frozen이고 변환·재저장하지 않는다.

## 독립 track 계약

### Track A — C1 learned-candidate coordinator

- 6-entity Transformer와 learned candidate `K=4`를 사용한다.
- joint path, smooth speed, pickup dwell, candidate risk를 함께 예측한다.
- PPO에 feasibility/diversity/risk auxiliary loss를 더한다.
- 태그와 체크포인트 prefix는 각각 `c1_*`, `coord_c1_*`이다.

### Track B — B0 minimal coordinator

- flattened state를 받는 2-layer Tanh MLP(128)만 사용한다.
- 출력 action은 `A/B lateral bow + A/B slowdown`의 4개뿐이다.
- pickup은 항상 직선이고 carry 경로만 최대 1m lateral bow를 더한다. 기본 속도는 1.5m/s이며
  출력은 감속만 할 수 있다.
- loss는 `PPO + 0.5 value - 1e-4 entropy`뿐이다. candidate/diversity/risk/path auxiliary는 없다.
- 태그와 체크포인트 prefix는 각각 `b0_*`, `coord_b0_*`이다.

### Track C2 — K=1 collision-first residual coordinator

- C1의 6-entity Transformer, hard-anchor residual path와 acceleration 기반 smooth speed만
  유지하고 learned candidate를 `K=1`로 줄인다.
- pickup dwell은 1.5초로 고정하고 diversity/risk/best-of-K loss를 제거한다.
- 보조 loss는 time-aligned future collision, speed smoothness, safe-scene slowdown 억제 세 항뿐이다.
- 동일 경로를 1.5m/s로 주행한 counterfactual이 충돌할 때만 감속 비용을 면제한다.
- 태그와 체크포인트 prefix/schema는 `c2_*`, `coord_c2_*`, `tokenhsi-coord-c2-v1`이다.

### 서로 섞지 않는 규칙

- A와 B는 서로의 checkpoint로 초기화하지 않는다.
- 한 트랙의 모델·loss 수정은 다른 트랙 소스/설정에 자동 반영하지 않는다.
- 두 트랙이 공유하는 것은 frozen ms18 PTH, refreshed simulator state, 동일한 Free/Cross 평가
  protocol뿐이다.
- 수정은 반드시 새 태그로 실행하고, 한 번에 원인 하나를 크게 바꾼다. 잔수정과 fine sweep은
  하지 않으며 on/off 또는 10~50배 수준의 분명한 대조만 허용한다.
- 평가가 끝날 때까지 현재 프로세스를 중단하거나 덮어쓰지 않는다.

## 토이 실험 복잡도 제한

- Track A는 `state → K=4 path/speed → selector → frozen ms18`, Track B는
  `state → 4 actions → one path/speed → frozen ms18` 골격을 각각 유지한다.
- C2는 `state → K=1 residual path/smooth speed → frozen ms18`로 충돌 시점의 감속만
  분리해 검증하는 단순 대조군이다.
- 한 비교 실험에서는 원인 하나와 수정 하나만 바꾼다.
- 수정 순서는 loss/reward 계수, 기존 출력의 작은 수정, 마지막에만 encoder/decoder 구조다.
- C1에서 World Model, 별도 학습형 Planner, 추가 stage/head/dependency 표현을 도입하지 않는다.
- 기준 모델이 충돌 회피와 carry/place를 충분히 달성하면 더 복잡하게 만들지 않고 종료한다.
- loss와 reward도 수정 범위다. 지표상 원인이 확인되면 모델 구조보다 먼저 작은 loss/reward
  정의 또는 계수 변경으로 검증한다.
- 소수점 최적화를 위한 잔수정·촘촘한 sweep은 하지 않는다. 토이 가설이 명확히 갈리도록
  항의 on/off, 5~10배 계수 차이, K=1 대조, width/layer의 큰 변경만 비교한다.

## C1 단계

1. **구조 smoke** — pure-PyTorch shape/anchor/speed/checkpoint/gradient 검사.
2. **bridge smoke** — random-init은 성능이 아니라 PTH 분리 로드, fallback, top-view,
   320점/steer 관측 shape만 검사한다.
3. **closed-loop pilot** — ms18을 frozen으로 로드한 high-level PPO runner에서 coordinator만
   학습한다. Free/Cross/Close-box를 섞고 scenario id는 입력하지 않는다.
4. **판정** — carry/place 보호, HH/BB/HB 충돌, clearance, makespan, 속도 변화와 candidate
   collapse를 함께 측정한다.

1~4번과 두 seed의 Free/Cross 비교가 완료됐다. 2 env/1-iteration bridge smoke와 64-env
정식 학습에서 frozen ms18 rollout, gradient update, reset, numbered/latest PTH 저장을
통과했다. 실행 sidecar에는 executor PTH와 coordinator 설정을 함께 남긴다.

## 완료된 주요 결과

| 설정 | Cross success / collision | Free success / collision | 판정 |
|---|---:|---:|---|
| analytic root→box→goal | 0.698 / 0.119 | 0.921 / 0.300 | 공통 기준선 |
| C1 hard-invalid legacy | 0.164 / 0.087 | 0.432 / 0.271 | invalid sentinel로 학습 오염 |
| C1 continuous-cost | 0.122 / 0.064 | 0.505 / 0.303 | 자유 residual이 너무 커 task 붕괴 |
| C1 analytic-prior | **0.724 / 0.104** | **0.901 / 0.290** | Track A 첫 toy 성공 |
| C1 analytic-prior + collision 50 | 0.706 / 0.127 | 0.917 / 0.277 | 기존 C1보다 악화, 폐기 |
| B0 4-action MLP @100 | **0.755 / 0.112** | **0.928 / 0.237** | 최고 점수, invalid 12~15% |
| B1 invalid reward @500 | 0.736 / 0.137 | 0.904 / 0.302 | 허점은 감소, 성능 악화 |
| B2 safe-bow MLP @100 | 0.715 / 0.119 | 0.923 / 0.288 | invalid 0.1~0.2%, clean 기준 |

`c1_analyticprior_s0`는 pickup을 analytic straight로 고정하고 carry residual을 최대 1m로
제한했으며, path/speed head를 nominal 직선·1.5m/s에서 시작한다. Cross에서 analytic보다
success와 collision이 모두 개선됐고 Free 성능도 거의 보존돼 Track A 최종 기준이다.
held-out seed1에서도 Cross `0.729/0.108`, Free `0.926/0.216`이라 결과가 유지됐다.

## 병렬 실험 최종 판정

- Track A `c1_analyticprior_col50_s0`: 성공한 analytic-prior 구조는 그대로 두고 실제 collision
  reward 계수만 `2→50`으로 바꿨지만 250/500 모두 악화됐다. 기존 C1을 유지한다.
- Track B `b0_mlp_col50_s0`: 4-action minimal MLP, auxiliary loss 0, collision 계수 50인 독립
  baseline이다. 100 이후 200/300/400/500에서 성능이 표류했으므로 100이 early best다.
- Track B `b1_mlp_invalid1_col50_s0`: B0에서 모델·PPO·collision reward는 그대로 두고
  invalid fallback reward `-1`만 추가했다. invalid는 감소했지만 task/collision이 나빠져 폐기한다.
- Track B `b2_mlp_safebow_col50_s0`: reward는 B0 그대로 두고 carry bow를 carry 거리 이하로
  제한했다. 두 seed 모두 invalid `0.02~0.36%`이고 analytic 대비 평균 carry/place 저하는
  5%p 이내다. 실제 bridge 기본 추천은 이 100-iter PTH다.

최종 PTH는 서로 독립이며 둘 다 보존한다.

- Track A 표현력 기준: `runs/coord/c1_analyticprior_s0/coord_c1_latest.pth`
- Track B 최고 점수: `runs/coord/b0_mlp_col50_s0/coord_b0_000100.pth`
- Track B clean-runtime 추천: `runs/coord/b2_mlp_safebow_col50_s0/coord_b0_000100.pth`

토이 가설은 충분히 성립했다. 다음 단계는 구조/loss sweep이 아니라 선택한 PTH를 실제
state-only receding-horizon viewer에서 정성 확인하는 것이다.

## C2 collision-first 진행

2026-09-03에 기존 C1/B를 보존한 채 `c2_k1_collision_s0`를 시작했다.

```text
L_C2 = L_ppo + 0.5 L_value - 1e-4 entropy
     + 50 future_collision + 0.5 speed_smooth + 0.1 unnecessary_slow
r_C2 = r_ms18_carry + 2 phase_progress - 50 actual_collision - 0.01
     - 1 invalid_plan
```

`unnecessary_slow`는 같은 예측 경로를 전부 1.5m/s로 바꾼 counterfactual이 안전할 때만
적용한다. Cross 평가에서는 success/collision뿐 아니라 raw command speed bin의 lower-speed
비율과 두 agent 중 한 명만 감속한 episode 비율을 필수 판정값으로 쓴다.

### C2 중단 시점 인계 메모 (2026-09-03)

- 최종 의도는 명시적 yielder/속도 정답 없이 작은 MLP가 state에서 두 agent의 33개
  `(x,y,v)` 점을 함께 출력하고, 미래 충돌 loss만으로 감속 또는 우회를 학습하는 것이다.
- `c2_k1_jointpoint_future8_s0`의 50-iter seed0은 Cross success/collision
  `0.7448/0.0909`로 analytic `0.6979/0.1191`보다 좋았고 선택적 감속도 확인됐다.
  그러나 같은 PTH의 seed1/2/3 collision은 `0.1232/0.1859/0.1255`로 일반화되지 않았다.
- 실패 원인은 예측 경로상 충돌률은 거의 0인데 실제 충돌은 주로 carry phase에 남는다는
  점이다. 이상적인 경로 시간과 frozen ms18의 집기·보행 시간이 어긋나 정확히 같은 시각의
  위치만 비교하는 loss가 실제 충돌을 놓친다.
- `c2_k1_jointpoint_future8_t075_s0`는 비교 시각을 ±0.75초로 넓혔지만 기존 충돌 계수 50이
  너무 강해 20-iter seed0 success/collision/invalid가 `0.6914/0.1143/0.0619`로 악화돼
  기각했다.
- 현재 실행 중인 단일 후속은 `c2_k1_jointpoint_future8_t075_c10_s0`이다. 모델과 ±0.75초
  비교는 유지하고 future-collision 계수만 `50→10`으로 낮췄다. 이 실행이 끝나도 자동으로
  추가 구조/loss 실험을 만들지 않으며, 다음 goal에서 결과를 검토해 재개한다.
- 이 후속의 20-iter seed0 Cross는 success/collision/invalid
  `0.7435/0.1155/0.0046`이다. analytic `0.6979/0.1191` 대비 task는 보존했지만 충돌 개선은
  `0.36%p`뿐이라 아직 채택 근거가 아니며, 예약된 50-iter 평가를 기다린다.
- 50-iter seed0 Cross는 success/collision/invalid `0.7344/0.1200/0.0012`로 충돌이
  analytic보다 `0.09%p` 높다. 선택적 감속 episode는 `39.3%`지만 감속 중 충돌률 0이라는
  사실만으로 전체 충돌을 줄이지 못했다. 따라서 현 시점에서 `t075_c10`도 해결책으로
  채택하지 않는다. 실행은 사용자의 요청대로 500 iter까지 보존하되 추가 실험은 멈춘다.

## 이전 실험 기록

- `c1_cross_s0`: 2026-09-02 시작. 64 env, 500 iter, 10 iter 저장. 기존 hard-invalid
  auxiliary를 가진 기준선으로 보존한다. 초반 1~90 iter에서 invalid가 약 0.35→0.49로
  개선되지 않았고 aux가 `3~7e8`로 오염되는 원인을 확인했다.
- 다음 비교는 모델을 늘리지 않고, 선택용 `1e9`와 학습용 연속 constraint cost만 분리한
  단일 변경이다. 두 런은 같은 seed/배치와 Free/Cross 평가로 비교한다.

## frozen ms18 시간 보정 (2026-09-03)

이상적인 `거리/명령속도 + pickup 1.5초`가 실제 충돌 시각을 잘못 예측하는지 먼저
분리 측정했다. executor와 coordinator는 학습하지 않고, frozen ms18에 여섯 고정 profile을
보내 실제 반응만 저장했다.

- raw: `runs/coord_measurement/ms18_executor_sysid_v1_s{0,1}/cross/raw.npz`
- fit: `runs/coord_measurement/ms18_executor_sysid_v1/calibration.json`
- 명령속도: `[0.375, 0.75, 1.125, 1.5]m/s`
- approach 실속도: `[0.812, 0.898, 1.170, 1.401]m/s`
- carry 실속도: `[0.658, 0.838, 1.124, 1.310]m/s`
- pickup dwell: phase0 `0.900s`, phase1 remaining `0.533s`
- step 감속의 실제 반응 지연 중앙값: 약 `0.5~0.6s`

단순 4점 lookup+dwell만 적용해도 교차점 도착 오차 중앙값은 `1.219→1.038s`, p90은
`4.738→2.798s`로 줄었다. 두 agent 도착시간 차 오차 중앙값도 `1.006→0.767s`로 줄었다.
아직 step response까지 넣지는 않는다. 토이 실험답게 먼저 이 정적 보정 하나의 효과만 본다.

현재 비교 태그는 `c2_k1_jointpoint_future8_meastime_s0`이다. 기존
`c2_k1_jointpoint_future8_s0`와 모델·33개 `(x,y,v)` 출력·top-8 collision loss·계수는
동일하고, loss의 time alignment에만 `COORD_C2_MEASURED_EXECUTOR_TIMING=1`을 사용한다.
기존 ±0.75초 uncertainty는 동시에 쓰지 않는다.

첫 20-iter seed0 Cross는 success/collision/invalid
`0.7500/0.1210/0.0018`이다. task 성공은 높지만 collision은 analytic `0.1191`보다 아직
`0.19%p` 높아 초반 합격은 아니다. 학습을 중단하지 않고 예약된 50/100/250 평가로
추세를 확인한다.

500 iter와 자동 최종평가까지 완료됐다. Cross success/collision은 50에서
`0.7174/0.1119`, 100에서 `0.7318/0.1185`, 250에서 **`0.7109/0.1043`**, 500에서
`0.7201/0.1131`이다. Free-500은 `0.9245/0.2978`로 analytic Free
`0.921/0.300`을 보존했다. timing 보정은 seed0에서 task를 무너뜨리지 않고 충돌을 줄이는
방향은 확인했지만 후반 표류가 있어 250을 seed0 후보로 두고, 채택 전 다른 scene seed에서
재현성을 확인한다.

판정 순서:

1. 20/50/100/250 milestone에서 analytic 및 기존 future8 대비 실제 rigid-body collision을 본다.
2. 좋아지면 seed1/2로 재현성을 확인한다.
3. 그래도 예측상 안전하지만 실제 충돌이면 그때만 실제 rigid-body 근접 penalty를 high-level
   PPO reward에 별도 on/off한다. timing 보정과 reward 변경을 한 실험에 섞지 않는다.
4. step-response lag나 timing MLP는 위 두 단순 실험이 모두 실패할 때만 고려한다.

## random fixed-priority + consistency 비교 (2026-09-03)

두 agent가 동시에 감속하는 모호성을 없애기 위해 episode마다 한 명을 랜덤 priority로
정한다. priority는 직선 `root→box→goal`, 전 구간 `1.5m/s`로 고정하고 다른 한 명의
33개 `(x,y,v)`만 실제 조율 출력으로 사용한다. 모델 입력에는 role을 추가하지 않고 두
agent의 counterfactual yield 출력을 모두 만든 뒤 episode buffer가 사용할 쪽을 고른다.

기존 measured-timing C2의 모델·collision/efficiency/path loss는 유지한다. 추가 loss는
이전 계획을 실제 replan 간격만큼 전진시킨 뒤 현재 계획과 같은 절대 미래시각에서 비교하는
trajectory consistency 하나뿐이다. reset/phase 전환은 제외하며 collision 계수 50이
consistency 계수 1보다 강하므로 새 위험에는 대응할 수 있다.

- 태그: `c2_k1_jointpoint_future8_meastime_rprio_cons1_s0`
- 학습: 64 env, 500 iter, 10 iter마다 PTH
- 평가: Cross 20/50/100/250 milestone, 완료 후 Cross/Free 500
- 첫 iteration: invalid `0`, priority-agent1 비율 `0.493`, consistency `0.0181`

다음 loss 후보로 yield agent의 절대 완료시간이 아닌 동일 경로 전 구간을 `1.5m/s`로
주행했을 때 대비 **추가 지연**
`L_extra_delay = T_yield - T_yield,full-speed`를 둔다. 이것만으로는 감속 위치가
결정되지 않으므로, 채택한다면 구간별 full-speed 복원 counterfactual로 계산한
`L_locally_unneeded_slow`와 함께 비교한다. 현재 실행 중인 태그에는 두 항 모두 넣지 않는다.

## compact slowdown-window 두 트랙 (2026-09-03)

33점 각각의 속도를 독립 출력하던 구조를 토이 실험에 맞게 줄였다. 작은 MLP는 두 agent의
waypoint residual과 각 agent별 `(감속 중심, 폭, 세기)`만 한 번에 출력한다. decoder가 이를
거리축 cosine window로 펼쳐 최종 33개 `(x,y,v)`를 만들므로 ms18 bridge 계약은 같다.
episode마다 무작위 한 명은 직선·`1.5m/s` priority로 고정되고 다른 한 명만 실제로 양보한다.

속도 regularizer는 `speed_smooth`, `unnecessary_slow`, `speed_efficiency`를 모두 끄고,
같은 예측 경로를 전 구간 `1.5m/s`로 갔을 때보다 늘어난 양보 agent의 시간
`L_extra_delay` 하나만 사용한다. 안전은 measured executor timing의 top-8 future-collision,
경로는 residual, 재계획 안정성은 기존 consistency가 담당한다.

- `c2_k1_slowwin_w3_delay_s0`: 감속 폭 `0.75~3.0m`
- `c2_k1_slowwin_free_delay_s0`: 감속 폭 `0.75m~전체 남은 경로`
- 공통: Cross, 64 env, 250 iter, 10 iter 저장, 20/50/100/250 평가
- 두 실행의 유일한 실험 변수는 최대 감속 폭이다.

두 트랙 모두 250 iter까지 학습했지만 milestone evaluator의 실제 low command 비율은
약 `2.7~2.9%`에 머물렀고 학습 중 `extra_delay`도 거의 0이었다. 따라서 viewer 문제가
아니라 모델이 주로 waypoint 우회로 future-collision을 줄이고 slowdown depth를 사실상
사용하지 않은 것이다.

이를 분리하는 명시적 loss 대조군 `c2_k1_slowwin_explicit_s0`을 추가했다. 속도 정답이나
oracle trajectory는 사용하지 않는다. 양쪽 모두 직선·평속인 counterfactual에서 교차점을
찾고, episode priority 반대편인 yielder가 그 점에 1초 이상 늦게 도착하도록 gap hinge를
준다. 감속 window는 교차점이 박스 뒤면 왼쪽 경계, 박스 앞이면 오른쪽 경계를 박스 호좌표에
붙이며, 교차점을 1m 지난 구간은 다시 `1.5m/s`로 복원한다. reset 직후 계획 표본은 이 세
항에서 3배 가중한다. 감속 폭·세기 정답은 없고 collision/gap과 extra-delay 사이에서 정한다.

- loss 계수: future collision `50`, assigned gap `10`, box anchor `3`, post-cross restore `3`,
  extra delay/path residual/consistency 각 `1`
- speed smooth/unnecessary slow/speed efficiency/path smooth는 `0`
- Cross 64 env, 250 iter, 10 iter 저장, 20/50/100/250 자동 평가

## 감속 우선 time-cost 장기 비교 (2026-09-03)

기존 slowdown-window 두 트랙은 감속 추가시간만 명시적으로 비용화하고 우회는 작은 평균
waypoint residual로만 비용화해, 250 iter에서 사실상 감속 없이 우회로 수렴했다. 이를
같은 초 단위로 분해한다.

`delay_slow = T(predicted path, predicted speed) - T(predicted path, 1.5m/s)`

`delay_detour = T(predicted path, 1.5m/s) - T(straight path, 1.5m/s)`

새 두 트랙은 `delay_slow + 3 * delay_detour`를 사용한다. 충돌은 오직 실제 predicted
path/speed에서 계산하며, 우회도 허용하되 완료시간 효과가 비슷하면 감속을 선호한다.

- `c2_k1_slowwin_w3_timebias3_s0`: 최대 감속 폭 3m, detour time 가중치 3
- `c2_k1_slowwin_free_timebias3_s0`: 폭 제한 없음, detour time 가중치 3
- 둘 다 처음부터 3000 iter, 200마다 저장·Cross 평가
- 두 가중치 트랙이 끝난 lane은 각각 기존 `w3_delay`, `free_delay`의 250 PTH를 resume해
  3000까지 진행한다.
- 기존 `c2_k1_slowwin_explicit_s0`도 현재 250 실행과 평가가 끝나면 같은 PTH에서 3000까지
  resume한다. 동시에 실행되는 학습 lane은 최대 3개다.

## C3 conflict-attached slowdown window (2026-09-04)

긴 C2 태그와 자유로운 `(center,width,depth)` 표현을 끊고 새 계열을
`c3_crosswin_s0`로 시작한다. 작은 state-only MLP와 waypoint residual은 유지하되, agent별
속도 출력은 `(감속 길이, 감속 세기)` 두 값뿐이다. 두 예측 경로의 가장 가까운 지점을
교차점으로 계산하고 감속구간을 `[교차점-length, 교차점]`에 구조적으로 배치한다. 구간 안은
cosine 감속, 교차점과 이후는 정확히 `1.5m/s`다. 경로와 두 속도 파라미터는 같은 head에서
동시에 출력되며 GT trajectory/speed/scenario ID는 사용하지 않는다.

loss는 가장 안정적이던 timebias3를 그대로 사용한다:

`50 future_collision + path_residual + extra_delay + 3 detour_delay + consistency`.

episode random priority, measured ms18 timing, top-8 collision도 동일하다. 따라서 유일한 핵심
변경은 감속 window의 위치 자유도를 제거한 것이다. 2000 iter, 200마다 PTH 저장 및 별도
Cross 평가로 감속 사용률·실제 충돌·invalid를 본다.

## C4 general proximity collision (2026-09-04)

C2의 자유로운 `(감속 중심, 폭, 세기)` 출력을 유지하면서, 기존 1m hinge 충돌항만
연속적인 미래 proximity로 교체한다. 96개 time-aligned future root 위치에서 box 크기와
carry 여부로 agent별 동적 반경을 만들고, 정규화 거리 `rho`에 대해
`exp(-0.5*rho^2)`를 사용한다. 서로 가까워지는 시점에는 closing-rate를 곱해 더 강하게
벌점을 주고, 멀어지면 자동으로 작아진다. Cross 판정, 교차점, scenario ID는 사용하지 않는다.

- `c4_d3_s0`: proximity 5, detour-delay 3
- `c4_d10_s0`: proximity 5, detour-delay 10
- 나머지는 동일: free slowdown window, random one-yielder, measured timing, top-8,
  extra-delay/path-residual/consistency 각 1, 2000 iter, 200마다 저장·Cross 평가
- 초기 state 분포는 우선 그대로 두어 loss 변경 효과만 비교한다.

### C3/C4 depth dead-zone 수정 실험

기존 slowdown decoder는 zero-centred softplus 뒤에 clamp를 사용해 raw depth가 음수가
되는 순간 depth와 gradient가 모두 0이 됐다. 실제 C3 200/400/600 및 C4 두 200 PTH에서
대표 Cross state의 decoded depth가 모두 0이고 최저 명령속도도 1.5m/s임을 확인했다.
기존 실행은 보존·종료하고, 새 체크포인트 옵션 `smooth_depth`에서만
`depth=1.125*sigmoid(raw-3)`을 사용한다. 초기 감속량은 약 0.053m/s로 작지만 음수
영역에서도 gradient가 살아 있다.

- `c3_df_s0`: 기존 C3와 동일(detour 3), depth decoder만 수정
- `c4_df_d3_s0`: 기존 C4 d3와 동일, depth decoder만 수정
- `c4_df_d10_s0`: 기존 C4 d10과 동일, depth decoder만 수정
- 모두 fresh 2000 iter, 200마다 저장 및 Cross 평가

### C3/C4 conservative-risk 4-way (2026-09-04)

depth-fix 200 평가에서도 1.5m/s 미만 명령은 약 2.9%였고, 대표 Cross state에서
1.45m/s 미만 점은 세 모델 모두 0%였다. 감속 head는 살아났지만 collision 대비
extra-delay가 우세해 미세 감속만 선택했다. 해당 세 실행을 보존·종료하고 위험 거리와
collision 비중을 크게 늘린 네 진단 실험으로 교체한다.

- `c3_r1_s0`: clearance 1.5m, collision 150, extra-delay 0.2, detour 10
- `c3_r2_s0`: clearance 2.0m, collision 300, extra-delay 0.05, detour 20
- `c4_r1_s0`: proximity margin 0.25m/agent(기본 pair envelope 1.2m), collision 15,
  extra-delay 0.2, detour 10
- `c4_r2_s0`: proximity margin 0.50m/agent(기본 pair envelope 1.7m), collision 30,
  extra-delay 0.05, detour 20
- 모두 smooth-depth, fresh 2000 iter, 100마다 저장·Cross 평가

### C3b conflict-minimum profile (2026-09-04)

기존 C3는 감속구간을 교차점 전에 붙였지만 대칭 cosine이라 구간 중앙에서 최저속도를
찍고 교차점에서는 이미 1.5m/s로 복귀했다. 또한 학습된 길이가 교차점까지의 거의 전체
prefix가 되어 현재 root 바로 앞부터 감속하는 문제가 있었다. 기존 체크포인트 의미는
보존하고 `conflict_min_at_crossing` 옵션을 새로 둔다.

- 교차점 전: 모델이 예측한 `L_pre` 동안 1.5m/s에서 최저속도로 smoothstep 감속
- 교차점: 정확히 최저속도
- 교차점 후: `a_up=0.75m/s^2`로 1.5m/s까지 복귀하는 거리를 물리식으로 자동 결정
- 모델 출력은 기존처럼 agent별 `(L_pre, depth)` 두 값뿐이며 post 길이는 학습하지 않음
- `c3b_r1_s0`, `c3b_r2_s0`: 각각 기존 C3 r1/r2 loss 설정 그대로 fresh 2000 iter,
  100마다 저장·Cross 평가
- 기존 `c3_r1_s0`, `c3_r2_s0`은 500 PTH까지 보존 후 종료했고 C4 두 런은 유지함
### C5 pointwise risk-speed cap (실행 중)

- `c5_pointcaps_s0`: state-only joint MLP가 두 경로의 30개 waypoint residual과
  각 33점의 slowdown request를 동시에 출력한다.
- 각 request는 speed cap으로 변환된다. 역방향 braking envelope(`1.0m/s²`)와
  순방향 recovery envelope(`0.75m/s²`)가 위험점 앞 감속과 통과 후 복귀를
  자동 생성한다. 따라서 center/width 및 Cross 전용 교차점 입력이 없다.
- full-speed counterfactual의 time-aligned point risk는 위험 위치를 찾는
  학습 신호다. 실제 collision loss가 위험점 request를 키우고,
  `unnecessary_slow`는 `q*(1-stopgrad(risk))`로 안전한 위치의 request만
  제거한다. 물리적으로 파생된 braking tail 자체에는 이 벌점을 걸지 않는다.
- 랜덤 priority 한 명은 기존처럼 직선/평속으로 고정되고 반대 agent만 양보한다.
- 실행 스크립트: `scripts/coord/c5_pointcaps.sh`.

실행 갱신:

- 기존 `c4_r1_s0`, `c4_r2_s0` 학습과 watcher는 800 평가 후 종료했다.
- `c5_r1_s0`: point-risk localization `0.5`, detour-delay `10`
- `c5_r2_s0`: point-risk localization `2.0`, detour-delay `10`
- 두 C5 모두 fresh 2000 iter, 100마다 저장·Cross 평가한다. 나머지 조건은
  동일하여 안전한 위치의 감속 요청을 얼마나 강하게 제거할지만 비교한다.

### C6 crossing plateau (2026-09-04)

- 기존 C3b 실행은 체크포인트를 보존하고 종료한다.
- 경로 head는 유지하고 agent별 두 speed 값의 의미를 `(L_hold, depth)`로
  재정의한다. `[cross-L_hold, cross]` 전체를 동일한 최저속도로 유지한다.
- 최저속도 구간 앞의 감속과 교차점 뒤의 복귀는 `0.75m/s²` 물리 envelope로
  자동 생성한다. 따라서 speed-smooth loss는 사용하지 않는다.
- `c6_r1_s0`: extra-delay 0.5, `c6_r2_s0`: extra-delay 2.0. 나머지는
  proximity collision 30, margin 0.25m/agent, detour 10, consistency 1로 동일하다.
- 두 실험 모두 fresh 2000 iter, 100마다 PTH 저장 및 Cross 평가한다.

중간 판정(700/400): C5 r1/r2의 Cross `(success, collision, lower-command)`는 각각
700에서 `(0.677, 0.084, 0.105)`, `(0.723, 0.070, 0.053)`이다. r2의 collision은
600의 `0.133`에서 크게 흔들려 아직 재현된 개선으로 보지 않는다. r1은 감속을 더 쓰지만
task success를 잃는다. C6 r1/r2는 400에서 `(0.716, 0.077, 0.028)`,
`(0.716, 0.104, 0.029)`이며 학습 중 extra-delay도 약 `2e-4초`라 plateau depth가 사실상
0으로 붕괴했다. C6의 낮은 collision은 감속 성과가 아니라 우회/평가 변동이다.

원인 분리를 위해 학습은 그대로 두고 viewer에만 `MS_VIEW_TIMED_CROSS=1`을 추가했다.
평속 root→box→cross 시간에 measured speed와 pickup dwell을 적용해 도착을 동기화하며,
headless smoke의 예상 gap은 `0.000~0.001초`였다. 이 장면에서도 C5가 위험점에 request를
못 두면 다음 단순 수정 후보는 현재 leakage `q(1-r)`에 positive coverage `r(1-q)`를
추가하는 것이다. C6를 계속할 경우에는 구조가 아니라 nominal-conflict 조건의 arrival-gap
hinge를 추가해야 speed depth가 직접 학습된다.

### C7 joint-both general risk (2026-09-04)

C6의 Cross 전용 plateau는 500 PTH까지 보존하고 종료했다. 다음 주 실험은 C5의
state-only joint `(x,y,v)` MLP와 물리적 감속/복귀 envelope를 그대로 쓰되, episode
고정 priority를 제거하여 **두 agent의 예측 경로와 속도를 모두 executor에 적용**한다.

- 입력에 yield role, Cross 교차점, GT trajectory/speed는 추가하지 않는다.
- 안전항은 measured timing의 일반적인 time-aligned proximity risk(top-8)만 사용한다.
- `extra-delay`와 `detour-delay`를 같은 계수 `0.2`로 더한다. 두 항의 합은 각 agent가
  직선·평속으로 완료할 때보다 늘어난 총 시간이라 감속과 우회 중 한 방식을 지정하지 않는다.
- point-risk localization, speed efficiency, speed smooth, consistency, explicit window loss는
  모두 끈다. waypoint residual은 비현실적인 큰 우회만 막도록 `0.1`로 약하게 둔다.
- PPO 실제 충돌 계수는 기존 C2 기본값 `50`, auxiliary proximity 계수는 `30`이다.
- 태그 `c7_jointboth_s0`, Cross 64 env, fresh 2000 iter, 100마다 저장·별도 평가한다.
- 실행 스크립트: `scripts/coord/c7_jointboth.sh`.

C7은 100/200/300 Cross 평가에서 invalid가 `52.7/57.5/51.0%`였고, 이후 학습에서도
약 50% 이상을 유지했다. `path_residual=0.1`이 proximity 30에 비해 약해 양쪽 waypoint
head가 과도한 우회와 invalid curve로 도망간 것이 원인이다. 600 PTH까지 보존하고 종료했다.

다음 단일변수 진단 `c8_joint_risk_s0`은 C7과 완전히 동일하되 C5에서 안정성이 확인된
`path_residual=1.0`만 복원한다. 우선 300 iter, 100마다 저장·Cross 평가하며 invalid 2%
이하를 회복하는지 먼저 본다. 이 단계에서는 consistency/localization/새 모델을 추가하지 않는다.

C8도 6~12 iter부터 invalid가 `30~79%`로 폭증했다. 따라서 C7의 실패 원인을 residual
하나로 한정한 진단은 기각한다. C5의 낮은 invalid에는 `detour-delay=10`, consistency 1,
safe-point localization 0.5를 포함한 전체 안정화 조합이 기여한다.

`c9_jointboth_c5base_s0`은 C5-r1의 모델과 모든 loss 계수를 그대로 복원하고,
`random_priority=1→0`만 바꾼 진짜 단일변수 비교다. 300 iter, 100마다 Cross 평가하며,
여기서도 invalid가 커지면 양쪽 자유 경로 제어 자체를 제한해야 한다.

C9의 100/200/300 Cross `(success, collision, invalid, lower-command)`은 각각
`(0.715,0.114,0.023,0.178)`, `(0.608,0.113,0.010,0.337)`,
`(0.630,0.093,0.008,0.314)`다. 경로는 정상화됐지만 감속이 늘어도 실제 충돌 감소가
작고 task success만 떨어졌다. 300에서는 episode의 약 68%에서 두 agent가 각기 다른
시점에 모두 감속했지만, 같은 step의 동시 감속은 1.3%뿐이었다. 따라서 순간 동시 감속보다
한 joint plan 안에서 양보 책임이 양쪽으로 분산되는 것이 문제다.

C10은 yielder 정답이나 priority 입력 없이 agent별 추가 지연 `D0,D1`의 곱
`L_dual-delay = mean(D0*D1)`만 C9에 추가한다. 필요한 지연을 어느 한쪽에 모으면 0이고,
양쪽에 나누면 커지므로 state에 따라 모델 스스로 양보자를 고르게 한다. 계수 1과 5의
러프한 두 변형 `c10_d1_s0`, `c10_d5_s0`을 300 iter까지 비교한다.

운영 제한: coordinator GPU 학습은 동시에 최대 4개까지만 실행한다. 새 실험 시작 전
실제 `train_closed_loop` 프로세스 수를 확인하고, 4개가 차 있으면 종료/완료될 때까지 대기한다.
milestone watcher도 공용 `flock`을 사용해 평가를 한 번에 하나만 실행하며, 학습+평가
Python 프로세스가 이미 4개면 슬롯이 날 때까지 기다린다.

### C10 판정과 C11 learned persistent role (2026-09-04)

C10의 `dual_delay=D0*D1`은 양쪽에 지연을 나누는 것을 벌점으로 줬지만 역할 자체를
episode 동안 고정하지 못했다. 300 Cross 평가의
`(success, collision, invalid, lower-command)`은 `d1=(0.680,0.119,0.020,0.235)`,
`d5=(0.675,0.107,0.008,0.233)`로 C9 collision `0.093`보다 개선되지 않아 기각한다.

C11 `c11_learnedrole_s0`은 C5-r1의 경로·속도 구조와 loss를 그대로 두고 역할 결정만
바꾼다. 같은 state MLP latent에서 2-class priority logit을 추가하며, 첫 plan에서만
categorical PPO action으로 한 agent를 고른다. 선택된 priority agent는 episode 내내
직선·1.5m/s로 고정되고 반대 agent의 `(x,y,v)`만 적용된다. 이후 replan은 actual state를
반영하지만 역할은 다시 뽑지 않는다. role/scene/GT token은 입력하지 않고, 역할 log-prob도
첫 선택 step에서만 PPO ratio와 entropy에 포함한다.

- fresh 300 iter, 100마다 저장·Cross 평가
- smoke 2 iter와 C2 unit test 58개 통과
- 비교 기준은 C5-r2 1400의 success `0.719`, collision `0.033`, invalid `0.010`,
  lower-command `0.056`
- C11에서도 실제 collision이 높으면 다음 수정은 loss weight가 아니라 carried-box 실제
  위치/크기와 ms18 rollout 시간 오차를 future model에 반영하는 것이다.

### C11 판정과 C12 mechanism-neutral min-risk (2026-09-04)

C11 100/200/300의 `(success, collision)`은 각각 `(0.733,0.100)`,
`(0.695,0.091)`, `(0.646,0.109)`로 C5-r2 1400의 `(0.719,0.033)`보다 안전성이
크게 낮다. learned persistent role만 추가하는 가설은 기각한다.

현재 목표를 감속 우선에서 **충돌 없는 최소시간 trajectory**로 단순화한다. C12는
`future-risk 100 + extra-delay 1 + detour-delay 1 + path-residual 1 + consistency 1`만
사용한다. extra와 detour의 계수를 같게 두면 감속과 우회는 오직 완료시간 효과로 경쟁하며,
작은 path residual만 동률에서 덜 휘는 해를 선호한다. unnecessary-slow/speed-smooth 및
모든 explicit Cross loss는 끈다.

- `c12_minrisk_random_s0`: 검증된 episode random one-yielder
- `c12_minrisk_learned_s0`: C11 learned persistent yielder
- 두 실행은 role 선택 외에는 동일하며 fresh 300 iter, 100마다 Cross 평가한다.

### C12 판정과 C13 path guard (2026-09-04)

C12 300의 `(success, collision, invalid)`은 random role
`(0.664,0.063,0.238)`, learned role `(0.628,0.087,0.291)`이다. 위험 100에 비해
path residual 1이 너무 작아 독립 waypoint가 큰 우회·invalid로 도망갔다. learned role도
agent 한쪽으로 치우쳐 random보다 나빴다.

C13은 C12 random-role과 동일하고 path residual만 `10`, `30`으로 크게 바꾼다.
위험도와 감속/우회의 동일 시간비용은 유지하므로 특정 회피 방식은 지정하지 않는다.
300 iter에서 invalid 2% 이하와 collision 감소를 동시에 통과한 설정만 장기 학습한다.
