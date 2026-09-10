# TokenHSI-masteer 분석 기록

## 2026-09-10 — ms46 곡선 CLEAR + stack-first bootstrap 평가

### 목적과 판정 우선순위

- 대상: `ms46_ms18init_curveclear_stackfirst_1000_s0`
- 초기 정책: ms18 epoch 9000
- 학습: 1024 env, seed 0, 1000 iteration, epoch 9000→10000
- 고정 평가: initial, e9100, e9300, e9500, e9700, e10000을 각각 512 env로 평가
- 사용자 우선순위: base box를 조금 건드리는 것은 허용하고, 정밀 적재보다 먼저 실제
  end-to-end stack 성공을 만든다. 작은 base box 이동은 STACK에서 갱신되는 관측과
  current base pose를 따라가는 top target으로 보정할 수 있다고 본다.

따라서 이번 pilot의 일차 판정은 `STACK_FINAL_SUCCESS > 0`의 재현 여부다. pickup, PLACE,
base displacement와 foot contact는 최종 성공에 도달할 만큼 앞단 coverage가 있는지와
치명적 붕괴 여부를 확인하는 보조 지표다. 작은 접촉이나 displacement만으로 정책을
기각하지 않는다. 현재 controller도 `STACK_DROP_XY=0.35`라 base box가 원래 목표에서
0.35 m를 넘게 이동하기 전까지는 허용하고, STACK 중 top target은 매 step 현재 base
box pose를 따라 갱신한다.

ms46은 ms45에서 곡선 CLEAR 경로와 CLEAR 완료 뒤 base hold를 추가하고, top 접근 속도
scale 0.35, above-box 일회성 보상 +5, top hand-release shaping, 5-frame strict success와
성공 보상 +20을 넣었다. carry rehearsal은 0.65로 올렸다. 동시에 CLEAR retreat scale을
0.50→0.25, 거리 1.50→0.75 m, stall penalty를 1.50→0.75로 낮추고 grace를 5→20 frame으로
늘렸다. 따라서 ms45 대비 차이는 곡선 하나의 독립 효과가 아니다.

### PLACE 지표의 정확한 의미

표의 `PLACE`는 top box 적재도 아니고, base box의 엄격한 물리 안정 판정도 아니다.
비-rehearsal episode의 base agent가 다음 조건으로 CARRY→RELEASE transition을 기록한
비율이다.

1. 부모 carry metric의 `delivered`가 한 번 참이 됨: base box 중심과 base target의
   3-D 거리가 config의 `successThreshold=0.20 m` 이하.
2. 현재 base box의 높이 오차가 `STACK_Z_TOL=0.06 m` 이하.
3. 위 조건을 `STACK_ENTRY_STEPS=2` frame 연속 유지.

ms46은 `STACK_ENTRY_DELIVERED=1`, `STACK_ENTRY_LOWERED=1`,
`STACK_ENTRY_FOOT_GATE=0`이므로 코드에서 계산하는 현재 XY 0.10 m, 선속도 0.25 m/s,
각속도 1.0 rad/s, upright 10도, 발 이격 조건은 PLACE gate에 들어가지 않는다. 손 이격도
다음 RELEASE phase의 조건이라 PLACE에는 필요 없다. 또한 `delivered`는 최초 도달 뒤
episode 동안 latch되므로, 엄밀히는 최초 0.20 m 도달 이후 현재 z 조건을 2 frame 만족한
사건이다.

initial의 `PLACE=0.759`는 non-rehearsal base 440 episode 중 334건이 이 transition을
통과했다는 뜻이다. “334건이 안정적으로 놓였고 stack할 준비가 끝났다”는 뜻은 아니지만,
현재 목표인 stack-first bootstrap에 필요한 앞단 표본은 충분히 열려 있다는 뜻이다.

### 고정 checkpoint 평가

모든 평가는 동일한 ms46 sidecar와 74열 stage metric을 사용했고 `rc=0`이다. initial은
정확한 ms18 epoch 9000, 나머지는 해당 보존 checkpoint를 로드했다.

| checkpoint | base/top pickup | PLACE | RELEASE\|PLACE | CLEAR\|RELEASE | 전체 CLEAR | strict success |
|---|---:|---:|---:|---:|---:|---:|
| initial | 0.900 / 0.870 | 0.759 | 0.359 | 2/120 (0.017) | 2/440 (0.005) | 0/440 |
| e9100 | 0.860 / 0.832 | 0.724 | 0.891 | 13/304 (0.043) | 13/471 (0.028) | 0/471 |
| e9300 | 0.902 / 0.884 | 0.774 | 0.942 | 8/326 (0.025) | 8/447 (0.018) | 0/447 |
| e9500 | 0.883 / 0.866 | 0.760 | 0.935 | 5/329 (0.015) | 5/463 (0.011) | 0/463 |
| **e9700** | **0.874 / 0.874** | **0.770** | **0.978** | **20/354 (0.056)** | **20/470 (0.043)** | **0/470** |
| e10000 | 0.876 / 0.876 | 0.766 | 0.947 | 17/338 (0.050) | 17/466 (0.036) | 0/466 |

ms46 안에서는 e9700이 가장 좋은 pipeline checkpoint다. initial 대비 base pickup 하락은
2.6%p이고 top pickup은 0.4%p 증가했다. 둘 다 프로젝트의 학습 시드 변동 문턱 4.6%p보다
작으므로 pickup은 회귀 없이 보존된 것으로 본다. PLACE도 0.759→0.770으로 보존됐다.

`release_given_place=0.359→0.978`은 큰 변화라 RELEASE의 5-frame 손 이격 병목은
해결됐다고 판정한다. 반면 전체 CLEAR는 0.005→0.043, `clear_given_release`는
0.017→0.056에 그쳤고 strict final success는 전 checkpoint에서 0이다. 일반
`MS_EVAL_SUMMARY`의 `sr`는 carry harness 성공률이며 stack 성공으로 읽지 않는다.

### 실제 남은 병목

ms46은 `STACK_CLEAR_ARC_DIST=0.60`을 쓰므로 낮게 기록된 `clear_body`는 transition
hard gate가 아니다. 실제 CLEAR→STACK 조건은 양손 이격 유지와
`retreat_arc >= 0.60 m`다. e9700의 CLEAR 표본은 hand-clear step 비율 0.999였지만
retreat arc p50/p95가 0.437/0.600 m였다. 손은 잘 떼지만 대부분 곡선 경로를 문턱까지
완주하지 못한다.

e9700에서 STACK에 들어간 20건의 clear 시점 중앙값은 episode step 464였다. 600-step
episode에서 남은 시간은 중앙값 136 frame, 약 4.5초다. 그중 top carrier는 전부 pickup했고
85%가 2 m staging 목표에 도달한 이력이 있었다. 그러나 CLEAR 뒤 top은 staging 위치에서
stack 중심까지 약 2 m를 `STACK_TOP_SCALE=0.35`, 명목 속도 약 0.525 m/s로 이동해야 한다.
물리적으로 이동만 최소 약 114 frame이고, 이후 위치·속도·자세·양손 이격을 5 frame
유지해야 하므로 중앙 episode에는 여유가 거의 없다. 이는 직접 측정된 실패 gate가 아니라
설정과 event time에서 나온 시간-budget 추론이다.

학습 전체의 non-stationary metric에서는 non-rehearsal base 21,707 episode 중 CLEAR
2,198건, strict success 1건이었다. TensorBoard의 above bonus는 전체 1000 update 중
6회만 발생했고 strict success bonus는 iteration 약 232에서 한 번만 발생했다. 마지막
50 update에서는 `top_above=0`, 평균 STACK frame fraction은 약 0.00256이었다. 즉 실제
stack이 학습 중 한 번 발생하기는 했지만 반복되지 않았고, top phase 학습 신호가 너무
희소해 고정 평가로 일반화되지 않았다.

### ms45와의 관계 및 판정

ms45는 e9200/e9500에서 전체 CLEAR가 0.217/0.189까지 열렸지만 PLACE가 0.494/0.347로
무너졌고 strict success는 역시 0이었다. ms46은 그 반대로 PLACE 0.770과
`release_given_place=0.978`을 보존·회복했지만 전체 CLEAR가 0.043으로 좁아졌다.

따라서 판정은 다음과 같다.

- 성공: base/top pickup과 base PLACE 보존
- 성공: CARRY→RELEASE 뒤 손 떼기
- 실패: 곡선 CLEAR 0.60 m의 충분한 완주율 확보
- 실패: 재현 가능한 end-to-end stack 생성
- 현재 최선 진단 checkpoint: e9700
- 최종 채택: 하지 않음

사용자 우선순위에 따르면 다음 단계에서 base box의 작은 접촉을 더 줄이는 것은 핵심이
아니다. 먼저 CLEAR와 STACK을 직접 시작 상태로 샘플링하는 phase curriculum으로 top
접근·놓기 표본을 충분히 만들거나, 최소한 CLEAR 속도/episode horizon을 분리 진단해
끝단 성공을 재현해야 한다. 현재의 “stack-first” wrapper는 이름과 달리 모든 non-rehearsal
episode를 CARRY에서 시작하므로 STACK 자체를 bootstrap하지 않는다. 다음 계측에서는 top의
XY/Z, 선속도, 각속도, upright, hand-clear, base-shift gate를 각각 기록해야 최종 5-frame
success에서 무엇이 막혔는지 분리할 수 있다.

평가 원시는 다음 파일이다.

- `runs/results/masteer/eval_ms46_ms18init_curveclear_stackfirst_1000_s0__stage_einitial.npy`
- `runs/results/masteer/eval_ms46_ms18init_curveclear_stackfirst_1000_s0__stage_e9100.npy`
- `runs/results/masteer/eval_ms46_ms18init_curveclear_stackfirst_1000_s0__stage_e9300.npy`
- `runs/results/masteer/eval_ms46_ms18init_curveclear_stackfirst_1000_s0__stage_e9500.npy`
- `runs/results/masteer/eval_ms46_ms18init_curveclear_stackfirst_1000_s0__stage_e9700.npy`
- `runs/results/masteer/eval_ms46_ms18init_curveclear_stackfirst_1000_s0__stage_e10000.npy`

## 2026-09-08 — ms37 epoch 12000: RELEASE 해결, CLEAR 후측방 실행 실패

### 대상과 비교 조건

- 대상: `ms37_ms18init_seqrewardmask_side135_s0_try2`
- 초기 정책: ms18 epoch 9000을 ms37 sidecar로 평가한 `stage_einitial`
- 최종 정책: ms37 epoch 12000의 `stage_e12000`
- 과거 기준: ms24의 `possteer025 try5`, `possteer050 try3` 최신 수동 gate 진단
- 모두 512 env의 고정 체크포인트 평가 결과를 사용했다.

ms37 initial과 epoch 12000은 같은 sidecar를 사용하므로 학습 전후의 직접 비교다. 반면
ms24와 ms37은 phase entry 조건과 rehearsal 비율이 다르므로, `place` 같은 절대값은 정책
성능만이 아니라 controller gate 차이도 포함한다. ms24 비교에서는 pickup 보존,
`release_given_place`, RELEASE 이후 실제 retreat arc를 주로 본다.

### 핵심 결과

| 정책 | base pickup | place | release/place | clear/release | retreat arc p50/p95 | arc >= 0.6 m |
|---|---:|---:|---:|---:|---:|---:|
| ms24 possteer .25 | 0.899 | 0.049 | 0.983 | 0.018* | 0.020 / 0.082 m | 0/57 |
| ms24 possteer .50 | 0.915 | 0.039 | 0.956 | 0.023* | 0.003 / 0.116 m | 0/43 |
| ms37 initial | 0.881 | 0.746 | 0.409 | 0.003 | 0.011 / 0.167 m | 1/324 |
| **ms37 epoch 12000** | **0.896** | **0.800** | **0.980** | **0.003** | **0.083 / 0.192 m** | **2/731** |

`*` ms24의 clear event는 `arc >= 0.6 m` 성공이 아니다. 당시 CLEAR 완료 조건이
body-distance였기 때문에 RELEASE와 같은 frame에 찍힌 false positive가 포함된다. 최신
수동 평가에서는 ms24 두 정책 모두 실제 `arc >= 0.6 m`가 0건이었다.

ms37에서 pickup은 0.881에서 0.896으로 유지됐다. 프로젝트의 신호 문턱 0.05보다 작은
차이이므로 pickup 개선으로 주장하지 않고 **회귀 없이 보존**된 것으로 판정한다.

가장 큰 변화는 `release_given_place=0.409 -> 0.980`이다. sequential reward mask가
배치 후 손 떼기 병목을 확실히 해결했다. 반면 `clear_given_release`는 0.003으로 그대로다.
retreat arc는 다음처럼 짧은 움직임과 실제 후퇴 사이에서 급격히 끊긴다.

| 조건 | ms37 initial | ms37 epoch 12000 |
|---|---:|---:|
| arc >= 0.1 m / RELEASE 진입 | 0.167 | 0.430 |
| arc >= 0.2 m / RELEASE 진입 | 0.037 | 0.045 |
| arc >= 0.3 m / RELEASE 진입 | 0.015 | 0.011 |
| arc >= 0.6 m / RELEASE 진입 | 0.003 | 0.003 |

즉 0.1 m 정도의 자세 전환이나 첫 움직임은 늘었지만 이를 보행으로 연결하지 못했다.
CLEAR 평균 체류시간도 RELEASE 진입 episode당 약 132 step에서 263 step으로 늘었다.
정책은 손을 빨리 떼고 CLEAR에 일찍 진입한 뒤 남은 episode 대부분을 정지 상태로 보낸다.
viewer에서 보인 “손을 떼고 경로는 생기지만 뒤나 옆으로 가지 않음”과 정량 결과가
일치한다.

### ms24와의 관계

ms24는 native carry와 pickup은 잘 보존했지만 순차 task의 첫 gate를 거의 통과하지 못했다.

- ms24 possteer .25: place 58/1189, release 57/1189
- ms24 possteer .50: place 45/1166, release 43/1166
- place에 들어간 뒤 손 떼기는 0.956~0.983으로 높았다.
- 그러나 실제 0.6 m retreat는 두 정책 모두 0건이었다.

따라서 ms24의 첫 병목은 CARRY의 안정 배치 결합 gate였고, 그 뒤에는 현재와 같은 CLEAR
실행 병목이 이미 숨어 있었다. ms37은 entry를 `delivered + lowered` 2-frame으로 바꿔
place를 0.800까지 열고 sequential reward mask로 손 떼기까지 해결하여, 잠재되어 있던
CLEAR 병목을 주 병목으로 드러냈다.

전체 end-to-end 지표는 ms24가 아직 더 좋다.

| 지표 | ms24 .25/.50 | ms37 epoch 12000 |
|---|---:|---:|
| 평가 success rate | 0.840 / 0.858 | 0.772 |
| finish rate | 0.785 / 0.817 | 0.704 |
| root lateral error | 0.124 / 0.129 m | 0.160 m |
| lateral error > 0.5 m | 0.011 / 0.021 | 0.194 |

다만 ms24는 sequence phase에 거의 진입하지 않았고 ms37은 CLEAR에 오래 머무르므로 이
전체 지표 차이를 순수 carry 회귀로 해석할 수는 없다. base pickup이 0.896으로 보존된 점과,
CLEAR 노출이 늘면서 speed/path 지표가 악화된 점을 함께 봐야 한다.

### 이전 시도에서 병목이 이동한 과정

1. **ms30/ms31 — hard conjunction gate 병목**
   - place는 각각 11/1132, 13/1140뿐이었다.
   - 안정 배치와 네 발 이격이 같은 순간 성립해야 했고 CARRY joint gate 통과율은 약
     0.00014~0.00016이었다.
   - RELEASE에서도 hand와 foot을 동시에 요구해 두 번째 hard gate가 생겼다.
   - 이때의 낮은 phase 진입은 정책 실행보다 gate 설계가 원인이었다.

2. **ms32/ms33 — CLEAR 학습 노출 부족**
   - foot hard gate를 제거하고 arc 기반 CLEAR를 도입했지만 안정 배치 5-frame gate가
     남았다.
   - place는 0.039~0.042, 실제 clear는 0건이었다.
   - CLEAR를 경험하는 episode가 너무 적어 reward-only pilot이 후퇴를 학습할 충분한
     데이터를 만들지 못했다.

3. **ms34 — delivered entry 방향은 맞지만 실제 sidecar가 5-frame**
   - legacy delivered event를 entry 조건으로 쓰기 시작했다.
   - wrapper가 요청한 1-frame과 달리 저장 sidecar에는 `STACK_ENTRY_STEPS=5`가 남았다.
   - 직접 비교할 고정 체크포인트 평가가 없어 정량 결론은 제한한다.

4. **ms35 — entry 해결 후 carry-reward bridge가 손 떼기를 방해**
   - delivered+lowered entry로 place가 초기 0.730, epoch 9400에서 0.696까지 열렸다.
   - 하지만 `release_given_place`는 0.391에서 0.216으로 감소했다.
   - RELEASE 시작에서 live carry reward를 보간하자 손이 붙은 상태의 보상이 보존되어,
     정책이 손을 계속 붙잡는 쪽으로 학습됐다.
   - retreat arc p95도 0.116 m에서 0.059 m로 감소했다.

5. **ms36 — sequential reward mask 도입**
   - live carry reward 대신 phase 완료 reward를 상수로 넘겨 손 접촉 유인을 제거했다.
   - checkpoint 생성 전에 중단되어 독립적인 고정 정책 평가는 없다.

6. **ms37 — RELEASE 해결 후 CLEAR 실행이 최종 병목**
   - sequential reward mask로 손 떼기는 해결했다.
   - 후측방 경로 생성과 phase 전환도 정상이다.
   - 실제 policy action은 0.1 m 안팎의 초기 움직임 뒤 정지한다.

### 현재 병목 원인

#### 측정으로 확인된 것

1. **controller/path 생성 오류가 아니다.**
   손 이격 5-frame 뒤 CLEAR로 넘어가고 후측방 경로가 viewer에 나타난다. arc도 0에서
   p50 0.083 m까지 증가하며, 드물지만 최대 0.614 m까지 진행했다.

2. **body, foot, stability hard gate가 phase를 막는 것이 아니다.**
   ms37은 `STACK_CLEAR_HARD_GATE=0`, `STACK_RELEASE_FOOT_GATE=0`이다. CLEAR 완료의
   실질 조건은 손 이격 유지와 `arc_root >= 0.60 m`이고, 실패한 값은 arc다.

3. **정책은 짧은 자세 변화만 학습했다.**
   arc 0.1 m 비율은 크게 증가했지만 0.2 m 이상은 개선되지 않았다. 경로 신호를 완전히
   못 받는 것이 아니라, 첫 동작을 지속 보행으로 연결하지 못한다.

4. **안전 보호 기준도 통과하지 못했다.**

   ```text
   protect=FAIL
   pickup=0.896/0.850
   release_given_place=0.980/0.800
   base_disp_p95=0.213/0.150
   foot_clear=0.256 baseline=0.267
   retreat_arc0.60=0.002 baseline=0.001
   ```

   `learning=UP`은 0.6 m 성공이 1건에서 2건으로 늘었다는 기계적 판정일 뿐이다. 절대
   차이는 0.001이며 프로젝트의 0.05 신호 문턱보다 훨씬 작아 학습 증거가 아니다.

#### 가장 가능성 높은 해석

1. **CLEAR 정지 local optimum**
   sequential mask에서 CLEAR는 매 step `carry_done 1.6 + release_done 1.0 = 2.6`의
   상수 reward를 받는다. motion reward는 손 이격을 유지하면서 실제로 움직여야 생긴다.
   최종 정책도 CLEAR step 중 hand-clear 비율이 0.567이라 거의 절반의 step에서 motion
   reward가 차단된다. 안정하게 서 있으면 큰 기본 reward를 계속 얻지만, 후측방 보행을
   시작하면 넘어짐과 박스 교란 위험이 생겨 정지가 안전한 해가 된다.

2. **CLEAR 전용 표현과 학습 자유도 부족**
   ms37에서 trainable actor 파라미터는 `internal_adapt_mlp` 4개뿐이다. task encoder,
   transformer, composer는 모두 동결됐다. RELEASE/CLEAR 전용 phase token도 없고 정책은
   steering window가 정지에서 새 경로로 바뀌는 것만 보고 mode를 구분한다. 하나의 shared
   adapter가 기존 pickup/carry를 보존하면서 손 떼기와 object-free 135도 보행까지 새로
   만들어야 한다. 손 떼기는 학습했지만 질적으로 새로운 보행 mode까지 만들기에는 조건
   분리 또는 용량이 부족한 것으로 해석된다.

3. **유효한 CLEAR motion reward가 희소하다.**
   positive steering reward는 `hand_clear * path_quality * motion`이고, ms20 CLEAR 항도
   hand-clear와 motion으로 gate된다. 정지 상태에서는 signed progress와 speed reward가
   거의 0이며, exploration으로 첫 걸음을 내딛고 손 이격·박스 안정까지 유지한 sample만
   informative reward를 받는다. 반면 상수 완료 reward는 모든 CLEAR step에 주어진다.

### 판정

**ms37은 부분 성공이지만 전체 sequential stacking 정책으로는 채택하지 않는다.**

- 보존: pickup 통과
- 성공: CARRY→RELEASE entry 및 손 떼기
- 실패: CLEAR에서 0.6 m 후측방 이동
- 추가 실패: RELEASE 이후 base box displacement p95 보호 기준 초과

현재 우선순위는 gate를 더 완화하는 것이 아니다. CLEAR의 정지 보상 구조를 제거하거나
완료 reward를 potential/transition 형태로 바꾸고, CLEAR 전용 phase conditioning 또는
phase-specific residual을 통해 기존 carry와 후퇴 보행의 학습 경로를 분리해야 한다.

### 추가 진단: +500 iteration 체크포인트

`+500 iteration`은 절대 epoch 9500이다. 이 체크포인트를 평가하면 최종 실패를 뒤집는
용도가 아니라 **손 떼기 학습과 CLEAR 정지 수렴이 언제 발생했는지** 구분할 수 있다.

```bash
bash scripts/masteer/stack_stage_eval.sh \
  ms37_ms18init_seqrewardmask_side135_s0_try2 7 9500 512
```

- epoch 9500에서 이미 release가 높고 arc가 낮으면 정지 local optimum이 초기에 형성됐다.
- epoch 9500에서 release도 낮으면, 뒤쪽 학습이 대부분 손 떼기에 소비되어 CLEAR motion을
  학습할 시간이 부족했던 것이다.
- `500 env` 평가를 뜻한다면 별도 이득이 없으며 기존 비교 규약인 512 env를 유지한다.

## 2026-09-08 — ms38 dynamic carry mask와 virtual-retreat zero-shot 진단

### 1. 실험 목적과 설정

- tag: `ms38_ms18init_seqrewardmask_side135_dynmask_s0`
- 초기 체크포인트: ms18 epoch 9000
- 학습 구간: epoch 9000 → 12000, seed 0, 1024 env, GPU 7
- 주요 설정:
  - ms37의 sequential reward mask와 135° 후퇴 controller 유지
  - `STACK_DYNAMIC_CARRY_MASK=1`
  - CLEAR부터 SUCCESS까지 `new_carry`, `old_carry` token과 두 carry observation window를 모두 mask/zero 처리
  - `use_prior_knowledge=False`라서 `old_carry`는 원래도 비활성
  - 따라서 CLEAR의 실질 활성 token은 `[weight, self, steer]`이고 teammate token도 mask 상태
  - `MA_ADAPTER_ONLY=1`, `MA_FREEZE_NEW_CARRY=1`
  - actor에서 실제 gradient가 발생한 것은 `internal_adapt_mlp` 4/4 parameter뿐이며 tokenizer, transformer, composer는 freeze
  - 25% carry rehearsal, learning rate `2e-5`, horizon 32
  - sequential done reward `1.6 / 1.0 / 1.5`
  - CLEAR arc gate `0.60 m`, retreat path `1.50 m`, retreat scale `0.50`, side angle `45°`
  - CLEAR hard gate와 release foot gate는 비활성
- observation 340차원과 모델 구조는 그대로이며 token/layer를 새로 추가하지 않았다.
- 설정 근거:
  - `TokenHSI-masteer/CHANGELOG.md`의 ms38 항목
  - `runs/queue/logs/ms38_ms18init_seqrewardmask_side135_dynmask_s0.env`

### 2. 평가 해석 시 주의점

- 이 실험의 `.npy`는 stack 전용 73열 schema다.
- 일반 `MS_EVAL_SUMMARY`가 출력하는 `graspEp` 이후 lifecycle 항목은 이 schema를 잘못 해석한다. 예를 들어 episode 수를 넘는 `delivered` 값은 유효한 단계 통계가 아니다.
- 단계 병목은 `scripts/masteer/stack_stage_summary.py` 결과를 기준으로 판단한다.
- 일반 summary의 carry SR은 참고용이며 sequential stack 전체 성공률로 해석하면 안 된다.

### 3. ms38 epoch 9500 평가

- checkpoint: `Humanoid_00009500.pth`
- result: `runs/results/masteer/eval_ms38_ms18init_seqrewardmask_side135_dynmask_s0__stage_e9500.npy`
- log: `runs/queue/logs/eval_ms38_ms18init_seqrewardmask_side135_dynmask_s0__stage_e9500.log`
- 정상 종료 `rc=0`, 정확한 checkpoint load 확인
- 일반 carry summary: `sr=0.7339`, `eps=2816`, `fin=0.6275`, `lat=0.166`, `spd_err=0.552`

base cohort `n=1086`의 단계 결과:

| 단계 | episode | 전체 대비 |
|---|---:|---:|
| near | 971 | 89.4% |
| pick | 945 | 87.0% |
| place | 795 | 73.2% |
| release | 769 | 70.8% |
| CLEAR 통과 / STACK 진입 | 47 | 4.3% |
| physical fail | 135 | 12.4% |

조건부 전환율:

- `pick | near`: 97.3%
- `place | pick`: 84.0%
- `release | place`: 96.7%
- `clear | release`: 6.11% (`47 / 769`)

첫 정지 단계:

- approach 115 (10.6%)
- lift 26 (2.4%)
- carry/place 150 (13.8%)
- release 26 (2.4%)
- CLEAR stuck 722 (66.5%)
- STACK 진입 47 (4.3%)

CLEAR 세부:

- retreat arc median `0.201 m`, p90 `0.477 m`, p95 `0.600 m`
- `arc >= 0.5 m`: 9.1%
- `arc >= 0.599 m`: 6.11%이며 CLEAR를 통과한 47 episode와 일치
- released episode는 모두 hand-clear step을 한 번 이상 만들었지만, CLEAR 전체 step 중 hand-clear 비율은 36.9%
- CLEAR stuck 722개 중 stack-fail flag 없음 597개, fail 125개
- released cohort fail rate 16.25%
- base displacement median `0.0618 m`, p95 `0.3473 m`
- 역할별 `clear | release`: agent0 6.9%, agent1 5.4%

동일 epoch의 ms37과 비교하면 dynamic carry mask가 CLEAR 동작 자체는 개선했다.

- retreat arc median: `0.051 → 0.201 m`
- `clear | release`: `0.42% → 6.11%`
- 다만 place 비율은 `80.0% → 73.2%`로 낮아졌고 base displacement/failure도 악화했다.
- 결론: epoch 9500의 지배적 병목은 release 이후 `0.60 m` CLEAR 후퇴다.

### 4. ms38 epoch 10400 평가와 plateau 확인

- checkpoint: `/home/hwanhee/koo_cvpr/TokenHSI-masteer/output/masteer/ms38_ms18init_seqrewardmask_side135_dynmask_s0/Humanoid_08-11-34-33/nn/Humanoid_00010400.pth`
- result: `runs/results/masteer/eval_ms38_ms18init_seqrewardmask_side135_dynmask_s0__stage_e10400.npy`
- log: `runs/queue/logs/eval_ms38_ms18init_seqrewardmask_side135_dynmask_s0__stage_e10400.log`
- 512 env, GPU 7, 정상 종료 `rc=0`, 정확한 checkpoint load 확인
- 일반 carry summary: `sr=0.7446`, `eps=2750`, `fin=0.6458`, `lat=0.166`, `spd_err=0.556`

base cohort `n=1056`의 단계 결과:

| 단계 | episode | 전체 대비 |
|---|---:|---:|
| near | 940 | 89.0% |
| pick | 920 | 87.1% |
| place | 791 | 74.9% |
| release | 770 | 72.9% |
| CLEAR 통과 / STACK 진입 | 36 | 3.4% |
| physical fail | 122 | 11.6% |

조건부 전환율:

- `pick | near`: 97.87%
- `place | pick`: 85.87%
- `release | place`: 97.35%
- `clear | release`: 4.68% (`36 / 770`)

첫 정지 단계:

- approach 11.0%
- lift 1.9%
- carry/place 12.3%
- release 2.0%
- CLEAR stuck 69.5%
- STACK 진입 3.4%

CLEAR 세부:

- retreat arc median `0.200 m`, p90 `0.476 m`, p95 `0.591 m`
- `arc >= 0.5 m`: 9.35%
- `arc >= 0.599 m`: 4.94%
- released cohort fail rate 14.42%
- CLEAR stuck 중 stack-fail flag 없음 623개, fail 111개
- base displacement median `0.0583 m`, p95 `0.3007 m`
- 역할별 `clear | release`: agent0 `7 / 378 = 1.9%`, agent1 `29 / 392 = 7.4%`
- CLEAR step 비율: hand-clear 47.4%, foot-clear 53.5%, stable 59.0%
- CLEAR 진입 후 통과 지연 median 127 frame

epoch 9500 → 10400에서 일반 carry SR은 약 1.1%p 올랐지만, 핵심 CLEAR 지표는 개선되지 않았다.

- retreat arc median: `0.201 → 0.200 m`
- `clear | release`: `6.11% → 4.68%`
- 결론: 학습 초반이라서 생긴 일시적 문제라기보다 현재 표현/최적화 조건의 plateau다. 같은 설정으로 iteration만 늘리는 것으로 해결될 가능성은 낮다.

### 5. dynamic double-mask 실패 원인

다음 가능성들은 평가로 배제하거나 우선순위를 낮췄다.

- 표본 부족이 아니다. epoch 10400에서 released episode 770개, CLEAR step 166,610개가 관찰됐다.
- gradient가 완전히 끊긴 것도 아니다. adapter 4/4 parameter에 nonzero gradient가 있었다.
- 속도 명령 자체가 불가능한 범위도 아니다. 목표 속도는 약 `0.75 m/s`이고 기존 velocity curve가 약 `0.72 m/s`까지 낸다.

가장 가능성 높은 원인은 다음 조합이다.

1. `new_carry`와 `old_carry`를 동시에 mask하면 locomotion/carry motion basis까지 제거된다.
2. steering token은 원래 `new_carry`와 함께 조건으로 사용됐으며, steering만으로 보행을 생성하도록 학습된 token이 아니다.
3. frozen transformer/composer 입장에서는 CLEAR의 `[weight, self, steer]` subset이 학습 때 보지 못한 조합이다.
4. 모든 phase가 하나의 작은 shared adapter만 갱신하므로 CLEAR 후퇴 gradient와 기존 pickup/carry 보존 gradient가 충돌한다.
5. CLEAR motion reward가 현재 step의 `clear_now`로 gate되어 epoch 10400 기준 47.4% step에서만 활성화된다. 반면 sequential constant reward는 계속 남고 `0.35 * clear_score`도 부분 진행 상태를 보상하므로 약 `0.2 m`에서 머무는 정책이 생길 수 있다.

### 6. carry token 선택에 대한 판단

- 실제 box observation을 넣은 채 frozen `new_carry`를 활성화하면 carry token은 박스를 유지하려 하고 steering은 박스에서 멀어지게 하므로 의미 충돌이 생긴다.
- `new_carry`를 mask하면 위에서 확인한 OOD token subset과 motion basis 제거 문제가 생긴다.
- `old_carry`도 실제 box를 입력하면 같은 충돌 가능성이 있다. 또한 `use_prior_knowledge=False`였으므로 `old_carry + steering` 조합 역시 본 조합이 아니다.
- `new_carry` unfreeze는 정상 pickup/carry 표현을 훼손할 위험이 있다. 이전 ms27에서도 tokenizer unfreeze 뒤 pickup 붕괴가 관찰됐다.

따라서 모델 구조를 바꾸지 않는 가장 보수적인 해법은 학습된 `new_carry + steering` 조합을 유지하되, CLEAR에서는 실제 박스가 아니라 손 위치에 놓인 가상 박스와 후퇴 endpoint를 carry observation으로 제공하는 것이다. 코드의 `STACK_VIRTUAL_RETREAT_BOX=1`이 이 동작을 이미 지원한다.

- virtual box position: 두 손 위치의 평균
- virtual box velocity: root velocity
- virtual target: retreat endpoint
- 물리 actor나 collision/contact를 추가하는 것이 아니라 observation만 치환한다.
- 적용 구간: CLEAR부터 SUCCESS까지

### 7. epoch 10400 virtual-retreat zero-shot probe

동일한 epoch 10400 weight에 학습 없이 다음 두 값만 바꿨다.

```bash
STACK_DYNAMIC_CARRY_MASK=0
STACK_VIRTUAL_RETREAT_BOX=1
```

- result: `runs/results/masteer/eval_ms38_ms18init_seqrewardmask_side135_dynmask_s0__stage_e10400_virtualprobe.npy`
- log: `runs/queue/logs/eval_ms38_ms18init_seqrewardmask_side135_dynmask_s0__stage_e10400_virtualprobe.log`
- log에서 `virtual_retreat=1`, 정확한 epoch 10400 checkpoint load, dynamic carry-mask 미적용, `rc=0` 확인

base cohort `n=1537`의 결과:

| 지표 | dynamic mask e10400 | virtual-retreat e10400 |
|---|---:|---:|
| place / base | 74.9% | 54.7% (`841`) |
| release / base | 72.9% | 53.2% (`817`) |
| `release \| place` | 97.35% | 97.1% |
| `clear \| release` | 4.68% | 65.6% (`536 / 817`) |
| STACK 진입 / base | 3.4% | 34.9% (`536 / 1537`) |
| retreat arc median | 0.200 m | 0.609 m |
| retreat arc p95 | 0.591 m | 0.645 m |
| CLEAR 통과 지연 median | 127 frame | 38 frame |
| CLEAR hand-clear step | 47.4% | 91.6% |
| base displacement p95 | 0.301 m | 0.297 m |

virtual probe의 추가 수치:

- first stop: approach 21.6%, lift 2.7%, carry/place 21.2%, release 1.6%, CLEAR 18.3%, STACK 진입 34.9%
- physical fail / base: 11.3%
- released cohort fail rate: 약 20.4%
- base displacement mean `0.0763 m`, median `0.028 m`, p95 `0.2972 m`
- CLEAR step: hand-clear 91.6%, foot-clear 43.9%, stable 32.1%
- root-distance median `0.870 m`, p95 `1.035 m`
- 역할별 `clear | release`: agent0 61.2%, agent1 70.0%

핵심 해석:

- `clear | release`가 `4.68% → 65.6%`, STACK 진입이 `3.4% → 34.9%`로 증가했다.
- retreat arc median도 threshold를 넘는 `0.609 m`까지 올라갔다.
- 따라서 dynamic double-mask가 CLEAR 실행 병목의 직접 원인이었고, virtual carry observation은 학습 없이도 이를 대부분 해소했다.
- 현재 단계에서는 `new_carry` unfreeze나 `old_carry` 활성화를 먼저 시도할 근거가 없다.
- 동일 weight/후단 observation 변경인데 pre-CLEAR 비율도 달라진 것은 비동기 reset과 후단 trajectory 분기로 평가 episode 표본 구성이 달라졌기 때문일 가능성이 높다. matched evaluation 없이 upstream의 인과적 회귀로 단정하지 않는다.
- 이 73열 결과에는 최종 STACK 성공이 별도 보존되지 않으므로, CLEAR 통과를 전체 작업 성공으로 해석하지 않는다. 최종 end-to-end 결론 전에는 STACK 완료 지표를 추가로 기록해야 한다.

### 8. viewer 관찰과 남은 병목

viewer에서 virtual-retreat를 켜면 CLEAR 후 실제로 뒤로 이동하지만, 뒤로 이동한 다음 힘없이 넘어지는 현상이 관찰됐다. 이는 virtual observation이 후퇴 명령을 만들지 못하는 문제가 아니라, CLEAR 통과 이후 controller target이 이어지는 문제로 보는 것이 맞다.

현재 값은 다음처럼 서로 어긋난다.

- CLEAR 통과 gate: retreat arc `0.60 m`
- retreat path endpoint: `1.50 m`

CLEAR → STACK 전환 때 top agent만 새 target을 받고 base의 retreat path/scale은 즉시 정지하지 않는다. 따라서 base는 CLEAR gate를 넘은 뒤에도 남은 약 `0.90 m`를 STACK 중 계속 후퇴하려 하며, 낮아진 foot-clear/stable 비율과 함께 넘어질 수 있다.

### 9. 현재 채택 방향과 다음 검증

모델 구조는 변경하지 않고 다음 표현을 우선 채택한다.

```bash
STACK_DYNAMIC_CARRY_MASK=0
STACK_VIRTUAL_RETREAT_BOX=1
```

- `new_carry`와 steering은 활성 상태로 두고 둘 다 freeze한다.
- 실제 박스 대신 손에 붙은 virtual box observation을 사용한다.
- `old_carry` 활성화와 `new_carry` unfreeze는 보류한다.
- 이 결정은 아직 학습 기본값 변경이 아니라 평가와 후속 실험 방향이다.

먼저 모델/reward 수정 없이 viewer에서 후퇴 거리만 맞춘다.

```bash
STACK_RETREAT_DIST=0.75
STACK_CLEAR_ARC_DIST=0.60
STACK_RETREAT_SCALE=0.50
```

- 여전히 넘어지면 `STACK_RETREAT_SCALE=0.25`를 시험한다. 목표 속도는 약 `0.375 m/s`로 ms18 학습 속도 범위 안이다.
- 그래도 넘어지면 CLEAR → STACK 전환 순간 base/root의 현재 위치로 virtual goal과 steering path를 retarget하여 후퇴를 정지시키는 controller-level hold를 적용한다.
- 이 hold도 token/layer 추가나 모델 구조 변경 없이 구현할 수 있다.
## 2026-09-09 — ms41 500/3000 iteration: CLEAR 후퇴 학습과 base carry 붕괴

### 대상과 평가 무결성

- tag: `ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_3000_s0`
- 초기 체크포인트: ms18 epoch 9000
- 비교 체크포인트:
  - `stage_e9500`: +500 iteration, `Humanoid_00009500.pth`
  - `stage_elatest`: +3000 iteration 종료 뒤 최종 `Humanoid.pth`
- 둘 다 512 env, seed 0, 동일 sidecar와 동일 stage 평가 경로를 사용했다.
- 두 로그 모두 의도한 체크포인트를 로드했고 `rc=0`으로 끝났다.
- carry harness 성공률은 망가진 0회차를 제외하고 `ObjectSet_test_1/2`를 평균한 값이다.
- `stage_elatest`는 보존된 `Humanoid_00012000.pth`가 아니라 그 직후 마지막으로 다시
  저장된 `Humanoid.pth`다. 학습 종료 시점의 최종 정책이라는 의미에서 +3000 iteration
  결과로 본다.

평가 로그:

- `runs/queue/logs/eval_ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_3000_s0__stage_e9500.log`
- `runs/queue/logs/eval_ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_3000_s0__stage_elatest.log`

원시 stage metrics:

- `runs/results/masteer/eval_ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_3000_s0__stage_e9500.npy`
- `runs/results/masteer/eval_ms41_ms18init_negclear_zeroobs_nomask_steeradapt_foot010_3000_s0__stage_elatest.npy`

두 로그에는 motion loader의 `Error for key=` 메시지가 각각 630회씩 동일하게 있으며
traceback은 없다. 두 체크포인트 비교를 다르게 오염시키는 정황은 아니지만,
`PLAN_masteer.md`의 엄격한 `warn=0` 조건을 문자 그대로 만족하는 로그라고 기록하지는 않는다.

### 전체 평가 결과

| 지표 | epoch 9500 (+500) | latest (+3000) | 변화 |
|---|---:|---:|---:|
| carry harness `sr` | 0.5356 | 0.5522 | +0.0166 |
| 개별 finish `fin` | 0.4334 | 0.4890 | +0.0556 |
| 두 agent 모두 finish `both` | 0.2539 | 0.2422 | -0.0117 |
| 한 agent만 finish `one` | 0.4258 | 0.5449 | +0.1191 |
| root lateral error | 0.201 m | 0.205 m | +0.004 m |
| lateral error > 0.5 m | 0.317 | 0.355 | +0.038 |
| speed error | 0.650 | 0.713 | +0.063 |
| real speed | 0.534 | 0.446 | -0.088 |
| still steps | 103.55 | 154.18 | +50.63 |
| moving gait speed | 0.981 | 1.041 | +0.060 |
| collision episode | 0.3138 | 0.3307 | +0.0169 |

`sr` 증가는 1.66%p로 프로젝트 신호 문턱 `0.05`보다 작다. `fin`은 5.56%p 올랐지만
두 agent가 모두 끝낸 비율은 오히려 1.17%p 감소했고 한 agent만 끝낸 비율이 11.91%p
증가했다. 따라서 개별 finish 증가는 순차 stacking이나 pair-level 조율 개선이 아니다.

경로 lateral error는 사실상 같지만 speed error는 0.063 악화됐다. 실제 평균 속도는
줄고 moving gait는 빨라졌으므로, 느린 걸음으로 명령을 따른 것이 아니라 정지 frame이
늘어난 영향이 크다. 속도 곡선 기울기도 `0.590 → 0.576`으로 개선되지 않았다.

### 역할별 생애주기

| 역할/지표 | epoch 9500 | latest | 변화 |
|---|---:|---:|---:|
| rehearsal pickup | 0.775 | 0.858 | +0.083 |
| rehearsal delivered | 0.329 | 0.561 | +0.232 |
| base pickup | 0.832 | 0.869 | +0.037 |
| base grasp break | 0.086 | 0.547 | **+0.461** |
| base delivered / PLACE 진입 | 0.384 / 0.375 | 0.121 / 0.118 | **-0.263 / -0.257** |
| top pickup | 0.845 | 0.860 | +0.015 |
| top delivered | 0.661 | 0.724 | +0.063 |

3000 iteration에서 접근과 pickup은 유지되거나 좋아졌다. 붕괴 지점은 pickup 이후 base
carry/place다. base episode의 first-stop 분해에서 `carry/place`가 `0.458 → 0.754`로
증가했고, 10 frame 연속으로 손이 박스에서 멀어지거나 lift가 0.05 m 아래로 내려간
`grasp break`가 `0.086 → 0.547`로 증가했다. 반면 rehearsal과 top carrier는 개선됐다.
따라서 전체 `sr`와 `fin` 상승은 핵심 base sequence의 보존을 의미하지 않는다.

### CLEAR 동작은 조건부로 성공했다

| 순차 단계 | epoch 9500 | latest | 변화 |
|---|---:|---:|---:|
| PLACE / base | 0.375 | 0.118 | -0.257 |
| RELEASE / base | 0.164 | 0.074 | -0.090 |
| `release_given_place` | 0.439 | 0.628 | +0.189 |
| CLEAR 통과 / base | 0.054 | 0.053 | -0.001 |
| `clear_given_release` | 0.325 | 0.714 | +0.389 |
| CLEAR delay p50 | 66 frame | 52 frame | -14 frame |
| retreat root distance p50 | 0.638 m | 0.880 m | +0.242 m |
| retreat arc p50 | 0.221 m | 0.606 m | +0.385 m |
| base displacement p95 | 0.300 m | 0.116 m | -0.185 m |

latest는 RELEASE에 도달한 episode에서 목표 arc `0.60 m`를 실제로 수행한다. 손 이격을
유지한 CLEAR step 비율은 1.000이고, `clear_given_release`는 0.714, retreat arc 중앙값은
0.606 m다. epoch 9500의 arc 중앙값 0.221 m와 비교하면 ms41의 negative CLEAR reward가
정지·역방향 행동을 줄이고 후측방 이동을 학습시켰다는 증거다. release 이후 base box
displacement p95도 0.300 m에서 0.116 m로 감소했다.

그러나 end-to-end 통과율은 다음처럼 상쇄된다.

```text
epoch 9500:  PLACE 0.375 × RELEASE|PLACE 0.439 × CLEAR|RELEASE 0.325 ≈ 0.054
latest:      PLACE 0.118 × RELEASE|PLACE 0.628 × CLEAR|RELEASE 0.714 ≈ 0.053
```

즉 **CLEAR에 도달한 뒤의 실행은 해결됐지만 CLEAR까지 가는 base carry가 무너져 최종
CLEAR 통과율은 5.4%에서 5.3%로 그대로다.** 여기서 CLEAR 통과는 top agent가 실제 최종
stacking을 완료했다는 뜻이 아니라 `STACK` phase에 진입했다는 뜻이다. 현재 73열
metrics에는 최종 stack 성공 사건이 별도 보존되지 않는다.

### 학습 곡선과 원인 해석

TensorBoard의 각 지점 직전 100 iteration 평균은 다음과 같다.

| 학습 진행 | +500 | +1000 | +1500 | +2000 | +2500 | +3000 |
|---|---:|---:|---:|---:|---:|---:|
| RELEASE phase fraction | 0.065 | 0.030 | 0.015 | 0.013 | 0.011 | 0.010 |
| CLEAR phase fraction | 0.020 | 0.013 | 0.009 | 0.007 | 0.008 | 0.007 |
| move-ok rate | 0.316 | 0.413 | 0.421 | 0.455 | 0.469 | 0.573 |
| stall rate | 0.684 | 0.587 | 0.579 | 0.545 | 0.531 | 0.427 |
| reverse rate | 0.390 | 0.326 | 0.310 | 0.269 | 0.258 | 0.201 |
| CLEAR reward | -0.204 | -0.097 | -0.076 | -0.030 | -0.013 | +0.112 |

phase fraction은 체류시간에도 영향을 받으므로 단독 event rate로 해석하지 않는다. 다만
고정 checkpoint 평가에서도 PLACE와 RELEASE 절대 도달률이 함께 감소했기 때문에,
survivor의 CLEAR 행동 품질은 계속 좋아지는 동안 그 phase를 방문하는 base episode가
감소했다는 방향은 일치한다.

가장 가능성 높은 설명은 **shared steering tokenizer/internal adapter의 phase 간 간섭**이다.
ms41은 Transformer와 carry 경로를 동결한 채 steering tokenizer와 internal adapter만 학습한다.
CLEAR reward gradient가 드물게 살아남은 CLEAR sample의 후퇴를 최적화하는 동안, 같은
가중치를 사용하는 base carry 동작이 훼손됐다. `STACK_ZERO_CARRY_OBS=1`은 CLEAR 이후에만
적용되므로 관측 zeroing이 pre-CLEAR 입력을 직접 지운 것은 아니다. 또한 50% carry
rehearsal가 일반 carry와 top 역할은 개선했지만 stack sequence의 base-role carry를
보존하지 못했다.

이는 원인 후보에 대한 해석이며 gradient conflict를 직접 측정한 결과는 아니다. 정확한
원인을 확정하려면 phase별 gradient cosine이나 base-role carry 고정 probe가 필요하다.

### 판정

**ms41은 후퇴 primitive 학습에는 성공했지만 end-to-end sequential stacking 정책으로는
채택하지 않는다.**

- CLEAR 후퇴 능력 증명: `stage_elatest`
- 두 체크포인트 중 end-to-end 보존이 나은 선택: `stage_e9500`
- 최종 정책으로 latest 채택: 기각
- 이유: 조건부 CLEAR 성공 증분보다 base delivery 회귀가 커서 순 통과율이 늘지 않음

ms41 sidecar로 평가한 정확한 epoch 9000 initial 결과는 없다. 따라서 초기 정책 대비
회귀량을 엄밀히 주장하려면 `stage_einitial`을 같은 조건으로 추가해야 한다. 다음 checkpoint
선택은 전체 carry reward나 개별 `sr`가 아니라
`PLACE × release_given_place × clear_given_release`와 pair-level `both`를 함께 기준으로
해야 한다. 보존된 e10000~e11500 checkpoint를 같은 stage 평가로 비교하면 base carry가
크게 무너지기 전 CLEAR 동작이 가장 많이 열린 중간 Pareto 지점을 찾을 수 있다.

## ms42 — completed-CARRY 0.50 continuity pilot (500 iteration)

대상은 `ms42_ms18init_negclear_cont050_zeroobs_nomask_500_s0`이다. ms18 epoch 9000에서
시작해 GPU 7, 1024 env, 500 iteration을 학습했고 100 iteration마다 checkpoint를 보존했다.
모델 구조와 340-D observation은 바꾸지 않았다. ms41과 동일하게 Transformer, self/carry
encoder, composer를 동결하고 steering tokenizer와 `internal_adapt_mlp`만 학습했다.

ms41과의 유일한 reward 축은 다음 완료 보상이다.

```text
STACK_SEQUENTIAL_REWARD_MASK=1
STACK_CARRY_DONE_REWARD=0.50
STACK_RELEASE_DONE_REWARD=0.00
STACK_CLEAR_DONE_REWARD=0.00
```

`negative CLEAR`, post-CLEAR carry observation zeroing, attention mask OFF, rehearsal 0.50,
foot-box penalty 0.10은 그대로 유지했다. 따라서 이번 비교는 CARRY가 끝난 뒤 +0.50을
계속 지급해 phase 경계의 reward 절벽을 줄이는 효과를 본다.

### 검증과 계측

`_compute_reward`에 reward를 바꾸지 않는 CARRY 전용 TensorBoard 진단을 추가했다.
비-rehearsal base가 CARRY이고 pickup 이후인 표본만 남기고 다른 표본은 NaN으로 기록한다.

- `stack_reward/{carry_native,carry_foot_penalty}`
- `stack_state/{carry_target_dist,carry_grasp_ok,carry_break_rate,near_goal_not_delivered,carry_path_fraction}`

첫 smoke 태그 `smoke_ms42_cont050_2it_20260909_s0`는 `minibatch_size=2048`이 기존
`amp_minibatch_size=4096`보다 작아 학습 시작 전 assert로 종료됐다. 산출물은 삭제하지 않았다.
`smoke2_ms42_cont050_2it_20260909_s0`는 64 env, minibatch 4096, 2 iteration으로 정상
완료했고 새 scalar 7개와 epoch 9001~9003 checkpoint를 확인했다. 본 학습 첫 backward에서
steering tokenizer gradient 합은 468.0, adapter는 923.6이었고 new/old carry encoder,
self encoder, Transformer, composer는 모두 0이었다.

### checkpoint stage 평가

모든 평가는 GPU 7, 512 env, seed 0, 동일한 `stack_stage_eval.sh` 경로로 실행했고 `rc=0`이다.
initial은 sidecar의 정확한 ms18 epoch 9000 checkpoint를 다시 로드했다.

| checkpoint | base n | pickup | break | delivered | PLACE | RELEASE | CLEAR | release/place | clear/release |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 721 | 0.871 | 0.044 | 0.716 | 0.703 | 0.287 | 0.007 | 0.408 | 0.024 |
| 9100 (+100) | 718 | 0.861 | 0.036 | 0.719 | 0.708 | 0.206 | 0.006 | 0.291 | 0.027 |
| 9300 (+300) | 682 | 0.875 | 0.050 | 0.717 | 0.708 | 0.198 | 0.019 | 0.280 | 0.096 |
| 9500 (+500) | 719 | 0.857 | 0.043 | 0.652 | 0.645 | 0.175 | 0.086 | 0.272 | 0.492 |

ms42는 ms41의 pre-CLEAR 붕괴를 대부분 막았다. 같은 +500 시점에서 ms41 e9500은 base
`pickup=0.832`, `break=0.086`, `delivered=0.384`, `PLACE=0.375`, `CLEAR=0.054`였지만
ms42 e9500은 각각 `0.857/0.043/0.652/0.645/0.086`이다. delivery는 +26.8%p, grasp
break는 -4.3%p, end-to-end CLEAR는 +3.2%p다. carry freeze가 직접 원인이었다면 작은
완료 reward 상수만으로 이 차이가 나기 어렵기 때문에, ms41의 주원인은 frozen carry
encoder보다 phase reward 불연속이었다는 해석을 지지한다.

ms42 자체의 initial 대비로도 pickup과 break는 500 iteration 동안 거의 보존됐고 delivery는
6.4%p 감소에 그쳤다. 반면 `clear_given_release`는 2.4%에서 49.2%로 증가했다. 전체
CLEAR도 다음 곱과 일치한다.

```text
initial: PLACE 0.703 × RELEASE|PLACE 0.408 × CLEAR|RELEASE 0.024 ≈ 0.007
e9500:   PLACE 0.645 × RELEASE|PLACE 0.272 × CLEAR|RELEASE 0.492 ≈ 0.086
```

따라서 e9500은 평가한 checkpoint 중 end-to-end CLEAR가 가장 높다. 다만 이는 top agent의
최종 stacking 성공이 아니라 base가 `STACK` phase에 진입했다는 뜻이다.

### TensorBoard가 보여 주는 학습 방향

아래는 +100/+300/+500 부근의 phase-conditional scalar다.

| 지표 | +100 | +300 | +500 |
|---|---:|---:|---:|
| carry native reward | 0.744 | 0.771 | 0.758 |
| carry foot penalty | -0.0030 | -0.0028 | -0.0019 |
| carry grasp OK | 0.982 | 0.989 | 0.984 |
| carry break latched | 0.065 | 0.050 | 0.039 |
| near goal, not delivered | 0.093 | 0.114 | 0.117 |
| carry path fraction | 0.699 | 0.701 | 0.714 |
| CLEAR raw total | -0.395 | -0.260 | -0.049 |
| CLEAR move positive | 0.070 | 0.130 | 0.199 |
| move OK rate | 0.149 | 0.289 | 0.489 |
| stall rate | 0.851 | 0.711 | 0.511 |
| reverse rate | 0.502 | 0.378 | 0.190 |
| `v_along` (command 0.75 m/s) | -0.007 | 0.057 | 0.130 |
| retreat arc | 0.067 m | 0.113 m | 0.275 m |

`CLEAR raw total`은 sequential 완료 상수를 더하기 전에 기록한 phase-local 값이다. 실제
CLEAR base reward에는 표의 값에 carry floor +0.50이 더해진다. 후퇴 방향 속도, move
성공률, arc가 증가하고 stall/reverse가 감소했으므로 negative CLEAR reward는 의도한
행동을 학습시켰다. CARRY native reward와 grasp 상태도 안정적이어서 foot penalty 0.10이
현재 병목이라는 증거는 없다.

### 남은 병목과 다음 reward 수정

현재 first bottleneck은 `RELEASE|PLACE=0.272`이다. RELEASE에서는 carry floor +0.50에
`0.50*h + I(clear)*(0.30*support+0.20*stable)`이 붙는다. 두 손을 거의 떼되 0.15 m
문턱을 5 frame 유지하지 않으면 약 1.0/step을 받을 수 있지만, CLEAR로 넘어가면 RELEASE
항이 사라져 초기에는 `0.50 + CLEAR raw(-0.4 안팎)`으로 내려간다. 일회성 transition
bonus가 있어도 장기 value 기준으로 문턱 직전 체류가 유리해질 수 있다. 이는 평가에서
`release_given_place`가 0.408→0.272로 감소하고, 일단 CLEAR에 들어간 표본의 성공은
0.024→0.492로 증가한 패턴과 정확히 맞는다.

다음 pilot은 carry를 unfreeze하거나 attention mask를 켜지 않는다. 먼저 RELEASE의 `h`,
`clear_now`, support/stable, phase-local reward와 5-frame streak를 TensorBoard에 추가한 뒤,
RELEASE 완료값을 CLEAR에도 이어 주고 정지 CLEAR가 양수가 되지 않도록 stall penalty를
같이 맞추는 것이 최소 reward 수정이다. 시작점 후보는 `release_done=1.0`,
`clear stall penalty=1.5`이며, 둘은 경계 reward 연속성과 정지 해법 제거를 함께 만족하는
결합 파라미터로 다뤄야 한다. ms42 e9500은 현재까지 CARRY 보존과 CLEAR 학습을 동시에
확인한 최선 checkpoint지만, RELEASE 병목 때문에 최종 sequential 정책으로 확정하지 않는다.

평가 원시는 다음 네 파일이다.

- `runs/results/masteer/eval_ms42_ms18init_negclear_cont050_zeroobs_nomask_500_s0__stage_einitial.npy`
- `runs/results/masteer/eval_ms42_ms18init_negclear_cont050_zeroobs_nomask_500_s0__stage_e9100.npy`
- `runs/results/masteer/eval_ms42_ms18init_negclear_cont050_zeroobs_nomask_500_s0__stage_e9300.npy`
- `runs/results/masteer/eval_ms42_ms18init_negclear_cont050_zeroobs_nomask_500_s0__stage_e9500.npy`

## ms43 — RELEASE→CLEAR reward continuity (3000 iteration 실행)

ms42 e9500에서 CARRY는 보존되고 `clear_given_release=0.492`까지 좋아졌지만
`release_given_place=0.272`가 새 병목이었다. RELEASE 성공 경계의 step reward는 carry
floor 0.50과 release local 최대 약 1.0을 합쳐 약 1.5다. 반면 ms42는 CLEAR 전환 직후
release local 항이 사라져 carry floor 0.50과 음수 CLEAR local만 남았다. 손을 0.15 m
문턱 직전까지 떼되 5 frame 전환을 피하는 해법을 만들 수 있는 reward 불연속이다.

ms43은 모델 구조, observation, controller, freeze, zero carry observation과 attention
mask OFF를 전부 유지하고 다음 두 결합 파라미터만 바꾼다.

```text
STACK_CARRY_DONE_REWARD=0.50
STACK_RELEASE_DONE_REWARD=1.00
STACK_CLEAR_STALL_PEN_W=1.50
STACK_CLEAR_GRACE_STEPS=5
```

CLEAR grace 동안 완료 floor는 1.50이라 성공적인 RELEASE 경계와 연속이다. grace 뒤
정지하면 stall -1.50이 완료 floor를 모두 상쇄하고 base/foot penalty가 추가되므로
정지 CLEAR는 양수 해법이 아니다. 올바른 방향으로 최소 속도를 넘기면 stall이 사라지고
기존 move-positive가 붙는다.

### RELEASE 계측과 smoke

`humanoid_ma_sequential_stack_release.py`에 phase 밖은 NaN인 RELEASE scalar 12개를
추가했다. 실제 reward는 바꾸지 않는 진단이다.

- reward: `release_total`, `release_local`, `release_hand_positive`,
  `release_support_positive`, `release_foot_penalty`, `release_transition_bonus`
- state: `release_hand_factor`, `release_hand_clear_rate`, `release_streak_fraction`,
  `release_support`, `release_stable`, `release_foot_distance`

64-env 2-iteration smoke는 정상 완료했지만 RELEASE 표본이 없어 scalar가 기록되지 않았다.
표본 부족을 코드 실패와 구분하기 위해 `smoke30_ms43_rel100_stall150_20260909_s0`를
30 iteration 실행했다. 12개 tag가 각각 26 step 기록됐고 마지막 표본은
`release_total=0.612`, `hand_factor=0.260`, `hand_clear=0`, `streak=0`,
`foot_distance=0.166 m`였다. 따라서 현재 손 떼기 병목을 직접 관찰할 수 있다.

### 본 학습

태그는 `ms43_ms18init_negclear_carry050_release100_stall150_zeroobs_nomask_3000_s0`이다.
ms18 epoch 9000에서 GPU 7, 1024 env, 3000 iteration, 100-iteration checkpoint로
2026-09-09에 분리 실행했고 PID는 3840815다. 로그는
`runs/queue/logs/ms43_ms18init_negclear_carry050_release100_stall150_zeroobs_nomask_3000_s0.train.log`다.

sidecar에서 `MS_ITERS=3000`, carry/release/clear 완료값 `0.50/1.00/0.00`, stall 1.5,
grace 5를 확인했다. 첫 backward gradient 합은 steering tokenizer 468.0, adapter
923.6이며 new/old carry encoder, self encoder, Transformer, composer는 0이다. 첫 rollout
FPS는 step 약 2.9~3.6만, total 약 2.0~2.2만이고 metrics 저장까지 확인했다.

완료 후 최소 평가는 exact initial, e9500(+500), e10000(+1000), e11000(+2000),
e12000(+3000)이다. 판정 우선순위는 `release_given_place` 회복, CARRY delivery/break
보존, `clear_given_release`, 전체 CLEAR 순이며 TensorBoard의 hand factor와 5-frame
streak가 실제로 상승하는지도 함께 본다.
