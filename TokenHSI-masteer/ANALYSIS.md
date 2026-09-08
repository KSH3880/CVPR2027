# TokenHSI-masteer 분석 기록

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
