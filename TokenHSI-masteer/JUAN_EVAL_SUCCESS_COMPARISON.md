# MS75 Juan 기준 성공률 비교

## 요약

동일한 MS75 최종 checkpoint를 로컬 sequential-stack 환경에서 세 성공 판정으로 각각
평가했다. 환경별 첫 episode를 warm-up으로 버리고, 270개 환경에서 뒤의 두 episode만
사용하여 각 결과의 분모는 540 episode다.

| 성공 판정 | 성공 | 성공률 | 참고용 95% Wilson 구간 | 비정상 종료율 |
|---|---:|---:|---:|---:|
| `strict_done` | 16 / 540 | **2.96%** | 1.83–4.76% | 14.44% |
| `box_radius` | 287 / 540 | **53.15%** | 48.93–57.32% | 15.19% |
| `tokenhsi_carry` | 349 / 540 | **64.63%** | 60.51–68.55% | 16.11% |

- `box_radius - strict_done`: **+50.19%p**
- `tokenhsi_carry - box_radius`: **+11.48%p**
- `tokenhsi_carry - strict_done`: **+61.67%p**

핵심 결과는 정책이 top box를 허용 위치에 한 번 넣는 경우는 많지만, 속도·자세·평행
조건을 20 step 연속 유지해 `DONE`에 도달하는 경우는 매우 적다는 것이다. 비정상 종료율은
세 평가에서 14.4–16.1%로 비슷하므로 `strict_done`의 낮은 성공률은 주로 낙상보다
최종 안정화 gate를 통과하지 못한 데서 발생한다. 단, timeout은 비정상 종료율에 포함되지
않는다.

## 성공 판정 정의

| 판정 | 조건 | 해석 |
|---|---|---|
| `strict_done` | XY ≤ 0.10 m, Z ≤ 0.08 m, 선속도 ≤ 0.08 m/s, 각속도 ≤ 0.20 rad/s, upright ≤ 10°, base/top 평행 오차 ≤ 15°, base 이동 ≤ 0.50 m를 20 step 연속 만족 | 안정적인 최종 적층 완료 |
| `box_radius` | STACK phase에서 top 중심이 bottom의 지지 반경 안에 있고 Z 오차 < 0.08 m인 순간을 latch | 물리적 지지영역 진입 여부 |
| `tokenhsi_carry` | STACK phase에서 top 중심과 목표 중심의 3D 거리 ≤ 0.20 m인 순간을 latch | 원본 TokenHSI carry 호환 성공 |

`box_radius`와 `tokenhsi_carry`는 항상 포함관계가 아니다. 전자는 bottom 크기에 따라 XY
허용 반경이 0.156/0.297/0.403 m로 달라지는 원통형 조건이고, 후자는 고정 0.20 m 구형
조건이다. 따라서 `tokenhsi_carry`를 모든 크기에서 일관되게 더 느슨한 판정으로 해석하면
안 된다.

## 3×3 box 조합별 성공률

각 조합은 60 episode다. 괄호 안은 성공 episode 수다.

| Bottom / Top | `strict_done` | `box_radius` | `tokenhsi_carry` | TokenHSI − Box |
|---|---:|---:|---:|---:|
| 0.22 / 0.22 m | 1.7% (1) | 45.0% (27) | 56.7% (34) | +11.7%p |
| 0.22 / 0.42 m | 0.0% (0) | 56.7% (34) | 68.3% (41) | +11.7%p |
| 0.22 / 0.57 m | 1.7% (1) | 25.0% (15) | 36.7% (22) | +11.7%p |
| 0.42 / 0.22 m | 0.0% (0) | 78.3% (47) | 80.0% (48) | +1.7%p |
| 0.42 / 0.42 m | 6.7% (4) | 68.3% (41) | 68.3% (41) | 0.0%p |
| 0.42 / 0.57 m | 6.7% (4) | 36.7% (22) | 60.0% (36) | +23.3%p |
| 0.57 / 0.22 m | 1.7% (1) | **86.7% (52)** | 75.0% (45) | **−11.7%p** |
| 0.57 / 0.42 m | **8.3% (5)** | 58.3% (35) | 75.0% (45) | +16.7%p |
| 0.57 / 0.57 m | 0.0% (0) | 23.3% (14) | 61.7% (37) | +38.3%p |

`box_radius`에서 가장 쉬운 조합은 큰 bottom과 작은 top(86.7%)이고, 가장 어려운 조합은
큰 bottom과 큰 top(23.3%) 및 작은 bottom과 큰 top(25.0%)이다. 단순히 bottom의 지지
반경만으로 난도가 정해지지 않고, 큰 top box의 운반·배치 안정성이 함께 병목이다.

`tokenhsi_carry`가 전체 평균에서는 11.48%p 높지만, 큰 bottom/작은 top 조합에서는 오히려
`box_radius`가 11.7%p 높다. 이는 두 판정의 기하가 다르고 세 결과가 서로 다른 rollout에서
측정됐기 때문이다.

## 해석과 권장 보고 방식

1. **주 공간 배치 지표:** `box_radius = 53.15%`를 사용한다. 실제 bottom 크기에 맞춘
   지지영역 진입을 측정하므로 sequential stack의 공간적 의미와 가장 직접적으로 맞는다.
2. **최종 안정성 지표:** `strict_done = 2.96%`를 함께 보고한다. 두 수치의 큰 차이는
   “위치에는 도달하지만 안정적인 적층 완료는 거의 못 한다”는 현재 병목을 보여준다.
3. **기존 TokenHSI 비교:** `tokenhsi_carry = 64.63%`는 원본 0.20 m carry 판정과 비교할
   때만 사용한다. `box_radius`의 단순한 상위/하위 기준으로 해석하지 않는다.

## 비교 한계

- 세 판정은 같은 저장 trajectory를 재채점한 것이 아니라 같은 seed로 각각 다시 실행한
  독립 rollout이다. 실제로 비정상 종료 수가 78/82/87로 달라졌으므로 성공률 차이에는
  시뮬레이션 및 episode 표본 변동도 포함된다.
- Wilson 구간은 540 episode를 독립 표본으로 간주한 참고값이다. 같은 환경에서 두 measured
  episode가 나오므로 군집 상관을 반영한 엄밀한 불확실성 구간은 아니다.
- 성공률의 판정 민감도만 정확히 비교하려면 한 번의 rollout에서 세 성공 latch를 동시에
  기록해 episode별 paired 비교를 해야 한다.

## 원본 결과

- [`box_radius` 로그](../runs/results/masteer/juan_eval_ms75_3000_s0_juan_eval.log)
- [`strict_done` 로그](../runs/results/masteer/juan_eval_ms75_3000_s0_juan_strict_done.log)
- [`tokenhsi_carry` 로그](../runs/results/masteer/juan_eval_ms75_3000_s0_juan_tokenhsi_carry.log)
- 평가 대상: `Humanoid_17-19-10-12/nn/Humanoid.pth`
- 평가 seed: 0
- box grid: 0.22/0.42/0.57 m의 3×3 조합, 조합당 60 episode
