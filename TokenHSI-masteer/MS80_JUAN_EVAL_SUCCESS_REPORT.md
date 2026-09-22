# MS80 Juan sequential-stack 성공률 보고서

## 기술 요약

MS80 최종 checkpoint를 로컬 sequential-stack 환경에서 Juan 방식으로 평가한 결과,
공간 배치 성공률은 `box_radius` **60.19%**, 원본 TokenHSI 호환 판정은
`tokenhsi_carry` **69.17%**였지만, 20 step 안정화까지 완료한 `strict_done`은
**4.26%**였다.

이전 MS75 평가보다 관측 성공률은 각각 `+7.04`, `+4.54`, `+1.30%p` 높고 비정상
종료율도 낮았다. 그러나 세 기준과 두 정책은 동일 trajectory를 재채점한 paired 비교가
아니므로, 차이 전체를 정책 학습 효과로 단정할 수는 없다.

## 위치 진입은 개선됐지만 안정 적층은 여전히 병목이다

환경별 첫 episode를 warm-up으로 제외하고 1,080개 환경에서 두 episode씩 사용했다.
따라서 각 기준의 분모는 2,160 episode이며, 3×3 box 조합별 분모는 240 episode다.

| 성공 판정 | 성공 | 성공률 | 참고용 95% Wilson 구간 | 비정상 종료율 |
|---|---:|---:|---:|---:|
| `strict_done` | 92 / 2,160 | **4.26%** | 3.49–5.20% | 11.76% |
| `box_radius` | 1,300 / 2,160 | **60.19%** | 58.10–62.23% | 11.30% |
| `tokenhsi_carry` | 1,494 / 2,160 | **69.17%** | 67.19–71.08% | 11.62% |

- `box_radius - strict_done`: **+55.93%p**
- `tokenhsi_carry - box_radius`: **+8.98%p**
- `tokenhsi_carry - strict_done`: **+64.91%p**

비정상 종료율은 모든 판정에서 약 11–12%인데 `strict_done` 성공률은 4.26%뿐이다.
따라서 주된 병목은 낙상이나 bottom 과이동보다 top box를 목표에 둔 뒤 속도·자세·평행
조건을 20 step 연속 유지하는 최종 안정화다. Timeout은 비정상 종료율에 포함되지 않는다.

## 3×3 box 조합에서 큰 top box가 공간 배치를 어렵게 한다

| Bottom / Top | `strict_done` | `box_radius` | `tokenhsi_carry` | TokenHSI − Box |
|---|---:|---:|---:|---:|
| 0.22 / 0.22 m | 2.92% (7) | 60.00% (144) | 64.58% (155) | +4.58%p |
| 0.22 / 0.42 m | 2.08% (5) | 69.17% (166) | **80.83% (194)** | +11.66%p |
| 0.22 / 0.57 m | 5.42% (13) | 49.58% (119) | 55.00% (132) | +5.42%p |
| 0.42 / 0.22 m | 2.50% (6) | **78.33% (188)** | 79.58% (191) | +1.25%p |
| 0.42 / 0.42 m | 2.08% (5) | 77.08% (185) | 76.25% (183) | −0.83%p |
| 0.42 / 0.57 m | 3.33% (8) | 55.42% (133) | 58.33% (140) | +2.91%p |
| 0.57 / 0.22 m | 2.08% (5) | 75.83% (182) | 63.75% (153) | −12.08%p |
| 0.57 / 0.42 m | **14.58% (35)** | 62.50% (150) | 72.08% (173) | +9.58%p |
| 0.57 / 0.57 m | 3.33% (8) | **13.75% (33)** | 72.08% (173) | **+58.33%p** |

`box_radius`에서는 medium bottom/small top이 78.33%로 가장 높고, large/large가
13.75%로 크게 무너졌다. 반면 `tokenhsi_carry`는 large/large에서도 72.08%다.
`tokenhsi_carry`는 committed target과의 고정 0.20 m 거리만 보지만 `box_radius`는
현재 움직인 bottom 위의 실제 지지영역을 본다. 따라서 large/large의 58.33%p 격차는
legacy carry 성공이 실제 지지관계보다 훨씬 낙관적일 수 있음을 보여준다. 두 수치는 별도
rollout이므로 episode 단위 포함관계로 해석할 수는 없다.

엄격 성공 92건 중 35건(38.0%)이 large bottom/medium top 한 조합에 몰렸다. 다른 여덟
조합의 `strict_done`은 모두 5.42% 이하이므로, 안정 적층 능력은 box 조합에 강하게
의존한다.

## MS75 대비 공간 성공률은 상승했지만 조합별 개선은 균일하지 않다

| 성공 판정 | MS75 | MS80 | 관측 변화 |
|---|---:|---:|---:|
| `strict_done` | 2.96% (16/540) | 4.26% (92/2,160) | **+1.30%p** |
| `box_radius` | 53.15% (287/540) | 60.19% (1,300/2,160) | **+7.04%p** |
| `tokenhsi_carry` | 64.63% (349/540) | 69.17% (1,494/2,160) | **+4.54%p** |

MS80의 `box_radius` 증가는 small/large(`+24.58%p`)와 medium/large(`+18.75%p`)에서
크게 나타났다. 반대로 large/small은 `−10.84%p`, large/large는 `−9.58%p`였다.
전체 평균 상승만으로 모든 box 조합의 일반화가 좋아졌다고 결론 내리면 안 된다.

비정상 종료율은 판정별로 MS75 대비 `−2.68%p`(`strict_done`),
`−3.89%p`(`box_radius`), `−4.49%p`(`tokenhsi_carry`) 낮았다. 이는 MS80에서 rollout
생존성이 관측상 개선됐다는 근거지만, 서로 다른 표본 크기와 비-paired 실행이라는 한계가
있다.

## 성공 판정과 보고 기준

| 판정 | 조건 | 권장 용도 |
|---|---|---|
| `strict_done` | 위치·속도·upright·box 평행·base 이동 조건을 20 step 연속 만족 | 최종 안정 적층 완료율 |
| `box_radius` | STACK phase에서 top이 현재 bottom 지지 반경 안에 있고 Z 오차 < 0.08 m인 순간을 latch | **주 공간 배치 성공률** |
| `tokenhsi_carry` | STACK phase에서 top과 committed target의 3D 거리 ≤ 0.20 m인 순간을 latch | 원본 TokenHSI carry와의 호환 비교 |

헤드라인은 `box_radius 60.19%`와 `strict_done 4.26%`를 함께 보고하는 것이 적절하다.
`tokenhsi_carry 69.17%`는 legacy 참고값으로 분리해야 한다.

## 한계와 다음 확인

- 세 성공 기준은 같은 저장 trajectory를 재채점한 것이 아니라 각각 다시 실행한 결과다.
  비정상 종료 수가 244/251/254로 달라져 표본 변동이 확인된다.
- MS75는 기준당 540 episode, MS80은 2,160 episode라 표본 크기도 다르다.
- Wilson 구간은 episode 독립성을 가정한 참고값이다. 같은 환경에서 나온 두 measured
  episode의 군집 상관은 반영하지 않았다.
- 성공 판정 자체의 차이를 정확히 분리하려면 한 rollout에서 세 latch를 동시에 저장해
  paired episode 비교를 해야 한다.
- 후속 분석에서는 `box_radius` 성공 후 `strict_done` 실패 episode를 대상으로 top box의
  속도, 각속도, upright, 평행 오차 중 어느 gate가 가장 자주 끊기는지 집계하는 것이
  우선이다.

## 원본 결과

- [`strict_done` 로그](../runs/results/masteer/juan_eval_ms80_e12000_n2160_strict_done.log)
- [`box_radius` 로그](../runs/results/masteer/juan_eval_ms80_e12000_n2160_box_radius.log)
- [`tokenhsi_carry` 로그](../runs/results/masteer/juan_eval_ms80_e12000_n2160_tokenhsi_carry.log)
- [MS75 비교 보고서](JUAN_EVAL_SUCCESS_COMPARISON.md)
- 평가 checkpoint: `Humanoid_22-11-15-29/nn/Humanoid.pth`
- 평가 seed: 0
- 표본: 1,080 env × 2 measured episodes = 2,160 episodes/criterion
