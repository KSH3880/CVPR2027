# MS81 CLEAR goal 3-D Juan 평가 보고서

## 기술 요약

MS81은 MS18 epoch 9000에서 시작해, CLEAR에서 carry box 상태 39-D를 fade하면서 마지막
goal 3-D에 최종 후퇴점을 유지하고, STACK/SUCCESS에서는 box 39-D 재노출을 막도록
학습한 정책이다.

2,160 episode 평가 결과 `box_radius`는 **58.38%**, `tokenhsi_carry`는 **65.60%**,
`strict_done`은 **3.84%**였다. 동일 표본 규모의 MS80보다 각각 `−1.81`, `−3.56`,
`−0.42%p`로, CLEAR goal 3-D 추가가 전체 성공률 개선으로 이어지지는 않았다.

## 위치 진입은 유지됐지만 안정 적층은 개선되지 않았다

환경별 첫 episode를 warm-up으로 제외하고, 1,080개 환경에서 뒤의 두 episode를 사용했다.
각 기준의 분모는 2,160 episode이며 3×3 box 조합별 분모는 240 episode다.

| 성공 판정 | 성공 | 성공률 | 참고용 95% Wilson 구간 | 비정상 종료율 |
|---|---:|---:|---:|---:|
| `strict_done` | 83 / 2,160 | **3.84%** | 3.11–4.74% | 13.38% |
| `box_radius` | 1,261 / 2,160 | **58.38%** | 56.29–60.44% | 13.84% |
| `tokenhsi_carry` | 1,417 / 2,160 | **65.60%** | 63.57–67.58% | 15.37% |

- `box_radius - strict_done`: **+54.54%p**
- `tokenhsi_carry - box_radius`: **+7.22%p**
- `tokenhsi_carry - strict_done`: **+61.76%p**

공간 허용영역에는 절반 이상 진입하지만 20-step 안정 적층은 3.84%에 그친다. MS80과
마찬가지로 최종 위치 도달보다 속도·자세·평행 조건을 연속 유지하는 단계가 주 병목이다.
비정상 종료율은 13–15%이며 timeout은 여기에 포함되지 않는다.

## 큰 top box에서 성능 저하가 집중된다

| Bottom / Top | `strict_done` | `box_radius` | `tokenhsi_carry` | TokenHSI − Box |
|---|---:|---:|---:|---:|
| 0.22 / 0.22 m | 4.17% (10) | 56.25% (135) | 65.42% (157) | +9.17%p |
| 0.22 / 0.42 m | 2.50% (6) | 65.42% (157) | 72.50% (174) | +7.08%p |
| 0.22 / 0.57 m | 5.83% (14) | 38.75% (93) | 51.67% (124) | +12.92%p |
| 0.42 / 0.22 m | 1.25% (3) | 75.83% (182) | **74.58% (179)** | −1.25%p |
| 0.42 / 0.42 m | 3.75% (9) | **80.42% (193)** | 69.58% (167) | −10.84%p |
| 0.42 / 0.57 m | **7.50% (18)** | 45.00% (108) | 56.67% (136) | +11.67%p |
| 0.57 / 0.22 m | 1.67% (4) | 79.58% (191) | 62.50% (150) | −17.08%p |
| 0.57 / 0.42 m | 5.83% (14) | 69.17% (166) | 73.75% (177) | +4.58%p |
| 0.57 / 0.57 m | 2.08% (5) | **15.00% (36)** | 63.75% (153) | **+48.75%p** |

`box_radius` 최고 조합은 medium/medium 80.42%, 최저는 large/large 15.00%다. 작은 또는
medium bottom에 large top을 올리는 조합도 각각 38.75%, 45.00%로 낮다. CLEAR goal
추가 여부와 무관하게 큰 top box의 실제 지지영역 배치가 계속 주요 실패축이다.

large/large에서 `tokenhsi_carry`는 63.75%지만 실제 bottom 지지영역을 보는
`box_radius`는 15.00%다. committed target 근접 성공을 실제 적층 성공으로 해석하면
성능을 크게 과대평가할 수 있다.

## MS80 대비 CLEAR goal 3-D는 전체 평균을 높이지 못했다

MS80과 MS81은 모두 MS18 epoch 9000에서 3,000 iteration 학습했고, MS81은 CLEAR의
carry observation 마지막 3-D에 최종 후퇴점을 유지한 것이 핵심 차이다.

| 성공 판정 | MS80 | MS81 | 관측 변화 |
|---|---:|---:|---:|
| `strict_done` | 4.26% (92/2,160) | 3.84% (83/2,160) | **−0.42%p** |
| `box_radius` | 60.19% (1,300/2,160) | 58.38% (1,261/2,160) | **−1.81%p** |
| `tokenhsi_carry` | 69.17% (1,494/2,160) | 65.60% (1,417/2,160) | **−3.56%p** |

비정상 종료율도 MS81이 `strict_done +1.62%p`, `box_radius +2.54%p`,
`tokenhsi_carry +3.75%p` 높았다. 따라서 현재 결과만 보면 goal 3-D 추가는 전체 성공과
생존성 모두에서 채택 근거가 없다.

효과는 조합별로 균일하지 않다. `box_radius` 기준 MS81은 large/medium에서 `+6.67%p`,
large/small에서 `+3.75%p`, medium/medium에서 `+3.34%p`였지만, small/large는
`−10.83%p`, medium/large는 `−10.42%p`였다. 후퇴 goal이 특정 base 크기에는 도움을
줄 가능성이 있으나, 큰 top box 일반화를 해결하지는 못했다.

## 보고 기준과 결론

| 판정 | 의미 | 권장 용도 |
|---|---|---|
| `strict_done` | 위치·속도·upright·평행 조건을 20 step 연속 만족 | 최종 안정 적층 완료율 |
| `box_radius` | top이 현재 bottom 지지 반경과 Z 허용치에 한 번 진입 | **주 공간 배치 성공률** |
| `tokenhsi_carry` | committed target과 3D 거리 0.20 m 이내 | 원본 TokenHSI 호환 참고값 |

현재 결론은 **MS81 goal 3-D 설정을 MS80보다 우선 채택하지 않는다**이다. 헤드라인은
`box_radius 58.38%`, `strict_done 3.84%`로 보고하고, legacy 참고값으로
`tokenhsi_carry 65.60%`를 병기한다.

## 한계와 다음 확인

- MS80과 MS81은 동일 seed와 동일 표본 수지만 같은 저장 trajectory의 paired 재채점은
  아니다. 비정상 종료 수가 서로 달라 rollout 표본 변동이 존재한다.
- Wilson 구간은 2,160 episode를 독립으로 가정한 참고값이며, 같은 환경에서 나온 두
  measured episode의 군집 상관은 반영하지 않았다.
- goal 3-D가 CLEAR 제어 자체를 개선했는지는 성공률만으로 분리할 수 없다. CLEAR endpoint
  도달률·도달 시간·재접촉률을 MS80과 직접 비교해야 한다.
- 성공 판정 차이를 정확히 비교하려면 한 rollout에서 세 latch를 동시에 기록해야 한다.

## 원본 결과

- [`strict_done` 로그](../runs/results/masteer/juan_eval_ms81_e12000_n2160_strict_done.log)
- [`box_radius` 로그](../runs/results/masteer/juan_eval_ms81_e12000_n2160_box_radius.log)
- [`tokenhsi_carry` 로그](../runs/results/masteer/juan_eval_ms81_e12000_n2160_tokenhsi_carry.log)
- [MS80 비교 보고서](MS80_JUAN_EVAL_SUCCESS_REPORT.md)
- 평가 checkpoint: `Humanoid_22-21-42-46/nn/Humanoid.pth`
- 평가 seed: 0
- 표본: 1,080 env × 2 measured episodes = 2,160 episodes/criterion
