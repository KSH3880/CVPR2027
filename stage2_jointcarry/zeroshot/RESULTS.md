# 2026-10-15 zero-shot 결과

박스 크기로 정규화한 테이블 mesh 1종은 기존 정책으로 운반을 유지하지 못했다.
다른 모든 물체의 불가능을 뜻하지 않으며, 실제 크기의 원형 테이블 평가도 아니다.

## 유효한 carryWith 비교

- epoch21500 고정, 학습 없음. GPU6 MPS의 정책 추론 + CPU PhysX/CPU pipeline.
- 2명/4물체/16환경×3회 = 조건당48장면, episode600프레임(20초), 같은 시드20261015.
- 이미 들고 있는 paired carryWith RSI로 시작. 초기 payload state 오차≤1.2e−7, 목표 오차0.
- 테이블은 O0만 교체. 크기0.52×0.8×0.4m·질량16.64kg. 기존 bbox8점 관측·박스 파지 앵커·보상을 유지.
- 이 평가기는 edge_context_eval이며 single-agent ObjectSet 0회차 제외 규칙을 그대로 적용하는 하네스가 아니다. 3회 전체를 집계했다.

| 지표 | 박스 | 테이블 mesh |
|---|---:|---:|
| 최종 목표 XY 오차 평균 | 0.0497m | 2.4739m |
| 목표 XY 오차 감소 평균 | 2.4377m | 0.0136m |
| 시작점 대비 최대 XY 변위 평균 | 2.5798m | 0.7738m |
| 최종 중심 높이 >0.5m | 48/48 | 0/48 |
| 최종 HOLDING 대리 판정(사람별) | 93.75% | 0% |
| 기존 엄격한 scene 목표 완료 | 0% | 0% |

이동량은 미끄러짐/낙하 중 이동도 포함하므로 공동 운반 성공률로 해석하지 않는다.
박스도 엄격한 완료율0%이므로 배치까지의 task 성공 일반화를 판정하지 않는다.
높이·거리 궤적은 테이블의 운반 유지 실패를 뒷받침한다. 시각적 손 접촉 및 사람별 지지력은 측정하지 않았다.
현재 정책은 동일 bbox의 서로 다른 실제 mesh를 관측에서 구별하지 못한다. 박스 기준 RSI와 파지 앵커가 가구의 빈 공간에 놓일 수 있으므로, 임의 형상에 맞는 파지 일반화는 별도 문제다.
CPU 물리 결과를 기존 GPU PhysX 학습 성능과 직접 비교하지 않는다. 시드 하나의 탐색 결과다.

## 경로

- 원시 대조군: `../../runs/jointcarry_zeroshot/box_carrywith_gpu6_cpu_physx/`
- 원시 테이블: `../../runs/jointcarry_zeroshot/round_table_carrywith_gpu6_cpu_physx/`
- [집계 JSON](../../runs/jointcarry_zeroshot/carrywith_summary.json)
- [높이·목표 거리 그래프](../../runs/jointcarry_zeroshot/carrywith_comparison.png): 실선은48장면 평균, 음영은25–75분위.

## 기타 실행과 실패

CPU 형상 낙하120스텝, box/table 각1환경32프레임 CPU 추론 통과.
바닥 접근 loco 대조군(`box_gpu6_cpu_physx_v3`)은 정상 종료했지만 목표 완료0%였다.
`round_table_gpu6_cpu_physx`의 시뮬레이션은 결과를 저장했지만 실행 중 wrapper 편집 실수로 최종 셸rc127이 발생했다. 유효 비교에서 제외하며 로그/수치는 보존한다.
초기 `box_cpu_smoke`, `box_gpu6_cpu_physx`, `_v2`의 장치 설정/평가 진단 장치 불일치 실패도 보존한다.
정상 carryWith 두 실행은 rc0. checkpoint 파일 SHA256 불변. illegal memory access는 관측되지 않았다.
