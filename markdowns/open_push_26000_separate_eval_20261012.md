# OPEN/PUSH 26000 단독 평가 — 2026-10-12

## 조건과 완료 확인

같은 접촉+AT warmstart26000 정책을 OPEN/PUSH 두 평가로 분리했다. GPU6·headless·seed42, **각64환경×8반복=512장면/1024 agent-trial**이며 두 사람 모두 해당 평가의 과제만 수행한다. 두 실행 모두8반복과 최종JSON 저장 메시지를 확인했다. GPU2에서 중단된 실행과 앞선 혼합 평가는 포함하지 않는다.

checkpoint: `output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00026000.pth`

기존 평가 기본값인 보행 시작(`loco=1.0`), episodeLength600을 유지했다. PUSH는20kg·박스마찰0.6·정면2m목표이고, 후속 마찰1/랜덤목표/목표반대면 실험이 아니다.

## 주 결과: 학습 의도에 맞는 개인 목표 성공

| 항목 | OPEN 전용 | PUSH 전용 |
| --- | ---: | ---: |
| 종료 시 개인 성공 | **0/1024 (0%)** | **959/1024 (93.65%)** |
| 중간에 한 번이라도 개인 성공 | 0/1024 (0%) | 959/1024 (93.65%) |
| 에이전트0 최종 성공 | 0/512 (0%) | 480/512 (93.75%) |
| 에이전트1 최종 성공 | 0/512 (0%) | 479/512 (93.55%) |
| 비정상 종료(`terminate`) | 0/512 | 0/512 |

- OPEN: 손 접촉·자세·접촉에 따른 개방 credit을 만족하면서 목표각90°에서5초 지지를 완료해야 성공이다. 이번 정책은 이 조건을 한 번도 완료하지 못했다. 문을 전혀 움직이지 않았다는 결론은 아니며, 각도/접촉/지지시간별 실패 원인은 원본 집계로 구분할 수 없다.
- PUSH: AT 위치 만족도≥0.9·목표 높이오차≤1mm이면 성공이다. **목표 도달 후 손을 놓아도 성공**이며, 학습 보상도 `contact OR at_success`로 접촉 보상을 포화시킨다. 속도·기울기·종료 시 손 접촉을 이 성공에 추가 요구하지 않는다.

## 분석

PUSH는1024명 중959명이 목표 배치를 성공해93.65%다. 두 에이전트의 성공 수가480/479로 거의 같고, 반복별 성공률도90.62~99.22%로 유지된다. 이 표본에서 목표 배치는 잘 수행하지만65명은 성공하지 못했다. 실패 원인은 집계JSON만으로 단정하지 않는다.

OPEN은8반복 모두0%다. 따라서 이26000 정책은 현재 보행 시작과 엄격한 OPEN 완료 조건에서 문 열기 과제를 해결하지 못한다. 문90° 도달 여부, 손 접촉 유지, 자세 및5초 지지시간을 별도로 기록해야 병목을 찾을 수 있다.

PUSH의 개인 ever/final 성공 수가 모두959이므로 성공한 사람은 종료 시에도 목표 조건을 만족했다. 아래 접촉 장면 지표의 감소를 목표 배치가 무너졌다는 증거로 해석하면 안 된다.

## 참고: 기존 장면 지표의 조건 불일치

| 기존 원본 지표 | OPEN 전용 | PUSH 전용 |
| --- | ---: | ---: |
| 종료 시 장면 필수 edge 동시 성공 | 0/512 (0%) | 16/512 (3.12%) |
| 중간에 한 번이라도 장면 필수 edge 동시 성공 | 0/512 (0%) | 210/512 (41.02%) |

기존 평가기는 장면 성공에 PUSH 접촉 edge와 AT edge를 모두 요구한다. 학습은 AT 성공 후 release를 허용하므로 **PUSH3.12%는 학습 의도에 맞는 팀 목표 성공률이 아니다.** 이번 주 결과는 개인 목표93.65%다. release를 허용한 정확한 팀 동시 성공률은 원본JSON에 없으며, 두 사람의 목표 성공을 장면별로 함께 저장해야 계산할 수 있다. 평가 코드/학습 보상은 이번 정리에서 변경하지 않았다.

## 반복별 개인 성공률

| 반복 | OPEN | PUSH |
| --- | ---: | ---: |
| 1 | 0.00% | 96.09% |
| 2 | 0.00% | 92.19% |
| 3 | 0.00% | 94.53% |
| 4 | 0.00% | 92.97% |
| 5 | 0.00% | 90.62% |
| 6 | 0.00% | 91.41% |
| 7 | 0.00% | 92.19% |
| 8 | 0.00% | 99.22% |

## 원본과 재실행

- [OPEN 원본JSON](../output/open_push_26000_open_eval_gpu6_20261012/metrics/edge_context_20261012_153107.json) · [집계JSON](../output/open_push_26000_open_eval_gpu6_20261012/summary.json) · [로그](../output/open_push_26000_open_eval_gpu6_20261012/eval.log)
- [PUSH 원본JSON](../output/open_push_26000_push_eval_gpu6_20261012/metrics/edge_context_20261012_153213.json) · [집계JSON](../output/open_push_26000_push_eval_gpu6_20261012/summary.json) · [로그](../output/open_push_26000_push_eval_gpu6_20261012/eval.log)

```bash
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=open SEED=42 OUTPUT_PATH=output/open_push_26000_open_eval_gpu6_20261012 bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_test.sh output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00026000.pth 2 64 4 8
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push SEED=42 OUTPUT_PATH=output/open_push_26000_push_eval_gpu6_20261012 bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_test.sh output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00026000.pth 2 64 4 8
```
