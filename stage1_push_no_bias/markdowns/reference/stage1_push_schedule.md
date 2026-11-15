# Stage1 PUSH 고정 epoch 스케줄

`stage1_push_schedule`은 새 scratch 학습용이다. 기존 `stage1_push`의 보상·RSI·물리·PPO·900step·2명/2048환경/4물체를 유지하고 샘플링만 아래처럼 바꾼다. 실행 명령은 [config](../config.md)에 있다.

| 행동 | 0–30,000epoch | 35,000epoch | 40,000epoch 이후 |
| --- | ---: | ---: | ---: |
| AT | 20% | 22.5% | 25% |
| ON_TOP | 15% | 17.5% | 20% |
| SIT | 10% | 7.5% | 5% |
| CLIMB | 30% | 27.5% | 25% |
| PUSH | 25% | 25% | 25% |

30,000부터40,000까지 선형 보간하고 이후 고정한다. eval 결과·성공률·보상에 따른 조건은 없다. `max_epochs: -1`로 본학습의 epoch 종료를 해제했고 자동 eval·성공률 종료는 추가하지 않았다. 사용자의 `CHECK=1`·`MAX_ITERATIONS`는 검사용 명시적 제한으로 동작한다.

## 고정 자산과 실제 과제 샘플링

기존처럼 물체 크기와 PUSH 슬롯은 환경 생성 때 고정한다. 새 실험은 초기 비율의 largest-remainder 개수를 한 번 섞어 자산을 생성하므로 기본4096개 사람 슬롯 중1024개가 PUSH다. 기본2048환경의 초기/최종 과제 확률은 크기 호환성을 보존하면서 전체 자산 집단의 평균이 목표 비율과 일치하도록 보정한다. 두 확률 행렬을 선형 보간하므로 중간 epoch도 같은 목표를 따른다. 매 epoch 재설정하는 것은 이후 reset의 추출 확률이며 진행 중인 에피소드 과제는 유지한다. reset된 표본 비율은 유한 표본·환경별 reset 빈도에 따라 변동할 수 있다.

크기 호환 후보가 충분하지 않은 작은 viewer는 조건부 비율을 사용하고 ON_TOP 받침이 부족하면 기존 규칙대로 재샘플링한다. 고정 과제 viewer와 단독 평가의 강제 과제는 스케줄로 덮어쓰지 않는다. AMP demo의 carry/sit/climb/push family 비율도 같은 epoch의 목표 비율을 따르며 family 내부 전문가 혼합과 RSI 비율은 유지한다.

## 저장과 재개

스케줄 명세는 env와 PUSH checkpoint contract에 포함한다. PTH에는 `stage1_push_sampling_state`로 전역 epoch·진행률·목표 비율을 저장한다. resume은 저장 전역 epoch를 첫 reset 전에 적용하며 viewer/수동 eval도 PTH epoch의 분포를 복원한다. 예를 들어35000 PTH는 중간 비율을 사용한다. 기존 고정 분포 PTH와 스케줄 PTH의 혼용, 스케줄 변경 후 resume은 거부한다.

TensorBoard의 `relation/sampling/target_*`는 목표, `asset_expected_*`는 현재 고정 자산의 평균 추출 확률, `template_*`는 실제 reset 표본 비율이다. `schedule_epoch`·`schedule_fraction`으로 적용 시점을 확인한다. 이 스케줄은 학습률 warmup이 아니다.
