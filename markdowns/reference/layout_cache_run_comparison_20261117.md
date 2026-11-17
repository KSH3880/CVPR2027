# Layout-cache 두 run 비교 및 초기화 문제 진단

작성일: 2026-11-17. 두 run의 저장 설정, checkpoint 전이 기록, TensorBoard scalar를 읽어 비교했다. 원인을 분리하는 추가 학습·평가는 수행하지 않았다.

## 비교 대상

1. `output_etc/check/layout_cache_20261117/benchmark/vectorized/Stage2Sensor900BridgeBeforeFixedLayoutCache_17-03-13-59`
   - [저장 설정](../../output_etc/check/layout_cache_20261117/benchmark/vectorized/Stage2Sensor900BridgeBeforeFixedLayoutCache_17-03-13-59/relation_config.yaml)
   - [TensorBoard 로그](../../output_etc/check/layout_cache_20261117/benchmark/vectorized/Stage2Sensor900BridgeBeforeFixedLayoutCache_17-03-13-59/summaries/)
2. `output/stage2_sensor_900_bridge_before_fixed_layout_cache/Stage2Sensor900BridgeBeforeFixedLayoutCache_17-04-02-46`
   - [저장 설정](../../output/stage2_sensor_900_bridge_before_fixed_layout_cache/Stage2Sensor900BridgeBeforeFixedLayoutCache_17-04-02-46/relation_config.yaml)
   - [TensorBoard 로그](../../output/stage2_sensor_900_bridge_before_fixed_layout_cache/Stage2Sensor900BridgeBeforeFixedLayoutCache_17-04-02-46/summaries/)

## 전이 원본과 설정 차이

두 run 모두 `output/stage2/Stage2Distill900V4_00002000.pth`에서 전이했다. 1번의 epoch 20 checkpoint와 2번의 epoch 1000 checkpoint에 저장된 `sensor_base`의 경로, epoch 2000, SHA256이 모두 일치한다.

```text
a1be127ad7118ca137f9de6079a9d4bb123a9d34a3c2c9c9aef24b15290f055a
```

두 run의 학습 계약은 `square_sensor_frozen_actor_coordination_v1`이다. Actor encoder·coordination·sigma는 동결하고, action head·critic 전체·센서 모듈·AMP 판별기는 학습한다. 2번 epoch 1000 checkpoint에서 원본 대비 action head 6/6개, critic 136/136개 파라미터 텐서가 실제로 바뀌었고, actor encoder·coordination 130개는 동일했다.

저장 설정의 차이는 다음과 같다.

| 항목 | 1번 run | 2번 run |
| --- | --- | --- |
| `env.stateInit` | `Random`: 모션 기반 RSI | `Default`: RSI 제거, 관절각·초기 속도 0 |
| BEFORE 배치 | crossing 100% | 같은 출발 무대 20% + crossing 80% |
| 다리 중심 추가 보상 | 없음 | crossing 선행에 거리 기반 보상 추가, 가중치 0.2 |
| 추가 보상 포화 | 해당 없음 | payload가 다리 중심을 통과하면 raw=1 유지; 같은 무대 20%는 제외 |
| `max_epochs` | 19, 실제 기록은 epoch 1~20 | 1,000,000 |

`before_training` 블록 추가와 `stateInit`, `max_epochs` 변경 외에는 두 저장 설정이 동일하다. 다리 경계 RSI는 fixed 구성에서 실행되지 않으며, 위 `Random`은 사람의 초기 자세를 모션에서 추출하는 RSI를 뜻한다.

## 동일한 첫 20 epoch 비교

각 scalar의 epoch별 기록을 산술 평균했다. `rewards0/iter`는 에피소드 누적 보상이므로 에피소드 길이의 영향을 받으며, crossing 성공률과 같지 않다.

| 지표 | 1번 epoch 1~20 | 2번 epoch 1~20 |
| --- | ---: | ---: |
| 에피소드 누적 보상 (`rewards0/iter`) | 23.6932 | −1.1850 |
| 에피소드 길이 (`episode_lengths/iter`) | 122.90 step | 2.33 step |
| AMP 보상 (`info/disc_reward_mean`) | 0.6098 | 0.2473 |
| FPS total | 17,753.6 | 11,480.2 |
| 캐시 적중률 | 99.25% | 99.00% |

2번은 학습 시작 직후부터 에피소드가 매우 짧았다. 두 run 모두 초기 캐시 적중률은 약 99%였으므로, 이 초기 동작 차이를 캐시 미스로 설명할 수 없다.

진단 시점의 2번 최근 20 epoch(1460~1479) 평균 누적 보상은 2.3303, 에피소드 길이는 29.52 step이었다. Crossing 에피소드 종료 중 선행 낙상은 약 35.6%, 다른 에이전트 낙상은 약 64.4%였다. 초기보다 길어졌지만 1번 첫 20 epoch의 생존 시간에는 미치지 못했다.

## 해석과 확인 우선순위

**확인된 사실:** 두 run은 같은 전이 원본을 사용했지만, 2번은 첫 20 epoch부터 생존 시간이 122.9→2.3 step으로 크게 짧아졌다. 1번 벤치마크는 RSI를 유지한 run이었다. 그 결과를 RSI 제거 후에도 유지되는 성능으로 해석하면 안 된다.

**우선 검증할 가설:** `Random → Default`로 바꾸면서 전이 정책에 익숙한 모션 자세 대신 관절각 0인 자세에서 시작하게 된 것이 초기 균형 실패나 초기 배치 문제를 일으켰을 수 있다. 20% 같은 무대 배치와 추가 보상도 동시에 바뀌었으므로 RSI 제거만을 원인으로 확정할 수는 없다.

Action head는 계속 학습되며 전이 원본 행동을 유지하는 별도의 모방 손실이나 원본 정책과의 KL 제약은 없다. 따라서 기존 동작의 망각도 가능하다. 다만 2번이 처음부터 불안정했던 점을 먼저 설명해야 한다. Critic loss 감소만으로 정책 동작이 정상이라고 판단할 수도 없다.

다음 확인은 전이 원본과 현재 모델을 동일한 Default 초기 상태에서 비교하고, 같은 정책에서 초기화 방식만 바꿔 비교하는 것이다. 초기 자세·접촉·높이와 첫 수 step의 종료 원인을 함께 확인하면 초기화 문제와 학습 중 동작 망각을 구분하는 데 도움이 된다.

별도로, 500 epoch 이후에는 연속 커리큘럼의 거리 값이 매 epoch 바뀌는데 캐시가 정확한 거리 일치를 요구하여 적중률이 약 40%로 떨어지는 성능 병목이 있다. 이 문제는 이후 FPS 저하를 설명하지만, 2번 첫 20 epoch의 즉시 종료 문제와는 구분해서 다뤄야 한다.
