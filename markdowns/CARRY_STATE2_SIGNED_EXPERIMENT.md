# State2 + signed linear progress 실험

사용자 합의에 따라 state2의 설정을 유지하고 공통 progress 함수만 변경한 별도 실험이다. 기존 v0/state2의 기본 보상 정의, config, output은 유지한다. 이 문서는 v0 전체 spec을 대체하지 않는다.

## 변경 범위

- 프로파일: `RELATION_VARIANT=state2_signed`
- env config: `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state2_signed.yaml`
- 기존 state2 YAML과 파싱 결과의 차이는 `env.relationReward.progress.mode: signed_linear` 하나뿐이다.
- state delta weight=2.0, progress coefficient=0.2, soft gate, Holding/At phi, 성공 bonus/완료 처리, AMP/penalties, observation/RMS/network는 유지한다.
- 공통 XY 속도 투영은 기존처럼 source의 변위 / control dt와 post-step 목적지 방향을 사용한다. moving target의 상대 거리 감소율로 바꾸지 않는다.

```
v_parallel = dot((source_next.xy - source_prev.xy) / dt,
                 normalize(target_next.xy - source_next.xy))
P = clamp(v_parallel / 1.5, -1, 1)

R_H = 2 * delta_phi_H + 0.2 * (1 - g_H_previous) * P_H_to_O
R_A = g_H_previous * (2 * delta_phi_A + 0.2 * (1 - g_A_previous) * P_O_to_G)
```

위 식은 완료 전의 관계 보상이다. 완료 시 한 번의 bonus 5, 완료 후 해당 관계 보상 0, AMP 및 regularizer 지속 등 기존 처리도 그대로다. 목적지 XY 거리가 normalization epsilon 이하이면 P=0이다. 추가 거리 cutoff, pinning, 감속 mask, 유지 보상은 없다.

1.5는 signed 모드에서는 선호 속도의 peak가 아니라 정규화/포화 속도다. `velocity_scale: 5.0`은 state2와 config를 일치시키기 위해 남겨 두었으나 signed 모드에서는 사용하지 않는다.

`progress.mode`가 없거나 `gaussian`이면 기존 Gaussian 계산을 그대로 사용한다. signed 모드에서만 progress의 검증 범위를 [-1,1]로 넓혔고 phi는 여전히 [0,1]로 검증한다. 관측의 phi/gate/history에도 음수 progress를 추가하지 않는다.

## 학습 실행 — scratch

레포 루트에서:

```bash
RELATION_VARIANT=state2_signed \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh
```

기본 설정:

- env 2048 / minibatch 16384 / mini-epochs 6
- agents 2 / objects 3
- seed 9896 (기존 비교 실험과 동일)
- state2 checkpoint 이어학습이 아니라 state2 **설정 기반의 새 scratch 학습**
- train YAML은 기존 공통 `amp_ma_carry_relation.yaml`을 사용

저장 위치:

```
output/ma_carry_relation_state2_signed/CarryRelationState2Signed_<timestamp>/
```

500 epoch마다 번호가 붙은 checkpoint를 별도로 보존한다. `CarryRelationState2Signed.pth`는 해당 run 안의 갱신용 파일이다. 기존 v0/state2 경로를 덮어쓰지 않는다. `OUTPUT_PATH`, `SEED`, `RESUME_CHECKPOINT` 등을 외부 환경변수로 지정하면 스크립트가 그 값을 따르므로 scratch 기본 실행에서는 별도로 지정하지 않는다.

## 뷰어

signed checkpoint는 반드시 같은 프로파일로 연다.

```bash
RELATION_VARIANT=state2_signed HEADLESS=0 \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh \
  output/ma_carry_relation_state2_signed/CarryRelationState2Signed_<timestamp>/nn/CarryRelationState2Signed_00000500.pth \
  2 1 3 10
```

`<timestamp>`는 실제 run 폴더 이름으로 대체한다. state2와 signed는 reward config가 다르므로 checkpoint 상호 로딩은 기존 strict 검사에서 거부된다. 이 검사를 우회하는 warm-start 기능은 추가하지 않았다.

기존 공용 test 스크립트가 headless 옵션을 추가할 때 seed 인자를 지우던 부분도, 인자를 보존하도록 수정하고 실행 인자 테스트로 확인했다.

## 검증

- 기존 Gaussian과 signed의 수치/방향/좌표변환/XY-only/거리-zero/포화 테스트.
- 음수 progress의 gate 적용, state2의 delta weight=2 유지, 완료 후 억제, phi의 음수 거부.
- 신규 YAML이 state2에서 mode 하나만 다른지 확인.
- variant별 output/name/seed 분리와 train/test 실제 CLI 인자 검증. 이 인자 검사는 Python 실행을 대체하여 학습을 시작하지 않는다.
- 동일 reward config의 checkpoint metadata 허용, state2 ↔ signed의 교차 로딩 거부.
- 실제 Isaac Gym: env32 / M2 / O3 / seed9896, 미학습 정책으로 64 transitions. 음수 progress 3,246건 포함, state/progress 항과 pure tensor 계산 일치, reset/history/observation/clipping/완료 비종료 검사 통과.
- 실제 장기 학습이나 행동 성능 검증은 하지 않았다. 본 실험 학습은 사용자가 위 명령으로 시작한다.

전체 회귀 테스트: **68 passed**.

```bash
PYTHONPATH=tokenhsi /home/cvlab/anaconda3/envs/tokenhsi/bin/python -m pytest -q \
  tokenhsi/tests/test_relation_reward.py \
  tokenhsi/tests/test_relation_runtime.py \
  tokenhsi/tests/test_relation_variants.py \
  tokenhsi/tests/test_ma_relation_policy.py \
  tokenhsi/tests/test_ma_scene_policy.py \
  tokenhsi/tests/test_ma_scene_features.py
```

## TensorBoard 해석과 비교 기준

- `relation/progress_holding`, `relation/progress_at`: 새 P의 평균이므로 음수가 가능하다.
- `reward_terms/holding_velocity`, `reward_terms/at_velocity`: 기존처럼 coefficient와 gate가 적용된 값이고, 이제 역방향에서 음수가 가능하다. 최종 task weight 0.5 적용 전이다.
- state2의 state delta 항은 계속 2배다. `phi`/`satisfied`는 가중치 적용 값이 아니다.
- 기존과 reward 평균의 스케일/분포가 달라진다. 총 reward 상승만으로 성공을 판단하지 않는다.
- 같은 학습량에서 배정 박스 도달률, Holding 유지 중 목표 거리 감소, Holding 유지율/낙하, 목표 통과·왕복, 실제 목표 달성을 비교한다.
- 음수 운반 progress를 피하기 위한 gate 저하/놓기, 속도 포화로 인한 과속, 이동 target에 대한 잔여 Holding progress 등은 남는 위험이다. 이 구현이 해결을 보장하지 않는다.
