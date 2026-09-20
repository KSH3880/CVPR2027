# OnTop 혼합학습 config

[설정 파일](../tokenhsi/data/cfg/multi_agent/approach_distance_success_ontop_mixed.yaml)은 `state_relation_ontop_mixed_v1` 혼합학습에 연결되어 있다. 전용 train/test/VNC 스크립트와 복사 가능한 실행 명령은 [config.md의 15번](config.md#ontop-혼합학습-15번)에 모았다.

## 확정 설정

| 항목 | 설정 |
| --- | --- |
| 총 환경 / 사람 / 물체 | 2048 / 2 / 3 |
| carry + carry | 512환경 |
| carry + Ox에 독립 OnTop | 768환경 |
| carry + Oa에 의존 OnTop | 768환경 |
| 학습 방식 | 하나의 공유 정책, 처음부터 동시학습, 환경 그룹·비율 고정 |
| 역할 | reset마다 사람과 a/b 역할을 무작위 연결, episode 안에서는 유지 |
| 초기화·물리·AMP | 기존 `approach_distance_success` 설정 유지. 쌓인 상태로 초기화하지 않음 |
| Ox | 독립 OnTop의 고정 받침. reset 때 위치 재배치, Ox용 At edge 없음 |
| 상태 계수 | Holding 5, At 10, OnTop 10 |
| 성공 | 각자 terminal edge 상태값 ≥ 0.9, 상태 입력 Z 오차 ≤ 0.001 m |
| 성공 보상 | 기존처럼 자기 edge 포화 + 매-step 0.2. B 성공에 A 성공 조건 추가 없음 |
| 종료 | 기존 600 step·넘어짐 조건 유지, 성공으로 조기 종료하지 않음 |

기존 상자 크기·회전·높이·skill reset 분포는 YAML에 그대로 복사했다. Ox는 기존 플랫폼처럼 중력을 끄고 큰 질량과 속도 제한을 적용해 reset 가능한 고정 받침으로 만든다. GPU PhysX의 static collision shape는 root tensor로 재배치할 수 없어 `fix_base_link` 방식은 사용하지 않는다. 독립 시나리오에서는 Ox를 사람의 운반 상자로 배정하지 않는다. 이미 쌓인 초기 상태는 재샘플링한다. 기존의 다양한 상자 크기·높이에 대한 장기 쌓기 성능은 아직 검증하지 않았다.

## 입력과 gate

```text
Holding state: 양손 평균 위치 -> 자기 상자 중심
Holding progress: 사람 root -> 자기 상자 중심 (XY)
At state: 자기 상자 중심 -> goal 좌표
At progress: 자기 상자 중심 -> goal 좌표 (XY)
OnTop state: 위 상자 바닥면 중심 -> 받침 상자 윗면 중심
OnTop progress: 위 상자 중심 -> 받침 상자 중심 (XY)

phi = exp(-k * squared_distance(state_source, state_target))
P = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
g = sigmoid(30 * (previous_step_phi - 0.8))

G_Holding = 1
G_At = g_own_Holding
G_independent_OnTop = g_own_Holding
G_dependent_OnTop = min(g_own_Holding, g_other_At)

r_edge_before_success = G * (0.2 * phi + 0.2 * P)
```

면 중심은 box local offset에 실제 회전을 적용해 계산한다. OnTop의 target은 실제 받침 상자를 따라가며 Ga로 대체하지 않는다. 선행 gate는 shaping에 적용하고, 로컬 성공 시 기존 포화 지급은 gate를 덮어쓴다. 따라서 A가 목표에 오기 전 B가 쌓아 성공할 가능성은 허용하는 설계다.

## checkpoint 전이

```text
/home/hwanhee/ksh/approach_distance_success/output/approach_distance_success/ApproachDistanceSuccess_18-14-22-43/nn/ApproachDistanceSuccess_00018000.pth
```

실제 파일의 epoch 18000, Holding k=5를 확인했다. actor/critic relation embedding은 각각 `[8, 32]`다.

- 호환되는 모델 가중치·정규화 통계를 가져오고, 기존 학습 가능 파라미터를 추가로 freeze하지 않는다.
- actor/critic 모두 기존 relation 행을 복사하고 OnTop 행을 새로 초기화한다.
- optimizer·epoch/frame·환경 상태는 복원하지 않는 새로운 실험이다.
- 새 학습에서는 config의 checkpoint를 자동 전이한다. 경로 변경은 `TRANSFER_CHECKPOINT`를 사용한다. `transfer_report.json`에 177개 텐서 복사와 actor/critic 두 embedding의 확장을 기록한다. 예상 밖의 키·크기·기존 reward 차이는 거부한다. `RESUME_CHECKPOINT`는 이미 학습한 OnTop 모델의 optimizer·카운터를 포함한 전체 resume에만 사용한다.

## 정책 입력과 구현 범위

기존 Human223 / Object30 / Target1 tokenizer와 5차원 edge attribute encoder, Transformer 및 action/value head를 유지한다. 관측 끝에는 기존 18차원 relation 상태에 시나리오·A 역할의 사람 슬롯 번호 2개를 추가한다. 이 두 값은 정규화하거나 새 MLP에 넣지 않고, 저장된 rollout의 그래프·사용하지 않는 goal mask를 복원하는 데만 사용한다. 따라서 PPO minibatch 순서나 reset 이후의 실시간 역할 변경에 영향을 받지 않는다.

추가로 초기화하는 학습 파라미터는 actor/critic 각각의 OnTop embedding 행이다. 전체 token concat 인코더를 새로 만들지 않는다. 현재 실행 범위는 사람 2·물체 3·세 시나리오이며, 더 많은 사람과 임의의 관계 그래프는 후속 확장이다.

## 진단과 검증

- TensorBoard `relation/06_scenarios/`에서 시나리오별 현재 A/B 성공, 공동 성공·최종 성공·유지, B gate/phi와 A 역할의 사람0 비율을 확인한다.
- 의존 OnTop은 B 첫 로컬 성공 시 실제 Oa–Ga 거리를 기록한다. A 성공은 이 기록이나 B 보상의 추가 조건이 아니다. 성공 표본 수 0과 실제 거리 0을 구별할 수 있도록 표본 수도 제공한다.
- 평가 JSON에도 시나리오별 최종 공동 성공·유지·B 첫 성공 거리를 기록한다. `reward_terms`는 At/OnTop 상태·progress를 분리한다. 수정 전부터 실행 중인 학습은 다음 실행/resume 전까지 기존 통합 At 이름을 유지한다. 별도의 기존 `relation/.../at_*`·CSV 진단 슬롯은 각 사람의 terminal At/OnTop을 나타내므로 시나리오별 지표를 함께 본다.
- CPU 검증: carry 그룹의 기존 reward와 완전 일치, 이전 step gate·성공 포화, 회전된 면 좌표, 부분 reset, 역할 교환·batch 순서 변경의 정책 일관성, 미사용 goal 마스킹, 전이와 신규 행 gradient, 진단 기록을 확인한다.
- GPU 4에서 2048환경 전이 학습(epoch 2)·resume(epoch 3)·checkpoint 저장, TensorBoard scalar 167개 유한값, 16환경 평가, 저장 모델의 64 step 시뮬레이션 검증을 수행했다. 고정 Ox pose 유지·reward/history·관측 suffix·actor/critic batch 재배열 일치를 확인했다.
- 장기 본학습과 OnTop 수렴은 실행·검증하지 않았다. 짧은 확인 결과는 `output/approach_distance_success_ontop_mixed_check/`에만 저장한다.
