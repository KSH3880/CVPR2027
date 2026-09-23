# SIT/CLIMB edge context (18번)

17번 sampled OnTop을 보존하고 `state_relation_edge_interaction_v1`, schema 4로 분리한 scratch 실험이다. relation ID는 기존 6/7/8을 유지하고 `SIT=9`, `CLIMB=10`을 추가한다. packet은 계속 `[valid, src, dst, relation, owner, pre, term]`, edge context는 `[q_pre,q_term]`이다.

## Reward와 success

모든 edge는 `0.2 × progress + 0.2 × state + 0.2 × success`를 쓴다. own current success 또는 TERM target의 current success가 참이면 세 성분이 모두 1로 포화되어 edge당 0.6이다. PRE는 정책 context일 뿐 reward gate가 아니다.

공통 progress는 다음과 같다.

```text
P(d_xy) = 1 / (1 + max(d_xy - 0.5, 0) / 1.0)
```

SIT target은 원본 `humanoid_sit.py`처럼 object-local `tarSitPos`를 quaternion으로 world 변환한다. toy box에는 object별 metadata가 없으므로 연결된 TokenHSI train sit object 49개의 중앙값을 config에 공개했다.

```text
tarSitPos_median = [0, 0, 0.1381430834425038]
phi_sit = exp(-10 * ||root - sit_target||²)
success_sit = (phi_sit >= 0.9)
```

SIT progress 거리는 root와 변환된 sit target의 XY 거리다. 속도·각속도 penalty는 relation reward에 넣지 않는다.

CLIMB 데이터의 `tarClimbPos.z`는 38개 train object에서 bbox 반높이와 중앙값 오차가 0이었다. 따라서 회전 bbox의 world top을 원본 surface target으로 사용한다.

```text
z_surface = object_center_z + rotated_vertical_half_extent
climb_target = [object_center_x, object_center_y, z_surface + char_h]
phi_climb = exp(-10 * ||root - climb_target||²)
feet_error = |mean(left_foot_z, right_foot_z) - z_surface|
success_climb = (phi_climb >= 0.9) AND (feet_error <= 0.05 m)
```

CLIMB progress는 root와 object center의 XY 거리다. feet 값은 success 검증에만 쓰며 dense state/reward에는 들어가지 않는다. 원본 direction velocity, 1.5m/s target, velocity penalty도 사용하지 않는다.

## Graph sampling

agent별 최종 확률은 `HOLDING .10 / SIT .10 / CLIMB .10 / HOLDING_AT .25 / HOLDING_ON_TOP .20 / HOLDING_CLIMB .15 / HOLDING_SIT .10`이다. O=3 toy에서 두 사람이 동시에 SIT/CLIMB support를 소비하지 않도록 두 agent의 interaction 여부만 음의 상관으로 joint sample한다. 각 agent의 위 marginal 확률은 그대로 유지한다.

- standalone SIT/CLIMB은 자기 free assigned object, 준비된 상대 object, Ox 중 유효 target을 균등 선택한다.
- HOLDING+SIT/CLIMB은 들고 있는 자기 object를 target으로 쓰지 않는다.
- 상대 object를 쓰려면 상대 AT/ON_TOP placement edge가 있어야 하며 그 edge를 PRE에 자동 추가한다.
- HOLDING+AT/ON_TOP은 기존처럼 terminal relation 성공 뒤 release가 허용된다.
- HOLDING+SIT/CLIMB은 두 edge가 모두 required goal이고 TERM이 없다. 즉 물체를 든 상태로 앉거나 올라야 한다.
- reward에는 PRE를 곱하지 않으며 task reward 0.9/0.1 sharing과 기존 penalty/AMP/PPO 경계는 유지한다.

## AMP와 checkpoint

`dataset_loco_sit_carry_climb.yaml`을 사용한다. reset은 loco pose만 사용해 box/reference mismatch를 피하고, discriminator는 loco/sit/climb/omomo/pickUp/carryWith/putDown을 샘플한다. task relation과 AMP skill은 1:1로 묶지 않는다.

relation embedding은 11행이고 schema 4 metadata를 요구한다. schema 3 OnTop checkpoint는 embedding 크기와 계약이 다르므로 직접 resume/evaluate하지 않는다.

학습 sampler는 agent당 2 edge지만 [3-edge explicit graph](../../../tokenhsi/data/cfg/multi_agent/graphs/edge_interaction_three_edges.yaml)는 평가 override로 컴파일된다.

## Viewer

1환경 viewer는 `hold_sit` 기본이다. 아래 고정 preset을 차례로 보는 것이 디버깅에 적합하다.

```text
sit_only, climb_only, hold_sit, hold_climb,
at_then_sit, at_then_climb, ontop_then_climb
```

고정 preset 확인 뒤 `TASK_GRAPH=random`을 쓰면 실제 학습 graph를 episode별로 볼 수 있다. 분포 평가는 화면 없이 여러 환경에서 수행하는 편이 낫다. `TASK_ROLE_SWAP=1`은 고정 preset의 역할을 뒤집는다.
