# Stage 1 relation skill learning

`approach_distance_edge_context_stage1`은 coordination 전에 HOLDING/AT/ON_TOP/SIT/CLIMB 실행 자체를 scratch로 학습하는 schema 5 실험이다. 기존 OnTop·interaction config와 checkpoint는 변경하지 않는다.

## 그래프와 binding

2 agents / 3 logical objects에서 `O0`, `O1`은 각 agent의 assigned object이고 `OX`는 free object다. 실제 physical box는 reset마다 무작위 배정되지만 observation과 graph는 logical 순서로 정렬된다.

| local pattern | 확률 | binding |
| --- | ---: | --- |
| HOLDING | .10 | `H_i -> O_i` |
| SIT | .10 | `H_i -> O_i` |
| CLIMB | .10 | `H_i -> O_i` |
| HOLDING_AT | .25 | `H_i -> O_i`, `O_i -> G_i` |
| HOLDING_ON_TOP | .15 | `H_i -> O_i`, `O_i -> OX` |
| HOLDING_SIT | .15 | `H_i -> O_i`, `H_i -> OX` |
| HOLDING_CLIMB | .15 | `H_i -> O_i`, `H_i -> OX` |

OX를 쓰는 pattern은 scene당 한 agent만 받는다. 상대 agent의 object는 target/support로 사용하지 않는다. active edge는 모두 current required goal이며 PRE/END link가 없다.

## Context와 reward

active edge packet은 `[valid,src,dst,relation,owner,start,keep]`이고 Stage 1에서 `start=keep=1`이다. invalid edge는 packet 전체가 0이다. START/KEEP auxiliary weight는 0이며 reward gate나 downstream saturation은 없다.

각 edge는 own current success만으로 포화한다.

```text
R_e = .2 * state + .2 * distance_progress + .2 * own_success
P(d_xy) = 1 / (1 + max(d_xy - .5, 0) / 1.0)
```

own success이면 세 항을 모두 1로 포화해 최대 `.6`이다. agent task reward는 자기 edge 합이며 teammate `.9/.1` sharing을 사용하지 않는다. 기존 power/collision/box-speed penalty와 PPO의 `.5 task + .5 AMP`는 유지한다.

19·20번의 SIT/CLIMB은 interaction에서 확정한 정의를 그대로 쓴다. 21·22번의 box-top SIT 변경은 아래 별도 절을 따른다.

```text
SIT target = object position + rotated [0,0,0.1381430834425038]
phi_sit = exp(-10 * ||root - target||^2)
success_sit = phi_sit >= .9

CLIMB root target = [object_xy, rotated_bbox_top_z + char_h]
phi_climb = exp(-10 * ||root - target||^2)
success_climb = phi_climb >= .9 AND |mean(feet_z)-surface_z| <= .05
```

feet height는 CLIMB dense reward에 섞지 않는다. SIT/CLIMB velocity/direction reward도 사용하지 않는다.

## 실행

```bash
# scratch training
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_train.sh 2 2048 3

# local viewer
CKPT='/absolute/path/to/ApproachDistanceEdgeContextStage1.pth'
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=holding_sit \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_test.sh "$CKPT" 2 1 3 10

# headless evaluation
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_stage1 \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_test.sh "$CKPT" 2 64 3 3

# server viewer
TOKENHSI_GPU=0 TASK_GRAPH=holding_climb \
  bash tokenhsi/scripts/multi_agent/approach_distance_edge_context_stage1_vnc.sh "$CKPT"
```

Preset은 `holding|sit|climb|holding_at|holding_ontop|holding_sit|holding_climb|random_stage1`이다. 고정 preset은 agent 0에 선택한 pattern, agent 1에 HOLDING을 주며 `TASK_ROLE_SWAP=1`로 역할을 바꾼다.

같은 schema 5/config checkpoint만 resume·평가할 수 있다. schema 3 OnTop과 schema 4 interaction checkpoint는 자동 호환하지 않는다. 짧은 검증은 `OUTPUT_PATH=output/approach_distance_edge_context_stage1_check MAX_ITERATIONS=2`처럼 본학습과 분리한다.

## 20번 단독 SIT/CLIMB 바닥 reset 비교

`approach_distance_edge_context_stage1_ground_sit_climb.yaml`은 19번에서 `box.reset.groundStandaloneSitClimb: true`만 추가한다. 그래프를 먼저 뽑은 뒤, 단독 SIT/CLIMB edge가 자기 `O_i`를 가리키는 agent에 한해 매 reset 상자 중심을 `box_height/2`로 놓고 그 출발 선반을 비활성화한다. `holding_sit`·`holding_climb`의 SIT/CLIMB 대상 `OX`는 원래 바닥이다. HOLDING source의 선반, AT target 선반, ON_TOP source 선반과 `OX` 받침 상자, reward와 AMP는 유지한다. 20번 본학습은 사용자 선택에 따라 19번 checkpoint를 로드하지 않는 scratch다. 명령은 [config.md의 20번](config.md#stage-1-단독-sitclimb-바닥-20번)을 따른다.

## 21번 공통 박스 크기

`approach_distance_edge_context_stage1_common_boxes.yaml`은 20번 위에 X/Y 각각 0.40~0.65m, Z 0.25~0.55m의 0.05m 독립 크기 샘플을 추가한다. 크기는 physical asset 생성 시 환경·상자별로 정해지고 reset 중에는 바뀌지 않는다. 19·20번 고정 크기와 달리 크기를 새로 뽑으려면 환경 재생성이 필요하다. Stage 1 그래프는 두 상자 적층까지만 허용하므로 높이 예산도 가장 높은 두 상자 합으로 검사한다. 다른 실험의 3상자 검사에는 영향을 주지 않는다.

평평한 박스에 원본 의자 `tarSitPos` offset을 적용하면 목표 pelvis가 윗면보다 낮아지는 문제를 이 실험에서 수정한다.

```text
SIT target = [box_center_xy, rotated_bbox_top_z + 0.12 m]
phi_sit = exp(-10 * ||root - target||²)
success_sit = phi_sit >= 0.9
```

SIT의 XY progress, reward 가중치·포화와 CLIMB의 목표·feet 성공 조건은 그대로다. 기존 sampler 확률과 물체 무작위 할당은 유지하며 크기별 가능성 필터는 없다. 제안 크기 전 구간에서 행동 성공은 미검증이다. reward config가 다른 19·20번 checkpoint와는 호환하지 않고 scratch로 시작한다. 실행은 [config.md의 21번](config.md#stage-1-공통-박스-크기-21번)을 따른다.

## 22번 고정 박스 SIT 목표 수정

`approach_distance_edge_context_stage1_fixed_boxes_sit_fix.yaml`은 20번과 같은 고정 0.5×0.5×0.4m 박스·단독 SIT/CLIMB 바닥 reset·Stage 1 그래프·AMP·CLIMB reward를 쓰고, **SIT 목표만** 21번의 박스 윗면 + 0.12m로 변경한다. 바닥 상자의 SIT target Z는 0.52m다. 20번 checkpoint는 reward 정의가 달라 사용할 수 없다. 21번과 reward 정의는 같지만 크기 분포가 다르므로 checkpoint를 섞지 않고 별도 output에서 scratch로 학습한다. 실행은 [config.md의 22번](config.md#stage-1-고정-박스-sit-수정-22번)을 따른다.

## 23번 단일 edge + relation RSI

21번의 가변 크기 박스·box-top SIT target·CLIMB target/success·combined AMP·own-only reward를 유지하면서, agent마다 `HOLDING`, `SIT`, `CLIMB` 중 정확히 **하나**만 독립적으로 1/3 확률로 샘플한다. 모두 자기 `O_i`를 목표로 하고 `OX`는 방해물이다. graph packet은 기존 4-slot/7-field이므로 active 2개와 padding 2개다. PRE/TERM·AT/ON_TOP은 이 단계에 없다.

초기 모션 분포는 graph relation을 먼저 뽑고 그에 맞춰 `HOLDING: loco .5 + pickUp .5`, `SIT: loco .5 + sit .5`, `CLIMB: loco .5 + climb .5`를 선택한다. `carryWith`는 AMP에는 포함하되 RSI에서는 우선 끈다. 같은 seed 2048환경 단기 대조에서 `carryWith` 시작의 상자 속도 페널티가 커졌고, 박스를 원본에 가까운 0.4m 고정으로 해도 남았기 때문이다. 원본 TokenHSI motion ID/time sampling과 SIT/CLIMB 물체 pose의 XY/yaw를 재사용한다. 다만 원본 정적 chair/climb 물체를 그대로 만드는 게 아니라 21번 동적 박스를 해당 pose에 놓고 높이는 실제 박스의 바닥 중심으로 맞춘다. 초기 모션과 박스가 겹치거나 발이 지면 아래인 후보는 물리 reset 경로에서 재시도한다. 재시도 중 다른 RSI clip 또는 loco가 선택될 수 있다. 이 필터는 기하학적 휴리스틱이며 장기 물리 안정성 보증은 아니다.

schema 6으로 checkpoint를 21/22번과 분리한다. `relation/sampling/rsi_sit|climb_attempts`, `...rejection_rate`, `...start_target_distance`, `sampling/physical_retries|failures`를 보고 크기별 부적합 여부를 판단한다. viewer/eval은 기본 loco-only 출발이므로 RSI 도움과 별개로 처음부터 배운 행동을 확인한다. 명령은 [config.md의 23번](config.md#stage-1-단일-edge와-relation-rsi-23번)을 따른다.

## 24번 CLIMB-only, context 없는 RSI

23번의 박스 크기·바닥 reset·CLIMB root target·own-only edge reward 구조를 유지한다. 그래프 sampler의 CLIMB 확률을 1로 고정해 두 agent가 각자 자기 상자만 오른다. 정책에는 5-field `[valid,src,dst,relation,owner]` semantic packet만 넣으며 START/KEEP 필드·context 인코더·context reward는 없다. schema 7과 별도 checkpoint/output을 사용한다. AMP 시연 분포는 원본 CLIMB 설정에 가까운 loco/climb/climbNoRSI `.3/.4/.3`이고 RSI는 loco/climb `.5/.5`다.

```text
r_valid = sqrt(box_x² + box_y²)/2 + 0.3 m
P = 1/(1 + max(d_xy - r_valid, 0)/1.0)
phi = exp(-10 ||root - [box_xy, box_top_z + char_h]||²)
success = (phi >= 0.6) AND (|mean(feet_z) - box_top_z| <= 0.07 m)
R_edge = 0.2 P + 0.2 phi before success; 0.6 while current success
```

feet dense 보상은 추가하지 않는다. 0.6/7cm 성공과 0.7/5cm 엄격한 진단을 함께 기록하고, 양발 개별 높이 오차 최대값도 기록한다. 발 평균만으로 착지·양발 접촉을 보증하지 않는다. 명령은 [config.md의 24번](config.md#stage-1-climb-전용-무context-24번)을 따른다.
