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

19·20번의 SIT/CLIMB은 interaction에서 확정한 정의를 그대로 쓴다. 21번의 box-top SIT 변경은 아래 별도 절을 따른다.

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
