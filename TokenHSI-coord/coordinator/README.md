# Joint Coordinator

## 범위

현재 버전은 A=2 Carry의 토이 coordination만 다룬다. 입력은 refreshed simulator
state와 final goal뿐이다. GT trajectory, scenario id, `A가 양보` 같은 고정 후보 이름은
모델 입력이나 출력에 없다.

기존 결과를 덮어쓰지 않는 모델 트랙을 유지한다.

- **C1**: Transformer와 K=4 learned candidates를 쓰는 표현력 있는 기준 모델
- **B0**: 2-layer MLP가 lateral/slowdown 4개만 내는 최소 기준 모델
- **C2**: C1 residual path/smooth speed를 K=1로 줄인 collision-first 모델

둘은 frozen ms18 executor와 평가 protocol만 공유한다. 서로의 checkpoint로 초기화하거나
한 트랙의 모델/loss 변경을 다른 트랙에 섞지 않는다.

```text
actual A/B root·box·goal state
              ↓
     6 entity Transformer
              ↓
  K=4 learned candidate queries
              ↓
 joint path + smooth speed + pickup dwell
              ↓
time-aligned HH/BB/HB safety evaluation
              ↓
       selected joint future
              ↓
  320-point path + ms18 steer window
              ↓
        frozen ms18 executor
```

## 모델 출력

- `path_world`: `[B,4,2,33,2]`
- `speed`: `[B,4,2,33]`, `0.375..1.5m/s`
- `pickup_dwell`: `[B,4,2]`
- `candidate_value`: `[B,4]`
- `risk_logits`: `[B,4,3]` (`human-human`, `box-box`, `human-other-box`)

경로는 agent마다 `root→box→goal` 두 개의 cubic Bézier leg다. 0/16/32번 점은
root/box/goal로 hard anchor된다. 속도는 waypoint 33개를 독립 예측하지 않고 8개
bounded acceleration knot를 보간한 뒤

```text
v_next² = clip(v² + 2 a ds, v_min², v_max²)
```

로 적분한다. runtime에서도 `COORD_ACCEL_UP/DOWN`으로 replan 간 명령 변화를 한 번 더
제한한다.

## 검사

Isaac Gym 없이 실행된다.

```bash
cd TokenHSI-coord
python -m unittest discover -s coordinator/tests -v
```

무작위 초기화 PTH는 연결 smoke 전용이다. 이것을 성능 모델로 해석하면 안 된다.

```bash
cd TokenHSI-coord
python -m coordinator.init_checkpoint \
  --output ../runs/coord/checkpoints/c1_random_smoke.pth
```

## 로컬 viewer 연결

첫 positional 인자는 frozen ms18 executor tag/PTH이고 `COORD_CKPT`는 별도다.

```bash
COORD_CKPT=runs/coord/c1_analyticprior_s0/coord_c1_latest.pth \
COORD_REPLAN_STEPS=6 COORD_DRAW_CANDIDATES=1 \
MS_VIZ=7 ENVS=1 MS_CAM=top \
bash scripts/coord/view_local.sh ms18_maskteam_origscale_c06_s0
```

회색/청색/보라/녹색 얇은 선은 네 learned candidate이고, 기존의 밝은 바닥 띠는
실제로 선택되어 ms18에 전달된 경로다. top camera refresh 수정도 복사된 ms18 코드에
그대로 포함된다.

## frozen-ms18 closed-loop 학습

```bash
COORD_ITERS=200 COORD_ENVS=64 \
bash scripts/coord/train_local.sh c1_s0 ms18_maskteam_origscale_c06_s0
```

한 coordinator transition 동안 frozen ms18을 기본 6 action step 실행한다. PPO action은
네 후보의 control residual, acceleration knot, pickup dwell이며 optimizer에는 coordinator와
exploration std만 들어간다. 실제 rollout의 Carry reward, phase별 progress와 HH/BB/HB
collision cost를 사용하고, 경로 feasibility/diversity는 작은 auxiliary loss로 유지한다.
PTH와 `metrics.jsonl`은 `runs/coord/<tag>/`에 저장된다.

## B0 minimal 병렬 기준선

```text
actual A/B root·box·goal state
              ↓ flatten
       2-layer Tanh MLP(128)
              ↓
[A lateral, B lateral, A slowdown, B slowdown]
              ↓
analytic root→box + bowed box→goal path
              ↓
       frozen ms18 executor
```

B0에는 Transformer, learned candidate, risk head, path/diversity/feasibility auxiliary가 없다.
pickup leg는 항상 직선이고 carry leg만 최대 1m 옆으로 휠 수 있다. 속도는 평시 1.5m/s이며
감속 action만 허용한다. 공간상 속도는 17 waypoint에 걸쳐 선형 전이하고, runtime의 기존
temporal acceleration limit도 그대로 적용된다.

```bash
bash scripts/coord/simple_compare.sh \
  b0_mlp_col50_s0 ms18_maskteam_origscale_c06_s0
```

학습식은 의도적으로 아래 하나뿐이다.

```text
L_B0 = L_ppo + 0.5 L_value - 1e-4 entropy
r_B0 = r_ms18_carry + 2 phase_progress - 50 actual_collision - 0.01
```

체크포인트는 `coord_b0_*.pth`이고 C1의 `coord_c1_*.pth`와 schema부터 다르게 저장한다.
`scripts/coord/eval_one.sh`은 실행 sidecar의 `COORD_MODEL`을 읽어 올바른 loader를 선택한다.

B0 최고 점수 checkpoint는 다음처럼 본다.

```bash
COORD_MODEL=simple \
COORD_CKPT=runs/coord/b0_mlp_col50_s0/coord_b0_000100.pth \
MS_SCEN=cross MS_VIZ=7 ENVS=1 MS_CAM=top \
bash scripts/coord/view_local.sh ms18_maskteam_origscale_c06_s0
```

Cross 경로가 공간적으로만 교차하는 것이 아니라, 직선·평속과 측정된 ms18
approach/carry 속도 및 pickup dwell 기준으로 **같은 시각에 교차점에 도착하는** 장면만
보고 싶으면 viewer 전용 옵션을 켠다. 학습과 batch eval 초기화에는 적용되지 않는다.

```bash
MS_VIEW_TIMED_CROSS=1 MS_VIEW_TIMED_CROSS_TOL=0.25 \
COORD_CKPT=runs/coord/c5_r2_s0/coord_c2_000700.pth \
MS_SCEN=cross MS_VIZ=7 ENVS=1 MS_CAM=top \
bash scripts/coord/view_local.sh ms18_maskteam_origscale_c06_s0
```

기본값은 모든 viewer reset에 적용(`MS_VIEW_TIMED_CROSS_PROB=1.0`)한다. 로그의
`[coord-view timed-cross]` 행에서 예상 교차 도착시간 차이를 확인할 수 있다.

B2는 같은 MLP/loss에서 carry lateral bow만 현재 box→goal 거리 이하로 제한한다. 급곡선
invalid가 거의 없어 실제 bridge 기본 추천이다.

```bash
COORD_MODEL=simple \
COORD_CKPT=runs/coord/b2_mlp_safebow_col50_s0/coord_b0_000100.pth \
MS_SCEN=cross MS_VIZ=7 ENVS=1 MS_CAM=top \
bash scripts/coord/view_local.sh ms18_maskteam_origscale_c06_s0
```

## C2 K=1 collision-first

C2는 C1의 state-token Transformer와 33점 hard-anchor trajectory를 유지하되 learned
candidate를 하나로 줄인다. 실제 자유도는 agent별 두 Bézier leg의 네 control residual과
8 acceleration knot이며, 출력은 기존과 같은 33-waypoint path/speed다. pickup residual은
집기 안정성을 위해 직선으로 고정하고 속도는 approach부터 조절할 수 있다. learned dwell,
risk loss, diversity와 selector 학습은 사용하지 않는다.

```text
L_C2 = L_ppo + 0.5 L_value - 1e-4 entropy
     + 50 future_collision + 0.5 speed_smooth + 0.1 unnecessary_slow
r_C2 = r_ms18_carry + 2 phase_progress - 50 actual_collision - 0.01
     - 1 invalid_plan
```

`future_collision`은 예측 path/speed를 시간 정렬해 계산한 HH/BB/HB 거리 침범이다.
`unnecessary_slow`는 같은 경로의 1.5m/s counterfactual이 안전한 경우에만 감속을 벌한다.
따라서 평시는 최고 속도를 유지하고 full-speed conflict에서는 감속이 허용된다.

frozen ms18의 실제 시간 반응은 별도 system-ID로 잰다.

```bash
MS_MEASURE_ENVS=48 MS_MEASURE_STEPS=600 MS_MEASURE_SEED=0 \
bash scripts/coord/measure_ms18_executor.sh \
  ms18_executor_sysid_v1_s0 ms18_maskteam_origscale_c06_s0
```

두 측정을 합친 `runs/coord_measurement/ms18_executor_sysid_v1/calibration.json`의 4점
명령→실속도 lookup과 phase별 pickup dwell은
`COORD_C2_MEASURED_EXECUTOR_TIMING=1`일 때 loss의 미래 시간축에만 적용된다. 실제 ms18에
보내는 속도를 remap하거나 executor PTH를 바꾸지는 않는다. 옵션 기본값은 0이다.

```bash
bash scripts/coord/c2_collision_compare.sh \
  c2_k1_collision_s0 ms18_maskteam_origscale_c06_s0
```

체크포인트는 `coord_c2_*`와 독립 schema를 사용한다. viewer에서는 반드시
`COORD_MODEL=c2`를 함께 지정한다.

학습식은 다음으로 고정했다.

```text
L = L_ppo + 0.5 L_value - 1e-4 entropy + 0.01 L_aux
r = r_ms18_carry + 2 phase_progress - 2 actual_collision - 0.01
    - 0.2 invalid_plan - 0.05 unsafe_plan

L_aux = soft_best_of_K
      + 0.05 nominal_speed
      + 0.05 acceleration_smoothness
      + 0.10 candidate_diversity
      + 0.10 risk_BCE
```

`soft_best_of_K`의 후보 비용은 `100*predicted_collision + makespan + 0.05*length
+ 0.25*speed_roughness`다. 다양성은 K=4 control residual의 pair distance가 0.5보다
작을 때만 벌점을 주므로 후보에 좌/우/감속 같은 semantic 이름을 강제하지 않는다.
전체 loss에서 다양성의 실효 계수는 `0.001`이다. 선택기는 critic의 보정되지 않은
candidate별 offset을 쓰지 않고 실제 decoded future의 안전성·시간·길이만 사용한다.
학습 로그의 `diversity_penalty`는 실제 후보 간 거리가 아니라 이 벌점이다. 값이 0에
가까우면 후보들이 최소 분리거리 0.5를 만족한다는 뜻이며, 실제 경로 다양성은 평가의
`candidate_diversity`로 확인한다.
선택할 때 invalid 후보는 hard ban(`1e9`)하지만, 학습할 때는 그 sentinel을 loss에 넣지
않고 같은 실제 비용에 곡률·버퍼 초과의 연속 penalty를 더한다.

충돌 조율을 먼저 보려면 같은 trainer에 cross 배치를 준다.

```bash
MS_SCEN=cross COORD_ITERS=200 COORD_ENVS=64 \
bash scripts/coord/train_local.sh c1_cross_s0 ms18_maskteam_origscale_c06_s0
```

두 agent를 모두 모델이 관리하고 감속/우회를 미리 정하지 않는 C7은 다음처럼 실행한다.

```bash
bash scripts/coord/c7_jointboth.sh \
  c7_jointboth_s0 ms18_maskteam_origscale_c06_s0
```

C7의 auxiliary는 `30 * proximity-risk + 0.2 * (speed-delay + detour-delay)
+ 0.1 * path-residual`이다. 두 delay의 합은 각 agent의 직선·평속 완료시간 대비 추가시간이며,
고정 priority·명시적 교차점/window·consistency는 사용하지 않는다. 실제 rollout PPO에는
기존 C2와 같은 `-50 * actual_collision`이 적용된다.

이어서 학습할 때는 이전 coordinator만 `COORD_INIT`으로 지정한다. 저장 step 번호와
optimizer/exploration 상태도 이어지며, ms18 executor는 여전히 별도로 frozen load된다.

```bash
COORD_INIT=runs/coord/c1_cross_s0/coord_c1_latest.pth \
MS_SCEN=free COORD_ITERS=100 COORD_ENVS=64 \
bash scripts/coord/train_local.sh c1_free_continue_s0 ms18_maskteam_origscale_c06_s0
```

## 현재 경계

- coordinator 모델, selector, checkpoint, ms18 inference bridge까지 구현되어 있다.
- `compute_auxiliary_loss()`는 gradient와 candidate diversity/safety 학습 신호를 제공한다.
- random checkpoint는 fallback/shape smoke만을 위한 것이다.
- closed-loop PPO runner는 2 env/1 iteration에서 실제 frozen ms18 rollout, update,
  numbered/latest PTH 저장을 통과했다. 저장 PTH를 다시 읽은 headless viewer도 599-step
  episode를 연속 완주했다.
- C1과 minimal B0/B1/B2의 500-iter 정식 학습, seed0/1 Free/Cross 비교를 완료했다.
- Track A 최종은 `c1_analyticprior_s0/coord_c1_latest.pth`다. collision reward `2→50`은
  성능을 개선하지 못해 폐기했다.
- Track B 점수 best는 `b0_mlp_col50_s0/coord_b0_000100.pth`, invalid<1%의 clean best는
  `b2_mlp_safebow_col50_s0/coord_b0_000100.pth`다. 둘 다 100 이후 성능이 개선되지 않아
  early stopping을 채택한다.
- B2-100의 seed0 Cross/Free는 success `0.715/0.923`, collision `0.119/0.288`, invalid
  `0.19%/0.14%`다. seed1 invalid도 `0.02%/0.36%`였다.
