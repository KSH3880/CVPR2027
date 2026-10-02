# Stage 2 실행 가이드

Stage 2 설정·학습·평가·VNC 명령은 이 문서에서 관리한다. Stage 1은 [config.md](config.md)를 따른다. 모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- Config·checkpoint·실행 옵션 변경 시 **현재 실험 표와 학습·로컬 평가·서버 VNC 명령을 함께 갱신**한다.
- 새 실험은 표에 한 행만 추가하고, 명령은 아래 기존 섹션에만 넣는다. **실험별 별도 섹션·중복 명령·설계 설명·실측 로그를 쌓지 않는다.** 상세 설정은 YAML, 변경·검증은 [changelog.md](../changelog.md), 코드 위치는 [structure.md](structure.md)를 따른다.
- 학습은 **2048환경**이다. 실행 전 GPU 점유를 확인하고 `TOKENHSI_GPU`를 지정한다. 기존 학습을 임의 종료·재시작하지 않는다.
- 새 학습은 `STAGE1_CHECKPOINT`, 이어 학습은 `RESUME_CHECKPOINT`를 사용한다. 본학습에는 확인용 `MAX_ITERATIONS`·`OUTPUT_PATH`·resume 잔여값을 제거한다.
- Stage 1 actor encoder·actor 관측 정규화는 고정하고 협력 attention·확장 action head·critic·AMP를 학습한다. Unified semantic/owner HOLDING·Rescue의 checkpoint는 각 실험 계약에 맞춰 사용한다.
- Unified semantic은 일반 unified pth 전용이며 owner HOLDING·shared edge encoder pth를 받지 않는다. Shared9 이어 학습·평가는 Shared9 pth를 사용한다.
- Shared9 정적 거리/CPA checkpoint는 이어 학습·평가에서 서로 섞지 않는다. CPA는 같은 Stage 1 epoch8000에서 새로 시작하며 별도 output을 사용한다.
- 평가의 `random_scenario`는 해당 config의 **학습 시나리오·비율을 유지**한다. 사람/물체 확장 때문에 다른 실험의 샘플러로 자동 대체하지 않는다.
- 평가 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. 로컬 viewer는 `HEADLESS=0`, 화면 없는 평가는 `HEADLESS=1`이다.

## 현재 실험

| 실험 | Config | Stage 1 출발점 | 사람/물체 | Scene 비율 |
| --- | --- | --- | --- | --- |
| **Rescue Shared9 CPA** | [approach_stage2_rescue_shared9_cpa.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_shared9_cpa.yaml) | Rescue epoch8000 | 학습2/4·평가O≥M≥2 | Shared9 동일·사람–사람 CPA만(0.5·0.7m·할인0.99) |
| **Rescue Shared9** | [approach_stage2_rescue_shared9.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_shared9.yaml) | Rescue epoch8000 | 학습2/4·평가O≥M≥2 | AT 포함3조합 각15%·공유6조합 각7.5%·독립10% |
| Rescue KLClimb50 | [approach_stage2_rescue_klclimb50.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_klclimb50.yaml) | Rescue epoch8000 | 학습2/4·평가O≥M≥2 | climb/sit/stack/독립30/30/30/10% |
| Unified semantic | [approach_stage2_unified.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_unified.yaml) | 일반 unified | 2/4 | 30/30/30/10% |
| Unified owner HOLDING | [approach_stage2_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_unified_owner_holding.yaml) | unified owner HOLDING | 2/4 | 30/30/30/10% |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 plane(28번) | 2/3 | 30/30/30/10% |
| SIT plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | Stage 1 self_sum(34번) | 2/3 | 20/20/40/20% |

Shared9는 전 과제 loco RSI·무조건부 AMP를 사용한다. 학습 상자 크기(cm)는 AT/독립 X/Y40~60·Z30~50, 공유 받침 X50~65·Y65~70·Z45~50, 공유 ON_TOP source X/Y30~40·Z40~50이며 5cm 간격이다. 학습 환경별 크기 풀은 고정되고 reset마다 풀 안의 조합·역할·배정이 바뀐다.

## 학습

**Shared9 CPA 본학습 명령:** epoch8000에서 새로 시작한다. 2048환경·기본 종료 조건1,000,000 epoch이며 output은 `output/approach_stage2_rescue_shared9_cpa`다.

```bash
TOKENHSI_GPU=0 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_train.sh 2 2048 4
```

Shared9 정적 거리 기준 실험의 본학습 명령은 다음과 같다. output은 `output/approach_stage2_rescue_shared9`다.

```bash
TOKENHSI_GPU=0 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_train.sh 2 2048 4
```

다른 실험은 필요한 한 줄만 실행한다. Unified·29·SIT plane의 Stage 1 경로는 해당 실험의 pth로 설정한다.

```bash
GPU=0
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_train.sh 2 2048 4
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT='/absolute/path/to/Stage1Unified.pth' bash tokenhsi/scripts/multi_agent/approach_stage2_unified_train.sh 2 2048 4
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT='/absolute/path/to/Stage1OwnerHolding.pth' bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_train.sh 2 2048 4
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT='/absolute/path/to/Stage1Plane.pth' bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
TOKENHSI_GPU="$GPU" STAGE1_CHECKPOINT='/absolute/path/to/Stage1SelfSum.pth' bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_train.sh 2 2048 3
```

짧은 확인·이어 학습은 해당 실험의 wrapper에 적용한다. 아래는 Shared9 예제다.

```bash
TOKENHSI_GPU=0 MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_rescue_shared9_check bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_train.sh 2 2048 4
TOKENHSI_GPU=0 RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2RescueShared9.pth' bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_train.sh 2 2048 4
TOKENHSI_GPU=0 MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_rescue_shared9_cpa_check bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_train.sh 2 2048 4
TOKENHSI_GPU=0 RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2RescueShared9CPA.pth' bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_train.sh 2 2048 4
```

Checkpoint는 `output/<실험>/<run>/nn/`, TensorBoard는 `summaries/`, 전이 기록은 `stage2_transfer_report.json`에 저장된다. 정기 checkpoint 간격은500 epoch다.

## 로컬 평가와 서버 VNC

`SHARED9_CKPT`·`SHARED9_CPA_CKPT`·`RESCUE_CKPT`·`CKPT`는 해당 실험의 Stage 2 pth다. `STAGE1_ONLY=1`은 Rescue/Shared9/CPA에서 epoch8000을 직접 불러오는 학습 전 baseline이다. 현재 viewer와 같은 test/VNC 스크립트·사람/물체 수·`TASK_GRAPH`·평가 옵션을 유지하고 checkpoint와 `STAGE1_ONLY`만 바꾼다. 시나리오·asset·보상은 Stage 2 config를 유지하며 Stage 1 정책/RMS를 불러오고 협력 입력의 action 기여를 0으로 만든다. 같은 초기 seed로 비교하려면 양쪽 명령에 같은 `SEED`를 지정한다.

```bash
GPU=0
STAGE1_CKPT=stage1/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth
SHARED9_CKPT=output/approach_stage2_rescue_shared9/ApproachStage2RescueShared9_02-15-13-35/nn/ApproachStage2RescueShared9.pth
SHARED9_CPA_CKPT=output/approach_stage2_rescue_shared9_cpa/ApproachStage2RescueShared9CPA_02-17-05-08/nn/ApproachStage2RescueShared9CPA.pth
RESCUE_CKPT=output/approach_stage2_rescue_klclimb50/ApproachStage2RescueKLClimb50_01-20-37-44/nn/ApproachStage2RescueKLClimb50.pth
CKPT='/absolute/path/to/matching_stage2_checkpoint.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_test.sh "$SHARED9_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_test.sh "$SHARED9_CPA_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_test.sh "$STAGE1_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_test.sh "$STAGE1_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 HEADLESS=0 TASK_GRAPH=ontop_ontop bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_test.sh "$STAGE1_CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_test.sh "$RESCUE_CKPT" 4 1 5 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_test.sh "$STAGE1_CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack bash tokenhsi/scripts/multi_agent/approach_stage2_unified_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_climb bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_climb bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_climb bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_test.sh "$CKPT" 2 1 3 10
```

화면 없는 평가는 `HEADLESS=1`·16환경, 짧은 확인은 `EPISODE_LENGTH=32`로 실행한다. Shared9/CPA test·VNC 기본은 **Shared9 9조합**이며 `random_scenario`는 1환경에서도 reset마다 다시 뽑는다. 2명·4상자 비율은 학습과 같은 AT 3조합 각15%·나머지6조합 각7.5%·독립10%다. 평가 인원은 O≥M≥2로 확장하며 무작위 2인 그룹·홀수 잔여자의 독립 과제로 구성한다. 상자가 부족하면 가능한 조합만 재정규화하고, 불가능한 고정 조합은 오류로 알린다. 모든 그룹에 9조합을 허용하려면 O≥3⌊M/2⌋+(M mod 2)를 사용한다(4명·6상자 등).

고정 조합은 `TASK_GRAPH`에 `at_climb`, `at_sit`, `at_ontop`, `ontop_ontop`, `ontop_climb`, `ontop_sit`, `climb_climb`, `sit_climb`, `sit_sit` 중 하나를 지정하고, 역할 반전은 `TASK_ROLE_SWAP=1`이다. 평가 상자는 역할 호환 고정 세트를 사용한다: 공유 받침은 학습 범위, 나머지는 ordinary/payload 교집합인40×40×40~50cm. 1환경에서 AT·공유 조합을 모두 재샘플링하기 위한 평가용 크기이며 학습의 전체 크기 분포와는 구분한다. 학습 YAML·크기 풀·보상·정적/CPA 계약은 유지한다. `EVAL_SAMPLER=klclimb50`은 명시적인 3조합 대조 평가에만 사용한다. Shared9 wrapper는 `BOX_SIZE_RANGE`·`RELATION_GRAPH` override를 받지 않는다.

서버 VNC는 위 checkpoint 변수를 그대로 사용한다.

```bash
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_vnc.sh "$SHARED9_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_vnc.sh "$SHARED9_CPA_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_vnc.sh "$STAGE1_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_cpa_vnc.sh "$STAGE1_CKPT" 4 1 6 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 TASK_GRAPH=ontop_ontop bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_shared9_vnc.sh "$STAGE1_CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_vnc.sh "$RESCUE_CKPT" 4 1 5 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_vnc.sh "$STAGE1_CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_stack bash tokenhsi/scripts/multi_agent/approach_stage2_unified_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb bash tokenhsi/scripts/multi_agent/approach_stage2_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_vnc.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_vnc.sh "$CKPT" 2 1 3 10
```

3인 운반·공유 CLIMB 고정 viewer:

```bash
THREE_CKPT=output/approach_stage2_rescue_klclimb50/ApproachStage2RescueKLClimb50_01-20-37-44/nn/ApproachStage2RescueKLClimb50_00007500.pth
THREE_GRAPH=tokenhsi/data/cfg/multi_agent/graphs/stage2_three_agent_place_two_climb.yaml
TOKENHSI_GPU="$GPU" STAGE1_ONLY=0 HEADLESS=0 RELATION_GRAPH="$THREE_GRAPH" BOX_SIZE_RANGE=0.60,0.60,0.50,0.50 EPISODE_LENGTH=600 bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_test.sh "$THREE_CKPT" 3 1 4 10
TOKENHSI_GPU="$GPU" STAGE1_ONLY=0 RELATION_GRAPH="$THREE_GRAPH" BOX_SIZE_RANGE=0.60,0.60,0.50,0.50 EPISODE_LENGTH=600 bash tokenhsi/scripts/multi_agent/approach_stage2_rescue_klclimb50_vnc.sh "$THREE_CKPT" 3 1 4 10
```

평가 결과는 `output/<실험>/metrics/`·`diagnostics/`, baseline 전이 기록은 `stage1_only_import.json`에 저장된다. 실측·RSI 충돌 검사·attention 분석의 재현 명령은 [Stage 2 진단 도구](../tokenhsi/docs/stage2_diagnostics.md)를 따른다.
