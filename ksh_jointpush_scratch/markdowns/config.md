# 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 유지하는 실험은 아래 스물두 개다. 설정과 실험별 train/test/VNC 진입점을 함께 관리한다.
- 모든 실행은 기본 conda `tokenhsi`를 사용한다. 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 한 번 지정한다. 기존 프로세스는 보존한다.
- 학습 인자: `[agents] [envs] [objects]`. 기본값은 Stage1·Stage1 PUSH **2명·2048환경·4물체**, BEFORE·coupled **4명·1024환경·8토큰**이다. BEFORE는 인원 N≥1·토큰 M≥2N, coupled는 인원 N≥1·토큰 M≥N을 지정할 수 있다.
- viewer/진단 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 추론과 resume은 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다. `stage2_coupled_contact` 학습은 패널티를 적용하는 전용 환경 YAML을 읽는다.
- 본학습은 `output/`, 확인 학습·평가·viewer는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이며 학습 환경 수를 유지한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 기존 BEFORE의 기본 전이 가중치는 `checkpoints/Stage1_00078000.pth`, PUSH 기반 Stage2 설정은 `checkpoints/Stage1PushFinetune_00021000.pth`, `stage2_coupled_push_goal5`·`stage2_coupled_push_individual_goal`·`stage2_coupled_push_individual_goal_solo5`·`stage2_coupled_push_individual_goal_solo5_carry5`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`·`stage2_coupled_push_individual_goal_solo5_keep1`·`stage2_coupled_push_individual_goal_solo5_shrink1`·`stage2_coupled_push_individual_goal_solo5_carry5_ground`·`stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`·`stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg`은 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`다. 데이터·checkpoint 연결과 추가 옵션은 [USAGE](USAGE.md)를 따른다.
- 학습·평가·viewer 기본 제한 시간은 **Stage1 900step, Stage2 1200step**이다.
- 모든 test의 기본값은 viewer다. 로컬 추론은 `HEADLESS=0`, 화면 없는 진단은 `HEADLESS=1`이다. **정식 평가는 `*_eval.sh`**, 1명 단독 평가는 **Stage1 계열 전용 `stage1_single_eval.sh`·`stage1_push_single_eval.sh`**로 구분한다. 일반 평가는 저장된 학습 인원·물체 수를 유지한다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| [stage1.yaml](../tokenhsi/data/cfg/multi_agent/stage1.yaml) | [stage1.yaml](../tokenhsi/data/cfg/train/rlg/stage1.yaml) |
| [stage2_before_nonemask_rsi433_near50.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_nonemask_rsi433_near50.yaml) | [stage2_before_nonemask_rsi433_near50.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_nonemask_rsi433_near50.yaml) |
| [stage2_before.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before.yaml) | [stage2_before.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before.yaml) |
| [stage2_before_rsi433.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_rsi433.yaml) | [stage2_before_rsi433.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_rsi433.yaml) |
| [stage1_push.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push.yaml) | [stage1_push.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push.yaml) |
| [stage2_before_push.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push.yaml) | [stage2_before_push.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push.yaml) |
| [stage2_coupled_push.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push.yaml) | [stage2_coupled_push.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push.yaml) |
| [stage2_before_push_near50.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50.yaml) | [stage2_before_push_near50.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50.yaml) |
| [stage2_coupled_push_goal5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_goal5.yaml) | [stage2_coupled_push_goal5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_goal5.yaml) |
| [stage2_coupled_push_individual_goal.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal.yaml) | [stage2_coupled_push_individual_goal.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal.yaml) |
| [stage2_before_push_near50_pair50_source5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5.yaml) | [stage2_before_push_near50_pair50_source5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5.yaml) |
| [stage2_coupled_push_individual_goal_solo5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5.yaml) | [stage2_coupled_push_individual_goal_solo5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5.yaml) |
| [stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml) | [stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml) |
| [stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml) | [stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml) |
| [stage2_coupled_push_individual_goal_solo5_keep1.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1.yaml) | [stage2_coupled_push_individual_goal_solo5_keep1.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_keep1.yaml) |
| [stage2_coupled_push_individual_goal_solo5_shrink1.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1.yaml) | [stage2_coupled_push_individual_goal_solo5_shrink1.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_shrink1.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml) |
| [stage2_coupled_contact.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_contact.yaml) | [stage2_coupled_contact.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_contact.yaml) |
| [stage2_coupled_contact_scratch.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_contact_scratch.yaml) | [stage2_coupled_contact_scratch.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_contact_scratch.yaml) |

## 학습

```bash
# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_train.sh 4 1024 8

# stage2_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_train.sh 4 1024 8

# stage2_before_rsi433
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_train.sh 4 1024 8

# stage1_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_train.sh 2 2048 4

# stage2_before_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_train.sh 4 1024 8

# stage2_coupled_push
TOKENHSI_GPU=6 STAGE1_CHECKPOINT=pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth bash tokenhsi/scripts/multi_agent/stage2_coupled_push_train.sh 4 1024 8

# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_train.sh 4 1024 8

# stage2_coupled_push_goal5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 STAGE1_CHECKPOINT=pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none
TOKENHSI_GPU=3 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_keep1
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_shrink1
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_train.sh 4 1024 8

# stage2_coupled_contact
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_coupled_contact_train.sh 4 1024 8

# stage2_coupled_contact_scratch
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_coupled_contact_scratch_train.sh 4 1024 8
```

## 로컬 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 1 4 10

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_test.sh "$CKPT" 4 1 8 10

# stage2_before
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_test.sh "$CKPT" 4 1 8 10

# stage2_before_rsi433
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_test.sh "$CKPT" 4 1 8 10

# stage1_push
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_test.sh checkpoints/Stage1PushFinetune_00021000.pth 2 1 4 10

# stage2_before_push
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_goal5
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled
TOKENHSI_GPU=0 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none
TOKENHSI_GPU=3 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_keep1
TOKENHSI_GPU=5 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_shrink1
TOKENHSI_GPU=6 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_contact
TOKENHSI_GPU=7 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_contact_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_contact_scratch
TOKENHSI_GPU=7 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_contact_scratch_test.sh "$CKPT" 4 1 8 10
```

## 서버 VNC 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_vnc.sh "$CKPT" 2 1 4 10

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_vnc.sh "$CKPT" 4 1 8 10

# stage2_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_rsi433
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_vnc.sh "$CKPT" 4 1 8 10

# stage1_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_vnc.sh checkpoints/Stage1PushFinetune_00021000.pth 2 1 4 10

# stage2_before_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_goal5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled
TOKENHSI_GPU=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none
TOKENHSI_GPU=3 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_keep1
TOKENHSI_GPU=5 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_shrink1
TOKENHSI_GPU=6 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_contact
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_coupled_contact_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_contact_scratch
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_coupled_contact_scratch_vnc.sh "$CKPT" 4 1 8 10
```

## 추론 옵션

- `TASK_GRAPH=random_scenario`가 기본이다. Stage1: `sit`, `climb`, `holding_at`, `holding_ontop`, `sit_climb`. BEFORE: `before_sit`, `before_climb`, `before_stack`, `one_pair_sit`, `one_pair_climb`, `one_pair_stack`, `all_solo`, `sit`, `climb`, `holding_at`, `holding_ontop`.
- `EVAL_SKILLS`와 `EVAL_SKILL_PROBS`는 함께 지정한다. 예: `EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0`. BEFORE의 기본 초기화는 loco이며 후행 RSI는 별도 옵션으로 선택한다.
- `TASK_CAMERA=stack` 또는 `agent`로 viewer 카메라를 선택한다. 그 외 초기화·거리·resume·실행 환경 옵션은 [USAGE](USAGE.md)를 따른다.
- coupled test/VNC의 `COUPLED_ASSIGNMENT=random|ordered|distance|optimal`(기본 random)은 공동 사람↔구역 배정이다. ordered는 좌우 정렬, distance는 총 접근 거리 최소화, optimal은 교차 우선이다. 직접 실행은 `--coupled_assignment`; 학습/resume·정식 평가의 비random 변경은 받지 않는다. [계산 방식·인원별 탐색](USAGE.md#공통-stage2-push와-coupled).
- 개인 목표 coupled test/VNC의 `COUPLED_PEERS=all|keep3|shrink3`(기본 all)은 5명 이상 공동 그룹의 사람 attention 상대 선택이다. keep3은 미접촉 상대 우선 최대3명을 유지하고, shrink3은 접촉한 상대를 빼 전원 접촉 시0명으로 줄인다. 직접 실행은 `--coupled_peers`; [접촉·갱신 규칙](USAGE.md#공통-stage2-push와-coupled).
- 이후 사용자 공동 PUSH 평가는 ordered를 사용하며, 현재는 전용 비교 실행기로 채점한다. [ordered 평가 실행법](USAGE.md#공통-stage2-push와-coupled).

```bash
# stage2_coupled_push_individual_goal_solo5: ordered를 distance 또는 optimal로 교체
TOKENHSI_GPU=6 COUPLED_ASSIGNMENT=ordered TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5: 미접촉 우선3명 유지
TOKENHSI_GPU=5 PORT=6080 COUPLED_ASSIGNMENT=ordered COUPLED_PEERS=keep3 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 8 1 8 10

# stage2_coupled_push_individual_goal_solo5: 접촉 완료 상대 제외 → 전원 접촉 시0명
TOKENHSI_GPU=5 PORT=6080 COUPLED_ASSIGNMENT=ordered COUPLED_PEERS=shrink3 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 8 1 8 10
```

- 11000 PTH의 감량/액션 샘플링 비교 viewer는 아래 명령을 사용한다. 첫 인자8을4/6으로 바꾸면4/6명이고 물체는 자동2N이다. `mass` 모드에서 `MASS_SCALE=0.8`은20% 감량, `0.6`은40% 감량이다. `baseline`·`mass80`·`mass60`·`noise`도 선택할 수 있다. ordered·1환경·seed0·씬당1200step이고 마지막 인자가 순차 재생 횟수(생략 시10)다. `SEED`로 시작 장면을 바꾼다. 상세는 [USAGE](USAGE.md#공통-stage2-push와-coupled)를 따른다.

```bash
# stage2_coupled_push_individual_goal_solo5: 8명 감량20%
TOKENHSI_GPU=6 PORT=6086 HEADLESS=0 MASS_SCALE=0.8 bash output_etc/check/solo5_mass_noise_n68_20261101/view.sh 8 mass 10

# stage2_coupled_push_individual_goal_solo5: 8명 원래 질량+액션 노이즈
TOKENHSI_GPU=5 PORT=6085 HEADLESS=0 bash output_etc/check/solo5_mass_noise_n68_20261101/view.sh 8 noise 10

# stage2_coupled_push_individual_goal_solo5: 8명 감량40%
TOKENHSI_GPU=6 PORT=6084 HEADLESS=0 MASS_SCALE=0.6 bash output_etc/check/solo5_mass_noise_n68_20261101/view.sh 8 mass 10
```

- 정식 평가는 loco·Stage1 900step/Stage2 1200step·512환경×seed0/1/2이며 점/영역 ever를 함께 보고한다. 인원/물체/거리 override를 받지 않는다. `--envs`·`--seeds`를 줄인 실행은 검사 전용으로 표시한다. [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md).

```bash
CKPT=checkpoints/Stage1_00078000.pth

# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_eval.sh "$CKPT" --output output_etc/check/stage1_general

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_eval.sh /absolute/path/to/near50.pth --output output_etc/check/stage2_before_nonemask_rsi433_near50_general

# stage2_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_eval.sh /absolute/path/to/before.pth --output output_etc/check/stage2_before_general

# stage2_before_rsi433
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_eval.sh /absolute/path/to/rsi433.pth --output output_etc/check/stage2_before_rsi433_general

# stage1 전용 1명 단독 비교
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_single_eval.sh "$CKPT" --output output_etc/check/stage1_single
```

```bash
# stage1_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_eval.sh checkpoints/Stage1PushFinetune_00021000.pth --output output_etc/check/stage1_push_general

# stage1_push 전용 1명 단독 비교
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_single_eval.sh checkpoints/Stage1PushFinetune_00021000.pth --output output_etc/check/stage1_push_single
```

```bash
# stage2_before_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_general
```

- `stage2_before_push`도 일반 평가는 학습 인원을 유지한다. `TASK_GRAPH=push`는 모든 사람을 독립 PUSH로 고정하는 viewer 옵션이다. [BEFORE PUSH](reference/stage2_before_push.md).

- `stage1_push`는 체크포인트 없이 스크래치 학습한다. 추론은 PTH 내부 저장 환경도 읽으며 일반 평가는 해당 학습 분포를 유지한다. `TASK_GRAPH=push`로 PUSH를 관찰한다. 설정 상세는 [PUSH](reference/stage1_push.md)를 따른다.

```bash
# stage2_coupled_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_general
```

- `stage2_coupled_push` 계열은 인원 상한 없이 N≥1·토큰 M≥N을 지원한다. `TASK_GRAPH=all_coupled`는 지정 인원 전원 공동 PUSH, `all_solo`는 독립 행동 혼합, `push`는 전원 독립 PUSH다. 물체가 부족한 임의 과제 조합은 다시 샘플링한다. 전원 ON_TOP 고정인 `holding_ontop`은 M≥2N이어야 한다. [공통 Stage2 PUSH 설정](reference/stage2_coupled_push.md).

```bash
# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_near50_general
```

```bash
# stage2_coupled_push_goal5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_goal5_general
```

```bash
# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_general
```

```bash
# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_general
```

```bash
# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_near50_pair50_source5_general
```

- 개인 목표 solo5의 임의 인원·물체 수 관찰 예시다. N은 사람 수, M은 물체 토큰 수(M≥N)다. 혼합 장면은 `TASK_GRAPH=random_scenario`로 바꾼다. 로컬 viewer는 `_vnc.sh`를 `_test.sh`로 바꾸고 `HEADLESS=0`을 지정한다.

```bash
N=10
M=10

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=5 PORT=6085 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" "$N" 1 "$M" 10
```

```bash
# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_general
```

```bash
# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_ground_general

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_general

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_general
```
