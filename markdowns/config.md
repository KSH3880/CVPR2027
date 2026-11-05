# 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 유지하는 실험은 아래 27개다. 설정과 실험별 train/test/VNC 진입점을 함께 관리한다.
- 모든 실행은 기본 conda `tokenhsi`를 사용한다. 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 한 번 지정한다. 기존 프로세스는 보존한다.
- 학습 인자: `[agents] [envs] [objects]`. Stage1·Stage1 PUSH는 **2명·2048환경·4물체**, BEFORE·coupled는 **4명·1024환경·8토큰**이다. BEFORE 학습은 인원 N≥1·토큰 M≥2N이며, BEFORE PUSH viewer/test/VNC는 M≥N을 지원한다. 부족한 예산의 과제 재샘플링은 [USAGE](USAGE.md#before-push)를 따른다.
- viewer/진단 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 추론과 resume은 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다.
- 본학습은 `output/`, 확인 학습·평가·viewer는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이며 학습 환경 수를 유지한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 기존 BEFORE의 기본 전이 가중치는 `checkpoints/Stage1_00078000.pth`, PUSH 기반 Stage2 설정은 `checkpoints/Stage1PushFinetune_00021000.pth`, `stage2_coupled_push_goal5`·`stage2_coupled_push_individual_goal`·`stage2_coupled_push_individual_goal_solo5`·`stage2_before_push_near50_pair50_source5_virtual`·`stage2_before_push_near50_pair50_source5_virtual_asym`·`stage2_before_push_near50_pair50_source5_goal5`·`stage2_before_push_near50_pair50_source5_virtual_goal5`·`stage2_before_push_near50_pair50_source5_virtual_asym_goal5`는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`다. 데이터·checkpoint 연결과 추가 옵션은 [USAGE](USAGE.md)를 따른다.
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
| [stage2_before_push_near50_pair50_source5_virtual.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual.yaml) | [stage2_before_push_near50_pair50_source5_virtual.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_asym.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym.yaml) | [stage2_before_push_near50_pair50_source5_virtual_asym.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_asym.yaml) |
| [stage2_before_push_near50_pair50_source5_goal5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_goal5.yaml) | [stage2_before_push_near50_pair50_source5_goal5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_goal5.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_asym_goal5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5.yaml) | [stage2_before_push_near50_pair50_source5_virtual_asym_goal5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_asym_goal5.yaml) |
| [stage2_before_push_near50_pair50_source5_goal5_goalroute.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_goal5_goalroute.yaml) | [stage2_before_push_near50_pair50_source5_goal5_goalroute.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_goal5_goalroute.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute.yaml) | [stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg.yaml) |
| [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2.yaml) | [stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2.yaml) |

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

# stage2_before_push_near50_pair50_source5_virtual
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_asym
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_goal5
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_asym_goal5
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_goal5_goalroute
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_goalroute_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2_train.sh 4 1024 8
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

# stage2_before_push_near50_pair50_source5_virtual
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_asym
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_goal5
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_asym_goal5
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_goal5_goalroute
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_goalroute_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2
HEADLESS=0 TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2_test.sh "$CKPT" 4 1 8 10
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

# stage2_before_push_near50_pair50_source5_virtual
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_asym
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_goal5
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_asym_goal5
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_goal5_goalroute
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_goalroute_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2
TOKENHSI_GPU=4 PORT=6080 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2_vnc.sh "$CKPT" 4 1 8 10
```

## 추론 옵션

- `TASK_GRAPH=random_scenario`가 기본이다. Stage1: `sit`, `climb`, `holding_at`, `holding_ontop`, `sit_climb`. BEFORE: `before_sit`, `before_climb`, `before_stack`, `one_pair_sit`, `one_pair_climb`, `one_pair_stack`, `all_solo`, `sit`, `climb`, `holding_at`, `holding_ontop`.
- `EVAL_SKILLS`와 `EVAL_SKILL_PROBS`는 함께 지정한다. 예: `EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0`. BEFORE의 기본 초기화는 loco이며 후행 RSI는 별도 옵션으로 선택한다.
- `BEFORE_WAIT_TERMINATION=0`은 wait 계열 viewer의 대기 위반 종료만 끈다(기본1). 낙상·시간 종료는 유지한다. 상세는 [USAGE](USAGE.md#before-viewer-대기-종료-끄기).
- `BEFORE_AT_GOAL_DISTANCE=7`은 goalroute BEFORE viewer에서 선행 공유상자→AT 목표 XY 거리를7m로 고정한다. 생략하면 학습 샘플링을 유지하며 정식 평가에는 적용하지 않는다. [거리 지정](USAGE.md#before-추론-초기화와-거리).
- `BEFORE_FOLLOWER_GOAL_DISTANCE=4`는 goalroute BEFORE viewer에서 선행 목표→후행 사람 root XY 거리를4m로 고정한다(ON_TOP도 사람 기준). 직접 지정 시 near50 대신 수동 거리를 사용하며 물리 충돌 검사는 유지한다. [거리 지정](USAGE.md#before-추론-초기화와-거리).
- `TASK_CAMERA=stack` 또는 `agent`로 viewer 카메라를 선택한다. 그 외 초기화·거리·resume·실행 환경 옵션은 [USAGE](USAGE.md)를 따른다.

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

- `stage2_coupled_push`의 `TASK_GRAPH=all_coupled`는4명 공동 PUSH, `all_solo`는 독립 행동 혼합, `push`는 전원 독립 PUSH다. [공통 Stage2 PUSH 설정](reference/stage2_coupled_push.md).

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

```bash
# stage2_before_push_near50_pair50_source5_virtual
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_near50_pair50_source5_virtual_general
```

```bash
# stage2_before_push_near50_pair50_source5_virtual_asym
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_virtual_asym_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_near50_pair50_source5_virtual_asym_general
```
