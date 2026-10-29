# 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- Config·스크립트·실행 옵션 변경 시 아래 표와 학습·로컬 추론·VNC 명령을 함께 갱신한다. 새 컨피그는 표와 각 명령 목록의 맨 아래에 추가한다. 실험별 설명·별도 섹션을 늘리지 않는다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 지정한다. 기존 프로세스는 임의 종료·재시작하지 않는다.
- 학습 인자: `[agents] [envs] [objects]`. stage1 계열은 **2명·2048환경·4물체**, stage2는 **4명·1024환경·8물체**를 사용한다. `stage2_before`·`stage2_before_ground`도 기본값은 같으며 학습·평가에서 인원 N≥1, 물체 M≥N을 지정할 수 있다. 공유 받침 두 실험은 **4명·1024환경**이며 마지막 인자는 총 obj 토큰 수(기본·최소 `12`)다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 추론 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 해당 실험의 checkpoint를 사용한다. test/VNC는 학습 당시 저장된 `relation_config.yaml`을 기본으로 읽는다.
- 본학습은 `output/`, 확인 학습·test·VNC 결과는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이고 해당 실험의 환경 수(stage1 계열 2048, stage2 계열 1024)를 유지한다.
- 화면 없는 평가는 로컬 추론 명령의 `HEADLESS=0`을 `HEADLESS=1`로 바꾼다. 추가 옵션·resume·checkpoint 예외는 [USAGE](USAGE.md)를 따른다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| stage1.yaml (`../tokenhsi/data/cfg/multi_agent/stage1.yaml`, 당시 경로) | stage1.yaml (`../tokenhsi/data/cfg/train/rlg/stage1.yaml`, 당시 경로) |
| stage1_v2.yaml (`../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml`, 당시 경로) | stage1_v2.yaml (`../tokenhsi/data/cfg/train/rlg/stage1_v2.yaml`, 당시 경로) |
| [stage1_v2_fixed.yaml](../../../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml) | [stage1_v2_fixed.yaml](../../../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed.yaml) |
| stage2.yaml (`../tokenhsi/data/cfg/multi_agent/stage2.yaml`, 당시 경로) | stage2.yaml (`../tokenhsi/data/cfg/train/rlg/stage2.yaml`, 당시 경로) |
| stage2_shared_full_success.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_shared_full_success.yaml`, 당시 경로) | stage2_shared_full_success.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_shared_full_success.yaml`, 당시 경로) |
| stage2_shared_region_success.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_shared_region_success.yaml`, 당시 경로) | stage2_shared_region_success.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_shared_region_success.yaml`, 당시 경로) |
| stage2_before.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before.yaml`, 당시 경로) | stage2_before.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before.yaml`, 당시 경로) |
| stage2_before_ground.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground.yaml`, 당시 경로) | stage2_before_ground.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground.yaml`, 당시 경로) |
| stage2_before_ground_mix50.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix50.yaml`, 당시 경로) | stage2_before_ground_mix50.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix50.yaml`, 당시 경로) |
| stage2_before_ground_mix334_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix334_1200.yaml`, 당시 경로) | stage2_before_ground_mix334_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix334_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_actorbias0_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_actorbias0_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_actorbias0_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_actorbias0_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_actorbias0_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_actorbias0_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_actorbias0_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_actorbias0_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_carry50_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_carry50_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_carry50_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_carry50_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_fullrsi_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_fullrsi_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_fullrsi_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_fullrsi_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_adapter_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_adapter_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_adapter_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_adapter_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_adapter_bias0_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_adapter_bias0_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_adapter_bias0_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_adapter_bias0_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_bias0_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_bias0_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_bias0_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_bias0_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_nonemask_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_nonemask_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_nonemask_1200.yaml`, 당시 경로) |
| stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200.yaml`, 당시 경로) | stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200.yaml`, 당시 경로) |
| stage2_before_ground_atprogress_nonemask_rsi433_near50_1200.yaml (`../tokenhsi/data/cfg/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_1200.yaml`, 당시 경로) | stage2_before_ground_atprogress_nonemask_rsi433_near50_1200.yaml (`../tokenhsi/data/cfg/train/rlg/stage2_before_ground_atprogress_nonemask_rsi433_near50_1200.yaml`, 당시 경로) |
| [stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200.yaml](../../../tokenhsi/data/cfg/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200.yaml) | [stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200.yaml](../../../tokenhsi/data/cfg/train/rlg/stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200.yaml) |
| [stage2_before_ground_mix226_atprogress_objalias_1200.yaml](../../../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_objalias_1200.yaml) | [stage2_before_ground_mix226_atprogress_objalias_1200.yaml](../../../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_objalias_1200.yaml) |
| [stage2_before_ground_mix226_atprogress_objalias_rsi433_1200.yaml](../../../tokenhsi/data/cfg/multi_agent/stage2_before_ground_mix226_atprogress_objalias_rsi433_1200.yaml) | [stage2_before_ground_mix226_atprogress_objalias_rsi433_1200.yaml](../../../tokenhsi/data/cfg/train/rlg/stage2_before_ground_mix226_atprogress_objalias_rsi433_1200.yaml) |

## 학습

필요한 컨피그의 한 줄만 실행한다.

```bash
# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4

# stage1_v2
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4

# stage1_v2_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_train.sh 4 1024 8

# stage2_shared_full_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_train.sh 4 1024 12

# stage2_shared_region_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_train.sh 4 1024 12

# stage2_before
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_train.sh 4 1024 8

# stage2_before_ground
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_ground_train.sh 4 1024 8

# stage2_before_ground_mix50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix50_train.sh 4 1024 8

# stage2_before_ground_mix334_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00073000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix334_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00073000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_actorbias0_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_actorbias0_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_actorbias0_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_actorbias0_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_carry50_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_carry50_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_fullrsi_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_fullrsi_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_adapter_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_adapter_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_adapter_bias0_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_adapter_bias0_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_bias0_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_bias0_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_nonemask_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=5 STAGE1_CHECKPOINT=/home/hwanhee/juan/CVPR2027/stage1_combined/checkpoints/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200_train.sh 4 1024 8

# stage2_before_ground_atprogress_nonemask_rsi433_near50_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_1200_train.sh 4 1024 8

# stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_objalias_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_objalias_1200_train.sh 4 1024 8

# stage2_before_ground_mix226_atprogress_objalias_rsi433_1200
env -u CHECK -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u OUTPUT_PATH TOKENHSI_GPU=0 STAGE1_CHECKPOINT=../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_objalias_rsi433_1200_train.sh 4 1024 8
```

## 로컬 추론

`CKPT`를 실행할 컨피그의 checkpoint 경로로 바꾸고 한 줄만 실행한다.

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 1 4 10

# stage1_v2
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_test.sh "$CKPT" 2 1 4 10

# stage2
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_test.sh "$CKPT" 4 1 8 10

# stage2_shared_full_success
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_test.sh "$CKPT" 4 1 12 10

# stage2_shared_region_success
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_test.sh "$CKPT" 4 1 12 10

# stage2_before
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix50
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix50_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix334_1200
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix334_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_1200
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_1200
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_actorbias0_1200
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_actorbias0_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_actorbias0_1200
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_actorbias0_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_carry50_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_carry50_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_fullrsi_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_fullrsi_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_adapter_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_adapter_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_adapter_bias0_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_adapter_bias0_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_bias0_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_bias0_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_nonemask_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200
HEADLESS=0 TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_atprogress_nonemask_rsi433_near50_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_objalias_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_objalias_1200_test.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_objalias_rsi433_1200
HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_objalias_rsi433_1200_test.sh "$CKPT" 4 1 8 10
```

## 서버 VNC 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_vnc.sh "$CKPT" 2 1 4 10

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_vnc.sh "$CKPT" 4 1 8 10

# stage2_shared_full_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_vnc.sh "$CKPT" 4 1 12 10

# stage2_shared_region_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_vnc.sh "$CKPT" 4 1 12 10

# stage2_before
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_ground_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix50_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix334_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix334_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_actorbias0_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_actorbias0_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_actorbias0_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_actorbias0_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_carry50_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_carry50_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_fullrsi_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_fullrsi_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_adapter_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_adapter_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_adapter_bias0_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_adapter_bias0_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_bias0_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_bias0_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_nonemask_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_atprogress_nonemask_rsi433_near50_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_objalias_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_objalias_1200_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_ground_mix226_atprogress_objalias_rsi433_1200
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix226_atprogress_objalias_rsi433_1200_vnc.sh "$CKPT" 4 1 8 10
```

## 추론 옵션

아래 옵션을 test/VNC 명령의 `bash` 앞에 붙인다. 기본값은 `TASK_GRAPH=random_scenario EVAL_SKILLS=loco`다.

**TASK_GRAPH — 수행할 과제**

| 값 | 동작 |
| --- | --- |
| `random_scenario` | 사람마다 과제 무작위 선택 |
| `sit` | 모두 SIT |
| `climb` | 모두 CLIMB |
| `sit_climb` | 사람 순서대로 SIT·CLIMB 교대 |
| `carry_at` | 각자 물체를 자기 목표 위치로 운반 |
| `carry_ontop` | 각자 물체를 받침 위로 운반 |

`holding_at`은 `carry_at`, `holding_ontop`은 `carry_ontop`과 같다.

**EVAL_SKILLS — 시작할 참조 동작**

| 값 | 초기 동작 |
| --- | --- |
| `loco` | 걷기·접근 |
| `sit` | 앉기 |
| `climb` | 오르기 |
| `pickUp` | 물체 집기 |
| `carryWith` | 물체를 들고 운반 |
| `putDown` | 물체 내려놓기 |
| `climbNoRSI` | 별도 Climb 참조 모션 묶음 |
| `omomo` | OMOMO 운반 참조 모션 묶음 |

여러 skill은 쉼표로 나열하고 `EVAL_SKILL_PROBS`를 같은 순서·개수로 지정한다. 하나만 지정하면 기본 확률은 `1.0`이다. skill은 사람별로 추첨하며 과제별 자동 배정이나 정확한 모션 프레임 고정은 아니다.

```bash
# AT 운반 — 접근부터 시작
TASK_GRAPH=carry_at EVAL_SKILLS=loco

# AT 운반 — 들고 있는 동작에서 시작
TASK_GRAPH=carry_at EVAL_SKILLS=carryWith

# ON_TOP 운반 — 접근 50%, 운반 50%
TASK_GRAPH=carry_ontop EVAL_SKILLS=loco,carryWith EVAL_SKILL_PROBS=0.5,0.5
```

Stage2의 `TASK_GRAPH`는 `random_scenario`, `two_pairs_at`, `two_pairs_ontop`, `one_pair_at`, `one_pair_ontop`, `all_solo`, `sit`, `climb`, `carry_at`, `carry_ontop`을 지원한다. `carry_at`·`carry_ontop`은 전원 단독 운반이며, 공동 구성을 고정하려면 pair preset을 쓴다.

공동 그룹의 `EVAL_SKILLS`는 `loco`, `pickUp`, `carryWith`, `putDown`을 지원한다. Stage2 ON_TOP에는 `putDown`을 지정하지 않는다. SIT/CLIMB 참조 동작은 해당 전원 단독 preset에서 사용한다.

공유 받침 두 실험은 `random_scenario`, `all_solo`, `all_shared`, `sit`, `climb`, `carry_at`, `carry_ontop`, `shared_sit`, `shared_climb`, `shared_ontop`을 지원하며, `EVAL_SKILLS=loco`만 사용한다. `shared_*`는 전원이 하나의 받침을 공유한다.

`stage2_before`·`stage2_before_ground`는 인원 N≥1·물체 M≥N을 지원하며 기본값은 4명·8물체다. `EVAL_SKILLS=loco`만 지원한다. `TASK_GRAPH`는 `random_scenario`, `before_sit`, `before_climb`, `before_stack`(가능한 최대 2인 쌍, 홀수의 마지막 1명은 단독), `one_pair_sit`, `one_pair_climb`, `one_pair_stack`(1쌍), `all_solo`, `sit`, `climb`, `carry_at`, `carry_ontop`이다. pair preset은 2명 이상, 전원 `carry_ontop`은 물체 2N개 이상이 필요하다. 후행·ON_TOP 자기 상자의 근접 무작위 평가는 `BEFORE_FOLLOWER_NEAR=1`을 test/VNC 앞에 붙인다. 거리 규칙과 추가 옵션은 [USAGE](USAGE.md#stage2-before-평가-옵션)를 따른다.

`stage2_before_ground_mix50`는 같은 인원·물체·TASK_GRAPH 옵션을 사용한다. 기본 test/VNC는 50/50 혼합이며 `BEFORE_FOLLOWER_NEAR=1`을 붙이면 BEFORE 쌍에 학습의 가까운 배치만 적용한다. 전용 checkpoint와 기본 혼합 배치 안내는 [USAGE](USAGE.md#stage2-before-5050-혼합-배치)를 따른다.

`stage2_before_ground_mix334_1200` viewer의 거리 모드 선택은 [추가 실행 옵션](USAGE.md#stage2-before-mix334-1200)을 따른다.

`stage2_before_ground_mix226_1200` viewer의 거리 모드 선택은 [추가 실행 옵션](USAGE.md#stage2-before-mix226-1200)을 따른다.

`stage2_before_ground_mix226_atprogress_1200`의 추가 실행 옵션은 [USAGE](USAGE.md#stage2-before-mix226-at-progress-1200)를 따른다.

`stage2_before_ground_mix226_atprogress_fullrsi_1200`의 RSI·선후행 carryWith 뷰어 옵션은 [USAGE](USAGE.md#stage2-before-mix226-at-progress-full-rsi)를 따른다.
