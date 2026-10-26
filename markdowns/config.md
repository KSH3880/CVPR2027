# PUSH 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- conda 환경은 `tokenhsi_sm120`이며 wrapper가 자동 활성화한다.
- GPU는 `TOKENHSI_GPU`로 지정한다. 본학습은 2명·2048환경·4물체, 학습 시각화는 4환경, 추론은 1환경이다.
- GPU0 MPS는 `source mps/shell.sh` 후 `mps_start 0`으로 켜고 `mps_status 0`으로 확인한다. 새 실행은 자동 연결하며 이미 실행 중인 학습에는 재시작 전까지 적용되지 않는다.
- 새 학습은 fixed78000에서 시작한다. cuboid checkpoint 전이·재개·추가 옵션은 [USAGE](USAGE.md#push-원본-rsi가변-정육면체)를 따른다.
- 본학습은 `output/`, 학습 시각화·추론은 `output_etc/check/`에 저장한다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| [stage1_v2_fixed_push_360_raw.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw.yaml) | [stage1_v2_fixed_push_360_raw.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw.yaml) |
| [stage1_v2_fixed_push_360_raw_mass20to25.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_mass20to25.yaml) | [stage1_v2_fixed_push_360_raw_mass20to25.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw_mass20to25.yaml) |
| [stage1_v2_fixed_push_360_raw_cuboid.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid.yaml) | [stage1_v2_fixed_push_360_raw_cuboid.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw_cuboid.yaml) |
| [stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml) | [stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml) |
| [stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130.yaml) | [stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130.yaml) |

## 학습

새로 시작할 때 먼저 실행한다.

```bash
unset CHECK MAX_ITERATIONS OUTPUT_PATH PUSH_CHECKPOINT RESUME_CHECKPOINT STAGE1_PUSH_ENV_CONFIG STAGE1_CHECKPOINT MINIBATCH_SIZE OLD_TASK_FRACTION TEACHER_KL_COEF

# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=0 HEADLESS=1 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_train.sh 2 2048 4

# stage1_v2_fixed_push_360_raw_mass20to25
TOKENHSI_GPU=0 HEADLESS=1 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_mass20to25_train.sh 2 2048 4

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=0 HEADLESS=1 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_train.sh 2 2048 4

# stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform
TOKENHSI_GPU=0 HEADLESS=1 TEACHER_KL_COEF=0 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform_train.sh 2 2048 4

# stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130
TOKENHSI_GPU=0 HEADLESS=1 TEACHER_KL_COEF=0 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130_train.sh 2 2048 4
```

백그라운드 학습 상태·로그. `rsi_uniform`은 fixed78000에서 KL0·새 optimizer/epoch으로 시작한 GPU0 MPS 사용자 서비스이며 로그아웃 후에도 유지된다. raw·20~25kg variant·기존 cuboid는 중단 상태다.

```bash
# stage1_v2_fixed_push_360_raw
systemctl --user status tokenhsi-push-raw-mps-20261021-162424.service --no-pager
tail -f output/stage1_v2_fixed_push_360_raw_full_plate_mps_resume_20261021_162424/train.log

# stage1_v2_fixed_push_360_raw_mass20to25
systemctl --user status tokenhsi-push-mass20to25-20261021-144954.service --no-pager
tail -f output/stage1_v2_fixed_push_360_raw_mass20to25_20261021_144954/train.log

# stage1_v2_fixed_push_360_raw_cuboid
systemctl --user status tokenhsi-push-cuboid-resume-20261023-203500.service --no-pager
tail -f output/stage1_v2_fixed_push_360_raw_cuboid_mps_resume_20261023_203500/train.log

# stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform
systemctl --user status tokenhsi-push-rsi-uniform-kl0-20261024-194136.service --no-pager
tail -f output/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform_kl0_fixed78000_20261024_194136/train.log
systemctl --user status tokenhsi-push-rsi-uniform-18k-monitor-20261025.service --no-pager

# stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130
systemctl --user status tokenhsi-push-cuboid-m10to15-r20cm-a130-20261025.service --no-pager
```

18K 감시·조건부 전환은 [절차](reference/stage1_push_light130.md)를 따른다. 다음 variant 서비스는 전환 조건 충족 후 생성한다.

### noVNC 학습 시각화 — 4환경

별도 4환경 학습이다. 위 초기화 명령을 먼저 실행한다. 6080 포트를 포워딩한다.

```bash
# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=0 PORT=6080 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360_raw 2 4 4

# stage1_v2_fixed_push_360_raw_mass20to25
TOKENHSI_GPU=0 PORT=6082 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360_raw_mass20to25 2 4 4

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=0 PORT=6084 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360_raw_cuboid 2 4 4

# stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform
TOKENHSI_GPU=0 PORT=6086 TEACHER_KL_COEF=0 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform 2 4 4

# stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130
TOKENHSI_GPU=0 PORT=6088 TEACHER_KL_COEF=0 STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130 2 4 4
```

`http://localhost:6080/vnc.html?autoconnect=1&resize=remote`

20~25kg variant: `http://localhost:6082/vnc.html?autoconnect=1&resize=remote`

직육면체 variant: `http://localhost:6084/vnc.html?autoconnect=1&resize=remote`

RSI 균등 거리 variant: `http://localhost:6086/vnc.html?autoconnect=1&resize=remote`

10–15kg·20cm·130° variant: `http://localhost:6088/vnc.html?autoconnect=1&resize=remote`

## 로컬 추론

`CKPT`·`CKPT_MASS20TO25`·`CKPT_CUBOID`·`CKPT_RSI_UNIFORM`·`CKPT_LIGHT130`에 각 학습에서 저장된 checkpoint 경로를 지정한다. 마지막 두 인자는 반복 횟수·최대 제어 step이다.

```bash
# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_test.sh "$CKPT" 2 1 4 100 300

# stage1_v2_fixed_push_360_raw_mass20to25
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_mass20to25_test.sh "$CKPT_MASS20TO25" 2 1 4 100 300

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_test.sh "$CKPT_CUBOID" 2 1 4 100 1200

# stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform_test.sh "$CKPT_RSI_UNIFORM" 2 1 4 100 1200

# stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130_test.sh "$CKPT_LIGHT130" 2 1 4 100 1200
```

화면 없는 평가는 `HEADLESS=1`을 사용한다.

## 서버 VNC 추론

각 checkpoint를 지정하고 사용하는 포트를 포워딩한다.

```bash
# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=0 PORT=6081 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_vnc.sh "$CKPT" 2 1 4 100 300

# stage1_v2_fixed_push_360_raw_mass20to25
TOKENHSI_GPU=0 PORT=6083 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_mass20to25_vnc.sh "$CKPT_MASS20TO25" 2 1 4 100 300

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=0 PORT=6085 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_vnc.sh "$CKPT_CUBOID" 2 1 4 100 1200

# stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform
TOKENHSI_GPU=0 PORT=6087 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform_vnc.sh "$CKPT_RSI_UNIFORM" 2 1 4 100 1200

# stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130
TOKENHSI_GPU=0 PORT=6089 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130_vnc.sh "$CKPT_LIGHT130" 2 1 4 100 1200
```

`http://localhost:6081/vnc.html?autoconnect=1&resize=remote`

20~25kg variant: `http://localhost:6083/vnc.html?autoconnect=1&resize=remote`

직육면체 variant: `http://localhost:6085/vnc.html?autoconnect=1&resize=remote`

RSI 균등 거리 variant: `http://localhost:6087/vnc.html?autoconnect=1&resize=remote`

10–15kg·20cm·130° variant: `http://localhost:6089/vnc.html?autoconnect=1&resize=remote`

## 추론 옵션

| 옵션 | 값 |
| --- | --- |
| `TASK_GRAPH` | `push` |
| `EVAL_SKILLS` | `loco`: 접근 시작 / `push`: 접촉 RSI 시작 |
| `EVAL_SKILL_PROBS` | 단일 skill은 `1.0`, 혼합은 `0.5,0.5` 등 |
| `PUSH_SPAWN_MODE` | `random`(기본) / `straight`: 보행 시작 위치 |
| `PUSH_RSI_MODE` | `raw`(기본) / `corrected`: 접촉 RSI 자세 |
| 6명 attention 진단 | [실행·뷰어](USAGE.md#6명5과제-attention-진단): 한 환경의 H6/O7/G6 전체 토큰·레이어/head별 비교 |

혼합 시작: `EVAL_SKILLS=loco,push EVAL_SKILL_PROBS=0.5,0.5`.
