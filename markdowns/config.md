# Stage1 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- Config·스크립트·실행 옵션 변경 시 아래 표와 학습·로컬 추론·VNC 명령을 함께 갱신한다. 새 컨피그는 표와 각 명령 목록의 맨 아래에 추가한다. 실험별 설명·별도 섹션을 늘리지 않는다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 지정한다. 기존 프로세스는 임의 종료·재시작하지 않는다.
- 학습 인자: `[agents] [envs] [objects]`. **2명·2048환경·4물체**를 사용한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 추론 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 해당 실험의 checkpoint를 사용한다. test/VNC는 학습 당시 저장된 `relation_config.yaml`을 기본으로 읽는다.
- 본학습은 `output/`, 확인 학습·test·VNC 결과는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이고 환경 수는 2048을 유지한다.
- 화면 없는 평가는 로컬 추론 명령의 `HEADLESS=0`을 `HEADLESS=1`로 바꾼다. 추가 옵션·resume·checkpoint 예외는 [USAGE](USAGE.md)를 따른다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| [stage1.yaml](../tokenhsi/data/cfg/multi_agent/stage1.yaml) | [stage1.yaml](../tokenhsi/data/cfg/train/rlg/stage1.yaml) |
| [stage1_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml) | [stage1_v2.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2.yaml) |
| [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml) | [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed.yaml) |

## 학습

필요한 컨피그의 한 줄만 실행한다.

```bash
# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4

# stage1_v2
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4

# stage1_v2_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4
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
