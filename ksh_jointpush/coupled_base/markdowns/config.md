# 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 유지하는 실험은 아래 여덟 개다. 설정과 실험별 train/test/VNC 진입점을 함께 관리한다.
- 모든 실행은 기본 conda `tokenhsi`를 사용한다. 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 한 번 지정한다. 기존 프로세스는 보존한다.
- 학습 인자: `[agents] [envs] [objects]`. Stage1·Stage1 PUSH는 **2명·2048환경·4물체**, BEFORE·coupled는 **4명·1024환경·8토큰**이다. BEFORE의 인원 N≥1, 토큰 M≥2N을 지정할 수 있다.
- viewer/진단 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 추론과 resume은 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다.
- 본학습은 `output/`, 확인 학습·평가·viewer는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이며 학습 환경 수를 유지한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 기존 BEFORE의 기본 전이 가중치는 `checkpoints/Stage1_00078000.pth`, PUSH 기반 Stage2 설정은 `checkpoints/Stage1PushFinetune_00021000.pth`다. 데이터·checkpoint 연결과 추가 옵션은 [USAGE](USAGE.md)를 따른다.
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
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_train.sh 4 1024 8

# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_train.sh 4 1024 8
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
```

## 추론 옵션

- `TASK_GRAPH=random_scenario`가 기본이다. Stage1: `sit`, `climb`, `holding_at`, `holding_ontop`, `sit_climb`. BEFORE: `before_sit`, `before_climb`, `before_stack`, `one_pair_sit`, `one_pair_climb`, `one_pair_stack`, `all_solo`, `sit`, `climb`, `holding_at`, `holding_ontop`.
- `EVAL_SKILLS`와 `EVAL_SKILL_PROBS`는 함께 지정한다. 예: `EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0`. BEFORE의 기본 초기화는 loco이며 후행 RSI는 별도 옵션으로 선택한다.
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
