# 실행 가이드

모든 명령은 저장소 루트에서 실행한다. 사용할 컨피그의 명령 한 줄을 선택한다.

## 공통 규칙

- 기본 conda 환경은 `tokenhsi_jhh`이며 wrapper가 자동 활성화한다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 지정한다. 기존 프로세스는 임의 종료·재시작하지 않는다.
- 학습 인자: `[agents] [envs] [objects]`. Stage1/PUSH는 **2명·2048환경·4물체**, Stage2는 **4명·1024환경·8물체**다. 공유 받침 두 설정은 **4명·1024환경·12물체 토큰**을 사용한다.
- 추론 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 해당 실험의 checkpoint를 사용한다. test/VNC는 학습 당시 저장된 `relation_config.yaml`을 기본으로 읽는다.
- PUSH 계열 test/VNC는 마지막에 `[episode_steps]`를 추가할 수 있다. 예: `"$CKPT" 2 1 4 10 900`은 10회 반복·최대900step이며, 생략·환경변수 우선순위는 [USAGE](USAGE.md#push-추론-에피소드-길이)를 따른다.
- 본학습은 `output/`, 확인 학습·test·VNC 결과는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이고 해당 실험의 환경 수(stage1 계열 2048, stage2 계열 1024)를 유지한다.
- 화면 없는 평가는 로컬 추론 명령의 `HEADLESS=0`을 `HEADLESS=1`로 바꾼다. 전이·재개·추가 옵션은 [USAGE](USAGE.md), 세부 설계는 [reference](reference/)를 따른다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| [stage1.yaml](../tokenhsi/data/cfg/multi_agent/stage1.yaml) | [stage1.yaml](../tokenhsi/data/cfg/train/rlg/stage1.yaml) |
| [stage1_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml) | [stage1_v2.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2.yaml) |
| [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml) | [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed.yaml) |
| [stage2.yaml](../tokenhsi/data/cfg/multi_agent/stage2.yaml) | [stage2.yaml](../tokenhsi/data/cfg/train/rlg/stage2.yaml) |
| [stage2_shared_full_success.yaml](../tokenhsi/data/cfg/multi_agent/stage2_shared_full_success.yaml) | [stage2_shared_full_success.yaml](../tokenhsi/data/cfg/train/rlg/stage2_shared_full_success.yaml) |
| [stage2_shared_region_success.yaml](../tokenhsi/data/cfg/multi_agent/stage2_shared_region_success.yaml) | [stage2_shared_region_success.yaml](../tokenhsi/data/cfg/train/rlg/stage2_shared_region_success.yaml) |
| [stage1_v2_fixed_push.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push.yaml) | [stage1_v2_fixed_push.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push.yaml) |
| [stage1_v2_fixed_push_360.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360.yaml) | [stage1_v2_fixed_push_360.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360.yaml) |
| [stage1_v2_fixed_push_360_raw.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw.yaml) | [stage1_v2_fixed_push_360_raw.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw.yaml) |
| [stage1_v2_fixed_push_360_raw_cuboid.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push_360_raw_cuboid.yaml) | [stage1_v2_fixed_push_360_raw_cuboid.yaml](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed_push_360_raw_cuboid.yaml) |
| [chair_sit_climb_ft.yaml](../tokenhsi/data/cfg/multi_agent/chair_sit_climb_ft.yaml) | [chair_sit_climb_ft.yaml](../tokenhsi/data/cfg/train/rlg/chair_sit_climb_ft.yaml) |

## 학습

새 학습은 아래 환경변수를 해제한 뒤 해당 명령을 실행한다. PUSH·PUSH 360은 Stage1 fixed60000, PUSH raw·raw cuboid는 `output/stage1/Stage1V2Fixed_00073000.pth`에서 시작한다. 기존 PUSH 전이와 학습 재개는 [USAGE](USAGE.md#push-360도-접근-전이)를 따른다.

```bash
unset CHECK MAX_ITERATIONS OUTPUT_PATH PUSH_CHECKPOINT RESUME_CHECKPOINT STAGE1_PUSH_ENV_CONFIG STAGE1_CHECKPOINT
```

```bash
# stage1
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4

# stage1_v2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4

# stage1_v2_fixed
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_train.sh 4 1024 8

# stage2_shared_full_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_train.sh 4 1024 12

# stage2_shared_region_success
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_train.sh 4 1024 12

# stage1_v2_fixed_push
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_train.sh 2 2048 4

# stage1_v2_fixed_push_360
TOKENHSI_GPU=5 OUTPUT_PATH=output/stage1_v2_fixed_push_360_midpoint bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_train.sh 2 2048 4

# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_train.sh 2 2048 4

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_train.sh 2 2048 4
```


```bash
# chair_sit_climb_ft
bash tokenhsi/scripts/multi_agent/chair_sit_climb_ft_train.sh
```

### noVNC 학습 시각화 — 4환경

실제 4환경을 사용하는 별도 학습이다. 위 초기화 명령을 먼저 실행하며 결과는 `output_etc/check/<컨피그>/train_vnc/`에 저장한다. `6080` 포트를 포워딩하고 아래 주소에 접속한다.

`http://localhost:6080/vnc.html?autoconnect=1&resize=remote`

```bash
# stage1
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1 2 4 4

# stage1_v2
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2 2 4 4

# stage1_v2_fixed
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed 2 4 4

# stage2
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage2 4 4 8

# stage2_shared_full_success
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage2_shared_full_success 4 4 12

# stage2_shared_region_success
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage2_shared_region_success 4 4 12

# stage1_v2_fixed_push
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push 2 4 4

# stage1_v2_fixed_push_360
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360 2 4 4

# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360_raw 2 4 4

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=5 PORT=6080 bash tokenhsi/scripts/multi_agent/train_vnc.sh stage1_v2_fixed_push_360_raw_cuboid 2 4 4
```

## 로컬 추론

모니터가 연결된 로컬 viewer다. `CKPT`는 사용할 컨피그의 checkpoint로 지정한다. 아래 경로는 PUSH raw 예시다.

```bash
CKPT='output/stage1_v2_fixed_push_360_raw/<run>/nn/Stage1V2FixedPush360Raw_<epoch>.pth'

# stage1
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 1 4 10

# stage1_v2
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_test.sh "$CKPT" 2 1 4 10

# stage2
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_test.sh "$CKPT" 4 1 8 10

# stage2_shared_full_success
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_test.sh "$CKPT" 4 1 12 10

# stage2_shared_region_success
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_test.sh "$CKPT" 4 1 12 10

# stage1_v2_fixed_push
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_push_360
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_test.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_test.sh "$CKPT" 2 1 4 10
```


```bash
# chair_sit_climb_ft
HEADLESS=0 bash tokenhsi/scripts/multi_agent/chair_sit_climb_ft_test.sh "$CKPT"
```

## 서버 VNC 추론

기본 **2명·1환경** 테스트이며 Stage2는 기본 4명이다. `CKPT`를 사용할 컨피그의 checkpoint로 지정한다. 아래 경로는 PUSH raw 예시다. `6081` 포트를 포워딩하고 아래 주소에 접속한다.

`http://localhost:6081/vnc.html?autoconnect=1&resize=remote`

```bash
CKPT='output/stage1_v2_fixed_push_360_raw/<run>/nn/Stage1V2FixedPush360Raw_<epoch>.pth'

# stage1
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage1_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage1_v2_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_vnc.sh "$CKPT" 2 1 4 10

# stage2
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage2_vnc.sh "$CKPT" 4 1 8 10

# stage2_shared_full_success
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage2_shared_full_success_vnc.sh "$CKPT" 4 1 12 10

# stage2_shared_region_success
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage2_shared_region_success_vnc.sh "$CKPT" 4 1 12 10

# stage1_v2_fixed_push
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_push_360
TOKENHSI_GPU=5 PORT=6081 TASK_GRAPH=push TASK_CAMERA=stack EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_vnc.sh "$CKPT" 2 1 4 10

# stage1_v2_fixed_push_360_raw
TOKENHSI_GPU=5 PORT=6081 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_vnc.sh "$CKPT" 2 1 4 10 300

# stage1_v2_fixed_push_360_raw_cuboid
TOKENHSI_GPU=5 PORT=6081 TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_vnc.sh "$CKPT" 2 1 4 10 300
```


```bash
# chair_sit_climb_ft
PORT=6083 bash tokenhsi/scripts/multi_agent/chair_sit_climb_ft_vnc.sh "$CKPT"
```

## 추론 옵션

아래 옵션을 test/VNC 명령의 `bash` 앞에 붙인다. 기본 시작 동작은 `EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0`이다. 기본 과제는 일반 설정에서 `random_scenario`, PUSH 계열에서 `push`다.

**TASK_GRAPH — Stage1/PUSH에서 수행할 과제**

| 값 | 동작 |
| --- | --- |
| `random_scenario` | 사람마다 과제 무작위 선택 |
| `sit` | 모두 SIT |
| `climb` | 모두 CLIMB |
| `sit_climb` | 사람 순서대로 SIT·CLIMB 교대 |
| `carry_at` | 각자 물체를 자기 목표 위치로 운반 |
| `carry_ontop` | 각자 물체를 받침 위로 운반 |
| `push` | 지정 면에서 박스를 목표로 밀기 — PUSH 계열 전용 |

`holding_at`은 `carry_at`, `holding_ontop`은 `carry_ontop`과 같다.

`random_scenario`의 기존/PUSH 혼합 평가는 최소 2환경이 필요하다.

PUSH 360 추론의 직선/360도 보행 시작 배치는 [USAGE](USAGE.md#push-360-추론-배치)에서 선택한다.

PUSH 접촉 RSI의 원본/보정 자세는 `PUSH_RSI_MODE=raw|corrected`로 선택한다. 기본값과 실행 예시는 [USAGE](USAGE.md#push-추론-rsi-자세)를 따른다.

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
| `push` | PUSH 접촉 자세 — `TASK_GRAPH=push`에서 사용 |

여러 skill은 쉼표로 나열하고 `EVAL_SKILL_PROBS`를 같은 순서·개수로 지정한다. 하나만 지정하면 기본 확률은 `1.0`이다. skill은 사람별로 추첨하며 과제별 자동 배정이나 정확한 모션 프레임 고정은 아니다.

```bash
# AT 운반 — 접근부터 시작
TASK_GRAPH=carry_at EVAL_SKILLS=loco

# AT 운반 — 들고 있는 동작에서 시작
TASK_GRAPH=carry_at EVAL_SKILLS=carryWith

# ON_TOP 운반 — 접근 50%, 운반 50%
TASK_GRAPH=carry_ontop EVAL_SKILLS=loco,carryWith EVAL_SKILL_PROBS=0.5,0.5

# PUSH — 보행 시작
TASK_GRAPH=push EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0

# PUSH — 접촉 자세 시작
TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0
```

Stage2의 `TASK_GRAPH`는 `random_scenario`, `two_pairs_at`, `two_pairs_ontop`, `one_pair_at`, `one_pair_ontop`, `all_solo`, `sit`, `climb`, `carry_at`, `carry_ontop`을 지원한다. `carry_at`·`carry_ontop`은 전원 단독 운반이며, 공동 구성을 고정하려면 pair preset을 쓴다.

공동 그룹의 `EVAL_SKILLS`는 `loco`, `pickUp`, `carryWith`, `putDown`을 지원한다. Stage2 ON_TOP에는 `putDown`을 지정하지 않는다. SIT/CLIMB 참조 동작은 해당 전원 단독 preset에서 사용한다.

공유 받침 두 실험은 `random_scenario`, `all_solo`, `all_shared`, `sit`, `climb`, `carry_at`, `carry_ontop`, `shared_sit`, `shared_climb`, `shared_ontop`을 지원하며, `EVAL_SKILLS=loco`만 사용한다. `shared_*`는 전원이 하나의 받침을 공유한다.

의자 좌판·등받이 FT의 상세 조건과 현재 실행은 [상위 README](../../README.md)를 따른다.

물체별 독립 SIT/CLIMB 평가(Box/Chair/Cabinet/Table): [실행·조건](../evaluation/object_comparison/README.md). 저장소 루트에서 `python3 evaluation/object_comparison/run_eval.py`를 실행한다.

## 독립 폴더 실행 위치

모든 명령은 `/home/hwanhee/juan/CVPR2027/stage1_combined`에서 실행한다. 본학습은 `MAX_ITERATIONS`를 해제하면 YAML 기본20,000회 적용. 현재 실행 중인 기존 학습과 중복 실행하지 않는다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage1_combined
unset CHECK MAX_ITERATIONS RESUME_CHECKPOINT
bash tokenhsi/scripts/multi_agent/chair_sit_climb_ft_train.sh
```
