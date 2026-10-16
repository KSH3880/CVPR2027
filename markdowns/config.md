# SA4 공동운반 실행 가이드

## 빠른 확인

- [Human SA4](#human-sa4): 실제 payload 중심 관측.
- [SA4 GraspView](#sa4-graspview): 공동운반에서 사람별 반대 파지 앵커 관측.
- AT20 추가 설정: [stage2_at20](#stage2_at20), [stage2_grasp_at20](#stage2_grasp_at20).
- 두 모델에 기존 AT 조건과 AT20 조건을 제공한다. 모델·성공조건이 다른 Stage2 체크포인트는 서로 재개·평가하지 않는다.
- 학습은 2명·2048환경·4물체다. 짧은 검증도 환경 수를 유지하고 반복 횟수만 줄인다.

## 준비와 공통 설정

명령은 저장소 루트에서 실행한다. Isaac Gym이 설치된 `tokenhsi` conda 환경을 사용한다.

```bash
nvidia-smi
source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/scripts/prepare_joint_carry.py --data-root /path/to/TokenHSI/tokenhsi/data
```

원본 TokenHSI 데이터는 읽기 전용 심링크로 연결한다. `TOKENHSI_DATA_ROOT`로도 지정할 수 있다.
보정 AMP12개와 FullPickup RSI는 Git에 포함된다. Stage1 PTH, 학습 결과와 `output/rsi_cache`는 포함되지 않는다.
첫 학습 시 단독 size RSI 캐시를 생성하므로 초기 준비가 오래 걸릴 수 있다.

공통: 공동80%·단독20%, AMP 원본80/후진10/옆걸음10, paired pickup100/carryWith129/putDown135,
Goal360, 공동 상대 CPA release+1m, 공동 loco의50%에 받침을 사용한다.
데이터 준비 → 단독 size RSI 캐시 → 공동/단독 reset → AMP reference → PPO 학습은 두 모델의 공통 정규 경로다.
`jointCarryTraining`으로 policy와 AMP 계약을 검증한다. 보정 AMP는 필수 입력이며 누락 시 학습을 시작하지 않는다.
캐시는 단독 환경의 크기·skill에 대해 생성하고 simulator·asset·motion·코드 fingerprint가 같으면 재사용한다.
공동 paired RSI는 캐시와 분리해100/129/135프레임을 사용한다. 실제 reset의 AMP 이력과 전문가 reference 샘플링도 분리한다.
신규 PTH는 정규 학습 계약으로 저장하며 기존 SA PTH의 계약도 호환 검사 후 불러온다.

Actor encoder와 scene RMS는 동결하고 SA4/action head를 학습한다. Critic은 전체 학습한다.
SELF/COUPLED/NONE bias는 모두 학습하며 사람·task·entity 순열을 유지한다.

`STAGE1_CHECKPOINT`에는 CarryDistill30000 PTH를 지정한다. 기존 체크포인트의 파라미터 이름과 크기를 유지한다.
재개는 해당 실험의 `RESUME_CHECKPOINT=/path/to/checkpoint.pth`로 한다.
기존 프로세스는 종료하거나 재시작하지 않는다. GPU는 명령 앞의 `TOKENHSI_GPU`로 한 번 지정한다.

## 전체 실행 파일

| 실험 | 파일 접두사 | AT 성공 거리 | 제공 파일 |
| --- | --- | --- | --- |
| Human SA4 | `stage2` | 약10.26cm | env/train YAML, `_train.sh`, `_test.sh`, `_vnc.sh` |
| SA4 GraspView | `stage2_grasp` | 약10.26cm | env/train YAML, `_train.sh`, `_test.sh`, `_vnc.sh` |
| Human SA4 AT20 | `stage2_at20` | 20cm | env/train YAML, `_train.sh`, `_test.sh`, `_vnc.sh` |
| SA4 GraspView AT20 | `stage2_grasp_at20` | 20cm | env/train YAML, `_train.sh`, `_test.sh`, `_vnc.sh` |

각 접두사에 대해 `tokenhsi/data/cfg/multi_agent/<접두사>.yaml`,
`tokenhsi/data/cfg/train/rlg/<접두사>.yaml`, `tokenhsi/scripts/multi_agent/<접두사>_{train,test,vnc}.sh`를 사용한다.
기본 output도 `output/<접두사>`로 분리된다.

기존 설정은 `exp(-10 × 3차원 거리²) ≥ 0.9`와 높이 차이≤1mm를 유지한다.
AT20은 `success.at_distance_tolerance: 0.2`로 **박스 중심과 목표점의 3차원 거리≤20cm AND 높이 차이 절댓값≤1mm**를 판정한다.
공동·단독 AT 모두 적용되며 기존 HOLDING gate, ON_TOP, 연속 state/progress 보상은 유지한다.
성공 보너스와 성공 지표에는 새 판정이 적용된다. 성공 시 자동 종료하지 않는 설정도 유지한다.

이름만 바뀐 `stage2`/`stage2_grasp`은 기존 해당 SA PTH를 사용할 수 있다.
AT20은 성공조건 계약이 달라 기존10.26cm PTH의 재개·평가를 거부한다.
AT20 학습은 Stage1 전이로 시작하고 이후 해당 AT20 PTH로 재개·평가한다.

## Human SA4

원본634차원 관측에서 Stage1 human64를 추출해 human SA4로 전달한다.

```bash
EXPERIMENT=stage2
SCRIPTS=tokenhsi/scripts/multi_agent

# Stage1 전이 학습
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 STAGE1_CHECKPOINT=/path/to/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
bash "$SCRIPTS/${EXPERIMENT}_train.sh" 2 2048 4

PTH=/path/to/this_experiment/checkpoint.pth

# 로컬 viewer; 화면 없는 평가는 HEADLESS=1
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash "$SCRIPTS/${EXPERIMENT}_test.sh" "$PTH" 2 1 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith \
bash "$SCRIPTS/${EXPERIMENT}_vnc.sh" "$PTH" 2 1 4 10
```

## SA4 GraspView

공동 장면에서 reset 때 반대 파지 앵커를 배정하고 episode 동안 유지한다. 앵커는 로컬 Y의 ±(반폭−0.15m), 현재 상자에서 ±0.25m다. 실제 물체·목표·물리·보상은 동일하다.

A/B 관측634차원을 각각 정규화해1268차원으로 저장하고 각 view의 자기 human64를 SA4에 넣는다. 단독에서는 view가 같아 encoder를 한 번만 실행한다. 현재2명·4물체 전용이다.

```bash
EXPERIMENT=stage2_grasp
SCRIPTS=tokenhsi/scripts/multi_agent

# Stage1 전이 학습
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 STAGE1_CHECKPOINT=/path/to/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
bash "$SCRIPTS/${EXPERIMENT}_train.sh" 2 2048 4

PTH=/path/to/this_experiment/checkpoint.pth

# 로컬 viewer; 화면 없는 평가는 HEADLESS=1
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash "$SCRIPTS/${EXPERIMENT}_test.sh" "$PTH" 2 1 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith \
bash "$SCRIPTS/${EXPERIMENT}_vnc.sh" "$PTH" 2 1 4 10
```

`TASK_GRAPH=joint_carry_ontop` 또는 `independent`로 과제를 선택한다.
`EVAL_SKILLS=loco/pickUp/carryWith/putDown`을 사용한다. 공동 ON_TOP의 putDown RSI는 지원하지 않는다.
짧은 검증은 `OUTPUT_PATH=output/<실험명>_check MAX_ITERATIONS=2`로 본학습과 분리한다.
현재 제공하는 평가 스크립트는 네 설정 모두2명·4물체 기준이다.

## stage2_at20

stage2의 AT 성공 반경20cm 비교 설정이다. 높이 허용오차는1mm다.

```bash
EXPERIMENT=stage2_at20
SCRIPTS=tokenhsi/scripts/multi_agent

# Stage1 전이 학습
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 STAGE1_CHECKPOINT=/path/to/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
bash "$SCRIPTS/${EXPERIMENT}_train.sh" 2 2048 4

PTH=/path/to/this_at20_experiment/checkpoint.pth

# 로컬 viewer; 화면 없는 평가는 HEADLESS=1
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash "$SCRIPTS/${EXPERIMENT}_test.sh" "$PTH" 2 1 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith \
bash "$SCRIPTS/${EXPERIMENT}_vnc.sh" "$PTH" 2 1 4 10
```

## stage2_grasp_at20

stage2_grasp의 AT 성공 반경20cm 비교 설정이다. 높이 허용오차는1mm다.

```bash
EXPERIMENT=stage2_grasp_at20
SCRIPTS=tokenhsi/scripts/multi_agent

# Stage1 전이 학습
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 STAGE1_CHECKPOINT=/path/to/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
bash "$SCRIPTS/${EXPERIMENT}_train.sh" 2 2048 4

PTH=/path/to/this_at20_experiment/checkpoint.pth

# 로컬 viewer; 화면 없는 평가는 HEADLESS=1
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash "$SCRIPTS/${EXPERIMENT}_test.sh" "$PTH" 2 1 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith \
bash "$SCRIPTS/${EXPERIMENT}_vnc.sh" "$PTH" 2 1 4 10
```

## 검증

```bash
PYTHONPATH=tokenhsi python -m pytest tokenhsi/tests -q
```

`tests/check_sa_checkpoint.py`는 Stage1 전이 두 모델과 기존 SA4 epoch8000 PTH를 원본 코드의
행동·가치·RMS 결과와 비교한다. `tests/check_sa_environment.py`는 AT/ON_TOP/단독의
geometry·HOLDING gate·보상·reset·CPA 해제를 비교한다. `--help`로 인자를 확인한다.
원본 정리 검증은 `output/cleanup_check/`, 이름 변경·AT20 검증은 `output/at20_check/`에 저장한다. 최신 수행 결과는 [changelog](../changelog.md)를 따른다.
