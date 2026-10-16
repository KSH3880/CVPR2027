# SA4 공동운반 실행 가이드

## 빠른 확인

- [Human SA4](#human-sa4): 실제 payload 중심 관측.
- [SA4 GraspView](#sa4-graspview): 공동운반에서 사람별 반대 파지 앵커 관측.
- AT20 추가 설정: [stage2_at20](#stage2_at20), [stage2_grasp_at20](#stage2_grasp_at20).
- 4명 랜덤 그룹: [stage2_at20_4agent / stage2_grasp_at20_4agent](#4명-랜덤-그룹-at20).
- 두 모델에 기존 AT 조건과 AT20 조건을 제공한다. 모델·성공조건이 다른 Stage2 체크포인트는 서로 재개·평가하지 않는다.
- 기존 네 설정은 2명·2048환경·4물체, 신규 두 설정은 4명·2048환경·관측 물체8슬롯이다. 짧은 검증도 환경 수를 유지하고 반복 횟수만 줄인다.

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

기존 2명 설정: 공동80%·단독20%, AMP 원본80/후진10/옆걸음10, paired pickup100/carryWith129/putDown135,
Goal360, 공동 상대 CPA release+1m, 공동 loco의50%에 받침을 사용한다.
데이터 준비 → 단독 size RSI 캐시 → 공동/단독 reset → AMP reference → PPO 학습은 두 모델의 공통 정규 경로다.
`jointCarryTraining`으로 policy와 AMP 계약을 검증한다. 보정 AMP는 필수 입력이며 누락 시 학습을 시작하지 않는다.
캐시는 단독 환경의 크기·skill에 대해 생성하고 simulator·asset·motion·코드 fingerprint가 같으면 재사용한다.
공동 paired RSI는 캐시와 분리해100/129/135프레임을 사용한다. 실제 reset의 AMP 이력과 전문가 reference 샘플링도 분리한다.
신규 PTH는 정규 학습 계약으로 저장하며 기존 SA PTH의 계약도 호환 검사 후 불러온다.

Actor encoder와 scene RMS는 동결하고 SA4/action head를 학습한다. Critic은 전체 학습한다.
SELF/COUPLED/NONE bias는 모두 학습하며 사람·task·entity 순열을 유지한다.
2명·4명 전체 여섯 train 설정에서 `cache_frozen_actor: true`로 동결 encoder 특징을 PPO 업데이트 동안 재사용한다. MA의 미사용 `next_obses` 버퍼 제거도 공통 적용한다. 실행 중인 프로세스에는 소급 적용되지 않으며 새 실행 또는 같은 실험 checkpoint 재개부터 반영된다.
2명 활성화 검증은 CPU45개와 GPU6·2048환경의 기본/Grasp 대표 설정 학습·저장·재로드로 확인했다. 기록은 `output/two_agent_cache_check/validation.json`에 있다. 2명 전후 속도와 전체 VRAM 절감량은 별도 측정하지 않았다.

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
| Human SA4 AT20 4명 | `stage2_at20_4agent` | 20cm | env/train YAML, `_train.sh`, `_test.sh`, `_vnc.sh` |
| SA4 GraspView AT20 4명 | `stage2_grasp_at20_4agent` | 20cm | env/train YAML, `_train.sh`, `_test.sh`, `_vnc.sh` |

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


## 4명 랜덤 그룹 AT20

`stage2_at20_4agent`와 `stage2_grasp_at20_4agent`는 각 episode마다 4명의 단독/2명 공동팀 구성을 다시 뽑는다. 순서를 섞고 미배정 사람을 확률0.8로 다른 미배정 사람과 묶는다. 공동팀 수0/1/2의 reset 확률은0.8%/35.2%/64%, 사람 기준 공동 비율은81.6%다. 그룹별 AT/ON_TOP은 독립50:50이며 episode 중 팀과 목표는 유지된다. 실제 timestep 비율은 episode 길이 때문에 reset 비율과 다를 수 있다.

### 상자·관측·배치

- 물리 상자12개: 단독 payload4/support4, 공동 payload2/support2. 단독 크기는 기존 Stage1 크기 샘플러로 환경 생성 때 정하고 크기별 RSI 캐시를 쓴다. 공동 크기는 payload `[0.52, 0.80, 0.40]`, support `[0.52, 0.80, 0.30]`이다. reset마다 물리 크기를 바꾸지 않는다.
- 동시에 쓰는 논리 물체는 최대8개. 비활성 물리 상자는 플레이 영역 밖(환경 중심 XY+1000m, actor 간2m)의 지면에 격리하고 관측에서 제거한다. 빈 물체·사용하지 않는 목표 토큰은 GTA encoder attention에서 제외한다. 실제 사람의 NONE 관계는 제외하지 않는다. scene RMS는 Stage1 값을 동결한다.
- 공동 loco 받침2개는 별도 물리 actor이며 관측 물체에 포함하지 않는다. 각 공동팀 loco의50%에 기존 높이 규칙을 적용한다. 받침 높이를 적용한 뒤 배치를 검사한다.
- `envSpacing: 10`으로 확대했다. 그룹별 yaw·중심을 무작위로 뽑고, 다른 그룹의 사람/상자·목표와 겹치면 배치만 다시 뽑는다. 초기 몸체는 환경 중심에서 각 축9.5m 이내에 두며, 팀 구성이나 RSI frame을 배치 실패 때문에 바꾸지 않는다.
- raw 관측1268, GraspView5072(사람별4개 view). GraspView는 해당 사람의 공동 payload만 옮기고 단독은 raw view를 쓴다. 사람4명은 하나의 Human SA4에서 처리한다.

### RSI·AMP·보상

공동 RSI는 팀별로 독립 frame을 선택하고 원본 두 사람의 역할을 무작위 배정한다. root/body/DOF/속도와 AMP reset 이력이 실제 사람 ID를 따른다. 공동 AMP 이력은 reset 현재 상태 반복, 단독은 크기 캐시에서 선택한 motion history다. AMP 전문가 원본80/후진10/옆걸음10은 locomotion family에만 적용한다.

공동 AT의 loco/pickUp/carryWith/putDown은0.4/0.1/0.4/0.1, 공동 ON_TOP은0.5/0.1/0.4/0이다. 단독은 기존0.4/0.1/0.4/0.1과 size-cache 가용성 규칙을 사용한다. 한 팀의 ON_TOP이 다른 AT팀의 putDown을 막지 않는다. Goal360 거리2.2~2.8m, putDown의 reference terminal XY 예외를 유지한다.

AT20와 ON_TOP 보상 수식은 기존과 같다. 반대 앵커 gate는 해당 공동팀에만 적용하고 단독은 자기 HOLDING을 사용한다. CPA 해제도 해당 pair에만 적용한 뒤 사람별 최대 위반을 계산하므로 다른 팀에 대한 충돌 페널티가 남는다.

`groups/solo_at`, `solo_ontop`, `joint_at`, `joint_ontop` 아래에 성공·holding gate·초기 RSI holding·loco pickup·사람별 holding·group/human transition 수·reset skill 비율을 기록한다. pair 통계는 팀 대표 한 명만 세고, human 통계는 두 사람을 각각 센다. 일반 학습 로그의 frame/시간과 함께 비교한다.

### 학습

이 서버의 검증 GPU는 **5 또는6만 사용**한다. 새 두 실험 모두 같은 Stage1 PTH에서 시작하며 기존2명 Stage2 checkpoint로 재개하지 않는다. GraspView 여부와4명 그룹 계약은 저장·재개·평가에서 확인한다.

```bash
TOKENHSI_GPU=5 STAGE1_CHECKPOINT=/path/to/CarryDistill_00030000.pth \
  bash tokenhsi/scripts/multi_agent/stage2_at20_4agent_train.sh 4 2048 8
TOKENHSI_GPU=6 STAGE1_CHECKPOINT=/path/to/CarryDistill_00030000.pth \
  bash tokenhsi/scripts/multi_agent/stage2_grasp_at20_4agent_train.sh 4 2048 8
```

짧은 검증은 같은 명령에 `MAX_ITERATIONS=1 OUTPUT_PATH=output/<실험명>_check`를 추가한다. 환경 수는2048을 유지한다. 재개는 `STAGE1_CHECKPOINT` 대신 해당 실험의 `RESUME_CHECKPOINT`를 지정한다.

### 로컬 viewer / headless 평가

```bash
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_at20_4agent_test.sh /path/to/stage2_at20_4agent.pth 4 16 8 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_grasp_at20_4agent_test.sh /path/to/stage2_grasp_at20_4agent.pth 4 16 8 3
```

화면 없는 평가는 `HEADLESS=1`. 기본은 `TASK_GRAPH=random_scenario`, `EVAL_SKILLS=loco`다. 강제 배치는 `two_pairs_at`, `two_pairs_ontop`, `one_pair_at`, `one_pair_ontop`, `all_solo_at`, `all_solo_ontop`, `mixed_groups`를 제공한다. `mixed_groups`는 공동1팀 AT + 단독2명 ON_TOP이다. `EVAL_SKILLS=loco,pickUp,carryWith,putDown EVAL_SKILL_PROBS=.4,.1,.4,.1`로 여러 skill을 평가할 수 있다. 공동 ON_TOP에서는 putDown을 제외하고 남은 가중치를 사용하며, 허용 동작이 하나도 없으면 오류를 낸다.

### 서버 VNC viewer

```bash
TOKENHSI_GPU=5 PORT=6085 bash tokenhsi/scripts/multi_agent/stage2_at20_4agent_vnc.sh /path/to/stage2_at20_4agent.pth 4 1 8 10
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/stage2_grasp_at20_4agent_vnc.sh /path/to/stage2_grasp_at20_4agent.pth 4 1 8 10
```

VNC wrapper와 내부 추론은 같은 GPU를 전달받는다. viewer 카메라는 활성 물체만 포함한다. 각 실험의 env/train YAML·학습·추론·VNC 파일과 기본 output을 분리했다. 공통 encoder/RSI/AMP 구현은 공유하며, 기존2명 경로는 별도 회귀 검증한다.


### 검증 상태 (2026-10-16)

4명 두 설정은 `cache_frozen_actor: true`로 동결 actor encoder의 출력을 rollout마다 한 번 계산해 PPO mini-epoch에서 재사용한다. 정규화 후 scene별 특징을 저장하고 minibatch 관측과 같은 순서로 선택한다. token 순열은 특징 계산 시 적용하며 human SA 순열은 매 forward마다 유지한다. Actor SA/head와 critic은 매 업데이트에서 계산·학습하고, RSI·AMP reference·보상·환경 수는 유지한다. MA 학습에서 사용하지 않는 `next_obses` 버퍼는 제거하며 bootstrap용 `next_values`는 유지한다.

실험명·checkpoint 형식·학습 명령은 그대로다. 이미 실행 중인 프로세스에는 적용되지 않으며 해당 실험의 저장 checkpoint에서 재개하면 적용된다. 새 실험을 만들거나 Stage1부터 다시 학습할 필요는 없다. 기존2명 네 설정에도 같은 actor 캐시를 활성화했다.

캐시 경로는 출력·gradient의 FP32 허용오차 내 동등성을 검사한다. 순열 난수 호출 횟수가 달라지므로 동일 seed의 전체 학습 궤적이 비트 단위로 같다는 의미는 아니다.

최적화 검사: CPU45개, 실제 관측의 CPU/GPU 출력·gradient 비교 및 최적화 전 checkpoint 재개 통과. GPU6/MPS·2048환경에서 Grasp 전후 각4 epoch를 실행했다. warmup 제외 업데이트 평균27.33→23.70초, 전체38.40→35.12초, PyTorch peak allocated29.50→27.09GiB였다. 공유 GPU에서의 짧은 비교이며 전체 VRAM 측정은 아니다. 기록은 `output/optimization_check/`에 있다.

CPU41개, 실제 물리26조합·부분 reset·역할/속도/AMP 매핑·비활성 actor 격리 검사 통과. GPU5/6에서 두 설정 모두2048환경·epoch2 학습/저장·strict 재로드를 확인했다. frozen actor encoder/scene RMS 불변, SA/head/critic 갱신 및 전체 tensor finite를 확인했다. 두 모델의 headless 평가와 VNC wrapper를 통한 실제 viewer 평가도 정상 종료했다. 사용자 로컬 데스크톱에서는 별도 실행하지 않았다. 장기 학습과 일반화 성능 평가는 아직 수행하지 않았다.

검증 기록: `output/four_agent_check/validation.json`, `checkpoints.json`, `physics_release/physics.json`, `cpu.log`. 마지막 검증용 체크포인트는 `output/four_agent_check/release/` 아래 각 실험의 `nn/`에 있다.
