# 랜덤 박스 · 참조 RSI 두 방식

학습 가속 목표, 기하 기준의 한계, 사용자 실행 impact_sweep 결과와 후속 제안은
[논의·검사 아카이브](reference_rsi_archive.md)에 정리했다. 아래 기하 후보 부족을
학습 가능한 박스 크기 한계로 해석하지 않는다. 현재 프로세스 상태는 직접 확인한다.

## 기준과 현재 결과

기준은 `/home/hwanhee/koo_cvpr/stage1_fix`에서 실행 중인 PID 289953의
`approach_scenario_stage1_unified_shared_random_binding.yaml`이다. 기존 실행은
종료·재시작하지 않았다. ksh 저장소와 baseline은 수정하지 않았다.

두 실험은 기존 2인·4물체·2048환경·무작위 binding·shared EdgeEncoder·혼합 AMP·
near-start·보상·skill 초기 비율을 유지하고 `referenceRsi`만 추가한다.
CLIMB의 loco/climb 선택은 기준과 같은 50/50이다. 과거 slow_near_start의 35/65를
동시에 이식하지 않아 변경축을 분리했다.

- **time_late:** climb 모션을 뽑은 경우 70%는 시간상 65~98%의 안전한 후보,
  30%는 원래 RSI 허용 구간의 안전한 후보에서 뽑는다.
- **semantic_late:** climb 모션을 뽑은 경우 70%는 한 발 이상이 상자 XY 영역에 들어가고
  발 body origin 높이가 상판 -7cm~+30cm인 안전한 후보, 30%는 원래 허용 구간에서 뽑는다.
  이것은 참조 모션의 공간적 진행 판정이며 정책 rollout 상태나 완전한 CLIMB 성공 판정이 아니다.
- 후반 CLIMB 후보는 기존 slow_near_start처럼 일반 RSI 제외 구간을 넘어갈 수 있지만,
  기하 검사를 통과해야 한다. 다른 skill은 원래 제외 구간을 유지한다.

distill pth는 **probe에서 action 계산에만** 쓴다. 학생 학습은 scratch이며 teacher
weight·optimizer·KL loss·rollout 초기 상태 풀을 사용하지 않는다. teacher의 3물체/rescue
보상 메타데이터와 target 4물체 보상은 다르다. probe에서만 동일 모델 구조를 strict weight
load로 확인하고 reward/scene 계약을 분리했으며, 일반 학습·평가 로더는 우회하지 않는다.
90% 성공률은 이번 검사에서 재검증하지 않았다.

GPU 6, 동일 teacher, seed 0, 2048환경 × 2회 reset × 첫 10 control step의 예비 결과:

| 초기화 | 박스 속도 >5m/s 장면 | 박스 Δv >3m/s 장면 | 최대 박스 속도 |
|---|---:|---:|---:|
| 현재 기준 | 175/4096 (4.272%) | 382/4096 (9.326%) | 15.73m/s |
| 안전 검사 없이 CLIMB 후반만 70% | 250/4096 (6.104%) | 513/4096 (12.524%) | 24.03m/s |
| 안전 시간 후반 + 전 장면 검사 | 2/4096 (0.049%) | 25/4096 (0.610%) | 13.17m/s |
| 안전 행동 단계 + 전 장면 검사 | 7/4096 (0.171%) | 37/4096 (0.903%) | 5.93m/s |

최종 두 probe에서 NaN·첫 10step 종료는 0이고 부분 reset의 비대상 scene/AMP 이력 유지도
통과했다. 이 수치는 scene별 최대값이다. 5m/s·3m/s는 진단 threshold이며 폭발의 확정 판정이나
최종 task 성공률이 아니다. 반복/seed 산포 및 학습 수렴은 미검증이다.

## 후보 부족 — 반드시 해석에 포함

참조 데이터는 기존 `dataset_loco_sit_carry_climb.yaml`의 모션과 물체 궤적이다.
X/Y 0.40~0.65m, Z 0.25~0.55m의 252개 크기에 대해 반프레임 간격으로 검사했다.

| skill | 원래 허용 구간에 안전 후보가 있는 크기 |
|---|---:|
| sit | 252/252 |
| climb | 252/252 |
| pickUp | 252/252 |
| carryWith | 14/252 |
| putDown | 79/252 |

이는 실제 충돌체를 이용한 **보수적 기하 screen** 결과다. 구체/캡슐 및 발 mesh 정점을
검사하지만 mesh triangle 내부만 교차하는 사례와 물리 안정성을 완전히 보장하지 않는다.
원본 모션의 양손 간격이 큰 박스와 맞지 않아, 어떤 크기는 시간을 바꿔도 후보가 부족하다.

후반 후보만 없으면 같은 크기·skill의 일반 후보로 대체한다. 해당 skill 전체 후보가 없으면
config에 명시된 `empty_skill: loco`에 따라 **같은 크기의 loco 참조 RSI**로 대체하고 계수한다.
박스 크기를 다시 뽑거나 teacher 상태를 넣지 않는다. 이 대체를 'carry RSI 유지'로 해석하면 안 된다.

실제 채택된 carryWith 초기화는 기준 2047/8192 agent에서 시간 방식 106/8192,
행동 단계 방식 108/8192로 줄었다. 시간 방식의 CLIMB 채택 시점 중 65~98%는 70.6%,
행동 단계 방식에서는 40.9%였다. 속도 급증 감소에는 skill 분포 변화도 포함되므로,
**이 결과만으로 수렴 개선이나 두 방식의 우열을 확정하지 않는다.**

carry 후반 RSI를 큰 박스에도 최대한 유지하려면 다음 변경축은 양손/팔 참조 자세 보정 또는
해당 크기에 맞는 추가 참조 모션이다. 그 작업 전에 현 결과를 기준으로 삼고, 다른 크기를
재추첨하거나 관통 threshold만 크게 풀어서 후보 수를 늘리지 않는다.

## 초기화 구현

1. reset 시작 시 논리→물리 물체 배정을 정하고 scene 재추첨 동안 유지한다.
2. 담당 물리 박스의 실제 크기로 후보표를 조회한다. 모션 선택은 가능한 clip 사이에서 원래
   clip weight를 재정규화하고, 선택한 clip 안에서 시점을 균등 샘플링한다.
3. 원본 모션에서 사람 자세·속도와 AMP history를 초기화한다. teacher history는 사용하지 않는다.
4. carry 계열 박스의 위치 보간 및 quaternion slerp와 일치하는 선속도/월드 각속도를 복원한다.
   바닥 높이 clamp가 적용된 경우 수직 속도를 0으로 한다. sit/climb 물체는 정지한다.
5. 기존 box-box/goal 검사와 함께 **모든 박스 × 두 사람의 모든 충돌체**를 검사한다.
   깊이 1cm 초과 관통은 scene 후보 재추첨으로 처리한다. 정상 파지의 얕은 접촉은 허용한다.
6. accepted 모션 ID/시점과 AMP 이력 연결은 기존 reset 집계 경로를 사용한다.

일반 평가에서는 기존 loco 시작을 사용하고 안전 RSI 모션 편향을 적용하지 않는다.
후보표와 새 관통 screen도 일반 평가에서는 비활성화해 기준 평가 분포를 유지한다.
training resume은 `reference_rsi.config`와 후보표 SHA256을 검사한다. 실험 시작 후 후보표를
덮어쓰지 말고, 새 후보표는 새 경로와 새 실험으로 관리한다.

## 실행 명령

아래 명령은 `cd /home/hwanhee/koo_cvpr/stage1_fix` 후 실행한다. GPU는 6/7만 허용한다.
현재 후보표와 pth는 이미 준비되어 있어 재생성/재복사가 필요하지 않다.

CPU 후보표 신규 생성 예시(기존 실험 입력과 다른 파일명 사용):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  tokenhsi/scripts/multi_agent/build_reference_rsi_bank.py \
  --output ../runs/reference_rsi/reference_bank_new.npz
```

distill 충돌 probe(학습 없음, training RSI 사용):

```bash
TOKENHSI_GPU=6 RSI_PROBE_OUTPUT=../runs/reference_rsi/probe_time_new \
  bash tokenhsi/scripts/multi_agent/probe_reference_rsi.sh time_late
TOKENHSI_GPU=6 RSI_PROBE_OUTPUT=../runs/reference_rsi/probe_semantic_new \
  bash tokenhsi/scripts/multi_agent/probe_reference_rsi.sh semantic_late
```

기본 probe는 2048환경·4회 reset·10step이며 `RSI_PROBE_ROUNDS`, `RSI_PROBE_STEPS`,
`TASK_GRAPH=climb|holding_at|holding_ontop`으로 검사 범위를 좁힐 수 있다.
`baseline`은 현재 289953의 설정, `raw_time_late`는 probe에서만 기존 CLIMB 후반 70%를
적용하고 새 안전 필터/속도 복원을 사용하지 않는 비교다. 각 결과는 새 경로에 저장한다.

1-iteration scratch 학습 확인(teacher 사용 없음):

```bash
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=../runs/reference_rsi/time_check_new \
  bash tokenhsi/scripts/multi_agent/approach_reference_rsi_time_late_train.sh
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=../runs/reference_rsi/semantic_check_new \
  bash tokenhsi/scripts/multi_agent/approach_reference_rsi_semantic_late_train.sh
```

위 candidate 부족을 반영해 파일럿을 진행할 경우, 두 train wrapper는 기본 3000 iteration이다.
기존 학습과 다른 output을 사용하고 GPU 점유를 확인한다. 이번 작업에서는 장기 학습을 실행하지 않는다.

일반 시작 평가와 가드가 연결된 VNC:

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_reference_rsi_time_late_test.sh /absolute/path/student.pth
TOKENHSI_GPU=6 PORT=6110 bash tokenhsi/scripts/multi_agent/approach_reference_rsi_time_late_vnc.sh /absolute/path/student.pth
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_reference_rsi_semantic_late_test.sh /absolute/path/student.pth
TOKENHSI_GPU=6 PORT=6111 bash tokenhsi/scripts/multi_agent/approach_reference_rsi_semantic_late_vnc.sh /absolute/path/student.pth
```

VNC는 `scripts/tokenhsi_gui.sh`의 UUID guard를 사용한다. 이번 검증은 headless만 수행했으며
VNC 화면은 실행하지 않았다.

## 산출물과 검증

### 크기별 물리 충격 검사 (impact_sweep)

`impact_sweep`은 후보표의 1cm 관통 필터 없이, 기존 크기 범위와 실제 참조 모션을
사용해 검사한다. 상자 관련 5종의 RSI 시간을 지정 구간에서 샘플링하며 원래 RSI 제외
구간도 검사 대상으로 포함한다. loco 시점과 기존 장면 배치 검사는 유지하므로,
전 조합을 동일 횟수로 검사하는 exhaustive sweep은 아니다. carry 박스의 참조 속도를
복원하고 distill 정책으로 행동한다. 제어 방식의 영향이 포함되며 자세 유지 제어와의
대조 검사는 아직 구현하지 않았다.

```bash
TOKENHSI_GPU=6 RSI_PROBE_ROUNDS=32 RSI_PROBE_STEPS=20 \
  RSI_PROBE_PHASE=0.65,0.98 \
  RSI_PROBE_OUTPUT=../runs/reference_rsi/impact_late_seed0 \
  bash tokenhsi/scripts/multi_agent/probe_reference_rsi.sh impact_sweep
```

`agents.jsonl`에는 실제 담당 박스별 크기, 행동, 모션 ID/phase, 첫 2 control step의
최대 Δv, 전체 최대 속도/각속도, carry 계열의 참조 위치 이탈을 기록한다.
`by_size_skill_phase.csv`는 크기·행동·10% phase 구간별 표본 수와 충격 p50/p95/최댓값을
집계한다. 종료율은 agent별 실패가 아닌 **해당 장면의 종료율**이다. >3m/s Δv와 >5m/s
속도는 진단용 집계이며 안전 후보의 확정 threshold는 아니다. 소표본 또는 누락 조합을
안전 판정으로 해석하지 않는다. 이 실행은 학습하지 않으며 후보표도 자동 변경하지 않는다.

새 옵션은 Python·셸 문법 검증을 수행했으며 GPU 실행 결과는 사용자 실행 후 확인한다.

- 후보표/크기별 수량: `../../runs/reference_rsi/reference_bank_mesh.{npz,json}`.
- 최종 teacher 검사: `../../runs/reference_rsi/probe_{time_late,semantic_late}_mesh/{summary.json,scenes.jsonl,run.log}`.
- 기준 teacher 검사: `../../runs/reference_rsi/probe_baseline/`.
- 초기 불완전 screen 결과: `../../runs/reference_rsi/probe_{time_late,semantic_late}/`; 최종 결과와 구분한다.
- 짧은 학생 학습: `../../runs/reference_rsi/train_check_{time_late,semantic_late}/`.
- 두 학생 checkpoint의 일반 시작 평가: `../../runs/reference_rsi/eval_check_{time_late,semantic_late}/metrics/`.
  각각 64환경·32step·3회 평가가 정상 종료했다. 1-iteration 학생의 성공률은 0이며,
  checkpoint 로드와 평가 실행 경로만 확인한 결과다.
- CPU 테스트: `test_reference_rsi.py`, 기존 shared-random/unified 테스트 포함 20개 통과.

연구 비교는 같은 크기·과제 분포와 일반 시작 평가를 유지해야 한다. 후보 대체율, skill별 실제
초기화 비율, first-step/10-step 충격과 종료, 성공률 대비 학습 frame·벽시계 시간을 함께 기록한다.
