# 2번 후보 추출 · 3번 현재 학생 재검사

수정된 measurement_version=2의 사용자 64회 재검사도 완료됐다.
[최신 학생 검사 분석](../../runs/reference_rsi/student_candidates_seed1_v2/ANALYSIS.md)을
우선 읽는다. 예비 후보 42,555개 관측, 7,250개 미측정. 3회 이상 관측·모든 첫 Δv≤2m/s
후보는 16,294개이며 이는 파지/회전/학습 성공을 확정한 안전 후보 수가 아니다.

사용자 실행의 2048환경×64회 학생 검사 분석은
[ANALYSIS.md](../../runs/reference_rsi/student_candidates_seed1/ANALYSIS.md)에 있다.
해당 실행의 손 거리 증가 지표는 초기 pose 기준 문제로 파지 판정에서 제외한다.
이후 검사 코드의 measurement_version=2는 새 참조 FK로 초기 거리를 계산한다.

2026-10-02 구현. 학습 가속 논의의 5번은 실제 비교 학습이며 이 문서의 두 명령은
학습하지 않는다. 실행 중인 학습에는 적용되지 않는다.

## 2번: 기존 distill 검사에서 예비 원본 시점 추출 (CPU)

```bash
cd /home/hwanhee/koo_cvpr/stage1_fix
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  tokenhsi/utils/impact_candidates.py \
  --source ../runs/reference_rsi/impact_late_seed0/agents.jsonl \
  --reference-bank ../runs/reference_rsi/reference_bank_mesh.npz \
  --output ../runs/reference_rsi/impact_candidates_v2.json
```

기본 기준은 첫 2step Δv ≤1m/s preferred, 1~2m/s conditional이며 2m/s 초과를 제외한다.
선택 기준은 미검증 제안이다. 원본 크기·skill·motion ID·정확한 phase를 저장한다.
박스 크기를 줄이거나 teacher rollout 상태를 저장하지 않는다. scene 종료는 동료 때문일
수 있어 이 추출에서는 종료 여부로 필터하지 않는다. finite 상태만 사용한다.

`impact_candidates_v1.json`은 구현 검증 중 이미 생성했다. 총 49,805개이며 preferred
34,169개, conditional 15,636개다. 위 명령은 덮어쓰지 않도록 새 v2 경로를 사용한다.
생성 파일과 `.summary.json`을 함께 보관한다. 단일 관측으로 고른 **예비 후보**이며
파지·수직 튕김·회전까지 확인된 학습용 안전 후보표는 아니다.

## 3번: 후보를 원본 RSI로 재생하고 학생 정책으로 검사 (GPU 6/7)

먼저 GPU 점유를 확인한 뒤 headless 검사만 실행한다. 아래는 PID 1184210 학습에서
저장된 16500회 학생 checkpoint를 고정해서 검사하는 예시다. 더 최신 번호 파일을
사용하려면 checkpoint 인자만 바꾼다. wrapper는 시작 때 해당 pth를 output 안으로 복사해
전체 검사 동안 고정한다. 소스 학습 프로세스에는 연결·수정하지 않는다.

```bash
TOKENHSI_GPU=6 SEED=1 RSI_PROBE_ENVS=128 \
  RSI_PROBE_ROUNDS=1 RSI_PROBE_STEPS=20 \
  RSI_PROBE_OUTPUT=../runs/reference_rsi/student_candidates_smoke_user \
  bash tokenhsi/scripts/multi_agent/probe_student_candidates.sh \
  /home/hwanhee/koo_cvpr/stage1_fix/output/shared_random_binding_resume1099002_full_amp_7500/SharedRandomBindingResume1099002FullAMP7500_01-22-53-00/nn/SharedRandomBindingResume1099002FullAMP7500_00016500.pth \
  /home/hwanhee/koo_cvpr/runs/reference_rsi/impact_candidates_v2.json
```

정상 완료 후 표본을 늘린 검사:

```bash
TOKENHSI_GPU=6 SEED=1 RSI_PROBE_ENVS=2048 \
  RSI_PROBE_ROUNDS=64 RSI_PROBE_STEPS=20 \
  RSI_PROBE_OUTPUT=../runs/reference_rsi/student_candidates_seed1 \
  bash tokenhsi/scripts/multi_agent/probe_student_candidates.sh \
  /home/hwanhee/koo_cvpr/stage1_fix/output/shared_random_binding_resume1099002_full_amp_7500/SharedRandomBindingResume1099002FullAMP7500_01-22-53-00/nn/SharedRandomBindingResume1099002FullAMP7500_00016500.pth \
  /home/hwanhee/koo_cvpr/runs/reference_rsi/impact_candidates_v2.json
```

GPU 번호는 6/7만 허용한다. 서로 다른 검사는 순서대로 실행한다. 완료 output을
덮어쓰지 않으며 실패 결과도 보관하고 재실행 시 새 output 경로를 사용한다.
현재 shared/full AMP 학생 구조와 환경 설정을 사용하고 일반 checkpoint 계약을 먼저
검사한다. distill 전용 구조나 메타데이터 우회를 학생 로드에 사용하지 않는다.
`RSI_STUDENT_CFG`로 다른 학생의 환경을 지정하려면 pth와 구조/보상 계약이 일치해야 한다.

## 검사에서 실제로 달라지는 것

- 박스의 실제 크기에 맞춰 원본 motion ID/phase를 재생한다. 물체 binding은 시점 선택
  전에 고정하고 배치 재추첨 동안 유지한다. 기존 과제·skill 선택 비율은 유지한다.
- 가능한 clip 사이에서 원래 clip weight를 유지하고, clip 안에서는 preferred 후보에
  conditional보다 두 배 가중치를 준다. 이는 검사 샘플 배분이지 학습 sampler가 아니다.
- 해당 size/skill 후보가 없으면 같은 크기·같은 skill의 기존 RSI를 검사한다.
  박스 크기를 바꾸거나 loco로 대체하지 않는다. `candidate_id: null`로 구분한다.
- 참조 사람/박스 상태로 초기화한다. carry 박스 선/각속도를 함께 복원한다.
- 기존 장면 배치와 sit/climb reset 검사는 유지하며 새 1cm 후보 screen은 사용하지 않는다.
- 초기 자세 유지 제어 대조는 아직 없다. 학생의 deterministic action을 사용한다.

## 출력과 판정

- `candidate_validation.json`: 모든 예비 후보별 student_n, Δv 최대값/2m/s 초과 수,
  수직 Δv, 상승량, 각속도, 손–박스 중심 거리 증가, 사람 root 높이 감소.
  재생되지 않은 후보는 `unmeasured`, 측정 후보도 `observed_not_safety_certified`다.
- `agents.jsonl`: 개별 reset의 candidate_id와 추가 물리 지표.
- `summary.json`: policy_role=student, 고정 pth SHA, 후보 재생 수와 전체 장면 통계.
- `by_size_skill_phase.csv`: 크기·skill·시간 구간별 집계. 후보 없는 원래 RSI도 포함되므로
  후보만 평가할 때는 agents.jsonl의 candidate_id 또는 candidate_validation.json을 사용한다.
- `student_snapshot.pth`: 검사 내 고정 정책. train checkpoint로 새로 학습한 파일이 아니다.

손–박스 중심 거리 증가는 이탈의 대리 지표이지 접촉/파지 성공 판정이 아니다.
상승량과 회전은 정상 참조 동작도 포함한다. 사람 root 높이 감소도 넘어짐의 확정 판정이
아니다. 후보별 표본 수를 확인하고 수직/회전/이탈 threshold를 검토한 뒤 별도 학습 후보표를
확정해야 한다. 이 명령은 자동으로 학습용 안전 판정이나 70/30 학습 설정을 만들지 않는다.

## 구현 검증

CPU 후보 분류·정확한 원본 시점·중복/비유한 제외·없는 조합의 크기/skill 유지와 기존
RSI 회귀 테스트 24개 통과. 최초 작은 GPU 검사에서 minibatch 크기 제약을 발견해
검사 전용 크기 4096으로 AMP minibatch 제약까지 보정했다. GPU 완료 결과는 changelog에 실제 실행 결과로 기록한다.

GPU 6, 128환경×1회×20step, 16500회 학생 checkpoint로 정상 완료했다.
124개 후보 일치, NaN 0, 부분 reset 비대상 상태/AMP 보존 통과.
일치 후보 중 16개가 학생 첫2step Δv 2m/s를 넘었다. 따라서 teacher 예비 후보를
그대로 안전 판정하지 않는다. 결과는 `../../runs/reference_rsi/student_candidates_smoke_dev_v4/`.
이는 실행 검증이며 전체 후보의 학생 검증이나 수렴 검증이 아니다.
