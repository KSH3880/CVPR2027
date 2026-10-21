# 학생 반복 검사 기반 크기별 원본 RSI 샘플링

2026-10-02 구현. [확장 학생 분석](../../runs/reference_rsi/student_candidates_xyz20_65_seed1/ANALYSIS.md)의 원본 motion ID/phase 후보를 사용한다. distill/student rollout 자세나 정책 가중치를 초기화에 사용하지 않는다.

## 선택 순서

1. 기존 build에서 XYZ 각각 0.20~0.65m, 5cm 간격의 박스를 생성한다. 매 reset에서 논리/물리 할당을 먼저 정하고 재시도 중에도 크기와 할당을 유지한다. 후보 수에 따라 물리 크기를 재추첨하지 않는다.
2. 기존 templateRsi 비율로 각 agent의 RSI skill을 선택한다. loco/참조 비율 자체는 변경하지 않는다.
3. 실제 담당 박스 크기와 같은 skill에서 student 관측 n≥3, 모든 초기 Δv≤2m/s 후보를 선택한다. carryWith만 모든 관측의 손 거리 증가≤0.20m 조건을 추가한다. 손 거리 조건은 파지 성공을 보증하지 않는다.
4. pickUp/carryWith/putDown에는 확률 0.5로 90~98% 후보 pool을 요청하고, 나머지 0.5는 검증된 65~98% 전체 pool에서 선택한다. 전체 pool에도 90~98% 후보가 포함되므로 실제 선택 시점의 절반이 반드시 90% 이전이라는 뜻은 아니다. 그 크기·skill의 늦은 pool에 후보가 3개 이상일 때 우선 요청을 적용한다. 이것은 초기 비교 설정이며 최적 학습 비율로 검증된 값이 아니다.
5. 늦은 pool이 부족하면 같은 크기의 65~90% 후보를 우선 사용한다. 덜 늦은 후보도 없으면 남아 있는 같은 크기의 반복 저충격 후보를 사용한다. sit/climb은 65~98% pool 전체에서 선택하고 90~98% 추가 편향은 주지 않는다.
6. 반복 저충격 후보가 아예 없는 크기·skill은 **같은 크기·같은 skill의 원래 허용 RSI**로 돌아가고 누적 횟수를 기록한다. 작은 박스나 loco로 대체하지 않는다. 이 fallback은 저충격 검증된 초기화가 아니다.

후보 pool 안에서는 원본 motion clip 가중치를 유지한다. 같은 clip에서는 student 최대 Δv≤1 후보 가중치 2, 나머지≤2 후보 가중치 1이다. teacher tier로 학습 가중치를 정하지 않는다. 후보표를 시작 시 CPU에서 구성해 GPU에 전달하고 reset 샘플링은 크기별 Python 반복 없이 일괄 처리한다.

사람은 기존 참조 자세·속도로 초기화하고 pickUp/carryWith/putDown 박스도 같은 참조 시점의 위치·회전 및 선형/각속도로 초기화한다. 바닥 clamp가 적용되면 수직 속도를 조정한다. 기존 scene 배치 및 SIT/CLIMB 관통 검사도 유지한다. 이전 geometry bank의 추가 1cm 전신 screen은 적용하지 않는다. 이번 후보를 검사했던 물리 조건에 맞춘 선택이며 후보가 실제 학습에서 안정적이라고 확정하지 않는다.

## 설정과 실행

- 환경: `tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_student_rsi.yaml`
- 모드: `referenceRsi.mode: student_candidates`
- 후보: `../runs/reference_rsi/student_candidates_xyz20_65_seed1/student_low_impact_candidates.json`
- 새 성공 사각형 progress와 task별 full AMP 활성. 관찰과 성공 조건은 기존 구현 사용.
- task graph는 agent별 HOLDING 10%, SIT 10%, CLIMB 20%, HOLDING_ON_TOP 30%, HOLDING_AT 30%로 샘플링한다. RSI skill 혼합은 유지한다. CLIMB의 원래 제외 시점은 후보 선택 때 반복 검사된 원본 후반 시점으로 덮어쓸 수 있다. 단, 후보 없는 fallback은 원래 제외 규칙을 따른다.
- 기본 출력: `../runs/student_rsi`. 기존 checkpoint와 progress 보상 계약이 달라 일반 resume하지 않는다.

새 학습 명령(이번 작업에서는 실행하지 않음):

```bash
cd /home/hwanhee/koo_cvpr/stage1_fix
TOKENHSI_GPU=6 SEED=42 \
bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_student_rsi_train.sh 2 2048 4
```

GPU 6/7만 허용한다. 평가 wrapper는 `approach_scenario_stage1_student_rsi_test.sh`이며 평가에서 학습 후보 sampler를 사용하지 않는다. 실행 중인 학습/평가는 변경하지 않았다.

TensorBoard `sampling/reference_cumulative/` 아래 attempts, late_requests, late_selected,
same_size_general_fallback, missing_size_skill_original, skill별 draws/missing_original/general_fallback을 확인한다. 누적 수는 reset 재시도를 포함한 시도 수다. 원래 RSI fallback과 덜 늦은 검증 후보 fallback은 서로 다르다.

## coverage 및 제한

원래 62,949후보에서 carry 손 거리 조건으로 741개를 제외한다. 후보 pool 크기 수는
pickUp 997 / carryWith 941 / putDown 936 / sit 905 / climb 962 (각 1000개 중).
기하 reset 재시도 때문에 실제 accepted 분포가 pool의 분포와 같다고 가정하지 않는다.

CLIMB는 낮은 박스 Δv와 상판 위 성공 유지가 다르다. 이번 sampler는 반복 저충격 시점 선택을 구현했으며 사람 root 높이 감소를 정상 하강/실패로 분리하지 않는다. 학습 중 CLIMB 실제 성공·유지 시간 및 fallback 비율을 확인한다. 기존 RSI/보상과 동일 조건 비교 없이는 수렴 가속을 주장하지 않는다.

## 검증

task 비율 변경 후 `validate_unified_env`도 student_candidates의 YAML 분포를 허용하도록 수정했다. 음수·합계 오류 등 sampler 계약은 계속 검사한다. 실제 student_rsi YAML 검증 및 기존 unified/random-binding/student RSI CPU 회귀 23개 통과. GPU 학습 실행은 별도로 확인해야 한다.

CPU 회귀 42개 통과: 크기 유지, 같은 skill 원래 RSI fallback, 늦은 후보 부족 시 덜 늦은 pool, n/Δv/carry 손 거리 조건, putDown 정상 release의 일괄 제외 방지, 원본 clip 가중치 및 student 품질 가중치, climb 끝부분 편향 방지, task AMP 및 기존 geometry RSI/progress 회귀.
실제 후보 fingerprint·mesh hash·1000개 크기를 CPU MotionLib stub으로 확인했다. 이는 Isaac Gym 실행 검증을 대체하지 않는다. shell 문법/변경 파일 공백 확인. GPU는 현재 작업들로 사용 중이므로 새 GPU 실행과 장기 학습은 하지 않았다.

로그: `../runs/reference_rsi/student_rsi_sampler_tests.log`, `student_rsi_real_bank_check.log`.
