# 후반 hybrid RSI

실행 가능한 별도 실험이다. 기반은 student_rsi이며 크기·reward·success_region progress·조건부 AMP·shared edge encoder와 task 비율(HOLDING/SIT/CLIMB/ON_TOP/AT=10/10/20/30/30%)을 유지한다. 기존 실행에 자동 반영되지 않는다.

## 선택 방식

- XYZ 각 축 20~65cm/5cm 간격의 실제 물체 크기를 먼저 고정한다. 후보 실패·성공률로 크기나 task를 재추첨하지 않는다.
- pickUp/carryWith/putDown: 원본 clip당 구간별 8개 시점의 PD 유지 물리 검사를 초기 실행에서 준비한다. 선택은 **80~95 → 70~80 → 60~70 → 50~60%** 중 후보가 남은 가장 후반 구간을 사용한다. 50% 이전·95% 이후는 제외한다. 예비 구간은 학습 중 탈락에 대비해 함께 검사한다.
- 모든 구간이 비면 동일 크기·task에서 loco 접근 시작을 사용한다. ON_TOP putDown 후보는 운반 물체 XYZ와 받침 XYZ의 정확한 조합으로 구분한다. ON_TOP RSI는 loco/pickUp/carryWith/putDown=50/10/30/10%다.
- 초기 PD 0.1초 검사: 유한값, root 속도 변화≤3m/s, 박스 속도≤5m/s, root·박스 이동≤0.3m. 실제 크기·밀도·floor 및 ON_TOP 받침을 반영한다. 캐시는 data/asset/physics/spec/code fingerprint로 구분하며 실제 자산에 존재하는 크기 조합만 준비한다.
- 현재 학습 정책의 실제 reset 이후 첫 2 control step을 관측한다. 위 조건에 더해 step간 박스 Δv≤2m/s, carryWith 손-박스 평균 거리 증가≤0.2m를 검사한다. 후보 관측 3회 이상에서 불안정 관측이 하나라도 있으면 해당 후보를 제외한다. 성공률은 후보 입장 조건으로 사용하지 않는다.
- 후반 구간 내 반복 검증된 후보가 있으면 70% 확률로 우선 선택하고 나머지는 해당 구간 전체 물리 후보에서 선택한다. 원본 clip 가중치는 후보 시점 수와 무관하게 유지한다. 1000 epoch마다 관측 창을 초기화하여 개선된 현재 정책에서 다시 검사한다.
- SIT/CLIMB은 기존 학생 반복 검사 후보 파일을 유지한다. 새 후반 단계 탐색은 carry 계열에 적용한다. 별도 teacher 정책은 사용하지 않는다.

## 실행

저장소 루트에서 실행한다. 최초 시작에는 물리 후보 검사 시간이 들며 이후 동일 조건은 캐시를 사용한다. 기존 학생 후보 파일 `../runs/reference_rsi/student_candidates_xyz20_65_seed1/student_low_impact_candidates.json`이 필요하다.

```bash
# 점유 확인 후 GPU 6 본학습, scratch / 2048환경
nvidia-smi
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_train.sh 2 2048 4

# 별도 짧은 검증
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_scenario_stage1_hybrid_late_rsi_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_train.sh 2 2048 4

# 로컬 viewer
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_test.sh <checkpoint.pth> 2 16 4 3

# 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_test.sh <checkpoint.pth> 2 16 4 3

# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_vnc.sh <checkpoint.pth> 2 1 4 10

# 동일 hybrid 실험 이어 학습: 같은 seed와 물리/config/후보 목록 필요
TOKENHSI_GPU=6 SEED=42 RESUME_CHECKPOINT=<hybrid_checkpoint.pth> bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_train.sh 2 2048 4
```

기본 output은 `output/approach_scenario_stage1_hybrid_late_rsi/<run>/`이다. 기존 student/size_rsi checkpoint의 학습 resume는 후보 계약이 달라 거부한다. 평가에서는 학습용 후보 검사를 수행하지 않고 loco에서 시작한다. 본학습 전에 셸에 남은 MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT를 확인한다.

## 진단

2026-10-03에는 사용자 요청으로 기존 GPU 6 학생 실행을 종료하고 위 hybrid 본학습을 seed42·2048환경으로 scratch 시작했다. 실행 기록은 [latest_launch.json](../output/approach_scenario_stage1_hybrid_late_rsi/latest_launch.json)에 있으며 실제 PID·train.log·run_directory를 확인할 수 있다. 백그라운드 세션을 분리했으며 시작 확인에서 TensorBoard epoch 1~6 진행과 물리 캐시 재사용을 확인했다.

같은 날 중단된 본학습을 1000 epoch checkpoint에서 재개했다. 이전 로그는 1049까지였으므로 1001부터 다시 진행한다. MPS 정상 연결·optimizer/RSI 상태 복원·1001~1003 진행을 확인했고 실행 기록에 이전 run과 resume checkpoint를 함께 남겼다. 이어진 추이 집계에는 이전 run의 1~1000과 새 run의 1001 이후를 사용해 재실행 구간을 중복 집계하지 않는다.

500 epoch마다 `<run>/diagnostics/`에 PID별 CSV를 저장한다. 후보 관측·관측 창은 checkpoint에 저장하며 episode 진단은 새 실행 세션에서 다시 집계한다.

- `hybrid_xyz_episodes_<PID>.csv`: 모든 task의 source XYZ별 누적 starts/steps/completed/final_success/ever_success/initial_success/reference_starts/loco_starts. 종료 성공률=final_success/completed, 한 번이라도 성공률=ever_success/completed. 1000 epoch 구간 비교 시 동일 PID의 누적 완료·성공 count 차이로 계산한다.
- `hybrid_candidates_<PID>.csv`: 현재 관측 창에서 실제 관측된 후보의 skill/source XYZ/support XYZ/clip/phase/band/observations/failures. 관측되지 않은 후보를 실패로 해석하지 않는다. 전체 PD 후보 목록은 `output/rsi_cache/hybrid_late/*.json`에 있다.
- TensorBoard의 기존 edge bias·task progress·성공 진단도 유지한다. 물리 통과와 실제 task 성공은 별개다. 학습 속도·수렴 개선은 장기 비교 전까지 미확인이다.

구현: `utils/hybrid_late_rsi.py`, `env/tasks/multi_agent/hybrid_rsi_screen.py`, `hybrid_rsi_runtime.py`. 검증: `tests/test_hybrid_late_rsi.py`와 기존 RSI/unified 회귀.

2026-10-03 검증: CPU 39개 통과. GPU 6·2048환경 짧은 학습과 checkpoint 저장, 동일 후보 cache hit 및 resume 관측 누적(3202→5079), 16환경·30step headless 평가 경로를 확인했다. 이는 실행 검증이며 장기 학습 성능 비교는 아니다. 로컬 GUI·VNC는 shell 문법과 연결 경로를 확인했다.
