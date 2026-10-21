# 5개 task의 Z별 checkpoint 평가

`evaluate_z_sweep.sh`는 평가 전용이며 학습·resume용 config가 아니다. 학생 RSI checkpoint를 기존 strict loader로 읽는다. 기본 checkpoint는 PID 2978991의 11000 epoch 저장본이다.

## 비교 조건

- 모든 박스 X/Y=0.40/0.40m, yaw=0. 기본 Z=0.40m. `--x`, `--y`, `--base-z`로 변경 가능하다.
- HOLDING/AT/ON_TOP/SIT/CLIMB: source 세로 크기 Z=0.20..0.65m, 0.05m 간격; source 밑면의 바닥 위 높이=0..0.50m, 0.10m 간격.
- 여기서 source는 SIT/CLIMB에서는 앉거나 올라갈 대상 박스를 뜻한다.
- ON_TOP: source와 받침 세로 크기를 별도로 변경; 받침 밑면 높이도 별도 변경한다. source sweep 때 받침은 기본 Z, 받침 sweep 때 source는 기본 Z다.
- AT: goal 높이도 별도 변경한다. `goal_bottom_z`는 source를 goal에 놓았을 때 밑면 높이이며, 실제 goal 중심 Z는 이 값+source Z/2다.
- 박스·goal을 높일 때 기존 두 source/two target 충돌 받침을 사용한다. 공중 자유낙하 초기화로 높이를 비교하지 않는다. 받침은 기존 0.4×0.4×0.02m·중력 비활성·고밀도 동적 actor다. 발은 기존 platform collision filter를 따른다.
- canonical binding은 평가 프로세스에서만 적용한다. 각 scene 두 agent가 같은 task를 수행하고 서로 다른 source/goal/support를 사용한다. 원본 YAML의 disjoint binding 계약·학습 sampler를 변경하지 않는다.
- 시작은 loco이며 agent 높이는 변경하지 않는다. source는 각 agent root XY+[0,1.5]m, goal/ON_TOP 받침은 root XY+[0,3]m이다. 나머지 상자는 root XY+[0,-1.5]m에 둔다. 모든 조건에서 같은 배치 규칙을 사용하지만 시작 loco frame은 무작위다.
- 기본 102조건×16scene=1632 평가 환경, 600step×4회. 조건당 64scene/128agent episode. 2048환경 규칙은 학습에 적용하며 이 실행은 평가다.
- 박스 밀도는 기존 100kg/m³이므로 Z에 따라 질량도 바뀐다. 크기만의 순수 인과 효과로 해석하지 않는다. ON_TOP 받침은 기존처럼 동적 박스다.

## 실행

저장소 루트에서 실행한다. 실행 전 `nvidia-smi`로 GPU 점유를 확인한다. 새 output 경로만 허용해 결과를 덮어쓰지 않는다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/evaluate_z_sweep.sh \
  /home/hwanhee/koo_cvpr/runs/student_rsi/ApproachScenarioStage1StudentRsi_02-18-35-00/nn/ApproachScenarioStage1StudentRsi_00011000.pth \
  --output output/student_rsi_z_sweep_11000_xy40_seed42
```

검증용은 `--smoke --scenes-per-condition 1 --repeats 1 --episode-length 30`을 붙이고 별도 output을 사용한다. smoke의 짧은 성공률은 본평가 결과가 아니다. `--prepare-only`는 GPU 없이 조건·config만 생성한다. 이 평가는 headless 전용이며 학습/local viewer/VNC wrapper를 제공하는 학습 실험이 아니다.

## 결과

- `manifest.json`: checkpoint, 단위, 조건, 배치·시작 상태·질량 조건.
- `evaluation_config.yaml`: 평가에 사용한 snapshot. `evaluationBoxSizes`는 실제 생성 asset 크기이며 `--eval`이 없으면 거부된다.
- `trials.csv`: condition/repeat/scene/agent별 초기·도달·종료 성공과 종료 step, 실제 초기 박스/goal 높이.
- `summary.csv`, `summary.json`: 조건별 count/rate. 성공 판정은 기존 `relation_runtime.done`의 required goal을 그대로 사용한다. AT/ON_TOP은 HOLDING 부가 edge 성공과 혼동하지 않는다.
- `success_by_z.png`: task별 크기/높이 그래프. 점선은 한번이라도 도달, 실선은 종료 시 성공이며 agent별 비율이다. summary는 두 agent 모두 성공한 scene 비율과 해당 이항 Wilson 구간도 별도로 기록한다. 같은 scene의 두 agent는 상관될 수 있고 시작 frame도 무작위이므로 작은 차이를 확정적 순위로 해석하지 않는다.

초기 성공도 별도 기록하며 도달률에는 포함한다. 평가 시작부터 첫 종료까지 한 episode만 수집하고, 이미 종료된 scene의 reset 후 episode는 다시 집계하지 않는다.

## 2026-10-03 완료 결과

[전체 표·그래프](../output/student_rsi_z_sweep_11000_xy40_seed42/REPORT.md). 조건당 128agent 기록, 합계13,056회 완료. 실제 초기 source 밑면 높이 오차<1e-6m, 초기 성공0, 중복/누락0. CPU20개·GPU smoke·600step 본평가 통과.

종료 성공률은 HOLDING Z20cm=0%, 25~65cm=90.6~100%; ON_TOP source40cm=75%/65cm=19.5%, 받침30cm=75.8%/65cm=26.6%. AT 바닥 목표는0%, goal 밑면20cm=49.2%. SIT 최고1.6%, CLIMB0%는 loco 시작 조건의 결과다. 전체 조건과 도달/종료 구분은 결과 표를 참조한다.
